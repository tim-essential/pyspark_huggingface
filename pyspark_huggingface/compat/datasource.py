from typing import TYPE_CHECKING, Iterator, Optional, Union

import pyspark
from packaging import version


if version.parse(pyspark.__version__) >= version.parse("4.0.0.dev2"):
    from pyspark.sql.datasource import DataSource, DataSourceArrowWriter, DataSourceReader, DataSourceWriter, InputPartition, WriterCommitMessage
else:
    class DataSource:
        def __init__(self, options):
            self.options = options

    class DataSourceArrowWriter:
        ...

    class DataSourceReader:
        ...
    
    class DataSourceWriter:
        ...
    
    class InputPartition:
        ...
    
    class WriterCommitMessage:
        ...
    

    import os
    import logging

    from pyspark.sql.readwriter import DataFrameReader as _DataFrameReader

    if TYPE_CHECKING:
        import pyarrow as pa
        from pyspark.sql.dataframe import DataFrame
        from pyspark.sql.readwriter import PathOrPaths
        from pyspark.sql.types import StructType
        from pyspark.sql._typing import OptionalPrimitiveType


    _orig_format = _DataFrameReader.format

    def _new_format(self: _DataFrameReader, source: str) -> _DataFrameReader:
        self._format = source
        return _orig_format(self, source)
    
    _DataFrameReader.format = _new_format

    _orig_option = _DataFrameReader.option

    def _new_option(self: _DataFrameReader, key, value) -> _DataFrameReader:
        if not hasattr(self, "_options"):
            self._options = {}
        self._options[key] = value
        return _orig_option(self, key, value)
    
    _DataFrameReader.option = _orig_option

    _orig_options = _DataFrameReader.options

    def _new_options(self: _DataFrameReader, **options) -> _DataFrameReader:
        if not hasattr(self, "_options"):
            self._options = {}
        self._options.update(options)
        return _orig_options(self, **options)
    
    _DataFrameReader.options = _orig_options

    _orig_load = _DataFrameReader.load

    class _unpack_dict(dict):
        ...

    class _ArrowPipe:

        def __init__(self, *fns):
            self.fns = fns

        def __call__(self, iterator: Iterator["pa.RecordBatch"]):
            for record_batch in iterator:
                for data in record_batch.to_pylist():
                    for fn in self.fns:
                        data = fn(**data) if isinstance(data, _unpack_dict) else fn(data)
                    yield from data

    def _new_load(
        self: _DataFrameReader,
        path: Optional["PathOrPaths"] = None,
        format: Optional[str] = None,
        schema: Optional[Union["StructType", str]] = None,
        **options: "OptionalPrimitiveType",
    ) -> "DataFrame":
        if (format or getattr(self, "_format", None)) == "huggingface":
            from dataclasses import asdict
            from pyspark_huggingface.huggingface import HuggingFaceDatasets
            
            source = HuggingFaceDatasets(options={**getattr(self, "_options", {}), **options, "path": path}).get_source()
            schema = schema or source.schema()
            reader = source.reader(schema)
            partitions = reader.partitions()
            partition_cls = type(partitions[0])
            rdd = self._spark.sparkContext.parallelize([asdict(partition) for partition in partitions], len(partitions))
            df = self._spark.createDataFrame(rdd)
            return df.mapInArrow(_ArrowPipe(_unpack_dict, partition_cls, reader.read), schema)
            
        return _orig_load(self, path=path, format=format, schema=schema, **options)

    _DataFrameReader.load = _new_load

    if not os.environ.get("SPARK_ENV_LOADED"):
        logging.getLogger(__name__).warning(f"huggingface datasource enabled for pyspark {pyspark.__version__} (backport from pyspark 4)")
