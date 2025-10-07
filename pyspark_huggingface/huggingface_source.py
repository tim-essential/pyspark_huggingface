import ast
import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Callable, Any

from pyspark.sql.pandas.types import from_arrow_schema
from pyspark.sql.types import StructType
from pyspark_huggingface.compat.datasource import (
    DataSource,
    DataSourceReader,
    InputPartition,
)


logger = logging.getLogger(__name__)


def _with_retries(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Execute a callable with exponential backoff and jitter on HF Hub 429 rate limits.
    Mirrors the retry policy used in the sink's commit logic.
    """
    from huggingface_hub.errors import HfHubHTTPError

    max_retries = 15
    base_delay = 1  # seconds

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except HfHubHTTPError as e:
            if (
                getattr(e, "response", None) is not None
                and getattr(e.response, "status_code", None) == 429
                and attempt < max_retries
            ):
                base_exponential_delay = base_delay * (2**attempt)
                jitter = random.uniform(0, base_exponential_delay)
                delay = base_exponential_delay / 2 + jitter / 2
                logger.warning(
                    f"Rate limited on attempt {attempt + 1}/{max_retries + 1} for {getattr(func, '__name__', repr(func))}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            else:
                raise


if TYPE_CHECKING:
    from datasets import DatasetBuilder, IterableDataset


class HuggingFaceSource(DataSource):
    """
    A DataSource for reading and writing HuggingFace Datasets in Spark.

    This data source allows reading public datasets from the HuggingFace Hub directly into Spark
    DataFrames. The schema is automatically inferred from the dataset features. The split can be
    specified using the `split` option. The default split is `train`.

    Name: `huggingface`

    Data Source Options:
    - split (str): Specify which split to retrieve. Default: train
    - config (str): Specify which subset or configuration to retrieve.
    - streaming (bool): Specify whether to read a dataset without downloading it.

    Notes:
    -----
    - Currently it can only be used with public datasets. Private or gated ones are not supported.

    Examples
    --------

    Load a public dataset from the HuggingFace Hub.

    >>> df = spark.read.format("huggingface").load("stanfordnlp/imdb")
    DataFrame[text: string, label: bigint]

    >>> df.show()
    +--------------------+-----+
    |                text|label|
    +--------------------+-----+
    |I rented I AM CUR...|    0|
    |"I Am Curious: Ye...|    0|
    |...                 |  ...|
    +--------------------+-----+

    Load a specific split from a public dataset from the HuggingFace Hub.

    >>> spark.read.format("huggingface").option("split", "test").load("stanfordnlp/imdb").show()
    +--------------------+-----+
    |                text|label|
    +--------------------+-----+
    |I love sci-fi and...|    0|
    |Worth the enterta...|    0|
    |...                 |  ...|
    +--------------------+-----+

    Enable predicate pushdown for Parquet datasets.

    >>> spark.read.format("huggingface") \
    ...     .option("filters", '[("language_score", ">", 0.99)]') \
    ...     .option("columns", '["text", "language_score"]') \
    ...     .load("HuggingFaceFW/fineweb-edu") \
    ...     .show()
    +--------------------+------------------+                                       
    |                text|    language_score|
    +--------------------+------------------+
    |died Aug. 28, 181...|0.9901925325393677|
    |Coyotes spend a g...|0.9902171492576599|
    |...                 |               ...|
    +--------------------+------------------+
    """

    DEFAULT_SPLIT: str = "train"

    def __init__(self, options):
        super().__init__(options)
        from datasets import load_dataset_builder

        if "path" not in options or not options["path"]:
            raise Exception("You must specify a dataset name.")

        from huggingface_hub import get_token

        kwargs = dict(self.options)
        self.dataset_name = kwargs.pop("path")
        self.config_name = kwargs.pop("config", None)
        self.split = kwargs.pop("split", self.DEFAULT_SPLIT)
        self.revision = kwargs.pop("revision", None)
        self.streaming = kwargs.pop("streaming", "true").lower() == "true"
        self.token = kwargs.pop("token", None) or get_token()
        self.endpoint = kwargs.pop("endpoint", None)
        # Remove columns option as it's for column selection, not dataset configuration
        self.columns = kwargs.pop("columns", None)
        for arg in kwargs:
            if kwargs[arg].lower() == "true":
                kwargs[arg] = True
            elif kwargs[arg].lower() == "false":
                kwargs[arg] = False
            else:
                try:
                    kwargs[arg] = ast.literal_eval(kwargs[arg])
                except ValueError:
                    pass

        # Raise the right error if the dataset doesn't exist (with retries for rate limiting)
        api = self._get_api()
        _with_retries(
            api.repo_info,
            self.dataset_name,
            repo_type="dataset",
            revision=self.revision,
        )

        # Build dataset builder and create streaming dataset with retries
        self.builder = _with_retries(
            load_dataset_builder,
            self.dataset_name,
            self.config_name,
            token=self.token,
            revision=self.revision,
            **kwargs,
        )
        streaming_dataset = _with_retries(self.builder.as_streaming_dataset)
        if self.split not in streaming_dataset:
            raise Exception(
                f"Split {self.split} is invalid. Valid options are {list(streaming_dataset)}"
            )

        self.streaming_dataset = streaming_dataset[self.split]
        if not self.streaming_dataset.features:
            self.streaming_dataset = _with_retries(
                self.streaming_dataset._resolve_features
            )

    def _get_api(self):
        from huggingface_hub import HfApi

        return HfApi(
            token=self.token, endpoint=self.endpoint, library_name="pyspark_huggingface"
        )

    @classmethod
    def name(cls):
        return "huggingfacesource"

    def schema(self):
        """
        Return the Spark StructType schema.

        If the user provided a `columns` option, we must return a schema that
        matches exactly those columns (and order). Otherwise Spark will expect
        the full dataset schema while the reader yields batches with a subset
        of columns, leading to Arrow vector child index errors at runtime.
        """
        full_schema = from_arrow_schema(self.streaming_dataset.features.arrow_schema)
        if self.columns:
            # Lazily parse the columns option (JSON/python-literal list of names)
            try:
                requested_columns = ast.literal_eval(self.columns)
                if not isinstance(requested_columns, (list, tuple)):
                    requested_columns = [str(requested_columns)]
            except (ValueError, SyntaxError):
                # Fall back to full schema field order if parsing fails
                requested_columns = [field.name for field in full_schema.fields]

            # Filter and preserve order; ignore any names not present in the schema
            filtered_fields = [
                field for field in full_schema.fields if field.name in requested_columns
            ]
            # If any requested columns are not in the schema, just ignore them silently
            # to avoid raising here; the read() path also selects only existing columns.
            if filtered_fields and len(filtered_fields) <= len(full_schema.fields):
                return StructType(filtered_fields)
        return full_schema

    def reader(self, schema: StructType) -> "HuggingFaceDatasetsReader":
        return HuggingFaceDatasetsReader(
            schema,
            builder=self.builder,
            split=self.split,
            streaming_dataset=self.streaming_dataset if self.streaming else None,
            columns=self.columns,
        )

    def _with_retries(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a callable with exponential backoff and jitter on HF Hub 429 rate limits.
        Mirrors the retry policy used in the sink's commit logic.
        """
        from huggingface_hub.errors import HfHubHTTPError

        max_retries = 15
        base_delay = 1  # seconds

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except HfHubHTTPError as e:
                if (
                    getattr(e, "response", None) is not None
                    and getattr(e.response, "status_code", None) == 429
                    and attempt < max_retries
                ):
                    base_exponential_delay = base_delay * (2**attempt)
                    jitter = random.uniform(0, base_exponential_delay)
                    delay = base_exponential_delay / 2 + jitter / 2
                    logger.warning(
                        f"Rate limited on attempt {attempt + 1}/{max_retries + 1} for {getattr(func, '__name__', repr(func))}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    raise


@dataclass
class Shard(InputPartition):
    """Represents a dataset shard."""

    index: int


class HuggingFaceDatasetsReader(DataSourceReader):
    def __init__(
        self,
        schema: StructType,
        builder: "DatasetBuilder",
        split: str,
        streaming_dataset: Optional["IterableDataset"],
        columns: Optional[str] = None,
    ):
        self.schema = schema
        self.builder = builder
        self.split = split
        self.streaming_dataset = streaming_dataset
        self.columns = columns
        # Get and validate the split name

    def partitions(self) -> Sequence[Shard]:
        if self.streaming_dataset:
            return [Shard(index=i) for i in range(self.streaming_dataset.num_shards)]
        else:
            return [Shard(index=0)]

    def read(self, partition: Shard):
        if self.columns:
            # Parse the columns option if provided (it's a JSON string)
            import ast

            try:
                columns = ast.literal_eval(self.columns)
            except (ValueError, SyntaxError):
                # If parsing fails, use all columns from schema
                columns = [field.name for field in self.schema.fields]
        else:
            columns = [field.name for field in self.schema.fields]
        if self.streaming_dataset:
            shard = self.streaming_dataset.shard(
                num_shards=self.streaming_dataset.num_shards, index=partition.index
            )
            if shard._ex_iterable.iter_arrow:
                for _, pa_table in shard._ex_iterable.iter_arrow():
                    yield from pa_table.select(columns).to_batches()
            else:
                for _, example in shard:
                    yield example
        else:
            # Non-streaming path: prepare and load with retries
            _with_retries(self.builder.download_and_prepare)
            dataset = _with_retries(self.builder.as_dataset, self.split)
            # Get the underlying arrow table of the dataset
            table = dataset._data
            yield from table.select(columns).to_batches()
