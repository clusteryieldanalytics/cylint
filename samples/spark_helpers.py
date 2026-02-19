"""Shared utilities for Spark pipelines.

Common patterns used across all jobs. This module demonstrates
clean PySpark patterns that should not trigger any lint findings.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import List, Optional


def get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def read_table_with_date_filter(
    table: str,
    date_col: str,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Read a table with a date range filter applied at scan time."""
    spark = get_spark()
    return spark.table(table).filter(
        (F.col(date_col) >= start_date) & (F.col(date_col) <= end_date)
    )


def add_audit_columns(df: DataFrame) -> DataFrame:
    """Add standard audit columns to a DataFrame."""
    return df.select(
        "*",
        F.current_timestamp().alias("_etl_loaded_at"),
        F.lit("spark-etl").alias("_etl_source"),
        F.input_file_name().alias("_source_file"),
    )


def safe_divide(
    df: DataFrame,
    numerator: str,
    denominator: str,
    result_col: str,
    default: float = 0.0,
) -> DataFrame:
    """Division that handles nulls and zeros gracefully."""
    return df.withColumn(
        result_col,
        F.when(
            (F.col(denominator).isNull()) | (F.col(denominator) == 0),
            F.lit(default),
        ).otherwise(F.col(numerator) / F.col(denominator)),
    )


def deduplicate(
    df: DataFrame,
    key_columns: List[str],
    order_col: str,
    keep: str = "latest",
) -> DataFrame:
    """Deduplicate a DataFrame keeping the latest or earliest record."""
    from pyspark.sql.window import Window

    order_expr = F.col(order_col).desc() if keep == "latest" else F.col(order_col).asc()
    window = Window.partitionBy(key_columns).orderBy(order_expr)

    return df.withColumn("_rn", F.row_number().over(window)) \
        .filter(F.col("_rn") == 1) \
        .drop("_rn")


def write_partitioned(
    df: DataFrame,
    path: str,
    partition_cols: List[str],
    mode: str = "overwrite",
    max_records_per_file: Optional[int] = None,
) -> None:
    """Write a DataFrame with standard partitioning settings."""
    writer = df.write.mode(mode).partitionBy(*partition_cols)
    if max_records_per_file:
        writer = writer.option("maxRecordsPerFile", max_records_per_file)
    writer.parquet(path)


def collect_small(df: DataFrame, max_rows: int = 1000) -> list:
    """Safely collect a small DataFrame with a limit guard."""
    return df.limit(max_rows).collect()


def to_pandas_safe(df: DataFrame, max_rows: int = 100_000) -> "pd.DataFrame":
    """Convert to pandas with a safety limit."""
    return df.limit(max_rows).toPandas()
