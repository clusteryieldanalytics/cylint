"""Daily user events ETL pipeline.

Reads raw event data from S3, enriches with user dimensions,
and writes to the curated layer for downstream analytics.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType


spark = SparkSession.builder \
    .appName("daily_events_etl") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()


def load_events(date: str) -> DataFrame:
    return spark.read.parquet(f"s3://datalake/raw/events/dt={date}")


def load_user_dim() -> DataFrame:
    return spark.read.parquet("s3://datalake/curated/dim_users")


# --- Normalization UDFs ---
# "We wrote these early on before we knew about builtins"

normalize_email = udf(lambda x: x.lower(), StringType())
trim_whitespace = udf(lambda x: x.strip(), StringType())
get_length = udf(lambda x: len(x), IntegerType())


def clean_events(df: DataFrame) -> DataFrame:
    """Apply data quality rules to raw events."""
    # Normalize email addresses
    df = df.withColumn("email", normalize_email("email"))
    # Trim event names
    df = df.withColumn("event_name", trim_whitespace("event_name"))
    # Add payload length for monitoring
    df = df.withColumn("payload_len", get_length("payload"))
    return df


def add_derived_columns(df: DataFrame) -> DataFrame:
    """Add business logic columns."""
    derived_cols = [
        "event_hour", "event_dayofweek", "event_month",
        "event_quarter", "is_weekend", "session_bucket",
    ]
    for col_name in derived_cols:
        if col_name == "event_hour":
            df = df.withColumn(col_name, F.hour("event_timestamp"))
        elif col_name == "event_dayofweek":
            df = df.withColumn(col_name, F.dayofweek("event_timestamp"))
        elif col_name == "event_month":
            df = df.withColumn(col_name, F.month("event_timestamp"))
        elif col_name == "event_quarter":
            df = df.withColumn(col_name, F.quarter("event_timestamp"))
        elif col_name == "is_weekend":
            df = df.withColumn(
                col_name,
                F.when(F.dayofweek("event_timestamp").isin(1, 7), True).otherwise(False),
            )
        elif col_name == "session_bucket":
            df = df.withColumn(
                col_name,
                F.floor(F.unix_timestamp("event_timestamp") / 1800),
            )
    return df


def enrich_with_users(events: DataFrame, users: DataFrame) -> DataFrame:
    """Join events with user dimension."""
    return events.join(users, on="user_id", how="left")


def build_session_summary(df: DataFrame) -> DataFrame:
    """Aggregate events into session-level summaries."""
    return df.groupBy("user_id", "session_bucket") \
        .agg(
            F.count("*").alias("event_count"),
            F.min("event_timestamp").alias("session_start"),
            F.max("event_timestamp").alias("session_end"),
            F.countDistinct("event_name").alias("distinct_events"),
        )


def run_etl(date: str):
    # Load
    events = load_events(date)
    users = load_user_dim()

    # Transform
    cleaned = clean_events(events)
    enriched = add_derived_columns(cleaned)
    with_users = enrich_with_users(enriched, users)

    # Cache for reuse in two write paths
    with_users.cache()

    # Write enriched events
    with_users.repartition(200).write \
        .mode("overwrite") \
        .partitionBy("event_month") \
        .parquet(f"s3://datalake/curated/events/dt={date}")

    # Write session summaries
    sessions = build_session_summary(with_users)
    sessions.write \
        .mode("overwrite") \
        .parquet(f"s3://datalake/curated/sessions/dt={date}")

    with_users.unpersist()


if __name__ == "__main__":
    import sys
    run_etl(sys.argv[1])
