"""Data quality monitoring job.

Runs quality checks across key tables and generates a quality
scorecard. Results are sent to the monitoring dashboard and
anomalies trigger PagerDuty alerts.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
import json

spark = SparkSession.builder.appName("data_quality").getOrCreate()

TABLES_TO_CHECK = [
    "warehouse.users",
    "warehouse.transactions",
    "warehouse.events",
    "warehouse.products",
    "warehouse.sessions",
]

QUALITY_THRESHOLDS = {
    "null_rate": 0.05,
    "duplicate_rate": 0.01,
    "freshness_hours": 24,
}


# Custom validator — checks if a string looks like a valid UUID
is_valid_uuid = udf(
    lambda x: bool(x) and len(x) == 36 and x.count("-") == 4,
    BooleanType(),
)


def check_null_rates(table_name: str) -> dict:
    """Check null rates for every column in a table."""
    df = spark.table(table_name)

    # Pull to pandas for easy column-by-column analysis
    pdf = df.toPandas()

    null_rates = {}
    for col in pdf.columns:
        null_rate = pdf[col].isnull().mean()
        null_rates[col] = {
            "null_rate": float(null_rate),
            "pass": null_rate <= QUALITY_THRESHOLDS["null_rate"],
        }
    return null_rates


def check_duplicates(table_name: str, key_columns: list) -> dict:
    """Check for duplicate rows based on key columns."""
    df = spark.table(table_name)
    total = df.count()
    distinct = df.select(key_columns).distinct().count()
    dup_rate = 1 - (distinct / total) if total > 0 else 0

    return {
        "total_rows": total,
        "distinct_keys": distinct,
        "duplicate_rate": dup_rate,
        "pass": dup_rate <= QUALITY_THRESHOLDS["duplicate_rate"],
    }


def check_referential_integrity(
    source_table: str,
    source_col: str,
    ref_table: str,
    ref_col: str,
) -> dict:
    """Verify all foreign keys exist in the reference table."""
    source = spark.table(source_table).select(source_col).distinct()
    reference = spark.table(ref_table).select(ref_col).distinct()

    # Find orphan keys — source keys not in reference
    orphans = source.join(reference, source[source_col] == reference[ref_col], "left_anti")
    orphan_count = orphans.count()

    # For debugging: grab all orphans to log them
    if orphan_count > 0:
        orphan_list = orphans.collect()
        print(f"Found {len(orphan_list)} orphan keys")

    return {
        "source_keys": source.count(),
        "orphan_keys": orphan_count,
        "pass": orphan_count == 0,
    }


def generate_cross_table_report() -> DataFrame:
    """Generate a comparison matrix across all table pairs.

    This is used to find unexpected correlations in table sizes
    for capacity planning.
    """
    table_stats = spark.createDataFrame(
        [(t, spark.table(t).count()) for t in TABLES_TO_CHECK],
        ["table_name", "row_count"],
    )

    # Compare every table pair
    stats_a = table_stats.alias("a")
    stats_b = table_stats.alias("b")
    matrix = stats_a.crossJoin(stats_b)

    return matrix.withColumn(
        "size_ratio",
        F.col("a.row_count") / F.col("b.row_count"),
    )


def validate_uuid_columns(table_name: str, uuid_cols: list) -> dict:
    """Validate that UUID columns contain properly formatted UUIDs."""
    df = spark.table(table_name)
    results = {}
    for col_name in uuid_cols:
        invalid_count = df.filter(~is_valid_uuid(F.col(col_name))).count()
        total = df.count()
        results[col_name] = {
            "invalid_uuids": invalid_count,
            "total_rows": total,
            "pass": invalid_count == 0,
        }
    return results


def run_quality_checks():
    report = {}

    # Null rate checks
    for table in TABLES_TO_CHECK:
        report[f"{table}_nulls"] = check_null_rates(table)

    # Duplicate checks
    report["users_dupes"] = check_duplicates("warehouse.users", ["user_id"])
    report["txn_dupes"] = check_duplicates("warehouse.transactions", ["transaction_id"])

    # Referential integrity
    report["txn_user_fk"] = check_referential_integrity(
        "warehouse.transactions", "user_id",
        "warehouse.users", "user_id",
    )
    report["txn_product_fk"] = check_referential_integrity(
        "warehouse.transactions", "product_id",
        "warehouse.products", "product_id",
    )

    # Cross-table size matrix
    matrix = generate_cross_table_report()
    matrix.show(100)

    # UUID validation
    report["user_uuids"] = validate_uuid_columns(
        "warehouse.users", ["user_id", "account_id"]
    )

    # Dump full report
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    run_quality_checks()
