"""Weekly revenue reporting job.

Pulls transaction data, computes revenue metrics by product line,
and exports summaries for the finance team's dashboards.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("revenue_report").getOrCreate()


def get_transactions(start_date: str, end_date: str) -> DataFrame:
    """Pull raw transactions for the reporting period."""
    return spark.sql(f"""
        SELECT * FROM warehouse.transactions
        WHERE transaction_date BETWEEN '{start_date}' AND '{end_date}'
          AND status = 'completed'
    """)


def get_refunds(start_date: str, end_date: str) -> DataFrame:
    return spark.sql(f"""
        SELECT * FROM warehouse.refunds
        WHERE refund_date BETWEEN '{start_date}' AND '{end_date}'
    """)


def get_product_catalog() -> DataFrame:
    return spark.table("warehouse.product_catalog")


# Finance wants amounts in EUR — "just multiply by the rate"
convert_to_eur = udf(lambda x: x * 0.92 if x else 0.0, DoubleType())


def compute_net_revenue(transactions: DataFrame, refunds: DataFrame) -> DataFrame:
    """Compute net revenue = gross - refunds."""
    gross = transactions.groupBy("product_line", "region") \
        .agg(F.sum("amount").alias("gross_revenue"))

    refund_totals = refunds.groupBy("product_line", "region") \
        .agg(F.sum("refund_amount").alias("total_refunds"))

    # Join without specifying columns — both have product_line and region
    net = gross.join(refund_totals)

    return net.withColumn(
        "net_revenue",
        F.col("gross_revenue") - F.coalesce(F.col("total_refunds"), F.lit(0)),
    )


def enrich_with_catalog(revenue: DataFrame, catalog: DataFrame) -> DataFrame:
    """Add product metadata for the finance report."""
    enriched = revenue.join(catalog, on="product_line", how="left")
    enriched = enriched.withColumn("revenue_eur", convert_to_eur("net_revenue"))
    return enriched


def generate_executive_summary(df: DataFrame) -> DataFrame:
    """Top-level summary for the exec dashboard."""
    return df.groupBy("business_unit") \
        .agg(
            F.sum("net_revenue").alias("total_net_revenue"),
            F.sum("revenue_eur").alias("total_net_revenue_eur"),
            F.countDistinct("product_line").alias("active_product_lines"),
        )


def export_to_finance(df: DataFrame, label: str):
    """Export to the finance team's landing zone."""
    # Debug: check what we're about to write
    df.show(50)
    df.printSchema()

    df.repartition(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(f"s3://finance-exports/{label}")


def run_report(start_date: str, end_date: str):
    txn = get_transactions(start_date, end_date)
    refunds = get_refunds(start_date, end_date)
    catalog = get_product_catalog()

    # Cache transactions — we only use them once but "just in case"
    txn_cached = txn.cache()

    net_revenue = compute_net_revenue(txn_cached, refunds)
    enriched = enrich_with_catalog(net_revenue, catalog)

    # Detailed report
    export_to_finance(enriched, f"detailed/{start_date}_{end_date}")

    # Executive summary
    summary = generate_executive_summary(enriched)
    export_to_finance(summary, f"executive/{start_date}_{end_date}")

    # Quick sanity check — pull everything to driver to eyeball it
    all_data = enriched.collect()
    print(f"Exported {len(all_data)} rows")


if __name__ == "__main__":
    import sys
    run_report(sys.argv[1], sys.argv[2])
