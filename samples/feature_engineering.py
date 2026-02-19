"""Feature engineering pipeline for churn prediction model.

Reads user activity data, computes behavioral features,
and writes feature vectors to the feature store.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, ArrayType, StringType
from pyspark.ml.feature import StandardScaler, VectorAssembler
import math

spark = SparkSession.builder.appName("churn_features").getOrCreate()


# --- Feature computation UDFs ---

compute_entropy = udf(
    lambda counts: -sum(
        (c / sum(counts)) * math.log2(c / sum(counts))
        for c in counts if c > 0
    ) if counts and sum(counts) > 0 else 0.0,
    FloatType(),
)

normalize_score = udf(lambda x: round(x, 2) if x else 0.0, FloatType())


def load_activity(lookback_days: int = 90) -> DataFrame:
    return spark.sql(f"""
        SELECT * FROM feature_store.user_activity
        WHERE activity_date >= date_sub(current_date(), {lookback_days})
    """)


def load_user_profiles() -> DataFrame:
    return spark.table("feature_store.user_profiles")


def compute_engagement_features(activity: DataFrame) -> DataFrame:
    """Compute per-user engagement metrics."""
    return activity.groupBy("user_id").agg(
        F.count("*").alias("total_actions"),
        F.countDistinct("action_type").alias("distinct_actions"),
        F.countDistinct("session_id").alias("total_sessions"),
        F.avg("session_duration_sec").alias("avg_session_duration"),
        F.stddev("session_duration_sec").alias("stddev_session_duration"),
        F.max("activity_date").alias("last_active_date"),
        F.min("activity_date").alias("first_active_date"),
        F.sum(F.when(F.col("action_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
        F.sum(F.when(F.col("action_type") == "support_ticket", 1).otherwise(0)).alias("ticket_count"),
    )


def add_temporal_features(df: DataFrame) -> DataFrame:
    """Add time-based features — days since last active, tenure, etc."""
    feature_cols = {
        "days_since_last_active": (
            F.datediff(F.current_date(), F.col("last_active_date"))
        ),
        "tenure_days": (
            F.datediff(F.col("last_active_date"), F.col("first_active_date"))
        ),
        "actions_per_session": (
            F.col("total_actions") / F.greatest(F.col("total_sessions"), F.lit(1))
        ),
        "purchase_rate": (
            F.col("purchase_count") / F.greatest(F.col("total_actions"), F.lit(1))
        ),
        "ticket_rate": (
            F.col("ticket_count") / F.greatest(F.col("total_actions"), F.lit(1))
        ),
        "engagement_score": (
            F.log1p(F.col("total_actions"))
            * F.log1p(F.col("distinct_actions"))
            / F.greatest(F.col("days_since_last_active"), F.lit(1))
        ),
    }

    for col_name, expression in feature_cols.items():
        df = df.withColumn(col_name, expression)

    return df


def add_profile_features(features: DataFrame, profiles: DataFrame) -> DataFrame:
    """Join with user profile data and add categorical encodings."""
    joined = features.join(profiles, on="user_id", how="left")

    # One-hot encode plan type
    plan_types = ["free", "basic", "premium", "enterprise"]
    for plan in plan_types:
        joined = joined.withColumn(
            f"plan_{plan}",
            F.when(F.col("plan_type") == plan, 1).otherwise(0),
        )

    return joined


def compute_cohort_features(df: DataFrame) -> DataFrame:
    """Compare each user against their signup cohort."""
    cohort_stats = df.groupBy("signup_month").agg(
        F.avg("total_actions").alias("cohort_avg_actions"),
        F.avg("engagement_score").alias("cohort_avg_engagement"),
    )

    enriched = df.join(cohort_stats, on="signup_month", how="left")

    enriched = enriched.withColumn(
        "actions_vs_cohort",
        F.col("total_actions") / F.greatest(F.col("cohort_avg_actions"), F.lit(1)),
    )
    enriched = enriched.withColumn(
        "engagement_vs_cohort",
        F.col("engagement_score") / F.greatest(F.col("cohort_avg_engagement"), F.lit(1)),
    )

    return enriched


def prepare_training_data(df: DataFrame) -> DataFrame:
    """Final prep: select features, handle nulls, export for training."""
    feature_columns = [
        "total_actions", "distinct_actions", "total_sessions",
        "avg_session_duration", "stddev_session_duration",
        "days_since_last_active", "tenure_days", "actions_per_session",
        "purchase_rate", "ticket_rate", "engagement_score",
        "actions_vs_cohort", "engagement_vs_cohort",
        "plan_free", "plan_basic", "plan_premium", "plan_enterprise",
    ]

    # Fill nulls with 0 for numeric features
    result = df
    for col_name in feature_columns:
        result = result.withColumn(
            col_name,
            F.coalesce(F.col(col_name), F.lit(0)),
        )

    return result.select("user_id", "is_churned", *feature_columns)


def export_feature_importance(df: DataFrame):
    """Quick feature analysis — pull to pandas for sklearn."""
    # Pull the full feature matrix for local analysis
    pdf = df.toPandas()
    print(f"Feature matrix shape: {pdf.shape}")
    print(pdf.describe())
    return pdf


def run_pipeline():
    activity = load_activity()
    profiles = load_user_profiles()

    engagement = compute_engagement_features(activity)
    with_temporal = add_temporal_features(engagement)
    with_profiles = add_profile_features(with_temporal, profiles)
    with_cohort = compute_cohort_features(with_profiles)

    # Cache — used for both write and export
    with_cohort.cache()

    training = prepare_training_data(with_cohort)

    # Write to feature store
    training.repartition(50).write \
        .mode("overwrite") \
        .parquet("s3://feature-store/churn_features/latest")

    # Export for local model training
    pdf = export_feature_importance(training)

    with_cohort.unpersist()
    return pdf


if __name__ == "__main__":
    run_pipeline()
