"""Sample PySpark pipeline with anti-patterns for testing."""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

spark = SparkSession.builder.getOrCreate()

# CY001: .collect() on unfiltered DataFrame
raw_data = spark.read.parquet("/data/events")
all_rows = raw_data.collect()

# CY002: UDF where builtin exists
lower_udf = udf(lambda x: x.lower(), StringType())
df = spark.read.parquet("/data/users")
df2 = df.withColumn("name_lower", lower_udf("name"))

# CY003: .withColumn() in loop
columns = ["col_a", "col_b", "col_c", "col_d"]
result = spark.read.csv("/data/input.csv")
for col in columns:
    result = result.withColumn(col, F.trim(F.col(col)))

# CY004: SELECT * in SQL string
summary = spark.sql("SELECT * FROM events WHERE date > '2024-01-01'")

# CY005: .cache() without reuse
expensive = spark.read.parquet("/data/large_table")
cached = expensive.cache()
output = cached.filter(F.col("status") == "active").write.parquet("/out")

# CY006: .toPandas() on unfiltered DataFrame
raw = spark.read.parquet("/data/metrics")
pandas_df = raw.toPandas()

# CY007: cross join
users = spark.table("users")
products = spark.table("products")
crossed = users.crossJoin(products)
also_crossed = users.join(products)

# CY008: .repartition() before write
final = spark.read.parquet("/data/output_staging")
final.repartition(10).write.parquet("/data/output")

# --- These should NOT trigger findings ---

# Filtered collect is fine
filtered_collect = spark.read.parquet("/data/x").filter(F.col("id") == 1).collect()

# toPandas after agg is fine
agg_result = spark.read.parquet("/data/x").groupBy("cat").agg(F.count("*")).toPandas()

# join with condition is fine
joined = users.join(products, on="user_id")

# coalesce before write is fine
spark.read.parquet("/data/x").coalesce(4).write.parquet("/out2")
