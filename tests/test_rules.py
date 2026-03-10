"""Tests for cylint PySpark linter."""

import unittest
from cylint.engine import LintEngine
from cylint.models import Severity
from cylint.config import Config, _parse_simple_yaml


class RuleTestBase(unittest.TestCase):
    """Base class for rule tests."""

    def setUp(self):
        self.engine = LintEngine()

    def lint(self, source: str) -> list:
        return self.engine.lint_source(source, filepath="test.py")

    def assert_rule_found(self, source: str, rule_id: str, count: int = 1):
        findings = self.lint(source)
        matched = [f for f in findings if f.rule_id == rule_id]
        self.assertEqual(
            len(matched), count,
            f"Expected {count} {rule_id} finding(s), got {len(matched)}. "
            f"All findings: {[f.rule_id for f in findings]}"
        )
        return matched

    def assert_no_findings(self, source: str, rule_id: str | None = None):
        findings = self.lint(source)
        if rule_id:
            matched = [f for f in findings if f.rule_id == rule_id]
            self.assertEqual(
                len(matched), 0,
                f"Expected no {rule_id} findings, got: {matched}"
            )
        else:
            self.assertEqual(len(findings), 0, f"Expected no findings, got: {findings}")


class TestCY001Collect(RuleTestBase):
    """CY001: .collect() on unfiltered DataFrame."""

    def test_unfiltered_collect(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
rows = df.collect()
""", "CY001")

    def test_filtered_collect_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
rows = df.filter(df.id > 10).collect()
""", "CY001")

    def test_limited_collect_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
rows = df.limit(100).collect()
""", "CY001")

    def test_collect_on_filtered_df(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
filtered = df.where(df.status == "active")
rows = filtered.collect()
""", "CY001")

    def test_collect_on_sql(self):
        self.assert_rule_found("""
df = spark.sql("SELECT * FROM table")
rows = df.collect()
""", "CY001")

    def test_non_df_collect_no_finding(self):
        """collect() on non-DataFrame should not trigger."""
        self.assert_no_findings("""
my_list = [1, 2, 3]
result = my_list.collect()
""", "CY001")


class TestCY002UDF(RuleTestBase):
    """CY002: UDF where builtin exists."""

    def test_lambda_lower(self):
        self.assert_rule_found("""
from pyspark.sql.functions import udf
lower_udf = udf(lambda x: x.lower())
""", "CY002")

    def test_lambda_upper(self):
        self.assert_rule_found("""
from pyspark.sql.functions import udf
upper_udf = udf(lambda x: x.upper())
""", "CY002")

    def test_lambda_strip(self):
        self.assert_rule_found("""
trim_udf = udf(lambda x: x.strip())
""", "CY002")

    def test_lambda_len(self):
        self.assert_rule_found("""
len_udf = udf(lambda x: len(x))
""", "CY002")

    def test_lambda_abs(self):
        self.assert_rule_found("""
abs_udf = udf(lambda x: abs(x))
""", "CY002")

    def test_complex_udf_no_finding(self):
        """Complex UDF with no simple builtin equivalent."""
        self.assert_no_findings("""
custom_udf = udf(lambda x: x[:3] + "-" + x[3:])
""", "CY002")

    def test_none_check_udf(self):
        self.assert_rule_found("""
null_udf = udf(lambda x: x is None)
""", "CY002")


class TestCY003WithColumnLoop(RuleTestBase):
    """CY003: .withColumn() in loop."""

    def test_for_loop_withcolumn(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
cols = ["a", "b", "c"]
for col in cols:
    df = df.withColumn(col, df[col].cast("string"))
""", "CY003")

    def test_single_withcolumn_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
df = df.withColumn("new_col", df.old_col + 1)
""", "CY003")

    def test_while_loop_withcolumn(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
cols = ["a", "b"]
i = 0
while i < len(cols):
    df = df.withColumn(cols[i], df[cols[i]].cast("int"))
    i += 1
""", "CY003")

    def test_withcolumns_in_loop_no_finding(self):
        """.withColumns() uses dict API, single plan node — no O(n²) issue."""
        self.assert_no_findings("""
df = spark.read.parquet("/data")
batches = [{"a": "1", "b": "2"}, {"c": "3"}]
for batch in batches:
    df = df.withColumns(batch)
""", "CY003")


class TestCY004SelectStar(RuleTestBase):
    """CY004: SELECT * in SQL strings."""

    def test_select_star(self):
        self.assert_rule_found("""
df = spark.sql("SELECT * FROM events")
""", "CY004")

    def test_select_star_with_where(self):
        """SELECT * even with WHERE is still flagged."""
        self.assert_rule_found("""
df = spark.sql("SELECT * FROM events WHERE date > '2024-01-01'")
""", "CY004")

    def test_select_columns_no_finding(self):
        self.assert_no_findings("""
df = spark.sql("SELECT id, name, date FROM events")
""", "CY004")

    def test_fstring_select_star(self):
        self.assert_rule_found("""
table = "events"
df = spark.sql(f"SELECT * FROM {table}")
""", "CY004")


class TestCY005Cache(RuleTestBase):
    """CY005: .cache() without reuse."""

    def test_cache_single_use(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
cached = df.cache()
result = cached.count()
""", "CY005")

    def test_cache_no_use(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
cached = df.cache()
""", "CY005")

    def test_cache_multiple_uses_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
cached = df.cache()
count = cached.count()
sample = cached.limit(10).collect()
cached.write.parquet("/out")
""", "CY005")

    def test_non_df_cache_no_finding(self):
        """cache() on a non-DataFrame should not trigger."""
        self.assert_no_findings("""
result = some_object.cache()
""", "CY005")


class TestCY006ToPandas(RuleTestBase):
    """CY006: .toPandas() on unfiltered DataFrame."""

    def test_unfiltered_topandas(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
pdf = df.toPandas()
""", "CY006")

    def test_filtered_topandas_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
pdf = df.filter(df.x > 0).toPandas()
""", "CY006")

    def test_agg_topandas_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
pdf = df.groupBy("cat").agg({"val": "sum"}).toPandas()
""", "CY006")


class TestCY007CrossJoin(RuleTestBase):
    """CY007: Cross join detection."""

    def test_explicit_crossjoin(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.crossJoin(b)
""", "CY007")

    def test_join_without_condition(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b)
""", "CY007")

    def test_join_with_on_no_finding(self):
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, on="id")
""", "CY007")

    def test_join_with_positional_condition(self):
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, a.id == b.id)
""", "CY007")

    def test_string_join_no_finding(self):
        """str.join() must not trigger CY007."""
        self.assert_no_findings("""
cols = ["a", "b", "c"]
result = ",".join(cols)
""", "CY007")

    def test_untracked_join_no_finding(self):
        """join() on a non-DataFrame variable must not trigger."""
        self.assert_no_findings("""
items = get_items()
result = items.join(other)
""", "CY007")


class TestCY008Repartition(RuleTestBase):
    """CY008: .repartition() before .write()."""

    def test_repartition_write(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition(10).write.parquet("/out")
""", "CY008")

    def test_coalesce_write_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.coalesce(4).write.parquet("/out")
""", "CY008")

    def test_repartition_without_write_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df2 = df.repartition(10)
""", "CY008")

    # --- Fire cases ---

    def test_small_count_before_partitioned_write(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition(10).write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_large_count_before_write_no_finding(self):
        """repartition(200) is increasing partitions — skip."""
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition(200).write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_column_no_partition_by(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition("date").write.parquet("/out")
""", "CY008")

    def test_column_different_partition_by(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition("user_id").write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_large_count_plus_column_no_finding(self):
        """repartition(200, "date") is increasing — skip."""
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition(200, "date").write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_small_count_plus_column(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition(10, "date").write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_broader_columns_than_partition_by(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition("date", "user_id").write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_boundary_count_101_no_finding(self):
        """Count just above threshold — skip."""
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition(101).write.parquet("/out")
""", "CY008")

    def test_boundary_count_100_fires(self):
        """Count at threshold — still fires."""
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition(100).write.parquet("/out")
""", "CY008")

    def test_no_args_repartition(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.repartition().write.parquet("/out")
""", "CY008")

    # --- Suppress cases (column-aligned repartition) ---

    def test_exact_column_match_suppressed(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition("date").write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_multi_column_match_suppressed(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition("year", "month").write.partitionBy("year", "month").parquet("/out")
""", "CY008")

    def test_subset_of_partition_by_suppressed(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition("date").write.partitionBy("date", "region").parquet("/out")
""", "CY008")

    def test_f_col_style_suppressed(self):
        self.assert_no_findings("""
from pyspark.sql import functions as F
df = spark.read.parquet("/in")
df.repartition(F.col("date")).write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_col_bare_style_suppressed(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition(col("date")).write.partitionBy("date").parquet("/out")
""", "CY008")

    def test_intermediate_methods_suppressed(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition("date").write.mode("overwrite").partitionBy("date").parquet("/out")
""", "CY008")

    def test_no_repartition_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.write.partitionBy("date").parquet("/out")
""", "CY008")


class TestConfig(unittest.TestCase):
    """Test configuration parsing."""

    def test_parse_yaml(self):
        config = _parse_simple_yaml("""
min-severity: warning
rules:
  CY001: warning
  CY004: off
  CY008: info
exclude:
  - tests/
  - notebooks/scratch/
""")
        self.assertEqual(config.min_severity, Severity.WARNING)
        self.assertEqual(config.rules["CY001"], Severity.WARNING)
        self.assertIsNone(config.rules["CY004"])
        self.assertEqual(config.rules["CY008"], Severity.INFO)
        self.assertEqual(config.exclude, ["tests/", "notebooks/scratch/"])

    def test_empty_config(self):
        config = _parse_simple_yaml("")
        self.assertEqual(config.min_severity, Severity.INFO)
        self.assertEqual(config.rules, {})
        self.assertEqual(config.exclude, [])


class TestEngineConfig(unittest.TestCase):
    """Test engine with configuration."""

    def test_disabled_rule(self):
        engine = LintEngine(disabled_rules={"CY001"})
        findings = engine.lint_source("""
df = spark.read.parquet("/data")
rows = df.collect()
""")
        self.assertEqual(
            len([f for f in findings if f.rule_id == "CY001"]), 0
        )

    def test_severity_filter(self):
        engine = LintEngine(min_severity=Severity.WARNING)
        findings = engine.lint_source("""
df = spark.sql("SELECT * FROM events")
""")
        # CY004 is INFO by default, should be filtered out
        self.assertEqual(
            len([f for f in findings if f.rule_id == "CY004"]), 0
        )

    def test_severity_override(self):
        engine = LintEngine(enabled_rules={"CY004": Severity.CRITICAL})
        findings = engine.lint_source("""
df = spark.sql("SELECT * FROM events")
""")
        matched = [f for f in findings if f.rule_id == "CY004"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].severity, Severity.CRITICAL)


class TestExitCodes(unittest.TestCase):
    def test_clean_exit_zero(self):
        engine = LintEngine()
        result = engine.lint_source("""
x = 1 + 2
""")
        from cylint.models import LintResult
        lr = LintResult(findings=result, files_scanned=1)
        self.assertEqual(lr.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
