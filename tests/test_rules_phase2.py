"""Tests for Phase 2 lint rules: CY009–CY016."""

import unittest
from cylint.engine import LintEngine
from cylint.models import Severity


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


# ---------------------------------------------------------------------------
# CY009 — UDF in filter / where
# ---------------------------------------------------------------------------
class TestCY009UdfFilter(RuleTestBase):
    """CY009: UDF in .filter()/.where() blocks predicate pushdown."""

    def test_registered_udf_variable(self):
        self.assert_rule_found("""
from pyspark.sql.functions import udf
my_filter = udf(lambda x: x > 0)
df = spark.table("orders")
df.filter(my_filter(col("amount")))
""", "CY009")

    def test_inline_udf_call(self):
        self.assert_rule_found("""
df = spark.table("orders")
df.filter(udf(lambda x: x > 0)(col("amount")))
""", "CY009")

    def test_udf_decorator(self):
        self.assert_rule_found("""
@udf("boolean")
def is_valid(x): return x is not None
df = spark.table("orders")
df.filter(is_valid(col("status")))
""", "CY009")

    def test_pandas_udf_decorator(self):
        self.assert_rule_found("""
@pandas_udf("boolean")
def check_val(x): return x > 0
df = spark.table("orders")
df.where(check_val(col("amount")))
""", "CY009")

    def test_lambda_in_filter(self):
        self.assert_rule_found("""
df = spark.table("orders")
df.filter(lambda row: row.status == "active")
""", "CY009")

    def test_where_with_udf(self):
        self.assert_rule_found("""
cleaner = udf(lambda x: x.strip() != "")
df = spark.table("orders")
df.where(cleaner(col("name")))
""", "CY009")

    def test_builtin_expression_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
df.filter(F.col("amount") > 0)
""", "CY009")

    def test_builtin_when_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
df.filter(F.when(col("x") > 0, True).otherwise(False))
""", "CY009")

    def test_builtin_regexp_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
df.filter(F.regexp_extract(col("s"), r"\\d+", 0) != "")
""", "CY009")

    def test_non_tracked_df_no_finding(self):
        """Filter on a non-tracked variable should not fire."""
        self.assert_no_findings("""
my_filter = udf(lambda x: x > 0)
something.filter(my_filter(col("x")))
""", "CY009")


# ---------------------------------------------------------------------------
# CY010 — Missing explicit join type
# ---------------------------------------------------------------------------
class TestCY010JoinType(RuleTestBase):
    """CY010: .join() without explicit how= argument."""

    def test_join_no_how_keyword(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, on="user_id")
""", "CY010")

    def test_join_list_condition_no_how(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, ["user_id", "date"])
""", "CY010")

    def test_join_expr_condition_no_how(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, a.id == b.id)
""", "CY010")

    def test_join_with_how_keyword_no_finding(self):
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, on="user_id", how="inner")
""", "CY010")

    def test_join_with_how_left_no_finding(self):
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, on="user_id", how="left")
""", "CY010")

    def test_join_positional_how_no_finding(self):
        """Third positional arg is how — do not flag."""
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, "user_id", "left")
""", "CY010")


# ---------------------------------------------------------------------------
# CY011 — Column transformations in a loop
# ---------------------------------------------------------------------------
class TestCY011LoopColumns(RuleTestBase):
    """CY011: Per-column DataFrame transformation inside a loop."""

    def test_withcolumnrenamed_in_loop(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
renames = [("old1", "new1"), ("old2", "new2")]
for old, new in renames:
    df = df.withColumnRenamed(old, new)
""", "CY011")

    def test_drop_in_loop(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
drop_cols = ["a", "b", "c"]
for c in drop_cols:
    df = df.drop(c)
""", "CY011")

    def test_withcolumn_defers_to_cy003(self):
        """CY011 should NOT fire on .withColumn() lines where CY003 fires."""
        findings = self.lint("""
df = spark.read.parquet("/data")
cols = ["a", "b", "c"]
for col in cols:
    df = df.withColumn(col, df[col].cast("string"))
""")
        cy003 = [f for f in findings if f.rule_id == "CY003"]
        cy011 = [f for f in findings if f.rule_id == "CY011"]
        self.assertGreater(len(cy003), 0, "CY003 should fire")
        self.assertEqual(len(cy011), 0, "CY011 should not duplicate CY003")

    def test_single_rename_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
df = df.withColumnRenamed("old", "new")
""", "CY011")

    def test_non_df_loop_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/data")
for row in df.collect():
    process(row)
""", "CY011")


# ---------------------------------------------------------------------------
# CY012 — Debug methods in production code
# ---------------------------------------------------------------------------
class TestCY012DebugMethods(RuleTestBase):
    """CY012: Debug/inspection methods left in production code."""

    def test_show(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
df.show()
""", "CY012")

    def test_show_with_n(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
df.show(20)
""", "CY012")

    def test_display(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
df.display()
""", "CY012")

    def test_printschema(self):
        self.assert_rule_found("""
df = spark.read.parquet("/data")
df.printSchema()
""", "CY012")

    def test_explain(self):
        findings = self.assert_rule_found("""
df = spark.read.parquet("/data")
df.explain()
""", "CY012")
        # explain should be INFO severity
        self.assertEqual(findings[0].severity, Severity.INFO)

    def test_show_severity_is_warning(self):
        findings = self.assert_rule_found("""
df = spark.read.parquet("/data")
df.show()
""", "CY012")
        self.assertEqual(findings[0].severity, Severity.WARNING)

    def test_non_df_show_no_finding(self):
        self.assert_no_findings("""
my_list = [1, 2, 3]
my_list.show()
""", "CY012")


# ---------------------------------------------------------------------------
# CY013 — coalesce(1) before write
# ---------------------------------------------------------------------------
class TestCY013CoalesceWrite(RuleTestBase):
    """CY013: .coalesce(1) before .write() — single-executor bottleneck."""

    def test_coalesce_1_write_parquet(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.coalesce(1).write.parquet("s3://out")
""", "CY013")

    def test_coalesce_1_write_save_as_table(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.coalesce(1).write.mode("overwrite").saveAsTable("prod.output")
""", "CY013")

    def test_coalesce_1_write_csv(self):
        self.assert_rule_found("""
df = spark.read.parquet("/in")
df.filter(df.x > 0).coalesce(1).write.csv("/out")
""", "CY013")

    def test_coalesce_4_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.coalesce(4).write.parquet("/out")
""", "CY013")

    def test_coalesce_1_no_write_no_finding(self):
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.coalesce(1).show()
""", "CY013")

    def test_repartition_1_not_caught_by_cy013(self):
        """CY013 is specifically for coalesce(1), not repartition(1)."""
        self.assert_no_findings("""
df = spark.read.parquet("/in")
df.repartition(1).write.parquet("/out")
""", "CY013")


# ---------------------------------------------------------------------------
# CY014 — Repeated actions without cache
# ---------------------------------------------------------------------------
class TestCY014RepeatedActions(RuleTestBase):
    """CY014: Multiple terminal actions without .cache()."""

    def test_two_actions_no_cache(self):
        findings = self.assert_rule_found("""
df = spark.table("orders")
n = df.count()
rows = df.collect()
""", "CY014")
        self.assertEqual(findings[0].action_count, 2)

    def test_three_actions_no_cache(self):
        findings = self.assert_rule_found("""
df = spark.table("orders")
n = df.count()
sample = df.collect()
df.write.parquet("s3://out")
""", "CY014")
        self.assertEqual(findings[0].action_count, 3)

    def test_cached_before_second_action_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders").cache()
n = df.count()
sample = df.collect()
""", "CY014")

    def test_cached_via_assignment_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
df = df.cache()
n = df.count()
sample = df.collect()
""", "CY014")

    def test_single_action_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
n = df.count()
""", "CY014")

    def test_severity_is_critical(self):
        findings = self.assert_rule_found("""
df = spark.table("orders")
n = df.count()
rows = df.collect()
""", "CY014")
        self.assertEqual(findings[0].severity, Severity.CRITICAL)

    # --- Cache tracking fixes ---

    def test_chained_cache_in_assignment_no_finding(self):
        """df = spark.table(...).filter(...).cache() should detect cache."""
        self.assert_no_findings("""
df = spark.table("orders").filter(col("date") > "2025-01-01").cache()
n = df.count()
df.write.parquet("s3://out")
""", "CY014")

    def test_cache_to_new_variable_no_finding(self):
        """df2 = df.cache() should detect cache on df2."""
        self.assert_no_findings("""
df = spark.table("orders")
df2 = df.cache()
n = df2.count()
df2.write.parquet("s3://out")
""", "CY014")

    def test_cache_propagated_through_chain_no_finding(self):
        """Cache on parent should propagate to derived DataFrame."""
        self.assert_no_findings("""
df = spark.table("orders").cache()
df2 = df.filter(col("date") > "2025-01-01")
n = df2.count()
df2.write.parquet("s3://out")
""", "CY014")

    def test_standalone_cache_expression_no_finding(self):
        """df.cache() as standalone expression should detect cache."""
        self.assert_no_findings("""
df = spark.table("orders")
df.cache()
n = df.count()
df.write.parquet("s3://out")
""", "CY014")

    def test_persist_variant_no_finding(self):
        """df = spark.table(...).persist() should detect cache."""
        self.assert_no_findings("""
df = spark.table("orders").persist()
n = df.count()
df.write.parquet("s3://out")
""", "CY014")

    # --- Debug action exclusion ---

    def test_show_plus_write_no_finding(self):
        """show + write = only 1 real action, should not fire."""
        self.assert_no_findings("""
df = spark.table("orders")
df.show()
df.write.parquet("s3://out")
""", "CY014")

    def test_display_plus_write_no_finding(self):
        """display + write = only 1 real action."""
        self.assert_no_findings("""
df = spark.table("orders")
df.display()
df.write.parquet("s3://out")
""", "CY014")

    def test_show_display_write_no_finding(self):
        """show + display + write = still 1 real action."""
        self.assert_no_findings("""
df = spark.table("orders")
df.show()
df.display()
df.write.parquet("s3://out")
""", "CY014")

    def test_printschema_plus_write_no_finding(self):
        """printSchema + write = only 1 real action."""
        self.assert_no_findings("""
df = spark.table("orders")
df.printSchema()
df.write.parquet("s3://out")
""", "CY014")

    # --- Should still fire ---

    def test_count_plus_write_fires(self):
        """Two real actions without cache should fire."""
        findings = self.assert_rule_found("""
df = spark.table("orders")
n = df.count()
df.write.parquet("s3://out")
""", "CY014")
        self.assertEqual(findings[0].action_count, 2)

    def test_show_count_write_fires(self):
        """show + count + write = 2 real actions (count + write)."""
        findings = self.assert_rule_found("""
df = spark.table("orders")
df.show()
n = df.count()
df.write.parquet("s3://out")
""", "CY014")
        self.assertEqual(findings[0].action_count, 2)

    def test_multiple_writes_fires(self):
        """Multiple writes without cache should fire."""
        findings = self.assert_rule_found("""
df = spark.table("orders")
df.write.parquet("s3://parquet-out")
df.write.csv("s3://csv-out")
""", "CY014")
        self.assertEqual(findings[0].action_count, 2)

    def test_cache_on_different_variable_fires(self):
        """Cache on a different variable should not suppress."""
        findings = self.assert_rule_found("""
df = spark.table("orders")
df2 = spark.table("events").cache()
n = df.count()
df.write.parquet("s3://out")
""", "CY014")
        self.assertEqual(findings[0].action_count, 2)


# ---------------------------------------------------------------------------
# CY015 — Non-equi join
# ---------------------------------------------------------------------------
class TestCY015NonEquiJoin(RuleTestBase):
    """CY015: Non-equi join condition producing implicit cartesian product."""

    def test_lit_true_join(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, F.lit(True))
""", "CY015")

    def test_greater_than_condition(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, a["ts"] > b["ts"])
""", "CY015")

    def test_not_equal_condition(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, a["ts"] != b["ts"])
""", "CY015")

    def test_bare_true_literal(self):
        self.assert_rule_found("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, True)
""", "CY015")

    def test_equi_join_no_finding(self):
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, a["id"] == b["id"])
""", "CY015")

    def test_string_condition_no_finding(self):
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, on="user_id", how="inner")
""", "CY015")

    def test_mixed_equi_and_range_no_finding(self):
        """A condition with == alongside >= should NOT fire (has equi key)."""
        self.assert_no_findings("""
a = spark.table("a")
b = spark.table("b")
c = a.join(b, (a.id == b.id) & (a.ts >= b.start))
""", "CY015")

    # --- Keyword on= cases ---

    def test_keyword_on_greater_than(self):
        self.assert_rule_found("""
df = spark.table("orders")
other = spark.table("events")
df.join(other, on=df["ts"] > other["start"])
""", "CY015")

    def test_keyword_on_lit_true(self):
        self.assert_rule_found("""
from pyspark.sql import functions as F
df = spark.table("orders")
other = spark.table("events")
df.join(other, on=F.lit(True), how="left")
""", "CY015")

    def test_keyword_on_not_equal(self):
        self.assert_rule_found("""
df = spark.table("orders")
other = spark.table("events")
df.join(other, on=df["ts"] != other["ts"], how="inner")
""", "CY015")

    def test_keyword_on_non_column_expr(self):
        self.assert_rule_found("""
from pyspark.sql import functions as F
df = spark.table("orders")
other = spark.table("events")
df.join(other, on=F.length(F.col("name")) > 5)
""", "CY015")

    def test_keyword_on_string_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
other = spark.table("events")
df.join(other, on="user_id")
""", "CY015")

    def test_keyword_on_list_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
other = spark.table("events")
df.join(other, on=["user_id", "date"])
""", "CY015")

    def test_keyword_on_equi_no_finding(self):
        self.assert_no_findings("""
df = spark.table("orders")
other = spark.table("events")
df.join(other, on=df["id"] == other["id"], how="left")
""", "CY015")

    def test_string_join_no_finding(self):
        """String .join() should not fire CY015."""
        self.assert_no_findings("""
result = ",".join(["a", "b", "c"])
""", "CY015")

    def test_no_duplicate_with_cy007(self):
        """CY015 should not fire on lines already flagged by CY007."""
        findings = self.lint("""
a = spark.table("a")
b = spark.table("b")
c = a.crossJoin(b)
""")
        cy007 = [f for f in findings if f.rule_id == "CY007"]
        cy015 = [f for f in findings if f.rule_id == "CY015"]
        self.assertGreater(len(cy007), 0, "CY007 should fire")
        self.assertEqual(len(cy015), 0, "CY015 should not duplicate CY007")


# ---------------------------------------------------------------------------
# CY016 — Invalid escape sequence
# ---------------------------------------------------------------------------
class TestCY016InvalidEscape(RuleTestBase):
    """CY016: Invalid escape sequence captured during parse."""

    def test_invalid_escape_in_string(self):
        self.assert_rule_found(r"""
import re
pattern = re.compile('\d+')
""", "CY016")

    def test_multiple_invalid_escapes(self):
        findings = self.assert_rule_found(r"""
import re
p1 = '\d+'
p2 = '\s+'
""", "CY016", count=2)

    def test_raw_string_no_finding(self):
        self.assert_no_findings(r"""
import re
pattern = re.compile(r'\d+')
""", "CY016")

    def test_valid_escapes_no_finding(self):
        self.assert_no_findings("""
msg = "hello\\nworld"
path = "C:\\\\Users"
tab = "col1\\tcol2"
""", "CY016")

    def test_cy016_disabled(self):
        engine = LintEngine(disabled_rules={"CY016"})
        findings = engine.lint_source(r"""
import re
pattern = re.compile('\d+')
""", filepath="test.py")
        cy016 = [f for f in findings if f.rule_id == "CY016"]
        self.assertEqual(len(cy016), 0)

    def test_severity_is_info(self):
        findings = self.assert_rule_found(r"""
x = '\d'
""", "CY016")
        self.assertEqual(findings[0].severity, Severity.INFO)


if __name__ == "__main__":
    unittest.main()
