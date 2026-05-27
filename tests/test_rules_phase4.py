"""Tests for Phase 4 lint rules: CY018, CY020, CY025, CY031."""

import unittest
from cylint.engine import LintEngine


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
# CY018 — Schema inference on read
# ---------------------------------------------------------------------------

class TestCY018SchemaInference(RuleTestBase):
    """CY018: spark.read.csv()/json() without .schema()."""

    def test_csv_no_schema_fires(self):
        self.assert_rule_found(
            'df = spark.read.csv("data/orders.csv")',
            "CY018",
        )

    def test_json_no_schema_fires(self):
        self.assert_rule_found(
            'df = spark.read.json("data/events.json")',
            "CY018",
        )

    def test_format_csv_load_fires(self):
        self.assert_rule_found(
            'df = spark.read.format("csv").load("data/orders.csv")',
            "CY018",
        )

    def test_format_json_load_fires(self):
        self.assert_rule_found(
            'df = spark.read.format("json").load("data/events/")',
            "CY018",
        )

    def test_infer_schema_true_string_fires(self):
        self.assert_rule_found(
            'df = spark.read.option("inferSchema", "true").csv("data/f.csv")',
            "CY018",
        )

    def test_infer_schema_true_bool_fires(self):
        self.assert_rule_found(
            'df = spark.read.option("inferSchema", True).csv("data/f.csv")',
            "CY018",
        )

    def test_multiple_options_no_schema_fires(self):
        src = '''\
df = (spark.read
    .option("header", "true")
    .option("delimiter", "|")
    .csv("data/orders.csv"))
'''
        self.assert_rule_found(src, "CY018")

    def test_with_schema_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.schema(order_schema).csv("data/orders.csv")',
            "CY018",
        )

    def test_infer_schema_false_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.option("inferSchema","false").csv("data/f.csv")',
            "CY018",
        )

    def test_parquet_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.parquet("data/orders.parquet")',
            "CY018",
        )

    def test_delta_format_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.format("delta").load("data/orders/")',
            "CY018",
        )

    def test_table_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.table("orders")',
            "CY018",
        )

    def test_orc_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.orc("data/orders.orc")',
            "CY018",
        )

    def test_avro_format_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.format("avro").load("data/orders/")',
            "CY018",
        )

    def test_format_parquet_load_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.format("parquet").load("data/orders/")',
            "CY018",
        )

    def test_sqlcontext_read_csv_fires(self):
        self.assert_rule_found(
            'df = sqlContext.read.csv("data/orders.csv")',
            "CY018",
        )

    def test_infer_schema_false_bool_no_fire(self):
        self.assert_no_findings(
            'df = spark.read.option("inferSchema", False).csv("data/f.csv")',
            "CY018",
        )

    def test_schema_kwarg_csv_no_fire(self):
        src = 'df = spark.read.csv("path", schema="nr integer,lang string")'
        self.assert_no_findings(src, "CY018")

    def test_schema_kwarg_json_no_fire(self):
        src = 'df = spark.read.json("path", schema=my_schema)'
        self.assert_no_findings(src, "CY018")

    def test_schema_kwarg_load_no_fire(self):
        src = 'df = spark.read.format("csv").load("path", schema=s)'
        self.assert_no_findings(src, "CY018")


# ---------------------------------------------------------------------------
# CY020 — .count() emptiness check
# ---------------------------------------------------------------------------

class TestCY020CountEmptiness(RuleTestBase):
    """CY020: .count() used only to check emptiness."""

    def test_count_gt_zero_fires(self):
        src = '''\
df = spark.table("orders")
if df.count() > 0:
    process(df)
'''
        self.assert_rule_found(src, "CY020")

    def test_count_eq_zero_fires(self):
        src = '''\
df = spark.table("orders")
if df.count() == 0:
    pass
'''
        self.assert_rule_found(src, "CY020")

    def test_count_ne_zero_fires(self):
        src = '''\
df = spark.table("orders")
if df.count() != 0:
    process(df)
'''
        self.assert_rule_found(src, "CY020")

    def test_reversed_comparison_fires(self):
        src = '''\
df = spark.table("orders")
if 0 < df.count():
    process(df)
'''
        self.assert_rule_found(src, "CY020")

    def test_not_count_fires(self):
        src = '''\
df = spark.table("orders")
if not df.count():
    pass
'''
        self.assert_rule_found(src, "CY020")

    def test_assigned_then_compared_fires(self):
        src = '''\
df = spark.table("orders")
n = df.count()
if n > 0:
    process(df)
'''
        self.assert_rule_found(src, "CY020")

    def test_count_gt_100_no_fire(self):
        """Threshold comparison — not an emptiness check."""
        src = '''\
df = spark.table("orders")
if df.count() > 100:
    sample(df)
'''
        self.assert_no_findings(src, "CY020")

    def test_count_used_as_value_no_fire(self):
        """Count used for actual value, not emptiness."""
        src = '''\
df = spark.table("orders")
total = df.count()
print(f"Found {total} rows")
'''
        self.assert_no_findings(src, "CY020")

    def test_count_in_arithmetic_no_fire(self):
        src = '''\
df = spark.table("orders")
avg = total_amount / df.count()
'''
        self.assert_no_findings(src, "CY020")

    def test_count_passed_to_function_no_fire(self):
        src = '''\
df = spark.table("orders")
log_metric("rows", df.count())
'''
        self.assert_no_findings(src, "CY020")

    def test_dual_use_no_fire(self):
        """Count assigned and used for both emptiness and logging — cannot replace."""
        src = '''\
df = spark.table("orders")
n = df.count()
print(f"row count: {n}")
if n > 0:
    process(df)
'''
        self.assert_no_findings(src, "CY020")

    def test_gte_one_fires(self):
        src = '''\
df = spark.table("orders")
if df.count() >= 1:
    process(df)
'''
        self.assert_rule_found(src, "CY020")

    def test_lt_one_fires(self):
        src = '''\
df = spark.table("orders")
if df.count() < 1:
    pass
'''
        self.assert_rule_found(src, "CY020")

    def test_count_gte_zero_tautology_no_fire(self):
        """count >= 0 is always true — not an emptiness check."""
        src = '''\
df = spark.table("orders")
if df.count() >= 0:
    process(df)
'''
        self.assert_no_findings(src, "CY020")

    def test_count_lt_zero_contradiction_no_fire(self):
        """count < 0 is always false — not an emptiness check."""
        src = '''\
df = spark.table("orders")
if df.count() < 0:
    process(df)
'''
        self.assert_no_findings(src, "CY020")

    def test_while_count_fires(self):
        src = '''\
df = spark.table("orders")
while df.count() > 0:
    process(df)
'''
        self.assert_rule_found(src, "CY020")

    def test_assert_count_fires(self):
        src = '''\
df = spark.table("orders")
assert df.count() > 0
'''
        self.assert_rule_found(src, "CY020")

    def test_count_lte_zero_fires(self):
        """count <= 0 is semantically 'is empty' for non-negative counts."""
        src = '''\
df = spark.table("orders")
if df.count() <= 0:
    process(df)
'''
        self.assert_rule_found(src, "CY020")


# ---------------------------------------------------------------------------
# CY025 — Missing .unpersist()
# ---------------------------------------------------------------------------

class TestCY025MissingUnpersist(RuleTestBase):
    """CY025: .cache()/.persist() without .unpersist()."""

    def test_cache_no_unpersist_fires(self):
        src = '''\
df = spark.table("orders").cache()
result = df.groupBy("date").count()
result.write.parquet("output/")
'''
        self.assert_rule_found(src, "CY025")

    def test_persist_no_unpersist_fires(self):
        src = '''\
df = spark.table("orders")
df.persist()
n = df.count()
df.write.parquet("output/")
'''
        self.assert_rule_found(src, "CY025")

    def test_partial_unpersist_fires_on_uncovered(self):
        """Two caches, only one unpersisted — fires on the other."""
        src = '''\
df1 = spark.table("orders").cache()
df2 = spark.table("events").cache()
process(df1, df2)
df2.unpersist()
'''
        matched = self.assert_rule_found(src, "CY025", count=1)
        self.assertIn("df1", matched[0].message)

    def test_unpersist_different_var_fires(self):
        src = '''\
df = spark.table("orders").cache()
other = spark.table("events")
other.unpersist()
'''
        self.assert_rule_found(src, "CY025")

    def test_with_unpersist_no_fire(self):
        src = '''\
df = spark.table("orders").cache()
result = df.groupBy("date").count()
result.write.parquet("output/")
df.unpersist()
'''
        self.assert_no_findings(src, "CY025")

    def test_unpersist_in_finally_no_fire(self):
        src = '''\
df = spark.table("orders").cache()
try:
    process(df)
finally:
    df.unpersist()
'''
        self.assert_no_findings(src, "CY025")

    def test_no_cache_no_fire(self):
        src = '''\
df = spark.table("orders")
df.write.parquet("output/")
'''
        self.assert_no_findings(src, "CY025")

    def test_nested_function_no_double_fire(self):
        """Cache in nested function should fire once, not twice."""
        src = '''\
def outer():
    def inner():
        df = spark.table("orders").cache()
        df.count()
'''
        self.assert_rule_found(src, "CY025", count=1)

    def test_partial_unpersist_no_false_finding(self):
        """Verify the unfired variable (df2) is NOT in any finding."""
        src = '''\
df1 = spark.table("orders").cache()
df2 = spark.table("events").cache()
process(df1, df2)
df2.unpersist()
'''
        findings = self.lint(src)
        cy025 = [f for f in findings if f.rule_id == "CY025"]
        self.assertEqual(len(cy025), 1)
        self.assertNotIn("df2", cy025[0].message)

    def test_chained_filter_cache_fires(self):
        src = '''\
df = spark.table("orders").filter("status='active'").cache()
result = df.groupBy("date").count()
'''
        self.assert_rule_found(src, "CY025")

    # --- Annotated assignment support ---

    def test_annotated_assign_cache_fires(self):
        """df: DataFrame = spark.table(...).cache() should be detected."""
        src = '''\
df: object = spark.table("orders").cache()
result = df.groupBy("date").count()
'''
        self.assert_rule_found(src, "CY025")

    def test_annotated_assign_with_unpersist_no_fire(self):
        src = '''\
df: object = spark.table("orders").cache()
result = df.groupBy("date").count()
df.unpersist()
'''
        self.assert_no_findings(src, "CY025")

    # --- Attribute receiver support (self.df) ---

    def test_self_df_cache_fires(self):
        """self.df.cache() standalone should be detected."""
        src = '''\
def run(self):
    self.df = spark.table("orders")
    self.df.cache()
    result = self.df.groupBy("date").count()
'''
        self.assert_rule_found(src, "CY025")

    def test_self_df_cache_with_unpersist_no_fire(self):
        src = '''\
def run(self):
    self.df = spark.table("orders")
    self.df.cache()
    result = self.df.groupBy("date").count()
    self.df.unpersist()
'''
        self.assert_no_findings(src, "CY025")

    def test_self_df_assigned_cache_fires(self):
        """self.df = spark.table(...).cache() in assignment."""
        src = '''\
def setup(self):
    self.df = spark.table("orders").cache()
    n = self.df.count()
'''
        self.assert_rule_found(src, "CY025")

    def test_self_df_assigned_cache_with_unpersist_no_fire(self):
        src = '''\
def setup(self):
    self.df = spark.table("orders").cache()
    n = self.df.count()
    self.df.unpersist()
'''
        self.assert_no_findings(src, "CY025")

    # --- Unpersist in assignment context ---

    def test_unpersist_in_assignment_no_fire(self):
        """result = df.unpersist() should count as unpersist."""
        src = '''\
df = spark.table("orders").cache()
result = df.groupBy("date").count()
cleaned = df.unpersist()
'''
        self.assert_no_findings(src, "CY025")

    # --- Persist variant with attribute receiver ---

    def test_self_df_persist_fires(self):
        src = '''\
def run(self):
    self.df = spark.table("orders")
    self.df.persist()
    result = self.df.groupBy("date").count()
'''
        self.assert_rule_found(src, "CY025")
        matched = self.assert_rule_found(src, "CY025")
        self.assertIn("self.df", matched[0].message)
        self.assertIn("persist", matched[0].message)


# ---------------------------------------------------------------------------
# CY031 — for row in df.collect()
# ---------------------------------------------------------------------------

class TestCY031CollectIteration(RuleTestBase):
    """CY031: for row in df.collect() — driver-side iteration."""

    def test_collect_in_for_loop_fires(self):
        src = '''\
df = spark.table("orders")
for row in df.collect():
    process(row["amount"])
'''
        self.assert_rule_found(src, "CY031")

    def test_to_local_iterator_fires(self):
        src = '''\
df = spark.table("orders")
for row in df.toLocalIterator():
    process(row["amount"])
'''
        self.assert_rule_found(src, "CY031")

    def test_chained_collect_fires(self):
        src = '''\
df = spark.table("orders")
for row in df.filter("status='active'").collect():
    send_email(row["email"])
'''
        self.assert_rule_found(src, "CY031")

    def test_assigned_then_iterated_fires(self):
        src = '''\
df = spark.table("orders")
rows = df.collect()
for row in rows:
    process(row)
'''
        self.assert_rule_found(src, "CY031")

    def test_topandas_iterrows_fires(self):
        src = '''\
df = spark.table("orders")
for idx, row in df.toPandas().iterrows():
    process(row)
'''
        self.assert_rule_found(src, "CY031")

    def test_list_comprehension_fires(self):
        src = '''\
df = spark.table("orders")
names = [row["name"] for row in df.collect()]
'''
        self.assert_rule_found(src, "CY031")

    def test_python_list_no_fire(self):
        src = '''\
items = [1, 2, 3]
for item in items:
    print(item)
'''
        self.assert_no_findings(src, "CY031")

    def test_collect_not_in_loop_no_fire(self):
        src = '''\
df = spark.table("orders")
result = df.collect()
print(len(result))
'''
        self.assert_no_findings(src, "CY031")

    def test_pandas_iterrows_not_pyspark_no_fire(self):
        """Pandas DataFrame iteration (not PySpark) should not fire."""
        src = '''\
import pandas as pd
pdf = pd.read_csv("data.csv")
for idx, row in pdf.iterrows():
    process(row)
'''
        self.assert_no_findings(src, "CY031")

    def test_to_local_iterator_message(self):
        """toLocalIterator should have its own message."""
        src = '''\
df = spark.table("orders")
for row in df.toLocalIterator():
    process(row)
'''
        findings = self.lint(src)
        matched = [f for f in findings if f.rule_id == "CY031"]
        self.assertEqual(len(matched), 1)
        self.assertIn("toLocalIterator", matched[0].message)

    def test_topandas_iterrows_message(self):
        """toPandas().iterrows() should have its own message."""
        src = '''\
df = spark.table("orders")
for idx, row in df.toPandas().iterrows():
    process(row)
'''
        findings = self.lint(src)
        matched = [f for f in findings if f.rule_id == "CY031"]
        self.assertEqual(len(matched), 1)
        self.assertIn("toPandas", matched[0].message)

    def test_dict_comprehension_fires(self):
        src = '''\
df = spark.table("orders")
d = {row["key"]: row["value"] for row in df.collect()}
'''
        self.assert_rule_found(src, "CY031")

    def test_topandas_assigned_then_iterated_fires(self):
        """pdf = df.toPandas(); for r in pdf.iterrows() should fire."""
        src = '''\
df = spark.table("orders")
pdf = df.toPandas()
for idx, row in pdf.iterrows():
    process(row)
'''
        self.assert_rule_found(src, "CY031")

    def test_chained_select_collect_fires(self):
        src = '''\
df = spark.table("orders")
for row in df.select("a").collect():
    process(row)
'''
        self.assert_rule_found(src, "CY031")

    def test_assigned_collect_list_comprehension_fires(self):
        src = '''\
df = spark.table("orders")
rows = df.collect()
names = [r["name"] for r in rows]
'''
        self.assert_rule_found(src, "CY031")


# ---------------------------------------------------------------------------
# CY032 — drop-before-select
# ---------------------------------------------------------------------------

class TestCY032DropSelect(RuleTestBase):
    """CY032: .drop() followed by .select() is redundant."""

    def test_drop_then_select_fires(self):
        src = '''\
df = spark.table("orders")
result = df.drop("tmp").select("id", "name")
'''
        self.assert_rule_found(src, "CY032")

    def test_drop_then_select_chained_fires(self):
        src = '''\
df = spark.table("orders")
result = df.filter(df.id > 0).drop("tmp").select("id", "name")
'''
        self.assert_rule_found(src, "CY032")

    def test_drop_alone_no_finding(self):
        src = '''\
df = spark.table("orders")
result = df.drop("tmp")
'''
        self.assert_no_findings(src, "CY032")

    def test_select_alone_no_finding(self):
        src = '''\
df = spark.table("orders")
result = df.select("id", "name")
'''
        self.assert_no_findings(src, "CY032")

    def test_untracked_df_no_finding(self):
        src = '''\
result = some_list.drop(0).select("x")
'''
        self.assert_no_findings(src, "CY032")

    def test_select_before_drop_no_finding(self):
        src = '''\
df = spark.table("orders")
result = df.select("id", "name").drop("id")
'''
        self.assert_no_findings(src, "CY032")


# ---------------------------------------------------------------------------
# CY033 — collect-list-dedup
# ---------------------------------------------------------------------------

class TestCY033CollectListDedup(RuleTestBase):
    """CY033: array_distinct(collect_list(x)) should be collect_set(x)."""

    def test_bare_functions_fire(self):
        self.assert_rule_found(
            'result = df.agg(array_distinct(collect_list("col")))',
            "CY033",
        )

    def test_module_prefix_fire(self):
        self.assert_rule_found(
            'result = df.agg(F.array_distinct(F.collect_list("col")))',
            "CY033",
        )

    def test_mixed_prefix_fire(self):
        self.assert_rule_found(
            'result = df.agg(array_distinct(F.collect_list("col")))',
            "CY033",
        )

    def test_collect_set_no_finding(self):
        self.assert_no_findings(
            'result = df.agg(collect_set("col"))',
            "CY033",
        )

    def test_array_distinct_non_collect_list_no_finding(self):
        self.assert_no_findings(
            'result = array_distinct(some_array)',
            "CY033",
        )

    def test_collect_list_alone_no_finding(self):
        self.assert_no_findings(
            'result = df.agg(collect_list("col"))',
            "CY033",
        )


# ---------------------------------------------------------------------------
# CY034 — rdd-collect
# ---------------------------------------------------------------------------

class TestCY034RddCollect(RuleTestBase):
    """CY034: df.rdd.collect() should be replaced with .toPandas()."""

    def test_rdd_collect_on_tracked_df_fires(self):
        src = '''\
df = spark.table("orders")
rows = df.rdd.collect()
'''
        self.assert_rule_found(src, "CY034")

    def test_rdd_collect_on_chained_df_fires(self):
        src = '''\
df = spark.table("orders")
rows = df.filter(df.id > 0).rdd.collect()
'''
        self.assert_rule_found(src, "CY034")

    def test_df_collect_without_rdd_no_finding(self):
        """Plain .collect() is handled by CY001, not CY034."""
        src = '''\
df = spark.table("orders")
rows = df.limit(10).collect()
'''
        self.assert_no_findings(src, "CY034")

    def test_rdd_collect_on_untracked_no_finding(self):
        src = 'rows = some_rdd.collect()'
        self.assert_no_findings(src, "CY034")

    def test_topandas_no_finding(self):
        src = '''\
df = spark.table("orders")
pdf = df.toPandas()
'''
        self.assert_no_findings(src, "CY034")


# ---------------------------------------------------------------------------
# CY035 — union-column-order
# ---------------------------------------------------------------------------

class TestCY035UnionByName(RuleTestBase):
    """CY035: .union() / .unionAll() should be .unionByName()."""

    def test_union_on_tracked_df_fires(self):
        src = '''\
df1 = spark.table("orders")
df2 = spark.table("orders_archive")
result = df1.union(df2)
'''
        self.assert_rule_found(src, "CY035")

    def test_union_all_on_tracked_df_fires(self):
        src = '''\
df1 = spark.table("orders")
df2 = spark.table("orders_archive")
result = df1.unionAll(df2)
'''
        self.assert_rule_found(src, "CY035")

    def test_union_by_name_no_finding(self):
        src = '''\
df1 = spark.table("orders")
df2 = spark.table("orders_archive")
result = df1.unionByName(df2)
'''
        self.assert_no_findings(src, "CY035")

    def test_untracked_union_no_finding(self):
        """Python set.union() should not be flagged."""
        src = 'result = set_a.union(set_b)'
        self.assert_no_findings(src, "CY035")

    def test_union_in_chain_fires(self):
        src = '''\
df1 = spark.table("orders")
df2 = spark.table("orders_archive")
result = df1.filter(df1.id > 0).union(df2)
'''
        self.assert_rule_found(src, "CY035")


# ---------------------------------------------------------------------------
# CY036 — insert-into-deprecated
# ---------------------------------------------------------------------------

class TestCY036InsertInto(RuleTestBase):
    """CY036: df.write.insertInto() should be replaced with writeTo API."""

    def test_insert_into_no_overwrite_fires(self):
        src = '''\
df = spark.table("orders")
df.write.insertInto("output_table")
'''
        self.assert_rule_found(src, "CY036")

    def test_insert_into_overwrite_false_fires(self):
        src = '''\
df = spark.table("orders")
df.write.insertInto("output_table", overwrite=False)
'''
        self.assert_rule_found(src, "CY036")

    def test_insert_into_overwrite_true_fires(self):
        src = '''\
df = spark.table("orders")
df.write.insertInto("output_table", overwrite=True)
'''
        findings = self.assert_rule_found(src, "CY036")
        self.assertIn("overwritePartitions", findings[0].message)

    def test_write_to_append_no_finding(self):
        src = '''\
df = spark.table("orders")
df.writeTo("output_table").append()
'''
        self.assert_no_findings(src, "CY036")

    def test_untracked_df_no_finding(self):
        src = 'some_obj.write.insertInto("table")'
        self.assert_no_findings(src, "CY036")

    def test_insert_into_append_message(self):
        src = '''\
df = spark.table("orders")
df.write.insertInto("output_table")
'''
        findings = self.assert_rule_found(src, "CY036")
        self.assertIn("append", findings[0].message)


# ---------------------------------------------------------------------------
# CY037 — functions-no-alias
# ---------------------------------------------------------------------------

class TestCY037FunctionsAlias(RuleTestBase):
    """CY037: pyspark.sql.functions should be imported as F."""

    def test_bare_import_fires(self):
        self.assert_rule_found(
            'from pyspark.sql import functions',
            "CY037",
        )

    def test_wrong_alias_fires(self):
        self.assert_rule_found(
            'from pyspark.sql import functions as funcs',
            "CY037",
        )

    def test_correct_alias_no_finding(self):
        self.assert_no_findings(
            'from pyspark.sql import functions as F',
            "CY037",
        )

    def test_other_pyspark_import_no_finding(self):
        self.assert_no_findings(
            'from pyspark.sql import SparkSession',
            "CY037",
        )

    def test_functions_from_other_module_no_finding(self):
        self.assert_no_findings(
            'from mymodule import functions',
            "CY037",
        )

    def test_combined_import_with_correct_alias_no_finding(self):
        self.assert_no_findings(
            'from pyspark.sql import SparkSession, functions as F',
            "CY037",
        )

    def test_combined_import_without_alias_fires(self):
        self.assert_rule_found(
            'from pyspark.sql import SparkSession, functions',
            "CY037",
        )


if __name__ == "__main__":
    unittest.main()
