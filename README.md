# cylint

A PySpark linter that catches various antipatterns.

Static analysis for PySpark code. No Spark runtime needed. Zero dependencies. Runs anywhere Python runs.

## Install

```bash
pip install cylint
```

## Usage

```bash
# Lint files or directories
cy lint src/pipelines/

# JSON output for CI
cy lint --format json src/

# Only warnings and critical
cy lint --min-severity warning .
```

Example output:

```
pipeline.py:47:8: CY003 [critical] .withColumn() inside a loop creates O(n²) plan complexity.
  Use .select([...]) with all column expressions instead.

pipeline.py:82:4: CY001 [warning] .collect() called without filtering.
  Consider .limit(N).collect(), .take(N), or using .show() for inspection.

pipeline.py:103:4: CY005 [warning] .cache() with single downstream use.
  Cache is only beneficial when the same DataFrame is used in multiple actions.

Found 3 issues (1 critical, 2 warnings) in 1 file.
```

## Rules

| Rule | Severity | What it catches |
|------|----------|----------------|
| [CY001](https://clusteryield.app/analysis-reference.html#CY001) | warning | `.collect()` without `.filter()` or `.limit()` — the #1 OOM cause |
| [CY002](https://clusteryield.app/analysis-reference.html#CY002) | warning | UDF where a builtin exists (e.g. `udf(lambda x: x.lower())` → `F.lower()`) |
| [CY003](https://clusteryield.app/analysis-reference.html#CY003) | critical | `.withColumn()` in a loop — creates O(n²) Catalyst plans |
| [CY004](https://clusteryield.app/analysis-reference.html#CY004) | info | `SELECT *` in `spark.sql()` strings — prevents column pruning |
| [CY005](https://clusteryield.app/analysis-reference.html#CY005) | warning | `.cache()` / `.persist()` with ≤1 downstream use — wastes memory |
| [CY006](https://clusteryield.app/analysis-reference.html#CY006) | warning | `.toPandas()` on unfiltered DataFrame — collects everything to driver |
| [CY007](https://clusteryield.app/analysis-reference.html#CY007) | critical | `.crossJoin()` or `.join()` without condition — cartesian product |
| [CY008](https://clusteryield.app/analysis-reference.html#CY008) | info | `.repartition()` before `.write()` — unnecessary shuffle |
| [CY009](https://clusteryield.app/analysis-reference.html#CY009) | critical | UDF in `.filter()`/`.where()` — blocks predicate pushdown |
| [CY010](https://clusteryield.app/analysis-reference.html#CY010) | warning | `.join()` without explicit `how=` — ambiguous join type |
| [CY011](https://clusteryield.app/analysis-reference.html#CY011) | warning | `.withColumnRenamed()`/`.drop()` in a loop — O(n²) plan nodes |
| [CY012](https://clusteryield.app/analysis-reference.html#CY012) | warning | `.show()`/`.display()`/`.printSchema()` left in production code |
| [CY013](https://clusteryield.app/analysis-reference.html#CY013) | warning | `.coalesce(1)` before `.write()` — single-executor bottleneck |
| [CY014](https://clusteryield.app/analysis-reference.html#CY014) | critical | Multiple actions without `.cache()` — recomputes full lineage each time |
| [CY015](https://clusteryield.app/analysis-reference.html#CY015) | critical | Non-equi `.join()` condition — implicit cartesian product |
| [CY016](https://clusteryield.app/analysis-reference.html#CY016) | info | Invalid escape sequence in string literal — use raw strings for regex |
| [CY017](https://clusteryield.app/analysis-reference.html#CY017) | warning | `Window.orderBy()` without `.partitionBy()` — full-table sort into one partition |
| [CY018](https://clusteryield.app/analysis-reference.html#CY018) | warning | `spark.read.csv()`/`.json()` without explicit schema — double file scan |
| [CY020](https://clusteryield.app/analysis-reference.html#CY020) | warning | `.count() == 0` for emptiness check — full scan wasted |
| [CY025](https://clusteryield.app/analysis-reference.html#CY025) | warning | `.cache()`/`.persist()` without `.unpersist()` — memory leak |
| [CY031](https://clusteryield.app/analysis-reference.html#CY031) | warning | `for row in df.collect()` — driver-side row iteration defeats Spark |

List all rules:

```bash
cy rules
```

## How it works

`cylint` uses Python's `ast` module to parse your source files and track DataFrame variables through assignment chains. It knows that anything coming from `spark.read.*`, `spark.sql()`, or `spark.table()` is a DataFrame, and follows method chains from there.

No type stubs. No Spark installation. No imports resolved. Just fast, heuristic analysis that catches the patterns that matter.

## Configuration

Out of the box, every rule runs at its default severity with no exclusions. No config file needed.

If a rule doesn't apply to your codebase, or you want to skip certain directories, drop a `.cylint.yml` in your project root or add a `[tool.cylint]` section to your existing `pyproject.toml`. The linter picks it up automatically.

### .cylint.yml

```yaml
# Only fail on warnings and above (ignore info-level findings)
min-severity: warning

rules:
  CY004: off        # we use SELECT * intentionally in dynamic queries
  CY008: warning    # promote repartition-before-write to warning

exclude:
  - tests/
  - vendor/
  - notebooks/scratch/
```

### pyproject.toml

```toml
[tool.cylint]
min-severity = "warning"
exclude = ["tests/", "notebooks/scratch/"]

[tool.cylint.rules]
CY004 = "off"
CY008 = "warning"
```

## Inline Suppression

Suppress individual findings with `# cy:ignore` comments:

```python
df.collect()  # cy:ignore CY001

# Suppress multiple rules
df.show()  # cy:ignore CY001,CY012

# Suppress all rules on a line
df.collect()  # cy:ignore
```

## CI Integration

### GitHub Actions

```yaml
name: PySpark Lint
on: pull_request

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install cylint
      - run: cy lint --format github src/
```

The `--format github` flag outputs findings as workflow annotations — they appear inline on the PR diff.

### pre-commit

```yaml
repos:
  - repo: https://github.com/clusteryield/cylint
    hooks:
      - id: spark-lint
        args: [--min-severity, warning]
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | No findings |
| 1 | Warnings or info findings |
| 2 | Critical findings |

## License

Apache 2.0
