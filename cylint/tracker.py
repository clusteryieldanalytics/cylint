"""AST-based heuristic DataFrame variable tracker.

Tracks variables that are likely Spark DataFrames by following
assignment chains from known DataFrame sources (spark.read.*,
spark.sql(), spark.table(), etc.) and method chains on tracked
DataFrames.
"""

import ast
from dataclasses import dataclass, field
from typing import Optional


# Methods that return a new DataFrame when called on a DataFrame
DATAFRAME_METHODS = frozenset({
    "select", "selectExpr", "filter", "where", "groupBy", "groupby",
    "agg", "orderBy", "sort", "sortWithinPartitions", "limit", "drop",
    "dropDuplicates", "drop_duplicates", "distinct", "union", "unionAll",
    "unionByName", "intersect", "intersectAll", "subtract", "exceptAll",
    "join", "crossJoin", "withColumn", "withColumnRenamed",
    "withColumnsRenamed", "withColumns", "alias", "toDF", "sample",
    "sampleBy", "na", "fillna", "dropna", "replace", "repartition",
    "coalesce", "cache", "persist", "unpersist", "checkpoint",
    "localCheckpoint", "hint", "repartitionByRange",
    "transform", "withMetadata", "withWatermark",
})

# Methods that indicate a terminal action (not returning a DataFrame)
ACTION_METHODS = frozenset({
    "collect", "count", "first", "head", "take", "takeOrdered",
    "show", "display", "toPandas", "write", "writeStream",
    "foreach", "foreachBatch", "printSchema", "explain",
    "describe", "summary", "toLocalIterator", "toJSON",
})

# Methods that indicate filtering/limiting has been applied
FILTER_METHODS = frozenset({
    "filter", "where", "limit", "head", "take", "sample", "agg",
    "groupBy", "groupby",
})


@dataclass
class ChainInfo:
    """Information about a DataFrame's method chain."""
    source_line: int
    has_filter: bool = False
    has_cache: bool = False
    has_persist: bool = False
    use_count: int = 0  # times referenced after definition


@dataclass
class DataFrameTracker:
    """Tracks variables that are likely DataFrames through AST analysis."""
    dataframes: dict[str, ChainInfo] = field(default_factory=dict)

    def track(self, name: str, line: int, chain_info: Optional[ChainInfo] = None):
        """Register a variable as a tracked DataFrame."""
        if chain_info is None:
            chain_info = ChainInfo(source_line=line)
        self.dataframes[name] = chain_info

    def is_tracked(self, name: str) -> bool:
        return name in self.dataframes

    def get_info(self, name: str) -> Optional[ChainInfo]:
        return self.dataframes.get(name)

    def record_use(self, name: str):
        if name in self.dataframes:
            self.dataframes[name].use_count += 1


def unwrap_cache(node: ast.expr) -> tuple[ast.expr, bool]:
    """Unwrap outer .cache()/.persist() calls, returning (inner_node, had_cache).

    Allows callers to detect patterns like spark.table("orders").cache()
    where cache wraps a spark source expression.
    """
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("cache", "persist")):
        return node.func.value, True
    return node, False


def is_spark_source(node: ast.expr) -> bool:
    """Check if an expression is a known DataFrame source.

    Matches patterns like:
        spark.read.csv(...)
        spark.read.parquet(...)
        spark.read.format(...).load()
        spark.sql("...")
        spark.table("...")
        spark.createDataFrame(...)
        spark.range(...)

    Also sees through wrapping .cache()/.persist() calls.
    """
    node, _ = unwrap_cache(node)
    if isinstance(node, ast.Call):
        func = node.func
        # spark.sql(...), spark.table(...), spark.createDataFrame(...), spark.range(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in ("spark", "sqlContext", "sc"):
                if func.attr in ("sql", "table", "createDataFrame", "range"):
                    return True
        # spark.read.csv(...), spark.read.parquet(...), etc.
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            if isinstance(func.value.value, ast.Name):
                if func.value.value.id in ("spark", "sqlContext"):
                    if func.value.attr == "read":
                        return True
        # spark.read.format("...").load()
        if isinstance(func, ast.Attribute) and func.attr == "load":
            inner = func.value
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                if inner.func.attr == "format":
                    return True
        # Chained method on spark.read (e.g., spark.read.option(...).csv(...))
        if _is_read_chain(node):
            return True
    return False


def _is_read_chain(node: ast.AST) -> bool:
    """Recursively check if a call chain starts from spark.read."""
    if isinstance(node, ast.Call):
        return _is_read_chain(node.func)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                if node.value.value.id in ("spark", "sqlContext") and node.value.attr == "read":
                    return True
        return _is_read_chain(node.value)
    return False


def is_dataframe_method_chain(node: ast.expr, tracker: DataFrameTracker) -> Optional[str]:
    """Check if an expression is a method chain on a tracked DataFrame.

    Returns the root DataFrame variable name if found, None otherwise.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in DATAFRAME_METHODS:
                return _find_root_df(func.value, tracker)
    return None


def _find_root_df(node: ast.expr, tracker: DataFrameTracker) -> Optional[str]:
    """Walk a method chain to find the root DataFrame variable."""
    if isinstance(node, ast.Name):
        if tracker.is_tracked(node.id):
            return node.id
        return None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return _find_root_df(node.func.value, tracker)
    if isinstance(node, ast.Attribute):
        return _find_root_df(node.value, tracker)
    return None


def get_chain_methods(node: ast.expr) -> list[str]:
    """Extract all method names from a chained call expression.

    E.g., df.filter(...).select(...).collect() → ['filter', 'select', 'collect']
    """
    methods = []
    current = node
    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                methods.append(current.func.attr)
                current = current.func.value
            else:
                break
        elif isinstance(current, ast.Attribute):
            methods.append(current.attr)
            current = current.value
        else:
            break
    methods.reverse()
    return methods


def chain_has_filter(node: ast.expr) -> bool:
    """Check if a method chain includes any filtering/limiting operation."""
    methods = get_chain_methods(node)
    return bool(FILTER_METHODS & set(methods))


def find_root_name(node: ast.expr) -> Optional[str]:
    """Find the root variable name in a chain expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return find_root_name(node.func)
    if isinstance(node, ast.Attribute):
        return find_root_name(node.value)
    return None
