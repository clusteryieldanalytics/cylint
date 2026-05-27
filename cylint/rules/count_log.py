"""CY032: .count() used only for logging or display — triggers a full scan."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name

# Logging method names that indicate the call is display-only
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "critical", "log", "exception"}
)

# Bare function names treated as display-only calls
_PRINT_NAMES = frozenset({"print", "display"})

_MESSAGE = (
    ".count() on `{df}` triggers a full DataFrame scan but the result is "
    "only used for display/logging. Remove the call or cache the count if "
    "it is needed elsewhere."
)

_SUGGESTION = (
    "If you only need to confirm data exists, use `df.head(1)`. "
    "If the count is reused, assign it once and reference the variable."
)


def _name_looks_like_logger(name: str) -> bool:
    """Return True if *name* case-insensitively contains 'log'.

    Matches: logger, LOG, logging, app_logger, myLog, audit_log, …
    """
    return "log" in name.lower()


def _receiver_looks_like_logger(node: ast.expr) -> bool:
    """Return True if any segment of the attribute chain looks like a logger.

    Handles: logger.info, self.logger.info, app.log.debug, logging.info, …
    """
    current = node
    while True:
        if isinstance(current, ast.Name):
            return _name_looks_like_logger(current.id)
        if isinstance(current, ast.Attribute):
            if _name_looks_like_logger(current.attr):
                return True
            current = current.value
        else:
            return False


def _is_log_or_print_call(node: ast.Call) -> bool:
    """Return True if *node* is a print(), display(), or logging call."""
    func = node.func
    # print(...) / display(...)
    if isinstance(func, ast.Name) and func.id in _PRINT_NAMES:
        return True
    # logger.info(...) / logging.debug(...) / self.logger.warning(...) / …
    if isinstance(func, ast.Attribute):
        if func.attr in _LOG_METHODS and _receiver_looks_like_logger(func.value):
            return True
    return False


def _find_tracked_count(node: ast.expr, tracker: DataFrameTracker) -> ast.Call | None:
    """Recursively search *node* for a tracked df.count() call.

    Descends into f-string (JoinedStr) FormattedValue children so that
    patterns like f"rows={df.count()}" are detected.

    Returns the count call AST node if found, else None.
    """
    if not isinstance(node, ast.AST):
        return None

    # Direct: df.count()
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "count"
        and not node.args
        and not node.keywords
    ):
        root = find_root_name(node.func.value)
        if root is not None and tracker.is_tracked(root):
            return node

    # f-string: descend into FormattedValue children
    if isinstance(node, ast.JoinedStr):
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                result = _find_tracked_count(value.value, tracker)
                if result is not None:
                    return result
        return None

    # Recurse into child nodes (e.g. str concat, nested calls used as args)
    for child in ast.iter_child_nodes(node):
        result = _find_tracked_count(child, tracker)
        if result is not None:
            return result

    return None


@register_rule
class CountLogRule(BaseRule):
    META = RuleMeta(
        rule_id="CY021",
        name="count-log",
        description=".count() used only for logging/display — triggers a full scan",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []

        for node in ast.walk(tree):
            # --- Pattern 1: Bare expression statement --- df.count()
            if isinstance(node, ast.Expr):
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "count"
                    and not call.args
                    and not call.keywords
                ):
                    root = find_root_name(call.func.value)
                    if root is not None and tracker.is_tracked(root):
                        findings.append(self._make_finding(
                            filepath=filepath,
                            line=call.lineno,
                            col=call.col_offset,
                            message=_MESSAGE.format(df=root),
                            suggestion=_SUGGESTION,
                        ))
                continue

            # --- Pattern 2: .count() inside print/display/log call args ---
            if not isinstance(node, ast.Call):
                continue
            if not _is_log_or_print_call(node):
                continue

            for arg in node.args:
                count_node = _find_tracked_count(arg, tracker)
                if count_node is not None:
                    root = find_root_name(count_node.func.value)
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=count_node.lineno,
                        col=count_node.col_offset,
                        message=_MESSAGE.format(df=root),
                        suggestion=_SUGGESTION,
                    ))
                    break  # one finding per log call is enough

        return findings
