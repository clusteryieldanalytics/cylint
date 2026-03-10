"""CLI entry point for cylint linter.

Usage:
    cy lint [files/dirs...]
    cy lint --format json src/
    cy lint --min-severity warning src/
    cy rules                          # list available rules
"""

import argparse
import sys
from pathlib import Path

from cylint.config import Config
from cylint.engine import LintEngine
from cylint.formatters import text as text_fmt
from cylint.formatters import json as json_fmt
from cylint.formatters import github as github_fmt
from cylint.models import Severity
from cylint.rules import get_all_rules


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        prog="cy",
        description="cylint: PySpark anti-pattern linter",
    )
    subparsers = parser.add_subparsers(dest="command")

    # cy lint
    lint_parser = subparsers.add_parser("lint", help="Lint PySpark files")
    lint_parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to lint (default: current directory)",
    )
    lint_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "github"],
        default="text",
        help="Output format (default: text)",
    )
    lint_parser.add_argument(
        "--min-severity",
        choices=["info", "warning", "critical"],
        default=None,
        help="Minimum severity to report",
    )
    lint_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    lint_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    lint_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Paths to exclude (can be repeated)",
    )
    lint_parser.add_argument(
        "--export-cells",
        action="store_true",
        help="Include Databricks notebook cell maps in JSON output",
    )

    # cy rules
    rules_parser = subparsers.add_parser("rules", help="List available rules")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "rules":
        return _cmd_rules()

    if args.command == "lint":
        return _cmd_lint(args)

    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    """Execute the lint command."""
    # Load config
    config = Config.find_and_load()

    # CLI args override config
    min_severity = config.min_severity
    if args.min_severity:
        min_severity = Severity.from_string(args.min_severity)

    exclude = config.exclude + args.exclude

    # Build rule overrides
    enabled_rules: dict[str, Severity] = {}
    disabled_rules: set[str] = set()
    for rule_id, severity in config.rules.items():
        if severity is None:
            disabled_rules.add(rule_id)
        else:
            enabled_rules[rule_id] = severity

    # Create engine and run
    engine = LintEngine(
        enabled_rules=enabled_rules or None,
        disabled_rules=disabled_rules or None,
        min_severity=min_severity,
    )

    result = engine.lint_paths(args.paths, exclude=exclude)

    # Format output
    use_color = not args.no_color and sys.stdout.isatty()

    if args.format == "json":
        print(json_fmt.format_result(result, export_cells=args.export_cells))
    elif args.format == "github":
        print(github_fmt.format_result(result))
    else:
        print(text_fmt.format_result(result, use_color=use_color))

    return result.exit_code


def _cmd_rules() -> int:
    """List all available rules."""
    rules = get_all_rules()
    print(f"\nAvailable rules ({len(rules)}):\n")
    for rule_id in sorted(rules.keys()):
        rule_cls = rules[rule_id]
        meta = rule_cls.META
        print(f"  {meta.rule_id}  [{meta.default_severity}]  {meta.name}")
        print(f"         {meta.description}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
