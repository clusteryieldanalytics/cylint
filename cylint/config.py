"""Configuration file loader.

Supports:
  - pyproject.toml: [tool.cylint] section
  - .cylint.yml / .cylint.yaml (if PyYAML is available)
  - Inline suppression: # cy:ignore or # cy:ignore CY001,CY003
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from cylint.models import Severity

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


@dataclass
class Config:
    """Linter configuration."""
    min_severity: Severity = Severity.INFO
    exclude: list[str] = field(default_factory=list)
    rules: dict[str, Severity | None] = field(default_factory=dict)  # rule_id → severity or None (disabled)

    @staticmethod
    def find_and_load(start_dir: str | None = None) -> "Config":
        """Find and load config from project root, walking up from start_dir."""
        search_dir = Path(start_dir) if start_dir else Path.cwd()

        # Walk up looking for config files
        for directory in [search_dir] + list(search_dir.parents):
            # Check pyproject.toml
            pyproject = directory / "pyproject.toml"
            if pyproject.exists():
                config = _load_pyproject(pyproject)
                if config is not None:
                    return config

            # Check .cylint.yml
            for name in (".cylint.yml", ".cylint.yaml"):
                yml_path = directory / name
                if yml_path.exists():
                    return _load_yaml(yml_path)

        return Config()


def _load_pyproject(path: Path) -> "Config | None":
    """Load config from pyproject.toml [tool.cylint] section."""
    if tomllib is None:
        return None

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None

    tool_config = data.get("tool", {}).get("cylint")
    if tool_config is None:
        return None

    return _parse_config_dict(tool_config)


def _load_yaml(path: Path) -> "Config":
    """Load config from YAML file."""
    # Use a simple parser to avoid PyYAML dependency
    # Supports the subset we need: simple key-value and lists
    try:
        text = path.read_text(encoding="utf-8")
        return _parse_simple_yaml(text)
    except Exception:
        return Config()


def _parse_simple_yaml(text: str) -> "Config":
    """Minimal YAML-like parser for our config format.

    Handles:
      min-severity: warning
      rules:
        CY001: warning
        CY004: off
      exclude:
        - tests/
        - notebooks/
    """
    config = Config()
    current_section = None
    lines = text.strip().split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        if indent == 0 and ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if key == "min-severity" and value:
                try:
                    config.min_severity = Severity.from_string(value)
                except (KeyError, ValueError):
                    pass
            elif key in ("rules", "exclude"):
                current_section = key
                if value:
                    current_section = None  # inline value, not a section
            else:
                current_section = None

        elif indent > 0 and current_section:
            if current_section == "rules":
                # CY001: warning
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip().strip("'\"")
                if value.lower() == "off":
                    config.rules[key] = None
                else:
                    try:
                        config.rules[key] = Severity.from_string(value)
                    except (KeyError, ValueError):
                        pass
            elif current_section == "exclude":
                # - tests/
                if stripped.startswith("- "):
                    config.exclude.append(stripped[2:].strip().strip("'\""))

    return config


def _parse_config_dict(data: dict) -> "Config":
    """Parse a config dictionary (from TOML or similar)."""
    config = Config()

    if "min-severity" in data:
        try:
            config.min_severity = Severity.from_string(data["min-severity"])
        except (KeyError, ValueError):
            pass

    if "exclude" in data:
        config.exclude = list(data["exclude"])

    if "rules" in data:
        for rule_id, value in data["rules"].items():
            if isinstance(value, str):
                if value.lower() == "off":
                    config.rules[rule_id] = None
                else:
                    try:
                        config.rules[rule_id] = Severity.from_string(value)
                    except (KeyError, ValueError):
                        pass

    return config


def parse_inline_suppression(comment: str) -> set[str] | None:
    """Parse a cy:ignore comment.

    Returns:
        Set of rule IDs to suppress, or empty set to suppress all.
        None if comment is not a suppression.
    """
    comment = comment.strip().lstrip("#").strip()
    if not comment.startswith("cy:ignore"):
        return None

    rest = comment[len("cy:ignore"):].strip()
    if not rest:
        return set()  # suppress all rules on this line

    return {r.strip() for r in rest.split(",") if r.strip()}
