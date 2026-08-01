import json
import os
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

ScraperRouter = import_module("tldw_Server_API.app.core.Web_Scraping.scraper_router").ScraperRouter

PINNED_PREDECESSOR_COMMIT = "c2a1695744032245acdb1cd115dd888586dc9623"
DEFAULT_PREDECESSOR_ROOT = Path("/private/tmp/tldw-phase4-predecessor-c2a16957")
ROUTER_URL = "https://example.com/path"
_PLAN_FIELDS = (
    "backend",
    "handler",
    "ua_profile",
    "impersonate",
    "extra_headers",
    "cookies",
    "respect_robots",
    "proxies",
    "strategy_order",
    "schema_rules",
    "llm_settings",
    "regex_settings",
    "cluster_settings",
)


def _ordinary_yaml_cases() -> list[dict[str, Any]]:
    boundaries: list[tuple[str, Any]] = [
        ("null", None),
        ("empty-string", ""),
        ("string", "value"),
        ("scalar", 7),
        ("empty-list", []),
        ("list", ["value", None, 7]),
        ("empty-mapping", {}),
        ("mapping", {"nested": None}),
    ]
    cases = [
        {"name": f"{field}-{name}", "rule": {field: value}}
        for field in (
            "backend",
            "handler",
            "ua_profile",
            "impersonate",
            "extra_headers",
            "cookies",
            "respect_robots",
            "proxies",
            "strategy_order",
        )
        for name, value in boundaries
    ]
    cases.extend(
        {
            "name": f"{field}-mixed-mapping",
            "rule": {
                field: {
                    "X-Null": None,
                    "X-Integer": 7,
                    "X-List": ["value"],
                    "X-Mapping": {"nested": True},
                }
            },
        }
        for field in ("extra_headers", "cookies", "proxies")
    )
    cases.extend(
        {
            "name": f"{field}-list-pairs",
            "rule": {field: [["X-Null", None], ["X-Integer", 7]]},
        }
        for field in ("extra_headers", "cookies", "proxies")
    )
    for primary, alias in (
        ("schema_rules", "schema"),
        ("llm_settings", "llm"),
        ("regex_settings", "regex"),
        ("cluster_settings", "cluster"),
    ):
        for name, value in boundaries:
            cases.append(
                {
                    "name": f"{primary}-{name}-before-valid-alias",
                    "rule": {primary: value, alias: {"source": "alias"}},
                }
            )
            cases.append(
                {
                    "name": f"{alias}-{name}",
                    "rule": {alias: value},
                }
            )
    return cases


_ORDINARY_YAML_CASES = _ordinary_yaml_cases()


def _capture_current(case: dict[str, Any]) -> dict[str, Any]:
    rules = {"domains": {"example.com": case["rule"]}}
    result: dict[str, Any] = {"name": case["name"]}
    for path in ("validation", "direct", "validated"):
        try:
            if path == "validation":
                result[path] = ScraperRouter.validate_rules(rules)
            elif path == "direct":
                result[path] = asdict(ScraperRouter(rules).resolve(ROUTER_URL))
            else:
                cleaned = ScraperRouter.validate_rules(rules)
                result[path] = asdict(ScraperRouter(cleaned).resolve(ROUTER_URL))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - assertion payload only
            result[path] = {"__error__": type(exc).__name__}
    return result


@lru_cache(maxsize=1)
def _capture_predecessor() -> dict[str, dict[str, Any]]:
    configured_root = os.environ.get("TLDW_PHASE4_PREDECESSOR_ROOT")
    predecessor_root = Path(configured_root) if configured_root else DEFAULT_PREDECESSOR_ROOT
    if not predecessor_root.is_dir():
        pytest.skip(f"pinned predecessor checkout is unavailable: {predecessor_root}")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=predecessor_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == PINNED_PREDECESSOR_COMMIT

    script = """
import json
import sys
from dataclasses import asdict

from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter

url = sys.argv[1]
cases = json.loads(sys.argv[2])
results = []
for case in cases:
    rules = {"domains": {"example.com": case["rule"]}}
    result = {"name": case["name"]}
    for path in ("validation", "direct", "validated"):
        try:
            if path == "validation":
                result[path] = ScraperRouter.validate_rules(rules)
            elif path == "direct":
                result[path] = asdict(ScraperRouter(rules).resolve(url))
            else:
                cleaned = ScraperRouter.validate_rules(rules)
                result[path] = asdict(ScraperRouter(cleaned).resolve(url))
        except Exception as exc:
            result[path] = {"__error__": type(exc).__name__}
    results.append(result)
print(json.dumps(results, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, ROUTER_URL, json.dumps(_ORDINARY_YAML_CASES)],
        cwd=predecessor_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {result["name"]: result for result in json.loads(completed.stdout)}


def _current_by_name() -> dict[str, dict[str, Any]]:
    return {case["name"]: _capture_current(case) for case in _ORDINARY_YAML_CASES}


def _successful_differences(path: str) -> list[dict[str, Any]]:
    predecessor = _capture_predecessor()
    current = _current_by_name()
    differences = []
    for name, expected in predecessor.items():
        if "__error__" in expected[path]:
            continue
        if current[name][path] != expected[path]:
            differences.append(
                {
                    "name": name,
                    "predecessor": expected[path],
                    "current": current[name][path],
                }
            )
    return differences


@pytest.mark.parametrize("path", ["validation", "direct", "validated"])
def test_ordinary_yaml_results_match_pinned_predecessor_independently(path: str) -> None:
    assert _successful_differences(path) == []


def test_current_router_remains_fail_safe_where_pinned_predecessor_raised() -> None:
    predecessor = _capture_predecessor()
    current = _current_by_name()
    predecessor_errors = [
        (name, path)
        for name, result in predecessor.items()
        for path in ("validation", "direct", "validated")
        if "__error__" in result[path]
    ]

    assert predecessor_errors
    assert all("__error__" not in current[name][path] for name, path in predecessor_errors)


def test_matrix_covers_every_non_regex_rule_field_and_alias_pair() -> None:
    covered_fields = {next(iter(case["rule"])) for case in _ORDINARY_YAML_CASES}

    assert set(_PLAN_FIELDS) - {"schema_rules", "llm_settings", "regex_settings", "cluster_settings"} <= covered_fields
    assert {
        "schema_rules",
        "schema",
        "llm_settings",
        "llm",
        "regex_settings",
        "regex",
        "cluster_settings",
        "cluster",
    } <= covered_fields
