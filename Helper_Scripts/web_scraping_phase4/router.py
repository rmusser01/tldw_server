"""Router cases for immutable Phase 4 predecessor fixtures."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

ROUTER_URL = "https://example.com/path"


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


def _capture_path(router_class: Any, rule: dict[str, Any], path: str) -> dict[str, Any]:
    rules = {"domains": {"example.com": rule}}
    try:
        if path == "validation":
            value = router_class.validate_rules(rules)
        elif path == "direct":
            value = asdict(router_class(rules).resolve(ROUTER_URL))
        else:
            cleaned = router_class.validate_rules(rules)
            value = asdict(router_class(cleaned).resolve(ROUTER_URL))
    except Exception as exc:  # noqa: BLE001 - exception type is predecessor behavior
        return {"status": "error", "type": type(exc).__name__}
    return {"status": "ok", "value": value}


def build_router_cases(router_class: Any) -> list[dict[str, Any]]:
    """Capture direct and validated router behavior from the loaded predecessor."""
    return [
        {
            "expected": {
                path: _capture_path(router_class, case["rule"], path) for path in ("validation", "direct", "validated")
            },
            "name": case["name"],
            "rule": case["rule"],
        }
        for case in _ordinary_yaml_cases()
    ]
