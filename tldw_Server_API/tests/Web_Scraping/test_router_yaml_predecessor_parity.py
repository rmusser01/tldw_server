import json
from copy import deepcopy
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as fixture_generator

ScraperRouter = import_module("tldw_Server_API.app.core.Web_Scraping.scraper_router").ScraperRouter

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase4"
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
    "url_patterns",
)
_PREDECESSOR_ERRORS = {
    ("handler-scalar", "direct", "AttributeError"),
    ("handler-scalar", "validated", "AttributeError"),
    ("handler-list", "direct", "AttributeError"),
    ("handler-list", "validated", "AttributeError"),
    ("handler-mapping", "direct", "AttributeError"),
    ("handler-mapping", "validated", "AttributeError"),
    ("extra_headers-null", "direct", "TypeError"),
    ("extra_headers-string", "direct", "ValueError"),
    ("extra_headers-scalar", "direct", "TypeError"),
    ("extra_headers-list", "direct", "ValueError"),
    ("cookies-null", "direct", "TypeError"),
    ("cookies-string", "direct", "ValueError"),
    ("cookies-scalar", "direct", "TypeError"),
    ("cookies-list", "direct", "ValueError"),
    ("proxies-null", "direct", "TypeError"),
    ("proxies-null", "validated", "TypeError"),
    ("proxies-string", "direct", "ValueError"),
    ("proxies-string", "validated", "ValueError"),
    ("proxies-scalar", "direct", "TypeError"),
    ("proxies-scalar", "validated", "TypeError"),
    ("proxies-list", "direct", "ValueError"),
    ("proxies-list", "validated", "ValueError"),
    ("url_patterns-mixed-types", "direct", "TypeError"),
    ("url_patterns-all-non-string", "direct", "TypeError"),
}
_APPROVED_CHANGE_4_URL_PATTERN_DIVERGENCES = {
    "url_patterns-mixed-types:direct",
    "url_patterns-all-non-string:direct",
    "url_patterns-all-non-string:validation",
}

pytestmark = pytest.mark.integration


def _load_router_cases() -> list[dict[str, Any]]:
    with fixture_generator.fixture_publication_reader(
        FIXTURE_ROOT,
        source_root=REPO_ROOT,
    ) as locked_root:
        manifest = json.loads((locked_root / "manifest.json").read_text(encoding="ascii"))
        payload = json.loads((locked_root / manifest["cases"]["router"]).read_text(encoding="ascii"))
    assert payload["category"] == "router"
    assert type(payload["cases"]) is list
    assert payload["cases"]
    return payload["cases"]


@pytest.fixture(scope="session")
def router_cases() -> list[dict[str, Any]]:
    return _load_router_cases()


def _capture_current(case: dict[str, Any], path: str) -> dict[str, Any]:
    rules = {"domains": {"example.com": deepcopy(case["rule"])}}
    try:
        if path == "validation":
            value = ScraperRouter.validate_rules(rules)
        elif path == "direct":
            value = asdict(ScraperRouter(rules).resolve(ROUTER_URL))
        else:
            cleaned = ScraperRouter.validate_rules(rules)
            value = asdict(ScraperRouter(cleaned).resolve(ROUTER_URL))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - assertion payload only
        return {"status": "error", "type": type(exc).__name__}
    return {"status": "ok", "value": value}


def _assert_approved_change_4_url_pattern_divergence(
    case: dict[str, Any],
    path: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    divergence = f"{case['name']}:{path}"
    assert divergence in _APPROVED_CHANGE_4_URL_PATTERN_DIVERGENCES
    assert actual != expected

    if path == "validation":
        assert expected == {
            "status": "ok",
            "value": {"domains": {"example.com": {"url_patterns": []}}},
        }
        assert actual == {"status": "ok", "value": {"domains": {}}}
        return

    assert expected == {"status": "error", "type": "TypeError"}
    assert actual == {
        "status": "ok",
        "value": asdict(ScraperRouter({}).resolve(ROUTER_URL)),
    }


@pytest.mark.parametrize("path", ["validation", "direct", "validated"])
def test_ordinary_yaml_results_match_predecessor_except_approved_change_4(
    path: str,
    router_cases: list[dict[str, Any]],
) -> None:
    differences = []
    approved_divergences: set[str] = set()
    for case in router_cases:
        expected = case["expected"][path]
        actual = _capture_current(case, path)
        divergence = f"{case['name']}:{path}"
        if divergence in _APPROVED_CHANGE_4_URL_PATTERN_DIVERGENCES:
            _assert_approved_change_4_url_pattern_divergence(
                case,
                path,
                actual,
                expected,
            )
            approved_divergences.add(divergence)
            continue
        if actual != expected:
            differences.append(
                {
                    "name": case["name"],
                    "expected": expected,
                    "actual": actual,
                }
            )

    assert differences == []
    assert approved_divergences == {
        divergence for divergence in _APPROVED_CHANGE_4_URL_PATTERN_DIVERGENCES if divergence.endswith(f":{path}")
    }


def test_router_fixture_has_strict_contract_and_complete_field_coverage(
    router_cases: list[dict[str, Any]],
) -> None:
    covered_fields: set[str] = set()
    predecessor_errors: list[tuple[str, str, str]] = []
    for case in router_cases:
        assert set(case) == {"expected", "name", "rule"}
        assert type(case["name"]) is str and case["name"]
        assert type(case["rule"]) is dict and case["rule"]
        assert set(case["expected"]) == {"validation", "direct", "validated"}
        for path, expected in case["expected"].items():
            assert expected["status"] in {"ok", "error"}
            assert set(expected) == ({"status", "value"} if expected["status"] == "ok" else {"status", "type"})
            if expected["status"] == "error":
                predecessor_errors.append((case["name"], path, expected["type"]))
        covered_fields.update(case["rule"])

    assert set(predecessor_errors) == _PREDECESSOR_ERRORS
    assert (
        set(_PLAN_FIELDS)
        - {
            "schema_rules",
            "llm_settings",
            "regex_settings",
            "cluster_settings",
        }
        <= covered_fields
    )
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
