import json
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
)
_PREDECESSOR_ERROR_CASE_PATHS = frozenset(
    {
        ("handler-scalar", "direct"),
        ("handler-scalar", "validated"),
        ("handler-list", "direct"),
        ("handler-list", "validated"),
        ("handler-mapping", "direct"),
        ("handler-mapping", "validated"),
        ("extra_headers-null", "direct"),
        ("extra_headers-string", "direct"),
        ("extra_headers-scalar", "direct"),
        ("extra_headers-list", "direct"),
        ("cookies-null", "direct"),
        ("cookies-string", "direct"),
        ("cookies-scalar", "direct"),
        ("cookies-list", "direct"),
        ("proxies-null", "direct"),
        ("proxies-null", "validated"),
        ("proxies-string", "direct"),
        ("proxies-string", "validated"),
        ("proxies-scalar", "direct"),
        ("proxies-scalar", "validated"),
        ("proxies-list", "direct"),
        ("proxies-list", "validated"),
    }
)
_EXPECTED_FAIL_SAFE_RESULT = {
    "status": "ok",
    "value": {
        "backend": "auto",
        "cluster_settings": None,
        "cookies": {},
        "domain": "example.com",
        "extra_headers": {},
        "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
        "impersonate": "chrome120",
        "llm_settings": None,
        "proxies": {},
        "regex_settings": None,
        "respect_robots": True,
        "schema_rules": None,
        "strategy_order": None,
        "ua_profile": "chrome_120_win",
        "url": "https://example.com/path",
    },
}


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


_ROUTER_CASES = _load_router_cases()


def _capture_current(case: dict[str, Any], path: str) -> dict[str, Any]:
    rules = {"domains": {"example.com": case["rule"]}}
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


@pytest.mark.parametrize("path", ["validation", "direct", "validated"])
def test_ordinary_yaml_results_match_checked_predecessor_fixture(path: str) -> None:
    differences = []
    for case in _ROUTER_CASES:
        expected = case["expected"][path]
        if expected["status"] == "error":
            continue
        actual = _capture_current(case, path)
        if actual != expected:
            differences.append(
                {
                    "name": case["name"],
                    "expected": expected,
                    "actual": actual,
                }
            )

    assert differences == []


def test_current_router_remains_fail_safe_where_checked_predecessor_raised() -> None:
    predecessor_errors = [
        (case, path)
        for case in _ROUTER_CASES
        for path in ("validation", "direct", "validated")
        if case["expected"][path]["status"] == "error"
    ]
    actual_pairs = frozenset((case["name"], path) for case, path in predecessor_errors)

    assert len(predecessor_errors) == 22
    assert len(actual_pairs) == 22
    assert actual_pairs == _PREDECESSOR_ERROR_CASE_PATHS, {
        "missing": sorted(_PREDECESSOR_ERROR_CASE_PATHS - actual_pairs),
        "unexpected": sorted(actual_pairs - _PREDECESSOR_ERROR_CASE_PATHS),
    }

    mismatches = [
        (case["name"], path)
        for case, path in predecessor_errors
        if _capture_current(case, path) != _EXPECTED_FAIL_SAFE_RESULT
    ]
    assert mismatches == []


def test_predecessor_error_contract_rejects_arbitrary_ok_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        globals(),
        "_capture_current",
        lambda _case, _path: {"status": "ok", "value": {"tampered": True}},
    )

    with pytest.raises(AssertionError):
        test_current_router_remains_fail_safe_where_checked_predecessor_raised()


def test_router_fixture_has_strict_contract_and_complete_field_coverage() -> None:
    covered_fields: set[str] = set()
    for case in _ROUTER_CASES:
        assert set(case) == {"expected", "name", "rule"}
        assert type(case["name"]) is str and case["name"]
        assert type(case["rule"]) is dict and case["rule"]
        assert set(case["expected"]) == {"validation", "direct", "validated"}
        for expected in case["expected"].values():
            assert expected["status"] in {"ok", "error"}
            assert set(expected) == ({"status", "value"} if expected["status"] == "ok" else {"status", "type"})
        covered_fields.update(case["rule"])

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
