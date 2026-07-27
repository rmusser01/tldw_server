from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from lxml import html

from tldw_Server_API.app.core.Watchlists import fetchers
from tldw_Server_API.app.core.Web_Scraping import selectors
from tldw_Server_API.app.core.Web_Scraping.selectors import caches, engine, schema
from tldw_Server_API.tests.Web_Scraping.phase4_fixture_contracts import (
    assert_predecessor_behavior,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase4" / "selectors.json"
SELECTOR_ENV = {
    "WATCHLIST_SELECTOR_MAX_EXPR_LEN": "512",
    "WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS": "12",
    "WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS": "8",
    "WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES": "10",
}


def _fixture_cases() -> list[dict[str, Any]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["category"] == "selectors"
    return payload["cases"]


@pytest.fixture(autouse=True)
def _reset_selector_state(monkeypatch: pytest.MonkeyPatch):
    for name, value in SELECTOR_ENV.items():
        monkeypatch.setenv(name, value)
    engine.reload_selector_guardrails_from_env()
    selectors.clear_selector_caches()
    yield
    selectors.clear_selector_caches()
    engine.reload_selector_guardrails_from_env()


@pytest.mark.parametrize("case", _fixture_cases(), ids=lambda case: case["name"])
def test_selector_fixture_matches_predecessor_contract(case: dict[str, Any]) -> None:
    if case["operation"] == "validate":
        result = selectors.validate_selector_rules(
            case["rules"],
            html_text=case.get("html"),
            include_counts=case.get("include_counts", False),
        )
    else:
        assert case["operation"] == "extract_schema_fields"
        result = selectors.extract_schema_fields(
            case["html"],
            case["base_url"],
            case["rules"],
        )

    actual = {
        "cache_stats": selectors.get_selector_cache_stats(),
        "result": result,
    }
    assert_predecessor_behavior(
        actual,
        case["expected"],
        behavior_change=case.get("behavior_change"),
        difference_contract=case.get("difference_contract"),
    )


def test_selector_facade_and_watchlists_exports_are_canonical() -> None:
    assert selectors.__all__ == [
        "clear_selector_caches",
        "extract_schema_fields",
        "get_selector_cache_stats",
        "validate_selector_rules",
    ]
    assert fetchers.validate_selector_rules is selectors.validate_selector_rules
    assert fetchers.extract_schema_fields is selectors.extract_schema_fields
    assert fetchers.get_selector_cache_stats is selectors.get_selector_cache_stats
    assert fetchers.clear_selector_caches is selectors.clear_selector_caches
    assert fetchers.reload_selector_guardrails_from_env is engine.reload_selector_guardrails_from_env


def test_validation_sanitizes_xpath_and_css_parser_errors() -> None:
    report = selectors.validate_selector_rules(
        {
            "content_xpath": "//private-token-article[",
            "content_selector": "css:private-token-div[",
        }
    )

    assert [entry["error"] for entry in report["errors"]] == [
        "selector_invalid",
        "selector_invalid",
    ]
    assert "Invalid expression" not in repr(report)
    assert "Expected selector" not in repr(report)


def test_validation_sanitizes_xpath_evaluation_errors() -> None:
    report = selectors.validate_selector_rules(
        {"content_xpath": "//article[phase4_unregistered_function()]"},
        html_text="<html><body><article>Body</article></body></html>",
    )

    assert report["errors"] == [
        {
            "key": "content_xpath",
            "selector": "//article[phase4_unregistered_function()]",
            "error": "selector_invalid",
        }
    ]
    assert "Unregistered function" not in repr(report)


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        (
            {"fields": [{"name": "title", "selector": "//h1"}]},
            {
                "url": "https://example.com/post",
                "extraction_successful": True,
                "schema_fields": {"title": "Example Title"},
                "title": "Example Title",
            },
        ),
        (
            {"baseFields": [{"name": "title", "selector": "//h1"}]},
            {
                "url": "https://example.com/post",
                "extraction_successful": True,
                "schema_fields": {"title": "Example Title"},
                "title": "Example Title",
            },
        ),
        (
            {"baseFields": {"title": {"selector": "//h1"}}},
            {
                "url": "https://example.com/post",
                "extraction_successful": True,
                "schema_fields": {"title": "Example Title"},
                "title": "Example Title",
            },
        ),
        (
            {"fields": {"title": {"selector": "//h1"}}},
            {
                "url": "https://example.com/post",
                "extraction_successful": False,
            },
        ),
        (
            {"baseSelector": "//article"},
            {
                "url": "https://example.com/post",
                "extraction_successful": False,
            },
        ),
    ],
    ids=[
        "fields-list-is-dsl",
        "base-fields-list-is-dsl",
        "base-fields-mapping-is-dsl",
        "fields-mapping-is-legacy",
        "base-selector-alone-is-legacy",
    ],
)
def test_schema_dsl_detection_preserves_predecessor_result_shapes(
    rules: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    result = selectors.extract_schema_fields(
        "<html><body><article><h1>Example Title</h1></article></body></html>",
        "https://example.com/post",
        rules,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("", False),
        ("css-abc", False),
        ("css-abcd", True),
        ("sc-abcde", False),
        ("jsx-abcd", False),
        ("CSS-abcd", False),
        ("Css-abcd", False),
        ("stableClass12", True),
    ],
)
def test_fragile_class_name_preserves_predecessor_boundaries(
    class_name: str,
    expected: bool,
) -> None:
    assert schema._is_fragile_class_name(class_name) is expected


@pytest.mark.parametrize(
    ("class_name", "expect_warning"),
    [
        ("css-abcd", True),
        ("sc-abcde", False),
        ("jsx-abcd", False),
        ("CSS-abcd", False),
        ("Css-abcd", False),
    ],
)
def test_fragile_class_warnings_preserve_prefix_and_case_parity(
    class_name: str,
    expect_warning: bool,
) -> None:
    report = selectors.validate_selector_rules(
        {"title_selector": f"css:.{class_name}"},
        html_text=f'<html><body><div class="{class_name}">Title</div></body></html>',
    )

    fragile_warnings = [warning for warning in report["warnings"] if warning["warning"] == "fragile_selector"]
    assert bool(fragile_warnings) is expect_warning


def test_endpoint_uses_canonical_validation_and_preserves_diagnostics() -> None:
    from tldw_Server_API.app.api.v1.endpoints import watchlists as endpoint

    assert endpoint.validate_selector_rules is selectors.validate_selector_rules

    diagnostics = endpoint._build_source_preview_diagnostics(
        fetch_mode="scrape_rules",
        scrape_rules={"content_xpath": "//article["},
    )

    assert diagnostics.selector_errors
    assert "selector_invalid" in diagnostics.selector_errors[0]
    assert "Invalid expression" not in diagnostics.selector_errors[0]


def test_regex_fields_preserve_groups_and_regex_replace_is_global() -> None:
    result = selectors.extract_schema_fields(
        "<article><h1>banana</h1><span>Order #12345 confirmed</span></article>",
        "https://example.com/post",
        {
            "name": "regex-semantics",
            "baseSelector": "//article",
            "baseFields": [
                {"name": "title", "selector": ".//h1"},
                {
                    "name": "order",
                    "type": "regex",
                    "selector": ".//span",
                    "pattern": r"Order #(?P<number>\d+)",
                    "group": "number",
                },
            ],
            "fields": [
                {
                    "name": "slug",
                    "type": "computed",
                    "from": "title",
                    "transforms": [{"name": "regex_replace", "pattern": "a", "repl": "x"}],
                }
            ],
        },
    )

    assert result["schema_fields"]["order"] == "12345"
    assert result["schema_fields"]["slug"] == "bxnxnx"


def test_invalid_and_oversized_regexes_preserve_existing_fallbacks() -> None:
    result = selectors.extract_schema_fields(
        "<article><h1>Original</h1><span>Value</span></article>",
        "https://example.com/post",
        {
            "name": "regex-fallbacks",
            "baseSelector": "//article",
            "baseFields": [{"name": "title", "selector": ".//h1"}],
            "fields": [
                {
                    "name": "invalid_field",
                    "type": "regex",
                    "selector": ".//span",
                    "pattern": "[",
                },
                {
                    "name": "oversized_field",
                    "type": "regex",
                    "selector": ".//span",
                    "pattern": "a" * 4_097,
                },
                {
                    "name": "invalid_transform",
                    "type": "computed",
                    "from": "title",
                    "transforms": [{"name": "regex_replace", "pattern": "[", "repl": "x"}],
                },
                {
                    "name": "oversized_transform",
                    "type": "computed",
                    "from": "title",
                    "transforms": [
                        {
                            "name": "regex_replace",
                            "pattern": "a" * 4_097,
                            "repl": "x",
                        }
                    ],
                },
            ],
        },
    )

    fields = result["schema_fields"]
    assert "invalid_field" not in fields
    assert "oversized_field" not in fields
    assert fields["invalid_transform"] == "Original"
    assert fields["oversized_transform"] == "Original"


def test_regex_paths_do_not_disclose_patterns_or_engine_errors(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_pattern = "private-regex-token["
    caplog.set_level(logging.DEBUG)

    result = selectors.extract_schema_fields(
        "<article><h1>Original</h1></article>",
        "https://example.com/post",
        {
            "name": "regex-logging",
            "baseSelector": "//article",
            "fields": [
                {
                    "name": "value",
                    "type": "regex",
                    "selector": ".//h1",
                    "pattern": secret_pattern,
                }
            ],
        },
    )
    captured = capsys.readouterr()
    disclosed = " ".join([caplog.text, captured.out, captured.err])

    assert result["schema_fields"] == {}
    assert secret_pattern not in disclosed
    assert "unterminated character set" not in disclosed


def test_selector_caches_are_bounded_lru_caches() -> None:
    document = html.fromstring("<html><body><p>value</p></body></html>")

    for index in range(caches._SELECTOR_CACHE_MAX):
        engine._select_nodes(document, f"//p[{index + 1}]")
    first = "//p[1]"
    second = "//p[2]"
    engine._select_nodes(document, first)
    engine._select_nodes(document, f"//p[{caches._SELECTOR_CACHE_MAX + 1}]")

    assert len(caches._XPATH_SELECTOR_CACHE) == caches._SELECTOR_CACHE_MAX
    assert first in caches._XPATH_SELECTOR_CACHE
    assert second not in caches._XPATH_SELECTOR_CACHE

    for index in range(caches._SELECTOR_CACHE_MAX + 1):
        caches._put_css_selector(f"p[data-cache-key='{index}']", object())

    stats = selectors.get_selector_cache_stats()
    assert stats == {
        "selector_xpath_cache_size": caches._SELECTOR_CACHE_MAX,
        "selector_css_cache_size": caches._SELECTOR_CACHE_MAX,
    }


def test_selector_cache_operations_are_thread_safe() -> None:
    def _exercise(index: int) -> None:
        document = html.fromstring("<html><body><p>value</p></body></html>")
        engine._select_nodes(document, f"//p[{index % 700 + 1}]")
        caches._put_css_selector(f"p[data-thread-key='{index % 700}']", object())
        if index % 31 == 0:
            selectors.clear_selector_caches()
        selectors.get_selector_cache_stats()

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_exercise, range(1_400)))

    stats = selectors.get_selector_cache_stats()
    assert stats["selector_xpath_cache_size"] <= caches._SELECTOR_CACHE_MAX
    assert stats["selector_css_cache_size"] <= caches._SELECTOR_CACHE_MAX
