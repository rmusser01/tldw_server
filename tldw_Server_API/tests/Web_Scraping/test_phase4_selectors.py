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


def test_generic_css_validation_and_extraction_share_the_compiler() -> None:
    rules = {"fields": [{"name": "title", "selector": "css:.headline"}]}
    html_text = '<article><h1 class="headline">Shared compiler</h1></article>'

    report = selectors.validate_selector_rules(rules, html_text=html_text, include_counts=True)
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
    )

    assert report == {
        "errors": [],
        "warnings": [],
        "selector_counts": {"fields.title": 1},
    }
    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": True,
        "schema_fields": {"title": "Shared compiler"},
        "title": "Shared compiler",
    }
    assert selectors.get_selector_cache_stats()["selector_css_cache_size"] == 1


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("css:table caption", "Quarterly table"),
        ("css:table thead tr td:nth-child(2)", "Header data"),
        ("css:table tbody tr:nth-child(1) td:nth-child(2)", "Row 1 data"),
        ("css:table tbody tr:nth-child(2) td:nth-child(1)", "Row 2 first"),
        ("css:table tbody tr td", "Row 1 data Row 2 first Row 2 second"),
        ("css:div.mixed span:nth-child(2)", "Second element"),
        ("css:table tbody tr:nth-child(3)", None),
    ],
)
def test_css_nth_child_counts_all_element_siblings_and_section_boundaries(
    selector: str,
    expected: str | None,
) -> None:
    html_text = """
    <article>
      <table>
        <caption>Quarterly table</caption>
        <thead><tr><th>Header label</th><td>Header data</td></tr></thead>
        <tbody>
          <tr><th>Row 1 label</th><td>Row 1 data</td></tr>
          <tr><td>Row 2 first</td><td>Row 2 second</td></tr>
        </tbody>
      </table>
      <div class="mixed"><em>First element</em><span>Second element</span></div>
    </article>
    """
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        {"fields": [{"name": "value", "selector": selector}]},
    )

    assert result["schema_fields"].get("value") == expected


def _computed_rules(template: str) -> dict[str, Any]:
    return {
        "fields": [
            {"name": "title", "selector": "//h1"},
            {"name": "computed", "type": "computed", "template": template},
        ]
    }


@pytest.mark.parametrize(
    "template",
    [
        "{title",
        "{title:5000000}",
        "{title.__class__}",
        "{title[0]}",
        "{title!r}",
    ],
)
def test_invalid_computed_templates_validate_and_extract_fail_soft(template: str) -> None:
    rules = _computed_rules(template)

    report = selectors.validate_selector_rules(rules)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
    )

    assert [entry["error"] for entry in report["errors"]] == ["selector_invalid"]
    assert result["schema_fields"] == {"title": "Headline"}


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("{{literal}} {title}", "{literal} Headline"),
        ("before {missing} after", "before  after"),
        ("Title: {title}", "Title: Headline"),
    ],
)
def test_safe_computed_templates_preserve_normal_rendering(
    template: str,
    expected: str,
) -> None:
    rules = _computed_rules(template)

    report = selectors.validate_selector_rules(rules)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
    )

    assert report["errors"] == []
    assert result["schema_fields"]["computed"] == expected


def test_oversized_computed_template_has_a_stable_complexity_error() -> None:
    code = "selector_too_complex:template_length>4096"
    rules = _computed_rules("x" * 4_097)

    report = selectors.validate_selector_rules(rules)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }


def test_oversized_computed_render_has_a_stable_complexity_error() -> None:
    code = "selector_too_complex:rendered_output>1000000"
    title = "x" * 500_001

    result = selectors.extract_schema_fields(
        f"<article><h1>{title}</h1></article>",
        "https://example.com/post",
        _computed_rules("{title}{title}"),
    )

    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }


def _nested_rules(depth: int) -> dict[str, Any]:
    field: dict[str, Any] = {"name": "leaf", "selector": "//h1"}
    for index in range(depth - 1):
        field = {
            "name": f"level_{index}",
            "type": "nested",
            "fields": [field],
        }
    return {"fields": [field]}


@pytest.mark.parametrize(
    ("rules", "limits", "code"),
    [
        (
            _nested_rules(33),
            None,
            "selector_too_complex:schema_depth>32",
        ),
        (
            {"fields": [{"name": f"field_{index}", "value": "x"} for index in range(257)]},
            None,
            "selector_too_complex:schema_fields>256",
        ),
        (
            {"fields": [{"name": f"field_{index}", "selector": "//h1"} for index in range(3)]},
            {"max_selector_evaluations": 2},
            "selector_too_complex:selector_evaluations>2",
        ),
    ],
    ids=["depth", "fields", "selector-evaluations"],
)
def test_schema_structure_and_evaluation_budgets_are_shared(
    rules: dict[str, Any],
    limits: Any,
    code: str,
) -> None:
    kwargs = {"_limits": schema._SchemaLimits(**limits)} if limits is not None else {}

    report = selectors.validate_selector_rules(
        rules,
        html_text="<article><h1>Headline</h1></article>",
        **kwargs,
    )
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
        **kwargs,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }


def test_five_hundred_nested_fields_fail_before_python_recursion() -> None:
    code = "selector_too_complex:schema_depth>32"
    rules = _nested_rules(500)

    report = selectors.validate_selector_rules(rules)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code


@pytest.mark.parametrize(
    ("rules", "html_text", "max_evaluations"),
    [
        (
            {
                "fields": [
                    {
                        "name": "nested",
                        "type": "nested",
                        "selector": "//section",
                        "fields": [{"name": "value", "selector": ".//span"}],
                    }
                ]
            },
            "<section><span>one</span></section>",
            1,
        ),
        (
            {
                "fields": [
                    {
                        "name": "nested_list",
                        "type": "nested_list",
                        "selector": "//section",
                        "fields": [{"name": "value", "selector": ".//span"}],
                    }
                ]
            },
            "<main><section><span>one</span></section><section><span>two</span></section></main>",
            2,
        ),
        (
            {
                "fields": [
                    {
                        "name": "items",
                        "type": "list",
                        "selector": "//ul",
                        "itemSelector": ".//li",
                    }
                ]
            },
            "<ul><li>one</li><li>two</li></ul>",
            1,
        ),
    ],
    ids=["nested", "nested-list", "list-item-selector"],
)
def test_nested_paths_count_every_selector_evaluation(
    rules: dict[str, Any],
    html_text: str,
    max_evaluations: int,
) -> None:
    limits = schema._SchemaLimits(max_selector_evaluations=max_evaluations)
    code = f"selector_too_complex:selector_evaluations>{max_evaluations}"

    report = selectors.validate_selector_rules(
        rules,
        html_text=html_text,
        _limits=limits,
    )
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code


def test_aggregate_match_budget_is_shared_by_validation_and_extraction() -> None:
    limits = schema._SchemaLimits(max_aggregate_matches=3)
    code = "selector_too_complex:selector_matches>3"
    rules = {"fields": [{"name": "items", "type": "list", "selector": "//li"}]}
    html_text = "<ul><li>1</li><li>2</li><li>3</li><li>4</li></ul>"

    report = selectors.validate_selector_rules(rules, html_text=html_text, _limits=limits)
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code


def test_aggregate_retained_output_budget_fails_without_partial_results() -> None:
    limits = schema._SchemaLimits(max_retained_output_chars=5)
    code = "selector_too_complex:retained_output_chars>5"

    result = selectors.extract_schema_fields(
        "<article><h1>123456</h1></article>",
        "https://example.com/post",
        {"fields": [{"name": "title", "selector": "//h1"}]},
        _limits=limits,
    )

    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }


@pytest.mark.parametrize(
    ("rules", "html_text", "extractor_name", "max_output_chars", "expected_calls"),
    [
        (
            {"fields": [{"name": "items", "type": "list", "selector": "//li"}]},
            "<ul>" + "".join("<li>x</li>" for _ in range(10)) + "</ul>",
            "_extract_text_from_node",
            1,
            2,
        ),
        (
            {
                "fields": [
                    {
                        "name": "items",
                        "type": "nested_list",
                        "selector": "//section",
                        "fields": [{"name": "value", "selector": ".//span"}],
                    }
                ]
            },
            "<main>" + "".join("<section><span>x</span></section>" for _ in range(10)) + "</main>",
            "_extract_text_from_node",
            1,
            2,
        ),
        (
            {"content_xpath": "//p"},
            "<article>" + "".join("<p>x</p>" for _ in range(10)) + "</article>",
            "coerce_value",
            1,
            2,
        ),
        (
            {
                "fields": [
                    {
                        "name": "items",
                        "type": "list",
                        "selector": "//li",
                        "itemType": "html",
                    }
                ]
            },
            "<ul>" + "".join("<li>x</li>" for _ in range(10)) + "</ul>",
            "_extract_html_from_node",
            10,
            2,
        ),
    ],
    ids=["list", "nested-list", "legacy-join", "html-serialization"],
)
def test_output_budget_stops_collection_materialization_incrementally(
    monkeypatch: pytest.MonkeyPatch,
    rules: dict[str, Any],
    html_text: str,
    extractor_name: str,
    max_output_chars: int,
    expected_calls: int,
) -> None:
    owner = engine if extractor_name == "coerce_value" else schema
    original = getattr(owner, extractor_name)
    calls = 0

    def counting_extract(node: Any) -> str | None:
        nonlocal calls
        calls += 1
        return original(node)

    monkeypatch.setattr(owner, extractor_name, counting_extract)
    if extractor_name == "coerce_value":
        monkeypatch.setattr(schema, extractor_name, counting_extract)
    code = f"selector_too_complex:retained_output_chars>{max_output_chars}"

    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=max_output_chars),
    )

    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }
    assert calls == expected_calls


@pytest.mark.parametrize(
    ("html_text", "replacement"),
    [
        ("<article><h1>xx</h1></article>", ""),
        ("<article><h1>x</h1></article>", "xx"),
    ],
    ids=["raw-input", "transform-result"],
)
def test_transform_materialization_uses_rendered_output_limit(
    html_text: str,
    replacement: str,
) -> None:
    code = "selector_too_complex:rendered_output>1"
    rules = {
        "fields": [
            {
                "name": "value",
                "selector": "//h1",
                "transforms": [{"name": "regex_replace", "pattern": "x", "repl": replacement}],
            }
        ]
    }

    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(
            max_retained_output_chars=1,
            max_rendered_output_chars=1,
        ),
    )

    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }


def test_transform_reduction_succeeds_under_final_retained_limit() -> None:
    rules = {
        "fields": [
            {
                "name": "value",
                "selector": "//h1",
                "transforms": [{"name": "regex_replace", "pattern": "x", "repl": ""}],
            }
        ]
    }

    result = selectors.extract_schema_fields(
        "<article><h1>xx</h1></article>",
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(
            max_retained_output_chars=1,
            max_rendered_output_chars=2,
        ),
    )

    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "schema_fields": {"value": ""},
    }


@pytest.mark.parametrize(
    ("selector", "html_text"),
    [
        ("//li", "<ul><li>aa</li><li>bb</li></ul>"),
        ("css:table td", "<table><tbody><tr><td>aa</td><td>bb</td></tr></tbody></table>"),
    ],
    ids=["list", "table"],
)
def test_compatibility_projection_is_charged_at_exact_output_boundary(
    selector: str,
    html_text: str,
) -> None:
    rules = {"fields": [{"name": "content", "type": "list", "selector": selector}]}

    exact = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=9),
    )
    one_over = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=8),
    )

    assert exact == {
        "url": "https://example.com/post",
        "extraction_successful": True,
        "schema_fields": {"content": ["aa", "bb"]},
        "content": "aa\nbb",
    }
    assert one_over == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": "selector_too_complex:retained_output_chars>8",
    }


def test_table_shaped_compatibility_projection_is_charged_separately() -> None:
    rules = {
        "fields": [
            {
                "name": "content",
                "type": "nested_list",
                "selector": "//tr",
                "fields": [{"name": "cell", "selector": ".//td"}],
            }
        ]
    }
    html_text = "<table><tr><td>aa</td></tr><tr><td>bb</td></tr></table>"

    exact = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=8),
    )
    one_over = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=7),
    )

    table = [{"cell": "aa"}, {"cell": "bb"}]
    assert exact == {
        "url": "https://example.com/post",
        "extraction_successful": True,
        "schema_fields": {"content": table},
        "content": table,
    }
    assert one_over == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": "selector_too_complex:retained_output_chars>7",
    }


@pytest.mark.parametrize("case", ["exact", "one-over"])
def test_default_projection_budget_without_large_dom(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    values = ["a" * 166_666, "b" * 166_666, "c" * 166_667] if case == "exact" else ["a" * 250_000, "b" * 250_000]
    value_iterator = iter(values)
    monkeypatch.setattr(schema, "_extract_text_from_node", lambda _node: next(value_iterator))
    html_text = "<ul>" + "".join("<li>x</li>" for _ in values) + "</ul>"

    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        {"fields": [{"name": "content", "type": "list", "selector": "//li"}]},
    )

    if case == "exact":
        assert result["extraction_successful"] is True
        assert result["schema_fields"]["content"] == values
        assert len(result["content"]) == 500_001
    else:
        assert result == {
            "url": "https://example.com/post",
            "extraction_successful": False,
            "error": "selector_too_complex:retained_output_chars>1000000",
        }


def test_irrelevant_item_selector_is_compile_only_at_evaluation_boundary() -> None:
    rules = {
        "fields": [
            {
                "name": "title",
                "selector": "//h1",
                "itemSelector": "//span[",
            }
        ]
    }
    limits = schema._SchemaLimits(max_selector_evaluations=1)

    report = selectors.validate_selector_rules(
        rules,
        html_text="<article><h1>Headline</h1></article>",
        include_counts=True,
        _limits=limits,
    )
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert report == {
        "errors": [
            {
                "key": "fields.title.item_selector",
                "selector": "//span[",
                "error": "selector_invalid",
            }
        ],
        "warnings": [],
        "selector_counts": {"fields.title": 1},
    }
    assert result["schema_fields"] == {"title": "Headline"}


@pytest.mark.parametrize(
    ("selectors_in_order", "max_evaluations", "expected_title", "expected_error"),
    [
        (["//h1", "//h2"], 1, "First", None),
        (["//missing", "//h2"], 2, "Second", None),
        (
            ["//missing", "//h2"],
            1,
            None,
            "selector_too_complex:selector_evaluations>1",
        ),
    ],
    ids=["first-nonempty", "fallback-exact-limit", "fallback-one-over"],
)
def test_fallback_validation_matches_extraction_evaluation_path(
    selectors_in_order: list[str],
    max_evaluations: int,
    expected_title: str | None,
    expected_error: str | None,
) -> None:
    rules = {"title_xpath": selectors_in_order}
    html_text = "<article><h1>First</h1><h2>Second</h2></article>"
    limits = schema._SchemaLimits(max_selector_evaluations=max_evaluations)

    report = selectors.validate_selector_rules(
        rules,
        html_text=html_text,
        _limits=limits,
    )
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    if expected_error:
        assert [entry["error"] for entry in report["errors"]] == [expected_error]
        assert result["error"] == expected_error
    else:
        assert report["errors"] == []
        assert result["title"] == expected_title


def test_schema_limit_defaults_are_explicit_and_conservative() -> None:
    assert (
        schema._SchemaLimits(
            max_depth=32,
            max_total_fields=256,
            max_selector_evaluations=512,
            max_aggregate_matches=10_000,
            max_retained_output_chars=1_000_000,
            max_template_length=4_096,
            max_rendered_output_chars=1_000_000,
        )
        == schema._DEFAULT_SCHEMA_LIMITS
    )


def test_validation_and_extraction_share_first_output_failure() -> None:
    rules = {
        "fields": [
            {"name": "first", "selector": "//h1"},
            {"name": "second", "selector": "//h2"},
        ]
    }
    html_text = "<article><h1>xx</h1><h2>y</h2></article>"
    limits = schema._SchemaLimits(
        max_selector_evaluations=1,
        max_retained_output_chars=1,
    )
    code = "selector_too_complex:retained_output_chars>1"

    report = selectors.validate_selector_rules(rules, html_text=html_text, _limits=limits)
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code


def test_validation_and_extraction_share_success_path() -> None:
    rules = {
        "fields": [
            {"name": "first", "selector": "//h1"},
            {"name": "second", "selector": "//h2"},
        ]
    }
    html_text = "<article><h1>xx</h1><h2>y</h2></article>"
    limits = schema._SchemaLimits(
        max_selector_evaluations=2,
        max_retained_output_chars=3,
    )

    report = selectors.validate_selector_rules(rules, html_text=html_text, _limits=limits)
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert report["errors"] == []
    assert result["schema_fields"] == {"first": "xx", "second": "y"}


@pytest.mark.parametrize(
    "rules",
    [
        {
            "fields": [
                {"name": "value", "selector": "//i"},
                {"name": "value", "selector": "//b"},
            ]
        },
        {
            "baseFields": [{"name": "value", "selector": "//i"}],
            "fields": [{"name": "value", "selector": "//b"}],
        },
        {
            "fields": [
                {"name": "value", "selector": "//i"},
                {"name": "value", "type": "computed", "value": "bb"},
            ]
        },
    ],
    ids=["ordinary", "base-then-field", "ordinary-then-computed"],
)
def test_replaced_output_slots_release_prior_reservations(rules: dict[str, Any]) -> None:
    result = selectors.extract_schema_fields(
        "<article><i>aa</i><b>bb</b></article>",
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=2),
    )

    assert result["schema_fields"] == {"value": "bb"}


def test_failed_output_slot_replacement_is_atomic() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=2))
    slot = ("schema_fields", "value")
    budget.retain_output("aa", slot=slot)

    with pytest.raises(schema._SchemaBudgetExceeded):
        budget.retain_output("bbb", slot=slot)

    assert budget.retained_output_chars == 2
    assert budget.output_slots == {slot: 2}


def test_nested_list_output_slots_include_item_indices_without_container_charge() -> None:
    rules = {
        "fields": [
            {
                "name": "items",
                "type": "nested_list",
                "selector": "//section",
                "fields": [
                    {"name": "value", "selector": ".//i"},
                    {"name": "value", "selector": ".//b"},
                ],
            }
        ]
    }
    html_text = "<main><section><i>aa</i><b>bb</b></section>" "<section><i>aa</i><b>bb</b></section></main>"

    exact = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=4),
    )
    one_over = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=3),
    )

    assert exact["schema_fields"] == {"items": [{"value": "bb"}, {"value": "bb"}]}
    assert one_over["error"] == "selector_too_complex:retained_output_chars>3"


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("//a", ["A", "B"]),
        ("//a/@href", ["a", "b"]),
        ("//a[position() = last()]", ["C"]),
        ("count(//a)", [3.0]),
        ("string(//a[1])", ["A"]),
    ],
    ids=["nodes", "attributes", "position-last", "number-scalar", "string-scalar"],
)
def test_bounded_xpath_preserves_node_set_attribute_and_scalar_behavior(
    selector: str,
    expected: list[Any],
) -> None:
    document = html.fromstring('<main><a href="a">A</a><a href="b">B</a><a href="c">C</a></main>')

    matches, failed = engine._select_nodes_with_status(
        document,
        selector,
        max_results=2,
    )

    actual = [item.text_content() if isinstance(item, html.HtmlElement) else item for item in matches]
    assert failed is False
    assert actual == expected


def test_schema_requests_only_remaining_match_capacity_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = engine._select_nodes_with_status
    observed_caps: list[int | None] = []

    def observe_selection(*args: Any, max_results: int | None = None, **kwargs: Any):
        observed_caps.append(max_results)
        return original(*args, max_results=max_results, **kwargs)

    monkeypatch.setattr(engine, "_select_nodes_with_status", observe_selection)
    monkeypatch.setattr(schema, "_select_nodes_with_status", observe_selection)

    result = selectors.extract_schema_fields(
        "<ul><li>a</li><li>b</li><li>c</li></ul>",
        "https://example.com/post",
        {"fields": [{"name": "items", "type": "list", "selector": "//li"}]},
        _limits=schema._SchemaLimits(max_aggregate_matches=1),
    )

    assert result["error"] == "selector_too_complex:selector_matches>1"
    assert observed_caps == [2]


def test_large_element_text_does_not_call_unbounded_text_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = schema.coerce_value

    def reject_element_coercion(value: Any) -> str | None:
        if isinstance(value, html.HtmlElement):
            raise AssertionError("bounded element text must use itertext")
        return original(value)

    monkeypatch.setattr(schema, "coerce_value", reject_element_coercion)
    result = selectors.extract_schema_fields(
        f"<article><h1>{'x' * 200_000}</h1></article>",
        "https://example.com/post",
        {"fields": [{"name": "value", "selector": "//h1"}]},
        _limits=schema._SchemaLimits(max_retained_output_chars=10),
    )

    assert result["error"] == "selector_too_complex:retained_output_chars>10"


def test_bounded_element_text_preserves_exact_predecessor_value() -> None:
    expected = "Lead inner tail"
    result = selectors.extract_schema_fields(
        "<article><h1>  Lead <span>inner</span> tail  </h1></article>",
        "https://example.com/post",
        {"fields": [{"name": "value", "selector": "//h1"}]},
        _limits=schema._SchemaLimits(max_retained_output_chars=len(expected)),
    )

    assert result["schema_fields"] == {"value": expected}


def test_large_html_does_not_call_unbounded_tostring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        schema.html,
        "tostring",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bounded HTML must use a counting writer")),
    )
    result = selectors.extract_schema_fields(
        f"<article><section>{'x' * 200_000}</section></article>",
        "https://example.com/post",
        {"fields": [{"name": "value", "type": "html", "selector": "//section"}]},
        _limits=schema._SchemaLimits(max_retained_output_chars=10),
    )

    assert result["error"] == "selector_too_complex:retained_output_chars>10"


def test_bounded_html_preserves_exact_predecessor_value() -> None:
    expected = "<section><p>Body &amp; more</p><br></section>"
    result = selectors.extract_schema_fields(
        "<article><section><p>Body &amp; more</p><br></section></article>",
        "https://example.com/post",
        {"fields": [{"name": "value", "type": "html", "selector": "//section"}]},
        _limits=schema._SchemaLimits(max_retained_output_chars=len(expected)),
    )

    assert result["schema_fields"] == {"value": expected}


def test_regex_transform_receives_active_rendered_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = schema.sub_untrusted
    observed_caps: list[int | None] = []

    def observe_sub(pattern: str, repl: str, value: str, **kwargs: Any):
        observed_caps.append(kwargs.get("max_output_chars"))
        kwargs.pop("max_output_chars", None)
        return original(pattern, repl, value, **kwargs)

    monkeypatch.setattr(schema, "sub_untrusted", observe_sub)
    result = selectors.extract_schema_fields(
        "<article><h1>x</h1></article>",
        "https://example.com/post",
        {
            "fields": [
                {
                    "name": "value",
                    "selector": "//h1",
                    "transforms": [{"name": "regex_replace", "pattern": "x", "repl": "xx"}],
                }
            ]
        },
        _limits=schema._SchemaLimits(max_rendered_output_chars=1),
    )

    assert result["error"] == "selector_too_complex:rendered_output>1"
    assert observed_caps == [1]


def test_prepend_and_append_transforms_preflight_known_expansion() -> None:
    rules = {
        "fields": [
            {
                "name": "value",
                "selector": "//h1",
                "transforms": [
                    {"name": "prepend", "value": "a"},
                    {"name": "append", "value": "b"},
                ],
            }
        ]
    }

    exact = selectors.extract_schema_fields(
        "<article><h1>x</h1></article>",
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_rendered_output_chars=3),
    )
    one_over = selectors.extract_schema_fields(
        "<article><h1>x</h1></article>",
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_rendered_output_chars=2),
    )

    assert exact["schema_fields"] == {"value": "axb"}
    assert one_over["error"] == "selector_too_complex:rendered_output>2"


def test_compile_only_selector_path_does_not_mutate_runtime_caches() -> None:
    engine.compile_selector("//h1", cache=False)
    engine.compile_selector("css:h1", cache=False)

    assert selectors.get_selector_cache_stats() == {
        "selector_xpath_cache_size": 0,
        "selector_css_cache_size": 0,
    }

    engine.compile_selector("//h1", cache=True)
    engine.compile_selector("css:h1", cache=True)

    assert selectors.get_selector_cache_stats() == {
        "selector_xpath_cache_size": 1,
        "selector_css_cache_size": 1,
    }


def _alternate_rules(depth: int) -> dict[str, Any]:
    rules: dict[str, Any] = {"title_xpath": "//h1"}
    for _index in range(depth):
        rules = {"alternates": [rules]}
    return rules


@pytest.mark.parametrize("depth", [33, 1_500])
def test_deep_alternates_return_stable_schema_depth_error(depth: int) -> None:
    rules = _alternate_rules(depth)
    code = "selector_too_complex:schema_depth>32"

    report = selectors.validate_selector_rules(rules)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code


def test_wide_alternates_share_schema_field_limit() -> None:
    rules = {"alternates": [{"title_xpath": "//h1"} for _index in range(257)]}
    code = "selector_too_complex:schema_fields>256"

    report = selectors.validate_selector_rules(rules)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code
