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


def test_validation_reports_every_legacy_selector_alias_like_predecessor() -> None:
    report = selectors.validate_selector_rules(
        {
            "title_xpath": "//h1",
            "title_selector": "css:.css-abc123",
        },
        html_text='<article><h1 class="css-abc123">Headline</h1></article>',
        include_counts=True,
    )

    assert report == {
        "errors": [],
        "warnings": [
            {
                "key": "title_selector",
                "selector": "css:.css-abc123",
                "warning": "fragile_selector",
                "detail": "fragile class 'css-abc123'",
            }
        ],
        "selector_counts": {
            "title_xpath": 1,
            "title_selector": 1,
        },
    }


def test_validation_reports_missing_alternate_like_predecessor() -> None:
    report = selectors.validate_selector_rules(
        {
            "title_xpath": "//h1",
            "alternates": [{"title_selector": "css:.missing"}],
        },
        html_text="<article><h1>Headline</h1></article>",
        include_counts=True,
    )

    assert report == {
        "errors": [],
        "warnings": [
            {
                "key": "alternates[0].title_selector",
                "selector": "css:.missing",
                "warning": "no_matches",
            }
        ],
        "selector_counts": {
            "title_xpath": 1,
            "alternates[0].title_selector": 0,
        },
    }


def test_validation_reports_pagination_and_supplemental_dsl_selectors() -> None:
    legacy_report = selectors.validate_selector_rules(
        {"pagination": {"next_selector": "css:.missing-next"}},
        html_text="<article><h1>Headline</h1></article>",
        include_counts=True,
    )
    dsl_report = selectors.validate_selector_rules(
        {
            "fields": [
                {
                    "name": "title",
                    "selector": "//h1",
                    "itemSelector": "//span",
                }
            ]
        },
        html_text="<article><h1>Headline</h1><span>Detail</span></article>",
        include_counts=True,
    )

    assert legacy_report == {
        "errors": [],
        "warnings": [],
        "selector_counts": {"pagination.next_selector": 0},
    }
    assert dsl_report == {
        "errors": [],
        "warnings": [],
        "selector_counts": {
            "fields.title": 1,
            "fields.title.item_selector": 1,
        },
    }


def test_validation_deduplicates_exact_selector_specs_before_budgeting() -> None:
    report = selectors.validate_selector_rules(
        {"title_xpath": ["//missing", "//missing"]},
        html_text="<article><h1>Headline</h1></article>",
        include_counts=True,
        _limits=schema._SchemaLimits(max_selector_evaluations=1),
    )

    assert report == {
        "errors": [],
        "warnings": [
            {
                "key": "title_xpath",
                "selector": "//missing",
                "warning": "no_matches",
            }
        ],
        "selector_counts": {"title_xpath": 0},
    }


def test_validation_deduplicates_exact_dsl_specs_per_context() -> None:
    field = {"name": "title", "selector": "//h1"}
    report = selectors.validate_selector_rules(
        {"fields": [field, dict(field)]},
        html_text="<article><h1>Headline</h1></article>",
        include_counts=True,
        _limits=schema._SchemaLimits(max_selector_evaluations=1),
    )

    assert report == {
        "errors": [],
        "warnings": [],
        "selector_counts": {"fields.title": 1},
    }


def test_validation_budgets_all_distinct_configured_selectors() -> None:
    report = selectors.validate_selector_rules(
        {
            "title_xpath": "//h1",
            "title_selector": "css:.headline",
        },
        html_text='<article><h1 class="headline">Headline</h1></article>',
        _limits=schema._SchemaLimits(max_selector_evaluations=1),
    )

    assert report == {
        "errors": [
            {
                "key": "schema",
                "selector": "",
                "error": "selector_too_complex:selector_evaluations>1",
            }
        ],
        "warnings": [],
    }


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


def test_regex_field_uses_regex_dialect_for_variable_length_lookbehind() -> None:
    result = selectors.extract_schema_fields(
        "<article><span>AB123</span></article>",
        "https://example.com/post",
        {
            "fields": [
                {
                    "name": "identifier",
                    "type": "regex",
                    "selector": "//span",
                    "pattern": r"(?<=\b[A-Z]{1,3})(\d+)",
                    "group": 1,
                }
            ]
        },
    )

    assert result["schema_fields"] == {"identifier": "123"}


def test_regex_replace_uses_regex_dialect_with_bounded_group_expansion() -> None:
    result = selectors.extract_schema_fields(
        "<article><h1>AB123</h1></article>",
        "https://example.com/post",
        {
            "fields": [
                {
                    "name": "identifier",
                    "selector": "//h1",
                    "transforms": [
                        {
                            "name": "regex_replace",
                            "pattern": r"(?<=\b[A-Z]{1,3})(\d+)",
                            "repl": r"[\1]",
                        }
                    ],
                }
            ]
        },
    )

    assert result["schema_fields"] == {"identifier": "AB[123]"}


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
        "{title[-1]}",
        "{title[first]}",
        "{title:{title}}",
        "{title!q}",
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
        ("{title!r}", "'Headline'"),
        ("{title[0]}", "H"),
        ("{title:10}", "Headline  "),
        ("{title:>10}", "  Headline"),
        ("{title!s}", "Headline"),
        ("{title!a}", "'Headline'"),
    ],
)
def test_computed_templates_preserve_bounded_predecessor_formatting(
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

    assert report == {"errors": [], "warnings": []}
    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": True,
        "schema_fields": {"title": "Headline", "computed": expected},
        "title": "Headline",
    }


@pytest.mark.parametrize(
    ("format_spec", "expected_length"),
    [
        (">65536", 65_536),
        (".65536", len("Headline")),
    ],
    ids=["width", "precision"],
)
def test_computed_template_format_components_have_an_explicit_hard_boundary(
    format_spec: str,
    expected_length: int,
) -> None:
    exact_width = 65_536
    exact_rules = _computed_rules(f"{{title:{format_spec}}}")
    one_over_spec = format_spec.replace(str(exact_width), str(exact_width + 1))
    one_over_rules = _computed_rules(f"{{title:{one_over_spec}}}")

    exact_report = selectors.validate_selector_rules(exact_rules)
    exact_result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        exact_rules,
    )
    one_over_report = selectors.validate_selector_rules(one_over_rules)
    one_over_result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        one_over_rules,
    )

    assert exact_report["errors"] == []
    assert len(exact_result["schema_fields"]["computed"]) == expected_length
    assert [entry["error"] for entry in one_over_report["errors"]] == ["selector_invalid"]
    assert one_over_result["schema_fields"] == {"title": "Headline"}


def test_computed_template_total_output_has_an_explicit_hard_boundary() -> None:
    exact_chars = 1_048_576

    exact, exact_error = schema._render_computed_template(
        "{title}",
        {"title": "x" * exact_chars},
        schema._SchemaLimits(),
        None,
        "rendered_output",
        None,
    )
    one_over, one_over_error = schema._render_computed_template(
        "{title}",
        {"title": "x" * (exact_chars + 1)},
        schema._SchemaLimits(),
        None,
        "rendered_output",
        None,
    )

    assert exact is not None
    assert len(exact) == exact_chars
    assert exact_error is None
    assert one_over is None
    assert one_over_error == "selector_invalid"


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


def test_injected_computed_template_limit_has_a_stable_complexity_error() -> None:
    code = "selector_too_complex:template_length>4096"
    rules = _computed_rules("x" * 4_097)

    limits = schema._SchemaLimits(max_template_length=4_096)
    report = selectors.validate_selector_rules(rules, _limits=limits)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result == {
        "url": "https://example.com/post",
        "extraction_successful": False,
        "error": code,
    }


def test_injected_computed_render_limit_has_a_stable_complexity_error() -> None:
    code = "selector_too_complex:rendered_output>1000000"
    title = "x" * 500_001

    result = selectors.extract_schema_fields(
        f"<article><h1>{title}</h1></article>",
        "https://example.com/post",
        _computed_rules("{title}{title}"),
        _limits=schema._SchemaLimits(max_rendered_output_chars=1_000_000),
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
            {"max_depth": 32},
            "selector_too_complex:schema_depth>32",
        ),
        (
            {"fields": [{"name": f"field_{index}", "value": "x"} for index in range(257)]},
            {"max_total_fields": 256},
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
    kwargs = {"_limits": schema._SchemaLimits(**limits)}

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

    limits = schema._SchemaLimits(max_depth=32)
    report = selectors.validate_selector_rules(rules, _limits=limits)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
        _limits=limits,
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
def test_injected_projection_budget_without_large_dom(
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
        _limits=schema._SchemaLimits(max_retained_output_chars=1_000_000),
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
        (
            ["//h1", "//h2"],
            1,
            "First",
            "selector_too_complex:selector_evaluations>1",
        ),
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
        if "error" in result:
            assert result["error"] == expected_error
        else:
            assert result["title"] == expected_title
    else:
        assert report["errors"] == []
        assert result["title"] == expected_title


def test_schema_limit_defaults_are_explicit_and_behavior_preserving() -> None:
    assert (
        schema._SchemaLimits(
            max_depth=None,
            max_total_fields=None,
            max_selector_evaluations=None,
            max_aggregate_matches=None,
            max_retained_output_chars=None,
            max_template_length=None,
            max_rendered_output_chars=None,
        )
        == schema._DEFAULT_SCHEMA_LIMITS
    )


def test_default_limits_preserve_depth_33_and_257_fields() -> None:
    depth_rules = _nested_rules(33)
    field_rules = {"fields": [{"name": f"field_{index}", "type": "computed", "value": "x"} for index in range(257)]}

    depth_report = selectors.validate_selector_rules(depth_rules)
    depth_result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        depth_rules,
    )
    field_report = selectors.validate_selector_rules(field_rules)
    field_result = selectors.extract_schema_fields(
        "<article></article>",
        "https://example.com/post",
        field_rules,
    )

    nested_value = depth_result["schema_fields"]
    for _index in range(32):
        nested_value = next(iter(nested_value.values()))
    assert nested_value == {"leaf": "Headline"}
    assert depth_report["errors"] == []
    assert field_report["errors"] == []
    assert len(field_result["schema_fields"]) == 257


def test_default_limits_preserve_more_than_512_selector_evaluations() -> None:
    rules = {"fields": [{"name": f"field_{index}", "selector": "//h1"} for index in range(513)]}
    html_text = "<article><h1>Headline</h1></article>"

    report = selectors.validate_selector_rules(rules, html_text=html_text)
    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
    )

    assert report["errors"] == []
    assert len(result["schema_fields"]) == 513


def test_default_limits_preserve_more_than_10000_aggregate_matches() -> None:
    match_count = 10_001
    html_text = "<ul>" + ("<li>x</li>" * match_count) + "</ul>"

    result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        {"fields": [{"name": "items", "type": "list", "selector": "//li"}]},
    )

    assert len(result["schema_fields"]["items"]) == match_count


def test_default_limits_preserve_more_than_one_million_retained_chars() -> None:
    value = "x" * 1_000_001

    result = selectors.extract_schema_fields(
        f"<article><h1>{value}</h1></article>",
        "https://example.com/post",
        {"fields": [{"name": "value", "selector": "//h1"}]},
    )

    assert result["schema_fields"]["value"] == value


def test_default_limits_preserve_more_than_one_million_rendered_chars() -> None:
    value = "x" * 500_001

    result = selectors.extract_schema_fields(
        f"<article><h1>{value}</h1></article>",
        "https://example.com/post",
        _computed_rules("{title}{title}"),
    )

    assert result["schema_fields"]["computed"] == value + value


def test_default_limits_preserve_templates_longer_than_4096_chars() -> None:
    template = "x" * 4_097

    report = selectors.validate_selector_rules(_computed_rules(template))
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        _computed_rules(template),
    )

    assert report["errors"] == []
    assert result["schema_fields"]["computed"] == template


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


@pytest.mark.parametrize(
    "rules",
    [
        {
            "fields": [
                {
                    "name": "value",
                    "type": "nested",
                    "fields": [{"name": "leaf", "selector": "//i"}],
                },
                {"name": "value", "selector": "//b"},
            ]
        },
        {
            "fields": [
                {
                    "name": "value",
                    "type": "nested",
                    "fields": [{"name": "leaf", "selector": "//i"}],
                },
                {"name": "value", "type": "computed", "value": "bb"},
            ]
        },
        {
            "baseFields": [
                {
                    "name": "value",
                    "type": "nested",
                    "fields": [{"name": "leaf", "selector": "//i"}],
                }
            ],
            "fields": [{"name": "value", "selector": "//b"}],
        },
        {
            "fields": [
                {"name": "value", "selector": "//i"},
                {
                    "name": "value",
                    "type": "nested",
                    "fields": [{"name": "leaf", "selector": "//b"}],
                },
            ]
        },
        {
            "fields": [
                {
                    "name": "value",
                    "type": "nested",
                    "fields": [{"name": "leaf", "selector": "//i"}],
                },
                {
                    "name": "value",
                    "type": "nested",
                    "fields": [{"name": "leaf", "selector": "//b"}],
                },
            ]
        },
    ],
    ids=[
        "nested-to-scalar",
        "nested-to-computed",
        "base-nested-to-field-scalar",
        "scalar-to-nested",
        "nested-to-nested",
    ],
)
def test_shape_changing_slot_replacement_boundaries(rules: dict[str, Any]) -> None:
    html_text = "<article><i>a</i><b>bb</b></article>"
    exact_limits = schema._SchemaLimits(max_retained_output_chars=2)
    one_over_limits = schema._SchemaLimits(max_retained_output_chars=1)
    one_over_code = "selector_too_complex:retained_output_chars>1"

    exact_report = selectors.validate_selector_rules(
        rules,
        html_text=html_text,
        _limits=exact_limits,
    )
    exact_result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=exact_limits,
    )
    one_over_report = selectors.validate_selector_rules(
        rules,
        html_text=html_text,
        _limits=one_over_limits,
    )
    one_over_result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=one_over_limits,
    )

    expected = {"value": "bb"}
    if rules["fields"][-1].get("type") == "nested":
        expected = {"value": {"leaf": "bb"}}
    assert exact_report["errors"] == []
    assert exact_result["schema_fields"] == expected
    assert [entry["error"] for entry in one_over_report["errors"]] == [one_over_code]
    assert one_over_result["error"] == one_over_code


def test_slot_subtree_replacement_preserves_exact_prefix_siblings() -> None:
    rules = {
        "fields": [
            {
                "name": "value",
                "type": "nested",
                "fields": [{"name": "leaf", "selector": "//i"}],
            },
            {"name": "value2", "selector": "//u"},
            {"name": "value", "selector": "//b"},
        ]
    }
    html_text = "<article><i>a</i><u>b</u><b>cc</b></article>"

    exact_report = selectors.validate_selector_rules(
        rules,
        html_text=html_text,
        _limits=schema._SchemaLimits(max_retained_output_chars=3),
    )
    exact_result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=3),
    )
    one_over_report = selectors.validate_selector_rules(
        rules,
        html_text=html_text,
        _limits=schema._SchemaLimits(max_retained_output_chars=2),
    )
    one_over_result = selectors.extract_schema_fields(
        html_text,
        "https://example.com/post",
        rules,
        _limits=schema._SchemaLimits(max_retained_output_chars=2),
    )

    code = "selector_too_complex:retained_output_chars>2"
    assert exact_report["errors"] == []
    assert exact_result["schema_fields"] == {"value": "cc", "value2": "b"}
    assert [entry["error"] for entry in one_over_report["errors"]] == [code]
    assert one_over_result["error"] == code


def test_failed_output_slot_replacement_is_atomic() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=2))
    slot = ("schema_fields", "value")
    budget.retain_output("aa", slot=slot)

    with pytest.raises(schema._SchemaBudgetExceeded):
        budget.retain_output("bbb", slot=slot)

    assert budget.retained_output_chars == 2
    assert budget.output_slots == {slot: 2}


def test_failed_output_slot_subtree_replacement_is_atomic() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=5))
    value_slot = ("schema_fields", "value")
    reservations = {
        value_slot + ("leaf",): "aa",
        ("schema_fields", "value2"): "b",
        ("root", "value"): "r",
    }
    for slot, value in reservations.items():
        budget.retain_output(value, slot=slot)

    assert budget.remaining_output_chars(value_slot) == 3
    with pytest.raises(
        schema._SchemaBudgetExceeded,
        match="selector_too_complex:retained_output_chars>5",
    ):
        budget.retain_output("xxxx", slot=value_slot)

    assert budget.retained_output_chars == 4
    assert budget.output_slots == {slot: len(value) for slot, value in reservations.items()}


class _IterationCountingSlots(dict[tuple[Any, ...], int]):
    def __init__(self, values: dict[tuple[Any, ...], int]) -> None:
        super().__init__(values)
        self.iteration_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        return super().__iter__()

    def items(self):
        self.iteration_calls += 1
        return super().items()


def _output_slot_index_signature(
    budget: schema._SchemaBudget,
) -> dict[tuple[Any, ...], tuple[int, tuple[Any, ...] | None, int]]:
    signature: dict[tuple[Any, ...], tuple[int, tuple[Any, ...] | None, int]] = {}
    pending = [((), budget._output_slot_index)]
    while pending:
        path, node = pending.pop()
        signature[path] = (node.subtree_chars, node.slot, node.slot_chars)
        pending.extend((path + (part,), child) for part, child in node.children.items())
    return signature


def _output_slot_index_node(
    budget: schema._SchemaBudget,
    path: tuple[Any, ...],
) -> schema._OutputSlotIndexNode:
    node = budget._output_slot_index
    for part in path:
        node = node.children[part]
    return node


def _assert_output_slot_index_consistent(budget: schema._SchemaBudget) -> None:
    indexed_slots: dict[tuple[Any, ...], int] = {}
    calculated_totals: dict[int, int] = {}
    pending = [(budget._output_slot_index, False)]
    while pending:
        node, expanded = pending.pop()
        if not expanded:
            pending.append((node, True))
            pending.extend((child, False) for child in node.children.values())
            continue
        expected = node.slot_chars if node.slot is not None else 0
        expected += sum(calculated_totals[id(child)] for child in node.children.values())
        assert node.subtree_chars == expected
        calculated_totals[id(node)] = expected
        if node.slot is not None:
            indexed_slots[node.slot] = node.slot_chars

    assert indexed_slots == budget.output_slots
    assert budget._output_slot_index.subtree_chars == budget.retained_output_chars


def test_output_slot_operations_do_not_iterate_the_global_mapping() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=1_000))
    for index in range(256):
        budget.retain_output("x", slot=("schema_fields", f"value{index}"))
    tracked_slots = _IterationCountingSlots(budget.output_slots)
    budget.output_slots = tracked_slots

    fresh_slot = ("schema_fields", "fresh")
    assert budget.remaining_output_chars(fresh_slot) == 744
    budget.retain_output("y", slot=fresh_slot)
    removed = budget.take_output_prefix(("schema_fields", "value128"))
    assert removed == {("schema_fields", "value128"): 1}
    budget.restore_output_prefix(("schema_fields", "value128"), removed)

    assert tracked_slots.iteration_calls == 0
    assert tracked_slots[fresh_slot] == 1


@pytest.mark.parametrize(
    "runtime_state",
    [
        {"selector_evaluations": 1},
        {"aggregate_matches": 1},
        {"retained_output_chars": 1},
        {"output_slots": {("schema_fields", "value"): 1}},
        {"selection_observer": lambda *_args: None},
        {"enforce_output": False},
    ],
)
def test_schema_budget_constructor_rejects_runtime_state(
    runtime_state: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        schema._SchemaBudget(schema._SchemaLimits(), **runtime_state)


def test_output_slot_snapshot_is_immutable_and_prefix_cannot_be_rebound() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=10))
    slot = ("schema_fields", "value")
    budget.retain_output("aa", slot=slot)
    snapshot = budget.take_output_prefix(slot)
    mutable_snapshot: Any = snapshot

    with pytest.raises(TypeError):
        mutable_snapshot[slot] = 3
    with pytest.raises(TypeError):
        del mutable_snapshot[slot]
    for method in ("clear", "pop", "popitem", "setdefault", "update"):
        assert not hasattr(snapshot, method)
    with pytest.raises(AttributeError):
        mutable_snapshot._prefix = ("schema_fields", "other")

    assert snapshot == {slot: 2}


def test_output_slot_snapshot_wrong_prefix_fails_atomically() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=10))
    prefix = ("schema_fields", "value")
    other_prefix = ("schema_fields", "other")
    budget.retain_output("aa", slot=prefix + ("leaf",))
    budget.retain_output("b", slot=other_prefix)
    snapshot = budget.take_output_prefix(prefix)
    before_slots = dict(budget.output_slots)
    before_total = budget.retained_output_chars
    before_index = _output_slot_index_signature(budget)
    snapshot_error = getattr(schema, "_OutputSlotSnapshotError", RuntimeError)

    with pytest.raises(
        snapshot_error,
        match="output_slot_snapshot_prefix_mismatch",
    ):
        budget.restore_output_prefix(other_prefix, snapshot)

    assert budget.output_slots == before_slots
    assert budget.retained_output_chars == before_total
    assert _output_slot_index_signature(budget) == before_index
    budget.restore_output_prefix(prefix, snapshot)
    _assert_output_slot_index_consistent(budget)


def test_output_slot_snapshot_reuse_fails_atomically() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=10))
    prefix = ("schema_fields", "value")
    budget.retain_output("aa", slot=prefix + ("leaf",))
    snapshot = budget.take_output_prefix(prefix)
    budget.restore_output_prefix(prefix, snapshot)
    before_slots = dict(budget.output_slots)
    before_total = budget.retained_output_chars
    before_index = _output_slot_index_signature(budget)
    snapshot_error = getattr(schema, "_OutputSlotSnapshotError", RuntimeError)

    with pytest.raises(
        snapshot_error,
        match="output_slot_snapshot_already_used",
    ):
        budget.restore_output_prefix(prefix, snapshot)

    assert budget.output_slots == before_slots
    assert budget.retained_output_chars == before_total
    assert _output_slot_index_signature(budget) == before_index
    _assert_output_slot_index_consistent(budget)


def test_output_slot_rollback_reattaches_the_detached_subtree_index() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=10))
    prefix = ("schema_fields", "value")
    budget.retain_output("aa", slot=prefix + ("left",))
    budget.retain_output("bb", slot=prefix + ("right",))
    detached_index = _output_slot_index_node(budget, prefix)

    snapshot = budget.take_output_prefix(prefix)
    assert snapshot._owns_detached_subtree
    budget.restore_output_prefix(prefix, snapshot)

    assert _output_slot_index_node(budget, prefix) is detached_index
    assert not snapshot._owns_detached_subtree
    with pytest.raises(AttributeError):
        _ = snapshot._index_subtree
    _assert_output_slot_index_consistent(budget)


def test_output_slot_index_stays_consistent_across_rollback_and_zero_slots() -> None:
    budget = schema._SchemaBudget(schema._SchemaLimits(max_retained_output_chars=5))
    value_slot = ("schema_fields", "value")
    empty_slot = ("schema_fields", "empty")
    reservations = {
        value_slot + ("leaf",): "aa",
        ("schema_fields", "value2"): "b",
        ("root", "value"): "r",
        empty_slot: "",
    }
    for slot, value in reservations.items():
        budget.retain_output(value, slot=slot)
    _assert_output_slot_index_consistent(budget)

    before_slots = dict(budget.output_slots)
    before_index = _output_slot_index_signature(budget)
    with pytest.raises(
        schema._SchemaBudgetExceeded,
        match="selector_too_complex:retained_output_chars>5",
    ):
        budget.retain_output("xxxx", slot=value_slot)
    assert budget.output_slots == before_slots
    assert _output_slot_index_signature(budget) == before_index
    _assert_output_slot_index_consistent(budget)

    snapshot = budget.take_output_prefix(value_slot)
    assert snapshot == {value_slot + ("leaf",): 2}
    assert value_slot not in _output_slot_index_signature(budget)
    _assert_output_slot_index_consistent(budget)

    budget.restore_output_prefix(value_slot, snapshot)
    assert budget.output_slots == before_slots
    _assert_output_slot_index_consistent(budget)

    budget.retain_output("bb", slot=value_slot)
    assert budget.output_slots == {
        ("schema_fields", "value2"): 1,
        ("root", "value"): 1,
        empty_slot: 0,
        value_slot: 2,
    }
    _assert_output_slot_index_consistent(budget)

    empty_snapshot = budget.take_output_prefix(empty_slot)
    assert empty_snapshot == {empty_slot: 0}
    assert empty_slot not in _output_slot_index_signature(budget)
    budget.restore_output_prefix(empty_slot, empty_snapshot)
    assert empty_slot in budget.output_slots
    _assert_output_slot_index_consistent(budget)


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
def test_injected_depth_limit_handles_deep_alternates_iteratively(depth: int) -> None:
    rules = _alternate_rules(depth)
    code = "selector_too_complex:schema_depth>32"
    limits = schema._SchemaLimits(max_depth=32)

    report = selectors.validate_selector_rules(rules, _limits=limits)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code


def test_wide_alternates_share_injected_schema_field_limit() -> None:
    rules = {"alternates": [{"title_xpath": "//h1"} for _index in range(257)]}
    code = "selector_too_complex:schema_fields>256"
    limits = schema._SchemaLimits(max_total_fields=256)

    report = selectors.validate_selector_rules(rules, _limits=limits)
    result = selectors.extract_schema_fields(
        "<article><h1>Headline</h1></article>",
        "https://example.com/post",
        rules,
        _limits=limits,
    )

    assert [entry["error"] for entry in report["errors"]] == [code]
    assert result["error"] == code
