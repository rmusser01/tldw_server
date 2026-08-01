"""Capture selector validation and extraction predecessor behavior."""

from __future__ import annotations

from typing import Any

from Helper_Scripts.web_scraping_phase4.shared import case


def build_selector_cases(fetchers: Any) -> list[dict[str, Any]]:
    schema_html = (
        "<html><body><article><h1> Example Title </h1>"
        '<time>2025-01-15</time><span class="views">1,234</span>'
        '<p class="content">First paragraph.</p><p class="content">Second paragraph.</p>'
        '<a class="more" href="/read-more">Read more</a>'
        '<span class="tag">News</span><span class="tag">Tech</span>'
        "</article></body></html>"
    )
    cases = [
        case(
            {
                "html": ("<html><body><h1>One</h1><h1>Two</h1>" '<div class="body">Body</div></body></html>'),
                "include_counts": True,
                "name": "validation_counts_and_warnings",
                "operation": "validate",
                "rules": {
                    "content_xpath": "//div[@class='body']",
                    "title_xpath": "//h1",
                },
            }
        ),
        case(
            {
                "behavior_change": 7,
                "difference_contract": "change_7_selector_invalid",
                "include_counts": False,
                "name": "invalid_xpath_error",
                "operation": "validate",
                "rules": {"content_xpath": "//article["},
            }
        ),
        case(
            {
                "include_counts": False,
                "name": "selector_complexity_guard",
                "operation": "validate",
                "rules": {"content_xpath": "//div" + "/span" * 200},
            }
        ),
        case(
            {
                "base_url": "https://example.com/post",
                "html": schema_html,
                "name": "schema_dsl_transforms_and_lists",
                "operation": "extract_schema_fields",
                "rules": {
                    "baseFields": [
                        {
                            "name": "title",
                            "selector": ".//h1",
                            "transforms": ["strip"],
                            "type": "text",
                        },
                        {
                            "join_with": "\n",
                            "name": "content",
                            "selector": ".//p[@class='content']",
                            "type": "text",
                        },
                        {
                            "name": "published",
                            "selector": ".//time",
                            "transforms": [{"format": "%Y-%m-%d", "name": "date_normalize"}],
                            "type": "text",
                        },
                        {
                            "name": "views",
                            "selector": ".//span[@class='views']",
                            "transforms": [{"name": "number_normalize"}],
                            "type": "text",
                        },
                    ],
                    "baseSelector": "//article",
                    "fields": [
                        {
                            "attribute": "href",
                            "name": "link",
                            "selector": ".//a[@class='more']",
                            "transforms": [{"name": "urljoin"}],
                            "type": "attribute",
                        },
                        {
                            "itemType": "text",
                            "name": "tags",
                            "selector": ".//span[@class='tag']",
                            "transforms": ["lowercase"],
                            "type": "list",
                        },
                        {
                            "from": "title",
                            "name": "slug",
                            "transforms": [
                                "lowercase",
                                {
                                    "name": "regex_replace",
                                    "pattern": r"\s+",
                                    "repl": "-",
                                },
                            ],
                            "type": "computed",
                        },
                    ],
                    "name": "article",
                },
            }
        ),
        case(
            {
                "base_url": "https://example.com/templates",
                "html": "<html><body><article><h1>Caf&#233; Title</h1></article></body></html>",
                "name": "computed_template_format_compatibility",
                "operation": "extract_schema_fields",
                "rules": {
                    "baseFields": [
                        {
                            "name": "title",
                            "selector": ".//h1",
                            "type": "text",
                        }
                    ],
                    "baseSelector": "//article",
                    "fields": [
                        {
                            "name": "formatted",
                            "template": (
                                "repr={title!r}|str={title!s}|ascii={title!a}|"
                                "first={title[0]}|left=[{title:20}]|"
                                "right=[{title:>20}]|missing=[{missing}]"
                            ),
                            "type": "computed",
                        }
                    ],
                    "name": "template_compatibility",
                },
            }
        ),
        case(
            {
                "base_url": "https://example.com/templates/mapping",
                "html": (
                    "<html><body><article><h1>Headline</h1>" '<span class="author">Ada</span></article></body></html>'
                ),
                "name": "computed_template_mapping_and_attribute_compatibility",
                "operation": "extract_schema_fields",
                "rules": {
                    "fields": [
                        {
                            "name": "title",
                            "selector": "//h1",
                            "type": "text",
                        },
                        {
                            "fields": [
                                {
                                    "name": "name",
                                    "selector": "//span[@class='author']",
                                }
                            ],
                            "name": "author",
                            "type": "nested",
                        },
                        {
                            "name": "byline",
                            "template": "By {author[name]}|type={title.__class__}",
                            "type": "computed",
                        },
                    ]
                },
            }
        ),
        case(
            {
                "base_url": "https://example.com/transforms",
                "html": "<html><body><article><h1>x</h1></article></body></html>",
                "name": "unknown_prepend_append_transform_noops",
                "operation": "extract_schema_fields",
                "rules": {
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
                },
            }
        ),
        case(
            {
                "html": "<html><body><h1>XPath title</h1><h2>CSS one</h2><h2>CSS two</h2></body></html>",
                "include_counts": True,
                "name": "validation_checks_both_title_aliases",
                "operation": "validate",
                "rules": {
                    "title_selector": "css:h2",
                    "title_xpath": "//h1",
                },
            }
        ),
        case(
            {
                "html": "<html><body><h1>Primary title</h1></body></html>",
                "include_counts": True,
                "name": "validation_preserves_missing_alternate_path",
                "operation": "validate",
                "rules": {
                    "alternates": [
                        {
                            "content_xpath": "//div[@class='missing']",
                        }
                    ],
                    "title_xpath": "//h1",
                },
            }
        ),
        case(
            {
                "html": (
                    "<html><body><article><h1>Headline</h1>" "<ul><li>One</li><li>Two</li></ul></article></body></html>"
                ),
                "include_counts": True,
                "name": "validation_dsl_item_selector_is_compile_only",
                "operation": "validate",
                "rules": {
                    "fields": [
                        {
                            "itemSelector": ".//li",
                            "name": "headline",
                            "selector": "//h1",
                            "type": "text",
                        }
                    ]
                },
            }
        ),
        case(
            {
                "html": (
                    '<html><body><article class="primary"><h2>Primary</h2></article>'
                    '<article class="secondary"><h2>Secondary</h2></article></body></html>'
                ),
                "include_counts": True,
                "name": "validation_dsl_nested_selector_uses_root_document",
                "operation": "validate",
                "rules": {
                    "fields": [
                        {
                            "fields": [
                                {
                                    "name": "heading",
                                    "selector": ".//h2",
                                    "type": "text",
                                }
                            ],
                            "name": "article",
                            "selector": "//article[@class='primary']",
                            "type": "nested",
                        }
                    ]
                },
            }
        ),
        case(
            {
                "base_url": "https://example.com/regex-fallback",
                "behavior_change": 4,
                "capture_regex_error": True,
                "difference_contract": "change_4_selector_regex_failure_returns_original",
                "html": "<html><body><article><p>AB123</p></article></body></html>",
                "name": "regex_replace_stdlib_invalid_lookbehind_fallback",
                "operation": "extract_schema_fields",
                "rules": {
                    "baseSelector": "//article",
                    "fields": [
                        {
                            "name": "content",
                            "selector": ".//p",
                            "transforms": [
                                {
                                    "name": "regex_replace",
                                    "pattern": "(?<=[A-Z]{1,3})(\\d+)",
                                    "repl": "[\\1]",
                                }
                            ],
                            "type": "text",
                        }
                    ],
                    "name": "regex_fallback",
                },
            }
        ),
        case(
            {
                "base_url": "https://example.com/regex-unicode",
                "html": "<html><body><article><p>e&#769;</p></article></body></html>",
                "name": "regex_replace_stdlib_combining_mark_parity",
                "operation": "extract_schema_fields",
                "rules": {
                    "baseSelector": "//article",
                    "fields": [
                        {
                            "name": "content",
                            "selector": ".//p",
                            "transforms": [
                                {
                                    "name": "regex_replace",
                                    "pattern": "\\w",
                                    "repl": "X",
                                }
                            ],
                            "type": "text",
                        }
                    ],
                    "name": "regex_unicode",
                },
            }
        ),
        case(
            {
                "base_url": "https://example.com/article",
                "html": (
                    "<html><body><main><h1>Legacy title</h1>"
                    '<div class="body"><p>One</p><p>Two</p></div>'
                    '<a rel="author">Grace Hopper</a></main></body></html>'
                ),
                "name": "legacy_selector_field_extraction",
                "operation": "extract_schema_fields",
                "rules": {
                    "author_xpath": ".//a[@rel='author']",
                    "base_xpath": "//main",
                    "content_xpath": ".//div[@class='body']/p",
                    "title_xpath": ".//h1",
                },
            }
        ),
    ]

    for fixture_case in cases:
        fetchers.clear_selector_caches()
        if fixture_case["operation"] == "validate":
            result = fetchers.validate_selector_rules(
                fixture_case["rules"],
                html_text=fixture_case.get("html"),
                include_counts=fixture_case["include_counts"],
            )
        elif fixture_case.get("capture_regex_error"):
            try:
                result = fetchers.extract_schema_fields(
                    fixture_case["html"],
                    fixture_case["base_url"],
                    fixture_case["rules"],
                )
            except fetchers.re.error:
                fixture_case["expected"] = {"outcome": "regex_error", "value": None}
            else:
                fixture_case["expected"] = {
                    "outcome": "returned",
                    "value": {
                        "cache_stats": fetchers.get_selector_cache_stats(),
                        "result": result,
                    },
                }
            fetchers.clear_selector_caches()
            continue
        else:
            result = fetchers.extract_schema_fields(
                fixture_case["html"],
                fixture_case["base_url"],
                fixture_case["rules"],
            )
        fixture_case["expected"] = {
            "cache_stats": fetchers.get_selector_cache_stats(),
            "result": result,
        }
        fetchers.clear_selector_caches()
    return cases
