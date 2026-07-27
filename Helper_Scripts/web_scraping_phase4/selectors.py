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
