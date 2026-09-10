"""Capture extraction-strategy predecessor behavior."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from Helper_Scripts.web_scraping_phase4.shared import FIXED_ENV, case, metric_patches


def _run_extraction_case(article: Any, fixture_case: Mapping[str, Any]) -> dict[str, Any]:
    with ExitStack() as stack:
        recorder = metric_patches(stack, article)
        stack.enter_context(patch.object(article.random, "uniform", lambda *_args, **_kwargs: 0.0))
        stack.enter_context(patch.dict(os.environ, FIXED_ENV, clear=False))
        article.clear_extraction_caches()
        operation = fixture_case["operation"]
        if operation == "regex":
            result = article.extract_regex_entities(
                fixture_case["html"],
                fixture_case["url"],
                mask_pii=fixture_case["mask_pii"],
            )
        elif operation == "jsonld":
            result = article.extract_jsonld_entities(
                fixture_case["html"],
                fixture_case["url"],
            )
        elif operation == "cluster":
            result = article.extract_cluster_entities(
                fixture_case["html"],
                fixture_case["url"],
                cluster_settings=fixture_case["cluster_settings"],
            )
        elif operation == "pipeline":
            fallback_result = fixture_case.get("fallback_result")

            def _fallback(_html: str, url: str) -> dict[str, Any]:
                return {"url": url, **dict(fallback_result or {})}

            result = article.extract_article_with_pipeline(
                fixture_case["html"],
                fixture_case["url"],
                strategy_order=fixture_case.get("strategy_order"),
                fallback_extractor=_fallback if fallback_result is not None else None,
                allow_llm_extraction=fixture_case["allow_llm_extraction"],
            )
        else:
            raise ValueError(f"Unknown extraction operation: {operation}")

        actual = {
            "cache_stats": article.get_extraction_cache_stats(),
            "metrics": recorder.events,
            "result": result,
        }
        article.clear_extraction_caches()
        return actual


def build_extraction_cases(article: Any) -> list[dict[str, Any]]:
    description_only_jsonld = (
        '<html><head><script type="application/ld+json">'
        '{"@context":"https://schema.org","@type":"Article",'
        '"headline":"Structured title","description":"Structured summary"}'
        "</script></head><body></body></html>"
    )
    cases = [
        case(
            {
                "html": (
                    "<html><head><title>Contacts</title></head><body>"
                    "Email demo@example.com or call +1 (415) 555-2671."
                    "</body></html>"
                ),
                "mask_pii": False,
                "name": "regex_catalog_matches",
                "operation": "regex",
                "url": "https://example.com/contacts",
            }
        ),
        case(
            {
                "html": (
                    '<html><head><script type="application/ld+json">'
                    '{"@context":"https://schema.org","@type":"NewsArticle",'
                    '"headline":"JSON-LD Title","author":{"@type":"Person","name":"Jane Doe"},'
                    '"datePublished":"2024-05-01","articleBody":"JSON-LD body text."}'
                    "</script></head><body></body></html>"
                ),
                "name": "jsonld_article",
                "operation": "jsonld",
                "url": "https://example.com/jsonld",
            }
        ),
        case(
            {
                "cluster_settings": {
                    "cluster_threshold": 0.1,
                    "embed_dims": 32,
                    "max_blocks": 10,
                    "method": "greedy",
                    "min_block_chars": 20,
                    "min_word_count": 4,
                    "prefilter_threshold": 0.0,
                    "tag_keywords": {
                        "research": ["research", "dataset"],
                        "security": ["security", "encryption"],
                    },
                    "tag_top_k": 2,
                },
                "html": (
                    "<html><head><title>Cluster Fixture</title></head><body><article>"
                    "<p>Security research explains encryption controls for a stable local fixture.</p>"
                    "<p>The research dataset contains deterministic examples for repeatable extraction.</p>"
                    "</article></body></html>"
                ),
                "name": "cluster_extraction",
                "operation": "cluster",
                "url": "https://example.com/cluster",
            }
        ),
        case(
            {
                "allow_llm_extraction": False,
                "fallback_result": {
                    "author": "Fixture Author",
                    "content": "Fallback body",
                    "date": "2026-07-27",
                    "extraction_successful": True,
                    "summary": "   ",
                    "title": "Fallback title",
                },
                "html": description_only_jsonld,
                "name": "jsonld_summary_carries_to_fallback",
                "operation": "pipeline",
                "strategy_order": ["jsonld", "trafilatura"],
                "url": "https://example.com/summary",
            }
        ),
        case(
            {
                "allow_llm_extraction": False,
                "fallback_result": {
                    "author": "Fixture Author",
                    "content": "Fallback body",
                    "date": "2026-07-27",
                    "extraction_successful": True,
                    "title": "Fallback title",
                },
                "html": "<html><body><p>Fallback input.</p></body></html>",
                "name": "unknown_strategy_is_traced",
                "operation": "pipeline",
                "strategy_order": ["mystery", "trafilatura"],
                "url": "https://example.com/unknown",
            }
        ),
        case(
            {
                "allow_llm_extraction": False,
                "behavior_change": 1,
                "difference_contract": "change_1_default_regex_non_terminal",
                "html": "<html><body>Contact predecessor@example.com</body></html>",
                "name": "default_regex_is_terminal_in_predecessor",
                "operation": "pipeline",
                "strategy_order": None,
                "url": "https://example.com/regex-default",
            }
        ),
    ]
    for fixture_case in cases:
        fixture_case["expected"] = _run_extraction_case(article, fixture_case)
    return cases
