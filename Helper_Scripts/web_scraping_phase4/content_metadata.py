"""Capture content-formatting and metadata predecessor behavior."""

from __future__ import annotations

from typing import Any

from Helper_Scripts.web_scraping_phase4.shared import case, normalize_formatted_metadata


def build_content_cases(article: Any) -> list[dict[str, Any]]:
    cases = [
        case(
            {
                "html": (
                    "<html><body><h1>Fixture &amp; Title</h1>"
                    "<p>First paragraph.</p>"
                    "<p>Second <strong>bold</strong> paragraph.</p></body></html>"
                ),
                "name": "paragraph_and_inline_formatting",
                "operation": "convert_html_to_markdown",
            }
        ),
        case(
            {
                "html": "<div>Lead<span>inline</span></div><p>Tail paragraph.</p>",
                "name": "mixed_block_and_paragraph_formatting",
                "operation": "convert_html_to_markdown",
            }
        ),
    ]
    for fixture_case in cases:
        fixture_case["expected"] = article.convert_html_to_markdown(fixture_case["html"])
    return cases


def _metadata_inspection(handler: Any, content: str) -> dict[str, Any]:
    metadata, clean_content = handler.extract_metadata(content)
    return {
        "clean_content": clean_content,
        "content_hash": handler.get_content_hash(content),
        "has_metadata": handler.has_metadata(content),
        "metadata": metadata,
        "stripped": handler.strip_metadata(content),
    }


def build_metadata_cases(article: Any) -> list[dict[str, Any]]:
    handler = article.ContentMetadataHandler
    canonical_envelope = (
        "  [METADATA]\n"
        '{"url":"https://example.com/article","literal":"brackets [{]} and \\"quotes\\""}\n'
        "[/METADATA]\n\nArticle body"
    )
    accepted_nested = '[METADATA]{"value":' + "[" * 63 + "0" + "]" * 63 + "}[/METADATA]\nArticle body"
    rejected_nested = '[METADATA]{"value":' + "[" * 64 + "0" + "]" * 64 + "}[/METADATA]\nArticle body"
    cases = [
        case(
            {
                "additional_metadata": {"author": "Ada", "language": "en"},
                "content": "Fixture body with caf\u00e9.",
                "name": "canonical_formatted_envelope",
                "operation": "format",
                "pipeline": "FixturePipeline",
                "url": "https://example.com/article",
            }
        ),
        case(
            {
                "content": canonical_envelope,
                "name": "canonical_envelope_inspection",
                "operation": "inspect",
            }
        ),
        case(
            {
                "content": '[METADATA]{"url":"https://example.com"}\nArticle body',
                "name": "malformed_envelope_passes_through",
                "operation": "inspect",
            }
        ),
        case(
            {
                "content": accepted_nested,
                "name": "nesting_boundary_is_accepted",
                "operation": "inspect",
            }
        ),
        case(
            {
                "content": rejected_nested,
                "name": "nesting_over_boundary_is_rejected",
                "operation": "inspect",
            }
        ),
        case(
            {
                "name": "metadata_only_changes_do_not_change_body_hash",
                "new_content": '[METADATA]{"version":2}[/METADATA]\nSame body',
                "old_content": '[METADATA]{"version":1}[/METADATA]\nSame body',
                "operation": "content_changed",
            }
        ),
        case(
            {
                "name": "body_changes_are_detected",
                "new_content": '[METADATA]{"version":2}[/METADATA]\nNew body',
                "old_content": '[METADATA]{"version":1}[/METADATA]\nOld body',
                "operation": "content_changed",
            }
        ),
    ]
    for fixture_case in cases:
        if fixture_case["operation"] == "format":
            fixture_case["expected"] = normalize_formatted_metadata(
                handler.format_content_with_metadata(
                    fixture_case["url"],
                    fixture_case["content"],
                    pipeline=fixture_case["pipeline"],
                    additional_metadata=fixture_case["additional_metadata"],
                )
            )
        elif fixture_case["operation"] == "inspect":
            fixture_case["expected"] = _metadata_inspection(
                handler,
                fixture_case["content"],
            )
        else:
            fixture_case["expected"] = handler.content_changed(
                fixture_case["old_content"],
                fixture_case["new_content"],
            )
    return cases
