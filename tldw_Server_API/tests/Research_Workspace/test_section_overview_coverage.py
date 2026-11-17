"""Structural overview coverage is derived from cited prose, not processing or judge verdicts."""

from __future__ import annotations

import copy
import importlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytestmark = pytest.mark.unit


def _sections():
    return [
        {"title": "1. Agency A", "status": "completed", "batches": [
            {"evidence": [{"id": "E9.8.7", "quote": "A proposes reform."}]},
        ]},
        {"title": "27. Financial Regulatory Agencies", "status": "completed", "batches": [
            {"evidence": [{"id": "E2.1.1", "quote": "B proposes oversight."}]},
        ]},
    ]


def _coverage(text, sections=None, **kwargs):
    module = importlib.import_module("tldw_Server_API.app.core.Research_Workspace.section_overview_coverage")
    return module.overview_coverage(text, _sections() if sections is None else sections, **kwargs)


def test_registered_ownership_does_not_infer_section_from_id_prefix():
    assert _coverage("Agency A proposes reform [E9.8.7].") == {
        "selected_sections": 2, "completed_sections": 2, "cited_sections": 1,
        "missing_sections": [{"index": 1, "title": "27. Financial Regulatory Agencies"}], "complete": False,
    }


@pytest.mark.parametrize("text", [
    "Both propose reforms [E9.8.7] [E2.1.1].",
    "Both propose reforms [E9.8.7], [E2.1.1].",
    "Both propose reforms [E9.8.7]; [E2.1.1].",
    "Both propose reforms [E9.8.7]\n[E2.1.1].",
    "Agency A proposes reform [E9.8.7].\n\nFinancial regulators propose oversight [E2.1.1].",
    "- Agency A proposes reform [E9.8.7].\n- Financial regulators propose oversight [E2.1.1].",
])
def test_prose_with_literal_grouped_or_separate_citations_covers_both_chapters(text):
    assert _coverage(text)["complete"] is True


@pytest.mark.parametrize("text", [
    "[E9.8.7] [E2.1.1]",
    "# Agency A [E9.8.7]\n\n## Financial Regulatory Agencies [E2.1.1]",
    "Agency A [E9.8.7]\n===\n\nFinancial Regulatory Agencies [E2.1.1]\n---",
    "1. Agency A [E9.8.7].\n27. Financial Regulatory Agencies [E2.1.1].",
    "**Agency A** [E9.8.7].\n\n**Financial Regulatory Agencies** [E2.1.1].",
    "```markdown\nA proposes reform [E9.8.7].\nB proposes oversight [E2.1.1].\n```",
    "~~~\nA proposes reform [E9.8.7].\nB proposes oversight [E2.1.1].\n~~~",
    "    A proposes reform [E9.8.7].\n    B proposes oversight [E2.1.1].",
    "`A proposes reform [E9.8.7].`\n\n`B proposes oversight [E2.1.1].`",
    "[E9.8.7](https://example.test/a)\n\n![E2.1.1](https://example.test/b)",
    "[E9.8.7]: https://example.test/a\n[E2.1.1]: https://example.test/b",
    "References: [E9.8.7] [E2.1.1].",
    "Source section: Agency A [E9.8.7].\n\nChapter: Financial Regulatory Agencies [E2.1.1].",
    "- **Source section:** Agency A [E9.8.7].\n- Section: 27. Financial Regulatory Agencies [E2.1.1].",
    "> ```\n> Agency A proposes reform [E9.8.7].\n> Financial regulators propose oversight [E2.1.1].\n> ```",
    "- ```\n  Agency A proposes reform [E9.8.7].\n  Financial regulators propose oversight [E2.1.1].\n  ```",
    "> - ~~~\n>   Agency A proposes reform [E9.8.7].\n>   Financial regulators propose oversight [E2.1.1].\n>   ~~~",
    "# References\n\nA proposes reform [E9.8.7].\n\n## Detail\nB proposes oversight [E2.1.1].",
    "<!-- A proposes reform [E9.8.7].\nB proposes oversight [E2.1.1]. -->",
    "A proposes reform E9.8.7. B proposes oversight E2.1.1.",
    "Both propose reform [E9.8.7–E2.1.1].",
    "Both propose reform [E9.8.7, E2.1.1–E2.1.5].",
    "A proposes reform [E1.1.1]. B proposes oversight [E27.1.1].",
    "1. [E9.8.7]\n2. [E2.1.1]",
])
def test_nonprose_inventory_or_unregistered_citations_cannot_cover_chapters(text):
    assert _coverage(text)["cited_sections"] == 0


@pytest.mark.parametrize("separator", [". ", "! ", "? ", "\n\n"])
def test_bare_citation_after_boundary_does_not_borrow_previous_prose(separator):
    assert _coverage("Agency A proposes reform [E9.8.7]" + separator + "[E2.1.1]")["cited_sections"] == 1


def test_reference_inventory_ends_at_next_same_level_heading():
    text = "## References\nA proposes reform [E9.8.7].\n\n### Details\nB proposes oversight [E2.1.1].\n\n## Proposals\nAgency A proposes reform [E9.8.7]."
    assert _coverage(text)["missing_sections"] == [{"index": 1, "title": "27. Financial Regulatory Agencies"}]


def test_ambiguous_registered_id_cannot_cover_either_owner():
    sections = _sections()
    sections[1]["batches"][0]["evidence"].append({"id": "E9.8.7", "quote": "Different owned passage."})
    assert _coverage("A proposes reform [E9.8.7].", sections)["cited_sections"] == 0


def test_duplicate_titles_do_not_merge_chapter_identity():
    sections = _sections()
    sections[1]["title"] = sections[0]["title"]
    assert _coverage("A proposes reform [E9.8.7].", sections)["missing_sections"] == [{"index": 1, "title": "1. Agency A"}]


def test_failed_and_unread_sections_are_separate_from_completed_overview_denominator():
    sections = _sections()
    sections[1]["status"] = "failed"
    sections.append({"title": "Unread chapter", "status": "unread", "batches": []})
    assert _coverage("A proposes reform [E9.8.7].", sections) == {
        "selected_sections": 3, "completed_sections": 1, "cited_sections": 1,
        "missing_sections": [], "complete": True,
    }


def test_source_validated_map_limits_saved_registered_citations():
    result = _coverage("Both propose reform [E9.8.7] [E2.1.1].", evidence_by_id={"E9.8.7": {"text": "A proposes reform."}})
    assert result["complete"] is False
    assert result["cited_sections"] == 0


def test_source_validation_cannot_authorize_an_unselected_id():
    result = _coverage("A proposes reform [E77.1.1].", evidence_by_id={"E77.1.1": {"text": "Unselected source."}})
    assert result["cited_sections"] == 0


def test_lone_carriage_return_keeps_bare_citation_paragraph_separate():
    assert _coverage("Agency A proposes reform [E9.8.7].\r\r[E2.1.1]")["cited_sections"] == 1


@pytest.mark.parametrize("label", ["Evidence IDs", "Citation IDs"])
def test_explicit_id_inventory_is_not_prose(label):
    assert _coverage(f"{label}: [E9.8.7, E2.1.1].")["cited_sections"] == 0


def test_numbered_chapter_title_inventory_is_not_prose():
    text = "**Chapter 1: Agency A** [E9.8.7].\n\n**Chapter 2: Financial Regulatory Agencies** [E2.1.1]."
    assert _coverage(text)["cited_sections"] == 0


@pytest.mark.parametrize("code", [
    "10. ```\n    sample [E2.1.1]\n    ```\n\n",
    "> ```\n> sample [E2.1.1]\n\n",
])
def test_container_fence_ends_before_root_prose(code):
    assert _coverage(code + "Agency A proposes reform [E9.8.7].")["cited_sections"] == 1


@pytest.mark.parametrize("sections", [[], [{"title": "Unread", "status": "unread", "batches": []}]])
def test_no_completed_chapter_does_not_report_complete_coverage(sections):
    assert _coverage("A proposes reform [E9.8.7].", sections)["complete"] is False


def test_coverage_ignores_saved_counts_and_judge_verdicts_and_does_not_mutate_input():
    sections = _sections()
    sections[0]["claim_verification"] = {"verdict": "unsupported"}
    original = copy.deepcopy(sections)
    assert _coverage("A proposes reform [E9.8.7]. B proposes oversight [E2.1.1].", sections)["complete"] is True
    assert sections == original


@given(st.permutations(["E9.8.7", "E2.1.1"]))
def test_registered_id_order_does_not_change_completed_coverage(ids):
    result = _coverage("Both propose reform " + " ".join(f"[{value}]" for value in ids))
    assert result["cited_sections"] == result["completed_sections"] == 2
