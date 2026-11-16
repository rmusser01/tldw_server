"""Source-bound contributor context must never turn a processing split into a byline."""

from __future__ import annotations

import copy
import importlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytestmark = pytest.mark.unit


def module():
    return importlib.import_module("tldw_Server_API.app.core.Research_Workspace.section_attribution")


def fixture(*, children=True, second_heading="**Change Agency**"):
    introduction = "# 1. Agency policy\n\n**Volume Editor**\n\n" if children else "# 1. Agency policy\n\n**Ada Example**\n\n"
    first = "**Keep Agency**\n\n**Ada Example**\n\n" if children else ""
    pages = [
        introduction + first + "The agency should remain. " * 70,
        "Running header\nThe first contributor's final sentence.\n\n"
        + (second_heading + "\n\n**Bert Sample**\n\n" if children else "")
        + "The agency should change. " * 70
        + "\n\n**ENDNOTES**\nAda Example and Bert Sample are cited in shared references.\n",
        "# 2. Next chapter\nUnselected Person\n",
    ]
    content = "\n\n".join(pages)
    offset, page_map = 0, []
    for number, page in enumerate(pages, 1):
        page_map.append({"page_number": number, "printed_page_label": str(number + 6),
                         "start_char": offset, "end_char": offset + len(page)})
        offset += len(page) + 2
    nodes = [{"id": "chapter", "title": "1. Agency policy", "level": 1, "target_start": 7}]
    if children:
        nodes.extend([
            {"id": "keep", "title": "Keep Agency", "parent_id": "chapter", "level": 2, "target_start": 7},
            {"id": "change", "title": "Change Agency", "parent_id": "chapter", "level": 2, "target_start": 8},
        ])
    nodes.append({"id": "next", "title": "2. Next chapter", "level": 1, "target_start": 9})
    for node in nodes:
        node.update(source="generated_toc", target_type="page")
    snapshot = {"content": content, "nodes": nodes, "page_map": page_map, "navigation_version": "nav1"}
    section = {"node_id": "chapter", "source_id": "source1", "media_id": 7, "navigation_version": "nav1",
               "start_char": 0, "end_char": page_map[2]["start_char"]}
    return section, snapshot


def author_at(name, text, source, contexts):
    start = source.index(text)
    return module().contextual_author(name, start, start + len(text), text, contexts)


def test_single_author_continues_after_a_processing_boundary_without_changing_source():
    section, snapshot = fixture(children=False)
    before = copy.deepcopy(snapshot)
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "The first contributor's final sentence.", snapshot["content"], contexts) == "Ada Example"
    assert snapshot == before
    heading = contexts[0]
    assert heading["source_id"] == "source1" and heading["navigation_version"] == "nav1"
    assert heading["heading_quote"] == snapshot["content"][heading["heading_start_char"]:heading["heading_end_char"]]
    assert heading["page_spans"][0]["page_number"] == 1


def test_same_page_boundary_keeps_closing_text_and_prevents_parent_or_sibling_attribution():
    section, snapshot = fixture()
    contexts = module().build_section_attribution_contexts(section, snapshot)
    source = snapshot["content"]
    assert author_at("Ada Example", "The first contributor's final sentence.", source, contexts) == "Ada Example"
    assert author_at("Bert Sample", "The first contributor's final sentence.", source, contexts) == ""
    assert author_at("Bert Sample", "The agency should change.", source, contexts) == "Bert Sample"
    assert author_at("Ada Example", "The agency should change.", source, contexts) == ""
    assert author_at("Volume Editor", "The agency should change.", source, contexts) == ""


def test_a_passage_straddling_contributors_cannot_claim_one_inherited_author():
    section, snapshot = fixture()
    contexts = module().build_section_attribution_contexts(section, snapshot)
    source = snapshot["content"]
    start = source.index("The first contributor's final sentence.")
    end = source.index("The agency should change.") + len("The agency should change.")
    assert module().contextual_author("Bert Sample", start, end, source[start:end], contexts) == ""


def test_selected_child_has_no_parent_or_sibling_context():
    section, snapshot = fixture()
    source = snapshot["content"]
    section.update(node_id="change", start_char=source.index("**Change Agency**"))
    contexts = module().build_section_attribution_contexts(section, snapshot)
    rendered = module().render_attribution_context(contexts)
    assert "Ada Example" not in rendered and "Volume Editor" not in rendered
    assert "Bert Sample" in rendered and "Unselected Person" not in rendered


@pytest.mark.parametrize("reference_heading", ["**ENDNOTES**", "## References", "Bibliography", "**Works Cited**"])
def test_reference_sections_do_not_authorize_even_a_literal_cited_name(reference_heading):
    section, snapshot = fixture(children=False)
    old = "**ENDNOTES**"
    replacement = reference_heading + " " * max(0, len(old) - len(reference_heading))
    delta = len(replacement) - len(old)
    snapshot["content"] = snapshot["content"].replace(old, replacement)
    snapshot["page_map"][1]["end_char"] += delta
    for field in ("start_char", "end_char"):
        snapshot["page_map"][2][field] += delta
    section["end_char"] += delta
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "Ada Example and Bert Sample are cited in shared references.", snapshot["content"], contexts) == ""
    assert all("shared references" not in context["heading_quote"] for context in contexts)


@pytest.mark.parametrize("heading", ["No matched heading", "**Change Agency**\n\n**Change Agency**"])
def test_unresolved_child_does_not_fall_back_to_first_author(heading):
    section, snapshot = fixture(second_heading=heading)
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "The agency should change.", snapshot["content"], contexts) == ""
    assert any(context["type"] == "unresolved" for context in contexts)


def test_out_of_page_child_heading_fails_closed():
    section, snapshot = fixture()
    snapshot["nodes"][2]["target_start"] = 9
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "The agency should change.", snapshot["content"], contexts) == ""


def test_unknown_author_and_absent_byline_are_not_invented():
    section, snapshot = fixture(children=False)
    snapshot["content"] = snapshot["content"].replace("Ada Example", "Anonymous??")
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "The agency should change.", snapshot["content"], contexts) == ""
    assert author_at(None, "The agency should change.", snapshot["content"], contexts) == ""


def test_legacy_literal_author_is_limited_to_actual_passage_not_another_batch_position():
    assert module().contextual_author("Ada", 0, 19, "Ada supports this.", []) == ""
    assert module().contextual_author("Ada", 30, 48, "Ada supports this.", []) == "Ada"
    assert module().contextual_author("Ada", 30, 39, "A policy.", []) == ""


def test_whole_document_selection_cannot_inherit_its_opening_byline():
    section, snapshot = fixture(children=False)
    section.update(node_id=None, start_char=0, end_char=len(snapshot["content"]))
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert contexts == []
    assert author_at("Ada Example", "The agency should change.", snapshot["content"], contexts) == ""
    assert module().contextual_author("Ada", 30, 48, "Ada supports this.", contexts) == "Ada"


def test_renderer_limits_context_to_the_actual_processing_range():
    section, snapshot = fixture()
    contexts = module().build_section_attribution_contexts(section, snapshot)
    source = snapshot["content"]
    start = source.index("The agency should change.")
    rendered = module().render_attribution_context(contexts, start, start + 20)
    assert "Bert Sample" in rendered and "Ada Example" not in rendered
    assert module().render_attribution_context(contexts, len(source), len(source) + 20) == ""


def test_later_child_keeps_main_byline_only_before_child_start():
    source = "# Chapter\n\n**Ada Example**\n\nMain policy.\n\n# Afterword\n\n**Bert Sample**\n\nLater essay."
    boundary = source.index("# Afterword")
    section = {"source_id": "s", "media_id": 1, "node_id": "root", "navigation_version": "v", "start_char": 0, "end_char": len(source)}
    snapshot = {"content": source, "page_map": [], "navigation_version": "v", "nodes": [
        {"id": "root", "title": "Chapter", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)},
        {"id": "tail", "parent_id": "root", "title": "Afterword", "level": 2, "target_type": "char_range", "target_start": boundary, "target_end": len(source)}]}
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "Main policy.", source, contexts) == "Ada Example"
    assert author_at("Ada Example", "Later essay.", source, contexts) == ""
    assert author_at("Bert Sample", "Later essay.", source, contexts) == "Bert Sample"


def test_verified_afterword_start_preserves_main_prefix_when_later_outline_entry_is_invalid():
    pages = ["# 1. Main chapter\n\n**Ada Example**\n\nMain policy.\n",
             "Running header\n\n**Afterword**\n\n**Bert Sample**\n\nLater essay.\n",
             "# 2. Next chapter\nOther text.\n"]
    source = "\n\n".join(pages)
    nodes = [
        {"id": "root", "title": "1. Main chapter", "level": 1, "target_start": 7},
        {"id": "tail", "title": "Afterword", "parent_id": "root", "level": 2, "target_start": 8},
        {"id": "bad", "title": "Bad trailing entry", "parent_id": "root", "level": 2, "target_start": 2025},
        {"id": "next", "title": "2. Next chapter", "level": 1, "target_start": 9},
    ]
    offset, page_map = 0, []
    for index, page in enumerate(pages):
        page_map.append({"page_number": index + 1, "printed_page_label": str(index + 7), "start_char": offset, "end_char": offset + len(page)})
        offset += len(page) + 2
    for node in nodes:
        node.update(target_type="page", source="generated_toc")
    section = {"source_id": "s", "media_id": 1, "node_id": "root", "navigation_version": "v", "start_char": 0, "end_char": page_map[2]["start_char"]}
    snapshot = {"content": source, "page_map": page_map, "navigation_version": "v", "nodes": nodes}
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "Main policy.", source, contexts) == "Ada Example"
    assert author_at("Ada Example", "Later essay.", source, contexts) == ""
    assert author_at("Bert Sample", "Later essay.", source, contexts) == ""
    unknown = next(context for context in contexts if context["type"] == "unresolved")
    assert (unknown["start_char"], unknown["end_char"]) == (source.index("**Afterword**"), section["end_char"])


@pytest.mark.parametrize("parent", ["keep", "change", "missing"])
def test_malformed_parent_relationships_terminate_without_inherited_attribution(parent):
    section, snapshot = fixture()
    snapshot["nodes"][1]["parent_id"] = parent
    if parent == "change":
        snapshot["nodes"][2]["parent_id"] = "keep"
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert author_at("Ada Example", "The agency should change.", snapshot["content"], contexts) == ""


def test_many_headings_preserve_an_explicit_omission_instead_of_silent_partial_context():
    paragraphs = [f"# Part {i}\n\n**Author {i}**\n\n" + "Policy details. " * 80 for i in range(40)]
    source = "\n\n".join(paragraphs)
    nodes = [{"id": "root", "title": "All parts", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)}]
    offset = 0
    for i, paragraph in enumerate(paragraphs):
        nodes.append({"id": str(i), "parent_id": "root", "title": f"Part {i}", "level": 2,
                      "target_type": "char_range", "target_start": offset, "target_end": offset + len(paragraph)})
        offset += len(paragraph) + 2
    section = {"source_id": "s", "media_id": 1, "node_id": "root", "navigation_version": "v", "start_char": 0, "end_char": len(source)}
    contexts = module().build_section_attribution_contexts(section, {"content": source, "nodes": nodes, "page_map": [], "navigation_version": "v"})
    assert any(context.get("omitted") for context in contexts)
    assert len(module().render_attribution_context(contexts)) <= 16_000


@pytest.mark.parametrize("field,value", [("navigation_version", "changed"), ("start_char", 1), ("node_id", "missing")])
def test_stale_selection_never_creates_verified_context(field, value):
    section, snapshot = fixture()
    section[field] = value
    contexts = module().build_section_attribution_contexts(section, snapshot)
    assert not any(context["type"] == "heading" for context in contexts)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=3000))
def test_heading_excerpt_is_bounded_exact_source_with_original_offsets(suffix):
    source = "# Heading\n\n**An Author**\n\n" + suffix
    section = {"source_id": "source", "media_id": 1, "node_id": "n", "navigation_version": "v",
               "start_char": 0, "end_char": len(source)}
    snapshot = {"content": source, "navigation_version": "v", "page_map": [], "nodes": [
        {"id": "n", "title": "Heading", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)}]}
    contexts = module().build_section_attribution_contexts(section, snapshot)
    for context in contexts:
        assert context["heading_quote"] == source[context["heading_start_char"]:context["heading_end_char"]]
        assert section["start_char"] <= context["heading_start_char"] <= context["heading_end_char"] <= section["end_char"]
        assert len(context["heading_quote"]) <= 800
    assert len(module().render_attribution_context(contexts)) <= 16_000
