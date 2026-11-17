"""Saved claim IDs must remain bound to exact footnotes through compression."""

import copy
import json
import re

import pytest

from tldw_Server_API.app.core.Research_Workspace import section_attribution as attribution
from tldw_Server_API.app.core.Research_Workspace import section_summaries as summaries
from tldw_Server_API.app.core.Research_Workspace.section_verification import section_evidence_map
from tldw_Server_API.tests.Research_Workspace.test_section_attribution import footnote_fixture

pytestmark = pytest.mark.unit


@pytest.fixture
def linked_section():
    section, snapshot = footnote_fixture()
    source = snapshot["content"]
    start, end = source.index("Transfer details."), source.index("56. Final")
    section["batches"] = [{"evidence": [{"id": "E2.3.6", "start_char": start, "end_char": end, "quote": source[start:end]}]}]
    section["attribution_contexts"] = attribution.build_section_attribution_contexts(section, snapshot)
    return section, snapshot


def projected(prompt):
    line = next(line for line in prompt.splitlines() if line.startswith('[{"type":'))
    return json.loads(line)


def bindings(prompt):
    return {key: value for context in projected(prompt) for key, value in context.get("evidence_ranges", {}).items()}


def test_label_split_and_multiple_notes_bind_saved_id_without_personal_author(linked_section):
    section, snapshot = linked_section
    evidence = section_evidence_map(section, snapshot)
    rendered = attribution.render_attribution_context(section["attribution_contexts"], evidence_by_id=evidence)
    assert bindings(rendered) == {"E2.3.6": [evidence["E2.3.6"]["start_char"], evidence["E2.3.6"]["end_char"]]}
    notes = next(context["footnotes"] for context in projected(rendered) if context["type"] == "reference")
    assert {note["label_quote"] for note in notes} >= {"53. ", "54. ", "[55. "}
    assert "not personal note authorship" in rendered
    assert attribution.contextual_author("Robert Bowes", evidence["E2.3.6"]["start_char"], evidence["E2.3.6"]["end_char"], evidence["E2.3.6"]["text"], section["attribution_contexts"]) == ""


@pytest.mark.parametrize("tamper", ["source_id", "media_id", "navigation_version", "range", "text", "unknown_id", "invalid_id_type", "invalid_item"])
def test_unvalidated_binding_does_not_supply_authority(linked_section, tamper):
    section, snapshot = linked_section
    evidence = section_evidence_map(section, snapshot)
    item = evidence["E2.3.6"]
    if tamper in {"source_id", "media_id", "navigation_version"}:
        item[tamper] = "wrong"
    elif tamper == "range":
        item["start_char"] = True
    elif tamper == "text":
        item.pop("text")
    elif tamper == "invalid_id_type":
        evidence[None] = evidence.pop("E2.3.6")
    elif tamper == "invalid_item":
        evidence["E2.3.6"] = None
    else:
        evidence["unregistered"] = evidence.pop("E2.3.6")
    assert not bindings(attribution.render_attribution_context(section["attribution_contexts"], evidence_by_id=evidence))


def test_ambiguous_note_does_not_bind_saved_id():
    section, snapshot = footnote_fixture(first_marker="[53]")
    source = snapshot["content"]
    start, end = source.index("Transfer details."), source.index("54. David")
    section["batches"] = [{"evidence": [{"id": "E2.3.6", "start_char": start, "end_char": end, "quote": source[start:end]}]}]
    section["attribution_contexts"] = attribution.build_section_attribution_contexts(section, snapshot)
    assert not bindings(attribution.render_attribution_context(section["attribution_contexts"], evidence_by_id=section_evidence_map(section, snapshot)))


def test_duplicate_or_wrong_quote_saved_ids_are_excluded_before_synthesis(linked_section):
    section, snapshot = linked_section
    duplicate = copy.deepcopy(section["batches"][0]["evidence"][0])
    section["batches"][0]["evidence"].extend([duplicate, {**duplicate, "id": "E2.3.7", "quote": "Not the source."}])
    assert summaries._synthesis_evidence(section, section_evidence_map(section, snapshot)) == {}


@pytest.mark.asyncio
async def test_bindings_survive_reductions_and_final_only_for_current_registered_ids(linked_section):
    section, snapshot = linked_section
    evidence = section_evidence_map(section, snapshot)
    evidence["E2.2.3"] = copy.deepcopy(evidence["E2.3.6"])
    parts = ["A reference claim [E2.2.3]. " + "Detail. " * 1800,
             "Transfer claim [E2.3.6]. Source mentions E9.9.9. " + "Detail. " * 1800]
    calls = []

    async def complete(prompt, **kwargs):
        supplied = set(re.findall(r"E\d+\.\d+\.\d+", " ".join(kwargs["supplied_parts"]))) & evidence.keys()
        assert set(bindings(prompt)) == supplied
        calls.append(kwargs["call_context"])
        return "The note links to the Bowes contribution. " + " ".join(f"[{key}]" for key in sorted(supplied))

    result = await summaries._synthesize_sections(parts, "Summarize the chapter.", complete=complete,
        report_progress=lambda *args: None, attribution_contexts=section["attribution_contexts"], attribution_evidence=evidence)
    assert [(call["reduction_round"], call["reduction_group"]) for call in calls] == [(1, 1), (1, 2), (0, 0)]
    assert "[E2.3.6]" in result and "E9.9.9" not in result


def test_binding_overflow_omits_all_note_authority_but_preserves_headings(linked_section):
    section, snapshot = linked_section
    original = section_evidence_map(section, snapshot)["E2.3.6"]
    evidence = {f"E2.3.{number}": original for number in range(1, 1001)}
    rendered = attribution.render_attribution_context(section["attribution_contexts"], evidence_by_id=evidence)
    assert len(rendered) <= 16_000
    assert not bindings(rendered)
    reference = next(item for item in projected(rendered) if item["type"] == "reference")
    assert reference["footnotes_omitted"] and not reference.get("footnotes")
    assert "Robert Bowes" in rendered and "The authors alone are responsible." in rendered


def test_body_only_render_is_byte_identical_with_empty_or_irrelevant_bindings(linked_section):
    section, snapshot = linked_section
    start = snapshot["content"].index("Transfer functions")
    before = attribution.render_attribution_context(section["attribution_contexts"], start, start + 18)
    assert attribution.render_attribution_context(section["attribution_contexts"], start, start + 18, evidence_by_id={}) == before
    assert attribution.render_attribution_context(section["attribution_contexts"], start, start + 18, evidence_by_id=section_evidence_map(section, snapshot)) == before


def test_body_only_overflow_keeps_existing_omission_bytes(linked_section):
    section, _ = linked_section
    contexts = section["attribution_contexts"][:1]
    contexts[0]["heading_quote"] = "body" * 8000
    assert attribution.render_attribution_context(contexts, evidence_by_id={}) == attribution.render_attribution_context(contexts)
