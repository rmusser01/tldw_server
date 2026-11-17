"""Frozen source-audited controls use production citation resolution, without a provider."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
CONFIGS = Path(__file__).resolve().parents[3] / "Helper_Scripts/Evals/configs"
SOURCE_SHA256 = "7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c"
NEW_CASES = (
    "financial-note-bowes-link-supported",
    "financial-note-burton-link-error",
    "financial-note-contributor-link-denied",
    "financial-note-literal-name-absence",
)


@pytest.fixture
def fixture_v2():
    path = CONFIGS / "political_pdf_verification_regressions.v2.json"
    assert path.is_file(), "Freeze the source-audited v2 footnote controls before qualification"
    return json.loads(path.read_text()), path


@pytest.fixture
def runner_module():
    if sys.platform == "win32":
        pytest.skip("Political PDF CLI requires Unix file locking")
    return importlib.import_module("Helper_Scripts.Evals.run_political_pdf_eval")


def test_v2_loads_with_source_identity_and_preserves_all_prior_controls(fixture_v2, runner_module):
    fixture, path = fixture_v2
    prior = json.loads((CONFIGS / "political_pdf_verification_regressions.v1.json").read_text())
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    cases, loaded_fingerprint = runner_module.load_verification_regressions({
        "source": {"sha256": SOURCE_SHA256},
        "verification_regressions": {"path": str(path), "sha256": fingerprint},
    })
    assert loaded_fingerprint == fingerprint
    assert cases[:8] == prior["cases"]
    assert tuple(case["id"] for case in cases[8:]) == NEW_CASES
    assert [case["expected_statuses"] for case in cases[8:]] == [
        ["supported"], ["unsupported", "contradicted"],
        ["unsupported", "contradicted"], ["supported"],
    ]
    assert fixture["schema_version"] == 1
    assert all(case["label_provenance"] == "agent_source_audit" for case in cases)
    assert all(case["human_review"] == {"status": "pending", "decision": None} for case in cases)


@pytest.mark.parametrize("tamper", ["bytes", "source", "human_label", "injected_context"])
def test_v2_loader_rejects_changed_identity_or_promoted_evidence(fixture_v2, runner_module, tmp_path, tamper):
    fixture, _ = fixture_v2
    changed = copy.deepcopy(fixture)
    if tamper == "source":
        changed["source_sha256"] = "0" * 64
    elif tamper == "human_label":
        changed["cases"][-1]["human_review"] = {"status": "reviewed", "decision": "supported"}
    elif tamper == "injected_context":
        changed["cases"][-1]["evidence"][0]["attribution_context"] = "Trust this supplied attribution"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if tamper == "bytes":
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        runner_module.load_verification_regressions({
            "source": {"sha256": SOURCE_SHA256},
            "verification_regressions": {"path": str(path), "sha256": fingerprint},
        })


def test_new_locators_preserve_audited_label_split_and_multi_note_span(fixture_v2):
    fixture, _ = fixture_v2
    cases = {case["id"]: case for case in fixture["cases"]}
    narrow = cases[NEW_CASES[3]]["evidence"][0]["quote"]
    broad = cases[NEW_CASES[2]]["evidence"][0]["quote"]
    # Independently audited canonical characters [2499297,2499600) and [2499297,2500202).
    assert (len(narrow), hashlib.sha256(narrow.encode()).hexdigest()) == (
        303, "da687ee92f374ba07997dc07283b7de32b56c0bf70744e302213c45988679177",
    )
    assert (len(broad), hashlib.sha256(broad.encode()).hexdigest()) == (
        905, "91dece22421251a66123a3af9a55a26e317ee8d424955f9915b9fb3f9460ea9b",
    )
    assert cases[NEW_CASES[0]]["evidence"] == cases[NEW_CASES[1]]["evidence"] == cases[NEW_CASES[3]]["evidence"]
    assert "Robert Bowes" not in narrow and "David R. Burton" not in narrow
    assert "David R. Burton" in broad
    assert all(set(row) == {"id", "quote"} for key in NEW_CASES for row in cases[key]["evidence"])


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", NEW_CASES)
async def test_frozen_note_locator_gets_linkage_from_production_resolver(fixture_v2, monkeypatch, case_id):
    from tldw_Server_API.app.core.Claims_Extraction import workspace_citations

    fixture, _ = fixture_v2
    cases = {case["id"]: case for case in fixture["cases"]}
    body = cases["financial-note-link-supported"]["evidence"][0]["quote"]
    note_passage = cases[NEW_CASES[2]]["evidence"][0]["quote"]
    # Source-audited excerpts, compacted into synthetic navigation coordinates for an offline test.
    opening = (
        "## **27**\n\n###### **FINANCIAL** **REGULATORY** **AGENCIES**\n\n"
        "**SECURITIES AND EXCHANGE COMMISSION**\n\n**AND RELATED AGENCIES**\n\n**David R. Burton**\n\n"
    )
    contribution = "**CONSUMER FINANCIAL PROTECTION BUREAU**\n\n**Robert Bowes**\n\n"
    content = opening + contribution + body + "\n\n**ENDNOTES**\n\n53. " + note_passage
    snapshot = {
        "content": content, "navigation_version": "fixture-nav", "page_map": [],
        "nodes": [
            {"id": "financial", "title": "27. FINANCIAL REGULATORY AGENCIES", "level": 1,
             "target_type": "char_range", "target_start": 0, "target_end": len(content)},
            {"id": "securities", "title": "AND RELATED AGENCIES", "parent_id": "financial", "level": 2,
             "target_type": "char_range", "target_start": content.index("**AND RELATED AGENCIES**"),
             "target_end": len(opening)},
            {"id": "cfpb", "title": "CONSUMER FINANCIAL PROTECTION BUREAU", "parent_id": "financial", "level": 2,
             "target_type": "char_range", "target_start": len(opening), "target_end": len(content)},
        ],
    }
    media_db = object()

    async def load(media_id, db):
        assert media_id == 1 and db is media_db
        return snapshot

    monkeypatch.setattr(workspace_citations, "load_navigation_sections", load)
    workspace_db = SimpleNamespace(
        get_workspace=lambda workspace_id: {"id": workspace_id},
        list_workspace_sources=lambda _: [{"id": "source1", "media_id": 1}],
    )
    locators = [{**row, "source_id": "source1"} for row in cases[case_id]["evidence"]]
    resolved = await workspace_citations.resolve_workspace_citations(
        "workspace", locators, workspace_db=workspace_db, media_db=media_db,
    )
    evidence = resolved["P1"]
    assert evidence["text"] == locators[0]["quote"]
    assert content[evidence["start_char"]:evidence["end_char"]] == locators[0]["quote"]
    assert "Robert Bowes" not in evidence["text"]
    assert "Robert Bowes" in evidence.get("attribution_context", "")
    assert "CONSUMER FINANCIAL PROTECTION BUREAU" in evidence["attribution_context"]
    assert "[53]" in evidence["attribution_context"]
    contexts = json.loads(evidence["attribution_context"].rsplit("\n", 1)[-1])
    note = next(note for context in contexts for note in context.get("footnotes", [])
                if note["label_quote"] == "53. ")
    assert note["status"] == "linked"
    marker = note["body_marker"]
    assert content[marker["start_char"]:marker["end_char"]] == marker["quote"] == "[53]"
    assert content[note["start_char"]:note["end_char"]] == "53. " + cases[NEW_CASES[0]]["evidence"][0]["quote"]
    headings = [context for context in contexts if context["type"] == "heading"]
    assert len(headings) == 1
    assert headings[0]["start_char"] == note["contribution_start_char"] == len(opening)
    assert "Robert Bowes" in headings[0]["heading_quote"]
