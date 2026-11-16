"""Strict Claims findings must survive the existing section Job lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod
from tldw_Server_API.tests.Research_Workspace.test_section_summary_jobs import args, request

pytestmark = pytest.mark.unit


@pytest.fixture
def generation(monkeypatch):
    text = "The chapter proposes buying at least 64 Next Generation Interceptors."
    snapshot = {
        "content": text, "navigation_version": "v1", "page_map": [],
        "nodes": [{"id": "dsi:1", "title": "Defense", "level": 1,
                   "target_type": "char_range", "target_start": 0, "target_end": len(text)}],
    }

    async def load(*args, **kwargs):
        return snapshot

    async def chat(**kwargs):
        content = (
            json.dumps({"summary": "A purchase is proposed.", "evidence": [{
                "claim": "Buy at least 64 Next Generation Interceptors.", "source_span_ids": ["P1"],
            }]})
            if "SOURCE:" in kwargs["messages_payload"][0]["content"]
            else "Maintain 64 interceptors [E1.1.1]."
        )
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(mod, "load_navigation_sections", load)
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    return args(request().settings), text


def report(text, *, status="unsupported"):
    return {
        "version": 1, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "verdict": "needs_revision", "provider": "openai", "model": "test-model",
        "coverage": {"total": 1, "evaluated": 1, "supported": 0, "uncovered": 0, "complete": True},
        "findings": [{"id": "C1", "text": text, "start": 0, "end": len(text),
                      "evidence_ids": ["E1.1.1"], "status": status,
                      "reason": "Buying at least 64 is not maintaining 64."}],
        "evidence": {}, "limits": {}, "context": {},
    }


@pytest.mark.asyncio
async def test_job_checks_extracted_claims_chapter_and_overview_and_exports_findings(generation, monkeypatch):
    options, source = generation
    calls = []

    async def verify(text, evidence_by_id, **kwargs):
        calls.append((text, copy.deepcopy(evidence_by_id)))
        return report(text)

    monkeypatch.setattr(mod, "verify_section_text", verify, raising=False)
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    section = analysis["sections"][0]
    assert section.get("claim_verification", {}).get("verdict") == "needs_revision"
    assert section["batches"][0]["claim_verification"]["coverage"]["complete"] is True
    assert analysis["synthesis_verification"]["verdict"] == "needs_revision"
    assert row["status"] == "complete"  # Generation completion is not semantic acceptance.
    assert len(calls) == 3
    assert all(call[1]["E1.1.1"]["text"] == source for call in calls)
    assert all(call[1]["E1.1.1"]["source_id"] == "source" for call in calls)
    assert "Buying at least 64 is not maintaining 64." in row["content"]
    assert "needs review" in row["content"].lower()


@pytest.mark.asyncio
async def test_verifier_failure_retains_generated_draft_as_unverified(generation, monkeypatch):
    options, _ = generation

    async def unavailable(*args, **kwargs):
        raise TimeoutError("provider details must not appear in findings")

    monkeypatch.setattr(mod, "verify_section_text", unavailable, raising=False)
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert analysis.get("synthesis_verification", {}).get("verdict") == "unverified"
    assert analysis["sections"][0]["status"] == "completed"
    assert "Maintain 64" in row["content"]
    assert "provider details" not in row["content"]


@pytest.mark.asyncio
async def test_retry_incomplete_checks_reuses_all_generated_text(generation, monkeypatch):
    options, _ = generation

    async def unavailable(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(mod, "verify_section_text", unavailable)
    await mod.process_section_summary(**options)

    async def no_generation(**kwargs):
        pytest.fail("Retrying checks must not generate another answer.")

    async def checked(text, evidence_by_id, **kwargs):
        return report(text)

    monkeypatch.setattr(mod, "perform_chat_api_call_async", no_generation)
    monkeypatch.setattr(mod, "verify_section_text", checked)
    await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert analysis["synthesis_verification"]["verdict"] == "needs_revision"


@pytest.mark.asyncio
async def test_resume_checks_old_completed_chapters_without_regenerating_them(generation, monkeypatch):
    options, _ = generation
    seen = []

    async def verify(text, evidence_by_id, **kwargs):
        seen.append(text)
        return report(text)

    monkeypatch.setattr(mod, "verify_section_text", verify, raising=False)
    await mod.process_section_summary(**options)
    saved = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    saved["sections"][0].pop("claim_verification", None)
    saved["sections"][0]["batches"][0].pop("claim_verification", None)
    seen.clear()
    original_chat = mod.perform_chat_api_call_async

    async def overview_only(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        assert "SOURCE:" not in prompt
        assert not prompt.startswith("Summarize all portions")
        return await original_chat(**kwargs)

    monkeypatch.setattr(mod, "perform_chat_api_call_async", overview_only)
    await mod.process_section_summary(**options)
    assert len(seen) == 2  # The identical overview retains its completed receipt.
    assert saved["sections"][0]["content"] == "Maintain 64 interceptors [E1.1.1]."


def test_changed_source_or_attribution_invalidates_completed_verification():
    from tldw_Server_API.app.core.Research_Workspace.section_verification import (
        evidence_digest,
        verification_is_current,
    )

    text = "A proposal [E1]."
    evidence = {"E1": {"text": "A proposal", "attribution_context": "A wrote the chapter."}}
    receipt = {**report(text), "evidence_sha256": evidence_digest(evidence)}
    assert verification_is_current(receipt, text, evidence)
    evidence["E1"]["attribution_context"] = "The chapter disclaims individual responsibility."
    assert not verification_is_current(receipt, text, evidence)


def test_batch_verification_retains_interpretations_and_shared_claim_passages():
    from tldw_Server_API.app.core.Research_Workspace.section_verification import batch_claim_text

    item = {"claim": "Buy at least 64.", "author": "Author A", "interpretation": "A purchase is proposed",
            "limitations": "Not enacted"}
    text = batch_claim_text({"evidence": [{**item, "id": "E1"}, {**item, "id": "E2"}]})
    assert text == "Buy at least 64. Attributed to: Author A. Interpretation: A purchase is proposed. Limitations: Not enacted. [E1] [E2]"


def test_stale_saved_quote_cannot_authorize_a_citation():
    from tldw_Server_API.app.core.Research_Workspace.section_verification import section_evidence_map

    evidence = section_evidence_map({"batches": [{"evidence": [{"id": "E1", "start_char": 0, "end_char": 3, "quote": "old"}]}]},
                                    {"content": "new"})
    assert evidence == {}


@pytest.mark.asyncio
async def test_section_adapter_uses_owned_credential_and_strict_raw_completion():
    from tldw_Server_API.app.core.Research_Workspace.section_verification import verify_section_text

    credential = object()
    source = "Context. " * 100 + "The chapter proposes buying at least 64 interceptors."
    settings = request().settings.model_copy(update={"provider": "openai", "model": "test-model"})

    async def chat(**kwargs):
        assert kwargs["_provider_call_credentials"] is credential
        assert kwargs["model"] == "test-model"
        assert source in kwargs["messages_payload"][0]["content"]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "label": "refuted", "confidence": 0.98, "rationale": "Buying at least 64 differs from maintaining 64.",
        })}}]}

    result = await verify_section_text("Maintain 64 interceptors [E1.1.1].", {"E1.1.1": {"text": source}},
                                       settings=settings, provider_credentials=credential, chat_fn=chat)
    assert result["findings"][0]["status"] == "contradicted"
    assert result["coverage"]["complete"] is True
