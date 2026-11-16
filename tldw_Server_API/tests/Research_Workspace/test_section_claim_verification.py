"""Strict Claims findings must survive the existing section Job lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
    return args(request().settings.model_copy(update={"provider": "openai", "model": "test-model"})), text


def report(text, *, status="unsupported", reason="Buying at least 64 is not maintaining 64.", evidence=None):
    evaluated = status in {"supported", "unsupported", "contradicted"}
    evidence = evidence or {"E1.1.1": {"text": "The chapter proposes buying at least 64 Next Generation Interceptors."}}
    return {
        "version": 1, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "verdict": "needs_revision" if status in {"unsupported", "contradicted", "uncited"} else "supported" if evaluated else "unverified",
        "provider": "openai", "model": "test-model",
        "coverage": {"total": 1, "evaluated": int(evaluated), "supported": int(status == "supported"),
                     "uncovered": int(not evaluated), "complete": evaluated},
        "findings": [{"id": "C1", "text": text, "start": 0, "end": len(text),
                      "evidence_ids": list(evidence) if evaluated else [], "status": status, "reason": reason}],
        "evidence": {key: {"text_sha256": hashlib.sha256(value["text"].encode()).hexdigest()} for key, value in evidence.items()},
        "limits": {}, "context": {},
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
    receipt = {**report(text, evidence=evidence), "evidence_sha256": evidence_digest(evidence)}
    assert verification_is_current(receipt, text, evidence, provider="openai", model="test-model")
    evidence["E1"]["attribution_context"] = "The chapter disclaims individual responsibility."
    assert not verification_is_current(receipt, text, evidence, provider="openai", model="test-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("status,reason", [("supported", "supported"), ("unsupported", "not entailed"),
    ("contradicted", "contradicted"), ("uncited", "missing_citation"),
    *(('not_checked', reason) for reason in ("text_or_unit_limit", "cited_evidence_limit", "report_size_limit"))])
async def test_resume_retains_terminal_receipts_without_rejudging(generation, monkeypatch, status, reason):
    options, _ = generation
    if status == "uncited":
        original_chat = mod.perform_chat_api_call_async

        async def with_heading(**kwargs):
            response = await original_chat(**kwargs)
            message = response["choices"][0]["message"]
            if "SOURCE:" in kwargs["messages_payload"][0]["content"]:
                parsed = json.loads(message["content"])
                parsed["evidence"][0]["claim"] = "# Extracted claim\n\n" + parsed["evidence"][0]["claim"]
                message["content"] = json.dumps(parsed)
            else:
                message["content"] = "# Policy overview\n\n" + message["content"]
            return response

        monkeypatch.setattr(mod, "perform_chat_api_call_async", with_heading)

    async def verify(text, evidence_by_id, **kwargs):
        if status == "uncited":
            from tldw_Server_API.app.core.Claims_Extraction.citation_support import verify_cited_text

            async def judge(*args, **kwargs):
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                    "label": "refuted", "confidence": 0.98, "rationale": "Buying differs from maintaining.",
                })}}]}

            return await verify_cited_text(text, evidence_by_id, provider="openai", model="test-model", analyze_fn=judge)
        return report(text, status=status, reason=reason, evidence=evidence_by_id)

    monkeypatch.setattr(mod, "verify_section_text", verify)
    await mod.process_section_summary(**options)
    before = copy.deepcopy(options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"])
    if status == "uncited":
        assert before["synthesis_verification"]["coverage"]["uncovered"] == 1
        assert before["synthesis_verification"]["findings"][0]["status"] == "uncited"

    async def no_provider_calls(*args, **kwargs):
        pytest.fail("Unchanged terminal receipts and drafts must not call a provider again.")

    monkeypatch.setattr(mod, "verify_section_text", no_provider_calls)
    monkeypatch.setattr(mod, "perform_chat_api_call_async", no_provider_calls)
    await mod.process_section_summary(**options)
    after = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert after == before  # Includes the exact negative/uncovered verdicts and coverage.


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["text_sha256", "evidence_sha256", "provider", "model"])
async def test_resume_rechecks_receipts_with_changed_bindings(generation, monkeypatch, changed):
    options, _ = generation
    calls = []

    async def verify(text, evidence_by_id, **kwargs):
        calls.append(text)
        return report(text, evidence=evidence_by_id)

    monkeypatch.setattr(mod, "verify_section_text", verify)
    await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    section = analysis["sections"][0]
    for receipt in (section["batches"][0]["claim_verification"], section["claim_verification"], analysis["synthesis_verification"]):
        receipt[changed] = "0" * 64 if changed.endswith("sha256") else "old-identity"
    calls.clear()
    await mod.process_section_summary(**options)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_failed_portion_resume_does_not_recheck_retained_terminal_batch(generation, monkeypatch):
    options, _ = generation
    original_load = mod.load_navigation_sections
    original_chat = mod.perform_chat_api_call_async
    tail_failed = False
    checked_stages = []

    async def load(*args, **kwargs):
        snapshot = copy.deepcopy(await original_load(*args, **kwargs))
        snapshot["content"] = snapshot["content"].ljust(24_000) + "A continuation proposes review."
        snapshot["nodes"][0]["target_end"] = len(snapshot["content"])
        return snapshot

    async def chat(**kwargs):
        nonlocal tail_failed
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:\n" in prompt and "A continuation" in prompt.split("SOURCE:\n", 1)[1] and not tail_failed:
            tail_failed = True
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "summary": "A continuation is present.", "evidence": [],
            })}}]}
        return await original_chat(**kwargs)

    async def verify(text, evidence_by_id, **kwargs):
        checked_stages.append(kwargs["context"]["stage"])
        return report(text, status="not_checked", reason="cited_evidence_limit", evidence=evidence_by_id)

    monkeypatch.setattr(mod, "load_navigation_sections", load)
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    monkeypatch.setattr(mod, "verify_section_text", verify)
    await mod.process_section_summary(**options)
    first = copy.deepcopy(options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"])
    assert first["sections"][0]["status"] == "failed"
    checked_stages.clear()
    await mod.process_section_summary(**options)
    after = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert after["sections"][0]["status"] == "completed"
    assert "Defense, extracted claims in portion 1" not in checked_stages
    assert len(checked_stages) == 3  # Newly completed tail, chapter, overview only.
    assert after["sections"][0]["batches"][0] == first["sections"][0]["batches"][0]
    assert after["generation_failures"] == first["generation_failures"]


@pytest.mark.parametrize("change", ["text", "evidence", "provider", "model", "missing", "empty findings",
    "error", "unknown limit", "deadline", "malformed counts", "missing evidence hash", "wrong slice", "uncovered text"])
def test_terminal_receipt_requires_current_complete_bindings(change):
    from tldw_Server_API.app.core.Research_Workspace.section_verification import (
        evidence_digest,
        verification_is_current,
    )

    text = "A proposal [E1]."
    evidence = {"E1": {"text": "A proposal"}}
    receipt = {**report(text, evidence=evidence), "evidence_sha256": evidence_digest(evidence)}
    provider, model = "openai", "test-model"
    if change == "text":
        text += " Another claim."
    elif change == "evidence":
        evidence["E1"]["text"] = "A different proposal"
    elif change == "provider":
        provider = "deepseek"
    elif change == "model":
        model = "another-model"
    elif change == "missing":
        receipt = None
    elif change == "empty findings":
        receipt.update(findings=[], coverage={"total": 0, "evaluated": 0, "supported": 0, "uncovered": 0, "complete": False}, verdict="unverified")
    elif change in {"error", "unknown limit", "deadline"}:
        receipt = {**report(text, status="error" if change == "error" else "not_checked", reason=change, evidence=evidence),
                   "evidence_sha256": evidence_digest(evidence)}
    elif change == "malformed counts":
        receipt["coverage"]["evaluated"] = True
    elif change == "missing evidence hash":
        receipt["evidence"]["E1"] = {}
    elif change == "wrong slice":
        receipt["findings"][0]["text"] = "Z" * len(text)
    else:
        receipt["findings"][0].update(text=text[2:], start=2)
    assert not verification_is_current(receipt, text, evidence, provider=provider, model=model)


@given(st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=40))
def test_terminal_receipt_cannot_hide_substantive_unicode_gap(prefix):
    from tldw_Server_API.app.core.Research_Workspace.section_verification import (
        evidence_digest,
        verification_is_current,
    )

    text = prefix + " A proposal [E1]."
    evidence = {"E1": {"text": "A proposal"}}
    receipt = {**report(text, evidence=evidence), "evidence_sha256": evidence_digest(evidence)}
    start = len(prefix) + 1
    receipt["findings"][0].update(text=text[start:], start=start)
    assert not verification_is_current(receipt, text, evidence, provider="openai", model="test-model")


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
