"""Strict support checks bind complete generated units only to their cited evidence."""

import asyncio
import hashlib
import json

import pytest

from tldw_Server_API.app.core.Claims_Extraction.citation_support import verify_cited_text


def response(label="supported", confidence=0.95, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps({
        "label": label, "confidence": confidence, "rationale": "Checked the whole cited unit.",
    })}}]}


@pytest.mark.asyncio
async def test_only_cited_complete_passages_reach_semantic_judge():
    seen = []
    evidence = {"E1.1.1": {"text": "A" * 600 + " The late qualifier applies."},
                "E2.1.1": {"text": "UNRELATED supporting material."}}

    async def analyze(provider, claim, prompt, key, system, temperature, **kwargs):
        seen.append((provider, claim, prompt, system, kwargs))
        return response()

    text = "The proposal applies only after review. [E1.1.1]"
    report = await verify_cited_text(text, evidence, provider="openai", model="test-judge", analyze_fn=analyze)
    assert report["verdict"] == "supported"
    assert report["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert report["coverage"] == {"total": 1, "evaluated": 1, "supported": 1, "uncovered": 0, "complete": True}
    assert evidence["E1.1.1"]["text"] in seen[0][2]
    assert "UNRELATED" not in seen[0][2]
    assert seen[0][0] == "openai" and seen[0][4]["model_override"] == "test-judge"
    assert report["findings"][0]["evidence_ids"] == ["E1.1.1"]


@pytest.mark.asyncio
async def test_uncited_paragraph_cannot_borrow_following_claim_citations():
    calls = []

    def analyze(*args, **kwargs):
        calls.append(args[1])
        return response()

    text = "An uncited summary invents a policy.\n\nThe actual proposal. [P1]"
    report = await verify_cited_text(text, {"P1": {"text": "The actual proposal."}},
        provider="openai", model="test-judge", analyze_fn=analyze)
    assert [row["status"] for row in report["findings"]] == ["uncited", "supported"]
    assert len(calls) == 1 and "invents" not in calls[0]
    assert report["verdict"] == "needs_revision" and not report["coverage"]["complete"]
    assert all(text[row["start"]:row["end"]] == row["text"] for row in report["findings"])


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [response("nei", 0.5), response(finish="length"),
    {"choices": [{"message": {"content": "not JSON"}, "finish_reason": "stop"}]},
    response(confidence=float("nan")), '{"label":"supported","confidence":1}'])
async def test_no_lexical_or_malformed_judgment_can_mark_supported(raw):
    async def analyze(*args, **kwargs):
        return raw

    report = await verify_cited_text('The identical quoted text. [1]',
        {"1": {"text": 'The identical quoted text.', "score": 1.0}},
        provider="openai", model="test-judge", analyze_fn=analyze)
    assert report["verdict"] != "supported"
    assert report["coverage"]["supported"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("text,evidence", [
    ("Claim. [P2]", {"P1": {"text": "Claim."}}),
    ("Claim. [P1]", {"P1": {"text": "x" * 16001}}),
    ("Claim. [P1-P3]", {"P1": {"text": "Claim."}, "P3": {"text": "Claim."}}),
])
async def test_missing_or_oversized_evidence_never_dispatches(text, evidence):
    async def analyze(*args, **kwargs):
        raise AssertionError("Unresolvable evidence must never reach a provider")

    report = await verify_cited_text(text, evidence, provider="openai", model="test-judge", analyze_fn=analyze)
    assert report["coverage"]["evaluated"] == 0
    assert report["findings"][0]["status"] == "not_checked"


@pytest.mark.asyncio
async def test_cancellation_propagates_without_claiming_a_receipt():
    async def analyze(*args, **kwargs):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await verify_cited_text("Claim. [P1]", {"P1": {"text": "Claim."}},
            provider="openai", model="test-judge", analyze_fn=analyze)


@pytest.mark.asyncio
async def test_wrapped_citations_form_one_claim_but_citation_only_block_is_unchecked():
    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args[1])
        return response()

    text = "Jointly supported proposition [P1]\n[P2]\n\n[P1]"
    report = await verify_cited_text(text, {"P1": {"text": "First."}, "P2": {"text": "Second."}},
        provider="openai", model="wrapped-test", analyze_fn=analyze)
    assert len(seen) == 1 and seen[0] == "Jointly supported proposition [P1]\n[P2]"
    assert report["findings"][0]["evidence_ids"] == ["P1", "P2"]
    assert report["findings"][1]["status"] == "not_checked"
    assert report["coverage"]["uncovered"] == 1


@pytest.mark.asyncio
async def test_independent_bullets_and_uncited_tail_keep_their_own_binding():
    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args[1])
        return response()

    text = "- An unsupported bullet\n- A cited bullet [Source 1]. An uncited tail."
    report = await verify_cited_text(text, {"1": {"text": "A cited bullet"}},
        provider="openai", model="bullet-test", analyze_fn=analyze)
    assert [row["status"] for row in report["findings"]] == ["uncited", "supported", "uncited"]
    assert seen == ["- A cited bullet [Source 1]"]
    assert all(row["text"] == text[row["start"]:row["end"]] for row in report["findings"])


@pytest.mark.asyncio
async def test_identical_attribution_context_is_deduplicated_without_dropping_passages():
    note = "The authors alone are responsible. " * 250
    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args[2])
        return response()

    evidence = {"P1": {"text": "First policy detail", "attribution_context": note},
        "P2": {"text": "Second policy detail", "attribution_context": note},
        "P3": {"text": "Uncited sibling", "attribution_context": "SIBLING CAVEAT"}}
    report = await verify_cited_text("Both details. [P1, P2]", evidence,
        provider="openai", model="context-test", analyze_fn=analyze)
    assert report["verdict"] == "supported"
    assert seen[0].count(note) == 1
    assert "First policy detail" in seen[0] and "Second policy detail" in seen[0]
    assert "SIBLING CAVEAT" not in seen[0]
    assert report["evidence"]["P2"]["attribution_context_sha256"] == hashlib.sha256(note.encode()).hexdigest()


@pytest.mark.asyncio
async def test_concurrency_and_caps_keep_unchecked_units_visible(monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import citation_support

    monkeypatch.setitem(citation_support.LIMITS, "max_units", 5)
    active = 0
    peak = 0
    entered = asyncio.Event()

    async def analyze(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 4:
            entered.set()
        await entered.wait()
        await asyncio.sleep(0)
        active -= 1
        return response()

    report = await verify_cited_text("\n\n".join(f"Claim {i}. [P1]" for i in range(7)),
        {"P1": {"text": "All claims"}}, provider="openai", model="capacity-test", analyze_fn=analyze)
    assert peak == 4
    assert [row["status"] for row in report["findings"]] == ["supported"] * 5 + ["not_checked"] * 2
    assert report["coverage"] == {"total": 7, "evaluated": 5, "supported": 5, "uncovered": 2, "complete": False}
    assert report["verdict"] == "unverified"


@pytest.mark.asyncio
async def test_provider_deadline_and_exception_are_unverified_without_secret_text(monkeypatch, caplog):
    from tldw_Server_API.app.core.Claims_Extraction import citation_support

    monkeypatch.setitem(citation_support.LIMITS, "timeout_seconds", 0.02)

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    async def failed(*args, **kwargs):
        raise RuntimeError("PRIVATE_CREDENTIAL_SENTINEL")

    for analyze in (stalled, failed):
        report = await verify_cited_text("A claim. [P1]", {"P1": {"text": "A claim"}},
            provider="openai", model="failure-test", analyze_fn=analyze)
        assert report["verdict"] == "unverified" and report["findings"][0]["status"] == "error"
        assert "PRIVATE_CREDENTIAL_SENTINEL" not in json.dumps(report) + caplog.text


@pytest.mark.asyncio
async def test_contradiction_is_distinct_from_insufficient_support():
    async def analyze(*args, **kwargs):
        return response("refuted")

    report = await verify_cited_text("The opposite policy. [P1]", {"P1": {"text": "The policy"}},
        provider="openai", model="refuted-test", analyze_fn=analyze)
    assert report["verdict"] == "needs_revision"
    assert report["findings"][0]["status"] == "contradicted"
    assert report["coverage"]["complete"] is True


@pytest.mark.asyncio
async def test_total_deadline_preserves_completed_checks_and_cancels_remaining(monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import citation_support

    monkeypatch.setitem(citation_support.LIMITS, "total_timeout_seconds", 0.03)
    cancelled = []

    async def analyze(*args, **kwargs):
        if "Fast" in args[1]:
            return response()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(args[1])

    report = await verify_cited_text("Fast. [P1]\n\nSlow. [P1]", {"P1": {"text": "Exact source"}},
        provider="openai", model="total-deadline", analyze_fn=analyze)
    assert [row["status"] for row in report["findings"]] == ["supported", "not_checked"]
    assert report["findings"][1]["reason"] == "verification_deadline"
    assert report["coverage"]["uncovered"] == 1 and report["verdict"] == "unverified"
    assert cancelled == ["Slow. [P1]"]


def test_splitter_preserves_every_nonwhitespace_character_and_offsets():
    from hypothesis import given
    from hypothesis import strategies as st

    from tldw_Server_API.app.core.Claims_Extraction.citation_support import _units

    @given(st.lists(st.text(alphabet="abc XYZé🌙", min_size=1, max_size=30).filter(lambda value: value.strip()), min_size=1, max_size=8))
    def check(parts):
        text = "\n\n".join(part + " [P1]" for part in parts)
        units = _units(text)
        covered = set()
        for unit in units:
            assert unit["text"] == text[unit["start"]:unit["end"]]
            assert not covered.intersection(range(unit["start"], unit["end"]))
            covered.update(range(unit["start"], unit["end"]))
        assert all(index in covered for index, char in enumerate(text) if not char.isspace())

    check()


@pytest.mark.asyncio
async def test_late_sync_adapter_awaitable_is_closed_after_total_deadline(monkeypatch):
    import inspect
    import threading

    from tldw_Server_API.app.core.Claims_Extraction import citation_support

    monkeypatch.setitem(citation_support.LIMITS, "total_timeout_seconds", 0.03)
    release = threading.Event()
    returned = []

    async def late_response():
        raise AssertionError("An abandoned provider coroutine must never start")

    def analyze(*args, **kwargs):
        release.wait(timeout=1)
        result = late_response()
        returned.append(result)
        return result

    try:
        report = await verify_cited_text("Claim [P1]", {"P1": {"text": "Exact source"}},
            provider="openai", model="late-awaitable", analyze_fn=analyze)
        release.set()
        for _ in range(100):
            if returned:
                break
            await asyncio.sleep(0.001)
        assert report["verdict"] == "unverified"
        assert len(returned) == 1 and inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED
    finally:
        release.set()
        for pending in returned:
            pending.close()
