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
    bindings = json.loads(seen[0].split("CITED PASSAGE CONTEXT BINDINGS", 1)[1].splitlines()[1])
    assert bindings == {"P1": None, "P2": None}
    assert report["evidence"]["P2"]["attribution_context_sha256"] == hashlib.sha256(note.encode()).hexdigest()


@pytest.mark.asyncio
async def test_mixed_contributor_contexts_keep_their_cited_passage_ranges():
    seen = []
    evidence = {
        "P3": {"text": "Large beneficiaries could access private financing.", "start_char": 2126649,
            "attribution_context": "THE EXPORT–IMPORT BANK SHOULD BE ABOLISHED\nVeronique de Rugy"},
        "P5": {"text": "EXIM largely substitutes for other export financing.", "start_char": 2393501,
            "attribution_context": "THE CASE FOR FREE TRADE\nKent Lassman"},
        "P12": {"text": "Uncited financing passage.", "attribution_context": "Jennifer Hazelton"},
    }
    for item in evidence.values():
        if "start_char" in item:
            item["end_char"] = item["start_char"] + len(item["text"])

    async def analyze(*args, **kwargs):
        seen.append(args[2])
        return response("nei")

    report = await verify_cited_text("De Rugy argues both financing propositions. [P3, P5]", evidence,
        provider="openai", model="binding-test", analyze_fn=analyze)
    groups = seen[0].split("CITED PASSAGE CONTEXT BINDINGS")[1:]
    assert len(groups) == 2
    for group, identifier in zip(groups, ("P3", "P5"), strict=True):
        item = evidence[identifier]
        assert json.loads(group.splitlines()[1]) == {identifier: [item["start_char"], item["end_char"]]}
        assert item["attribution_context"] in group
        assert report["evidence"][identifier]["attribution_context_sha256"] == hashlib.sha256(item["attribution_context"].encode()).hexdigest()
    assert "Jennifer Hazelton" not in seen[0]
    assert report["findings"][0]["status"] == "unsupported"


@pytest.mark.parametrize("start,end", [(None, None), (True, 2), (0, True), (-1, 0), (0, 0), (2, 1), ("0", 1), (0, 3)])
def test_context_binding_never_invents_or_coerces_invalid_source_ranges(start, end):
    from tldw_Server_API.app.core.Claims_Extraction.citation_support import _evidence_for_unit

    documents, context, error = _evidence_for_unit({"evidence_ids": ["P1"]},
        {"P1": {"text": "x", "start_char": start, "end_char": end, "attribution_context": "Author note"}})
    assert json.loads(context.splitlines()[1]) == {"P1": None}
    assert error is None and documents[0].content == "x"


@pytest.mark.asyncio
async def test_context_binding_wrapper_counts_toward_existing_evidence_budget(monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import citation_support

    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args)
        return response()

    monkeypatch.setitem(citation_support.LIMITS, "max_evidence_chars", 20)
    report = await verify_cited_text("Policy. [P1]", {"P1": {"text": "Policy", "attribution_context": "Author context"}},
        provider="openai", model="binding-test", analyze_fn=analyze)
    assert seen == []
    assert report["findings"][0]["reason"] == "cited_evidence_limit"


@pytest.mark.asyncio
@pytest.mark.parametrize("label,status", [("supported", "supported"), ("refuted", "contradicted"), ("nei", "unsupported")])
async def test_linked_note_context_guides_attribution_without_forcing_judgment(label, status):
    from tldw_Server_API.app.core.Research_Workspace.section_attribution import (
        build_section_attribution_contexts,
        render_attribution_context,
    )

    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args[2])
        return response(label)

    quote = "The reference supplies transfer details."
    source = ("# Agency\n**Ada Example**\nThe agency should change.[53]\n"
        "AUTHOR'S NOTE: These are collective views.\nENDNOTES\n53. " + quote + "\n54. Other reference.\n")
    node = {"id": "agency", "title": "Agency", "level": 1, "target_type": "char_range",
        "target_start": 0, "target_end": len(source)}
    snapshot = {"content": source, "nodes": [node], "navigation_version": "version", "page_map": []}
    section = {"node_id": "agency", "source_id": "source", "media_id": 1,
        "start_char": 0, "end_char": len(source), "navigation_version": "version"}
    contexts = build_section_attribution_contexts(section, snapshot)
    start = source.index(quote)
    context = render_attribution_context(contexts, start, start + len(quote))
    report = await verify_cited_text("This excerpt does not name an author. [P1]",
        {"P1": {"text": quote, "attribution_context": context},
            "P2": {"text": "UNRELATED POLICY", "attribution_context": "UNRELATED BYLINE"}},
        provider="openai", model="context-test", analyze_fn=analyze)
    assert report["findings"][0]["status"] == status
    assert "Ada Example" in seen[0] and "[53]" in seen[0] and "These are collective views." in seen[0]
    assert "policy-claim evidence or repair an unrelated citation):\nCITED PASSAGE CONTEXT BINDINGS" in seen[0]
    assert seen[0].endswith(context)
    assert "UNRELATED" not in seen[0]


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

    text = "\n\n".join(f"Claim {i}. [P1]" for i in range(7))
    report = await verify_cited_text(text,
        {"P1": {"text": "All claims"}}, provider="openai", model="capacity-test", analyze_fn=analyze)
    assert peak == 4
    assert [row["status"] for row in report["findings"]] == ["supported"] * 5 + ["not_checked"]
    assert report["coverage"] == {"total": 6, "evaluated": 5, "supported": 5, "uncovered": 1, "complete": False}
    tail = report["findings"][-1]
    assert tail["text"] == text[report["findings"][-2]["end"]:] and tail["evidence_ids"] == []
    assert report["verdict"] == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["x\n\n" * 20_000, " 𐐀 \n\n" * 20_000, "x\n\n" * 33_333 + "x"])
async def test_allowed_large_text_has_bounded_tasks_and_a_persistable_exact_overflow_tail(text, monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import citation_support
    from tldw_Server_API.app.core.Claims_Extraction.citation_support_types import validate_claim_verification

    created = []
    create_task = asyncio.create_task

    def capture_task(coroutine):
        task = create_task(coroutine)
        created.append(task)
        return task

    monkeypatch.setattr(citation_support.asyncio, "create_task", capture_task)
    report = await verify_cited_text(text, {}, provider="openai", model="bounded-test")
    assert len(created) <= 257 and len(report["findings"]) <= 257
    assert validate_claim_verification(report) is report
    assert report["verdict"] != "supported" and report["coverage"]["complete"] is False
    tail = report["findings"][-1]
    assert tail["status"] == "not_checked" and tail["reason"] == "text_or_unit_limit"
    assert tail["evidence_ids"] == [] and tail["end"] == len(text)
    assert tail["text"] == text[tail["start"]:] and tail["start"] == report["findings"][-2]["end"]


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum", [0, 1])
async def test_zero_or_one_unit_limit_preserves_all_remaining_text_without_more_judging(maximum, monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import citation_support

    monkeypatch.setitem(citation_support.LIMITS, "max_units", maximum)
    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args[1])
        return response()

    text = " \nFirst [P1].\n\nSecond [P1].\n\nThird [P1]. \n"
    report = await verify_cited_text(text, {"P1": {"text": "Exact source"}},
        provider="openai", model="zero-one-test", analyze_fn=analyze)
    assert len(seen) == maximum and len(report["findings"]) == maximum + 1
    tail = report["findings"][-1]
    start = report["findings"][0]["end"] if maximum else 0
    assert tail == {"id": f"claim-{maximum + 1}", "text": text[start:], "start": start,
                    "end": len(text), "evidence_ids": [], "status": "not_checked", "reason": "text_or_unit_limit"}


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", ["rationales", "provenance", "context"])
async def test_report_byte_overflow_replaces_judgments_with_one_exact_unchecked_text(overflow):
    from tldw_Server_API.app.core.Claims_Extraction.citation_support_types import validate_claim_verification
    from tldw_Server_API.app.core.Media.page_provenance import page_spans_for_range, validate_page_map

    text = "\n\n".join(f"Policy {i} [P1]" for i in range(256)) if overflow == "rationales" else "Policy [P1]"
    evidence = {"P1": {"text": "Policy"}}
    context = {"workspace_id": "owned-workspace"}
    if overflow == "provenance":
        page_map = validate_page_map([{"page_number": 1, "start_char": 0, "end_char": 6,
                                       "printed_page_label": "𐐀" * 500_000}], 6)
        evidence["P1"]["page_spans"] = page_spans_for_range(page_map, 0, 6)
    if overflow == "context":
        context["padding"] = "𐐀" * 500_000

    async def analyze(*args, **kwargs):
        rationale = "𐐀" * 2048 if overflow == "rationales" else "Exact support."
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "label": "supported", "confidence": 0.95, "rationale": rationale}, ensure_ascii=False)}}]}

    report = await verify_cited_text(text, evidence, provider="openai", model="size-test", analyze_fn=analyze, context=context)
    assert report["findings"] == [{"id": "claim-1", "text": text, "start": 0, "end": len(text),
        "evidence_ids": [], "status": "not_checked", "reason": "report_size_limit"}]
    assert report["coverage"] == {"total": 1, "evaluated": 0, "supported": 0, "uncovered": 1, "complete": False}
    assert report["verdict"] == "unverified" and report["evidence"] == {}
    assert report["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert report["context"] == ({"unavailable_reason": "report_size_limit"} if overflow == "context" else context)
    assert validate_claim_verification(report) is report


@pytest.mark.asyncio
async def test_core_text_too_large_for_lossless_receipt_raises_instead_of_truncating():
    with pytest.raises(ValueError, match="report_size_limit"):
        await verify_cited_text("𐐀" * 500_001, {}, provider="openai", model="size-test")


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
    slow_started = asyncio.Event()
    original_wait = asyncio.wait
    observed_outer_timeouts = []

    async def expire_after_fast_completion(tasks, *, timeout=None, **kwargs):
        if timeout != 0.03:
            return await original_wait(tasks, timeout=timeout, **kwargs)
        observed_outer_timeouts.append(timeout)
        # Exercise completed-result retention independently of cold config and
        # thread startup; separate deadline tests retain real elapsed timing.
        await asyncio.wait_for(slow_started.wait(), timeout=5)
        done, pending = await original_wait(tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        assert len(done) == len(pending) == 1
        return done, pending

    monkeypatch.setattr(citation_support.asyncio, "wait", expire_after_fast_completion)

    async def analyze(*args, **kwargs):
        if "Fast" in args[1]:
            return response()
        try:
            slow_started.set()
            await asyncio.Event().wait()
        finally:
            cancelled.append(args[1])

    report = await verify_cited_text("Fast. [P1]\n\nSlow. [P1]", {"P1": {"text": "Exact source"}},
        provider="openai", model="total-deadline", analyze_fn=analyze)
    assert [row["status"] for row in report["findings"]] == ["supported", "not_checked"]
    assert report["findings"][1]["reason"] == "verification_deadline"
    assert report["coverage"]["uncovered"] == 1 and report["verdict"] == "unverified"
    assert cancelled == ["Slow. [P1]"]
    assert observed_outer_timeouts == [0.03]


def test_splitter_preserves_every_nonwhitespace_character_and_offsets():
    from hypothesis import given
    from hypothesis import strategies as st

    from tldw_Server_API.app.core.Claims_Extraction.citation_support import _units

    @given(st.lists(st.text(alphabet="abc XYZé🌙", min_size=1, max_size=30).filter(lambda value: value.strip()), min_size=1, max_size=8),
           st.one_of(st.none(), st.integers(min_value=0, max_value=8)))
    def check(parts, maximum):
        text = "\n\n".join(part + " [P1]" for part in parts)
        units = _units(text, maximum)
        assert maximum is None or len(units) <= maximum + 1
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


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", [", ", ",", "; ", ";\n", ",\n", "\n, "])
async def test_punctuated_adjacent_citations_verify_jointly_without_citation_only_units(separator):
    seen = []

    async def analyze(*args, **kwargs):
        seen.append((args[1], args[2]))
        return response()

    text = f"The combined proposition ([P1]{separator}[P2])."
    report = await verify_cited_text(text, {"P1": {"text": "First supporting fact."}, "P2": {"text": "Second supporting fact."}},
        provider="openai", model="punctuation-test", analyze_fn=analyze)
    assert report["coverage"] == {"total": 1, "evaluated": 1, "supported": 1, "uncovered": 0, "complete": True}
    assert report["findings"][0]["evidence_ids"] == ["P1", "P2"]
    assert len(seen) == 1 and "First supporting fact." in seen[0][1] and "Second supporting fact." in seen[0][1]
    assert seen[0][0] == text[:-2]


@pytest.mark.parametrize("separator", [". ", ": ", "–", "-", ",, ", "\n\n", "\n- ", " and ", ", another claim "])
def test_adjacent_citation_group_does_not_cross_claim_or_range_boundaries(separator):
    from tldw_Server_API.app.core.Claims_Extraction.citation_support import _units

    units = _units(f"First claim [P1]{separator}[P2]")
    assert units[0]["evidence_ids"] == ["P1"]
    assert len(units) == 2 and units[1]["evidence_ids"] == ["P2"]


@pytest.mark.asyncio
async def test_punctuated_citations_alone_still_do_not_form_a_claim():
    report = await verify_cited_text("([P1], [P2])", {"P1": {"text": "First."}, "P2": {"text": "Second."}},
        provider="openai", model="punctuation-test")
    assert report["findings"] == [{"id": "claim-1", "text": "([P1], [P2]", "start": 0, "end": 11,
        "evidence_ids": ["P1", "P2"], "status": "not_checked", "reason": "citation_without_claim"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("citation", [
    "[1, PDF p. 872; printed p. 839]", "[Source 1, PDF p. 872]",
    "[1, PDF pp. 871–872; printed pp. 838-839]", "[1, PDF pp. 871 - 872]",
])
async def test_numeric_pdf_location_annotation_is_not_an_extra_evidence_id(citation):
    seen = []

    async def analyze(*args, **kwargs):
        seen.append(args[1])
        return response()

    text = "The source supports this proposal " + citation
    report = await verify_cited_text(text, {"1": {"text": "Exact supporting source"}},
        provider="openai", model="page-label-test", analyze_fn=analyze)
    assert report["verdict"] == "supported" and report["findings"][0]["evidence_ids"] == ["1"]
    assert seen == [text] and report["findings"][0]["text"] == text


@pytest.mark.asyncio
@pytest.mark.parametrize("citation", [
    "[1, PDF p. unknown]", "[1, PDF p. 0]", "[1, PDF p. 872; another source]",
    "[1, PDF p. 872, P2]", "[P1, PDF p. 872]", "[1, arbitrary locator]",
])
async def test_unknown_annotation_or_identifier_never_silently_authorizes_support(citation):
    report = await verify_cited_text("Claim " + citation, {"1": {"text": "Source"}, "P1": {"text": "Source"}},
        provider="openai", model="invalid-label-test")
    assert report["findings"][0]["status"] == "not_checked"
    assert report["findings"][0]["reason"] == "citation_not_resolved"
    assert len(report["findings"][0]["evidence_ids"]) > 1
