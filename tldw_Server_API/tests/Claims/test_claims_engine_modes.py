import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, Optional

import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import Claim, ClaimsEngine


class Doc:
    def __init__(self, id: str, content: str, score: float = 0.0):
        self.id = id
        self.content = content
        self.score = score


def _analyze_stub(api_name: str, input_data: Any, custom_prompt_arg: Optional[str] = None,
                  api_key: Optional[str] = None, system_message: Optional[str] = None,
                  temp: Optional[float] = None, **kwargs):
    # Extraction path (LLM-based extractor)
    if system_message and "extract" in system_message.lower() and isinstance(custom_prompt_arg, str):
        return '{"claims": [{"text": "Stub claim one."}]}'
    # Judge path
    if system_message and "fact-checking judge" in system_message:
        return '{"label": "supported", "confidence": 0.9, "rationale": "stub"}'
    return '{"claims": []}'


@pytest.mark.unit
def test_claims_engine_llm_only_labels_supported():
    engine = ClaimsEngine(_analyze_stub)
    answer = "Alpha. Beta."
    query = "Q"
    documents = [Doc("d1", "Alpha Beta context", 0.5)]

    async def _run():
        result = await engine.run(
            answer=answer,
            query=query,
            documents=documents,
            claim_extractor="auto",
            claim_verifier="llm",
            claims_top_k=2,
            claims_conf_threshold=0.5,
            claims_max=5,
        )
        claims = result.get("claims") or []
        assert claims, "No claims returned"
        assert all(c.get("label") in {"supported", "refuted", "nei"} for c in claims)
        # Our stub judge always marks supported
        assert any(c.get("label") == "supported" for c in claims)

    asyncio.run(_run())


@pytest.mark.unit
def test_claims_engine_nli_only_without_model_returns_nei():
    # No transformers/NLI model available in test env, so NLI path should return NEI without LLM fallback
    def _analyze_noop(api_name: str, input_data: Any, custom_prompt_arg: Optional[str] = None,
                      api_key: Optional[str] = None, system_message: Optional[str] = None,
                      temp: Optional[float] = None, **kwargs):
                          return '{"claims": []}'

    engine = ClaimsEngine(_analyze_noop)
    answer = "Acme was founded in 2000."
    query = "When was Acme founded?"
    documents = [Doc("d1", "Acme context", 0.1)]

    async def _run():
        result = await engine.run(
            answer=answer,
            query=query,
            documents=documents,
            claim_extractor="auto",  # LLM path returns empty -> fallback to heuristic
            claim_verifier="nli",
            claims_top_k=2,
            claims_conf_threshold=0.7,
            claims_max=3,
        )
        claims = result.get("claims") or []
        assert claims, "Expected at least one heuristic claim"
        assert all(c.get("label") == "nei" for c in claims)

    asyncio.run(_run())


@pytest.mark.unit
def test_claims_engine_uses_structured_response_format():
    observed_formats = []

    def _analyze_with_capture(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        **kwargs,
    ):
        observed_formats.append(kwargs.get("response_format"))
        if system_message and "fact-checking judge" in system_message:
            return '{"label": "nei", "confidence": 0.4, "rationale": "stub"}'
        return '{"claims": [{"text": "Captured claim."}]}'

    engine = ClaimsEngine(_analyze_with_capture)
    documents = [Doc("d1", "Captured claim context.", 0.5)]

    async def _run():
        result = await engine.run(
            answer="Captured claim.",
            query="Q",
            documents=documents,
            claim_extractor="llm",
            claim_verifier="llm",
            claims_max=2,
        )
        assert result.get("claims")

    asyncio.run(_run())
    assert any(isinstance(fmt, dict) for fmt in observed_formats if fmt is not None)


@pytest.mark.unit
def test_claims_engine_parse_error_records_parse_event_and_fallback(monkeypatch):
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod

    captured = {"parse": [], "fallback": []}

    def _record_parse(**kwargs):
        captured["parse"].append(kwargs)

    def _record_fallback(**kwargs):
        captured["fallback"].append(kwargs)

    monkeypatch.setattr(engine_mod, "record_claims_output_parse_event", _record_parse)
    monkeypatch.setattr(engine_mod, "record_claims_fallback", _record_fallback)

    def _analyze_invalid_json(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        **kwargs,
    ):
        return "not valid json"

    engine = ClaimsEngine(_analyze_invalid_json)
    documents = [Doc("d1", "Context for parse fallback path.", 0.1)]

    async def _run():
        result = await engine.run(
            answer=(
                "Iota fact sentence for extractor fallback. "
                "Kappa fact sentence for extractor fallback."
            ),
            query="Q",
            documents=documents,
            claim_extractor="llm",
            claim_verifier="nli",
            claims_max=2,
        )
        claims = result.get("claims") or []
        assert claims

    asyncio.run(_run())

    assert any(
        item.get("mode") == "extract" and item.get("outcome") == "error"
        for item in captured["parse"]
    )
    assert any(
        item.get("mode") == "extract" and item.get("reason") == "parse_error"
        for item in captured["fallback"]
    )


@pytest.mark.unit
def test_claims_engine_verify_parse_error_records_parse_event_and_fallback(monkeypatch):
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod

    captured = {"parse": [], "fallback": []}

    def _record_parse(**kwargs):
        captured["parse"].append(kwargs)

    def _record_fallback(**kwargs):
        captured["fallback"].append(kwargs)

    monkeypatch.setattr(engine_mod, "record_claims_output_parse_event", _record_parse)
    monkeypatch.setattr(engine_mod, "record_claims_fallback", _record_fallback)

    def _analyze_invalid_verify(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        **kwargs,
    ):
        return "not valid json"

    engine = ClaimsEngine(_analyze_invalid_verify)
    documents = [Doc("d1", "Verifier context for parse fallback path.", 0.6)]

    async def _run():
        result = await engine.run(
            answer="Lambda fact sentence for verifier fallback.",
            query="Q",
            documents=documents,
            claim_extractor="heuristic",
            claim_verifier="llm",
            claims_max=2,
        )
        claims = result.get("claims") or []
        assert claims

    asyncio.run(_run())

    assert any(
        item.get("mode") == "verify" and item.get("outcome") == "error"
        for item in captured["parse"]
    )
    assert any(
        item.get("mode") == "verify" and item.get("reason") == "parse_error"
        for item in captured["fallback"]
    )


@pytest.mark.unit
def test_claims_engine_records_response_format_selection(monkeypatch):
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod

    captured: list[dict[str, Any]] = []

    monkeypatch.setattr(
        engine_mod,
        "record_claims_response_format_selection",
        lambda **kwargs: captured.append(kwargs),
    )

    def _analyze_stub(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        **kwargs,
    ):
        if system_message and "fact-checking judge" in system_message:
            return '{"label": "nei", "confidence": 0.4, "rationale": "stub"}'
        return '{"claims": [{"text": "Captured metric claim."}]}'

    engine = ClaimsEngine(_analyze_stub)
    documents = [Doc("d1", "Captured metric claim context.", 0.5)]

    async def _run():
        result = await engine.run(
            answer="Captured metric claim context.",
            query="Q",
            documents=documents,
            claim_extractor="llm",
            claim_verifier="llm",
            claims_max=2,
        )
        assert result.get("claims")

    asyncio.run(_run())
    modes = {item.get("mode") for item in captured}
    assert "extract" in modes
    assert "verify" in modes


@pytest.mark.unit
def test_claims_engine_multi_pass_dedupes_first_pass_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure ClaimsEngine multi-pass extraction de-duplicates repeated claim text.

    The first-pass claim identity should win even when later passes repeat it.
    """
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod

    calls: dict[str, int] = {"extract": 0}

    monkeypatch.setattr(engine_mod, "_resolve_claims_extraction_passes", lambda: 3)
    monkeypatch.setattr(engine_mod, "_resolve_claims_context_window_chars", lambda: 64)

    def _analyze_multi_pass(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Return a repeated claim for extraction and a stub label for judge calls."""
        if system_message and "fact-checking judge" in system_message:
            return '{"label": "nei", "confidence": 0.4, "rationale": "stub"}'
        calls["extract"] += 1
        return '{"claims": [{"text": "Repeated multi-pass claim."}]}'

    engine = ClaimsEngine(_analyze_multi_pass)

    async def _run() -> None:
        """Run ClaimsEngine with Doc input and assert first-pass de-dupe behavior."""
        result: dict[str, Any] = await engine.run(
            answer="Repeated multi-pass claim.",
            query="Q",
            documents=[Doc("d1", "Repeated multi-pass claim context.", 0.5)],
            claim_extractor="llm",
            claim_verifier="nli",
            claims_max=5,
        )
        claims = result.get("claims") or []
        assert len(claims) == 1
        assert claims[0].get("text") == "Repeated multi-pass claim."

    run_coro: Coroutine[Any, Any, None] = _run()
    asyncio.run(run_coro)
    assert calls["extract"] == 3


@pytest.mark.unit
def test_dedupe_claims_replaces_missing_span_with_later_span():
    claims = [
        Claim(id="c1", text="Repeated multi-pass claim.", span=None),
        Claim(id="c2", text="repeated   multi-pass claim.", span=(4, 28)),
    ]

    deduped = ClaimsEngine._dedupe_claims(claims, max_claims=5)

    assert len(deduped) == 1
    assert deduped[0].id == "c1"
    assert deduped[0].span == (4, 28)


@pytest.mark.unit
def test_claims_engine_summary_includes_refuted_status_count() -> None:
    """Verify summary counters include refuted status when judge returns refuted."""

    def _analyze_refuted(
        api_name: str,
        input_data: Any,
        custom_prompt_arg: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        temp: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Return one claim for extraction and a refuted label for judge calls."""
        if system_message and "fact-checking judge" in system_message:
            return '{"label": "refuted", "confidence": 0.95, "rationale": "stub"}'
        return '{"claims": [{"text": "Claim to refute."}]}'

    engine = ClaimsEngine(_analyze_refuted)

    async def _run() -> None:
        """Run ClaimsEngine and assert refuted_status is counted in summary."""
        result: dict[str, Any] = await engine.run(
            answer="Claim to refute.",
            query="Q",
            documents=[Doc("d1", "Contradictory context.", 0.9)],
            claim_extractor="llm",
            claim_verifier="llm",
            claims_max=2,
        )
        summary = result.get("summary") or {}
        assert int(summary.get("refuted_status", 0)) >= 1

    run_coro: Coroutine[Any, Any, None] = _run()
    asyncio.run(run_coro)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_extractor_rejects_shared_pool_saturation_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saturated process-wide provider cap cannot queue claim extraction."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    holder_entered = threading.Event()
    holder_release = threading.Event()
    analyze_started = threading.Event()
    pool = BoundedDaemonPool(1)

    def hold_capacity() -> None:
        holder_entered.set()
        holder_release.wait(timeout=2.0)

    def analyze_claims(*_args: Any, **_kwargs: Any) -> str:
        analyze_started.set()
        return '{"claims": [{"text": "Provider claim."}]}'

    monkeypatch.setattr(engine_mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    holder = pool.start(
        hold_capacity,
        name="claims-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    try:
        for _ in range(1000):
            if holder_entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert holder_entered.is_set()

        claims = await engine_mod.LLMBasedClaimExtractor(analyze_claims).extract(
            "Fallback claim sentence.",
            max_claims=2,
        )

        assert analyze_started.is_set() is False
        assert [claim.text for claim in claims] == ["Fallback claim sentence."]
        assert pool.active_count == 1
    finally:
        holder_release.set()
        holder.join(timeout=1.0)

    assert analyze_started.is_set() is False
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_claim_verifier_fanout_never_exceeds_shared_provider_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claims concurrency cannot multiply the process-wide provider cap."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    release = threading.Event()
    two_started = threading.Event()
    counter_lock = threading.Lock()
    started = 0
    pool = BoundedDaemonPool(2)

    def analyze_verification(*_args: Any, **_kwargs: Any) -> str:
        nonlocal started
        with counter_lock:
            started += 1
            if started >= 2:
                two_started.set()
        release.wait(timeout=2.0)
        return '{"label": "supported", "confidence": 0.9, "rationale": "ok"}'

    monkeypatch.setattr(engine_mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(
        engine_mod,
        "suggest_claims_concurrency",
        lambda *, requested, **_kwargs: requested,
    )
    engine = ClaimsEngine(analyze_verification)
    claims = [Claim(id=f"c{index}", text=f"Claim {index}.") for index in range(4)]
    task = asyncio.create_task(
        engine.verify_claims_only(
            claims=claims,
            query="query",
            documents=[Doc("d1", "Evidence for all claims.", 0.9)],
            claim_verifier="llm",
            claims_concurrency=4,
        )
    )
    try:
        for _ in range(1000):
            if two_started.is_set():
                break
            await asyncio.sleep(0.001)
        assert two_started.is_set()
        await asyncio.sleep(0.03)

        with counter_lock:
            assert started == 2
        assert pool.active_count == 2

        release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert len(result.verifications) == 4
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_provider_timeout_and_cancellation_hold_capacity_until_actual_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both deadline and caller cancellation retain the admitted claims worker."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    async def run_case(*, cancel: bool) -> None:
        entered = threading.Event()
        release = threading.Event()
        lifecycle: list[str] = []

        class TrackingPool(BoundedDaemonPool):
            def _release_capacity(self) -> None:
                lifecycle.append("capacity-release")
                super()._release_capacity()

        def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
            lifecycle.append("provider-start")
            entered.set()
            release.wait(timeout=2.0)
            lifecycle.append("provider-exit")
            return '{"claims": [{"text": "Provider claim."}]}'

        pool = TrackingPool(1)
        monkeypatch.setattr(engine_mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
        monkeypatch.setattr(
            engine_mod,
            "CLAIMS_PROVIDER_CALL_TIMEOUT_SECONDS",
            0.01 if not cancel else 1.0,
            raising=False,
        )
        task = asyncio.create_task(
            engine_mod.LLMBasedClaimExtractor(blocking_analyze).extract(
                "Fallback claim sentence.",
                max_claims=2,
            )
        )
        try:
            for _ in range(1000):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            if cancel:
                task.cancel()
            await asyncio.sleep(0.03)

            assert task.done() is False
            assert pool.active_count == 1
            assert lifecycle == ["provider-start"]

            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1.0)
            else:
                claims = await asyncio.wait_for(task, timeout=1.0)
                assert [claim.text for claim in claims] == ["Fallback claim sentence."]
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert pool.active_count == 0
        assert lifecycle == [
            "provider-start",
            "provider-exit",
            "capacity-release",
        ]

    await run_case(cancel=False)
    await run_case(cancel=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_nli_verification_does_not_consume_provider_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local deterministic NLI remains independent of the remote-provider pool."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    holder_entered = threading.Event()
    holder_release = threading.Event()
    provider_called = threading.Event()
    pool = BoundedDaemonPool(1)

    def hold_capacity() -> None:
        holder_entered.set()
        holder_release.wait(timeout=2.0)

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> str:
        provider_called.set()
        raise AssertionError("local NLI must not dispatch a provider call")

    def local_nli(_payload: dict[str, str]) -> list[list[dict[str, Any]]]:
        return [[
            {"label": "entailment", "score": 0.99},
            {"label": "neutral", "score": 0.01},
        ]]

    monkeypatch.setattr(engine_mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    verifier = engine_mod.HybridClaimVerifier(forbidden_provider)
    verifier._nli = local_nli
    holder = pool.start(
        hold_capacity,
        name="nli-provider-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    try:
        for _ in range(1000):
            if holder_entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert holder_entered.is_set()

        result = await verifier.verify(
            claim=Claim(id="c1", text="Acme was founded in 2000."),
            query="When was Acme founded?",
            base_documents=[Doc("d1", "Acme was founded in 2000.", 0.9)],
            mode="nli",
            conf_threshold=0.7,
        )

        assert result.status.value == "verified"
        assert provider_called.is_set() is False
        assert pool.active_count == 1
    finally:
        holder_release.set()
        holder.join(timeout=1.0)

    assert provider_called.is_set() is False
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aps_extraction_saturation_degrades_locally_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APS cannot bypass the process-wide provider cap or queue secret work."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Chunking.strategies import propositions

    holder_entered = threading.Event()
    holder_release = threading.Event()
    aps_started = threading.Event()
    analyze_started = threading.Event()
    pool = BoundedDaemonPool(1)

    class BlockingAPSStrategy:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def chunk(self, **_kwargs: Any) -> list[str]:
            aps_started.set()
            return ["Provider-only proposition."]

    def hold_capacity() -> None:
        holder_entered.set()
        holder_release.wait(timeout=2.0)

    def analyze_claims(*_args: Any, **_kwargs: Any) -> str:
        analyze_started.set()
        return '["Provider-only proposition."]'

    monkeypatch.setattr(engine_mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(propositions, "PropositionChunkingStrategy", BlockingAPSStrategy)
    monkeypatch.setattr(engine_mod, "_resolve_claims_extraction_passes", lambda: 1)
    monkeypatch.setattr(engine_mod, "_resolve_claims_context_window_chars", lambda: 0)
    holder = pool.start(
        hold_capacity,
        name="aps-capacity-holder",
        exhaustion_message="test capacity exhausted",
    )
    try:
        for _ in range(1000):
            if holder_entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert holder_entered.is_set()

        claims, _mode = await ClaimsEngine(analyze_claims)._extract_claims_by_mode(
            answer="Fallback claim sentence.",
            claim_extractor="aps",
            claims_max=2,
            budget=None,
            job_context=None,
        )

        assert [claim.text for claim in claims] == ["Fallback claim sentence."]
        assert aps_started.is_set() is False
        assert analyze_started.is_set() is False
        assert pool.active_count == 1
    finally:
        holder_release.set()
        holder.join(timeout=1.0)

    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aps_extraction_deadline_keeps_event_loop_responsive_and_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late APS strategy drains its worker but cannot block the event loop."""
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as engine_mod
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Chunking.strategies import propositions

    started = threading.Event()
    release = threading.Event()
    lifecycle: list[str] = []

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            super()._release_capacity()
            lifecycle.append("capacity-release")

    class LateAPSStrategy:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def chunk(self, **_kwargs: Any) -> list[str]:
            lifecycle.append("aps-start")
            started.set()
            release.wait(timeout=1.0)
            lifecycle.append("aps-exit")
            return ["Late provider proposition."]

    async def heartbeat() -> None:
        for _ in range(1000):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()
        await asyncio.sleep(0.04)
        lifecycle.append("loop-heartbeat")
        release.set()

    pool = TrackingPool(1)
    monkeypatch.setattr(engine_mod, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(
        engine_mod,
        "CLAIMS_PROVIDER_CALL_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(propositions, "PropositionChunkingStrategy", LateAPSStrategy)
    engine = ClaimsEngine(lambda *_args, **_kwargs: None)
    safety_release = threading.Timer(0.15, release.set)
    safety_release.daemon = True
    safety_release.start()
    extraction = asyncio.create_task(
        engine._extract_aps_claim_texts("Late provider sentence.", 2)
    )
    loop_heartbeat = asyncio.create_task(heartbeat())
    try:
        claim_texts = await asyncio.wait_for(extraction, timeout=1.0)
        await asyncio.wait_for(loop_heartbeat, timeout=1.0)
    finally:
        release.set()
        safety_release.cancel()
        await asyncio.gather(extraction, loop_heartbeat, return_exceptions=True)

    assert claim_texts == []
    assert lifecycle == [
        "aps-start",
        "loop-heartbeat",
        "aps-exit",
        "capacity-release",
    ]
    assert pool.active_count == 0
