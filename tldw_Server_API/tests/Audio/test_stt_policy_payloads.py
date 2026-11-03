from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def moderation(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.tests.unit.test_moderation_execution_service import make_service

    service = make_service()
    monkeypatch.setattr(stt_policy, "get_moderation_service", lambda: service)
    return service


@pytest.mark.asyncio
async def test_many_segment_payload_uses_one_real_batch_and_preserves_fields(monkeypatch, tmp_path):
    import json
    import time

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.app.core.Moderation import execution_runner
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store
    from tldw_Server_API.app.core.Moderation.execution_models import ExecutionBounds
    from tldw_Server_API.tests.unit.test_moderation_execution_service import make_service

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    prepare_moderation_incident_store()
    service = make_service()
    service._load_builtin_pii_rules()  # Warm trusted imports before the scoped measurement.
    monkeypatch.setattr(stt_policy, "get_moderation_service", lambda: service)
    children = []

    def spawn(*args, **kwargs):
        child = execution_runner.subprocess.Popen(*args, **kwargs)
        children.append(child)
        return child

    owner = execution_runner.ModerationExecutionRunner(ExecutionBounds(), process_factory=spawn)
    launches = []
    execute = owner.execute

    def run(request, scope):
        launches.append(request.operation)
        return execute(request, scope)

    monkeypatch.setattr(owner, "execute", run)
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: owner)
    policy = stt_policy.STTPolicy(None, True, 0.0, True, False, [])
    segment_text = "Research notes for today."
    payload = {"type": "final", "text": " ".join([segment_text] * 100), "segments": [
        {"Text": segment_text, "start": i * 6, "end": (i + 1) * 6, "speaker": "researcher"} for i in range(100)
    ]}
    assert stt_policy.transcript_policy_input_chars(payload) == 5099
    started = time.monotonic()
    try:
        result = await stt_policy.apply_transcript_payload_policy_async(payload, policy=policy)
        elapsed = time.monotonic() - started
        assert result == payload
        assert result is not payload and result["segments"][0] is not payload["segments"][0]
        assert launches == ["redact_many"]
        assert len(children) == 1 and children[0].poll() == 0
        assert elapsed < 2
        assert not owner._entries
        print(json.dumps({"segments": 100, "input_chars": 5099, "builtin_rules": len(service._load_builtin_pii_rules()),
                          "elapsed_ms": round(elapsed * 1000, 3), "actual_launches": len(children),
                          "execute_attempts": len(launches), "correct": result == payload, "owned_after": len(owner._entries)}))
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_payload_failure_never_emits_text_or_segments(monkeypatch, tmp_path, moderation):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    prepare_moderation_incident_store()
    sent = []

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    def redact(text, policy, phase=None):
        if text == "segment private":
            raise ModerationExecutionError("worker_crashed")
        return "safe"

    monkeypatch.setattr(moderation, "redact_text", redact)
    policy = stt_policy.STTPolicy(None, True, 0.0, True, False, [])
    proxy = stt_policy.RedactingWebSocketProxy(Socket(), policy=policy)
    with pytest.raises(ModerationExecutionError):
        await proxy.send_json({"type": "final", "text": "private", "segments": [{"text": "segment private"}]})
    assert sent == []


@pytest.mark.asyncio
async def test_disabled_payload_policy_does_not_admit_content(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    sent = []

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    policy = stt_policy.STTPolicy(None, True, 0.0, False, False, [])
    payload = {"type": "final", "text": "x" * 1000001}
    await stt_policy.RedactingWebSocketProxy(Socket(), policy=policy).send_json(payload)
    assert sent == [payload]


@pytest.mark.asyncio
async def test_partial_opt_out_does_not_admit_skipped_text_but_checks_segments(monkeypatch, moderation):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    policy = stt_policy.STTPolicy(None, True, 0.0, True, True, [])

    def redact(text, policy, phase=None):
        assert text == "email"
        return "[PII]"

    monkeypatch.setattr(moderation, "redact_text", redact)
    payload = {"type": "partial", "text": "x" * 1000001, "segments": [{"text": "email"}]}
    result = await stt_policy.apply_transcript_payload_policy_async(payload, policy=policy)
    assert result["text"] == payload["text"]
    assert result["segments"] == [{"text": "[PII]"}]


def test_transcript_payload_policy_redacts_final_frames(monkeypatch, moderation) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    calls: list[dict[str, Any]] = []

    def _fake_apply_transcript_text_policy(
        text: str,
        policy,
        phase=None,
    ) -> str:
        calls.append({"text": text, "phase": phase, "replacement": policy.redact_replacement})
        return "[redacted]"

    policy = stt_policy.STTPolicy(
        org_id=None,
        delete_audio_after_success=True,
        audio_retention_hours=0.0,
        redact_pii=True,
        allow_unredacted_partials=False,
        redact_categories=["email"],
    )

    monkeypatch.setattr(moderation, "redact_text", _fake_apply_transcript_text_policy)

    payload = stt_policy.apply_transcript_payload_policy(
        {"type": "final", "text": "email me at user@example.test", "is_final": True},
        policy=policy,
    )

    assert payload["text"] == "[redacted]"
    assert calls == [
        {
            "text": "email me at user@example.test",
            "phase": None,
            "replacement": "[PII]",
        }
    ]


@pytest.mark.asyncio
async def test_late_field_output_failure_is_atomic_and_reports_once(monkeypatch, tmp_path, moderation):
    import copy
    import re

    from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.app.core.Moderation import moderation_service
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store
    from tldw_Server_API.app.core.Moderation.execution_models import ExecutionBounds
    from tldw_Server_API.app.core.Moderation.models import PatternRule

    path = tmp_path / "review.db"
    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(path))
    prepare_moderation_incident_store()
    monkeypatch.setattr(moderation_service, "ExecutionBounds", lambda: ExecutionBounds(max_output_chars=3))
    monkeypatch.setattr(moderation, "_load_builtin_pii_rules", lambda: [PatternRule(re.compile("a"), replacement="zz")])
    policy = stt_policy.STTPolicy(None, True, 0, True, False, [])
    sent = []

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    payload = {"type": "final", "text": "a", "segments": [{"Text": "a", "start": 0, "end": 6}]}
    original = copy.deepcopy(payload)
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        await stt_policy.RedactingWebSocketProxy(Socket(), policy=policy).send_json(payload)
    assert payload == original
    assert not sent
    [item] = ModerationReviewStore(path).list_items()["items"]
    assert item["context"]["input_chars"] == 2
    assert item["context"]["operation"] == "redact_many"


@pytest.mark.asyncio
async def test_cancelled_batch_never_sends_payload_and_reaps_eventual_child(monkeypatch, moderation):
    import asyncio
    import threading

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.app.core.Moderation import async_execution, execution_runner
    from tldw_Server_API.tests.unit.test_moderation_execution_runner import Child, runner

    entered, publish = threading.Event(), threading.Event()
    child = Child(stalled=True)
    sent = []

    def spawn(*args, **kwargs):
        entered.set()
        assert publish.wait(2)
        return child

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    owner = runner(spawn)
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: owner)
    payload = {"type": "final", "text": "private", "segments": [{"text": "private segment"}]}
    policy = stt_policy.STTPolicy(None, True, 0, True, False, [])
    task = asyncio.create_task(stt_policy.RedactingWebSocketProxy(Socket(), policy=policy).send_json(payload))
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        publish.set()
        async with asyncio.timeout(2):
            while async_execution._active:
                await asyncio.sleep(0.005)
        assert not owner._entries and child.killed.is_set()
        assert not sent
        assert payload["segments"] == [{"text": "private segment"}]
    finally:
        publish.set()
        owner.close()


def test_empty_audio_fields_keep_noop_guards_and_dual_key_mapping(monkeypatch, moderation):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    checked = []

    def redact(text, policy, phase=None):
        checked.append(text)
        return "safe:" + text

    monkeypatch.setattr(moderation, "redact_text", redact)
    payload = {"type": "final", "text": "", "segments": [
        {"text": "", "Text": "shadow", "start": 0}, {"Text": "a", "speaker": "one"},
    ]}
    result = stt_policy.apply_transcript_payload_policy(payload, policy=stt_policy.STTPolicy(None, True, 0, True, False, []))
    assert checked == ["a"]
    assert result == {"type": "final", "text": "", "segments": [
        {"text": "", "Text": "", "start": 0}, {"Text": "safe:a", "speaker": "one"},
    ]}
    assert payload["segments"][0]["Text"] == "shadow"
