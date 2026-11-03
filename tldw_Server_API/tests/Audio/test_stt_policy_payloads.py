from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_payload_failure_never_emits_text_or_segments(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.exception_types import ModerationExecutionError
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    prepare_moderation_incident_store()
    sent = []

    class Socket:
        async def send_json(self, payload):
            sent.append(payload)

    def redact(text, **kwargs):
        if text == "segment private":
            raise ModerationExecutionError("worker_crashed")
        return "safe"

    monkeypatch.setattr(stt_policy, "apply_transcript_text_policy", redact)
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
async def test_partial_opt_out_does_not_admit_skipped_text_but_checks_segments(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    policy = stt_policy.STTPolicy(None, True, 0.0, True, True, [])

    def redact(text, *, policy, is_partial):
        return text if is_partial else "[PII]"

    monkeypatch.setattr(stt_policy, "apply_transcript_text_policy", redact)
    payload = {"type": "partial", "text": "x" * 1000001, "segments": [{"text": "email"}]}
    result = await stt_policy.apply_transcript_payload_policy_async(payload, policy=policy)
    assert result["text"] == payload["text"]
    assert result["segments"] == [{"text": "[PII]"}]


def test_transcript_payload_policy_redacts_final_frames(monkeypatch) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy

    calls: list[dict[str, Any]] = []

    def _fake_apply_transcript_text_policy(
        text: str,
        *,
        policy: stt_policy.STTPolicy,
        is_partial: bool,
    ) -> str:
        calls.append({"text": text, "policy": policy, "is_partial": is_partial})
        return "[redacted]"

    policy = stt_policy.STTPolicy(
        org_id=None,
        delete_audio_after_success=True,
        audio_retention_hours=0.0,
        redact_pii=True,
        allow_unredacted_partials=False,
        redact_categories=["email"],
    )

    monkeypatch.setattr(stt_policy, "apply_transcript_text_policy", _fake_apply_transcript_text_policy)

    payload = stt_policy.apply_transcript_payload_policy(
        {"type": "final", "text": "email me at user@example.test", "is_final": True},
        policy=policy,
    )

    assert payload["text"] == "[redacted]"
    assert calls == [
        {
            "text": "email me at user@example.test",
            "policy": policy,
            "is_partial": False,
        }
    ]
