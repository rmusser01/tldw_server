from __future__ import annotations

from typing import Any


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
