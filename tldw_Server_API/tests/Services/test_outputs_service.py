from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services import outputs_service


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_output_to_media_db_uses_media_repository_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_db = object()

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 88, "media-uuid", "stored"

    fake_repo = _FakeRepo()
    seen_db: list[object] = []

    def _fake_get_media_repository(db):
        seen_db.append(db)
        return fake_repo

    monkeypatch.setattr(outputs_service, "get_media_repository", _fake_get_media_repository, raising=False)

    media_id = await outputs_service._ingest_output_to_media_db(
        media_db=media_db,
        output_id=17,
        title="Weekly Briefing",
        content="Rendered body",
        output_type="briefing",
        output_format="md",
        storage_path="weekly.md",
        template_id=9,
        run_id=33,
        item_ids=[1, 2],
        tags=["watchlist", "briefing"],
        variant_of=5,
    )

    assert media_id == 88
    assert seen_db == [media_db]
    assert len(fake_repo.calls) == 1
    payload = fake_repo.calls[0]
    assert payload["url"] == "output://17"
    assert payload["title"] == "Weekly Briefing"
    assert payload["media_type"] == "output_briefing"
    assert payload["content"] == "Rendered body"
    assert payload["keywords"] == ["watchlist", "briefing"]
    assert payload["transcription_model"] == "output"
    assert payload["overwrite"] is False
    assert payload["ingestion_date"]
    assert json.loads(str(payload["safe_metadata"])) == {
        "output_id": 17,
        "output_type": "briefing",
        "output_format": "md",
        "storage_path": "weekly.md",
        "template_id": 9,
        "run_id": 33,
        "item_ids": [1, 2],
        "variant_of": 5,
    }


@pytest.mark.unit
def test_infer_output_tts_provider_from_model_omnivoice() -> None:
    assert outputs_service._infer_output_tts_provider_from_model("omnivoice") == "omnivoice"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_preserves_explicit_omnivoice() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model="omnivoice",
        tts_voice=None,
        template_row=SimpleNamespace(
            metadata_json=json.dumps(
                {
                    "tts_default_model": outputs_service.DEFAULT_KITTEN_TTS_MODEL,
                    "tts_default_voice": outputs_service.DEFAULT_KITTEN_TTS_VOICE,
                    "tts_default_speed": 1.25,
                }
            )
        ),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == "auto"
    assert resolved_speed == 1.25


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_template_selected_omnivoice_forces_auto_voice() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model=None,
        tts_voice=None,
        template_row=SimpleNamespace(
            metadata_json=json.dumps(
                {
                    "tts_default_model": "omnivoice",
                    "tts_default_voice": "legacy-template-voice",
                    "tts_default_speed": 0.9,
                }
            )
        ),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == "auto"
    assert resolved_speed == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_template_selected_omnivoice_preserves_explicit_voice() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model=None,
        tts_voice="caller-selected",
        template_row=SimpleNamespace(
            metadata_json=json.dumps(
                {
                    "tts_default_model": "omnivoice",
                    "tts_default_voice": "legacy-template-voice",
                    "tts_default_speed": 0.9,
                }
            )
        ),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == "caller-selected"
    assert resolved_speed == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_omnivoice_blank_voice_falls_back_to_auto() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model="omnivoice",
        tts_voice="   ",
        template_row=SimpleNamespace(metadata_json=None),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == outputs_service.DEFAULT_OMNIVOICE_TTS_VOICE
    assert resolved_speed is None
