import json
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.phoneme_overrides import (
    PhonemeOverrideEntry,
    apply_overrides_to_text,
    filter_overrides_for_provider,
    load_override_entries,
    merge_override_entries,
    parse_override_entries,
)


@pytest.fixture
def captured_phoneme_override_logs():
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
        format="{message}",
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_apply_overrides_respects_lang_and_boundaries():
    entries = [
        PhonemeOverrideEntry(term="OpenAI", phonemes="ow p en aɪ", lang="en", boundary=True),
        PhonemeOverrideEntry(term="bonjour", phonemes="b ɔ̃ ʒ u ʁ", lang="fr", boundary=True),
    ]
    text = "OpenAI builds tools. bonjour monde."
    updated = apply_overrides_to_text(text, entries, lang_hint="en-US")

    assert "[[ow p en aɪ]]" in updated
    # French entry is skipped because lang_hint is English
    assert "bonjour" in updated


def test_merge_precedence_request_wins():
    global_entries = parse_override_entries({"demo": "AAA"})
    provider_entries = parse_override_entries([{"term": "demo", "phonemes": "BBB", "boundary": False}])
    request_entries = parse_override_entries({"demo": "CCC"})

    merged = merge_override_entries(global_entries, provider_entries, request_entries)

    assert len(merged) == 1
    assert merged[0].phonemes == "CCC"
    assert merged[0].boundary is True  # request entry falls back to default boundary


def test_parse_structured_config_supports_provider_scopes():
    raw = {
        "version": 1,
        "global": [{"term": "SQL", "phonemes": "S-Q-L", "lang": "en"}],
        "providers": {
            "kokoro": [{"term": "OpenAI", "phonemes": "oʊ p ən aɪ"}],
            "openai": [{"term": "OpenAI", "phonemes": "oh-pen-ai"}],
        },
    }
    entries = parse_override_entries(raw)
    kokoro_entries = filter_overrides_for_provider(entries, "kokoro")
    openai_entries = filter_overrides_for_provider(entries, "openai")

    assert any(e.term == "SQL" and e.provider is None for e in entries)
    assert any(e.term == "OpenAI" and e.provider == "kokoro" for e in entries)
    assert any(e.term == "OpenAI" and e.provider == "openai" for e in entries)
    assert any(e.term == "SQL" for e in kokoro_entries)  # global applies to all providers
    assert any(e.term == "OpenAI" and e.phonemes == "oʊ p ən aɪ" for e in kokoro_entries)
    assert any(e.term == "OpenAI" and e.phonemes == "oh-pen-ai" for e in openai_entries)


def test_parse_mapping_ignores_nested_values():
    entries = parse_override_entries({"demo": {"phonemes": "AAA"}, "ok": "BBB"})
    assert len(entries) == 1
    assert entries[0].term == "ok"
    assert entries[0].phonemes == "BBB"


def test_apply_overrides_handles_overlap_with_longer_term_first():
    entries = [
        PhonemeOverrideEntry(term="York", phonemes="YORK", boundary=True),
        PhonemeOverrideEntry(term="New York", phonemes="NEW_YORK", boundary=True),
    ]
    text = "I visited New York and York."
    updated = apply_overrides_to_text(text, entries, lang_hint="en")

    assert "[[NEW_YORK]]" in updated
    assert updated.count("[[YORK]]") == 1


def test_adapter_applies_request_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Write a small JSON override file so load_override_entries can find it
    override_file = tmp_path / "tts_phonemes.json"
    override_file.write_text(json.dumps({"demo": "AAA"}), encoding="utf-8")
    monkeypatch.setenv("TTS_PHONEME_OVERRIDES_PATH", str(override_file))

    adapter = KokoroAdapter({"kokoro_phoneme_overrides": {"demo": "BBB"}})
    request = TTSRequest(
        text="demo phrase",
        voice="af_bella",
        format=AudioFormat.MP3,
        extra_params={"phoneme_overrides": {"demo": "CCC"}},
    )

    # Ensure the global file is picked up
    assert load_override_entries(str(override_file))

    updated = adapter._apply_phoneme_overrides_to_text(request.text, request=request, lang_hint="en")

    assert "[[CCC]]" in updated
    # Request-level flag can disable overrides entirely
    request.extra_params["disable_phoneme_overrides"] = True
    assert adapter._phoneme_overrides_enabled_for_request(request) is False


@pytest.mark.asyncio
async def test_kokoro_generate_applies_overrides_in_generate_flow(monkeypatch: pytest.MonkeyPatch):
    adapter = KokoroAdapter(
        {
            "kokoro_lazy_init": True,
            "kokoro_phoneme_overrides": {"OpenAI": "provider"},
        }
    )
    adapter.use_onnx = True
    adapter._deferred_model_load = False

    async def _ensure_initialized() -> bool:
        return True

    captured: dict[str, str] = {}

    async def _generate_complete_kokoro(text: str, voice: str, lang: str, request: TTSRequest) -> bytes:
        captured["text"] = text
        return b"audio-bytes"

    monkeypatch.setattr(adapter, "ensure_initialized", _ensure_initialized)
    monkeypatch.setattr(adapter, "_ensure_audio_normalizer", lambda: None)
    monkeypatch.setattr(adapter, "_generate_complete_kokoro", _generate_complete_kokoro)

    request = TTSRequest(
        text="OpenAI ships tools.",
        voice="af_bella",
        format=AudioFormat.MP3,
        stream=False,
        extra_params={"phoneme_overrides": {"OpenAI": "request-level"}},
    )

    response = await adapter.generate(request)
    assert response.audio_data == b"audio-bytes"
    assert "[[request-level]]" in captured["text"]


def test_invalid_config_path_candidate_log_omits_raw_path_and_exception_text(
    captured_phoneme_override_logs,
):
    secret_path = "SECRET_CONFIG_PATH_\0invalid"

    entries = load_override_entries(secret_path)

    log_text = "\n".join(captured_phoneme_override_logs)
    assert entries == []
    assert "Skipping invalid config path candidate" in log_text
    assert "ValueError" in log_text
    assert "SECRET_CONFIG_PATH" not in log_text
    assert "File name too long" not in log_text


def test_failed_entry_parse_log_omits_raw_exception_text(captured_phoneme_override_logs):
    class SecretStrFailure:
        def __str__(self) -> str:
            raise ValueError("SECRET_PARSE_VALUE")

    entries = parse_override_entries([{"term": SecretStrFailure(), "phonemes": "AH"}])

    log_text = "\n".join(captured_phoneme_override_logs)
    assert entries == []
    assert "Failed to parse phoneme override entry" in log_text
    assert "ValueError" in log_text
    assert "SECRET_PARSE_VALUE" not in log_text


def test_failed_load_from_path_log_omits_raw_path_and_exception_text(
    tmp_path: Path,
    captured_phoneme_override_logs,
):
    override_file = tmp_path / "SECRET_LOAD_PATH_tts_phonemes.json"
    override_file.write_text("{not-valid-json", encoding="utf-8")

    entries = load_override_entries(str(override_file))

    log_text = "\n".join(captured_phoneme_override_logs)
    assert entries == []
    assert "Failed to load phoneme overrides" in log_text
    assert "JSONDecodeError" in log_text
    assert str(override_file) not in log_text
    assert "SECRET_LOAD_PATH" not in log_text
    assert "Expecting property name" not in log_text
