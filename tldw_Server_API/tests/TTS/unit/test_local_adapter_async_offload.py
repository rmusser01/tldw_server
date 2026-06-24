import inspect

import pytest

from tldw_Server_API.app.core.TTS.adapters import (
    chatterbox_adapter,
    dia_adapter,
    higgs_adapter,
    kokoro_adapter,
    vibevoice_adapter,
)

pytestmark = pytest.mark.unit


def test_blocking_local_generation_paths_use_thread_offload_helper():
    checked_sources = [
        inspect.getsource(chatterbox_adapter.ChatterboxAdapter._stream_audio_chatterbox),
        inspect.getsource(chatterbox_adapter.ChatterboxAdapter._stream_voice_conversion_chatterbox),
        inspect.getsource(dia_adapter.DiaAdapter._stream_audio_dia),
        inspect.getsource(higgs_adapter.HiggsAdapter._stream_audio_higgs),
        inspect.getsource(vibevoice_adapter.VibeVoiceAdapter._stream_audio_vibevoice),
    ]

    for source in checked_sources:
        assert "run_tts_blocking_call" in source

    kokoro_source = inspect.getsource(kokoro_adapter.KokoroAdapter._stream_audio_kokoro)
    assert "run_tts_blocking_next" in kokoro_source
