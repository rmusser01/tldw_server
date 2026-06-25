"""Regression tests for offloading blocking local TTS adapter calls."""

import pytest

from tldw_Server_API.app.core.TTS import utils as tts_utils
from tldw_Server_API.app.core.TTS.adapters import (
    chatterbox_adapter,
    dia_adapter,
    higgs_adapter,
    kokoro_adapter,
    vibevoice_adapter,
)

pytestmark = pytest.mark.unit


def _code_object_references_name(func, name: str) -> bool:
    """Return whether a function or nested function references a global name."""
    pending = [func.__code__]
    while pending:
        code = pending.pop()
        if name in code.co_names:
            return True
        pending.extend(
            const for const in code.co_consts
            if hasattr(const, "co_names") and hasattr(const, "co_consts")
        )
    return False


def test_blocking_local_generation_paths_use_thread_offload_helper():
    assert chatterbox_adapter.run_tts_blocking_call is tts_utils.run_tts_blocking_call
    assert dia_adapter.run_tts_blocking_call is tts_utils.run_tts_blocking_call
    assert higgs_adapter.run_tts_blocking_call is tts_utils.run_tts_blocking_call
    assert vibevoice_adapter.run_tts_blocking_call is tts_utils.run_tts_blocking_call
    assert kokoro_adapter.run_tts_blocking_next is tts_utils.run_tts_blocking_next

    assert _code_object_references_name(
        chatterbox_adapter.ChatterboxAdapter._stream_audio_chatterbox,
        "run_tts_blocking_call",
    )
    assert _code_object_references_name(
        chatterbox_adapter.ChatterboxAdapter._stream_voice_conversion_chatterbox,
        "run_tts_blocking_call",
    )
    assert _code_object_references_name(dia_adapter.DiaAdapter._stream_audio_dia, "run_tts_blocking_call")
    assert _code_object_references_name(higgs_adapter.HiggsAdapter._stream_audio_higgs, "run_tts_blocking_call")
    assert _code_object_references_name(vibevoice_adapter.VibeVoiceAdapter._stream_audio_vibevoice, "run_tts_blocking_call")
    assert _code_object_references_name(kokoro_adapter.KokoroAdapter._stream_audio_kokoro, "run_tts_blocking_next")
