import pytest

from tldw_Server_API.app.core.TTS.adapter_registry import (
    TTSAdapterFactory,
    TTSAdapterRegistry,
    TTSProvider,
)


@pytest.mark.unit
def test_audio_cpp_provider_aliases_resolve():
    assert TTSAdapterRegistry.resolve_provider("audio_cpp") == TTSProvider.AUDIO_CPP
    assert TTSAdapterRegistry.resolve_provider("audio-cpp") == TTSProvider.AUDIO_CPP
    assert TTSAdapterRegistry.resolve_provider("audiocpp") == TTSProvider.AUDIO_CPP


@pytest.mark.unit
def test_audio_cpp_model_aliases_are_namespaced_and_do_not_steal_pocket_tts():
    factory = TTSAdapterFactory({})

    assert factory.get_provider_for_model("audio_cpp:pocket-tts") == TTSProvider.AUDIO_CPP
    assert factory.get_provider_for_model("audio-cpp/pocket-tts") == TTSProvider.AUDIO_CPP
    assert factory.get_provider_for_model("audiocpp/pocket-tts") == TTSProvider.AUDIO_CPP
    assert factory.get_provider_for_model("pocket-tts") == TTSProvider.POCKET_TTS


@pytest.mark.unit
def test_audio_cpp_default_adapter_is_registered_lazily():
    assert (
        TTSAdapterRegistry.DEFAULT_ADAPTERS[TTSProvider.AUDIO_CPP]
        == "tldw_Server_API.app.core.TTS.adapters.audio_cpp_adapter.AudioCppTTSAdapter"
    )
