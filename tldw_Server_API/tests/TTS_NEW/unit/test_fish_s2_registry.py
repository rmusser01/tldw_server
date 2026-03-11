import pytest

from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider


@pytest.mark.unit
def test_fish_s2_aliases_resolve_to_provider():
    factory = TTSAdapterFactory(config={"providers": {"fish_s2": {"enabled": True}}})

    assert factory.get_provider_for_model("fish_s2") == TTSProvider.FISH_S2
    assert factory.get_provider_for_model("fish-s2-pro") == TTSProvider.FISH_S2
    assert factory.get_provider_for_model("s2-pro") == TTSProvider.FISH_S2
    assert factory.get_provider_for_model("fishaudio/s2-pro") == TTSProvider.FISH_S2
