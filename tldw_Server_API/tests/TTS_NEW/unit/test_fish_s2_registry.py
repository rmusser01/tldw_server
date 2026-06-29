import pytest

from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import ProviderStatus


@pytest.mark.unit
def test_fish_s2_aliases_resolve_to_provider():
    factory = TTSAdapterFactory(config={"providers": {"fish_s2": {"enabled": True}}})

    assert factory.get_provider_for_model("fish_s2") == TTSProvider.FISH_S2
    assert factory.get_provider_for_model("fish-s2-pro") == TTSProvider.FISH_S2
    assert factory.get_provider_for_model("s2-pro") == TTSProvider.FISH_S2
    assert factory.get_provider_for_model("fishaudio/s2-pro") == TTSProvider.FISH_S2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fish_s2_direct_dict_env_key_initializes_commercial_backend(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "fish-secret")

    class _MemoryMonitor:
        def is_memory_critical(self):
            return False

    class _ResourceManager:
        memory_monitor = _MemoryMonitor()

    async def _resource_manager():
        return _ResourceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_resource_manager",
        _resource_manager,
        raising=True,
    )

    factory = TTSAdapterFactory(
        config={
            "providers": {
                "fish_s2": {
                    "enabled": True,
                    "backend": "commercial_api",
                }
            }
        }
    )

    adapter = await factory.registry.get_adapter("fish_s2")

    assert adapter is not None
    assert adapter.status == ProviderStatus.AVAILABLE
