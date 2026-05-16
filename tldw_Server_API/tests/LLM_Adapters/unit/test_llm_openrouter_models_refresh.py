import configparser
import asyncio


def _fake_config_openrouter():
    cfg = configparser.ConfigParser()
    cfg.add_section("API")
    cfg.set("API", "default_api", "openrouter")
    cfg.set("API", "openrouter_api_key", "sk-or-test")
    cfg.set("API", "openrouter_model", "openrouter/auto")
    cfg.add_section("Local-API")
    return cfg


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def close(self):
        return None


def test_llm_models_metadata_refresh_openrouter_includes_live_models(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.llm_providers as llm_providers
    import tldw_Server_API.app.core.LLM_Calls.openrouter_model_inventory as openrouter_inventory

    calls = {"count": 0}

    def _fake_fetch(*_args, **_kwargs):
        calls["count"] += 1
        return _FakeResponse(
            {
                "data": [
                    {"id": "openrouter/auto"},
                    {"id": "z-ai/glm-4.6"},
                ]
            }
        )

    monkeypatch.setattr(llm_providers, "load_comprehensive_config", _fake_config_openrouter)
    monkeypatch.setattr(llm_providers, "list_provider_models", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(llm_providers, "apply_llm_provider_overrides_to_listing", lambda result: result)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {"openrouter": "sk-or-test"})
    monkeypatch.setattr(llm_providers, "list_image_models_for_catalog", lambda: [])
    monkeypatch.setattr(llm_providers, "_http_fetch", _fake_fetch)
    openrouter_inventory.clear_openrouter_model_cache()

    refreshed = asyncio.run(
        llm_providers.get_models_metadata(
            request=None,
            refresh_openrouter=True,
            model_type=None,
            input_modality=None,
            output_modality=None,
        )
    )
    refreshed_names = {m.get("name") for m in refreshed.get("models", [])}
    assert "z-ai/glm-4.6" in refreshed_names
    assert calls["count"] == 1

    cached = asyncio.run(
        llm_providers.get_models_metadata(
            request=None,
            refresh_openrouter=False,
            model_type=None,
            input_modality=None,
            output_modality=None,
        )
    )
    cached_names = {m.get("name") for m in cached.get("models", [])}
    assert "z-ai/glm-4.6" in cached_names
    assert calls["count"] == 1
