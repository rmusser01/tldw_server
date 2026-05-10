from __future__ import annotations

import importlib


def test_pricing_catalog_loads_from_config_file(monkeypatch):
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    # Force reload to ensure file overrides are applied afresh
    mod = importlib.import_module("tldw_Server_API.app.core.Usage.pricing_catalog")
    importlib.reload(mod)
    catalog = mod.get_pricing_catalog()

    # Should read from tldw_Server_API/Config_Files/model_pricing.json
    in_per_1k, out_per_1k, estimated = catalog.get_rates("openai", "gpt-4o")

    # Values set in Config_Files/model_pricing.json
    assert round(in_per_1k, 6) == 0.0025
    assert round(out_per_1k, 6) == 0.01
    # File override yields exact match, not estimated
    assert estimated is False


def test_commercial_provider_catalog_exposes_current_models(monkeypatch):
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    mod = importlib.import_module("tldw_Server_API.app.core.Usage.pricing_catalog")
    importlib.reload(mod)

    assert "gpt-5.5" in mod.list_provider_models("openai")
    assert "claude-opus-4-7" in mod.list_provider_models("anthropic")
    assert "mistral-large-2512" in mod.list_provider_models("mistral")
    assert "grok-4.3" in mod.list_provider_models("xai")
    assert "kimi-k2.6" in mod.list_provider_models("moonshot")
    assert "minimax-m2.7" in mod.list_provider_models("minimax")


def test_placeholder_entries_are_hidden_from_provider_models(monkeypatch):
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    mod = importlib.import_module("tldw_Server_API.app.core.Usage.pricing_catalog")
    importlib.reload(mod)

    assert "gpt-image-1" not in mod.list_provider_models("openai")
    assert "claude-2.1" not in mod.list_provider_models("anthropic")
    assert "gemini-3-pro-preview" not in mod.list_provider_models("google")
    assert "mixtral-8x7b-32768" not in mod.list_provider_models("groq")
    assert "grok-2" not in mod.list_provider_models("xai")
    assert all("cache-hit" not in name for name in mod.list_provider_models("moonshot"))


def test_estimated_catalog_entries_remain_marked_estimated(monkeypatch):
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    mod = importlib.import_module("tldw_Server_API.app.core.Usage.pricing_catalog")
    importlib.reload(mod)
    catalog = mod.get_pricing_catalog()

    in_per_1k, out_per_1k, estimated = catalog.get_rates("qwen", "qwen3-max-preview")

    assert round(in_per_1k, 6) == 0.00104
    assert round(out_per_1k, 6) == 0.00624
    assert estimated is True
