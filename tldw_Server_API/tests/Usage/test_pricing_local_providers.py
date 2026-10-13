"""Local providers cost nothing by default, as data an operator can change (LLM usage accounting spec §6)."""

import json

import pytest

from tldw_Server_API.app.core.Usage.pricing_catalog import UNKNOWN_BILLABLE_RATE, PricingCatalog
from tldw_Server_API.app.core.Usage.usage_tracker import compute_costs

pytestmark = pytest.mark.unit

LOCAL = ("llama.cpp", "kobold", "ooba", "tabbyapi", "vllm", "local-llm", "ollama", "aphrodite", "mlx")


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from the shipped catalog."""
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)


@pytest.mark.parametrize("provider", LOCAL)
def test_each_local_provider_is_free_and_not_estimated(provider: str) -> None:
    """Any model of a local provider is priced at zero, and the price is not an estimate."""
    rates, estimated = PricingCatalog().get_rate_details(provider, "some-local-model-q4")
    assert (rates["prompt"], rates["completion"], estimated) == (0.0, 0.0, False)


@pytest.mark.parametrize("alias", ["llamacpp", "LLaMA.cpp", "oobabooga", "tabby", "local_llm", "koboldcpp"])
def test_aliases_price_like_their_canonical_provider(alias: str) -> None:
    """The writers pass different spellings; an alias is priced as its canonical provider."""
    rates, estimated = PricingCatalog().get_rate_details(alias, "m")
    assert (rates["prompt"], rates["completion"], estimated) == (0.0, 0.0, False)


def test_a_local_call_records_zero_cost() -> None:
    """compute_costs, which log_llm_usage calls, stores zero for a local call."""
    assert compute_costs("ollama", "llama3.1:8b", 10_000, 1_000) == (0.0, 0.0, 0.0, False)


def test_unknown_providers_and_custom_endpoints_keep_the_unknown_rate() -> None:
    """A provider with no entry, including a custom OpenAI-compatible endpoint, is still estimated at the unknown rate."""
    for provider in ("custom-openai-api", "some-new-cloud"):
        rates, estimated = PricingCatalog().get_rate_details(provider, "m")
        assert (rates["prompt"], rates["completion"], estimated) == (
            UNKNOWN_BILLABLE_RATE["prompt"],
            UNKNOWN_BILLABLE_RATE["completion"],
            True,
        )


def test_the_huggingface_default_is_now_reachable() -> None:
    """The provider-wide default the file already had is used, and stays marked as an estimate."""
    rates, estimated = PricingCatalog().get_rate_details("huggingface", "some/model")
    assert (rates["prompt"], estimated) == (pytest.approx(0.05e-3), True)


def test_an_operator_can_price_a_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local provider pointed at a paid service can be given a price."""
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps({"vllm": {"default": {"prompt": 0.002, "completion": 0.004}}}))
    rates, _ = PricingCatalog().get_rate_details("vllm", "anything")
    assert (rates["prompt"], rates["completion"]) == (0.002, 0.004)


def test_a_model_match_beats_the_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model entry wins over the provider-wide default, even when the model name contains 'default'."""
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps({"vllm": {"llama": {"prompt": 0.001, "completion": 0.002}}}))
    rates, estimated = PricingCatalog().get_rate_details("vllm", "llama-default-q4")
    assert (rates["prompt"], rates["completion"], estimated) == (0.001, 0.002, True)


def test_the_provider_default_is_not_advertised_as_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider-wide price is not a model, so model validation for a local provider stays unarmed."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Usage.pricing_catalog import list_provider_models

    monkeypatch.setattr(chat_service, "_configured_models_for_provider", lambda _provider: ())
    assert "default" not in list_provider_models("vllm")
    assert chat_service.is_model_known_for_provider("vllm", "llama3.1:8b") is None
