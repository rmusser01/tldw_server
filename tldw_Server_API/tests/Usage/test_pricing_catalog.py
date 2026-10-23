from __future__ import annotations

import json
import os
import re

import pytest

from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog, list_provider_models, reset_pricing_catalog
from tldw_Server_API.app.core.Usage.usage_tracker import compute_costs


def test_pricing_defaults_and_partial_match() -> None:
    cat = PricingCatalog()
    pr, cr, est = cat.get_rates('openai', 'gpt-3.5-turbo')
    assert pr > 0 and cr > 0
    assert est is False

    # Partial model name should mark estimated True if not exact
    pr2, cr2, est2 = cat.get_rates('openai', 'gpt-4-something')
    assert pr2 >= 0 and cr2 >= 0
    assert est2 in (True, False)  # May map to gpt-4 baseline

    # Unknown provider/model → tiny sentinel with estimated True
    pr3, cr3, est3 = cat.get_rates('unknownprov', 'mymodel')
    assert pr3 > 0 and cr3 > 0 and est3 is True


def test_pricing_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {
        "openai": {"gpt-3.5-turbo": {"prompt": 0.123, "completion": 0.456}}
    }
    monkeypatch.setenv('PRICING_OVERRIDES', json.dumps(overrides))
    cat = PricingCatalog()
    pr, cr, est = cat.get_rates('openai', 'gpt-3.5-turbo')
    assert pr == pytest.approx(0.123)
    assert cr == pytest.approx(0.456)
    assert est is False


def test_pricing_env_override_supports_cache_read_write_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {
        "openai": {
            "cached-test": {
                "prompt": 0.010,
                "completion": 0.030,
                "cache_read": 0.001,
                "cache_write": 0.005,
            }
        }
    }
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps(overrides))
    cat = PricingCatalog()

    rates, estimated = cat.get_rate_details("openai", "cached-test")

    assert rates["prompt"] == pytest.approx(0.010)
    assert rates["completion"] == pytest.approx(0.030)
    assert rates["cache_read"] == pytest.approx(0.001)
    assert rates["cache_write"] == pytest.approx(0.005)
    assert estimated is False


def test_compute_costs_uses_cache_rates_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {
        "openai": {
            "cached-test": {
                "prompt": 0.010,
                "completion": 0.030,
                "cache_read": 0.001,
                "cache_write": 0.005,
            }
        }
    }
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps(overrides))
    reset_pricing_catalog()

    prompt_cost, completion_cost, total_cost, estimated = compute_costs(
        "openai",
        "cached-test",
        prompt_tokens=100,
        completion_tokens=10,
        cache_read_input_tokens=40,
        cache_write_input_tokens=20,
        billable_input_tokens=40,
    )

    assert prompt_cost == pytest.approx(((40 * 0.010) + (40 * 0.001) + (20 * 0.005)) / 1000.0)
    assert completion_cost == pytest.approx((10 * 0.030) / 1000.0)
    assert total_cost == pytest.approx(prompt_cost + completion_cost)
    assert estimated is False


def test_compute_costs_clamps_billable_input_tokens_to_uncached_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = {
        "openai": {
            "cached-test": {
                "prompt": 0.010,
                "completion": 0.030,
                "cache_read": 0.001,
                "cache_write": 0.005,
            }
        }
    }
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps(overrides))
    reset_pricing_catalog()

    prompt_cost, completion_cost, total_cost, estimated = compute_costs(
        "openai",
        "cached-test",
        prompt_tokens=100,
        completion_tokens=0,
        cache_read_input_tokens=80,
        cache_write_input_tokens=0,
        billable_input_tokens=100,
    )

    assert prompt_cost == pytest.approx(((20 * 0.010) + (80 * 0.001)) / 1000.0)
    assert completion_cost == 0.0
    assert total_cost == pytest.approx(prompt_cost)
    assert estimated is False


# Anthropic list prices per 1K tokens: prompt, completion, cache hit, 5-minute cache write, 1-hour cache write.
# Source: platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-20 (TASK-13858); 1-hour writes
# fetched 2026-10-22 (TASK-13878).
_FABLE_5_1 = {"prompt": 0.01, "completion": 0.05, "cache_read": 0.00025, "cache_write": 0.0125, "cache_write_1h": 0.02}
_FABLE_5 = {"prompt": 0.01, "completion": 0.05, "cache_read": 0.001, "cache_write": 0.0125, "cache_write_1h": 0.02}
_OPUS_5_5 = {"prompt": 0.004, "completion": 0.02, "cache_read": 0.0002, "cache_write": 0.005, "cache_write_1h": 0.008}
_OPUS_5 = {"prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625, "cache_write_1h": 0.01}
_OPUS_4_1 = {"prompt": 0.015, "completion": 0.075, "cache_read": 0.0015, "cache_write": 0.01875, "cache_write_1h": 0.03}
_SONNET_5_5 = {"prompt": 0.002, "completion": 0.01, "cache_read": 0.0001, "cache_write": 0.0025, "cache_write_1h": 0.004}
_SONNET_5 = {"prompt": 0.002, "completion": 0.01, "cache_read": 0.0002, "cache_write": 0.0025, "cache_write_1h": 0.004}
_SONNET_4 = {"prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375, "cache_write_1h": 0.006}
_HAIKU_4_5 = {"prompt": 0.001, "completion": 0.005, "cache_read": 0.0001, "cache_write": 0.00125, "cache_write_1h": 0.002}


@pytest.mark.parametrize(
    ("model", "rates"),
    [
        ("claude-fable-5-1", _FABLE_5_1),
        ("claude-fable-5", _FABLE_5),
        ("claude-opus-5-5", _OPUS_5_5),
        ("claude-opus-5", _OPUS_5),
        ("claude-opus-4-8", _OPUS_5),
        ("claude-opus-4-7", _OPUS_5),
        ("claude-opus-4-5", _OPUS_5),
        ("claude-opus-4-5-20251101", _OPUS_5),
        ("claude-opus-4-1-20250805", _OPUS_4_1),
        ("claude-sonnet-5-5", _SONNET_5_5),
        ("claude-sonnet-5", _SONNET_5),
        ("claude-sonnet-4-6", _SONNET_4),
        ("claude-sonnet-4-5-20250929", _SONNET_4),
        ("claude-haiku-4-5-20251001", _HAIKU_4_5),
    ],
)
def test_anthropic_model_ids_get_their_list_prices_and_cache_rates(
    model: str, rates: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, _estimated = PricingCatalog().get_rate_details("anthropic", model)

    assert got == pytest.approx(rates)


# OpenAI Standard-tier list prices per 1K tokens: input, cached input.
# Source: developers.openai.com/api/docs/pricing, fetched 2026-10-22 (TASK-13879).
_OPENAI_CACHED = {
    "gpt-5.5": (0.005, 0.0005),
    "gpt-5.4": (0.0025, 0.00025),
    "gpt-5.4-mini": (0.00075, 0.000075),
    "gpt-5.4-nano": (0.0002, 0.00002),
    "gpt-5.2": (0.00175, 0.000175),
    "gpt-5.1": (0.00125, 0.000125),
    "gpt-5": (0.00125, 0.000125),
    "gpt-5-mini": (0.00025, 0.000025),
    "gpt-5-nano": (0.00005, 0.000005),
    "gpt-4.1": (0.002, 0.0005),
    "gpt-4.1-mini": (0.0004, 0.0001),
    "gpt-4.1-nano": (0.0001, 0.000025),
    "gpt-4o": (0.0025, 0.00125),
    "gpt-4o-mini": (0.00015, 0.000075),
    "o1": (0.015, 0.0075),
    "o3": (0.002, 0.0005),
    "o3-mini": (0.0011, 0.00055),
    "o4-mini": (0.0011, 0.000275),
}


@pytest.mark.parametrize(("model", "rates"), sorted(_OPENAI_CACHED.items()))
def test_openai_models_carry_their_published_cached_input_rate(
    model: str, rates: tuple[float, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, _estimated = PricingCatalog().get_rate_details("openai", model)

    assert got["prompt"] == pytest.approx(rates[0])
    assert got["cache_read"] == pytest.approx(rates[1])


@pytest.mark.parametrize("model", ["gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.2-pro", "gpt-5-pro", "o1-pro"])
def test_openai_models_without_a_published_cached_rate_get_none(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, _estimated = PricingCatalog().get_rate_details("openai", model)

    assert "cache_read" not in got


def test_an_openai_row_with_cached_tokens_is_costed_at_the_cached_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 800}}
    normalized = normalize_llm_usage(provider="openai", usage=usage)

    prompt_cost, completion_cost, _total, _est = compute_costs(
        "openai",
        "gpt-4.1",
        1000,
        100,
        cache_read_input_tokens=normalized.cache_read_input_tokens,
        cache_write_input_tokens=normalized.cache_write_input_tokens,
        billable_input_tokens=normalized.billable_input_tokens,
    )

    # gpt-4.1 per 1K: input $0.002, cached $0.0005, output $0.008.
    assert prompt_cost == pytest.approx((200 * 0.002 + 800 * 0.0005) / 1000)
    assert completion_cost == pytest.approx(100 * 0.008 / 1000)


def test_dated_openai_snapshots_carry_their_alias_cached_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI's model pages name the default snapshot and give it the alias's price (gpt-5.5, gpt-4.1)."""
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    cat = PricingCatalog()
    checked = 0

    for model in list_provider_models("openai"):
        dated = re.match(r"^(.*)-(\d{4}-\d{2}-\d{2}|\d{8})$", model)
        if dated is None:
            continue
        alias = dated.group(1)
        snapshot, _s = cat.get_rate_details("openai", model)
        base, _b = cat.get_rate_details("openai", alias)
        if "cache_read" in base and snapshot["prompt"] == base["prompt"]:
            assert snapshot.get("cache_read") == pytest.approx(base["cache_read"]), model
            checked += 1

    assert checked >= 15


def test_o3_pro_is_priced_at_its_own_published_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, _estimated = PricingCatalog().get_rate_details("openai", "o3-pro")

    # developers.openai.com/api/docs/pricing: o3-pro input $20, output $80 per MTok, no cached input.
    assert got == pytest.approx({"prompt": 0.02, "completion": 0.08})


def test_compute_costs_prices_one_hour_cache_writes_at_the_one_hour_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    prompt_cost, _completion, _total, _est = compute_costs(
        "anthropic",
        "claude-opus-5",
        3000,
        0,
        cache_write_input_tokens=3000,
        cache_write_1h_input_tokens=2000,
        billable_input_tokens=0,
    )

    # Opus 5 per 1K: 5-minute write $0.00625 (1.25x), 1-hour write $0.010 (2x).
    assert prompt_cost == pytest.approx((1000 * 0.00625 + 2000 * 0.010) / 1000)


def test_one_hour_writes_fall_back_to_the_five_minute_rate_when_the_catalog_has_no_one_hour_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = {"anthropic": {"claude-test": {"prompt": 0.010, "completion": 0.030, "cache_write": 0.0125}}}
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps(overrides))
    reset_pricing_catalog()

    prompt_cost, _c, _t, _e = compute_costs(
        "anthropic", "claude-test", 100, 0, cache_write_input_tokens=100, cache_write_1h_input_tokens=100, billable_input_tokens=0
    )

    assert prompt_cost == pytest.approx(100 * 0.0125 / 1000)


def test_every_anthropic_cache_write_rate_has_a_one_hour_rate_at_twice_the_input_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    # list_provider_models reads the global catalog; an earlier test's override must not leak into it (review M9).
    reset_pricing_catalog()
    cat = PricingCatalog()

    for model in list_provider_models("anthropic"):
        rates, _estimated = cat.get_rate_details("anthropic", model)
        if "cache_write" in rates:
            # platform.claude.com/docs/en/about-claude/pricing: every 1h write is 2x the row's input price.
            assert rates["cache_write_1h"] == pytest.approx(2 * rates["prompt"]), model
