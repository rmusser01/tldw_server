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
        ("claude-opus-4-20250514", _OPUS_4_1),
        ("claude-opus-4-6", _OPUS_5),
        ("claude-sonnet-5-5", _SONNET_5_5),
        ("claude-sonnet-5", _SONNET_5),
        ("claude-sonnet-4-6", _SONNET_4),
        ("claude-sonnet-4-5-20250929", _SONNET_4),
        ("claude-sonnet-4-20250514", _SONNET_4),
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


# Haiku 5.5 per 1K tokens by prompt length. Source: platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-22.
_HAIKU_5_5_LOW = {"prompt": 0.0001, "completion": 0.0005, "cache_read": 0.00001, "cache_write": 0.000125, "cache_write_1h": 0.0002}
_HAIKU_5_5_HIGH = {"prompt": 0.0005, "completion": 0.0025, "cache_read": 0.00005, "cache_write": 0.000625, "cache_write_1h": 0.001}


@pytest.mark.parametrize(
    ("prompt_tokens", "rates"),
    [(0, _HAIKU_5_5_LOW), (1_000, _HAIKU_5_5_LOW), (100_000, _HAIKU_5_5_LOW), (100_001, _HAIKU_5_5_HIGH), (900_000, _HAIKU_5_5_HIGH)],
)
def test_haiku_5_5_is_priced_by_prompt_length(prompt_tokens: int, rates: dict[str, float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, estimated = PricingCatalog().get_rate_details("anthropic", "claude-haiku-5-5", prompt_tokens=prompt_tokens)

    assert got == pytest.approx(rates)
    assert estimated is False


def test_compute_costs_picks_the_haiku_5_5_tier_from_the_whole_prompt_including_cache_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    # 40,000 uncached + 50,000 cache read + 10,001 cache write = 100,001 prompt tokens: the high tier.
    prompt_cost, completion_cost, _t, _e = compute_costs(
        "anthropic", "claude-haiku-5-5", 100_001, 1000,
        cache_read_input_tokens=50_000, cache_write_input_tokens=10_001, billable_input_tokens=40_000,
    )
    low_prompt_cost, _lc, _lt, _le = compute_costs(
        "anthropic", "claude-haiku-5-5", 100_000, 0,
        cache_read_input_tokens=50_000, cache_write_input_tokens=10_000, billable_input_tokens=40_000,
    )

    assert prompt_cost == pytest.approx((40_000 * 0.0005 + 50_000 * 0.00005 + 10_001 * 0.000625) / 1000)
    assert completion_cost == pytest.approx(1000 * 0.0025 / 1000)
    assert low_prompt_cost == pytest.approx((40_000 * 0.0001 + 50_000 * 0.00001 + 10_000 * 0.000125) / 1000)


@pytest.mark.parametrize(
    ("model", "rates"),
    [
        ("claude-mythos-5-1", _FABLE_5_1),
        ("claude-mythos-5", _FABLE_5),
        ("claude-3-5-haiku-20241022", {"prompt": 0.0008, "completion": 0.004, "cache_read": 0.00008, "cache_write": 0.001, "cache_write_1h": 0.0016}),
        ("claude-opus-4-1-20250805", _OPUS_4_1),
        ("claude-opus-4-20250514", _OPUS_4_1),
        ("claude-sonnet-4-5-20250929", _SONNET_4),
        ("claude-sonnet-4-20250514", _SONNET_4),
        ("claude-opus-4-7", _OPUS_5),
        ("claude-sonnet-4-6", _SONNET_4),
    ],
)
def test_remaining_anthropic_api_ids_get_their_list_prices(model: str, rates: dict[str, float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, _estimated = PricingCatalog().get_rate_details("anthropic", model)

    assert got == pytest.approx(rates)


def test_listed_anthropic_models_have_no_dotted_names_and_no_invalid_dated_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks the shape of the picker's Anthropic list, not that every key is an API ID.

    claude-opus-4 and claude-sonnet-4 stay as substring price keys (the task's own note allows it) and
    claude-3-5-sonnet-latest stays until follow-up 1; neither is a documented ID, and neither has a dot or a date.
    """
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    # Dated IDs the API accepts (platform.claude.com/docs/en/about-claude/model-deprecations).
    dated_ids = {
        "claude-haiku-4-5-20251001",
        "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    }

    for model in list_provider_models("anthropic"):
        assert "." not in model, f"{model} is dotted; no current Anthropic API ID has a dot"
        if re.search(r"-\d{8}$", model):
            assert model in dated_ids, f"{model} is not an API ID"
        # The 4.6 generation and later have dateless IDs only.
        assert not re.search(r"claude-(opus|sonnet|haiku|fable|mythos)-(4-[6-9]|5(-\d)?)-\d{8}$", model), model


@pytest.mark.parametrize(
    ("dotted", "ids_rates"),
    [
        ("claude-haiku-4.5", _HAIKU_4_5),
        ("claude-sonnet-4.5", _SONNET_4),
        ("claude-opus-4.1", _OPUS_4_1),
        ("claude-opus-4.5", _OPUS_5),
    ],
)
def test_a_dotted_anthropic_model_name_is_still_priced_as_its_api_id(dotted: str, ids_rates: dict[str, float], monkeypatch: pytest.MonkeyPatch) -> None:
    """The dotted names stay in config defaults and the model router; they must not fall to the unknown-model rate."""
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    got, _estimated = PricingCatalog().get_rate_details("anthropic", dotted)

    assert got == pytest.approx(ids_rates)


# OpenAI long-context prices per 1K tokens: prompts over 272,000 tokens.
# Source: developers.openai.com/api/docs/pricing, "Long context" columns, fetched 2026-10-22 (TASK-13879).
_OPENAI_LONG = {
    "gpt-5.5": {"prompt": 0.01, "completion": 0.045, "cache_read": 0.001},
    "gpt-5.5-2026-04-23": {"prompt": 0.01, "completion": 0.045, "cache_read": 0.001},
    "gpt-5.4": {"prompt": 0.005, "completion": 0.0225, "cache_read": 0.0005},
    "gpt-5.4-2026-03-05": {"prompt": 0.005, "completion": 0.0225, "cache_read": 0.0005},
    "gpt-5.5-pro": {"prompt": 0.06, "completion": 0.27},
    "gpt-5.5-pro-2026-04-23": {"prompt": 0.06, "completion": 0.27},
    "gpt-5.4-pro": {"prompt": 0.06, "completion": 0.27},
    "gpt-5.4-pro-2026-03-05": {"prompt": 0.06, "completion": 0.27},
}
_OPENAI_SHORT = {
    "gpt-5.5": {"prompt": 0.005, "completion": 0.03, "cache_read": 0.0005},
    "gpt-5.5-2026-04-23": {"prompt": 0.005, "completion": 0.03, "cache_read": 0.0005},
    "gpt-5.4": {"prompt": 0.0025, "completion": 0.015, "cache_read": 0.00025},
    "gpt-5.4-2026-03-05": {"prompt": 0.0025, "completion": 0.015, "cache_read": 0.00025},
    "gpt-5.5-pro": {"prompt": 0.03, "completion": 0.18},
    "gpt-5.5-pro-2026-04-23": {"prompt": 0.03, "completion": 0.18},
    "gpt-5.4-pro": {"prompt": 0.03, "completion": 0.18},
    "gpt-5.4-pro-2026-03-05": {"prompt": 0.03, "completion": 0.18},
}


@pytest.mark.parametrize("model", sorted(_OPENAI_LONG))
def test_openai_long_context_prompts_pay_the_long_context_rates(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    cat = PricingCatalog()

    at_limit, _e1 = cat.get_rate_details("openai", model, prompt_tokens=272_000)
    over_limit, _e2 = cat.get_rate_details("openai", model, prompt_tokens=272_001)

    assert at_limit == pytest.approx(_OPENAI_SHORT[model])
    assert over_limit == pytest.approx(_OPENAI_LONG[model])


def test_compute_costs_bills_a_long_openai_prompt_whole_at_the_long_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    # 300,000 prompt tokens, 100,000 of them cached: the whole request pays gpt-5.5's long-context prices.
    prompt_cost, completion_cost, _t, _e = compute_costs(
        "openai", "gpt-5.5", 300_000, 1_000, cache_read_input_tokens=100_000, billable_input_tokens=200_000
    )

    assert prompt_cost == pytest.approx((200_000 * 0.01 + 100_000 * 0.001) / 1000)
    assert completion_cost == pytest.approx(1_000 * 0.045 / 1000)


@pytest.mark.parametrize("model", ["claude-mythos-5-1", "claude-mythos-5"])
def test_hidden_models_are_priced_but_not_listed(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    got, estimated = PricingCatalog().get_rate_details("anthropic", model)

    assert got["prompt"] == pytest.approx(0.01)
    assert got["completion"] == pytest.approx(0.05)
    assert estimated is False
    assert model not in list_provider_models("anthropic")


def test_the_hidden_flag_survives_an_override_that_repeats_it(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {"anthropic": {"claude-test-hidden": {"prompt": 0.01, "completion": 0.02, "hidden": True}}}
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps(overrides))
    reset_pricing_catalog()

    got, _estimated = PricingCatalog().get_rate_details("anthropic", "claude-test-hidden")

    assert got["prompt"] == pytest.approx(0.01)
    assert "claude-test-hidden" not in list_provider_models("anthropic")
    monkeypatch.delenv("PRICING_OVERRIDES")
    reset_pricing_catalog()


def test_fast_mode_costs_twice_list_price_on_every_token_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    args = {"cache_read_input_tokens": 40_000, "cache_write_input_tokens": 10_000, "billable_input_tokens": 50_000}

    list_prompt, list_completion, _t, _e = compute_costs("anthropic", "claude-opus-5", 100_000, 2_000, **args)
    fast_prompt, fast_completion, fast_total, _e2 = compute_costs("anthropic", "claude-opus-5", 100_000, 2_000, speed="fast", **args)

    assert fast_prompt == pytest.approx(2 * list_prompt)
    assert fast_completion == pytest.approx(2 * list_completion)
    assert fast_total == pytest.approx(fast_prompt + fast_completion)


def test_fast_mode_input_and_output_match_the_published_fast_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    # platform.claude.com/docs/en/about-claude/pricing, "Fast mode pricing": Opus 5.5 $8/$40, Opus 5 and 4.8 $10/$50 per MTok.
    for model, in_per_mtok, out_per_mtok in [("claude-opus-5-5", 8, 40), ("claude-opus-5", 10, 50), ("claude-opus-4-8", 10, 50)]:
        prompt_cost, completion_cost, _t, _e = compute_costs("anthropic", model, 1_000_000, 1_000_000, speed="fast", billable_input_tokens=1_000_000)
        assert prompt_cost == pytest.approx(in_per_mtok), model
        assert completion_cost == pytest.approx(out_per_mtok), model


def test_us_only_inference_costs_one_point_one_times_on_every_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    args = {
        "cache_read_input_tokens": 40_000,
        "cache_write_input_tokens": 10_000,
        "cache_write_1h_input_tokens": 4_000,
        "billable_input_tokens": 50_000,
    }

    list_prompt, list_completion, _t, _e = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, **args)
    us_prompt, us_completion, _t2, _e2 = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, inference_geo="us", **args)
    global_prompt, global_completion, _t3, _e3 = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, inference_geo="global", **args)

    assert us_prompt == pytest.approx(1.1 * list_prompt)
    assert us_completion == pytest.approx(1.1 * list_completion)
    assert (global_prompt, global_completion) == (list_prompt, list_completion)


def test_fast_mode_and_us_only_inference_stack_with_cache_rates_and_the_haiku_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    fast_us, _c, _t, _e = compute_costs(
        "anthropic", "claude-opus-5", 10_000, 0,
        cache_write_input_tokens=10_000, cache_write_1h_input_tokens=10_000, billable_input_tokens=0, speed="fast", inference_geo="us",
    )
    # Opus 5 1-hour write $0.010 per 1K, x2 fast, x1.1 US-only.
    assert fast_us == pytest.approx(10_000 * 0.010 * 2 * 1.1 / 1000)

    haiku_us, _c2, _t2, _e2 = compute_costs("anthropic", "claude-haiku-5-5", 100_001, 0, billable_input_tokens=100_001, inference_geo="us")
    # Haiku 5.5 high tier $0.0005 per 1K, x1.1 (the page: the multiplier applies to the higher prices too).
    assert haiku_us == pytest.approx(100_001 * 0.0005 * 1.1 / 1000)


@pytest.mark.parametrize(
    "tier",
    [
        {"above_tokens": 200_000, "completion": 0.01},
        {"above_tokens": 200_000, "prompt": 0.01},
        {"prompt": 0.02, "completion": 0.06},
        {"above_tokens": 0, "prompt": 0.02, "completion": 0.06},
    ],
    ids=["no-prompt", "no-completion", "no-above-tokens", "zero-above-tokens"],
)
def test_an_incomplete_long_prompt_tier_is_ignored_with_a_warning(tier: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tier without a prompt or completion rate would bill that side at $0; one without a threshold would apply
    to every request. Such a tier is dropped and the base rates stay (review M2)."""
    from loguru import logger

    overrides = {"openai": {"tiered-test": {"prompt": 0.01, "completion": 0.03, "long_prompt": tier}}}
    monkeypatch.setenv("PRICING_OVERRIDES", json.dumps(overrides))
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        cat = PricingCatalog()
    finally:
        logger.remove(sink)

    for prompt_tokens in (1, 300_000):
        got, _estimated = cat.get_rate_details("openai", "tiered-test", prompt_tokens=prompt_tokens)
        assert got == pytest.approx({"prompt": 0.01, "completion": 0.03}), prompt_tokens
    assert any("long_prompt" in line and "tiered-test" in line for line in warnings), warnings


# platform.claude.com/docs/en/manage-claude/data-residency, fetched 2026-10-23: "The `inference_geo` parameter is
# supported on Claude 4.6 and later models. Requests with `inference_geo` on Claude Opus 4.5, Claude Sonnet 4.5
# (deprecated), Claude Haiku 4.5, or earlier models return a 400 error." Pricing: "Older models: ... Standard
# pricing applies." "Legacy models are unaffected by this migration."
@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-5-20251101",
        "claude-opus-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4.5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
    ],
)
def test_us_only_inference_is_not_charged_on_models_before_claude_4_6(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    standard = compute_costs("anthropic", model, 10_000, 1_000, billable_input_tokens=10_000)
    reported_us = compute_costs("anthropic", model, 10_000, 1_000, billable_input_tokens=10_000, inference_geo="us")

    assert reported_us[:3] == pytest.approx(standard[:3])


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-8", "claude-opus-5", "claude-haiku-5-5", "claude-mythos-5-1", "claude-fable-5"],
)
def test_us_only_inference_is_charged_on_claude_4_6_and_later(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()

    standard = compute_costs("anthropic", model, 10_000, 1_000, billable_input_tokens=10_000)
    reported_us = compute_costs("anthropic", model, 10_000, 1_000, billable_input_tokens=10_000, inference_geo="us")

    assert reported_us[0] == pytest.approx(1.1 * standard[0])
    assert reported_us[1] == pytest.approx(1.1 * standard[1])
