"""Service tiers, Gemini cache reads and OpenRouter's reported cost are priced (TASK-13936)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from tldw_Server_API.app.core.Usage import llm_usage_normalizer, usage_tracker
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog, list_provider_models, reset_pricing_catalog
from tldw_Server_API.app.core.Usage.usage_tracker import compute_costs

pytestmark = pytest.mark.unit

# Vendor pages give prices per million tokens; catalog rates are per thousand.
MTOK = 1000.0

# developers.openai.com/api/docs/pricing, "Batch", "Flex" and "Fast" text-token tables, fetched 2026-10-26.
# (input, output, cached input) per million tokens, short context; None where the page shows "-".
# The page renamed Priority to Fast on July 30, 2026. The Fast-mode guide (guides/priority-processing): "For GPT-5.6
# and earlier models, the response returns `priority` whether the request specifies `priority` or `fast`."
OPENAI_TIER_PRICES: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "gpt-5.5": {"batch": (2.50, 15.00, 0.25), "flex": (2.50, 15.00, 0.25), "priority": (12.50, 75.00, 1.25)},
    "gpt-5.5-pro": {"batch": (15.00, 90.00, None), "flex": (15.00, 90.00, None)},
    "gpt-5.4": {"batch": (1.25, 7.50, 0.13), "flex": (1.25, 7.50, 0.13), "priority": (5.00, 30.00, 0.50)},
    "gpt-5.4-pro": {"batch": (15.00, 90.00, None), "flex": (15.00, 90.00, None)},
    "gpt-5.4-mini": {"batch": (0.375, 2.25, 0.0375), "flex": (0.375, 2.25, 0.0375), "priority": (1.50, 9.00, 0.15)},
    "gpt-5.4-nano": {"batch": (0.10, 0.625, 0.01), "flex": (0.10, 0.625, 0.01)},
    "gpt-5.2": {"batch": (0.875, 7.00, 0.0875), "flex": (0.875, 7.00, 0.0875), "priority": (3.50, 28.00, 0.35)},
    "gpt-5.2-pro": {"batch": (10.50, 84.00, None)},
    "gpt-5.1": {"batch": (0.625, 5.00, 0.0625), "flex": (0.625, 5.00, 0.0625), "priority": (2.50, 20.00, 0.25)},
    "gpt-5": {"batch": (0.625, 5.00, 0.0625), "flex": (0.625, 5.00, 0.0625), "priority": (2.50, 20.00, 0.25)},
    "gpt-5-mini": {"batch": (0.125, 1.00, 0.0125), "flex": (0.125, 1.00, 0.0125), "priority": (0.45, 3.60, 0.045)},
    "gpt-5-nano": {"batch": (0.025, 0.20, 0.0025), "flex": (0.025, 0.20, 0.0025)},
    "gpt-5-pro": {"batch": (7.50, 60.00, None)},
    "gpt-4.1": {"batch": (1.00, 4.00, None), "priority": (3.50, 14.00, 0.875)},
    "gpt-4.1-mini": {"batch": (0.20, 0.80, None), "priority": (0.70, 2.80, 0.175)},
    "gpt-4.1-nano": {"batch": (0.05, 0.20, None), "priority": (0.20, 0.80, 0.05)},
    "gpt-4o": {"batch": (1.25, 5.00, None), "priority": (4.25, 17.00, 2.125)},
    "gpt-4o-2024-05-13": {"batch": (2.50, 7.50, None), "priority": (8.75, 26.25, None)},
    "gpt-4o-mini": {"batch": (0.075, 0.30, None), "priority": (0.25, 1.00, 0.125)},
    "o1": {"batch": (7.50, 30.00, None)},
    "o1-pro": {"batch": (75.00, 300.00, None)},
    "o3-pro": {"batch": (10.00, 40.00, None)},
    "o3": {"batch": (1.00, 4.00, None), "flex": (1.00, 4.00, 0.25), "priority": (3.50, 14.00, 0.875)},
    "o4-mini": {"batch": (0.55, 2.20, None), "flex": (0.55, 2.20, 0.138), "priority": (2.00, 8.00, 0.50)},
    "o3-mini": {"batch": (0.55, 2.20, None)},
    "gpt-4-turbo-2024-04-09": {"batch": (5.00, 15.00, None)},
    "gpt-4-0613": {"batch": (15.00, 30.00, None)},
    "gpt-3.5-turbo-0125": {"batch": (0.25, 0.75, None)},
}
# The long-context columns (prompts over 272,000 tokens) of the same tables. Where the page shows "-" there
# (gpt-5.5 and gpt-5.4 Fast, gpt-5.5-pro Batch and Flex), the tier has no long-context price.
OPENAI_LONG_TIER_PRICES: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "gpt-5.5": {"batch": (5.00, 22.50, 0.50), "flex": (5.00, 22.50, 0.50)},
    "gpt-5.4": {"batch": (2.50, 11.25, 0.25), "flex": (2.50, 11.25, 0.25)},
    "gpt-5.4-pro": {"batch": (30.00, 135.00, None), "flex": (30.00, 135.00, None)},
}


def _rates(prices: tuple[float, float, float | None]) -> dict[str, float]:
    """Catalog rates (per 1K tokens) for a page row (per 1M tokens)."""
    prompt, completion, cached = prices
    rates = {"prompt": prompt / MTOK, "completion": completion / MTOK}
    if cached is not None:
        rates["cache_read"] = cached / MTOK
    return rates


@pytest.fixture(autouse=True)
def _shipped_catalog(monkeypatch: pytest.MonkeyPatch):
    """Price with the shipped catalog only."""
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    yield
    reset_pricing_catalog()


@pytest.fixture
def costed_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The llm_usage_log rows log_llm_usage would insert, costed by the shipped catalog."""
    captured: list[dict[str, Any]] = []

    class _Repo:
        """Stands in for AuthnzUsageRepo and keeps the inserted rows."""

        def __init__(self, _pool: Any) -> None:
            """No database."""

        async def insert_llm_usage_log(self, **kwargs: Any) -> None:
            """Record the row instead of writing it."""
            captured.append(kwargs)

    async def _none() -> None:
        """No pool and no daily ledger."""

    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
    monkeypatch.setattr(usage_tracker, "get_db_pool", _none)
    monkeypatch.setattr(usage_tracker, "_get_tokens_daily_ledger", _none)
    monkeypatch.setattr(usage_tracker, "AuthnzUsageRepo", _Repo)
    return captured


async def _log(provider: str, model: str, usage: dict[str, Any], prompt_tokens: int, completion_tokens: int) -> None:
    """Record one call the way the provider-level recorder does."""
    await usage_tracker.log_llm_usage(
        user_id=5,
        key_id=None,
        endpoint="internal",
        operation="llm_call",
        provider=provider,
        model=model,
        status=200,
        latency_ms=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_metadata=usage,
    )


# --- OpenAI service tiers (AC #5) and unpriced tiers (AC #6) ---


@pytest.mark.parametrize(
    ("model", "tier", "prices"),
    sorted((model, tier, prices) for model, tiers in OPENAI_TIER_PRICES.items() for tier, prices in tiers.items()),
)
def test_an_openai_service_tier_is_priced_at_its_published_rate(model: str, tier: str, prices: tuple) -> None:
    got, _estimated = PricingCatalog().get_rate_details("openai", model, service_tier=tier)

    assert got == pytest.approx(_rates(prices))


@pytest.mark.parametrize(
    ("model", "tier", "prices"),
    sorted((model, tier, prices) for model, tiers in OPENAI_LONG_TIER_PRICES.items() for tier, prices in tiers.items()),
)
def test_an_openai_service_tier_prices_a_prompt_over_272k_tokens_at_its_long_context_rate(
    model: str, tier: str, prices: tuple
) -> None:
    cat = PricingCatalog()

    at_limit, _e1 = cat.get_rate_details("openai", model, prompt_tokens=272_000, service_tier=tier)
    over_limit, _e2 = cat.get_rate_details("openai", model, prompt_tokens=272_001, service_tier=tier)

    assert at_limit == pytest.approx(_rates(OPENAI_TIER_PRICES[model][tier]))
    assert over_limit == pytest.approx(_rates(prices))


@pytest.mark.parametrize(
    ("model", "tier"),
    [("gpt-5.5", "priority"), ("gpt-5.4", "priority"), ("gpt-5.5-pro", "batch"), ("gpt-5.5-pro", "flex")],
)
def test_a_long_prompt_on_a_tier_with_no_long_context_price_costs_the_standard_long_context_rate(
    model: str, tier: str
) -> None:
    """The page shows "-" for these tiers' long-context columns: no price, so the standard one (AC #6)."""
    cat = PricingCatalog()

    standard_long, _e1 = cat.get_rate_details("openai", model, prompt_tokens=272_001)
    tier_long, _e2 = cat.get_rate_details("openai", model, prompt_tokens=272_001, service_tier=tier)

    assert tier_long == pytest.approx(standard_long)


def test_dated_openai_snapshots_carry_their_alias_service_tiers() -> None:
    """A snapshot priced like its alias (2026-10-23 pricing plan) takes the alias's tier rates too."""
    cat = PricingCatalog()
    checked = 0

    for model in list_provider_models("openai"):
        dated = re.match(r"^(.*)-(\d{4}-\d{2}-\d{2})$", model)
        if dated is None or dated.group(1) not in OPENAI_TIER_PRICES:
            continue
        alias = dated.group(1)
        if cat.get_rate_details("openai", model)[0] != cat.get_rate_details("openai", alias)[0]:
            continue  # priced on its own row (gpt-4o-2024-05-13)
        for tier in OPENAI_TIER_PRICES[alias]:
            got = cat.get_rate_details("openai", model, service_tier=tier)[0]
            assert got == pytest.approx(cat.get_rate_details("openai", alias, service_tier=tier)[0]), (model, tier)
        checked += 1

    assert checked >= 23  # the 23 snapshots catalogued on 2026-10-26; a new one is checked by the loop above


@pytest.mark.parametrize("tier", ["default", "scale", "auto", "standard", "", "ultrafast", None])
def test_a_service_tier_the_catalog_has_no_rate_for_costs_the_standard_rate(tier: str | None) -> None:
    args = {"cache_read_input_tokens": 400, "billable_input_tokens": 600}

    standard = compute_costs("openai", "gpt-5-mini", 1000, 100, **args)

    assert compute_costs("openai", "gpt-5-mini", 1000, 100, service_tier=tier, **args) == standard


def test_a_tier_a_model_has_no_row_for_costs_the_standard_rate() -> None:
    """gpt-4.1 has Batch and Fast rows but no Flex row; Anthropic publishes no Priority Tier token price."""
    assert compute_costs("openai", "gpt-4.1", 1000, 100, service_tier="flex") == compute_costs(
        "openai", "gpt-4.1", 1000, 100
    )
    assert compute_costs("anthropic", "claude-sonnet-4-6", 1000, 100, service_tier="priority") == compute_costs(
        "anthropic", "claude-sonnet-4-6", 1000, 100
    )


def test_a_tier_with_no_cached_input_price_costs_cached_tokens_at_the_tier_input_rate() -> None:
    """gpt-4.1 Batch shows "-" for cached input: every prompt token pays the Batch input rate."""
    prompt_cost, completion_cost, _t, _e = compute_costs(
        "openai", "gpt-4.1", 1000, 100, cache_read_input_tokens=800, billable_input_tokens=200, service_tier="batch"
    )

    assert prompt_cost == pytest.approx(1000 * 1.00 / 1_000_000)
    assert completion_cost == pytest.approx(100 * 4.00 / 1_000_000)


def test_the_response_service_tier_is_folded_into_a_copy_of_its_usage_block() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    response = {"object": "chat.completion", "service_tier": "flex", "usage": usage}

    assert llm_usage_normalizer.usage_with_service_tier(response) == {**usage, "service_tier": "flex"}
    assert "service_tier" not in usage
    assert llm_usage_normalizer.usage_with_service_tier({"usage": usage}) is usage
    assert llm_usage_normalizer.usage_with_service_tier({"service_tier": "flex"}) is None
    assert llm_usage_normalizer.usage_with_service_tier(
        {"usage": {"service_tier": "batch"}, "service_tier": "flex"}
    ) == {"service_tier": "batch"}
    assert llm_usage_normalizer.usage_with_service_tier("not a response") is None


@pytest.mark.parametrize(
    ("tier", "prices"),
    [
        ("flex", (0.125, 1.00, 0.0125)),
        ("priority", (0.45, 3.60, 0.045)),
        ("batch", (0.125, 1.00, 0.0125)),
        ("default", (0.25, 2.00, 0.025)),
    ],
)
async def test_an_openai_call_is_costed_at_the_service_tier_its_response_reports(
    costed_rows: list[dict[str, Any]], tier: str, prices: tuple[float, float, float]
) -> None:
    response = {
        "object": "chat.completion",
        "service_tier": tier,
        "usage": {
            "prompt_tokens": 10_000,
            "completion_tokens": 1_000,
            "total_tokens": 11_000,
            "prompt_tokens_details": {"cached_tokens": 4_000},
        },
    }

    await _log("openai", "gpt-5-mini", llm_usage_normalizer.usage_with_service_tier(response), 10_000, 1_000)

    (row,) = costed_rows
    in_mtok, out_mtok, cached_mtok = prices
    assert row["prompt_cost_usd"] == pytest.approx((6_000 * in_mtok + 4_000 * cached_mtok) / 1_000_000)
    assert row["completion_cost_usd"] == pytest.approx(1_000 * out_mtok / 1_000_000)
    assert json.loads(row["raw_usage_metadata_json"])["service_tier"] == tier


def test_the_reported_service_tier_is_read_case_insensitively() -> None:
    normalized = normalize_llm_usage(provider="openai", usage={"prompt_tokens": 5, "service_tier": " Flex "})

    assert normalized.service_tier == "flex"


# --- Anthropic Batch (AC #4) ---

# platform.claude.com/docs/en/about-claude/pricing, "Batch processing", fetched 2026-10-26:
# "a 50 percent discount on both input and output tokens". (batch input, batch output) per million tokens.
ANTHROPIC_BATCH_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (5.00, 25.00),
    "claude-mythos-5-1": (5.00, 25.00),
    "claude-fable-5": (5.00, 25.00),
    "claude-mythos-5": (5.00, 25.00),
    "claude-opus-5-5": (2.00, 10.00),
    "claude-opus-5": (2.50, 12.50),
    "claude-opus-4-8": (2.50, 12.50),
    "claude-opus-4-7": (2.50, 12.50),
    "claude-opus-4-6": (2.50, 12.50),
    "claude-opus-4-5": (2.50, 12.50),
    "claude-opus-4-1": (7.50, 37.50),
    "claude-opus-4": (7.50, 37.50),
    "claude-sonnet-5-5": (1.00, 5.00),
    "claude-sonnet-5": (1.00, 5.00),
    "claude-sonnet-4-6": (1.50, 7.50),
    "claude-sonnet-4-5": (1.50, 7.50),
    "claude-sonnet-4": (1.50, 7.50),
    "claude-haiku-4-5": (0.50, 2.50),
    "claude-3-5-haiku-20241022": (0.40, 2.00),
}


@pytest.mark.parametrize(("model", "prices"), sorted(ANTHROPIC_BATCH_PRICES.items()))
def test_an_anthropic_batch_call_costs_the_published_batch_price(model: str, prices: tuple[float, float]) -> None:
    prompt_cost, completion_cost, _t, _e = compute_costs(
        "anthropic", model, 1_000_000, 1_000_000, billable_input_tokens=1_000_000, service_tier="batch"
    )

    assert (prompt_cost, completion_cost) == pytest.approx(prices)


def test_an_anthropic_batch_call_on_haiku_5_5_halves_both_prompt_length_tiers() -> None:
    """Page rows: Haiku 5.5 batch input $0.05 up to 100,000 prompt tokens, $0.25 over."""
    low = compute_costs(
        "anthropic", "claude-haiku-5-5", 100_000, 0, billable_input_tokens=100_000, service_tier="batch"
    )
    high = compute_costs(
        "anthropic", "claude-haiku-5-5", 100_001, 0, billable_input_tokens=100_001, service_tier="batch"
    )

    assert low[0] == pytest.approx(100_000 * 0.05 / 1_000_000)
    assert high[0] == pytest.approx(100_001 * 0.25 / 1_000_000)


def test_the_anthropic_batch_discount_stacks_with_the_cache_rates_and_us_only_inference() -> None:
    """Cache multipliers "stack with other pricing modifiers, including the Batch API discount and data residency"."""
    args = {
        "cache_read_input_tokens": 40_000,
        "cache_write_input_tokens": 10_000,
        "cache_write_1h_input_tokens": 4_000,
        "billable_input_tokens": 50_000,
        "inference_geo": "us",
    }

    standard = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, **args)
    batch = compute_costs("anthropic", "claude-sonnet-4-6", 100_000, 2_000, service_tier="batch", **args)

    assert batch[0] == pytest.approx(standard[0] * 0.5)
    assert batch[1] == pytest.approx(standard[1] * 0.5)


def test_a_provider_without_a_batch_price_is_not_discounted() -> None:
    assert compute_costs("mistral", "mistral-large-2512", 1000, 100, service_tier="batch") == compute_costs(
        "mistral", "mistral-large-2512", 1000, 100
    )


# --- Gemini cached content (AC #1) ---

# ai.google.dev/gemini-api/docs/pricing, Standard tier, paid, text/image/video, fetched 2026-10-26:
# (input, output, context caching) per million tokens, prompts up to 200k tokens.
GEMINI_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3-flash-preview": (0.50, 3.00, 0.05),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
}
# The same rows for prompts over 200k tokens.
GEMINI_LONG_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-3.1-pro-preview": (4.00, 18.00, 0.40),
    "gemini-2.5-pro": (2.50, 15.00, 0.25),
}
# Catalogued -preview entries the page does not list, priced exactly as their GA model, so they take its cache rate.
# The 2.0 Flash entries are off the page too, and Google publishes no cache rate for them, so they get none.
GEMINI_PREVIEW_ALIASES: dict[str, str] = {
    "gemini-2.5-flash-preview": "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025": "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite",
}


@pytest.mark.parametrize(("model", "prices"), sorted(GEMINI_PRICES.items()))
def test_a_gemini_model_carries_its_published_context_caching_rate(
    model: str, prices: tuple[float, float, float]
) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model, prompt_tokens=200_000)

    assert got == pytest.approx(_rates(prices))


@pytest.mark.parametrize(("model", "ga_model"), sorted(GEMINI_PREVIEW_ALIASES.items()))
def test_a_gemini_2_5_preview_entry_carries_its_ga_model_context_caching_rate(model: str, ga_model: str) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model)

    assert got == pytest.approx(_rates(GEMINI_PRICES[ga_model]))


@pytest.mark.parametrize(("model", "prices"), sorted(GEMINI_LONG_PRICES.items()))
def test_a_gemini_prompt_over_200k_tokens_pays_the_long_prompt_rates(
    model: str, prices: tuple[float, float, float]
) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model, prompt_tokens=200_001)

    assert got == pytest.approx(_rates(prices))


async def test_a_gemini_reply_costs_its_cached_tokens_at_the_cache_read_rate(costed_rows: list[dict[str, Any]]) -> None:
    """The adapter keeps cachedContentTokenCount, which promptTokenCount includes."""
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    shaped = GoogleAdapter._normalize_to_openai_shape(
        {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 10_000,
                "candidatesTokenCount": 100,
                "totalTokenCount": 10_100,
                "cachedContentTokenCount": 8_000,
            },
        }
    )

    await _log("google", "gemini-2.5-flash", shaped["usage"], 10_000, 100)

    (row,) = costed_rows
    assert row["cache_read_input_tokens"] == 8_000
    # gemini-2.5-flash per 1M: input $0.30, context caching $0.03, output $2.50.
    assert row["prompt_cost_usd"] == pytest.approx((2_000 * 0.30 + 8_000 * 0.03) / 1_000_000)
    assert row["completion_cost_usd"] == pytest.approx(100 * 2.50 / 1_000_000)


# --- OpenRouter's reported cost (AC #2) ---


async def test_an_openrouter_call_is_costed_from_the_cost_openrouter_reports(costed_rows: list[dict[str, Any]]) -> None:
    """An uncatalogued model is not costed at the unknown-model rate when OpenRouter reports the call's cost."""
    usage = {"prompt_tokens": 1_000, "completion_tokens": 200, "total_tokens": 1_200, "cost": 0.0042, "is_byok": False}

    await _log("openrouter", "acme/brand-new-model", usage, 1_000, 200)

    (row,) = costed_rows
    assert row["total_cost_usd"] == pytest.approx(0.0042)
    assert row["prompt_cost_usd"] + row["completion_cost_usd"] == pytest.approx(0.0042)
    assert row["estimated"] is False


async def test_an_openrouter_byok_call_adds_the_upstream_provider_charge(costed_rows: list[dict[str, Any]]) -> None:
    """With BYOK, cost is OpenRouter's fee and cost_details.upstream_inference_cost what the provider charged."""
    usage = {
        "prompt_tokens": 194,
        "completion_tokens": 2,
        "total_tokens": 196,
        "cost": 0.95,
        "is_byok": True,
        "cost_details": {"upstream_inference_cost": 19},
    }

    await _log("openrouter", "openai/gpt-5.5", usage, 194, 2)

    (row,) = costed_rows
    assert row["total_cost_usd"] == pytest.approx(19.95)


@pytest.mark.parametrize("cost", [None, "0.5", -1, float("nan"), float("inf"), True])
async def test_an_openrouter_call_without_a_usable_cost_falls_back_to_the_catalog(
    costed_rows: list[dict[str, Any]], cost: Any
) -> None:
    usage = {"prompt_tokens": 1_000, "completion_tokens": 200, "total_tokens": 1_200, "cost": cost}

    await _log("openrouter", "openai/gpt-5.5", usage, 1_000, 200)

    (row,) = costed_rows
    # The catalogued openrouter/openai/gpt-5.5 rate: $0.005 / $0.03 per 1K.
    assert row["total_cost_usd"] == pytest.approx((1_000 * 0.005 + 200 * 0.03) / 1000)


def test_only_openrouter_usage_reports_a_cost() -> None:
    usage = {"prompt_tokens": 1_000, "completion_tokens": 200, "cost": 0.0042}

    assert normalize_llm_usage(provider="openrouter", usage=usage).reported_cost_usd == pytest.approx(0.0042)
    assert normalize_llm_usage(provider="openai", usage=usage).reported_cost_usd is None


def test_a_reported_cost_is_split_by_the_catalog_shares() -> None:
    prompt_cost, completion_cost, total, estimated = compute_costs(
        "openrouter", "openai/gpt-5.5", 1_000, 200, reported_cost_usd=0.022
    )

    # Catalog: 1,000 x $0.005 + 200 x $0.03 per 1K = $0.005 + $0.006.
    assert (prompt_cost, completion_cost, total) == pytest.approx((0.01, 0.012, 0.022))
    assert estimated is False
