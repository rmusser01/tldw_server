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
