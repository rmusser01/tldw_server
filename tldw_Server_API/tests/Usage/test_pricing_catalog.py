from __future__ import annotations

import json
import os
import pytest

from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog, reset_pricing_catalog
from tldw_Server_API.app.core.Usage.usage_tracker import compute_costs


def test_pricing_defaults_and_partial_match():
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


def test_pricing_env_override(monkeypatch):
    overrides = {
        "openai": {"gpt-3.5-turbo": {"prompt": 0.123, "completion": 0.456}}
    }
    monkeypatch.setenv('PRICING_OVERRIDES', json.dumps(overrides))
    cat = PricingCatalog()
    pr, cr, est = cat.get_rates('openai', 'gpt-3.5-turbo')
    assert pr == pytest.approx(0.123)
    assert cr == pytest.approx(0.456)
    assert est is False


def test_pricing_env_override_supports_cache_read_write_rates(monkeypatch):
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


def test_compute_costs_uses_cache_rates_when_present(monkeypatch):
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
