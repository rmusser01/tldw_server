from __future__ import annotations

import importlib

import pytest


def _capture_pricing_warnings(module):
    messages: list[str] = []
    sink_id = module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    return messages, sink_id


def test_pricing_overrides_env(monkeypatch):
    """Environment PRICING_OVERRIDES should override default rates and support partial matches.

    Verifies that an override for a provider/model is picked up and that
    requesting a model name with the override as a substring returns the
    same rates but marked as estimated.
    """
    monkeypatch.setenv(
        "PRICING_OVERRIDES",
        '{"OpenAI": {"gpt-xyz-test": {"prompt": 0.123, "completion": 0.456}}}',
    )

    # Import inside test so env is set before class is constructed
    from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()

    # Exact match should not be estimated
    p_in, p_out, est = catalog.get_rates("openai", "gpt-xyz-test")
    assert pytest.approx(p_in, rel=1e-6) == 0.123
    assert pytest.approx(p_out, rel=1e-6) == 0.456
    assert est is False

    # Partial model match should be estimated=True but same rates
    p_in2, p_out2, est2 = catalog.get_rates("openai", "gpt-xyz-test-v2")
    assert pytest.approx(p_in2, rel=1e-6) == 0.123
    assert pytest.approx(p_out2, rel=1e-6) == 0.456
    assert est2 is True


def test_pricing_overrides_env_preserves_estimated_metadata(monkeypatch):
    monkeypatch.setenv(
        "PRICING_OVERRIDES",
        '{"Qwen": {"qwen-new-current": {"prompt": 0.001, "completion": 0.002, "estimated": true}}}',
    )

    from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()

    p_in, p_out, est = catalog.get_rates("qwen", "qwen-new-current")
    assert pytest.approx(p_in, rel=1e-6) == 0.001
    assert pytest.approx(p_out, rel=1e-6) == 0.002
    assert est is True
    assert "qwen-new-current" in catalog._catalog["qwen"]


def test_pricing_overrides_env_parse_warning_is_sanitized(monkeypatch):
    mod = importlib.import_module("tldw_Server_API.app.core.Usage.pricing_catalog")
    monkeypatch.setenv("PRICING_OVERRIDES", "{invalid")
    messages, sink_id = _capture_pricing_warnings(mod)

    try:
        mod.PricingCatalog()
    finally:
        mod.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to parse PRICING_OVERRIDES" in joined
    assert "Expecting property name" not in joined


def test_pricing_overrides_file_warning_is_sanitized(monkeypatch):
    mod = importlib.import_module("tldw_Server_API.app.core.Usage.pricing_catalog")

    class ExplodingPath:
        def __init__(self, *_args):
            pass

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self, self, self, self]

        def __truediv__(self, _path):
            return self

        def exists(self):
            return True

        def read_text(self):
            raise OSError("pricing file exploded at /private/model_pricing.json")

    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    monkeypatch.setattr(mod, "Path", ExplodingPath)
    messages, sink_id = _capture_pricing_warnings(mod)

    try:
        mod.PricingCatalog()
    finally:
        mod.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to load pricing overrides file" in joined
    assert "pricing file exploded" not in joined
    assert "/private/model_pricing.json" not in joined
