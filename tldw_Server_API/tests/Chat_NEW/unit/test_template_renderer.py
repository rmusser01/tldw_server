from __future__ import annotations

import os
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Templating.template_renderer import (
    TemplateContext,
    TemplateEnv,
    TemplateOptions,
    _clear_template_cache,
    _ENV,
    render,
)
import tldw_Server_API.app.core.Templating.template_renderer as template_renderer

pytestmark = pytest.mark.unit


def test_basic_now_renders_year():


    ctx = TemplateContext(env=TemplateEnv(timezone="UTC"))
    out = render("Today is {{ now('%Y', tz='UTC') }}", ctx)
    assert str(datetime.now(timezone.utc).year) in out


def test_strict_undefined_fallbacks_to_original():


     # Unknown name should not raise; renderer returns original text
    ctx = TemplateContext()
    tpl = "Hello {{ unknown_variable }}"
    out = render(tpl, ctx)
    assert out == tpl


def test_random_gated_off_fallbacks():


     # When random is not allowed, randint is undefined and render should fallback
    ctx = TemplateContext()
    tpl = "Roll: {{ randint(1, 6) }}"
    opts = TemplateOptions(allow_random=False)
    out = render(tpl, ctx, opts)
    assert out == tpl


def test_random_allowed_is_deterministic_with_seed():


    ctx = TemplateContext()
    tpl = "Roll: {{ randint(1, 6) }}"
    opts = TemplateOptions(allow_random=True, random_seed=123)
    out1 = render(tpl, ctx, opts)
    out2 = render(tpl, ctx, opts)
    assert out1 == out2
    assert out1.startswith("Roll: ")


def test_max_output_cap_truncates():


    big = "x" * 3000
    ctx = TemplateContext(extra={"big": big})
    opts = TemplateOptions(max_output_chars=2000)
    out = render("{{ big }}", ctx, opts)
    assert len(out) == 2000
    assert out == "x" * 2000


def test_block_tag_construct_falls_back_to_original():


    ctx = TemplateContext()
    tpl = "{% if true %}x{% endif %}"
    out = render(tpl, ctx)
    assert out == tpl


def test_compiled_template_cache_reuses_compiled_template(monkeypatch):


    _clear_template_cache()
    calls = {"n": 0}
    original_from_string = _ENV.from_string

    def counting_from_string(src: str) -> Any:
        calls["n"] += 1
        return original_from_string(src)

    monkeypatch.setattr(_ENV, "from_string", counting_from_string)

    ctx = TemplateContext(extra={"v": "ok"})
    opts = TemplateOptions(cache_max_entries=16)
    tpl = "{{ v }}"

    out1 = render(tpl, ctx, opts)
    out2 = render(tpl, ctx, opts)

    assert out1 == "ok"
    assert out2 == "ok"
    assert calls["n"] == 1


def test_runtime_arithmetic_errors_fallback_to_original():
    ctx = TemplateContext()
    tpl = "{{ 1 / 0 }}"

    out = render(tpl, ctx, TemplateOptions())

    assert out == tpl


def test_oversized_range_errors_fallback_to_original():
    ctx = TemplateContext()
    tpl = "{{ range(1000000)|list|length }}"

    out = render(tpl, ctx, TemplateOptions())

    assert out == tpl


def test_expensive_string_multiplication_is_rejected_before_rendering():
    ctx = TemplateContext()
    tpl = "{{ 'x' * 100 }}"

    out = render(tpl, ctx, TemplateOptions(max_output_chars=20))

    assert out == tpl


def test_extra_objects_cannot_call_public_methods():
    class Probe:
        def __init__(self) -> None:
            self.hit = False

        def public_method(self) -> str:
            self.hit = True
            return "CALLED"

    probe = Probe()
    tpl = "{{ obj.public_method() }}"

    out = render(tpl, TemplateContext(extra={"obj": probe}), TemplateOptions())

    assert out == tpl
    assert probe.hit is False


def test_default_now_uses_context_timezone(monkeypatch):
    real_datetime = datetime

    class FrozenDateTime:
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            value = real_datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
            if tz is not None:
                return value.astimezone(tz)
            return value

    monkeypatch.setattr(template_renderer, "datetime", FrozenDateTime)
    ctx = TemplateContext(env=TemplateEnv(timezone="America/Los_Angeles"))

    out = render("{{ now('%Y-%m-%d') }}", ctx, TemplateOptions())

    assert out == "2025-12-31"


def test_unused_renderer_api_surface_is_removed():
    option_names = {field.name for field in fields(TemplateOptions)}

    assert "allow_external_calls" not in option_names
    assert not hasattr(template_renderer, "TemplateRenderError")
