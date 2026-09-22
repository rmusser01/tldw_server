"""Regression guard for TASK-13313 -- the native-HTTP kill switch must be reachable.

Every adapter with a native HTTP path gates it on `_use_native_http()`, which reads
`LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>` and returns False for 0/false/no/off. When it is
False the adapter raises a deliberate, documented

    RuntimeError("<Adapter> native HTTP disabled by configuration")

rather than falling back -- see the "If disabled explicitly, raise clear error rather
than falling back" comment in openai_adapter.py.

Four adapters short-circuit that gate on the test environment and so can never reach it
under pytest:

    groq, openrouter:        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
    anthropic, custom_openai: `_use_native_http()` itself opens with
                              `if os.getenv("PYTEST_CURRENT_TEST"): return True`

Since PYTEST_CURRENT_TEST is set for every test, the switch is unreachable in CI: no
test can set it to a false value and observe anything. An operator who sets it in
production gets the RuntimeError on every call, and the suite cannot tell them.

`openai` and `bedrock` already use the bare `if self._use_native_http():` form and are
parametrised here as the controls -- they passed before the fix, which is what makes
this a test of the gating rather than of the adapters.

Note the first term is also dead weight on its own: `_prefer_httpx_in_tests()` is
defined as `bool(os.getenv("PYTEST_CURRENT_TEST"))`, so those two conditions are the
same check written twice.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# (import path, class name, env suffix, message prefix)
_ADAPTERS = [
    ("openai_adapter", "OpenAIAdapter", "OPENAI", "OpenAIAdapter"),
    ("groq_adapter", "GroqAdapter", "GROQ", "GroqAdapter"),
    ("openrouter_adapter", "OpenRouterAdapter", "OPENROUTER", "OpenRouterAdapter"),
    ("anthropic_adapter", "AnthropicAdapter", "ANTHROPIC", "AnthropicAdapter"),
    (
        "custom_openai_adapter",
        "CustomOpenAIAdapter",
        "CUSTOM_OPENAI",
        "CustomOpenAIAdapter",
    ),
]

_FALSE_SPELLINGS = ["0", "false", "no", "off"]


def _adapter(module_name: str, class_name: str):
    import importlib

    module = importlib.import_module(
        f"tldw_Server_API.app.core.LLM_Calls.providers.{module_name}"
    )
    return getattr(module, class_name)()


def _request() -> dict:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "some-model",
        "api_key": "test-key",
    }


@pytest.mark.parametrize(
    ("module_name", "class_name", "env_suffix", "message_prefix"), _ADAPTERS
)
def test_disabling_native_http_is_observable(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    env_suffix: str,
    message_prefix: str,
) -> None:
    monkeypatch.setenv(f"LLM_ADAPTERS_NATIVE_HTTP_{env_suffix}", "0")
    adapter = _adapter(module_name, class_name)

    with pytest.raises(RuntimeError) as excinfo:
        adapter.chat(_request())

    assert "native HTTP disabled by configuration" in str(excinfo.value), (
        f"{class_name}.chat did not reach its disabled branch with "
        f"LLM_ADAPTERS_NATIVE_HTTP_{env_suffix}=0. The gate is short-circuited on "
        "PYTEST_CURRENT_TEST, so the kill switch cannot be exercised by any test and "
        "an operator who sets it in production gets an error CI can never see. "
        f"Got instead: {excinfo.value!r}"
    )
    assert str(excinfo.value).startswith(message_prefix), (
        f"expected {class_name}'s own message, got {str(excinfo.value)!r}"
    )


@pytest.mark.parametrize(
    ("module_name", "class_name", "env_suffix", "message_prefix"), _ADAPTERS
)
def test_streaming_honours_the_same_switch(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    env_suffix: str,
    message_prefix: str,
) -> None:
    monkeypatch.setenv(f"LLM_ADAPTERS_NATIVE_HTTP_{env_suffix}", "0")
    adapter = _adapter(module_name, class_name)

    with pytest.raises(RuntimeError) as excinfo:
        # stream() returns an iterable for some adapters and raises eagerly for
        # others; draining it makes both shapes reach the same assertion.
        list(adapter.stream(_request()))

    assert "native HTTP disabled by configuration" in str(excinfo.value), (
        f"{class_name}.stream did not reach its disabled branch: {excinfo.value!r}"
    )


@pytest.mark.parametrize(("module_name", "class_name", "env_suffix", "_prefix"), _ADAPTERS)
@pytest.mark.parametrize("spelling", _FALSE_SPELLINGS)
def test_every_documented_false_spelling_disables(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    env_suffix: str,
    _prefix: str,
    spelling: str,
) -> None:
    """The adapters advertise 0/false/no/off; pin all four, and the casing."""
    monkeypatch.setenv(f"LLM_ADAPTERS_NATIVE_HTTP_{env_suffix}", spelling.upper())
    adapter = _adapter(module_name, class_name)

    assert adapter._use_native_http() is False, (
        f"{class_name} did not treat {spelling.upper()!r} as disabling"
    )


@pytest.mark.parametrize(("module_name", "class_name", "env_suffix", "_prefix"), _ADAPTERS)
def test_unset_switch_leaves_native_http_enabled(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    env_suffix: str,
    _prefix: str,
) -> None:
    """Control: removing the test-env short-circuit must not change the default."""
    monkeypatch.delenv(f"LLM_ADAPTERS_NATIVE_HTTP_{env_suffix}", raising=False)
    adapter = _adapter(module_name, class_name)

    assert adapter._use_native_http() is True, (
        f"{class_name} stopped defaulting to native HTTP"
    )
