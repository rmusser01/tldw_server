"""Stage 1 of TASK-13342: the three boolean coercers that resolve the wrong way.

Each of these is a private re-implementation of "is this string true?", and each gets a
different answer from the canonical `core/testing.py:is_truthy`. Two of them fail *open*
on a security-relevant switch, which is the reason this is fixed ahead of the wider
consolidation described in
`Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md`.

1. `TTS/adapters/audio_cpp_config.py:_as_bool` ends `return bool(value)`, so any
   unrecognised non-empty string is True. It gates `allow_remote_base_url`, whose False
   value is what makes `validate_base_url` enforce the loopback check -- so an operator
   writing `n` for "no" disables the guard and the adapter accepts an arbitrary remote
   host. ADR-026 governs that boundary. The same parser also gates `managed`, which
   spawns a subprocess.

2. `LLM_Calls/providers/google_adapter.py:_env_flag` ends
   `lowered not in {"0","false","no","off",""}`, so `disabled` is True. It gates whether
   caller-supplied URLs are forwarded for Google to fetch server-side.

3. `RAG/rag_service/request_resolution.py:_is_truthy_value` omits `y`, which
   `core/testing.py:is_truthy` accepts. `SEARCH_QUERY_CLASSIFICATION=y` therefore
   resolves off while `RAG_GUARDRAILS_STRICT=y`, parsed elsewhere by the canonical
   helper, resolves on -- in the same request.

The rule these pin: an unrecognised value must never turn a guard off, and a spelling
the canonical parser accepts must not be silently rejected.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.testing import is_truthy

pytestmark = pytest.mark.unit

# Spellings a human plausibly writes meaning "no". None may resolve True.
_MEANT_AS_FALSE = ["n", "N", "no", "none", "nope", "disabled", "off", "0", "false"]

# Spellings the canonical parser accepts as true.
_MEANT_AS_TRUE = ["1", "true", "yes", "y", "on"]


def test_the_canonical_parser_is_the_comparison_point() -> None:
    """Pin what `is_truthy` does, since the assertions below are relative to it."""
    assert is_truthy("y") is True
    assert is_truthy("yes") is True
    assert is_truthy("on") is True
    assert is_truthy("disabled") is False
    assert is_truthy("n") is False


@pytest.mark.parametrize("value", _MEANT_AS_FALSE)
def test_audio_cpp_as_bool_never_resolves_a_no_to_true(value: str) -> None:
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_config import _as_bool

    assert _as_bool(value, default=False) is False, (
        f"_as_bool({value!r}) is True, so allow_remote_base_url={value} turns the "
        "loopback guard OFF -- the opposite of what the operator wrote. The same parser "
        "gates `managed`, which spawns a subprocess."
    )


@pytest.mark.parametrize("value", ["nope", "disabled", "none", "maybe", "0 "])
def test_audio_cpp_unrecognised_value_keeps_the_loopback_guard_on(value: str) -> None:
    """The consequence, asserted through the guard rather than the parser.

    Deliberately uses spellings that are *unrecognised* rather than "n". "n" is now in
    the explicit falsy set, so testing with it would pass even if the fail-closed
    fallback were reverted -- verified by reverting it and watching this test still
    pass. These values exercise the fallback itself, which is the half that protects
    against a typo rather than against a known word.
    """
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_config import (
        AudioCppConfig,
        _as_bool,
    )
    from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError

    assert _as_bool(value, default=False) is False

    with pytest.raises(TTSValidationError) as excinfo:
        AudioCppConfig.from_provider_config(
            {
                "base_url": "http://attacker.example.com:8080",
                "extra_params": {"allow_remote_base_url": value},
            }
        )
    assert excinfo.value.error_code == "remote_base_url_disabled", (
        f"a remote base_url was accepted with allow_remote_base_url={value!r}"
    )


def test_audio_cpp_still_honours_the_documented_true_spellings() -> None:
    """Control: failing closed must not break the switch for people who do enable it."""
    from tldw_Server_API.app.core.TTS.adapters.audio_cpp_config import _as_bool

    for value in ("1", "true", "yes", "on"):
        assert _as_bool(value, default=False) is True, value
    assert _as_bool(None, default=True) is True
    assert _as_bool(True) is True
    assert _as_bool(False, default=True) is False


@pytest.mark.parametrize("value", ["disabled", "nope", "none", "n"])
def test_google_env_flag_does_not_fail_open(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import google_adapter

    monkeypatch.setenv("TLDW_TEST_GOOGLE_FLAG", value)
    assert google_adapter._env_flag("TLDW_TEST_GOOGLE_FLAG") is False, (
        f"_env_flag returned True for {value!r}; this gates whether caller-supplied "
        "URLs are forwarded for Google to fetch server-side"
    )


def test_google_env_flag_still_honours_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control."""
    from tldw_Server_API.app.core.LLM_Calls.providers import google_adapter

    for value in ("1", "true", "yes", "y", "on"):
        monkeypatch.setenv("TLDW_TEST_GOOGLE_FLAG", value)
        assert google_adapter._env_flag("TLDW_TEST_GOOGLE_FLAG") is True, value
    monkeypatch.delenv("TLDW_TEST_GOOGLE_FLAG", raising=False)
    assert google_adapter._env_flag("TLDW_TEST_GOOGLE_FLAG") is False


@pytest.mark.parametrize("value", _MEANT_AS_TRUE)
def test_rag_truthiness_matches_the_canonical_parser(value: str) -> None:
    from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
        _is_truthy_value,
    )

    assert _is_truthy_value(value) is is_truthy(value), (
        f"_is_truthy_value({value!r}) disagrees with core is_truthy, so the same "
        "spelling means different things to different flags in one request"
    )


@pytest.mark.parametrize("value", _MEANT_AS_FALSE)
def test_rag_truthiness_rejects_the_no_spellings(value: str) -> None:
    """Control: widening the true set must not make a 'no' resolve true."""
    from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
        _is_truthy_value,
    )

    assert _is_truthy_value(value) is False, value
