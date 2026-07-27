import json
import logging
import re
from typing import Any

import pytest
import regex as regex_engine

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael
from tldw_Server_API.app.core.Web_Scraping import safe_regex as safe_regex_module
from tldw_Server_API.app.core.Web_Scraping import scraper_router as scraper_router_module
from tldw_Server_API.app.core.Web_Scraping.safe_regex import (
    SafeRegexLimits,
    SafeRegexResult,
    SafeRegexSubResult,
    search_untrusted,
    sub_untrusted,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter

_RECURSIVE_PATTERN = "(" * 500 + "a" + ")" * 500
_LEGACY_INCOMPATIBLE_PATTERNS = (
    r"(?V1)example",
    r"(?|example|never)",
    r"(?P<host>example)(?P<host>\.com)",
)


class _FakeCompiled:
    def __init__(self, *, match: Any = None, error: Exception | None = None) -> None:
        self.match = match
        self.error = error
        self.calls: list[tuple[str, float]] = []

    def search(self, value: str, *, timeout: float) -> Any:
        self.calls.append((value, timeout))
        if self.error is not None:
            raise self.error
        return self.match


class _FakeSubCompiled:
    def __init__(self, *, value: str | None = None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[tuple[Any, str, float]] = []

    def sub(self, repl: Any, value: str, *, timeout: float) -> str:
        self.calls.append((repl, value, timeout))
        if self.error is not None:
            raise self.error
        assert self.value is not None
        return self.value


def _install_fake_compile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    match: Any = None,
    error: Exception | None = None,
) -> tuple[_FakeCompiled, dict[str, Any]]:
    compiled = _FakeCompiled(match=match, error=error)
    observed: dict[str, Any] = {}

    def _fake_compile(pattern: str, flags: int) -> _FakeCompiled:
        observed["pattern"] = pattern
        observed["flags"] = flags
        return compiled

    monkeypatch.setattr(safe_regex_module, "_compile_pattern", _fake_compile)
    return compiled, observed


def _install_router_budget_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    elapsed_per_search: float,
    matching_pattern: str | None = None,
    rejected_patterns: set[str] | None = None,
) -> list[tuple[str, float]]:
    state = {"now": 0.0}
    calls: list[tuple[str, float]] = []
    rejected = rejected_patterns or set()

    def _fake_monotonic() -> float:
        return state["now"]

    def _fake_search(
        pattern: str,
        _value: str,
        *,
        limits: SafeRegexLimits,
    ) -> SafeRegexResult:
        calls.append((pattern, limits.timeout_s))
        state["now"] += min(elapsed_per_search, limits.timeout_s)
        if pattern in rejected:
            return SafeRegexResult(matched=False, code="regex_invalid")
        return SafeRegexResult(matched=pattern == matching_pattern)

    monkeypatch.setattr(
        scraper_router_module,
        "_monotonic",
        _fake_monotonic,
        raising=False,
    )
    monkeypatch.setattr(scraper_router_module, "search_untrusted", _fake_search)
    return calls


def _install_regex_llm_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    raw_payload = json.dumps(payload)

    def _fake_call(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": raw_payload}}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
            "model": "gpt-test",
        }

    monkeypatch.setattr(chat_service, "perform_chat_api_call", _fake_call)


def _generate_regex(html: str) -> dict[str, Any]:
    return ael.generate_regex_pattern_from_llm(
        html,
        "https://example.com",
        llm_settings={"provider": "openai"},
    )


def test_safe_regex_defaults_are_exact() -> None:
    assert SafeRegexLimits() == SafeRegexLimits(
        max_pattern_chars=4_096,
        max_input_chars=8_192,
        timeout_s=0.100,
    )


def test_invalid_pattern_returns_stable_code() -> None:
    result = search_untrusted("[", "sample")

    assert result == SafeRegexResult(matched=False, code="regex_invalid")


@pytest.mark.parametrize(
    "pattern",
    [r"(?V1)a", r"(?|a|b)"],
    ids=["version-directive", "branch-reset"],
)
def test_regex_only_constructs_are_rejected_before_engine_compile(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
) -> None:
    compiled, observed = _install_fake_compile(monkeypatch)

    result = search_untrusted(pattern, "a")

    assert result == SafeRegexResult(matched=False, code="regex_invalid")
    assert observed == {}
    assert compiled.calls == []


def test_duplicate_named_group_is_rejected_before_engine_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, observed = _install_fake_compile(monkeypatch)

    result = search_untrusted(r"(?P<value>a)(?P<value>b)", "ab")

    assert result == SafeRegexResult(matched=False, code="regex_invalid")
    assert observed == {}
    assert compiled.calls == []


def test_locale_flag_for_string_pattern_is_rejected_before_engine_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, observed = _install_fake_compile(monkeypatch)

    result = search_untrusted("a", "a", flags=re.LOCALE)

    assert result == SafeRegexResult(matched=False, code="regex_invalid")
    assert observed == {}
    assert compiled.calls == []


def test_incompatible_stdlib_flags_are_rejected_before_engine_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, observed = _install_fake_compile(monkeypatch)

    result = search_untrusted("a", "a", flags=re.ASCII | re.UNICODE)

    assert result == SafeRegexResult(matched=False, code="regex_invalid")
    assert observed == {}
    assert compiled.calls == []


def test_recursive_compile_failure_returns_stable_code_without_disclosure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)

    result = search_untrusted(_RECURSIVE_PATTERN, "a")
    captured = capsys.readouterr()
    disclosed = " ".join([repr(result), caplog.text, captured.out, captured.err])

    assert result == SafeRegexResult(matched=False, code="regex_invalid")
    assert _RECURSIVE_PATTERN not in disclosed
    assert "maximum recursion depth exceeded" not in disclosed


def test_pattern_size_boundary_accepts_exact_limit_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, observed = _install_fake_compile(monkeypatch)

    exact_result = search_untrusted("a" * 4_096, "sample")
    over_result = search_untrusted("a" * 4_097, "sample")

    assert exact_result.code is None
    assert observed["pattern"] == "a" * 4_096
    assert compiled.calls == [("sample", 0.100)]
    assert over_result == SafeRegexResult(matched=False, code="regex_too_large")


def test_router_input_boundary_accepts_exact_limit_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match = object()
    compiled, _observed = _install_fake_compile(monkeypatch, match=match)

    exact_result = search_untrusted("x", "a" * 8_192)
    over_result = search_untrusted("x", "a" * 8_193)

    assert exact_result == SafeRegexResult(matched=True, match=match)
    assert compiled.calls == [("a" * 8_192, 0.100)]
    assert over_result == SafeRegexResult(matched=False, code="regex_too_large")


def test_generated_input_boundary_accepts_exact_limit_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _observed = _install_fake_compile(monkeypatch)
    limits = SafeRegexLimits(max_input_chars=1_000_000)

    exact_result = search_untrusted("x", "a" * 1_000_000, limits=limits)
    over_result = search_untrusted("x", "a" * 1_000_001, limits=limits)

    assert exact_result.code is None
    assert compiled.calls == [("a" * 1_000_000, 0.100)]
    assert over_result == SafeRegexResult(matched=False, code="regex_too_large")


def test_default_timeout_is_passed_to_compiled_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _observed = _install_fake_compile(monkeypatch)

    result = search_untrusted("x", "sample")

    assert result.code is None
    assert compiled.calls == [("sample", 0.100)]


def test_injected_timeout_maps_engine_timeout_to_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _observed = _install_fake_compile(
        monkeypatch,
        error=TimeoutError("engine detail must remain private"),
    )

    result = search_untrusted(
        "x",
        "sample",
        limits=SafeRegexLimits(timeout_s=0.025),
    )

    assert compiled.calls == [("sample", 0.025)]
    assert result == SafeRegexResult(matched=False, code="regex_timeout")


@pytest.mark.parametrize(
    "flags",
    [
        re.ASCII | re.IGNORECASE | re.MULTILINE,
        int(re.ASCII | re.IGNORECASE | re.MULTILINE),
    ],
)
def test_stdlib_regex_flags_are_normalized_for_regex_engine(
    monkeypatch: pytest.MonkeyPatch,
    flags: int,
) -> None:
    _compiled, observed = _install_fake_compile(monkeypatch)

    result = search_untrusted("x", "sample", flags=flags)

    assert result.code is None
    assert observed["flags"] == int(regex_engine.ASCII | regex_engine.IGNORECASE | regex_engine.MULTILINE)


@pytest.mark.parametrize("flags", ["i", True, re.DEBUG, 1 << 20, -1])
def test_invalid_or_unsupported_flags_return_stable_code(flags: Any) -> None:
    result = search_untrusted("x", "sample", flags=flags)

    assert result == SafeRegexResult(matched=False, code="regex_invalid")


@pytest.mark.parametrize(
    "limits",
    [
        SafeRegexLimits(max_pattern_chars=0),
        SafeRegexLimits(max_pattern_chars=-1),
        SafeRegexLimits(max_pattern_chars=1.5),
        SafeRegexLimits(max_input_chars=0),
        SafeRegexLimits(max_input_chars=-1),
        SafeRegexLimits(max_input_chars=1.5),
        SafeRegexLimits(timeout_s=0),
        SafeRegexLimits(timeout_s=-0.1),
        SafeRegexLimits(timeout_s=float("nan")),
        SafeRegexLimits(timeout_s=float("inf")),
        SafeRegexLimits(timeout_s="0.1"),
    ],
)
def test_nonsensical_limits_return_stable_code(limits: SafeRegexLimits) -> None:
    result = search_untrusted("x", "sample", limits=limits)

    assert result == SafeRegexResult(matched=False, code="regex_invalid")


def test_injected_limits_cannot_raise_pattern_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _compiled, _observed = _install_fake_compile(monkeypatch)

    result = search_untrusted(
        "a" * 4_097,
        "sample",
        limits=SafeRegexLimits(max_pattern_chars=10_000),
    )

    assert result == SafeRegexResult(matched=False, code="regex_too_large")


def test_injected_limits_cannot_raise_generated_input_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _compiled, _observed = _install_fake_compile(monkeypatch)

    result = search_untrusted(
        "x",
        "a" * 1_000_001,
        limits=SafeRegexLimits(max_input_chars=2_000_000),
    )

    assert result == SafeRegexResult(matched=False, code="regex_too_large")


def test_injected_limits_cannot_raise_timeout_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _observed = _install_fake_compile(monkeypatch)

    result = search_untrusted(
        "x",
        "sample",
        limits=SafeRegexLimits(timeout_s=1.0),
    )

    assert result.code is None
    assert compiled.calls == [("sample", 0.100)]


def test_match_result_preserves_groups_and_span() -> None:
    value = "Order #12345 confirmed"

    result = search_untrusted(r"Order\s+#(\d+)", value, flags=re.IGNORECASE)

    assert result.matched is True
    assert result.code is None
    assert result.match is not None
    assert result.match.group(1) == "12345"
    assert result.match.span() == (0, 12)


def test_valid_no_match_has_no_failure_code() -> None:
    result = search_untrusted(r"Order\s+#(\d+)", "No order here")

    assert result == SafeRegexResult(matched=False)


def test_sub_untrusted_preserves_global_group_replacement_semantics() -> None:
    result = sub_untrusted(
        r"(\w+),\s*(\w+)",
        r"\2 \1",
        "Lovelace, Ada; Hopper, Grace",
    )

    assert result == SafeRegexSubResult(value="Ada Lovelace; Grace Hopper")


@pytest.mark.parametrize(
    ("replacement", "value"),
    [(r"\2", "a"), (r"\2", "no match"), (r"\g<missing>", "a")],
)
def test_sub_untrusted_rejects_invalid_group_reference_templates(
    replacement: str,
    value: str,
) -> None:
    result = sub_untrusted(r"(a)", replacement, value)

    assert result == SafeRegexSubResult(code="regex_invalid")


def test_sub_untrusted_bounds_replacement_template_size() -> None:
    exact_result = sub_untrusted("x", "a" * 4_096, "x")
    over_result = sub_untrusted("x", "a" * 4_097, "x")

    assert exact_result == SafeRegexSubResult(value="a" * 4_096)
    assert over_result == SafeRegexSubResult(code="regex_too_large")


def test_sub_untrusted_bounds_amplified_output_size() -> None:
    replacement = "a" * 4_000

    exact_result = sub_untrusted("x", replacement, "x" * 250)
    over_result = sub_untrusted("x", replacement, "x" * 250 + "y")

    assert exact_result == SafeRegexSubResult(value="a" * 1_000_000)
    assert over_result == SafeRegexSubResult(code="regex_too_large")


def test_sub_untrusted_bounds_backreference_output_amplification() -> None:
    exact_result = sub_untrusted(r"(a+)", r"\1" * 500, "a" * 2_000)
    over_result = sub_untrusted(r"(a+)", r"\1" * 501, "a" * 2_000)

    assert exact_result == SafeRegexSubResult(value="a" * 1_000_000)
    assert over_result == SafeRegexSubResult(code="regex_too_large")


@pytest.mark.parametrize(
    ("pattern", "value", "expected_code"),
    [
        ("[", "sample", "regex_invalid"),
        ("a" * 4_097, "sample", "regex_too_large"),
        ("a", "x" * 8_193, "regex_too_large"),
        (r"(?V1)a", "a", "regex_invalid"),
    ],
)
def test_sub_untrusted_uses_search_limits_and_stdlib_dialect(
    pattern: str,
    value: str,
    expected_code: str,
) -> None:
    result = sub_untrusted(pattern, "replacement", value)

    assert result == SafeRegexSubResult(code=expected_code)


def test_sub_untrusted_passes_the_bounded_timeout_and_sanitizes_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _FakeSubCompiled(error=TimeoutError("private engine detail"))
    monkeypatch.setattr(safe_regex_module, "_compile_pattern", lambda *_args: compiled)

    result = sub_untrusted(
        "x",
        "replacement",
        "sample",
        limits=SafeRegexLimits(timeout_s=0.025),
    )

    assert len(compiled.calls) == 1
    replacement, value, timeout = compiled.calls[0]
    assert callable(replacement)
    assert (value, timeout) == ("sample", 0.025)
    assert result == SafeRegexSubResult(code="regex_timeout")


def test_sub_untrusted_does_not_disclose_pattern_or_engine_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_pattern = "private-sub-pattern"
    secret_error = "private-sub-error"
    compiled = _FakeSubCompiled(error=ValueError(secret_error))
    monkeypatch.setattr(safe_regex_module, "_compile_pattern", lambda *_args: compiled)
    caplog.set_level(logging.DEBUG)

    result = sub_untrusted(secret_pattern, "replacement", "sample")
    captured = capsys.readouterr()
    disclosed = " ".join([repr(result), caplog.text, captured.out, captured.err])

    assert result == SafeRegexSubResult(code="regex_invalid")
    assert secret_pattern not in disclosed
    assert secret_error not in disclosed


def test_raw_pattern_and_exception_are_not_returned_or_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_pattern = "private-pattern-token"
    secret_exception = "private-engine-exception"

    def _raising_compile(_pattern: str, _flags: int) -> Any:
        raise ValueError(secret_exception)

    monkeypatch.setattr(safe_regex_module, "_compile_pattern", _raising_compile)
    caplog.set_level(logging.DEBUG)

    result = search_untrusted(secret_pattern, "sample")
    captured = capsys.readouterr()
    disclosed = " ".join([repr(result), caplog.text, captured.out, captured.err])

    assert result == SafeRegexResult(matched=False, code="regex_invalid")
    assert secret_pattern not in disclosed
    assert secret_exception not in disclosed


def test_router_validation_drops_invalid_and_oversized_patterns() -> None:
    rules = {
        "domains": {
            "example.com": {
                "backend": "curl",
                "url_patterns": [r"/article/\d+$", "[", "a" * 4_097, 123],
            }
        }
    }

    cleaned = ScraperRouter.validate_rules(rules)

    assert cleaned["domains"]["example.com"]["url_patterns"] == [r"/article/\d+$"]


def test_router_validation_discards_rule_when_all_configured_patterns_are_rejected() -> None:
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": ["[", "a" * 4_097],
                }
            }
        }
    )

    assert "example.com" not in cleaned["domains"]
    assert ScraperRouter(cleaned).resolve("https://example.com/article").backend == "auto"


def test_router_validation_discards_rule_when_pattern_cap_prevents_a_survivor() -> None:
    patterns = ["["] * scraper_router_module._MAX_URL_PATTERNS + [r"/article$"]

    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": patterns,
                }
            }
        }
    )

    assert "example.com" not in cleaned["domains"]


def test_router_validation_discards_rule_when_budget_expires_before_a_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(
        monkeypatch,
        elapsed_per_search=0.100,
        rejected_patterns={"rejected"},
    )

    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": ["rejected", "survivor"],
                }
            }
        }
    )

    assert "example.com" not in cleaned["domains"]
    assert [pattern for pattern, _timeout in calls] == ["rejected"]


def test_router_validation_preserves_explicit_empty_pattern_constraint() -> None:
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": [],
                }
            }
        }
    )

    assert cleaned["domains"]["example.com"]["url_patterns"] == []
    assert ScraperRouter(cleaned).resolve("https://example.com/article").backend == "curl"


def test_router_validation_keeps_rule_when_at_least_one_pattern_survives() -> None:
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": ["[", r"/article$"],
                }
            }
        }
    )

    assert cleaned["domains"]["example.com"]["url_patterns"] == [r"/article$"]
    assert ScraperRouter(cleaned).resolve("https://example.com/article").backend == "curl"


def test_router_validation_discards_rule_with_non_list_pattern_constraint() -> None:
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": r"/article$",
                }
            }
        }
    )

    assert "example.com" not in cleaned["domains"]


def test_router_validation_caps_configured_pattern_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(monkeypatch, elapsed_per_search=0.0)
    patterns = [f"pattern-{index}" for index in range(40)]

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": {"url_patterns": patterns}}})

    assert scraper_router_module._MAX_URL_PATTERNS == 32
    assert cleaned["domains"]["example.com"]["url_patterns"] == patterns[:32]
    assert [pattern for pattern, _timeout in calls] == patterns[:32]


def test_router_validation_uses_one_shared_pattern_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(monkeypatch, elapsed_per_search=0.040)
    patterns = [f"pattern-{index}" for index in range(10)]

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": {"url_patterns": patterns}}})

    assert cleaned["domains"]["example.com"]["url_patterns"] == patterns[:3]
    assert [pattern for pattern, _timeout in calls] == patterns[:3]
    assert [timeout for _pattern, timeout in calls] == pytest.approx([0.100, 0.060, 0.020])


def test_router_validation_drops_legacy_incompatible_patterns() -> None:
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "url_patterns": [*_LEGACY_INCOMPATIBLE_PATTERNS, r"/valid$"],
                }
            }
        }
    )

    assert cleaned["domains"]["example.com"]["url_patterns"] == [r"/valid$"]


def test_router_validation_drops_recursive_pattern_without_raising() -> None:
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": [_RECURSIVE_PATTERN],
                }
            }
        }
    )

    assert "example.com" not in cleaned["domains"]


def test_directly_constructed_router_invalid_pattern_fails_open() -> None:
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": ["["],
                }
            }
        }
    )

    plan = router.resolve("https://example.com/article")

    assert plan.backend == "auto"


def test_directly_constructed_router_recursive_pattern_fails_open() -> None:
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": [_RECURSIVE_PATTERN],
                }
            }
        }
    )

    plan = router.resolve("https://example.com/article")

    assert plan.backend == "auto"


def test_direct_router_caps_pattern_count_before_late_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(
        monkeypatch,
        elapsed_per_search=0.0,
        matching_pattern="match-after-cap",
    )
    patterns = [f"no-match-{index}" for index in range(32)]
    patterns.append("match-after-cap")
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": patterns,
                }
            }
        }
    )

    plan = router.resolve("https://example.com/article")

    assert plan.backend == "auto"
    assert [pattern for pattern, _timeout in calls] == patterns[:32]


def test_direct_router_uses_one_shared_pattern_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(monkeypatch, elapsed_per_search=0.040)
    patterns = [f"no-match-{index}" for index in range(10)]
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": patterns,
                }
            }
        }
    )

    plan = router.resolve("https://example.com/article")

    assert plan.backend == "auto"
    assert [pattern for pattern, _timeout in calls] == patterns[:3]
    assert [timeout for _pattern, timeout in calls] == pytest.approx([0.100, 0.060, 0.020])


def test_direct_router_applies_match_before_shared_deadline_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(
        monkeypatch,
        elapsed_per_search=0.040,
        matching_pattern="match-now",
    )
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": ["no-match", "match-now", "later"],
                }
            }
        }
    )

    plan = router.resolve("https://example.com/article")

    assert plan.backend == "curl"
    assert [pattern for pattern, _timeout in calls] == ["no-match", "match-now"]
    assert [timeout for _pattern, timeout in calls] == pytest.approx([0.100, 0.060])


def test_direct_router_legacy_incompatible_patterns_fail_open() -> None:
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": list(_LEGACY_INCOMPATIBLE_PATTERNS),
                }
            }
        }
    )

    plan = router.resolve("https://example.com/article")

    assert plan.backend == "auto"


def test_router_enforces_exact_input_boundary() -> None:
    prefix = "https://example.com/"
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": [r"^https://example\.com/"],
                }
            }
        }
    )

    exact_plan = router.resolve(prefix + ("a" * (8_192 - len(prefix))))
    over_plan = router.resolve(prefix + ("a" * (8_193 - len(prefix))))

    assert exact_plan.backend == "curl"
    assert over_plan.backend == "auto"


def test_router_pattern_timeout_is_a_non_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _observed = _install_fake_compile(
        monkeypatch,
        error=TimeoutError("engine detail must remain private"),
    )
    router = ScraperRouter(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": [r"(a+)+$"],
                }
            }
        }
    )

    plan = router.resolve("https://example.com/" + ("a" * 200) + "!")

    first_timeout = compiled.calls[0][1]
    assert 0 < first_timeout <= 0.100
    assert plan.backend == "auto"


def test_domain_suffix_matching_does_not_use_safe_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_search(*_args: Any, **_kwargs: Any) -> SafeRegexResult:
        raise AssertionError("domain suffix matching must not use safe_regex")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.scraper_router.search_untrusted",
        _unexpected_search,
    )
    router = ScraperRouter({"domains": {"*.example.com": {"backend": "curl"}}})

    plan = router.resolve("https://sub.example.com/article")

    assert plan.backend == "curl"


def test_generated_regex_preserves_valid_result_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = "<html><body>Order #12345 confirmed.</body></html>"
    _install_regex_llm_response(
        monkeypatch,
        {"pattern": r"Order\s+#(\d+)", "flags": "i", "group": 1},
    )

    result = _generate_regex(html)

    start = html.index("Order #12345")
    assert result["success"] is True
    assert result["pattern"] == r"Order\s+#(\d+)"
    assert result["flags"] == "i"
    assert result["group"] == 1
    assert result["sample_match"] == "12345"
    assert result["sample_span"] == [start, start + len("Order #12345")]


def test_generated_regex_invalid_pattern_uses_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_regex_llm_response(monkeypatch, {"pattern": "["})

    result = _generate_regex("sample")

    assert result["success"] is False
    assert result["error"] == "regex_invalid"


def test_generated_regex_recursive_pattern_uses_stable_code_without_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_regex_llm_response(monkeypatch, {"pattern": _RECURSIVE_PATTERN})
    caplog.set_level(logging.DEBUG)

    result = _generate_regex("a")
    captured = capsys.readouterr()
    disclosed = " ".join([repr(result), caplog.text, captured.out, captured.err])

    assert result["success"] is False
    assert result["error"] == "regex_invalid"
    assert _RECURSIVE_PATTERN not in disclosed
    assert "maximum recursion depth exceeded" not in disclosed


@pytest.mark.parametrize(
    "payload",
    [
        {"pattern": _LEGACY_INCOMPATIBLE_PATTERNS[0]},
        {"pattern": _LEGACY_INCOMPATIBLE_PATTERNS[1]},
        {"pattern": _LEGACY_INCOMPATIBLE_PATTERNS[2]},
        {"pattern": "example", "flags": int(re.LOCALE)},
        {"pattern": "example", "flags": int(re.ASCII | re.UNICODE)},
    ],
    ids=[
        "version-directive",
        "branch-reset",
        "duplicate-group",
        "locale",
        "ascii-unicode",
    ],
)
def test_generated_regex_legacy_incompatible_input_uses_stable_code(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    _install_regex_llm_response(monkeypatch, payload)

    result = _generate_regex("example.com")

    assert result["success"] is False
    assert result["error"] == "regex_invalid"


def test_generated_regex_oversized_pattern_uses_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_regex_llm_response(monkeypatch, {"pattern": "a" * 4_097})

    result = _generate_regex("sample")

    assert result["success"] is False
    assert result["error"] == "regex_too_large"


def test_generated_regex_enforces_exact_sample_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_regex_llm_response(monkeypatch, {"pattern": "a$"})

    exact_result = _generate_regex("a" * 1_000_000)
    over_result = _generate_regex("a" * 1_000_001)

    assert exact_result["success"] is True
    assert exact_result["sample_match"] == "a"
    assert exact_result["sample_span"] == [999_999, 1_000_000]
    assert over_result["success"] is False
    assert over_result["error"] == "regex_too_large"


def test_generated_regex_timeout_uses_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_regex_llm_response(monkeypatch, {"pattern": r"(a+)+$"})
    compiled, _observed = _install_fake_compile(
        monkeypatch,
        error=TimeoutError("engine detail must remain private"),
    )

    result = _generate_regex(("a" * 200) + "!")

    assert compiled.calls[0][1] == 0.100
    assert result["success"] is False
    assert result["error"] == "regex_timeout"


def test_trusted_catalogs_remain_stdlib_regex_patterns() -> None:
    assert ael._BOILERPLATE_REGEXES
    assert all(isinstance(pattern, re.Pattern) for pattern in ael._BOILERPLATE_REGEXES)
    assert ael._REGEX_CATALOG
    assert all(isinstance(pattern, re.Pattern) for _label, pattern in ael._REGEX_CATALOG)
    assert ael._strip_boilerplate_sections("Keep this\nSubscribe now\nKeep that") == ("Keep this\nKeep that")
