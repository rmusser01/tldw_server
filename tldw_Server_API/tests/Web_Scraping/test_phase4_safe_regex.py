import json
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def _assert_match_matches_stdlib(
    result: SafeRegexResult,
    expected: re.Match[str] | None,
) -> None:
    assert result.matched is (expected is not None)
    assert result.code is None
    if expected is None:
        assert result.match is None
        return

    match = result.match
    assert match is not None
    assert bool(match) is True
    selectors: list[int | str] = list(range(expected.re.groups + 1))
    selectors.extend(expected.re.groupindex)
    assert match.group() == expected.group()
    assert match.group(*selectors) == expected.group(*selectors)
    assert match.groups() == expected.groups()
    assert match.groups("missing") == expected.groups("missing")
    assert match.groupdict() == expected.groupdict()
    assert match.groupdict("missing") == expected.groupdict("missing")
    for selector in selectors:
        assert match.group(selector) == expected.group(selector)
        assert match.span(selector) == expected.span(selector)
        assert match.start(selector) == expected.start(selector)
        assert match.end(selector) == expected.end(selector)


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
    assert observed["flags"] == int(
        regex_engine.ASCII | regex_engine.IGNORECASE | regex_engine.MULTILINE | regex_engine.VERSION0
    )


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


def test_ascii_stdlib_search_uses_explicit_regex_version0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    compiled = _FakeCompiled()

    def _fake_regex_compile(pattern: str, flags: int) -> _FakeCompiled:
        observed.update(pattern=pattern, flags=flags)
        return compiled

    monkeypatch.setattr(regex_engine, "compile", _fake_regex_compile)

    result = search_untrusted(r"[a-z]+", "ASCII", flags=re.IGNORECASE, dialect="stdlib")

    assert result == SafeRegexResult(matched=False)
    assert observed == {
        "pattern": r"[a-z]+",
        "flags": int(regex_engine.IGNORECASE | regex_engine.VERSION0),
    }
    assert compiled.calls == [("ASCII", 0.100)]


@pytest.mark.parametrize(
    ("pattern", "flags"),
    [
        ("i", re.IGNORECASE),
        ("I", re.IGNORECASE),
        ("[a-z]", re.IGNORECASE),
        ("[A-Z]", re.IGNORECASE),
    ],
    ids=["lower-literal", "upper-literal", "lower-class", "upper-class"],
)
@pytest.mark.parametrize("value", ["i", "I", "İ", "ı"], ids=["i", "I", "dotted-I", "dotless-i"])
def test_stdlib_flag_ignorecase_matches_re_search(
    pattern: str,
    flags: int,
    value: str,
) -> None:
    expected = re.search(pattern, value, flags)

    result = search_untrusted(pattern, value, flags=flags, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


@pytest.mark.parametrize(
    "pattern",
    ["(?i)i", "(?i)I", "(?i)[a-z]", "(?i)[A-Z]"],
    ids=["lower-literal", "upper-literal", "lower-class", "upper-class"],
)
@pytest.mark.parametrize("value", ["i", "I", "İ", "ı"], ids=["i", "I", "dotted-I", "dotless-i"])
def test_stdlib_inline_ignorecase_matches_re_search(pattern: str, value: str) -> None:
    expected = re.search(pattern, value)

    result = search_untrusted(pattern, value, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


@pytest.mark.parametrize(
    ("pattern", "value", "flags"),
    [
        (r"(?P<word>\w+)-(?P<digits>\d+)(?P<suffix>Ω)?", "prefix café-١٢٣ suffix", 0),
        (r"(?P<literal>İ)(?P<tail>stanbul)", "xx İstanbul yy", 0),
        (r"(?P<word>[^\W\d_]+)", "--élan--", 0),
        (r"(?P<lead>i)(?P<tail>stanbul)", "xx İstanbul yy", re.IGNORECASE),
    ],
    ids=["unicode-classes", "unicode-literal", "unicode-letter-class", "unicode-casefold"],
)
def test_stdlib_unicode_match_accessors_match_re_search(
    pattern: str,
    value: str,
    flags: int,
) -> None:
    expected = re.search(pattern, value, flags)

    result = search_untrusted(pattern, value, flags=flags, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"\u0130", "I"),
        (r"[\u0131]", "i"),
        (r"\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}", "ı"),
        (r"\N{LATIN SMALL LETTER DOTLESS I}", "İ"),
    ],
    ids=["unicode-escape", "class-unicode-escape", "named-dotted-I", "named-dotless-i"],
)
def test_stdlib_escaped_unicode_literals_match_re_search(pattern: str, value: str) -> None:
    expected = re.search(pattern, value, re.IGNORECASE)

    result = search_untrusted(pattern, value, flags=re.IGNORECASE, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


def test_stdlib_lone_surrogate_literal_matches_re_search() -> None:
    pattern = "\ud800"
    value = "prefix\ud800suffix"
    expected = re.search(pattern, value)

    result = search_untrusted(pattern, value, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


def test_stdlib_unicode_search_does_not_compile_with_regex_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_compile(_pattern: str, _flags: int) -> Any:
        raise AssertionError("Unicode stdlib search must use the isolated worker")

    monkeypatch.setattr(safe_regex_module, "_compile_pattern", _unexpected_compile)

    result = search_untrusted(r"(?P<city>İstanbul)", "xx İstanbul yy", dialect="stdlib")

    assert result.matched is True
    assert result.code is None
    assert result.match is not None
    assert result.match.group("city") == "İstanbul"


@pytest.mark.parametrize(
    ("pattern", "value", "flags"),
    [
        (r"(?P<word>[A-Z]+)-(\d+)", "ref-123", re.IGNORECASE),
        (r"(?i)(?P<word>[a-z]+)-(\d+)", "REF-123", 0),
        (r"^(?P<word>\w+)$", "ASCII_123", re.ASCII),
    ],
    ids=["flag-ignorecase", "inline-ignorecase", "ascii-classes"],
)
def test_stdlib_ascii_controls_match_re_search(
    pattern: str,
    value: str,
    flags: int,
) -> None:
    expected = re.search(pattern, value, flags)

    result = search_untrusted(pattern, value, flags=flags, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


@pytest.mark.parametrize(
    "pattern",
    [r"[[:alpha:]]", r"[[:digit:]]"],
)
def test_stdlib_ambiguous_ascii_sets_match_re_search(pattern: str) -> None:
    value = "a1:[]"
    expected = re.search(pattern, value)

    result = search_untrusted(pattern, value, dialect="stdlib")

    _assert_match_matches_stdlib(result, expected)


def test_stdlib_worker_startup_does_not_consume_the_engine_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout_s = 0.025
    startup_delay_s = 0.050
    original_popen = subprocess.Popen

    def _delayed_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        worker = original_popen(*args, **kwargs)
        time.sleep(startup_delay_s)
        return worker

    monkeypatch.setattr(subprocess, "Popen", _delayed_popen)
    started = time.monotonic()

    result = search_untrusted(
        r"(?P<city>İstanbul)",
        "İstanbul",
        limits=SafeRegexLimits(timeout_s=timeout_s),
        dialect="stdlib",
    )

    elapsed = time.monotonic() - started
    assert elapsed >= startup_delay_s
    assert result.matched is True
    assert result.code is None
    assert result.match is not None
    assert result.match.group("city") == "İstanbul"


def test_stdlib_unicode_search_is_safe_under_concurrent_calls() -> None:
    cases = [
        ("i", "ı", re.IGNORECASE),
        ("I", "İ", re.IGNORECASE),
        ("(?i)[a-z]", "ı", 0),
        ("(?i)[A-Z]", "İ", 0),
    ]

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        results = list(
            executor.map(
                lambda case: search_untrusted(
                    case[0],
                    case[1],
                    flags=case[2],
                    dialect="stdlib",
                ),
                cases,
            )
        )

    for result, (pattern, value, flags) in zip(results, cases, strict=True):
        _assert_match_matches_stdlib(result, re.search(pattern, value, flags))


def test_stdlib_worker_receives_bounded_json_on_stdin_not_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_pattern = r"(?P<secret>İstanbul-parent-pattern)"
    secret_value = "prefix İstanbul-parent-pattern suffix"
    requests: list[bytes] = []
    launches: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_popen = subprocess.Popen

    class _RecordingWorker:
        def __init__(self, worker: subprocess.Popen[bytes]) -> None:
            self._worker = worker

        def __getattr__(self, name: str) -> Any:
            return getattr(self._worker, name)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            payload = kwargs.get("input", args[0] if args else None)
            assert isinstance(payload, bytes)
            requests.append(payload)
            return self._worker.communicate(*args, **kwargs)

    def _recording_popen(*args: Any, **kwargs: Any) -> _RecordingWorker:
        launches.append((args, kwargs))
        return _RecordingWorker(original_popen(*args, **kwargs))

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)

    result = search_untrusted(secret_pattern, secret_value, dialect="stdlib")

    assert result.matched is True
    assert len(launches) == 1
    args, kwargs = launches[0]
    argv = args[0]
    assert isinstance(argv, list)
    assert all(secret_pattern not in argument for argument in argv)
    assert all(secret_value not in argument for argument in argv)
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert len(requests) == 1
    assert requests[0].endswith(b"\n")
    assert json.loads(requests[0]) == {
        "flags": 0,
        "pattern": secret_pattern,
        "value": secret_value,
    }


def test_catastrophic_stdlib_timeout_reaps_child_threads_and_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []
    execution_timeouts: list[float] = []
    original_popen = subprocess.Popen

    class _RecordingWorker:
        def __init__(self, worker: subprocess.Popen[bytes]) -> None:
            self._worker = worker

        def __getattr__(self, name: str) -> Any:
            return getattr(self._worker, name)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            timeout = kwargs.get("timeout")
            if timeout is not None:
                execution_timeouts.append(timeout)
            return self._worker.communicate(*args, **kwargs)

    def _recording_popen(*args: Any, **kwargs: Any) -> _RecordingWorker:
        worker = _RecordingWorker(original_popen(*args, **kwargs))
        spawned.append(worker)
        return worker

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)
    started = time.monotonic()

    try:
        result = search_untrusted(
            r"(?:a|aa)+İ$",
            "a" * 6_000 + "Ω",
            limits=SafeRegexLimits(timeout_s=0.020),
            dialect="stdlib",
        )
        elapsed = time.monotonic() - started

        assert result == SafeRegexResult(matched=False, code="regex_timeout")
        assert elapsed < 5.0
        assert len(spawned) == 1
        assert execution_timeouts[0] == 0.020
        assert all(worker.returncode is not None for worker in spawned)
        assert all(worker.stdin is not None and worker.stdin.closed for worker in spawned)
        assert all(worker.stdout is not None and worker.stdout.closed for worker in spawned)
        assert not any(
            thread.name == "safe-regex-startup-reader" and thread.is_alive() for thread in threading.enumerate()
        )
    finally:
        for worker in spawned:
            if worker.returncode is None:
                worker.kill()
                worker.wait(timeout=1.0)


def test_valid_no_match_has_no_failure_code() -> None:
    result = search_untrusted(r"Order\s+#(\d+)", "No order here")

    assert result == SafeRegexResult(matched=False)


def test_regex_dialect_accepts_variable_length_lookbehind_while_default_rejects_it() -> None:
    pattern = r"(?<=\b[A-Z]{1,3})(\d+)"

    default_result = search_untrusted(pattern, "AB123")
    regex_result = search_untrusted(pattern, "AB123", dialect="regex")

    assert default_result == SafeRegexResult(matched=False, code="regex_invalid")
    assert regex_result.matched is True
    assert regex_result.code is None
    assert regex_result.match is not None
    assert regex_result.match.group(1) == "123"


def test_sub_untrusted_preserves_global_group_replacement_semantics() -> None:
    result = sub_untrusted(
        r"(\w+),\s*(\w+)",
        r"\2 \1",
        "Lovelace, Ada; Hopper, Grace",
    )

    assert result == SafeRegexSubResult(value="Ada Lovelace; Grace Hopper")


def test_stdlib_substitution_rejects_variable_length_lookbehind() -> None:
    result = sub_untrusted(
        r"(?<=\b[A-Z]{1,3})(\d+)",
        r"[\1]",
        "AB123",
    )

    assert result == SafeRegexSubResult(code="regex_invalid")


def test_stdlib_substitution_matches_re_for_unicode_word_characters() -> None:
    value = "e\u0301"

    result = sub_untrusted(r"\w", "X", value)

    assert result == SafeRegexSubResult(value=re.sub(r"\w", "X", value))


def test_stdlib_substitution_matches_named_group_replacement() -> None:
    pattern = r"(?P<family>\w+),\s*(?P<given>\w+)"
    replacement = r"\g<given> \g<family>"
    value = "Lovelace, Ada; Hopper, Grace"

    result = sub_untrusted(pattern, replacement, value)

    assert result == SafeRegexSubResult(value=re.sub(pattern, replacement, value))


@pytest.mark.parametrize(
    "replacement",
    [r"\g<+1>", r"\g< 1>", r"\g<١>", r"\g<-1>"],
    ids=["plus", "space", "unicode-decimal", "negative"],
)
def test_stdlib_numeric_group_reference_grammar_matches_active_runtime(
    replacement: str,
) -> None:
    try:
        expected = SafeRegexSubResult(value=re.sub(r"(x)", replacement, "x"))
    except re.error:
        expected = SafeRegexSubResult(code="regex_invalid")

    assert sub_untrusted(r"(x)", replacement, "x") == expected


def test_legacy_numeric_group_reference_protocol_branch_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        safe_regex_module,
        "_STDLIB_LEGACY_NUMERIC_GROUP_IDS",
        True,
        raising=False,
    )

    accepted = {
        r"\g<+1>": "x",
        r"\g< 1>": "x",
        r"\g<١>": "x",
    }
    assert {replacement: sub_untrusted(r"(x)", replacement, "x") for replacement in accepted} == {
        replacement: SafeRegexSubResult(value=value) for replacement, value in accepted.items()
    }
    assert sub_untrusted(r"(x)", r"\g<-1>", "x") == SafeRegexSubResult(code="regex_invalid")

    exact = sub_untrusted(
        r"(x+)",
        r"\g<+1>\g<+1>",
        "xx",
        max_output_chars=4,
    )
    over = sub_untrusted(
        r"(x+)",
        r"\g<+1>\g<+1>",
        "xx",
        max_output_chars=3,
    )
    assert exact == SafeRegexSubResult(value="xxxx")
    assert over == SafeRegexSubResult(code="regex_too_large")


def test_regex_dialect_substitution_preserves_group_replacement_semantics() -> None:
    result = sub_untrusted(
        r"(?<=\b[A-Z]{1,3})(\d+)",
        r"[\1]",
        "AB123 CD4567",
        dialect="regex",
    )

    assert result == SafeRegexSubResult(value="AB[123] CD[4567]")


@pytest.mark.parametrize(
    "replacement",
    [
        r"\x41",
        r"\u0041",
        r"\U00000041",
        r"\N{LATIN CAPITAL LETTER A}",
        r"\400",
        r"\777",
    ],
)
def test_regex_dialect_replacement_escapes_match_regex_engine(replacement: str) -> None:
    expected = regex_engine.sub(r"(x)", replacement, "x")

    result = sub_untrusted(r"(x)", replacement, "x", dialect="regex")

    assert result == SafeRegexSubResult(value=expected)


def test_regex_dialect_mixed_replacement_template_matches_regex_engine() -> None:
    replacement = r"pre-\1-\x41-\u0041-\U00000041-\N{LATIN CAPITAL LETTER A}-\400-\777-post"
    expected = regex_engine.sub(r"(x)", replacement, "x")

    result = sub_untrusted(r"(x)", replacement, "x", dialect="regex")

    assert result == SafeRegexSubResult(value=expected)


@pytest.mark.parametrize(
    "name",
    [
        "CJK UNIFIED IDEOGRAPH-4E00",
        "HANGUL JUNGSEONG O-E",
        "VARIATION SELECTOR-1",
        "KEYCAP DIGIT ONE",
    ],
)
def test_regex_dialect_replacement_parser_rejects_engine_invalid_names(name: str) -> None:
    replacement = rf"\N{{{name}}}"
    with pytest.raises((regex_engine.error, TypeError)):
        regex_engine.sub("x", replacement, "x")

    parsed = safe_regex_module._parse_replacement_template("x", replacement, 0, "regex")

    assert parsed is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [("LF", "\n"), ("lf", "\n"), ("latin capital letter a", "A")],
)
def test_regex_dialect_replacement_parser_accepts_engine_valid_names(
    name: str,
    expected: str,
) -> None:
    replacement = rf"\N{{{name}}}"

    assert regex_engine.sub("x", replacement, "x") == expected
    assert safe_regex_module._parse_replacement_template("x", replacement, 0, "regex") == (
        1,
        (),
    )


def test_regex_dialect_decoded_escape_output_boundaries() -> None:
    exact_result = sub_untrusted(
        "x",
        r"\x41",
        "xx",
        max_output_chars=2,
        dialect="regex",
    )
    over_result = sub_untrusted(
        "x",
        r"\x41",
        "xx",
        max_output_chars=1,
        dialect="regex",
    )

    assert exact_result == SafeRegexSubResult(value="AA")
    assert over_result == SafeRegexSubResult(code="regex_too_large")


@pytest.mark.parametrize(
    "replacement",
    [
        r"\x4",
        r"\xGG",
        r"\u004",
        r"\u004G",
        r"\U0000004",
        r"\U00110000",
        r"\N",
        r"\N{}",
        r"\N{NO SUCH NAME}",
        r"\N{LATIN CAPITAL LETTER A",
        r"\778",
    ],
)
def test_regex_dialect_rejects_malformed_replacement_escapes(replacement: str) -> None:
    result = sub_untrusted(r"(x)", replacement, "x", dialect="regex")

    assert result == SafeRegexSubResult(code="regex_invalid")


@pytest.mark.parametrize(
    "replacement",
    [
        r"\x41",
        r"\u0041",
        r"\U00000041",
        r"\N{LATIN CAPITAL LETTER A}",
        r"\400",
        r"\777",
    ],
)
def test_stdlib_dialect_keeps_replacement_escape_rejections(replacement: str) -> None:
    result = sub_untrusted(r"(x)", replacement, "x")

    assert result == SafeRegexSubResult(code="regex_invalid")


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


def test_sub_untrusted_honors_injected_output_cap_at_exact_boundary() -> None:
    exact_result = sub_untrusted("x", "ab", "xx", max_output_chars=4)
    over_result = sub_untrusted("x", "ab", "xx", max_output_chars=3)

    assert exact_result == SafeRegexSubResult(value="abab")
    assert over_result == SafeRegexSubResult(code="regex_too_large")


def test_sub_untrusted_honors_injected_output_cap_for_backreferences() -> None:
    exact_result = sub_untrusted(r"(a+)", r"\1\1", "aa", max_output_chars=4)
    over_result = sub_untrusted(r"(a+)", r"\1\1", "aa", max_output_chars=3)

    assert exact_result == SafeRegexSubResult(value="aaaa")
    assert over_result == SafeRegexSubResult(code="regex_too_large")


def test_sub_untrusted_default_output_cap_remains_one_million() -> None:
    result = sub_untrusted("x", "a" * 4_000, "x" * 250)

    assert result == SafeRegexSubResult(value="a" * 1_000_000)


def test_stdlib_substitution_worker_receives_bounded_request_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_pattern = r"(?P<secret>x)"
    secret_replacement = r"\g<secret>-private-replacement"
    secret_value = "private-value-x"
    requests: list[bytes] = []
    launches: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_popen = subprocess.Popen

    class _RecordingWorker:
        def __init__(self, worker: subprocess.Popen[bytes]) -> None:
            self._worker = worker

        def __getattr__(self, name: str) -> Any:
            return getattr(self._worker, name)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            payload = kwargs.get("input", args[0] if args else None)
            assert isinstance(payload, bytes)
            requests.append(payload)
            return self._worker.communicate(*args, **kwargs)

    def _recording_popen(*args: Any, **kwargs: Any) -> _RecordingWorker:
        launches.append((args, kwargs))
        return _RecordingWorker(original_popen(*args, **kwargs))

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)

    result = sub_untrusted(
        secret_pattern,
        secret_replacement,
        secret_value,
        flags=re.IGNORECASE,
        max_output_chars=64,
    )

    assert result == SafeRegexSubResult(value="private-value-x-private-replacement")
    assert len(launches) == 1
    args, kwargs = launches[0]
    argv = args[0]
    assert isinstance(argv, list)
    assert all(secret_pattern not in argument for argument in argv)
    assert all(secret_replacement not in argument for argument in argv)
    assert all(secret_value not in argument for argument in argv)
    assert kwargs["stdin"] is subprocess.PIPE
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert len(requests) == 1
    assert requests[0].endswith(b"\n")
    assert json.loads(requests[0]) == {
        "flags": int(re.IGNORECASE),
        "legacy_numeric_group_ids": sys.version_info < (3, 12),
        "max_output_chars": 64,
        "operation": "sub",
        "pattern": secret_pattern,
        "repl": secret_replacement,
        "value": secret_value,
    }


def test_stdlib_substitution_rejects_request_over_worker_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safe_regex_module, "_MAX_WORKER_REQUEST_BYTES", 64)

    result = sub_untrusted("x", "replacement", "x")

    assert result == SafeRegexSubResult(code="regex_too_large")


def test_catastrophic_stdlib_substitution_timeout_reaps_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []
    execution_timeouts: list[float] = []
    original_popen = subprocess.Popen

    class _RecordingWorker:
        def __init__(self, worker: subprocess.Popen[bytes]) -> None:
            self._worker = worker

        def __getattr__(self, name: str) -> Any:
            return getattr(self._worker, name)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            timeout = kwargs.get("timeout")
            if timeout is not None:
                execution_timeouts.append(timeout)
            return self._worker.communicate(*args, **kwargs)

    def _recording_popen(*args: Any, **kwargs: Any) -> _RecordingWorker:
        worker = _RecordingWorker(original_popen(*args, **kwargs))
        spawned.append(worker)
        return worker

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)

    try:
        result = sub_untrusted(
            r"(?:a|aa)+$",
            "x",
            "a" * 6_000 + "!",
            limits=SafeRegexLimits(timeout_s=0.020),
        )

        assert result == SafeRegexSubResult(code="regex_timeout")
        assert len(spawned) == 1
        assert execution_timeouts[0] == 0.020
        assert all(worker.returncode is not None for worker in spawned)
        assert all(worker.stdin is not None and worker.stdin.closed for worker in spawned)
        assert all(worker.stdout is not None and worker.stdout.closed for worker in spawned)
        assert not any(
            thread.name == "safe-regex-startup-reader" and thread.is_alive() for thread in threading.enumerate()
        )
    finally:
        for worker in spawned:
            if worker.returncode is None:
                worker.kill()
                worker.wait(timeout=1.0)


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


@pytest.mark.parametrize(
    ("pattern", "value", "expected_code"),
    [
        ("a" * 4_097, "sample", "regex_too_large"),
        ("a", "x" * 8_193, "regex_too_large"),
        ("[", "sample", "regex_invalid"),
    ],
    ids=["pattern-size", "input-size", "invalid-syntax"],
)
def test_regex_dialect_preserves_size_and_sanitization_codes(
    pattern: str,
    value: str,
    expected_code: str,
) -> None:
    result = search_untrusted(pattern, value, dialect="regex")

    assert result == SafeRegexResult(matched=False, code=expected_code)


def test_regex_dialect_preserves_timeout_bound_and_sanitization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, _observed = _install_fake_compile(
        monkeypatch,
        error=TimeoutError("private regex dialect detail"),
    )

    result = search_untrusted(
        r"(?<=\b[A-Z]{1,3})(\d+)",
        "AB123",
        dialect="regex",
        limits=SafeRegexLimits(timeout_s=0.025),
    )

    assert compiled.calls == [("AB123", 0.025)]
    assert result == SafeRegexResult(matched=False, code="regex_timeout")


def test_regex_dialect_substitution_passes_bounded_timeout_and_sanitizes_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _FakeSubCompiled(error=TimeoutError("private engine detail"))
    monkeypatch.setattr(safe_regex_module, "_compile_pattern", lambda *_args: compiled)

    result = sub_untrusted(
        "x",
        "replacement",
        "sample",
        dialect="regex",
        limits=SafeRegexLimits(timeout_s=0.025),
    )

    assert len(compiled.calls) == 1
    replacement, value, timeout = compiled.calls[0]
    assert callable(replacement)
    assert (value, timeout) == ("sample", 0.025)
    assert result == SafeRegexSubResult(code="regex_timeout")


def test_regex_dialect_substitution_does_not_disclose_pattern_or_engine_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_pattern = "private-sub-pattern"
    secret_error = "private-sub-error"
    compiled = _FakeSubCompiled(error=ValueError(secret_error))
    monkeypatch.setattr(safe_regex_module, "_compile_pattern", lambda *_args: compiled)
    caplog.set_level(logging.DEBUG)

    result = sub_untrusted(secret_pattern, "replacement", "sample", dialect="regex")
    captured = capsys.readouterr()
    disclosed = " ".join([repr(result), caplog.text, captured.out, captured.err])

    assert result == SafeRegexSubResult(code="regex_invalid")
    assert secret_pattern not in disclosed
    assert secret_error not in disclosed


def test_stdlib_substitution_does_not_disclose_worker_inputs_or_error(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_pattern = "private-pattern-("
    secret_replacement = r"\g<private-replacement>"
    secret_value = "private-input"
    caplog.set_level(logging.DEBUG)

    result = sub_untrusted(secret_pattern, secret_replacement, secret_value)
    captured = capsys.readouterr()
    disclosed = " ".join([repr(result), caplog.text, captured.out, captured.err])

    assert result == SafeRegexSubResult(code="regex_invalid")
    assert secret_pattern not in disclosed
    assert secret_replacement not in disclosed
    assert secret_value not in disclosed


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


def test_router_validation_checks_survivor_after_32_invalid_patterns() -> None:
    patterns = ["["] * 32 + [r"/article$"]

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

    assert cleaned["domains"]["example.com"]["url_patterns"] == [r"/article$"]


def test_router_validation_checks_survivor_after_prior_pattern_search_time(
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

    assert cleaned["domains"]["example.com"]["url_patterns"] == ["survivor"]
    assert calls == [("rejected", 0.100), ("survivor", 0.100)]


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


def test_router_validation_checks_every_configured_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(monkeypatch, elapsed_per_search=0.0)
    patterns = [f"pattern-{index}" for index in range(40)]

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": {"url_patterns": patterns}}})

    assert cleaned["domains"]["example.com"]["url_patterns"] == patterns
    assert [pattern for pattern, _timeout in calls] == patterns


def test_router_validation_preserves_per_pattern_timeout_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_router_budget_fakes(monkeypatch, elapsed_per_search=0.040)
    patterns = [f"pattern-{index}" for index in range(10)]

    cleaned = ScraperRouter.validate_rules({"domains": {"example.com": {"url_patterns": patterns}}})

    assert cleaned["domains"]["example.com"]["url_patterns"] == patterns
    assert [pattern for pattern, _timeout in calls] == patterns
    assert [timeout for _pattern, timeout in calls] == pytest.approx([0.100] * 10)


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


def test_router_keeps_the_default_stdlib_dialect_for_variable_length_lookbehind() -> None:
    pattern = r"(?<=https?://)example\.com"
    cleaned = ScraperRouter.validate_rules(
        {
            "domains": {
                "example.com": {
                    "backend": "curl",
                    "url_patterns": [pattern],
                }
            }
        }
    )

    assert "example.com" not in cleaned["domains"]


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


def test_direct_router_applies_late_match_after_32_patterns(
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

    assert plan.backend == "curl"
    assert [pattern for pattern, _timeout in calls] == patterns


def test_direct_router_preserves_per_pattern_timeout_bounds(
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
    assert [pattern for pattern, _timeout in calls] == patterns
    assert [timeout for _pattern, timeout in calls] == pytest.approx([0.100] * 10)


def test_direct_router_stops_after_match_with_per_pattern_timeout(
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
    assert [timeout for _pattern, timeout in calls] == pytest.approx([0.100, 0.100])


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


def test_generated_regex_keeps_the_default_stdlib_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_regex_llm_response(
        monkeypatch,
        {"pattern": r"(?<=\b[A-Z]{1,3})(\d+)", "group": 1},
    )

    result = _generate_regex("AB123")

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


def test_generated_regex_preserves_exact_sample_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_regex_llm_response(monkeypatch, {"pattern": "a$"})

    exact_result = _generate_regex("a" * 1_000_000)

    assert exact_result["success"] is True
    assert exact_result["sample_match"] == "a"
    assert exact_result["sample_span"] == [999_999, 1_000_000]


def test_generated_regex_skips_one_over_sample_without_searching_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = "a" * 1_000_001
    _install_regex_llm_response(
        monkeypatch,
        {"pattern": "(a)$", "flags": "i", "group": 1},
    )
    searched_values: list[str] = []
    original_search = ael.search_untrusted

    def _recording_search(pattern: str, value: str, **kwargs: Any) -> SafeRegexResult:
        searched_values.append(value)
        return original_search(pattern, value, **kwargs)

    monkeypatch.setattr(ael, "search_untrusted", _recording_search)

    result = _generate_regex(sample)

    assert result["success"] is True
    assert result["pattern"] == "(a)$"
    assert result["flags"] == "i"
    assert result["group"] == 1
    assert result["sample_status"] == "skipped_input_too_large"
    assert "sample_match" not in result
    assert "sample_span" not in result
    assert "error" not in result
    assert searched_values == [""]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"pattern": "["}, "regex_invalid"),
        ({"pattern": "a" * 4_097}, "regex_too_large"),
    ],
    ids=["invalid-pattern", "oversized-pattern"],
)
def test_generated_regex_one_over_sample_still_rejects_invalid_pattern(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    expected_error: str,
) -> None:
    sample = "a" * 1_000_001
    _install_regex_llm_response(monkeypatch, payload)
    searched_values: list[str] = []
    original_search = ael.search_untrusted

    def _recording_search(pattern: str, value: str, **kwargs: Any) -> SafeRegexResult:
        searched_values.append(value)
        return original_search(pattern, value, **kwargs)

    monkeypatch.setattr(ael, "search_untrusted", _recording_search)

    result = _generate_regex(sample)

    assert result["success"] is False
    assert result["error"] == expected_error
    assert "sample_status" not in result
    assert searched_values == [""]


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
