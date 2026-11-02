"""Benign worker parity; real adversarial containment belongs to Task 2."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation import regex_worker
from tldw_Server_API.app.core.Moderation.execution_models import (
    ExecutionBounds,
    OutputBudget,
    RegexRuleSnapshot,
    SubstitutionRuleSnapshot,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import (
    decode_response,
    encode_request,
    snapshot_request,
    snapshot_substitution_request,
    snapshot_supervised_request,
)
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy, PatternRule
from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits, PolicyEvaluator
from tldw_Server_API.app.core.Moderation.supervised_evaluator import evaluate_supervised

LIMITS = EvaluationLimits(100, 10, 1000, None)


def base_request(
    operation="evaluate", text="hello hello", *, pattern="hello", replacement="[R]", limits=LIMITS, bounds=None
):
    policy = ModerationPolicy(
        enabled=True,
        block_patterns=[PatternRule(re.compile(pattern), action="redact", replacement=replacement, categories={"pii"})],
    )
    return snapshot_request(operation, text, policy, "input", limits, bounds or ExecutionBounds()), policy


@pytest.mark.parametrize("operation", ["evaluate", "redact", "redact_with_count"])
@pytest.mark.parametrize("phase", [None, "input", "output", "other"])
def test_worker_matches_evaluator_characterization(operation, phase):
    from dataclasses import replace

    request, policy = base_request(operation)
    request = replace(request, phase=phase)
    evaluator = PolicyEvaluator()
    if operation == "evaluate":
        expected = evaluator.evaluate_text(request.text, policy, phase, LIMITS, include_redacted_text=True)
    else:
        method = evaluator.redact_text if operation == "redact" else evaluator.redact_text_with_count
        expected = method(request.text, policy, phase, LIMITS)
    response = regex_worker.execute_request(request)
    assert response.reason is None
    assert response.result == expected


def test_literal_backslash_replacement_stays_literal():
    request, _ = base_request("redact", replacement=r"\1-literal")
    assert regex_worker.execute_request(request).result == r"\1-literal \1-literal"


@pytest.mark.parametrize("raw_count", [None, 0, -1, 2, "2", "bad"])
@pytest.mark.parametrize("scan", [1, 100])
def test_zero_length_and_raw_count_limits_keep_legacy_behavior(raw_count, scan):
    limits = EvaluationLimits(scan, 0, 1000, raw_count)
    request, policy = base_request("redact_with_count", "aa", pattern=r"(?=a)", limits=limits)
    try:
        expected = PolicyEvaluator().redact_text_with_count("aa", policy, "input", limits)
    except (TypeError, ValueError):
        assert regex_worker.execute_request(request).reason == "worker_crashed"
    else:
        assert regex_worker.execute_request(request).result == expected


def test_supervised_priority_first_tie_early_block_and_notify_are_preserved():
    rules = [RegexRuleSnapshot("late", 0, "notify", "both"), RegexRuleSnapshot("early", 0, "notify", "both")]
    result = evaluate_supervised("early late", rules, "input")
    assert (result.winner_index, result.match_span) == (0, (6, 10))
    rules += [RegexRuleSnapshot("late", 0, "block", "both"), RegexRuleSnapshot("early", 0, "block", "both")]
    request = snapshot_supervised_request("early late", rules, "input", ExecutionBounds())
    result = regex_worker.execute_request(request).result
    assert (result.winner_index, result.match_span, result.redacted_text) == (2, (6, 10), None)


def test_custom_substitution_preserves_templates_counts_and_invalid_rule_skips():
    rules = [
        SubstitutionRuleSnapshot("(", 0, "bad"),
        SubstitutionRuleSnapshot("(a)", 0, r"\9"),
        SubstitutionRuleSnapshot("(a)", 0, r"\1\1"),
        SubstitutionRuleSnapshot("aa", 0, "b"),
    ]
    request = snapshot_substitution_request("a a", rules, None, ExecutionBounds())
    result = regex_worker.execute_request(request).result
    assert (result.text, result.count, result.skipped_rule_indices) == ("b b", 4, (0, 1))


def test_template_group_expansion_is_sized_before_allocation(monkeypatch):
    request = snapshot_substitution_request(
        "a" * 100, [SubstitutionRuleSnapshot("(a+)", 0, r"\1" * 100)], None, ExecutionBounds(max_output_chars=200)
    )
    response = regex_worker.execute_request(request)
    assert response.reason == "size_exceeded"


def test_intermediate_expansion_aborts_before_join():
    request, _ = base_request(
        "redact", "aa", pattern="a", replacement="x" * 100, bounds=ExecutionBounds(max_output_chars=150)
    )
    assert regex_worker.execute_request(request).reason == "size_exceeded"


def test_debug_flag_cannot_write_protocol_or_stderr(capfd):
    request = snapshot_substitution_request(
        "hi", [SubstitutionRuleSnapshot("hi", int(re.DEBUG), "ok")], None, ExecutionBounds()
    )
    assert regex_worker.execute_request(request).result.text == "ok"
    assert capfd.readouterr() == ("", "")


def test_main_isolated_benign_child_resource_posture_and_imports():
    request, _ = base_request()
    child = subprocess.run(
        [sys.executable, "-I", "-S", str(Path(regex_worker.__file__).resolve())],
        input=encode_request(request),
        capture_output=True,
        timeout=10,
        check=True,
    )
    response = decode_response(child.stdout, request)
    assert response.result.action == "redact"
    assert child.stderr == b""
    assert response.resource_limits["core"] == (sys.platform != "win32")
    if sys.platform.startswith("linux"):
        assert all(response.resource_limits.values())


def test_optional_budget_raises_without_changing_default_semantics():
    request, policy = base_request("redact", "aa", pattern="a", replacement="xxxx")
    assert PolicyEvaluator().redact_text(request.text, policy, "input", LIMITS) == "xxxxxxxx"
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        PolicyEvaluator().redact_text(request.text, policy, "input", LIMITS, output_budget=OutputBudget(7))


def test_worker_child_import_graph_excludes_application_initializers():
    request, _ = base_request()
    code = (
        "import runpy,sys; "
        f"entry={str(Path(regex_worker.__file__).resolve())!r}; "
        "namespace=runpy.run_path(entry,run_name='__main__')"
    )
    # Inspect after main exits, in that child; no limits are set in pytest.
    code = code.replace(
        "namespace=runpy.run_path(entry,run_name='__main__')",
        "\ntry:\n runpy.run_path(entry,run_name='__main__')\n"
        "except SystemExit as e:\n assert e.code == 0\n"
        "assert 'tldw_Server_API.app.core.testing' not in sys.modules\n"
        "assert 'regex' not in sys.modules\n"
        "assert 'fastapi' not in sys.modules\n"
        "assert 'tldw_Server_API.app.core.Web_Scraping.safe_regex' not in sys.modules\n",
    )
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code], input=encode_request(request), capture_output=True, timeout=10
    )
    assert child.returncode == 0, child.stderr.decode()
    assert decode_response(child.stdout, request).reason is None


@pytest.mark.parametrize("template", [r"\1", r"\g<name>", r"\g<0>", r"\101", r"\n", r"\\", r"\&", "literal"])
def test_standalone_parser_template_parity(template):
    text = "ab a"
    rule = SubstitutionRuleSnapshot(r"(?P<name>a)(b)?", 0, template)
    request = snapshot_substitution_request(text, [rule], None, ExecutionBounds())
    result = regex_worker.execute_request(request).result
    assert (result.text, result.count) == re.subn(rule.pattern, template, text)


def test_no_match_invalid_template_is_still_skipped():
    request = snapshot_substitution_request("hello", [SubstitutionRuleSnapshot("x", 0, r"\9")], None, ExecutionBounds())
    assert regex_worker.execute_request(request).result.skipped_rule_indices == (0,)


def test_expansion_guard_precedes_match_expand(monkeypatch):
    compile_pattern = re.compile
    expansions = []

    class TrackedMatch:
        def __init__(self, match):
            self.match = match

        def span(self, *args):
            return self.match.span(*args)

        def expand(self, template):
            expansions.append(template)
            return self.match.expand(template)

    class TrackedPattern:
        def __init__(self, pattern, flags):
            self.pattern = compile_pattern(pattern, flags)
            self.groups = self.pattern.groups
            self.groupindex = self.pattern.groupindex

        def finditer(self, text):
            return (TrackedMatch(match) for match in self.pattern.finditer(text))

    monkeypatch.setattr(regex_worker.re, "compile", TrackedPattern)
    with pytest.raises(ModerationExecutionError) as error:
        regex_worker._substitute("a" * 100, [SubstitutionRuleSnapshot("(a+)", 0, r"\1" * 100)], OutputBudget(200))
    assert error.value.reason == "size_exceeded"
    assert expansions == []


def test_child_reports_limit_failure_without_decoding_policy():
    if sys.platform == "win32":
        pytest.skip("POSIX resource installation failure")
    request, _ = base_request("redact")
    code = (
        "import runpy,resource;\n"
        "def fail(*args):\n raise OSError('private diagnostic must not escape')\n"
        "resource.setrlimit=fail\n"
        f"runpy.run_path({str(Path(regex_worker.__file__).resolve())!r},run_name='__main__')\n"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        input=encode_request(request),
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert child.stderr == b""
    assert decode_response(child.stdout, request).reason == "worker_unavailable"


@pytest.mark.parametrize("operation", ["redact", "redact_with_count"])
def test_budget_also_checks_disabled_phase_and_no_rule_outputs(operation):
    method = getattr(PolicyEvaluator(), "redact_text" if operation == "redact" else "redact_text_with_count")
    for policy in (ModerationPolicy(), ModerationPolicy(input_enabled=False)):
        with pytest.raises(ModerationExecutionError, match="size_exceeded"):
            method("hello", policy, "input", LIMITS, output_budget=OutputBudget(1))


@pytest.mark.parametrize(
    "operation", ["evaluate", "redact", "redact_with_count", "supervised_evaluate", "custom_substitute"]
)
def test_all_five_variants_round_trip_success_and_failure(operation):
    from tldw_Server_API.app.core.Moderation.execution_protocol import decode_request, encode_response

    if operation == "supervised_evaluate":
        request = snapshot_supervised_request(
            "hi", [RegexRuleSnapshot("hi", 0, "redact", "both")], None, ExecutionBounds()
        )
    elif operation == "custom_substitute":
        request = snapshot_substitution_request(
            "hi", [SubstitutionRuleSnapshot("hi", 0, "bye")], None, ExecutionBounds()
        )
    else:
        request, _ = base_request(operation)
    request = decode_request(encode_request(request))
    response = regex_worker.execute_request(request)
    assert response.reason is None
    assert decode_response(encode_response(response), request) == response


def test_child_rejects_unverified_infinite_hard_resource_limit():
    if sys.platform == "win32":
        pytest.skip("POSIX resource verification")
    request, _ = base_request("redact")
    code = (
        "import runpy,resource;\n"
        "resource.setrlimit=lambda *args: None\n"
        "resource.RLIM_INFINITY=-1\n"
        "resource.getrlimit=lambda key: (0,resource.RLIM_INFINITY)\n"
        f"runpy.run_path({str(Path(regex_worker.__file__).resolve())!r},run_name='__main__')\n"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        input=encode_request(request),
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert child.stderr == b""
    assert decode_response(child.stdout, request).reason == "worker_unavailable"


def test_child_response_byte_cap_includes_resource_posture():
    from dataclasses import replace

    from tldw_Server_API.app.core.Moderation.execution_protocol import encode_response

    request, _ = base_request("redact", text="hello " * 100)
    cap = len(encode_response(regex_worker.execute_request(request)))
    request = replace(request, bounds=replace(request.bounds, max_response_bytes=cap))
    child = subprocess.run(
        [sys.executable, "-I", "-S", str(Path(regex_worker.__file__).resolve())],
        input=encode_request(request),
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert len(child.stdout) <= cap
    assert decode_response(child.stdout, request).reason == "size_exceeded"


def test_child_never_exceeds_even_an_unreportably_small_response_cap():
    request, _ = base_request("redact", bounds=ExecutionBounds(max_response_bytes=1))
    child = subprocess.run(
        [sys.executable, "-I", "-S", str(Path(regex_worker.__file__).resolve())],
        input=encode_request(request),
        capture_output=True,
        timeout=10,
    )
    assert child.returncode != 0
    assert child.stdout == child.stderr == b""
