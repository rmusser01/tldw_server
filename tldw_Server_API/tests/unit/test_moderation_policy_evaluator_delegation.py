from __future__ import annotations

import inspect
import re
import threading
from unittest.mock import Mock

import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service_module
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)


def _service() -> ModerationService:
    service = ModerationService.__new__(ModerationService)
    service._lock = threading.RLock()
    service._max_scan_chars = 10
    service._match_window_chars = 5
    service._max_fallback_scan_chars = 100
    service._max_replacements_per_pattern = 2
    service._policy_evaluator = PolicyEvaluator()
    return service


def _policy(action="redact"):
    return ModerationPolicy(
        enabled=True,
        input_action="block",
        output_action="redact",
        per_user_overrides=False,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret"),
                action=action,
                replacement="[R]",
            )
        ],
    )


def test_evaluation_limits_copy_raw_values():
    service = _service()
    service._max_scan_chars = "10"
    service._match_window_chars = None
    service._max_fallback_scan_chars = object()
    service._max_replacements_per_pattern = "bad"

    limits = service._evaluation_limits()

    assert limits.max_scan_chars == "10"
    assert limits.match_window_chars is None
    assert limits.max_fallback_scan_chars is service._max_fallback_scan_chars
    assert limits.max_replacements_per_pattern == "bad"


def test_evaluation_limits_wait_for_service_lock():
    service = _service()
    started = threading.Event()
    completed = threading.Event()

    def snapshot():
        started.set()
        service._evaluation_limits()
        completed.set()

    with service._lock:
        thread = threading.Thread(target=snapshot)
        thread.start()
        assert started.wait(timeout=1)
        assert not completed.wait(timeout=0.05)
    thread.join(timeout=1)

    assert completed.is_set()


def test_evaluation_limits_never_observe_reload_partial_assignments(
    monkeypatch,
):
    service = _service()
    service._global_policy = _policy("block")
    service._user_overrides = {}
    partial_assignment = threading.Event()
    release_reload = threading.Event()
    snapshot_complete = threading.Event()
    observed = []
    load_calls = 0

    def controlled_load_global_policy():
        nonlocal load_calls
        load_calls += 1
        service._max_scan_chars = 20
        if load_calls == 1:
            partial_assignment.set()
            release_reload.wait(timeout=1)
        service._match_window_chars = 6
        service._max_fallback_scan_chars = 200
        service._max_replacements_per_pattern = 3
        return service._global_policy

    service._load_global_policy = controlled_load_global_policy
    service._load_runtime_overrides_file = lambda: None
    service._load_user_overrides = lambda: {}
    monkeypatch.setattr(
        moderation_service_module,
        "load_and_log_configs",
        lambda: {},
    )

    reload_thread = threading.Thread(target=service.reload)
    reload_thread.start()
    assert partial_assignment.wait(timeout=1)

    def snapshot():
        observed.append(service._evaluation_limits())
        snapshot_complete.set()

    snapshot_thread = threading.Thread(target=snapshot)
    snapshot_thread.start()
    assert not snapshot_complete.wait(timeout=0.05)
    release_reload.set()
    reload_thread.join(timeout=1)
    snapshot_thread.join(timeout=1)

    assert not reload_thread.is_alive()
    assert not snapshot_thread.is_alive()
    assert observed == [EvaluationLimits(20, 6, 200, 3)]


def test_service_evaluation_and_redaction_use_separate_snapshots():
    first = EvaluationLimits(10, 5, 100, 2)
    second = EvaluationLimits(20, 6, 200, 3)
    evaluator = Mock()
    evaluator.evaluate_text.return_value = ModerationEvaluationResult(
        action="redact",
        matched_pattern="secret",
        match_span=(0, 6),
    )
    evaluator.redact_text.return_value = "[R]"
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(side_effect=[first, second])

    result = service.evaluate_text("secret", _policy(), "input")

    assert result.redacted_text == "[R]"
    assert evaluator.evaluate_text.call_args.args[3] is first
    assert evaluator.evaluate_text.call_args.kwargs == {
        "include_redacted_text": False,
    }
    assert evaluator.redact_text.call_args.args[3] is second


def test_check_and_decision_only_core_do_not_invoke_public_redaction():
    service = _service()
    service.redact_text = Mock(
        side_effect=AssertionError("redaction must not run"),
    )
    policy = _policy("redact")

    assert service.check_text("secret", policy, "input") == (True, "[R]")
    decision = service._evaluate_text_core(
        "secret",
        policy,
        "input",
        include_redacted_text=False,
    )

    assert decision.action == "redact"
    assert decision.redacted_text is None
    service.redact_text.assert_not_called()


def test_private_helper_descriptors_are_preserved():
    assert isinstance(
        inspect.getattr_static(ModerationService, "_effective_rule_categories"),
        classmethod,
    )
    assert isinstance(
        inspect.getattr_static(
            ModerationService,
            "_rule_matches_enabled_categories",
        ),
        classmethod,
    )
    for name in (
        "_rule_applies_to_phase",
        "_build_sanitized_snippet",
        "_apply_rule_redactions",
    ):
        assert isinstance(
            inspect.getattr_static(ModerationService, name),
            staticmethod,
        )


def test_service_method_parameter_names_and_kinds_are_preserved():
    expected = {
        "check_text": ("self", "text", "policy", "phase"),
        "build_sanitized_snippet": (
            "self",
            "text",
            "policy",
            "match_span",
            "pattern",
        ),
        "redact_text": ("self", "text", "policy", "phase"),
        "redact_text_with_count": ("self", "text", "policy", "phase"),
        "evaluate_text": ("self", "text", "policy", "phase"),
        "_evaluate_text_core": (
            "self",
            "text",
            "policy",
            "phase",
            "include_redacted_text",
        ),
        "_evaluate_action_internal": ("self", "text", "policy", "phase"),
        "evaluate_action": ("self", "text", "policy", "phase"),
        "evaluate_action_with_match": ("self", "text", "policy", "phase"),
        "_iter_scan_chunks": ("self", "text"),
        "_find_match_span": ("self", "pat", "text"),
        "_collect_rule_matches": ("self", "text", "pat"),
        "_apply_rule_redactions": ("text", "matches", "replacement"),
    }

    for name, parameter_names in expected.items():
        descriptor = inspect.getattr_static(ModerationService, name)
        target = descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
        signature = inspect.signature(
            target,
        )
        assert tuple(signature.parameters) == parameter_names

    core_signature = inspect.signature(
        ModerationService._evaluate_text_core,
    )
    assert core_signature.parameters["include_redacted_text"].kind is inspect.Parameter.KEYWORD_ONLY
    for method_name, parameter_name in (
        ("check_text", "phase"),
        ("build_sanitized_snippet", "pattern"),
        ("redact_text", "phase"),
        ("redact_text_with_count", "phase"),
        ("evaluate_text", "phase"),
    ):
        signature = inspect.signature(
            getattr(ModerationService, method_name),
        )
        assert signature.parameters[parameter_name].default is None

    for method_name in (
        "_evaluate_action_internal",
        "evaluate_action",
        "evaluate_action_with_match",
    ):
        signature = inspect.signature(
            getattr(ModerationService, method_name),
        )
        assert signature.parameters["phase"].default is inspect.Parameter.empty
