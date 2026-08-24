from __future__ import annotations

import inspect
import re
import threading
from typing import Any
from unittest.mock import Mock, call, sentinel

import pytest

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

pytestmark = pytest.mark.unit


_OBSOLETE_POLICY_HELPER_DELEGATES = (
    "_effective_rule_categories",
    "_rule_applies_to_phase",
    "_rule_matches_enabled_categories",
    "_build_sanitized_snippet",
    "_apply_rule_redactions",
)

_OBSOLETE_SCAN_HELPER_DELEGATES = (
    "_iter_scan_chunks",
    "_find_match_span",
    "_collect_rule_matches",
)


def test_obsolete_policy_helper_delegates_are_not_class_local():
    for name in _OBSOLETE_POLICY_HELPER_DELEGATES:
        assert name not in ModerationService.__dict__


def test_obsolete_scan_helper_delegates_are_not_class_local():
    for name in _OBSOLETE_SCAN_HELPER_DELEGATES:
        assert name not in ModerationService.__dict__


def _service() -> ModerationService:
    service = ModerationService.__new__(ModerationService)
    service._lock = threading.RLock()
    service._max_scan_chars = 10
    service._match_window_chars = 5
    service._max_fallback_scan_chars = 100
    service._max_replacements_per_pattern = 2
    service._policy_evaluator = PolicyEvaluator()
    return service


def _policy(action: Any = "redact") -> ModerationPolicy:
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


def test_constructor_owns_exactly_one_stateless_policy_evaluator(
    monkeypatch,
):
    constructed = []

    class RecordingPolicyEvaluator(PolicyEvaluator):
        def __init__(self) -> None:
            constructed.append(self)

    evaluator_factory = Mock(side_effect=RecordingPolicyEvaluator)
    load_configs = Mock(return_value={})
    load_policy = Mock(return_value=ModerationPolicy())
    load_runtime_overrides = Mock(return_value=None)
    load_user_overrides = Mock(return_value={})
    getenv = Mock(return_value=None)
    monkeypatch.setattr(
        moderation_service_module,
        "PolicyEvaluator",
        evaluator_factory,
    )
    monkeypatch.setattr(
        moderation_service_module,
        "load_and_log_configs",
        load_configs,
    )
    monkeypatch.setattr(
        moderation_service_module.os,
        "getenv",
        getenv,
    )
    monkeypatch.setattr(
        ModerationService,
        "_load_global_policy",
        load_policy,
    )
    monkeypatch.setattr(
        ModerationService,
        "_load_runtime_overrides_file",
        load_runtime_overrides,
    )
    monkeypatch.setattr(
        ModerationService,
        "_load_user_overrides",
        load_user_overrides,
    )

    service = ModerationService()

    evaluator_factory.assert_called_once_with()
    assert constructed == [service._policy_evaluator]
    assert service._policy_evaluator is constructed[0]
    assert isinstance(service._policy_evaluator, PolicyEvaluator)
    assert vars(service._policy_evaluator) == {}
    load_configs.assert_called_once_with()
    assert load_policy.call_args_list == [call(), call()]
    load_runtime_overrides.assert_called_once_with()
    load_user_overrides.assert_called_once_with()
    assert getenv.call_args_list == [
        call("MODERATION_MAX_SCAN_CHARS"),
        call("MODERATION_MAX_REPLACEMENTS_PER_PATTERN"),
        call("MODERATION_MATCH_WINDOW_CHARS"),
        call("MODERATION_MAX_FALLBACK_SCAN_CHARS"),
        call("MODERATION_BLOCKLIST_WRITE_DEBOUNCE_MS"),
    ]


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

    def snapshot() -> None:
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

    def controlled_load_global_policy() -> ModerationPolicy:
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

    def snapshot() -> None:
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


def test_build_sanitized_snippet_delegates_exactly_once():
    evaluator = Mock()
    evaluator.build_sanitized_snippet.return_value = sentinel.built_snippet
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(
        side_effect=AssertionError("snapshot must not run"),
    )

    result = service.build_sanitized_snippet(
        sentinel.snippet_text,
        sentinel.snippet_policy,
        sentinel.snippet_span,
        sentinel.snippet_pattern,
    )

    assert result is sentinel.built_snippet
    evaluator.build_sanitized_snippet.assert_called_once_with(
        sentinel.snippet_text,
        sentinel.snippet_policy,
        sentinel.snippet_span,
        sentinel.snippet_pattern,
    )
    service._evaluation_limits.assert_not_called()


def test_redact_text_delegates_exactly_once_with_one_snapshot():
    evaluator = Mock()
    evaluator.redact_text.return_value = sentinel.redacted_text
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(return_value=sentinel.redact_limits)

    result = service.redact_text(
        sentinel.redact_text_input,
        sentinel.redact_policy,
        sentinel.redact_phase,
    )

    assert result is sentinel.redacted_text
    service._evaluation_limits.assert_called_once_with()
    evaluator.redact_text.assert_called_once_with(
        sentinel.redact_text_input,
        sentinel.redact_policy,
        sentinel.redact_phase,
        sentinel.redact_limits,
    )


def test_redact_text_with_count_delegates_exactly_once_with_one_snapshot():
    evaluator = Mock()
    evaluator.redact_text_with_count.return_value = sentinel.redaction_with_count
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(return_value=sentinel.count_limits)

    result = service.redact_text_with_count(
        sentinel.count_text,
        sentinel.count_policy,
        sentinel.count_phase,
    )

    assert result is sentinel.redaction_with_count
    service._evaluation_limits.assert_called_once_with()
    evaluator.redact_text_with_count.assert_called_once_with(
        sentinel.count_text,
        sentinel.count_policy,
        sentinel.count_phase,
        sentinel.count_limits,
    )


def test_decision_only_evaluation_delegates_exactly_once_with_one_snapshot():
    evaluator = Mock()
    evaluator.evaluate_text.return_value = sentinel.decision_only_result
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(return_value=sentinel.decision_limits)
    service.redact_text = Mock(
        side_effect=AssertionError("public redaction must not run"),
    )

    result = service._evaluate_text_core(
        sentinel.decision_only_text,
        sentinel.decision_only_policy,
        sentinel.decision_only_phase,
        include_redacted_text=False,
    )

    assert result is sentinel.decision_only_result
    service._evaluation_limits.assert_called_once_with()
    evaluator.evaluate_text.assert_called_once_with(
        sentinel.decision_only_text,
        sentinel.decision_only_policy,
        sentinel.decision_only_phase,
        sentinel.decision_limits,
        include_redacted_text=False,
    )
    service.redact_text.assert_not_called()


def test_service_evaluation_and_redaction_use_separate_snapshots():
    first = EvaluationLimits(10, 5, 100, 2)
    second = EvaluationLimits(20, 6, 200, 3)
    evaluator = Mock()
    decision = ModerationEvaluationResult(
        action="redact",
        matched_pattern=sentinel.decision_pattern,
        category=sentinel.decision_category,
        match_span=sentinel.decision_span,
        sample=sentinel.decision_sample,
    )
    decision_before = vars(decision).copy()
    evaluator.evaluate_text.return_value = decision
    evaluator.redact_text.return_value = sentinel.public_redacted_text
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(side_effect=[first, second])
    service.redact_text = Mock(wraps=service.redact_text)

    result = service.evaluate_text(
        sentinel.decision_text,
        sentinel.decision_policy,
        sentinel.decision_phase,
    )

    assert service._evaluation_limits.call_args_list == [call(), call()]
    evaluator.evaluate_text.assert_called_once_with(
        sentinel.decision_text,
        sentinel.decision_policy,
        sentinel.decision_phase,
        first,
        include_redacted_text=False,
    )
    service.redact_text.assert_called_once_with(
        sentinel.decision_text,
        sentinel.decision_policy,
        phase=sentinel.decision_phase,
    )
    evaluator.redact_text.assert_called_once_with(
        sentinel.decision_text,
        sentinel.decision_policy,
        sentinel.decision_phase,
        second,
    )
    assert evaluator.mock_calls == [
        call.evaluate_text(
            sentinel.decision_text,
            sentinel.decision_policy,
            sentinel.decision_phase,
            first,
            include_redacted_text=False,
        ),
        call.redact_text(
            sentinel.decision_text,
            sentinel.decision_policy,
            sentinel.decision_phase,
            second,
        ),
    ]
    assert vars(decision) == decision_before
    assert result is not decision
    assert result.action is decision.action
    assert result.redacted_text is sentinel.public_redacted_text
    assert result.matched_pattern is decision.matched_pattern
    assert result.category is decision.category
    assert result.match_span is decision.match_span
    assert result.sample is decision.sample


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
