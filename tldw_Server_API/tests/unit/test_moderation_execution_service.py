"""Activation contracts: no parent matching, shared deadlines, atomic failure."""

from __future__ import annotations

import re
import threading
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.exception_types import MODERATION_EXECUTION_REASONS, ModerationExecutionError
from tldw_Server_API.app.core.Moderation import execution_runner
from tldw_Server_API.app.core.Moderation.execution_models import execution_scope
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy, ModerationService, PatternRule
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator

pytestmark = pytest.mark.unit


def make_service() -> ModerationService:
    service = ModerationService.__new__(ModerationService)
    service._lock = threading.RLock()
    service._policy_evaluator = PolicyEvaluator()
    service._max_scan_chars = 200000
    service._match_window_chars = 4096
    service._max_fallback_scan_chars = 800000
    service._max_replacements_per_pattern = 1000
    return service


def make_policy(**kwargs) -> ModerationPolicy:
    return ModerationPolicy(enabled=True, block_patterns=[PatternRule(re.compile("secret"), action="redact")], **kwargs)


@pytest.fixture(autouse=True)
def incident_db(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    prepare_moderation_incident_store()


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
@pytest.mark.parametrize(
    "method",
    [
        "evaluate_text",
        "check_text",
        "redact_text",
        "redact_text_with_count",
        "evaluate_action",
        "evaluate_action_with_match",
    ],
)
def test_direct_redaction_never_returns_partial_text(monkeypatch, reason, method):
    runner = Mock()
    runner.execute.side_effect = ModerationExecutionError(reason)
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    with pytest.raises(ModerationExecutionError, match=reason):
        getattr(make_service(), method)("secret", make_policy(), "output")


def test_no_parent_regex_execution():
    service = make_service()
    service._policy_evaluator = Mock(side_effect=AssertionError("parent evaluator"))
    service._policy_evaluator.evaluate_text.side_effect = AssertionError("parent evaluator")
    service._policy_evaluator.redact_text.side_effect = AssertionError("parent redaction")
    assert service.evaluate_text("a secret", make_policy(), "output").redacted_text == "a [REDACTED]"


def test_override_dispatch_preserved_with_shared_budget(monkeypatch):
    service = make_service()
    scopes = []
    real = execution_runner.get_execution_runner()

    def execute(request, scope):
        scopes.append(scope)
        return real.execute(request, scope)

    monkeypatch.setattr(
        execution_runner, "get_execution_runner", lambda: type("Runner", (), {"execute": staticmethod(execute)})()
    )
    service.redact_text = Mock(wraps=service.redact_text)
    policy = make_policy()
    with execution_scope() as scope:
        result = service.evaluate_action_with_match("secret", policy, "output")
    assert result == ("redact", "[REDACTED]", "secret", "uncategorized", (0, 6))
    assert scopes == [scope, scope]
    service.redact_text.assert_called_once_with("secret", policy, phase="output")


def test_evaluate_then_redact_does_not_reset_deadline(monkeypatch):
    service = make_service()
    runner = Mock()
    deadlines = []

    def execute(request, scope):
        from tldw_Server_API.app.core.Moderation.execution_models import WorkerResponse
        from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

        deadlines.append(scope.deadline)
        if request.operation == "evaluate":
            scope.deadline = 0.0
            return WorkerResponse("evaluate", ModerationEvaluationResult(action="redact"))
        raise ModerationExecutionError("deadline_exceeded")

    runner.execute.side_effect = execute
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    with pytest.raises(ModerationExecutionError, match="deadline_exceeded"):
        service.evaluate_text("secret", make_policy(), "output")
    assert len(deadlines) == 2
    assert deadlines[1] == 0.0


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
def test_disabled_and_noop_guards_preserve_existing_semantics(monkeypatch, reason):
    runner = Mock()
    runner.execute.side_effect = ModerationExecutionError(reason)
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    service = make_service()
    disabled = make_policy()
    disabled.enabled = False
    assert service.evaluate_text("secret", disabled, "output").action == "pass"
    assert service.redact_text("", disabled) == ""
    assert service.redact_text("secret", ModerationPolicy()) == "secret"
    assert service.redact_text("secret", make_policy(output_enabled=False), "output") == "secret"
    runner.execute.assert_not_called()


def test_disabled_redaction_still_applies_rules():
    policy = make_policy()
    policy.enabled = False
    assert make_service().redact_text("secret", policy) == "[REDACTED]"


@pytest.mark.parametrize("method", ["_evaluate_action_internal", "evaluate_action", "evaluate_action_with_match"])
def test_tuple_facades_retain_evaluate_override(method):
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    service = make_service()
    service.evaluate_text = Mock(return_value=ModerationEvaluationResult(action="warn", matched_pattern="override"))
    assert getattr(service, method)("secret", make_policy(), "input")[0] == "warn"
    service.evaluate_text.assert_called_once()


def test_reporting_failure_never_replaces_original_error(monkeypatch):
    from tldw_Server_API.app.core.Moderation import execution_incidents

    runner = Mock()
    runner.execute.side_effect = ModerationExecutionError("worker_crashed")
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    monkeypatch.setattr(
        execution_incidents, "report_execution_failure", Mock(side_effect=RuntimeError("reporter failed"))
    )
    with pytest.raises(ModerationExecutionError, match="worker_crashed"):
        make_service().evaluate_text("secret", make_policy(), "input")


@pytest.mark.asyncio
async def test_service_and_public_async_wrapper_report_once(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
    from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async
    from tldw_Server_API.app.core.Moderation.execution_models import FailureMetadata

    runner = Mock()
    runner.execute.side_effect = ModerationExecutionError("worker_crashed")
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    with pytest.raises(ModerationExecutionError):
        await run_moderation_async(
            lambda: make_service().evaluate_action("secret", make_policy(), "input"),
            metadata=FailureMetadata("evaluate", "input", 6),
        )
    store = ModerationReviewStore(str(tmp_path / "review.db"))
    assert store.list_items()["total"] == 1
    assert len(store.list_audit()["events"]) == 1


@pytest.mark.parametrize("method", ["evaluate_text", "redact_text", "redact_text_with_count"])
@pytest.mark.parametrize("rule", [None, object()])
def test_unserializable_policy_fails_with_sanitized_typed_error(method, rule):
    policy = make_policy()
    policy.block_patterns = [rule]
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        getattr(make_service(), method)("private text", policy, "input")
