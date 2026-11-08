"""Operational failure must stop current Chat, Audio and Workflow callers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.exception_types import MODERATION_EXECUTION_REASONS, ModerationExecutionError
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def incident_db(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store

    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    prepare_moderation_incident_store()


def failing_service(reason):
    service = Mock()
    service.get_effective_policy.return_value = ModerationPolicy(enabled=True)
    service.evaluate_action_with_match.side_effect = ModerationExecutionError(reason)
    service.evaluate_action.side_effect = ModerationExecutionError(reason)
    service.check_text.return_value = (False, None)
    return service


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
async def test_input_failure_never_calls_provider(reason):
    from tldw_Server_API.app.core.Chat.chat_service import moderate_input_messages

    service = failing_service(reason)
    provider = Mock()
    request_data = SimpleNamespace(messages=[SimpleNamespace(role="user", content="private text")])
    with pytest.raises(HTTPException) as caught:
        await moderate_input_messages(request_data, None, service, None, Mock(), None, None, "1")
        provider()
    assert caught.value.status_code == 503
    assert caught.value.detail["error_code"] == "moderation_unavailable"
    provider.assert_not_called()
    service.check_text.assert_not_called()


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
async def test_output_failure_never_returns_or_saves_affected_reply(reason):
    from tldw_Server_API.app.core.Chat.moderation_pipeline import (
        OutputModerationRuntime,
        apply_output_safety_to_choices,
    )

    service = failing_service(reason)
    save = Mock()
    with pytest.raises(HTTPException) as caught:
        result = await apply_output_safety_to_choices(
            choices=[],
            fallback_content="private text",
            fallback_content_text="private text",
            runtime=OutputModerationRuntime(None, "1", None, Mock(), moderation_getter=lambda: service),
        )
        save(result)
    assert caught.value.status_code == 503
    assert caught.value.detail["error_code"] == "moderation_unavailable"
    save.assert_not_called()
    service.check_text.assert_not_called()


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
async def test_compatibility_fallback_does_not_retry_failed_evaluation(reason):
    from tldw_Server_API.app.core.Chat.moderation_pipeline import _evaluate_output_moderation

    service = failing_service(reason)
    with pytest.raises(ModerationExecutionError):
        _evaluate_output_moderation(
            moderation=service, eff_policy=ModerationPolicy(enabled=True), target_text="private text"
        )
    service.check_text.assert_not_called()


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
async def test_transcript_failure_is_not_a_success_placeholder(monkeypatch, reason):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_policy
    from tldw_Server_API.tests.unit.test_moderation_execution_service import make_service

    service = make_service()
    service.redact_text = Mock(side_effect=ModerationExecutionError(reason))
    monkeypatch.setattr(stt_policy, "get_moderation_service", lambda: service)
    policy = stt_policy.STTPolicy(None, True, 0.0, True, False, [])
    with pytest.raises(ModerationExecutionError):
        stt_policy.apply_transcript_payload_policy(
            {"type": "final", "text": "private text", "segments": [{"text": "private segment"}]},
            policy=policy,
        )


@pytest.mark.parametrize("simulated", [False, True])
@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
async def test_workflow_custom_timeout_discards_base_redaction_and_stops_chaining(monkeypatch, simulated, reason):
    from tldw_Server_API.app.core.Moderation import execution_runner, moderation_service
    from tldw_Server_API.app.core.Workflows.adapters.llm import moderation
    from tldw_Server_API.tests.unit.test_moderation_execution_service import make_policy, make_service

    service = make_service()
    service.get_effective_policy = lambda uid: make_policy()
    monkeypatch.setattr(moderation_service, "get_moderation_service", lambda: service)
    owner = execution_runner.get_execution_runner()
    scopes = []

    def execute(request, scope):
        scopes.append(scope)
        if len(scopes) == 2:
            raise ModerationExecutionError(reason)
        return owner.execute(request, scope)

    runner = Mock()
    runner.execute.side_effect = execute
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    monkeypatch.setattr(moderation, "is_test_mode", lambda: simulated)
    with pytest.raises(ModerationExecutionError) as caught:
        await moderation.run_moderation_adapter(
            {"action": "redact", "text": "secret private text", "patterns": ["private"]}, {}
        )
    assert caught.value.reason == reason
    assert len(scopes) == 2
    assert scopes[0] is scopes[1]


async def test_supervised_timeout_blocks_before_guardian_fallback_or_provider():
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Chat.chat_service import moderate_input_messages

    service = failing_service("worker_crashed")
    engine = SimpleNamespace(
        build_moderation_policy_overlay=lambda uid, policy, **kwargs: policy,
        check_text_async=AsyncMock(side_effect=ModerationExecutionError("deadline_exceeded")),
    )
    with pytest.raises(HTTPException) as caught:
        await moderate_input_messages(
            SimpleNamespace(messages=[SimpleNamespace(role="user", content="private")]),
            None,
            service,
            None,
            Mock(),
            None,
            None,
            "1",
            supervised_policy_engine=engine,
            dependent_user_id="1",
        )
    assert caught.value.status_code == 503
    service.evaluate_action_with_match.assert_not_called()


async def test_moderation_test_http_failure_is_safe(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import moderation
    from tldw_Server_API.app.api.v1.schemas.moderation_schemas import ModerationTestRequest

    service = failing_service("worker_crashed")
    service.evaluate_text.side_effect = ModerationExecutionError("worker_crashed")
    monkeypatch.setattr(moderation, "get_moderation_service", lambda: service)
    with pytest.raises(HTTPException) as caught:
        await moderation.test_moderation(ModerationTestRequest(text="private"))
    assert caught.value.status_code == 503
    assert caught.value.detail["error_code"] == "moderation_unavailable"
    assert "private" not in str(caught.value.detail)


@pytest.mark.parametrize("mode", ["guardian_sync", "guardian_async", "custom", "simulation"])
async def test_pathological_guardian_and_custom_rules_are_contained(mode, tmp_path):
    import sys
    import textwrap
    from pathlib import Path

    from tldw_Server_API.tests.helpers.moderation_process_harness import run_isolated_probe
    from tldw_Server_API.tests.unit.test_moderation_process_harness import (
        test_outer_harness_timeout_kills_probe_and_grandchild,
        test_probe_hard_exit_and_invoker_disconnect_still_clean_tree,
    )

    # Never admit a hostile probe after a failed independent ownership proof.
    test_outer_harness_timeout_kills_probe_and_grandchild(tmp_path)
    test_probe_hard_exit_and_invoker_disconnect_still_clean_tree(tmp_path)
    root = str(Path(__file__).resolve().parents[3])
    code = f"import sys; sys.path.insert(0, {root!r})\n" + textwrap.dedent("""
        import asyncio, os, re, tempfile, threading, time
        from types import SimpleNamespace
        from loguru import logger
        logger.remove()
        directory = tempfile.TemporaryDirectory()
        os.environ["MODERATION_REVIEW_DB_PATH"] = directory.name + "/review.db"
        from tldw_Server_API.app.core.Moderation.execution_incidents import prepare_moderation_incident_store
        prepare_moderation_incident_store()
        from tldw_Server_API.app.core.exception_types import ModerationExecutionError
        from tldw_Server_API.app.core.Moderation import moderation_service
        from tldw_Server_API.app.core.Moderation.supervised_policy import SupervisedPolicyEngine
        from tldw_Server_API.app.core.Workflows.adapters.llm import moderation
        from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
        mode = sys.argv[1]
        engine = SupervisedPolicyEngine(None)
        engine._get_compiled_policies = lambda *a, **k: [(SimpleNamespace(action="notify", phase="input"), re.compile("(a+)+$"), None)]
        moderation_service.get_moderation_service = lambda: SimpleNamespace(
            get_effective_policy=lambda uid: ModerationPolicy(),
            redact_text_with_count=lambda text, policy: (text, 0))
        moderation.is_test_mode = lambda: mode == "simulation"
        text = "a" * 100000 + "!"
        async def main():
            ticks = []
            async def heartbeat():
                while True:
                    ticks.append(1)
                    await asyncio.sleep(0.02)
            task = asyncio.create_task(heartbeat())
            try:
                if mode.startswith("guardian"):
                    try:
                        if mode == "guardian_sync":
                            await asyncio.to_thread(engine.check_text, text, "child")
                        else:
                            await engine.check_text_async(text, "child")
                    except ModerationExecutionError as error:
                        assert error.reason == "deadline_exceeded"
                    else:
                        raise AssertionError("failed open")
                else:
                    try:
                        await moderation.run_moderation_adapter({"action": "redact", "text": text, "patterns": ["(a+)+$"]}, {})
                    except ModerationExecutionError as error:
                        assert error.reason == "deadline_exceeded"
                    else:
                        raise AssertionError("failed open")
                assert len(ticks) >= 10, len(ticks)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        asyncio.run(main())
        from tldw_Server_API.app.core.Moderation import async_execution, execution_incidents
        end = time.monotonic() + 1
        with async_execution._lock:
            active = async_execution._active
        while active and time.monotonic() < end:
            time.sleep(0.01)
            with async_execution._lock:
                active = async_execution._active
        assert active == 0
        reporter = execution_incidents._report_thread
        if reporter is not None:
            reporter.join(1)
            assert not reporter.is_alive()
        directory.cleanup()
        print("contained and responsive")
    """)
    result = run_isolated_probe([sys.executable, "-I", "-c", code, mode])
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout.strip() == b"contained and responsive"


@pytest.mark.parametrize("mode", ["start", "resume"])
async def test_workflow_moderation_failure_stops_real_engine_chain(monkeypatch, tmp_path, mode):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
    from tldw_Server_API.app.core.Moderation import execution_runner
    from tldw_Server_API.app.core.Workflows import engine as engine_module
    from tldw_Server_API.app.core.Workflows.adapters.llm import moderation

    db = WorkflowsDatabase(str(tmp_path / "workflow.db"))
    runner = Mock()
    runner.execute.side_effect = ModerationExecutionError("worker_crashed")
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    monkeypatch.setattr(moderation, "is_test_mode", lambda: True)
    monkeypatch.setattr(engine_module.WorkflowScheduler, "_inst", None)
    downstream = []
    webhook = AsyncMock()
    monkeypatch.setattr(engine_module.WorkflowEngine, "_maybe_send_completion_webhook", webhook)

    async def after(config, context):
        downstream.append(True)
        return {"text": "continued"}

    monkeypatch.setattr(
        engine_module, "get_adapter", lambda name: moderation.run_moderation_adapter if name == "moderation" else after
    )
    definition = {
        "name": "moderation-failure",
        "steps": [
            {
                "id": "moderate",
                "type": "moderation",
                "retry": 2,
                "on_failure": "after",
                "config": {"action": "redact", "text": "private text", "patterns": ["private"]},
            },
            {"id": "after", "type": "log", "config": {}},
        ],
    }
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    try:
        engine = engine_module.WorkflowEngine(db)
        if mode == "start":
            await engine.start_run("run")
        else:
            db.update_run_status("run", "waiting_human")
            await engine.continue_run(
                "run", after_step_id="", next_step_id="moderate", last_outputs={"text": "private cached snapshot"}
            )
        run = db.get_run("run")
        assert run.status == "failed"
        assert run.status_reason == "moderation_unavailable"
        assert downstream == []
        assert runner.execute.call_count == 1
        webhook.assert_not_called()
        assert run.outputs_json is None
        attempts = db.list_step_attempts(run_id="run", step_id="moderate")
        assert len(attempts) == 1
        assert attempts[0]["reason_code_core"] == "moderation_unavailable"
        assert not attempts[0]["retryable"]
        assert "private" not in str(attempts[0]["metadata_json"]["failure_envelope"])
    finally:
        db.close()


@pytest.mark.parametrize("wrapper", ["parallel", "retry", "map"])
async def test_nested_workflow_controls_propagate_moderation_failure(monkeypatch, wrapper):
    from tldw_Server_API.app.core.Moderation import execution_runner
    from tldw_Server_API.app.core.Workflows.adapters.control import flow, state
    from tldw_Server_API.app.core.Workflows.adapters.llm import moderation

    runner = Mock()
    runner.execute.side_effect = ModerationExecutionError("worker_crashed")
    monkeypatch.setattr(execution_runner, "get_execution_runner", lambda: runner)
    monkeypatch.setattr(moderation, "is_test_mode", lambda: True)
    config = {"action": "redact", "text": "private", "patterns": ["private"]}
    if wrapper == "parallel":
        monkeypatch.setattr(flow, "get_adapter", lambda name: moderation.run_moderation_adapter)
        async def call():
            return await flow.run_parallel_adapter(
                {"steps": [{"type": "moderation", "config": config}], "fail_fast": False}, {}
            )
    elif wrapper == "map":
        monkeypatch.setattr(flow, "get_adapter", lambda name: moderation.run_moderation_adapter)
        async def call():
            return await flow.run_map_adapter(
                {"items": ["private"], "step": {"type": "moderation", "config": config}}, {}
            )
    else:
        monkeypatch.setattr(state, "get_adapter", lambda name: moderation.run_moderation_adapter)
        async def call():
            return await state.run_retry_adapter(
                {"step_type": "moderation", "step_config": config, "max_retries": 2, "backoff_base": 0, "backoff_max": 0},
                {},
            )
    with pytest.raises(ModerationExecutionError):
        await call()
    assert runner.execute.call_count == 1


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
async def test_nested_workflow_call_preserves_terminal_moderation_failure(monkeypatch, reason):
    import sys
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Workflows import engine
    from tldw_Server_API.app.core.Workflows.adapters.control.orchestration import run_workflow_call_adapter

    # The legacy workflow_call DB accessor is supplied by existing adapter tests too.
    db = Mock()
    db.get_run.return_value = SimpleNamespace(
        status="failed", status_reason="moderation_unavailable", error=reason,
        outputs={"text": "private cached output"},
    )
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.workflows_db", SimpleNamespace(get_workflows_db=lambda: db))
    monkeypatch.setattr(engine.WorkflowEngine, "start_run", AsyncMock())
    with pytest.raises(ModerationExecutionError) as caught:
        await run_workflow_call_adapter({"workflow_id": "child"}, {})
    assert caught.value.reason == reason


async def test_nested_workflow_call_does_not_swallow_typed_failure(monkeypatch):
    import sys

    from tldw_Server_API.app.core.Workflows.adapters.control.orchestration import run_workflow_call_adapter

    db = Mock()
    db.get_workflow.side_effect = ModerationExecutionError("worker_crashed")
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.workflows_db", SimpleNamespace(get_workflows_db=lambda: db))
    with pytest.raises(ModerationExecutionError, match="worker_crashed"):
        await run_workflow_call_adapter({"workflow_id": "child"}, {})
