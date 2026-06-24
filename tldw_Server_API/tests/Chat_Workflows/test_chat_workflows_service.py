import json
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat_Workflows.service import (
    ChatWorkflowConflictError,
    ChatWorkflowService,
)
from tldw_Server_API.app.core.Chat_Workflows.question_renderer import (
    ChatWorkflowQuestionRenderer,
)
from tldw_Server_API.app.core.DB_Management.ChatWorkflows_DB import ChatWorkflowsDatabase


@pytest.fixture
def fake_chat_workflows_db(tmp_path: Path) -> ChatWorkflowsDatabase:
    return ChatWorkflowsDatabase(
        db_path=tmp_path / "chat_workflows.db",
        client_id="test",
    )


def _dialogue_template(
    *,
    max_rounds: int = 4,
    extra_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    steps = [
        {
            "id": "debate",
            "step_index": 0,
            "step_type": "dialogue_round_step",
            "base_question": "State your thesis.",
            "question_mode": "stock",
            "context_refs": [],
            "dialogue_config": {
                "goal_prompt": "Stress-test the thesis.",
                "opening_prompt_mode": "base_question",
                "user_role_label": "Author",
                "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                "max_rounds": max_rounds,
                "finish_conditions": ["clear compromise"],
                "context_refs": [],
                "debate_instruction_prompt": "Challenge weak assumptions.",
                "moderator_instruction_prompt": "Return structured control output only.",
            },
        }
    ]
    steps.extend(extra_steps or [])
    return {
        "id": 30,
        "title": "Socratic",
        "version": 1,
        "steps": steps,
    }


def test_start_run_uses_template_snapshot(fake_chat_workflows_db):
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)

    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 3,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What do you want?",
                    "question_mode": "stock",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    assert run["template_version"] == 3
    assert run["current_step_index"] == 0
    assert json.loads(run["template_snapshot_json"])["steps"][0]["id"] == "goal"


def test_start_run_preserves_step_context_refs_from_saved_template_rows(fake_chat_workflows_db):
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)

    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 3,
            "steps": [
                {
                    "step_id": "goal",
                    "step_index": 0,
                    "base_question": "What do you want?",
                    "question_mode": "stock",
                    "context_refs_json": '[{"kind":"note","id":"note-1"}]',
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    snapshot = json.loads(run["template_snapshot_json"])

    assert snapshot["steps"][0]["context_refs"] == [{"kind": "note", "id": "note-1"}]


def test_renderer_falls_back_to_base_question_on_error(fake_chat_workflows_db):
    class FailingRenderer:
        async def render_question(self, **kwargs):
            raise RuntimeError("provider offline with secret-token")

    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=FailingRenderer(),
    )
    question = service._render_question_sync(
        step={"base_question": "What is your goal?", "question_mode": "llm_phrased"},
        prior_answers=[],
        context_snapshot=[],
    )

    assert question["displayed_question"] == "What is your goal?"
    assert question["fallback_used"] is True
    assert question["question_generation_meta"]["mode"] == "fallback"
    assert question["question_generation_meta"]["reason"] == "renderer_error"
    assert question["question_generation_meta"]["error_type"] == "RuntimeError"
    assert "secret-token" not in json.dumps(question["question_generation_meta"])


def test_missing_llm_phrased_renderer_reports_step_model_fallback(
    fake_chat_workflows_db: ChatWorkflowsDatabase,
) -> None:
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)
    question = service._render_question_sync(
        step={
            "base_question": "What is your goal?",
            "question_mode": "llm_phrased",
            "model": "gpt-test",
        },
        prior_answers=[],
        context_snapshot=[],
    )

    assert question["displayed_question"] == "What is your goal?"
    assert question["fallback_used"] is True
    assert question["question_generation_meta"] == {
        "mode": "fallback",
        "reason": "renderer_unavailable",
        "model": "gpt-test",
    }


@pytest.mark.asyncio
async def test_default_llm_phrased_renderer_reports_unavailable_fallback(
    fake_chat_workflows_db: ChatWorkflowsDatabase,
) -> None:
    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=ChatWorkflowQuestionRenderer(),
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 1,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What do you want?",
                    "question_mode": "llm_phrased",
                    "phrasing_instructions": "Make this warmer.",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    current_step = await service.get_current_step(run["run_id"])

    assert current_step["displayed_question"] == "What do you want?"
    assert current_step["fallback_used"] is True
    assert current_step["question_generation_meta"] == {
        "mode": "fallback",
        "reason": "renderer_unavailable",
        "model": None,
    }


@pytest.mark.asyncio
async def test_get_current_step_reuses_rendered_question(fake_chat_workflows_db):
    class CountingRenderer:
        def __init__(self):
            self.calls = 0

        async def render_question(self, **kwargs):
            self.calls += 1
            return {
                "displayed_question": f"Rendered question #{self.calls}",
                "question_generation_meta": {"calls": self.calls},
                "fallback_used": False,
            }

    renderer = CountingRenderer()
    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=renderer,
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 1,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What do you want?",
                    "question_mode": "llm_phrased",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    first = await service.get_current_step(run["run_id"])
    second = await service.get_current_step(run["run_id"])

    assert first["displayed_question"] == "Rendered question #1"
    assert second["displayed_question"] == "Rendered question #1"
    assert renderer.calls == 1


@pytest.mark.asyncio
async def test_record_answer_completes_run_after_last_step(fake_chat_workflows_db):
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 1,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What is your goal?",
                    "question_mode": "stock",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    result = await service.record_answer(
        run_id=run["run_id"],
        step_index=0,
        answer_text="Ship a feature",
    )

    answers = fake_chat_workflows_db.list_answers(run["run_id"])

    assert result["status"] == "completed"
    assert result["current_step_index"] == 1
    assert answers[0]["displayed_question"] == "What is your goal?"


@pytest.mark.asyncio
async def test_record_answer_rejects_stale_step_submission(fake_chat_workflows_db):
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 1,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What is your goal?",
                    "question_mode": "stock",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    await service.record_answer(
        run_id=run["run_id"],
        step_index=0,
        answer_text="Ship a feature",
    )

    with pytest.raises(ValueError, match="step submission"):
        await service.record_answer(
            run_id=run["run_id"],
            step_index=0,
            answer_text="Try again",
        )


@pytest.mark.asyncio
async def test_record_answer_replays_matching_idempotent_submission(fake_chat_workflows_db):
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 1,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What is your goal?",
                    "question_mode": "stock",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    first = await service.record_answer(
        run_id=run["run_id"],
        step_index=0,
        answer_text="Ship a feature",
        idempotency_key="answer-1",
    )
    replay = await service.record_answer(
        run_id=run["run_id"],
        step_index=0,
        answer_text="Ship a feature",
        idempotency_key="answer-1",
    )

    assert first["status"] == "completed"
    assert replay["status"] == "completed"
    assert len(fake_chat_workflows_db.list_answers(run["run_id"])) == 1


@pytest.mark.asyncio
async def test_record_answer_rejects_idempotency_key_reuse_for_different_answer(fake_chat_workflows_db):
    service = ChatWorkflowService(db=fake_chat_workflows_db, question_renderer=None)
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 10,
            "title": "Discovery",
            "version": 1,
            "steps": [
                {
                    "id": "goal",
                    "step_index": 0,
                    "base_question": "What is your goal?",
                    "question_mode": "stock",
                    "context_refs": [],
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    await service.record_answer(
        run_id=run["run_id"],
        step_index=0,
        answer_text="Ship a feature",
        idempotency_key="answer-1",
    )

    with pytest.raises(ChatWorkflowConflictError, match="different answer"):
        await service.record_answer(
            run_id=run["run_id"],
            step_index=0,
            answer_text="Ship something else",
            idempotency_key="answer-1",
        )


@pytest.mark.asyncio
async def test_respond_to_round_continues_same_step(fake_chat_workflows_db):
    class FakeDialogueOrchestrator:
        async def run_round(self, **kwargs):
            return {
                "debate_llm_message": "Counterargument",
                "moderator_decision": "continue",
                "moderator_summary": "Push harder on the weakest premise.",
                "next_user_prompt": "Defend your evidence.",
            }

    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=FakeDialogueOrchestrator(),
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 20,
            "title": "Socratic",
            "version": 1,
            "steps": [
                {
                    "id": "debate",
                    "step_index": 0,
                    "step_type": "dialogue_round_step",
                    "base_question": "State your thesis.",
                    "question_mode": "stock",
                    "context_refs": [],
                    "dialogue_config": {
                        "goal_prompt": "Stress-test the thesis.",
                        "opening_prompt_mode": "base_question",
                        "user_role_label": "Author",
                        "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "max_rounds": 4,
                        "finish_conditions": ["clear compromise"],
                        "context_refs": [],
                        "debate_instruction_prompt": "Challenge weak assumptions.",
                        "moderator_instruction_prompt": "Return structured control output only.",
                    },
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    result = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    assert result["status"] == "active"
    assert result["current_step_index"] == 0
    assert result["current_step_kind"] == "dialogue_round_step"
    assert result["current_round_index"] == 1
    assert result["current_prompt"] == "Defend your evidence."
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["moderator_decision"] == "continue"


@pytest.mark.asyncio
async def test_respond_to_round_replays_completed_round_with_same_idempotency_key(
    fake_chat_workflows_db,
):
    class CountingDialogueOrchestrator:
        def __init__(self):
            self.calls = 0

        async def run_round(self, **kwargs):
            self.calls += 1
            return {
                "debate_llm_message": "Counterargument",
                "moderator_decision": "continue",
                "moderator_summary": "Push harder on the weakest premise.",
                "next_user_prompt": "Defend your evidence.",
            }

    orchestrator = CountingDialogueOrchestrator()
    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=orchestrator,
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template=_dialogue_template(),
        source_mode="saved_template",
        selected_context_refs=[],
    )

    first = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )
    replay = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    rounds = fake_chat_workflows_db.list_rounds(run["run_id"], 0)

    assert first["current_round_index"] == 1
    assert replay["current_round_index"] == 1
    assert orchestrator.calls == 1
    assert len(rounds) == 1


@pytest.mark.asyncio
async def test_respond_to_round_replays_final_round_after_run_completion(
    fake_chat_workflows_db,
):
    class CountingDialogueOrchestrator:
        def __init__(self):
            self.calls = 0

        async def run_round(self, **kwargs):
            self.calls += 1
            return {
                "debate_llm_message": "Counterargument",
                "moderator_decision": "finish",
                "moderator_summary": "The thesis has been adequately tested.",
                "next_user_prompt": None,
            }

    orchestrator = CountingDialogueOrchestrator()
    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=orchestrator,
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template=_dialogue_template(),
        source_mode="saved_template",
        selected_context_refs=[],
    )

    first = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )
    replay = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    assert first["status"] == "completed"
    assert replay["status"] == "completed"
    assert orchestrator.calls == 1
    assert len(fake_chat_workflows_db.list_rounds(run["run_id"], 0)) == 1


@pytest.mark.asyncio
async def test_respond_to_round_rejects_idempotency_key_reuse_for_different_message(
    fake_chat_workflows_db,
):
    class CountingDialogueOrchestrator:
        async def run_round(self, **kwargs):
            return {
                "debate_llm_message": "Counterargument",
                "moderator_decision": "continue",
                "moderator_summary": "Push harder on the weakest premise.",
                "next_user_prompt": "Defend your evidence.",
            }

    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=CountingDialogueOrchestrator(),
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template=_dialogue_template(),
        source_mode="saved_template",
        selected_context_refs=[],
    )

    await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    with pytest.raises(ChatWorkflowConflictError, match="different dialogue round"):
        await service.respond_to_round(
            run_id=run["run_id"],
            round_index=0,
            user_message="A different message.",
            idempotency_key="round-1",
        )


@pytest.mark.asyncio
async def test_respond_to_round_rejects_idempotency_key_reuse_in_later_dialogue_step(
    fake_chat_workflows_db: ChatWorkflowsDatabase,
) -> None:
    class CountingDialogueOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def run_round(self, **kwargs: Any) -> dict[str, str | None]:
            self.calls += 1
            return {
                "debate_llm_message": "Counterargument",
                "moderator_decision": "finish",
                "moderator_summary": "The thesis has been adequately tested.",
                "next_user_prompt": None,
            }

    orchestrator = CountingDialogueOrchestrator()
    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=orchestrator,
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template=_dialogue_template(
            extra_steps=[
                {
                    "id": "second-debate",
                    "step_index": 1,
                    "step_type": "dialogue_round_step",
                    "base_question": "Restate your thesis.",
                    "question_mode": "stock",
                    "context_refs": [],
                    "dialogue_config": {
                        "goal_prompt": "Stress-test the revised thesis.",
                        "opening_prompt_mode": "base_question",
                        "user_role_label": "Author",
                        "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "max_rounds": 4,
                        "finish_conditions": ["clear compromise"],
                        "context_refs": [],
                        "debate_instruction_prompt": "Challenge weak assumptions.",
                        "moderator_instruction_prompt": "Return structured control output only.",
                    },
                }
            ]
        ),
        source_mode="saved_template",
        selected_context_refs=[],
    )

    first = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    assert first["current_step_index"] == 1

    with pytest.raises(ChatWorkflowConflictError, match="different dialogue round"):
        await service.respond_to_round(
            run_id=run["run_id"],
            round_index=0,
            user_message="My thesis is sound.",
            idempotency_key="round-1",
        )

    assert orchestrator.calls == 1


@pytest.mark.asyncio
async def test_respond_to_round_finish_advances_to_next_step(fake_chat_workflows_db):
    class FakeDialogueOrchestrator:
        async def run_round(self, **kwargs):
            return {
                "debate_llm_message": "Your premise fails.",
                "moderator_decision": "finish",
                "moderator_summary": "The thesis has been adequately tested.",
                "next_user_prompt": None,
            }

    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=FakeDialogueOrchestrator(),
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 21,
            "title": "Socratic",
            "version": 1,
            "steps": [
                {
                    "id": "debate",
                    "step_index": 0,
                    "step_type": "dialogue_round_step",
                    "base_question": "State your thesis.",
                    "question_mode": "stock",
                    "context_refs": [],
                    "dialogue_config": {
                        "goal_prompt": "Stress-test the thesis.",
                        "opening_prompt_mode": "base_question",
                        "user_role_label": "Author",
                        "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "max_rounds": 4,
                        "finish_conditions": ["clear compromise"],
                        "context_refs": [],
                        "debate_instruction_prompt": "Challenge weak assumptions.",
                        "moderator_instruction_prompt": "Return structured control output only.",
                    },
                },
                {
                    "id": "reflection",
                    "step_index": 1,
                    "step_type": "question_step",
                    "base_question": "What changed after the dialogue?",
                    "question_mode": "stock",
                    "context_refs": [],
                },
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    result = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    assert result["status"] == "active"
    assert result["current_step_index"] == 1
    assert result["current_step_kind"] == "question_step"
    assert result["current_step"]["displayed_question"] == "What changed after the dialogue?"


@pytest.mark.asyncio
async def test_respond_to_round_marks_round_failed_when_orchestrator_errors(fake_chat_workflows_db):
    class FailingDialogueOrchestrator:
        async def run_round(self, **kwargs):
            raise RuntimeError("provider offline")

    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=FailingDialogueOrchestrator(),
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 22,
            "title": "Socratic",
            "version": 1,
            "steps": [
                {
                    "id": "debate",
                    "step_index": 0,
                    "step_type": "dialogue_round_step",
                    "base_question": "State your thesis.",
                    "question_mode": "stock",
                    "context_refs": [],
                    "dialogue_config": {
                        "goal_prompt": "Stress-test the thesis.",
                        "opening_prompt_mode": "base_question",
                        "user_role_label": "Author",
                        "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "max_rounds": 4,
                        "finish_conditions": ["clear compromise"],
                        "context_refs": [],
                        "debate_instruction_prompt": "Challenge weak assumptions.",
                        "moderator_instruction_prompt": "Return structured control output only.",
                    },
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    with pytest.raises(RuntimeError, match="provider offline"):
        await service.respond_to_round(
            run_id=run["run_id"],
            round_index=0,
            user_message="My thesis is sound.",
            idempotency_key="round-1",
        )

    rounds = fake_chat_workflows_db.list_rounds(run["run_id"], 0)
    refreshed_run = fake_chat_workflows_db.get_run(run["run_id"])

    assert rounds[0]["status"] == "failed"
    assert refreshed_run["active_round_index"] == 0


@pytest.mark.asyncio
async def test_respond_to_round_allows_retry_after_failed_attempt_with_same_idempotency_key(
    fake_chat_workflows_db,
):
    class FlakyDialogueOrchestrator:
        def __init__(self):
            self.calls = 0

        async def run_round(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider offline")
            return {
                "debate_llm_message": "Your premise still has gaps.",
                "moderator_decision": "finish",
                "moderator_summary": "The round completed after retry.",
                "next_user_prompt": None,
            }

    orchestrator = FlakyDialogueOrchestrator()
    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=orchestrator,
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 23,
            "title": "Socratic",
            "version": 1,
            "steps": [
                {
                    "id": "debate",
                    "step_index": 0,
                    "step_type": "dialogue_round_step",
                    "base_question": "State your thesis.",
                    "question_mode": "stock",
                    "context_refs": [],
                    "dialogue_config": {
                        "goal_prompt": "Stress-test the thesis.",
                        "opening_prompt_mode": "base_question",
                        "user_role_label": "Author",
                        "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "max_rounds": 4,
                        "finish_conditions": ["clear compromise"],
                        "context_refs": [],
                        "debate_instruction_prompt": "Challenge weak assumptions.",
                        "moderator_instruction_prompt": "Return structured control output only.",
                    },
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    with pytest.raises(RuntimeError, match="provider offline"):
        await service.respond_to_round(
            run_id=run["run_id"],
            round_index=0,
            user_message="My thesis is sound.",
            idempotency_key="round-1",
        )

    retried = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )

    rounds = fake_chat_workflows_db.list_rounds(run["run_id"], 0)

    assert retried["status"] == "completed"
    assert orchestrator.calls == 2
    assert len(rounds) == 1
    assert rounds[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_respond_to_round_preserves_current_prompt_when_continue_omits_next_prompt(
    fake_chat_workflows_db,
):
    class PromptingDialogueOrchestrator:
        def __init__(self):
            self.calls = 0

        async def run_round(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "debate_llm_message": "Counterargument one.",
                    "moderator_decision": "continue",
                    "moderator_summary": "Push on the weakest premise.",
                    "next_user_prompt": "Defend your evidence.",
                }
            return {
                "debate_llm_message": "Counterargument two.",
                "moderator_decision": "continue",
                "moderator_summary": "Stay on the current point.",
                "next_user_prompt": None,
            }

    service = ChatWorkflowService(
        db=fake_chat_workflows_db,
        question_renderer=None,
        dialogue_orchestrator=PromptingDialogueOrchestrator(),
    )
    run = service.start_run(
        tenant_id="default",
        user_id="user-1",
        template={
            "id": 24,
            "title": "Socratic",
            "version": 1,
            "steps": [
                {
                    "id": "debate",
                    "step_index": 0,
                    "step_type": "dialogue_round_step",
                    "base_question": "State your thesis.",
                    "question_mode": "stock",
                    "context_refs": [],
                    "dialogue_config": {
                        "goal_prompt": "Stress-test the thesis.",
                        "opening_prompt_mode": "base_question",
                        "user_role_label": "Author",
                        "debate_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "moderator_llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
                        "max_rounds": 4,
                        "finish_conditions": ["clear compromise"],
                        "context_refs": [],
                        "debate_instruction_prompt": "Challenge weak assumptions.",
                        "moderator_instruction_prompt": "Return structured control output only.",
                    },
                }
            ],
        },
        source_mode="saved_template",
        selected_context_refs=[],
    )

    first = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=0,
        user_message="My thesis is sound.",
        idempotency_key="round-1",
    )
    second = await service.respond_to_round(
        run_id=run["run_id"],
        round_index=1,
        user_message="Here is my evidence.",
        idempotency_key="round-2",
    )

    assert first["current_prompt"] == "Defend your evidence."
    assert second["current_prompt"] == "Defend your evidence."
    assert second["current_round_index"] == 2
