import pytest
from pydantic import ValidationError
from pydantic.json_schema import models_json_schema

from tldw_Server_API.app.api.v1.schemas.chat_workflows import (
    ChatWorkflowRunResponse,
    ChatWorkflowTemplateCreate,
    ChatWorkflowTemplateResponse,
    ChatWorkflowTemplateStep,
    ChatWorkflowTranscriptMessage,
    GenerateDraftRequest,
    GenerateDraftResponse,
    StartRunRequest,
    SubmitAnswerRequest,
)


def test_generate_draft_request_requires_goal():
    req = GenerateDraftRequest(
        goal="Plan my migration",
        desired_step_count=4,
        context_refs=[],
    )

    assert req.goal == "Plan my migration"
    assert req.desired_step_count == 4


def test_template_openapi_refs_are_stable_across_input_and_output_modes():
    """Avoid Pydantic-version-specific ``-Input``/``-Output`` component names."""
    _, schema = models_json_schema(
        [
            (ChatWorkflowTemplateCreate, "validation"),
            (ChatWorkflowTemplateResponse, "serialization"),
            (GenerateDraftResponse, "serialization"),
            (StartRunRequest, "validation"),
        ]
    )

    template_components = sorted(
        name for name in schema["$defs"] if name.startswith("ChatWorkflowTemplate")
    )

    assert template_components == [
        "ChatWorkflowTemplateCreate",
        "ChatWorkflowTemplateDraft",
        "ChatWorkflowTemplateResponse",
        "ChatWorkflowTemplateStep",
    ]


def test_answer_request_rejects_empty_answer():
    with pytest.raises(ValidationError):
        SubmitAnswerRequest(step_index=0, answer_text="  ")


def test_template_step_normalizes_llm_phrased_question_mode():
    step = ChatWorkflowTemplateStep(
        id="goal",
        step_index=0,
        base_question="What is your goal?",
        question_mode="llm-phrased",
        context_refs=[],
    )

    assert step.question_mode == "llm_phrased"


def test_start_run_request_requires_template_reference():
    with pytest.raises(ValidationError):
        StartRunRequest(selected_context_refs=[])


def test_submit_answer_request_strips_idempotency_key():
    req = SubmitAnswerRequest(
        step_index=0,
        answer_text="Ship a feature",
        idempotency_key="  replay-1  ",
    )

    assert req.idempotency_key == "replay-1"


def test_template_step_accepts_dialogue_round_step():
    step = ChatWorkflowTemplateStep.model_validate(
        {
            "id": "debate-step",
            "step_index": 0,
            "step_type": "dialogue_round_step",
            "base_question": "State your thesis.",
            "dialogue_config": {
                "goal_prompt": "Stress-test the user's thesis.",
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
    )

    assert step.step_type == "dialogue_round_step"
    assert step.dialogue_config is not None
    assert step.dialogue_config.user_role_label == "Author"


def test_template_step_rejects_dialogue_round_without_config():
    with pytest.raises(ValidationError):
        ChatWorkflowTemplateStep.model_validate(
            {
                "id": "debate-step",
                "step_index": 0,
                "step_type": "dialogue_round_step",
                "base_question": "State your thesis.",
            }
        )


def test_template_step_requires_opening_prompt_text_for_custom_prompt():
    with pytest.raises(ValidationError):
        ChatWorkflowTemplateStep.model_validate(
            {
                "id": "debate-step",
                "step_index": 0,
                "step_type": "dialogue_round_step",
                "base_question": "State your thesis.",
                "dialogue_config": {
                    "goal_prompt": "Stress-test the user's thesis.",
                    "opening_prompt_mode": "custom_prompt",
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
        )


def test_run_response_accepts_round_history_and_current_prompt():
    run = ChatWorkflowRunResponse.model_validate(
        {
            "run_id": "run-1",
            "template_version": 1,
            "status": "active",
            "current_step_index": 0,
            "current_step_kind": "dialogue_round_step",
            "current_prompt": "Defend your evidence.",
            "current_round_index": 1,
            "selected_context_refs": [],
            "rounds": [
                {
                    "round_index": 0,
                    "user_message": "Here is my thesis.",
                    "debate_llm_message": "Counterargument",
                    "moderator_decision": "continue",
                    "moderator_summary": "Push on the weakest claim.",
                    "next_user_prompt": "Defend your evidence.",
                }
            ],
            "answers": [],
        }
    )

    assert run.current_step_kind == "dialogue_round_step"
    assert run.current_prompt == "Defend your evidence."
    assert len(run.rounds) == 1


def test_transcript_message_accepts_dialogue_roles():
    debate_message = ChatWorkflowTranscriptMessage.model_validate(
        {
            "role": "debate_llm",
            "content": "Counterargument",
            "step_index": 0,
        }
    )
    moderator_message = ChatWorkflowTranscriptMessage.model_validate(
        {
            "role": "moderator",
            "content": "Push on the weakest claim.",
            "step_index": 0,
        }
    )

    assert debate_message.role == "debate_llm"
    assert moderator_message.role == "moderator"
