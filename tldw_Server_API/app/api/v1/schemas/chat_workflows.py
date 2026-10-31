"""Pydantic request and response models for Chat Workflows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


def _strip_required(value: str, *, field_name: str) -> str:
    """Normalize a required string field and reject blank values."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def _strip_optional(value: str | None) -> str | None:
    """Normalize an optional string field and coerce blank values to None."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class ChatWorkflowLLMSelection(BaseModel):
    """Workflow-safe LLM selection fields for dialogue participants."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1)
    provider: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0, le=1)

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        """Trim the required model identifier."""
        return _strip_required(value, field_name="model")

    @field_validator("provider", mode="before")
    @classmethod
    def _strip_provider(cls, value: str | None) -> str | None:
        """Trim the optional provider identifier."""
        return _strip_optional(value)


class ChatWorkflowDialogueConfig(BaseModel):
    """Configuration for a moderated dialogue step."""

    model_config = ConfigDict(extra="forbid")

    goal_prompt: str = Field(..., min_length=1)
    opening_prompt_mode: Literal["base_question", "custom_prompt"] = "base_question"
    opening_prompt_text: str | None = None
    user_role_label: str = Field(..., min_length=1)
    debate_llm_config: ChatWorkflowLLMSelection
    moderator_llm_config: ChatWorkflowLLMSelection
    max_rounds: int = Field(..., ge=1, le=12)
    finish_conditions: list[str] = Field(default_factory=list)
    context_refs: list[dict[str, Any]] = Field(default_factory=list)
    debate_instruction_prompt: str = Field(..., min_length=1)
    moderator_instruction_prompt: str = Field(..., min_length=1)

    @field_validator(
        "goal_prompt",
        "user_role_label",
        "debate_instruction_prompt",
        "moderator_instruction_prompt",
    )
    @classmethod
    def _strip_required_text(cls, value: str, info: ValidationInfo) -> str:
        """Trim required dialogue config text fields."""
        return _strip_required(value, field_name=info.field_name)

    @field_validator("opening_prompt_text", mode="before")
    @classmethod
    def _strip_opening_prompt_text(cls, value: str | None) -> str | None:
        """Trim the optional custom opening prompt."""
        return _strip_optional(value)

    @field_validator("opening_prompt_mode", mode="before")
    @classmethod
    def _normalize_opening_prompt_mode(cls, value: str) -> str:
        """Normalize custom/base opening prompt mode values."""
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_")
        return value

    @field_validator("finish_conditions")
    @classmethod
    def _strip_finish_conditions(cls, value: list[str]) -> list[str]:
        """Trim finish-condition strings and drop empties."""
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def _validate_opening_prompt(self) -> ChatWorkflowDialogueConfig:
        """Require a custom opening prompt when the mode asks for one."""
        if self.opening_prompt_mode == "custom_prompt" and not self.opening_prompt_text:
            raise ValueError("custom_prompt mode requires opening_prompt_text")
        return self


class ChatWorkflowTemplateStep(BaseModel):
    """A single authored step in a workflow template or run snapshot."""

    model_config = ConfigDict(extra="forbid", json_schema_mode_override="validation")

    id: str = Field(..., min_length=1)
    step_index: int = Field(..., ge=0)
    step_type: Literal["question_step", "dialogue_round_step"] = "question_step"
    label: str | None = None
    base_question: str = Field(..., min_length=1)
    question_mode: Literal["stock", "llm_phrased"] = "stock"
    phrasing_instructions: str | None = None
    context_refs: list[dict[str, Any]] = Field(default_factory=list)
    dialogue_config: ChatWorkflowDialogueConfig | None = None

    @field_validator("id", "base_question")
    @classmethod
    def _strip_required_fields(cls, value: str, info: ValidationInfo) -> str:
        """Trim required text fields while preserving the field name in errors."""
        return _strip_required(value, field_name=info.field_name)

    @field_validator("label", "phrasing_instructions", mode="before")
    @classmethod
    def _strip_optional_fields(cls, value: str | None) -> str | None:
        """Trim optional text fields before validation."""
        return _strip_optional(value)

    @field_validator("question_mode", "step_type", mode="before")
    @classmethod
    def _normalize_modes(cls, value: str) -> str:
        """Accept hyphenated mode values and normalize them to schema form."""
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_")
        return value

    @model_validator(mode="after")
    def _validate_step_shape(self) -> ChatWorkflowTemplateStep:
        """Ensure the dialogue fields match the selected step type."""
        if self.step_type == "dialogue_round_step" and self.dialogue_config is None:
            raise ValueError("dialogue_round_step requires dialogue_config")
        if self.step_type == "question_step" and self.dialogue_config is not None:
            raise ValueError("question_step must not include dialogue_config")
        return self


class ChatWorkflowTemplateDraft(BaseModel):
    """A reusable or generated linear workflow definition."""

    model_config = ConfigDict(extra="forbid", json_schema_mode_override="validation")

    title: str = Field(..., min_length=1)
    description: str | None = None
    version: int = Field(default=1, ge=1)
    steps: list[ChatWorkflowTemplateStep] = Field(..., min_length=1)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        """Trim the draft title and reject blanks."""
        return _strip_required(value, field_name="title")

    @field_validator("description", mode="before")
    @classmethod
    def _strip_description(cls, value: str | None) -> str | None:
        """Trim the optional draft description."""
        return _strip_optional(value)


class ChatWorkflowTemplateCreate(ChatWorkflowTemplateDraft):
    """Request body for creating a persisted workflow template."""


class ChatWorkflowTemplateUpdate(BaseModel):
    """Patch body for modifying an existing workflow template."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None
    steps: list[ChatWorkflowTemplateStep] | None = None
    status: Literal["active", "archived"] | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _strip_optional_title(cls, value: str | None) -> str | None:
        """Trim the optional title override."""
        return _strip_optional(value)

    @field_validator("description", mode="before")
    @classmethod
    def _strip_optional_description(cls, value: str | None) -> str | None:
        """Trim the optional description override."""
        return _strip_optional(value)

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> ChatWorkflowTemplateUpdate:
        """Reject empty update payloads that would be a no-op."""
        if (
            self.title is None
            and self.description is None
            and self.steps is None
            and self.status is None
        ):
            raise ValueError("at least one template field must be provided")
        return self


class GenerateDraftRequest(BaseModel):
    """Request body for draft generation from a goal statement."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=1)
    base_question: str | None = None
    desired_step_count: int = Field(default=4, ge=1, le=12)
    context_refs: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("goal")
    @classmethod
    def _strip_goal(cls, value: str) -> str:
        """Trim the generation goal and reject blanks."""
        return _strip_required(value, field_name="goal")

    @field_validator("base_question", mode="before")
    @classmethod
    def _strip_base_question(cls, value: str | None) -> str | None:
        """Trim the optional seed question."""
        return _strip_optional(value)


class GenerateDraftResponse(BaseModel):
    """Response body containing the generated workflow draft."""

    model_config = ConfigDict(extra="forbid")

    template_draft: ChatWorkflowTemplateDraft


class StartRunRequest(BaseModel):
    """Request body for starting a workflow run from a template or draft."""

    model_config = ConfigDict(extra="forbid")

    template_id: int | None = Field(default=None, ge=1)
    template_draft: ChatWorkflowTemplateDraft | None = None
    selected_context_refs: list[dict[str, Any]] = Field(default_factory=list)
    question_renderer_model: str | None = None

    @field_validator("question_renderer_model", mode="before")
    @classmethod
    def _strip_question_renderer_model(cls, value: str | None) -> str | None:
        """Trim the optional renderer model override."""
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_template_source(self) -> StartRunRequest:
        """Require exactly one workflow source when starting a run."""
        has_template_id = self.template_id is not None
        has_template_draft = self.template_draft is not None
        if has_template_id == has_template_draft:
            raise ValueError("provide exactly one of template_id or template_draft")
        return self


class SubmitAnswerRequest(BaseModel):
    """Request body for answering the current workflow step."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=0)
    answer_text: str = Field(..., min_length=1)
    idempotency_key: str | None = None

    @field_validator("answer_text")
    @classmethod
    def _strip_answer(cls, value: str) -> str:
        """Trim the answer body and reject blanks."""
        return _strip_required(value, field_name="answer_text")

    @field_validator("idempotency_key", mode="before")
    @classmethod
    def _strip_idempotency_key(cls, value: str | None) -> str | None:
        """Trim the optional idempotency key."""
        return _strip_optional(value)


class SubmitRoundRequest(BaseModel):
    """Request body for responding to the current dialogue round."""

    model_config = ConfigDict(extra="forbid")

    user_message: str = Field(..., min_length=1)
    idempotency_key: str | None = None

    @field_validator("user_message")
    @classmethod
    def _strip_user_message(cls, value: str) -> str:
        """Trim the user dialogue message and reject blanks."""
        return _strip_required(value, field_name="user_message")

    @field_validator("idempotency_key", mode="before")
    @classmethod
    def _strip_round_idempotency_key(cls, value: str | None) -> str | None:
        """Trim the optional round idempotency key."""
        return _strip_optional(value)


class ContinueChatResponse(BaseModel):
    """Response body for continuing a completed run as free chat."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str


class ChatWorkflowTemplateResponse(ChatWorkflowTemplateDraft):
    """Persisted workflow template payload returned by the API."""

    id: int
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None


class ChatWorkflowAnswerResponse(BaseModel):
    """Serialized answer entry for a workflow run."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    step_index: int
    displayed_question: str
    answer_text: str
    question_generation_meta: dict[str, Any] = Field(default_factory=dict)
    answered_at: str | None = None


class ChatWorkflowTranscriptMessage(BaseModel):
    """A transcript message synthesized from a workflow run."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant", "user", "debate_llm", "moderator"]
    content: str
    step_index: int


class ChatWorkflowRoundResponse(BaseModel):
    """Serialized round state for a dialogue workflow step."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(..., ge=0)
    user_message: str
    debate_llm_message: str | None = None
    moderator_decision: Literal["continue", "finish"] | None = None
    moderator_summary: str | None = None
    next_user_prompt: str | None = None
    status: Literal["pending", "completed", "failed"] = "completed"
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator(
        "user_message",
        "debate_llm_message",
        "moderator_summary",
        "next_user_prompt",
        mode="before",
    )
    @classmethod
    def _strip_round_text(cls, value: str | None) -> str | None:
        """Trim optional round transcript text fields."""
        return _strip_optional(value)


class ChatWorkflowTranscriptResponse(BaseModel):
    """Transcript response for replaying a workflow run as chat messages."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    messages: list[ChatWorkflowTranscriptMessage] = Field(default_factory=list)


class ChatWorkflowRunResponse(BaseModel):
    """Current run state, including answered history and current question."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    template_id: int | None = None
    template_version: int
    status: str
    current_step_index: int
    started_at: str | None = None
    completed_at: str | None = None
    canceled_at: str | None = None
    free_chat_conversation_id: str | None = None
    selected_context_refs: list[dict[str, Any]] = Field(default_factory=list)
    current_question: str | None = None
    current_step_kind: Literal["question_step", "dialogue_round_step"] | None = None
    current_prompt: str | None = None
    current_round_index: int | None = Field(default=None, ge=0)
    rounds: list[ChatWorkflowRoundResponse] = Field(default_factory=list)
    answers: list[ChatWorkflowAnswerResponse] = Field(default_factory=list)
