from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .flashcards import (
    DeckReviewPromptSide,
    DeckSchedulerSettingsEnvelope,
    DeckSchedulerType,
    _coerce_scheduler_settings_envelope,
)


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    MULTI_SELECT = "multi_select"
    MATCHING = "matching"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"


class QuizSourceType(str, Enum):
    MEDIA = "media"
    NOTE = "note"
    FLASHCARD_DECK = "flashcard_deck"
    FLASHCARD_CARD = "flashcard_card"
    QUIZ_ATTEMPT = "quiz_attempt"
    QUIZ_ATTEMPT_QUESTION = "quiz_attempt_question"


AnswerValue = int | str | list[int] | dict[str, str]


class SourceCitation(BaseModel):
    source_type: Optional[QuizSourceType] = None
    source_id: Optional[str] = Field(None, min_length=1)
    label: Optional[str] = None
    quote: Optional[str] = None
    media_id: Optional[int] = Field(None, ge=1)
    chunk_id: Optional[str] = None
    timestamp_seconds: Optional[float] = Field(None, ge=0)
    source_url: Optional[str] = None


class QuizGenerateSource(BaseModel):
    source_type: QuizSourceType
    source_id: str = Field(..., min_length=1)


class QuizCreate(BaseModel):
    name: str = Field(..., description="Quiz name")
    description: Optional[str] = Field(None, description="Optional quiz description")
    workspace_tag: Optional[str] = Field(None, description="Optional workspace tag (e.g., 'workspace:<slug-or-id>')")
    workspace_id: Optional[str] = Field(None, description="Canonical owning workspace ID; null means general scope")
    media_id: Optional[int] = Field(None, description="Source media ID for AI-generated quizzes")
    source_bundle_json: Optional[list[QuizGenerateSource]] = Field(
        None, description="Optional canonical mixed-source bundle used to generate this quiz"
    )
    time_limit_seconds: Optional[int] = Field(None, ge=1, description="Optional time limit in seconds")
    passing_score: Optional[int] = Field(None, ge=0, le=100, description="Passing score percentage")


class QuizUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    workspace_tag: Optional[str] = None
    media_id: Optional[int] = None
    source_bundle_json: Optional[list[QuizGenerateSource]] = None
    time_limit_seconds: Optional[int] = Field(None, ge=1)
    passing_score: Optional[int] = Field(None, ge=0, le=100)
    expected_version: Optional[int] = None


class QuizResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    workspace_tag: Optional[str] = None
    workspace_id: Optional[str] = None
    media_id: Optional[int] = None
    source_bundle_json: Optional[list[QuizGenerateSource]] = None
    total_questions: int
    time_limit_seconds: Optional[int] = None
    passing_score: Optional[int] = None
    deleted: bool
    client_id: str
    version: int
    created_at: Optional[str] = None
    last_modified: Optional[str] = None


class QuizListResponse(BaseModel):
    items: list[QuizResponse]
    count: int


class QuestionCreate(BaseModel):
    question_type: QuestionType
    question_text: str
    options: Optional[list[str]] = Field(None, description="Multiple choice options")
    correct_answer: AnswerValue
    explanation: Optional[str] = None
    hint: Optional[str] = None
    hint_penalty_points: int = Field(0, ge=0)
    source_citations: Optional[list[SourceCitation]] = None
    points: int = Field(1, ge=0)
    order_index: int = 0
    tags: Optional[list[str]] = None


class QuestionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    options: Optional[list[str]] = None
    correct_answer: Optional[AnswerValue] = None
    explanation: Optional[str] = None
    hint: Optional[str] = None
    hint_penalty_points: Optional[int] = Field(None, ge=0)
    source_citations: Optional[list[SourceCitation]] = None
    points: Optional[int] = Field(None, ge=0)
    order_index: Optional[int] = None
    tags: Optional[list[str]] = None
    expected_version: Optional[int] = None


class QuestionPublicResponse(BaseModel):
    id: int
    quiz_id: int
    question_type: QuestionType
    question_text: str
    options: Optional[list[str]] = None
    hint: Optional[str] = None
    hint_penalty_points: int = Field(0, ge=0)
    source_citations: Optional[list[SourceCitation]] = None
    points: int
    order_index: int
    tags: Optional[list[str]] = None
    deleted: bool
    client_id: str
    version: int
    created_at: Optional[str] = None
    last_modified: Optional[str] = None


class QuestionAdminResponse(QuestionPublicResponse):
    correct_answer: Optional[AnswerValue] = None
    explanation: Optional[str] = None


class QuestionListResponse(BaseModel):
    items: list[QuestionPublicResponse | QuestionAdminResponse]
    count: int


class QuizAnswerInput(BaseModel):
    question_id: int
    user_answer: AnswerValue
    hint_used: Optional[bool] = None
    time_spent_ms: Optional[int] = None


class AttemptSubmitRequest(BaseModel):
    answers: list[QuizAnswerInput]


class AttemptAnswer(BaseModel):
    question_id: int
    user_answer: AnswerValue
    is_correct: bool
    correct_answer: Optional[AnswerValue] = None
    explanation: Optional[str] = None
    hint_used: Optional[bool] = None
    hint_penalty_points: Optional[int] = Field(None, ge=0)
    source_citations: Optional[list[SourceCitation]] = None
    points_awarded: Optional[int] = None
    time_spent_ms: Optional[int] = None


class AttemptResponse(BaseModel):
    id: int
    quiz_id: int
    started_at: str
    completed_at: Optional[str] = None
    score: Optional[int] = None
    total_possible: int
    time_spent_seconds: Optional[int] = None
    answers: list[AttemptAnswer] = Field(default_factory=list)
    questions: Optional[list[QuestionPublicResponse]] = None


class AttemptListResponse(BaseModel):
    items: list[AttemptResponse]
    count: int


class QuizRemediationConversionSummary(BaseModel):
    id: int
    attempt_id: int
    quiz_id: int
    question_id: int
    status: Literal["active", "superseded"]
    orphaned: bool = False
    superseded_count: int = Field(0, ge=0)
    superseded_by_id: Optional[int] = None
    target_deck_id: Optional[int] = None
    target_deck_name_snapshot: Optional[str] = None
    flashcard_count: int = Field(0, ge=0)
    flashcard_uuids_json: list[str] = Field(default_factory=list)
    source_ref_id: Optional[str] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    client_id: str
    version: int


class QuizRemediationConversionListResponse(BaseModel):
    attempt_id: int
    items: list[QuizRemediationConversionSummary] = Field(default_factory=list)
    count: int = Field(..., ge=0)
    superseded_count: int = Field(..., ge=0)


class QuizRemediationTargetDeck(BaseModel):
    id: int
    name: str


class QuizRemediationConvertRequest(BaseModel):
    question_ids: list[int] = Field(..., min_length=1)
    target_deck_id: Optional[int] = Field(None, ge=1)
    create_deck_name: Optional[str] = None
    create_deck_review_prompt_side: DeckReviewPromptSide = "front"
    create_deck_scheduler_type: Optional[DeckSchedulerType] = None
    create_deck_scheduler_settings: Optional[DeckSchedulerSettingsEnvelope] = None
    replace_active: bool = False

    @model_validator(mode="after")
    def _reject_explicit_null_create_deck_review_prompt_side(self) -> "QuizRemediationConvertRequest":
        if (
            "create_deck_review_prompt_side" in self.model_fields_set
            and self.create_deck_review_prompt_side is None
        ):
            raise ValueError("create_deck_review_prompt_side cannot be null")
        return self

    @model_validator(mode="before")
    def normalize_scheduler_settings(cls, data):
        if not isinstance(data, dict):
            return data
        if "create_deck_scheduler_settings" in data:
            data["create_deck_scheduler_settings"] = _coerce_scheduler_settings_envelope(
                data.get("create_deck_scheduler_settings")
            )
        return data

    @model_validator(mode="after")
    def validate_deck_target(self) -> "QuizRemediationConvertRequest":
        has_target_deck = self.target_deck_id is not None
        has_create_deck = bool((self.create_deck_name or "").strip())
        if has_target_deck == has_create_deck:
            raise ValueError("Provide exactly one of target_deck_id or create_deck_name")
        if "create_deck_review_prompt_side" in self.model_fields_set and not has_create_deck:
            raise ValueError("create_deck review orientation requires create_deck_name")
        if (self.create_deck_scheduler_settings is not None or self.create_deck_scheduler_type is not None) and not has_create_deck:
            raise ValueError("create_deck scheduler options require create_deck_name")
        return self


class QuizRemediationConvertResult(BaseModel):
    question_id: int
    status: Literal["created", "already_exists", "superseded_and_created", "failed"]
    conversion: Optional[QuizRemediationConversionSummary] = None
    flashcard_uuids: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class QuizRemediationConvertResponse(BaseModel):
    attempt_id: int
    quiz_id: int
    target_deck: Optional[QuizRemediationTargetDeck] = None
    results: list[QuizRemediationConvertResult] = Field(default_factory=list)
    created_flashcard_uuids: list[str] = Field(default_factory=list)


class QuizGenerateRequest(BaseModel):
    media_id: Optional[int] = Field(None, ge=1)
    sources: Optional[list[QuizGenerateSource]] = Field(None, min_length=1)
    num_questions: int = Field(10, ge=1, le=100)
    question_types: Optional[list[QuestionType]] = None
    difficulty: str = Field("mixed", description="easy, medium, hard, mixed")
    focus_topics: Optional[list[str]] = None
    model: Optional[str] = None
    api_provider: Optional[str] = None
    workspace_id: Optional[str] = Field(None, description="Canonical owning workspace ID; null means general scope")
    workspace_tag: Optional[str] = Field(None, description="Optional workspace tag (e.g., 'workspace:<slug-or-id>')")

    @model_validator(mode="after")
    def validate_media_id_or_sources(self) -> "QuizGenerateRequest":
        if self.media_id is None and not self.sources:
            raise ValueError("Either media_id or sources must be provided")
        return self


class QuizGenerateResponse(BaseModel):
    quiz: QuizResponse
    questions: list[QuestionAdminResponse]


class QuizImportQuestion(BaseModel):
    question_type: QuestionType
    question_text: str
    options: Optional[list[str]] = None
    correct_answer: AnswerValue
    explanation: Optional[str] = None
    hint: Optional[str] = None
    hint_penalty_points: int = Field(0, ge=0)
    source_citations: Optional[list[SourceCitation]] = None
    points: int = Field(1, ge=0)
    order_index: int = Field(0, ge=0)
    tags: Optional[list[str]] = None


class QuizImportQuiz(BaseModel):
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    workspace_tag: Optional[str] = None
    media_id: Optional[int] = None
    source_bundle_json: Optional[list[QuizGenerateSource]] = None
    time_limit_seconds: Optional[int] = Field(None, ge=1)
    passing_score: Optional[int] = Field(None, ge=0, le=100)


class QuizImportEntry(BaseModel):
    quiz: QuizImportQuiz
    questions: list[QuizImportQuestion] = Field(default_factory=list)


class QuizImportRequest(BaseModel):
    export_format: Optional[str] = Field(
        default=None,
        description="Expected export format marker, e.g. tldw.quiz.export.v1",
    )
    quizzes: list[QuizImportEntry]


class QuizImportItemResult(BaseModel):
    source_index: int = Field(..., ge=0, description="Index of the input quiz entry")
    quiz_id: int
    imported_questions: int = Field(..., ge=0)
    failed_questions: int = Field(..., ge=0)


class QuizImportError(BaseModel):
    source_index: int = Field(..., ge=0)
    quiz_name: Optional[str] = None
    question_index: Optional[int] = Field(default=None, ge=0)
    error: str


class QuizImportResponse(BaseModel):
    imported_quizzes: int = Field(..., ge=0)
    failed_quizzes: int = Field(..., ge=0)
    imported_questions: int = Field(..., ge=0)
    failed_questions: int = Field(..., ge=0)
    items: list[QuizImportItemResult] = Field(default_factory=list)
    errors: list[QuizImportError] = Field(default_factory=list)
