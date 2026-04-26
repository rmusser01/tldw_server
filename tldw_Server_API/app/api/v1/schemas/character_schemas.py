# character_schemas.py
# Description:
#
# Imports
import json
from datetime import datetime
from typing import Any, Literal, Optional, Union

#
# Third-party imports
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

#
######################################################################################################################
#
# --- Pydantic Schemas ---

# Maximum lengths for character fields to prevent unbounded database growth
MAX_NAME_LENGTH = 500
MAX_DESCRIPTION_LENGTH = 50_000
MAX_PERSONALITY_LENGTH = 50_000
MAX_SCENARIO_LENGTH = 50_000
MAX_SYSTEM_PROMPT_LENGTH = 100_000
MAX_FIRST_MESSAGE_LENGTH = 50_000
MAX_MESSAGE_EXAMPLE_LENGTH = 100_000
MAX_CREATOR_NOTES_LENGTH = 50_000
MAX_CREATOR_LENGTH = 500
MAX_VERSION_LENGTH = 100
MAX_IMAGE_BASE64_LENGTH = 20_000_000  # ~15MB image when decoded


class CharacterBase(BaseModel):
    name: Optional[str] = Field(None, max_length=MAX_NAME_LENGTH)
    description: Optional[str] = Field(None, examples=["A brave knight"], max_length=MAX_DESCRIPTION_LENGTH)
    personality: Optional[str] = Field(None, examples=["Stoic and honorable"], max_length=MAX_PERSONALITY_LENGTH)
    scenario: Optional[str] = Field(None, examples=["Guarding the ancient ruins"], max_length=MAX_SCENARIO_LENGTH)
    system_prompt: Optional[str] = Field(None, examples=["You are a helpful character."], max_length=MAX_SYSTEM_PROMPT_LENGTH)
    post_history_instructions: Optional[str] = Field(None, max_length=MAX_SYSTEM_PROMPT_LENGTH)
    first_message: Optional[str] = Field(None, examples=["Greetings, traveler!"], max_length=MAX_FIRST_MESSAGE_LENGTH)
    message_example: Optional[str] = Field(None, examples=["<START>\nUSER: Hello\nASSISTANT: Hi there!\n<END>"], max_length=MAX_MESSAGE_EXAMPLE_LENGTH)
    creator_notes: Optional[str] = Field(None, max_length=MAX_CREATOR_NOTES_LENGTH)
    alternate_greetings: Optional[Union[list[str], str]] = Field(None,
                                                                 description="List of strings or a JSON string representation of a list.",
                                                                 examples=[["Hello!", "Good day!"]])
    tags: Optional[Union[list[str], str]] = Field(None,
                                                  description="List of strings or a JSON string representation of a list.",
                                                  examples=[["fantasy", "knight"]])
    creator: Optional[str] = Field(None, max_length=MAX_CREATOR_LENGTH)
    character_version: Optional[str] = Field(None, max_length=MAX_VERSION_LENGTH)
    extensions: Optional[Union[dict[str, Any], str]] = Field(None,
                                                             description="Dictionary or a JSON string representation of a dictionary.")
    image_base64: Optional[str] = Field(None,
                                        description="Base64 encoded image string (without 'data:image/...;base64,' prefix).",
                                        max_length=MAX_IMAGE_BASE64_LENGTH)

    @field_validator("alternate_greetings", "tags", "extensions", mode="before")
    @classmethod
    def parse_json_string(cls, value: Any, info: ValidationInfo) -> Any:
        """Parse JSON strings and validate the resulting structure.

        Unlike the previous implementation that silently returned empty values on
        parse failure, this raises a ValueError to prevent invalid data from being
        accepted without the caller's knowledge.

        Empty String Conversion Behavior:
            - Empty or whitespace-only strings are converted to appropriate empty types:
              - alternate_greetings, tags: "" -> []
              - extensions: "" -> {}
            - This is intentional to allow form submissions with empty fields.
            - If you want to explicitly clear a field, pass the appropriate empty
              type ([] or {}) or null/None.

        Raises:
            ValueError: If the string contains invalid JSON or the parsed type
                doesn't match the expected type for the field.
        """
        if value is None:
            return value

        if isinstance(value, str):
            # Empty or whitespace-only string -> convert to appropriate empty type
            # This allows form submissions with empty fields to work correctly.
            # NOTE: This is intentional behavior. Pass [] or {} explicitly if needed.
            if not value.strip():
                if info.field_name in ["alternate_greetings", "tags"]:
                    return []
                if info.field_name == "extensions":
                    return {}
                return value

            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                # Raise error instead of silently returning empty - caller should know about invalid JSON
                raise ValueError(
                    f"Invalid JSON in field '{info.field_name}': {str(e)[:100]}"
                ) from e

            # Validate the parsed structure matches expected type
            if info.field_name in ["alternate_greetings", "tags"]:
                if not isinstance(parsed, list):
                    raise ValueError(
                        f"Field '{info.field_name}' must be a list, got {type(parsed).__name__}"
                    )
                # Validate list contents are strings
                for i, item in enumerate(parsed):
                    if not isinstance(item, str):
                        raise ValueError(
                            f"Field '{info.field_name}' must contain only strings, "
                            f"item at index {i} is {type(item).__name__}"
                        )
            elif info.field_name == "extensions":
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Field 'extensions' must be a dictionary, got {type(parsed).__name__}"
                    )

            return parsed

        # If already the correct type, validate it
        if info.field_name in ["alternate_greetings", "tags"]:
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if not isinstance(item, str):
                        raise ValueError(
                            f"Field '{info.field_name}' must contain only strings, "
                            f"item at index {i} is {type(item).__name__}"
                        )
        elif info.field_name == "extensions" and not isinstance(value, dict):
            raise ValueError(
                f"Field 'extensions' must be a dictionary, got {type(value).__name__}"
            )

        return value


class CharacterCreate(CharacterBase):
    name: str = Field(..., examples=["Sir Gideon"], min_length=1, max_length=MAX_NAME_LENGTH)


class CharacterUpdate(CharacterBase):
    pass  # All fields optional


class CharacterResponse(CharacterBase):
    id: int
    version: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    image_present: bool = False
    model_config = {"from_attributes": True}


class CharacterVersionEntry(BaseModel):
    change_id: int
    version: int = Field(ge=1)
    operation: str
    timestamp: Optional[datetime] = None
    client_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CharacterVersionListResponse(BaseModel):
    items: list[CharacterVersionEntry] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class CharacterVersionDiffField(BaseModel):
    field: str
    old_value: Any = None
    new_value: Any = None


class CharacterVersionDiffResponse(BaseModel):
    character_id: int = Field(gt=0)
    from_entry: CharacterVersionEntry
    to_entry: CharacterVersionEntry
    changed_fields: list[CharacterVersionDiffField] = Field(default_factory=list)
    changed_count: int = Field(default=0, ge=0)


class CharacterRevertRequest(BaseModel):
    target_version: int = Field(..., ge=1)


class CharacterListQueryResponse(BaseModel):
    items: list[CharacterResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1)
    has_more: bool = False


class CharacterTagOperationRequest(BaseModel):
    operation: Literal["rename", "merge", "delete"]
    source_tag: str = Field(..., min_length=1, max_length=200)
    target_tag: Optional[str] = Field(default=None, max_length=200)

    @field_validator("source_tag", "target_tag", mode="before")
    @classmethod
    def normalize_tag_strings(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_target_tag_requirements(self) -> "CharacterTagOperationRequest":
        if self.operation in {"rename", "merge"} and not self.target_tag:
            raise ValueError("target_tag is required for rename and merge operations")
        if self.operation == "delete":
            self.target_tag = None
        return self


class CharacterTagOperationResponse(BaseModel):
    operation: Literal["rename", "merge", "delete"]
    source_tag: str
    target_tag: Optional[str] = None
    matched_count: int = Field(default=0, ge=0)
    updated_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    updated_character_ids: list[int] = Field(default_factory=list)
    failed_character_ids: list[int] = Field(default_factory=list)


class CharacterImportResponse(BaseModel):
    id: int = Field(..., description="ID of the imported character")
    name: str = Field(..., description="Name of the imported character")
    message: str = Field(..., description="Import status message")
    character: Optional[CharacterResponse] = Field(
        default=None,
        description="Full character details when available"
    )


class MessageResponse(BaseModel):
    message: str


class DeletionResponse(MessageResponse):
    character_id: int = Field(gt=0)


class WorldBookDeletionResponse(MessageResponse):
    world_book_id: int = Field(gt=0)


class WorldBookEntryDeletionResponse(MessageResponse):
    entry_id: int = Field(gt=0)


class CharacterWorldBookDetachDeletionResponse(MessageResponse):
    world_book_id: int = Field(gt=0)
    detached_from_character_id: int = Field(gt=0)


class CharacterExemplarSource(BaseModel):
    type: Literal['audio_transcript', 'video_transcript', 'article', 'other'] = 'other'
    url_or_id: Optional[str] = None
    date: Optional[str] = None


class CharacterExemplarLabels(BaseModel):
    emotion: Literal['angry', 'neutral', 'happy', 'other'] = 'other'
    scenario: Literal['press_challenge', 'fan_banter', 'debate', 'boardroom', 'small_talk', 'other'] = 'other'
    rhetorical: list[str] = Field(default_factory=list)
    register: Optional[str] = None


class CharacterExemplarSafety(BaseModel):
    allowed: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)


class CharacterExemplarRights(BaseModel):
    public_figure: bool = True
    notes: Optional[str] = None


class CharacterExemplarIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    source: CharacterExemplarSource = Field(default_factory=CharacterExemplarSource)
    novelty_hint: Literal['post_cutoff', 'unknown', 'pre_cutoff'] = 'unknown'
    labels: CharacterExemplarLabels = Field(default_factory=CharacterExemplarLabels)
    safety: CharacterExemplarSafety = Field(default_factory=CharacterExemplarSafety)
    rights: CharacterExemplarRights = Field(default_factory=CharacterExemplarRights)
    length_tokens: Optional[int] = Field(default=None, ge=1, le=10_000)


class CharacterExemplarUpdate(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=100_000)
    source: Optional[CharacterExemplarSource] = None
    novelty_hint: Optional[Literal['post_cutoff', 'unknown', 'pre_cutoff']] = None
    labels: Optional[CharacterExemplarLabels] = None
    safety: Optional[CharacterExemplarSafety] = None
    rights: Optional[CharacterExemplarRights] = None
    length_tokens: Optional[int] = Field(default=None, ge=1, le=10_000)


class CharacterExemplarResponse(CharacterExemplarIn):
    id: str
    character_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None


class CharacterExemplarSearchFilter(BaseModel):
    emotion: Optional[Literal['angry', 'neutral', 'happy', 'other']] = None
    scenario: Optional[Literal['press_challenge', 'fan_banter', 'debate', 'boardroom', 'small_talk', 'other']] = None
    rhetorical: list[str] = Field(default_factory=list)


class CharacterExemplarSearchRequest(BaseModel):
    query: Optional[str] = None
    filter: CharacterExemplarSearchFilter = Field(default_factory=CharacterExemplarSearchFilter)
    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    use_embedding_scores: bool = Field(
        default=False,
        description="Enable embedding-backed hybrid ranking for search results.",
    )
    embedding_model_id: Optional[str] = Field(
        default=None,
        description="Optional embedding model override when hybrid ranking is enabled.",
    )


class CharacterExemplarSearchResponse(BaseModel):
    items: list[CharacterExemplarResponse]
    total: int = Field(default=0, ge=0)


class CharacterExemplarSelectionConfig(BaseModel):
    budget_tokens: int = Field(default=600, ge=1, le=20_000)
    max_exemplar_tokens: int = Field(default=120, ge=1, le=20_000)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    use_embedding_scores: bool = Field(
        default=False,
        description="Enable embedding-backed semantic scoring in selector debug flow.",
    )
    embedding_model_id: Optional[str] = Field(
        default=None,
        description="Optional embedding model override when semantic scoring is enabled.",
    )


class CharacterExemplarSelectionDebugRequest(BaseModel):
    user_turn: str = Field(..., min_length=1, max_length=100_000)
    selection_config: CharacterExemplarSelectionConfig = Field(default_factory=CharacterExemplarSelectionConfig)


class CharacterExemplarCoverage(BaseModel):
    openers: int = Field(default=0, ge=0)
    emphasis: int = Field(default=0, ge=0)
    enders: int = Field(default=0, ge=0)
    catchphrases_used: int = Field(default=0, ge=0)


class CharacterExemplarScore(BaseModel):
    id: str
    score: float


class CharacterExemplarSelectionDebug(BaseModel):
    selected: list[CharacterExemplarResponse]
    budget_tokens: int = Field(..., ge=0)
    coverage: CharacterExemplarCoverage
    scores: list[CharacterExemplarScore]


class CharacterExemplarDeletionResponse(BaseModel):
    message: str
    character_id: int
    exemplar_id: str

#
# End of character_schemas.py
######################################################################################################################
