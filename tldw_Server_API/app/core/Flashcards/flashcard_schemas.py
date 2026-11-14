"""Core-owned flashcard generation models, re-exported by the API schemas package (TASK-13950.15)."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

FlashcardCardType = Literal["basic", "basic_reverse", "cloze"]
FlashcardGenerationType = Literal["basic", "basic_reverse", "cloze", "true_false"]


class FlashcardPlanItem(BaseModel):
    """One requested flashcard generation type and its target count."""

    model_config = ConfigDict(extra="forbid")

    card_type: FlashcardGenerationType
    count: int = Field(..., ge=1, le=100, strict=True)


class FlashcardGenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source text to generate flashcards from")
    num_cards: Optional[int] = Field(None, ge=1, le=100, description="Requested number of generated cards")
    card_type: Optional[FlashcardCardType] = Field(None)
    card_plan: Optional[list[FlashcardPlanItem]] = None
    difficulty: Literal['easy', 'medium', 'hard', 'mixed'] = Field('mixed')
    focus_topics: list[str] = Field(default_factory=list)
    provider: Optional[str] = Field(None, description="Optional LLM provider override")
    model: Optional[str] = Field(None, description="Optional LLM model override")
    claims_verification_provider: Optional[str] = Field(None, description="Optional Claims verification LLM provider override")
    claims_verification_model: Optional[str] = Field(None, description="Optional Claims verification LLM model override")

    @model_validator(mode="before")
    @classmethod
    def _validate_raw_generation_shape(cls, value: Any) -> Any:
        """Validate raw fields that must remain strict only for planned requests."""
        if not isinstance(value, dict):
            return value
        has_plan = value.get("card_plan") is not None
        if has_plan and value.get("card_type") is not None:
            raise ValueError("card_plan and card_type are mutually exclusive")
        if has_plan and "num_cards" not in value:
            raise ValueError("num_cards is required when card_plan is present")
        if has_plan and type(value.get("num_cards")) is not int:
            raise ValueError("num_cards must be an integer when card_plan is present")
        return value

    @model_validator(mode="after")
    def _validate_card_plan(self) -> "FlashcardGenerateRequest":
        """Validate planned generation totals and fill legacy defaults."""
        if self.card_plan is not None:
            if len(self.card_plan) == 0:
                raise ValueError("card_plan cannot be empty")
            seen: set[str] = set()
            for row in self.card_plan:
                if row.card_type in seen:
                    raise ValueError("card_plan cannot contain duplicate card_type rows")
                seen.add(row.card_type)
            total = sum(row.count for row in self.card_plan)
            if self.num_cards != total:
                raise ValueError("num_cards must equal the sum of card_plan counts")
        else:
            self.num_cards = self.num_cards or 10
            self.card_type = self.card_type or "basic"
        return self
