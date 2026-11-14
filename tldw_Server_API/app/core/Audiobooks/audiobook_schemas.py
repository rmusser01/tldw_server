"""Core-owned audiobook alignment and subtitle models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SubtitleFormat = Literal["srt", "vtt", "ass"]
SubtitleMode = Literal["line", "sentence", "word_count", "highlight"]
SubtitleVariant = Literal["wide", "narrow", "centered"]
AlignmentEngine = Literal["kokoro"]


class AlignmentWord(BaseModel):
    """Alignment data for a single word."""

    word: str = Field(..., description="Word text")
    start_ms: int = Field(..., ge=0, description="Start time in milliseconds")
    end_ms: int = Field(..., ge=0, description="End time in milliseconds")
    char_start: int | None = Field(None, ge=0, description="Character offset start")
    char_end: int | None = Field(None, ge=0, description="Character offset end")


class AlignmentPayload(BaseModel):
    """Alignment payload used for subtitle rendering."""

    engine: AlignmentEngine = Field(..., description="Alignment engine")
    sample_rate: int = Field(..., ge=8000, description="Sample rate used by the aligner")
    words: list[AlignmentWord] = Field(..., description="Word alignment data")

    @field_validator("words")
    @classmethod
    def _validate_words(cls, value: list[AlignmentWord]) -> list[AlignmentWord]:
        if not value:
            raise ValueError("alignment words must not be empty")
        return value


class ChapterPreview(BaseModel):
    """Chapter preview from parse output."""

    chapter_id: str = Field(..., description="Chapter identifier")
    title: str | None = Field(None, description="Chapter title")
    start_offset: int = Field(..., ge=0, description="Start offset in characters")
    end_offset: int = Field(..., ge=0, description="End offset in characters")
    word_count: int = Field(..., ge=0, description="Word count in chapter")
