"""Bounded plain-text document drafts and private PDF-worker input."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_WORK_BYTES = 30 * 1024 * 1024
MAX_ANNOTATION_BYTES = 512 * 1024
PDF_ERRORS = {
    "invalid_document": "The document input is invalid or unsupported.",
    "page_overflow": "A generated page exceeds printable space. Shorten that page before saving.",
    "locked_pdf": "Unlock the original PDF before exporting its annotations.",
    "too_large": "The PDF exceeds the 20 MiB document limit.",
    "invalid_annotation": "A saved annotation has an invalid page number or exceeds the export limits.",
}


class DocumentDraftPage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heading: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=6000)

    @field_validator("heading", "text")
    @classmethod
    def plain_text(cls, value: str) -> str:
        """Preserve exact text while refusing blank or invalid control input."""
        if not value.strip() or any(ord(char) < 32 and char not in "\t\r\n" for char in value):
            raise ValueError("Enter nonblank plain text without control characters")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Text must contain valid Unicode") from exc
        return value


class DocumentRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    pages: list[DocumentDraftPage] = Field(min_length=1, max_length=10)
    _plain_title = field_validator("title")(DocumentDraftPage.plain_text.__func__)


class PdfAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(strict=True, ge=1)
    text: str = Field(max_length=6000)
    note: str = Field(default="", max_length=6000)
    color: Literal["yellow", "green", "blue", "pink"]
    kind: Literal["highlight", "page_note"]
