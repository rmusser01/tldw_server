# app/api/v1/schemas/skills_schemas.py
#
# Pydantic schemas for Skills API endpoints
#
# Imports
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

# 3rd-party Libraries
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

#
# Local Imports
#
#######################################################################################################################
#
# Constants

# Skill name validation: lowercase letters, numbers, and hyphens only
SKILL_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9-]{0,63}$')


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response
SUPPORTING_FILE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$')

# Aggregate limits for supporting files
MAX_SUPPORTING_FILES_COUNT = 20
MAX_SUPPORTING_FILE_BYTES = 500000
MAX_SUPPORTING_FILES_TOTAL_BYTES = 5 * 1024 * 1024  # 5MB


def _normalize_and_validate_skill_name(value: str) -> str:
    normalized = value.strip().lower()
    if not SKILL_NAME_PATTERN.match(normalized):
        raise ValueError(
            "Skill name must start with a letter, contain only lowercase letters, "
            "numbers, and hyphens, and be 1-64 characters long"
        )
    return normalized


def _validate_supporting_files(
    value: dict[str, str | None] | None,
    *,
    allow_deletes: bool,
) -> dict[str, str | None] | None:
    if value is None:
        return None

    non_null_count = 0
    total_bytes = 0
    for filename, content in value.items():
        # Validate filename format
        if not SUPPORTING_FILE_NAME_PATTERN.match(filename):
            raise ValueError(f"Invalid supporting file name: {filename}")
        # Don't allow SKILL.md as supporting file
        if filename.lower() == "skill.md":
            raise ValueError("SKILL.md cannot be a supporting file")

        if content is None:
            if allow_deletes:
                continue
            raise ValueError(f"Supporting file {filename} cannot be null")

        non_null_count += 1
        if non_null_count > MAX_SUPPORTING_FILES_COUNT:
            raise ValueError(
                f"Too many supporting files ({non_null_count}); maximum is {MAX_SUPPORTING_FILES_COUNT}"
            )
        # Limit individual content size
        file_bytes = len(content.encode("utf-8"))
        if file_bytes > MAX_SUPPORTING_FILE_BYTES:
            raise ValueError(f"Supporting file {filename} exceeds 500KB limit")
        total_bytes += file_bytes

    if total_bytes > MAX_SUPPORTING_FILES_TOTAL_BYTES:
        raise ValueError(
            f"Total supporting files size ({total_bytes} bytes) exceeds "
            f"{MAX_SUPPORTING_FILES_TOTAL_BYTES // (1024 * 1024)}MB limit"
        )
    return value


#######################################################################################################################
#
# Schemas:

class SkillFrontmatter(BaseModel):
    """Parsed frontmatter from SKILL.md file."""
    name: str | None = Field(None, description="Skill identifier (lowercase, hyphens, max 64 chars)")
    description: str | None = Field(None, max_length=1000, description="What the skill does")
    argument_hint: str | None = Field(None, max_length=100, description="Hint shown in UI (e.g., '[issue-number]')")
    disable_model_invocation: bool = Field(False, description="If true, only user can invoke (not auto-invoked by LLM)")
    user_invocable: bool = Field(True, description="If false, hidden from user UI (background knowledge only)")
    allowed_tools: list[str] | None = Field(None, description="Tools allowed without permission when skill is active")
    model: str | None = Field(None, description="Override model for this skill")
    context: Literal["inline", "fork"] = Field("inline", description="'inline' or 'fork' (fork runs in isolated subagent)")

    model_config = ConfigDict(extra='ignore')


class SkillBase(BaseModel):
    """Base schema for skill data."""
    name: str = Field(..., min_length=1, max_length=64, description="Skill identifier")
    description: str | None = Field(None, max_length=1000, description="What the skill does")
    argument_hint: str | None = Field(None, max_length=100, description="Hint shown in UI")
    disable_model_invocation: bool = Field(False, description="If true, only user can invoke")
    user_invocable: bool = Field(True, description="If false, hidden from user UI")
    allowed_tools: list[str] | None = Field(None, description="Tools allowed during skill execution")
    model: str | None = Field(None, description="Override model for this skill")
    context: Literal["inline", "fork"] = Field("inline", description="Execution context mode")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _normalize_and_validate_skill_name(value)


class SkillCreate(BaseModel):
    """Schema for creating a new skill."""
    name: str = Field(..., min_length=1, max_length=64, description="Skill identifier")
    content: str = Field(..., min_length=1, max_length=500000, description="Full SKILL.md content with optional frontmatter")
    supporting_files: dict[str, str] | None = Field(
        None,
        description="Additional files (e.g., {'reference.md': '...content...'})"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _normalize_and_validate_skill_name(value)

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        validated = _validate_supporting_files(
            value,
            allow_deletes=False,
        )
        return validated  # type: ignore[return-value]


class SkillUpdate(BaseModel):
    """Schema for updating an existing skill."""
    content: str | None = Field(None, min_length=1, max_length=500000, description="Full SKILL.md content")
    supporting_files: dict[str, str | None] | None = Field(
        None,
        description="Additional files to update/add. Set value to None to remove a file."
    )

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(
        cls,
        value: dict[str, str | None] | None,
    ) -> dict[str, str | None] | None:
        return _validate_supporting_files(
            value,
            allow_deletes=True,
        )


class SkillResponse(SkillBase):
    """Schema for skill responses."""
    id: str = Field(..., description="UUID of the skill")
    content: str = Field(..., description="SKILL.md content (markdown body without frontmatter)")
    supporting_files: dict[str, str] | None = Field(None, description="Additional files in the skill directory")
    directory_path: str = Field(..., description="Path to skill directory")
    created_at: datetime = Field(..., description="Timestamp of creation")
    last_modified: datetime = Field(..., description="Timestamp of last modification")
    version: int = Field(..., description="Version for optimistic locking")

    model_config = ConfigDict(from_attributes=True)


class SkillSummary(BaseModel):
    """Minimal skill info for listing."""
    name: str = Field(..., description="Skill identifier")
    description: str | None = Field(None, description="What the skill does")
    argument_hint: str | None = Field(None, description="Hint shown in UI")
    user_invocable: bool = Field(..., description="If false, hidden from user UI")
    disable_model_invocation: bool = Field(..., description="If true, only user can invoke")
    context: Literal["inline", "fork"] = Field(..., description="Execution context mode")

    model_config = ConfigDict(from_attributes=True)


class SkillsListResponse(BaseModel):
    """Response for listing skills."""
    skills: list[SkillSummary] = Field(..., description="List of skill summaries")
    count: int = Field(..., description="Number of skills in response")
    total: int = Field(..., description="Total number of skills")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical pagination metadata")

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class SkillExecuteRequest(BaseModel):
    """Request to execute/preview a skill."""
    args: str | None = Field(None, max_length=10000, description="Arguments to pass to the skill")


class SkillExecutionResult(BaseModel):
    """Result of skill execution."""
    skill_name: str = Field(..., description="Name of the executed skill")
    rendered_prompt: str = Field(..., description="Prompt with arguments substituted")
    allowed_tools: list[str] | None = Field(None, description="Tools allowed for this skill")
    model_override: str | None = Field(None, description="Model override if specified")
    execution_mode: Literal["inline", "fork"] = Field(..., description="How the skill was executed")
    fork_output: str | None = Field(None, description="Output from fork execution (if applicable)")

    model_config = ConfigDict(from_attributes=True)


class SkillImportRequest(BaseModel):
    """Request to import a skill from text content."""
    name: str | None = Field(
        None,
        min_length=1,
        max_length=64,
        description="Optional skill name override (uses frontmatter name if omitted)",
    )
    content: str = Field(..., min_length=1, max_length=500000, description="SKILL.md content")
    supporting_files: dict[str, str] | None = Field(None, description="Additional files")
    overwrite: bool = Field(False, description="If true, overwrite existing skill with same name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_and_validate_skill_name(value)

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        validated = _validate_supporting_files(
            value,
            allow_deletes=False,
        )
        return validated  # type: ignore[return-value]


class SkillContextPayload(BaseModel):
    """Skill context for injection into chat."""
    available_skills: list[SkillSummary] = Field(..., description="Skills available for invocation")
    context_text: str = Field(..., description="Formatted text for LLM context injection")

    model_config = ConfigDict(from_attributes=True)


#
# End of skills_schemas.py
#######################################################################################################################
