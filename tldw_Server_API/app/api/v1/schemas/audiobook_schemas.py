"""
Pydantic schemas for audiobook creation APIs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

SourceInputType = Literal["epub", "pdf", "txt", "md", "srt", "vtt", "ass"]
AudioFormat = Literal["wav", "mp3", "flac", "opus", "m4b"]
SubtitleFormat = Literal["srt", "vtt", "ass"]
SubtitleMode = Literal["line", "sentence", "word_count", "highlight"]
SubtitleVariant = Literal["wide", "narrow", "centered"]
AudiobookJobStatus = Literal["queued", "processing", "completed", "failed", "canceled"]
AudiobookArtifactType = Literal["audio", "subtitle", "package", "alignment"]
AudiobookArtifactScope = Literal["chapter", "merged"]
AlignmentEngine = Literal["kokoro"]


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class SourceRef(BaseModel):
    """Reference to an input source for audiobook processing."""

    input_type: SourceInputType = Field(..., description="Source type for the input")
    upload_id: str | None = Field(None, description="Upload id for a file previously uploaded")
    media_id: int | str | None = Field(None, description="Existing media id to read from")
    raw_text: str | None = Field(None, description="Raw text to process directly")

    model_config = {
        "json_schema_extra": {
            "example": {
                "input_type": "epub",
                "upload_id": "upload_4d8f",
            }
        }
    }

    @model_validator(mode="after")
    def _validate_payload(self) -> SourceRef:
        if not self.upload_id and not self.media_id and not (self.raw_text or "").strip():
            raise ValueError("source requires upload_id, media_id, or raw_text")
        return self


class ChapterSelection(BaseModel):
    """Chapter selection and voice overrides."""

    chapter_id: str = Field(..., description="Chapter identifier")
    include: bool = Field(..., description="Include or exclude chapter")
    voice: str | None = Field(None, description="Voice override for this chapter")
    speed: float | None = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Speed override for this chapter",
    )


class ChapterVoiceOverride(BaseModel):
    """Voice overrides used by voice profiles."""

    chapter_id: str = Field(..., description="Chapter identifier")
    voice: str | None = Field(None, description="Voice override for this chapter")
    speed: float | None = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Speed override for this chapter",
    )


class OutputOptions(BaseModel):
    """Audio output configuration."""

    merge: bool = Field(True, description="Merge chapters into a single audiobook output")
    per_chapter: bool = Field(True, description="Emit per-chapter audio outputs")
    formats: list[AudioFormat] = Field(..., description="Output audio formats")

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, value: list[AudioFormat]) -> list[AudioFormat]:
        if not value:
            raise ValueError("formats must include at least one audio format")
        if len(set(value)) != len(value):
            raise ValueError("formats must not contain duplicates")
        return value


class SubtitleOptions(BaseModel):
    """Subtitle generation options."""

    formats: list[SubtitleFormat] = Field(..., description="Subtitle formats to emit")
    mode: SubtitleMode = Field(..., description="Segmentation mode")
    variant: SubtitleVariant = Field(..., description="Styling variant")
    words_per_cue: int | None = Field(
        12,
        ge=1,
        description="Words per cue for word_count mode",
    )
    max_chars: int | None = Field(None, ge=10, description="Maximum characters per cue")
    max_lines: int | None = Field(None, ge=1, description="Maximum lines per cue")

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, value: list[SubtitleFormat]) -> list[SubtitleFormat]:
        if not value:
            raise ValueError("formats must include at least one subtitle format")
        if len(set(value)) != len(value):
            raise ValueError("formats must not contain duplicates")
        return value


class QueueOptions(BaseModel):
    """Job scheduling hints."""

    priority: int = Field(5, ge=1, le=10, description="Queue priority (1-10)")
    batch_group: str | None = Field(None, max_length=100, description="Optional batch group id")


class AudiobookJobItem(BaseModel):
    """Per-item override for batch jobs."""

    source: SourceRef = Field(..., description="Source reference for this item")
    tts_provider: str | None = Field(None, description="TTS provider override")
    tts_model: str | None = Field(None, description="TTS model override")
    voice_profile_id: str | None = Field(None, description="Voice profile id override")
    chapters: list[ChapterSelection] | None = Field(None, description="Per-item chapter selection")
    output: OutputOptions | None = Field(None, description="Per-item output options")
    subtitles: SubtitleOptions | None = Field(None, description="Per-item subtitle options")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


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


class AudiobookParseRequest(BaseModel):
    """Parse request for audiobook inputs."""

    source: SourceRef = Field(..., description="Source input reference")
    detect_chapters: bool = Field(True, description="Detect chapters automatically")
    custom_chapter_pattern: str | None = Field(None, description="Custom chapter regex pattern")
    language: str | None = Field(None, description="Language code for parsing")
    max_chars: int | None = Field(None, ge=1, description="Optional text truncation limit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": {"input_type": "epub", "upload_id": "upload_4d8f"},
                "detect_chapters": True,
                "custom_chapter_pattern": None,
                "language": "en",
            }
        }
    }


class AudiobookParseResponse(BaseModel):
    """Parse response with normalized text and chapter candidates."""

    project_id: str = Field(..., description="Project identifier")
    normalized_text: str = Field(..., description="Normalized text for processing")
    chapters: list[ChapterPreview] = Field(..., description="Detected chapter previews")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_id": "abk_01J7Y2M4G1",
                "normalized_text": "Chapter 1...\n\nChapter 2...",
                "chapters": [
                    {
                        "chapter_id": "ch_001",
                        "title": "Chapter 1",
                        "start_offset": 0,
                        "end_offset": 12458,
                        "word_count": 2450,
                    },
                    {
                        "chapter_id": "ch_002",
                        "title": "Chapter 2",
                        "start_offset": 12459,
                        "end_offset": 23877,
                        "word_count": 2201,
                    },
                ],
                "metadata": {
                    "title": "Example Book",
                    "author": "Example Author",
                    "source_type": "epub",
                },
            }
        }
    }


class AudiobookJobRequest(BaseModel):
    """Job creation payload for audiobook generation."""

    project_title: str = Field(..., min_length=1, max_length=200, description="Project title")
    source: SourceRef | None = Field(None, description="Single source reference")
    items: list[AudiobookJobItem] | None = Field(None, description="Batch items")
    tts_provider: str | None = Field(None, description="TTS provider override")
    tts_model: str | None = Field(None, description="TTS model override")
    voice_profile_id: str | None = Field(None, description="Voice profile id to apply")
    chapters: list[ChapterSelection] | None = Field(None, description="Chapter selection")
    output: OutputOptions | None = Field(None, description="Output options")
    subtitles: SubtitleOptions | None = Field(None, description="Subtitle options")
    queue: QueueOptions | None = Field(None, description="Queue options")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_title": "Example Book",
                "source": {"input_type": "epub", "upload_id": "upload_4d8f"},
                "voice_profile_id": "vp_01J7Y2NV6F",
                "chapters": [
                    {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0},
                    {"chapter_id": "ch_002", "include": True, "voice": "am_adam", "speed": 0.98},
                ],
                "output": {"merge": True, "per_chapter": True, "formats": ["mp3", "m4b"]},
                "subtitles": {"formats": ["srt", "vtt", "ass"], "mode": "sentence", "variant": "wide"},
                "queue": {"priority": 5, "batch_group": "client_batch_1"},
            }
        }
    }

    @model_validator(mode="after")
    def _validate_shape(self) -> AudiobookJobRequest:
        def _requires_subtitles(provider: str | None, model: str | None) -> bool:
            if provider:
                return str(provider).strip().lower() == "kokoro"
            if model:
                return str(model).strip().lower().startswith("kokoro")
            return True

        has_items = self.items is not None
        has_source = self.source is not None
        if has_items and has_source:
            raise ValueError("provide either items or source, not both")
        if not has_items and not has_source:
            raise ValueError("source is required when items are not provided")
        if has_items:
            if not self.items:
                raise ValueError("items must include at least one entry")
            if self.chapters is not None:
                raise ValueError("chapters cannot be set when items are provided")
            # Ensure output and subtitles resolve per item
            if self.output is None:
                for item in self.items:
                    if item.output is None:
                        raise ValueError("output required for each item or at top level")
            if self.subtitles is None:
                for item in self.items:
                    provider = item.tts_provider or self.tts_provider
                    model = item.tts_model or self.tts_model
                    if _requires_subtitles(provider, model) and item.subtitles is None:
                        raise ValueError("subtitles required for each item or at top level")
            for item in self.items:
                provider = item.tts_provider or self.tts_provider
                model = item.tts_model or self.tts_model
                requires_subtitles = _requires_subtitles(provider, model)
                if requires_subtitles:
                    if "subtitles" in item.model_fields_set and item.subtitles is None:
                        raise ValueError("subtitles cannot be null for kokoro items")
                    continue
                if item.subtitles is not None:
                    raise ValueError("subtitles are only supported for kokoro providers")
                if self.subtitles is not None and "subtitles" not in item.model_fields_set:
                    raise ValueError("subtitles must be null for non-kokoro items when defaults are set")
        else:
            if not self.chapters:
                raise ValueError("chapters must be provided for single-source jobs")
            if self.output is None:
                raise ValueError("output is required for single-source jobs")
            if _requires_subtitles(self.tts_provider, self.tts_model) and self.subtitles is None:
                raise ValueError("subtitles are required for single-source jobs")
            if not _requires_subtitles(self.tts_provider, self.tts_model) and self.subtitles is not None:
                raise ValueError("subtitles are only supported for kokoro providers")
        return self


class AudiobookJobCreateResponse(BaseModel):
    """Job creation response."""

    job_id: int = Field(..., description="Job identifier")
    project_id: str = Field(..., description="Project identifier")
    status: AudiobookJobStatus = Field(..., description="Initial job status")

    model_config = {
        "json_schema_extra": {"example": {"job_id": 12345, "project_id": "abk_01J7Y2M4G1", "status": "queued"}}
    }


class JobProgress(BaseModel):
    """Job progress metadata."""

    stage: str = Field(..., description="Current pipeline stage")
    chapter_index: int | None = Field(None, ge=0, description="Zero-based chapter index")
    chapters_total: int | None = Field(None, ge=0, description="Total chapters")
    item_index: int | None = Field(None, ge=0, description="Zero-based item index (batch jobs)")
    items_total: int | None = Field(None, ge=0, description="Total items (batch jobs)")
    percent: int | None = Field(None, ge=0, le=100, description="Percent complete")


class AudiobookJobStatusResponse(BaseModel):
    """Job status response."""

    job_id: int = Field(..., description="Job identifier")
    project_id: str = Field(..., description="Project identifier")
    status: AudiobookJobStatus = Field(..., description="Job status")
    progress: JobProgress | None = Field(None, description="Progress metadata")
    errors: list[str] = Field(default_factory=list, description="Error list")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": 12345,
                "project_id": "abk_01J7Y2M4G1",
                "status": "processing",
                "progress": {
                    "stage": "audiobook_tts",
                    "chapter_index": 2,
                    "chapters_total": 10,
                    "item_index": 0,
                    "items_total": 1,
                    "percent": 45,
                },
                "errors": [],
            }
        }
    }


class AudiobookProjectInfo(BaseModel):
    """Audiobook project summary."""

    project_db_id: int = Field(..., description="Internal project id")
    project_id: str | None = Field(None, description="External project id")
    title: str | None = Field(None, description="Project title")
    status: str | None = Field(None, description="Project status")
    source_ref: dict[str, Any] | None = Field(None, description="Source reference metadata")
    settings: dict[str, Any] | None = Field(None, description="Project settings")
    created_at: str = Field(..., description="Created timestamp")
    updated_at: str = Field(..., description="Updated timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_db_id": 12,
                "project_id": "abk_01J7Y2M4G1",
                "title": "Example Book",
                "status": "completed",
                "source_ref": {"input_type": "epub", "upload_id": "upload_4d8f"},
                "settings": {
                    "project_id": "abk_01J7Y2M4G1",
                    "project_title": "Example Book",
                    "language": "en",
                },
                "created_at": "2025-01-21T10:00:00+00:00",
                "updated_at": "2025-01-21T10:05:00+00:00",
            }
        }
    }


class AudiobookProjectListResponse(BaseModel):
    """List response for audiobook projects."""

    projects: list[AudiobookProjectInfo] = Field(..., description="Project list")
    total: int = Field(..., description="Total number of projects")
    limit: int = Field(..., description="Applied page limit")
    offset: int = Field(..., description="Applied page offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical pagination metadata")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> AudiobookProjectListResponse:
        return _default_offset_pagination_aliases(self)

    model_config = {
        "json_schema_extra": {
            "example": {
                "projects": [
                    {
                        "project_db_id": 12,
                        "project_id": "abk_01J7Y2M4G1",
                        "title": "Example Book",
                        "status": "completed",
                        "source_ref": {"input_type": "epub", "upload_id": "upload_4d8f"},
                        "settings": {"project_id": "abk_01J7Y2M4G1", "project_title": "Example Book"},
                        "created_at": "2025-01-21T10:00:00+00:00",
                        "updated_at": "2025-01-21T10:05:00+00:00",
                    }
                ],
                "total": 1,
                "limit": 100,
                "offset": 0,
                "pagination": {
                    "mode": "offset",
                    "total": 1,
                    "limit": 100,
                    "offset": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            }
        }
    }


class AudiobookProjectResponse(BaseModel):
    """Single project response."""

    project: AudiobookProjectInfo = Field(..., description="Project details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project": {
                    "project_db_id": 12,
                    "project_id": "abk_01J7Y2M4G1",
                    "title": "Example Book",
                    "status": "completed",
                    "source_ref": {"input_type": "epub", "upload_id": "upload_4d8f"},
                    "settings": {"project_id": "abk_01J7Y2M4G1", "project_title": "Example Book"},
                    "created_at": "2025-01-21T10:00:00+00:00",
                    "updated_at": "2025-01-21T10:05:00+00:00",
                }
            }
        }
    }


class AudiobookChapterInfo(BaseModel):
    """Audiobook chapter info."""

    id: int = Field(..., description="Chapter id")
    chapter_index: int = Field(..., description="Zero-based chapter index")
    title: str | None = Field(None, description="Chapter title")
    start_offset: int | None = Field(None, description="Start offset in characters")
    end_offset: int | None = Field(None, description="End offset in characters")
    voice_profile_id: str | None = Field(None, description="Voice profile id")
    speed: float | None = Field(None, description="Speed override")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chapter metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 101,
                "chapter_index": 0,
                "title": "Chapter 1",
                "start_offset": 0,
                "end_offset": 12458,
                "voice_profile_id": None,
                "speed": 1.0,
                "metadata": {"chapter_id": "ch_001", "item_index": 0},
            }
        }
    }


class AudiobookChapterListResponse(BaseModel):
    """List response for audiobook chapters."""

    project_id: str = Field(..., description="Project identifier")
    chapters: list[AudiobookChapterInfo] = Field(..., description="Chapter list")
    limit: int = Field(..., description="Applied page limit")
    offset: int = Field(..., description="Applied page offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical pagination metadata")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> AudiobookChapterListResponse:
        return _default_offset_pagination_aliases(self)

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_id": "abk_01J7Y2M4G1",
                "chapters": [
                    {
                        "id": 101,
                        "chapter_index": 0,
                        "title": "Chapter 1",
                        "start_offset": 0,
                        "end_offset": 12458,
                        "voice_profile_id": None,
                        "speed": 1.0,
                        "metadata": {"chapter_id": "ch_001", "item_index": 0},
                    }
                ],
                "limit": 200,
                "offset": 0,
                "pagination": {
                    "mode": "offset",
                    "total": None,
                    "limit": 200,
                    "offset": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            }
        }
    }


class ArtifactInfo(BaseModel):
    """Output artifact metadata."""

    artifact_type: AudiobookArtifactType = Field(..., description="Artifact category")
    format: str = Field(..., description="Artifact format")
    scope: AudiobookArtifactScope | None = Field(None, description="Artifact scope")
    chapter_id: str | None = Field(None, description="Chapter identifier")
    output_id: int = Field(..., description="Outputs table id")
    download_url: str = Field(..., description="Download URL")


class AudiobookArtifactsResponse(BaseModel):
    """Artifact list response."""

    project_id: str = Field(..., description="Project identifier")
    artifacts: list[ArtifactInfo] = Field(..., description="Artifact list")
    limit: int = Field(..., description="Applied page limit")
    offset: int = Field(..., description="Applied page offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical pagination metadata")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> AudiobookArtifactsResponse:
        return _default_offset_pagination_aliases(self)

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_id": "abk_01J7Y2M4G1",
                "artifacts": [
                    {
                        "artifact_type": "audio",
                        "format": "mp3",
                        "scope": "chapter",
                        "chapter_id": "ch_001",
                        "output_id": 456,
                        "download_url": "/api/v1/outputs/456/download",
                    },
                    {
                        "artifact_type": "subtitle",
                        "format": "srt",
                        "scope": "chapter",
                        "chapter_id": "ch_001",
                        "output_id": 789,
                        "download_url": "/api/v1/outputs/789/download",
                    },
                    {
                        "artifact_type": "alignment",
                        "format": "json",
                        "scope": "chapter",
                        "chapter_id": "ch_001",
                        "output_id": 790,
                        "download_url": "/api/v1/outputs/790/download",
                    },
                ],
                "limit": 200,
                "offset": 0,
                "pagination": {
                    "mode": "offset",
                    "total": None,
                    "limit": 200,
                    "offset": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            }
        }
    }


class VoiceProfileCreateRequest(BaseModel):
    """Create a voice profile."""

    name: str = Field(..., min_length=1, max_length=100, description="Profile name")
    default_voice: str = Field(..., description="Default voice id")
    default_speed: float = Field(..., ge=0.25, le=4.0, description="Default speed")
    chapter_overrides: list[ChapterVoiceOverride] | None = Field(
        default_factory=list,
        description="Optional chapter overrides",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Narrator + Dialog",
                "default_voice": "af_heart",
                "default_speed": 1.0,
                "chapter_overrides": [{"chapter_id": "ch_005", "voice": "am_adam", "speed": 0.98}],
            }
        }
    }


class VoiceProfileResponse(VoiceProfileCreateRequest):
    """Voice profile response."""

    profile_id: str = Field(..., description="Profile id")

    model_config = {
        "json_schema_extra": {
            "example": {
                "profile_id": "vp_01J7Y2NV6F",
                "name": "Narrator + Dialog",
                "default_voice": "af_heart",
                "default_speed": 1.0,
                "chapter_overrides": [{"chapter_id": "ch_005", "voice": "am_adam", "speed": 0.98}],
            }
        }
    }


class VoiceProfileListResponse(BaseModel):
    """List response for voice profiles."""

    profiles: list[VoiceProfileResponse] = Field(..., description="Voice profiles")
    total: int = Field(..., description="Total voice profiles matching the query")
    limit: int = Field(..., description="Applied page size")
    offset: int = Field(..., description="Applied offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical pagination metadata")

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> VoiceProfileListResponse:
        return _default_offset_pagination_aliases(self)


class VoiceProfileDeleteResponse(BaseModel):
    """Delete response for voice profiles."""

    profile_id: str = Field(..., description="Profile id")
    deleted: bool = Field(True, description="Deletion flag")


class SubtitleExportRequest(BaseModel):
    """Request to export subtitles from alignment data."""

    format: SubtitleFormat = Field(..., description="Subtitle format")
    mode: SubtitleMode = Field(..., description="Segmentation mode")
    variant: SubtitleVariant = Field(..., description="Styling variant")
    alignment: AlignmentPayload | None = Field(None, description="Alignment payload")
    alignment_output_id: int | None = Field(None, ge=1, description="Output id for stored alignment JSON")
    persist: bool | None = Field(None, description="Persist the generated subtitle output")
    cache_ttl_hours: int | None = Field(None, ge=1, description="Cache TTL (hours) when persisting")
    project_id: str | None = Field(None, description="External audiobook project id")
    chapter_id: str | None = Field(None, description="Chapter id for metadata linking")
    chapter_index: int | None = Field(None, ge=0, description="Chapter index for metadata linking")
    item_index: int | None = Field(None, ge=0, description="Item index for batch metadata")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata to persist")
    words_per_cue: int | None = Field(12, ge=1, description="Words per cue for word_count mode")
    max_chars: int | None = Field(None, ge=10, description="Maximum characters per cue")
    max_lines: int | None = Field(None, ge=1, description="Maximum lines per cue")

    @model_validator(mode="after")
    def _validate_alignment_source(self) -> SubtitleExportRequest:
        if self.alignment is None and self.alignment_output_id is None:
            raise ValueError("alignment or alignment_output_id is required")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "format": "srt",
                "mode": "sentence",
                "variant": "wide",
                "alignment": {
                    "words": [
                        {"word": "Hello", "start_ms": 0, "end_ms": 420},
                        {"word": "world", "start_ms": 450, "end_ms": 900},
                    ],
                    "engine": "kokoro",
                    "sample_rate": 24000,
                },
                "persist": False,
            }
        }
    }


class AudiobookErrorResponse(BaseModel):
    """Standard error response for audiobook endpoints."""

    error_code: str | None = Field(None, description="Machine-readable error code")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "invalid_request",
                "message": "Output formats must include at least one format",
                "details": {"field": "output.formats"},
            }
        }
    }


__all__ = [
    "SourceRef",
    "ChapterSelection",
    "ChapterVoiceOverride",
    "OutputOptions",
    "SubtitleOptions",
    "QueueOptions",
    "AudiobookJobItem",
    "AlignmentWord",
    "AlignmentPayload",
    "ChapterPreview",
    "AudiobookParseRequest",
    "AudiobookParseResponse",
    "AudiobookJobRequest",
    "AudiobookJobCreateResponse",
    "JobProgress",
    "AudiobookJobStatusResponse",
    "AudiobookProjectInfo",
    "AudiobookProjectListResponse",
    "AudiobookProjectResponse",
    "AudiobookChapterInfo",
    "AudiobookChapterListResponse",
    "ArtifactInfo",
    "AudiobookArtifactsResponse",
    "VoiceProfileCreateRequest",
    "VoiceProfileResponse",
    "VoiceProfileListResponse",
    "VoiceProfileDeleteResponse",
    "SubtitleExportRequest",
    "AudiobookErrorResponse",
]
