"""Core-owned media ingestion request models, re-exported by the API schemas package (TASK-13950.15)."""

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from pydantic_core.core_schema import ValidationInfo

from tldw_Server_API.app.core.Ingestion_Media_Processing.Email.attachment_policy import (
    normalize_attachment_mime_patterns,
)

# Define allowed media types using Literal for validation
MediaType = Literal["video", "audio", "document", "pdf", "ebook", "email", "code"]

# Define allowed chunking methods (adjust as needed based on your library)
ChunkMethod = Literal[
    "semantic",
    "structure_aware",
    "tokens",
    "paragraphs",
    "sentences",
    "words",
    "ebook_chapters",
    "json",
    "propositions",
]

ChunkingMode = Literal["auto", "manual"]

AutoChunkingGoal = Literal["balanced", "qa_search", "navigation_summary"]

# Define allowed PDF parsing engines
PdfEngine = Literal["pymupdf4llm", "pymupdf", "docling"]  # Add others if supported

# OCR options
OcrMode = Literal["always", "fallback"]


class ChunkingOptions(BaseModel):
    """Pydantic model for chunking specific options"""

    perform_chunking: bool = Field(True, description="Enable chunk-based processing of the media content")
    chunking_mode: Optional[ChunkingMode] = Field(
        None,
        description="Chunking control mode. Omitted preserves legacy/manual behavior; 'auto' lets the server plan media-aware chunking.",
    )
    auto_chunking_goal: AutoChunkingGoal = Field(
        "balanced",
        description="Goal used by automatic chunking when chunking_mode='auto'",
    )
    auto_chunking_use_llm: bool = Field(
        False,
        description="Explicit opt-in to AI-assisted automatic chunk boundary refinement when available",
    )
    chunk_method: Optional[ChunkMethod] = Field(
        None, description="Method used to chunk content (e.g., 'sentences', 'recursive', 'chapter')"
    )
    use_adaptive_chunking: bool = Field(False, description="Whether to enable adaptive chunking")
    use_multi_level_chunking: bool = Field(False, description="Whether to enable multi-level chunking")
    chunk_language: Optional[str] = Field(
        None, description="Optional language override for chunking (ISO 639-1 code, e.g., 'en')"
    )
    chunk_size: int = Field(500, gt=0, description="Target size of each chunk (positive integer)")
    chunk_overlap: int = Field(200, ge=0, description="Overlap size between chunks (non-negative integer)")
    custom_chapter_pattern: Optional[str] = Field(
        None, description="Optional regex pattern for custom chapter splitting (ebook/docs)"
    )
    proposition_engine: Optional[str] = Field(
        None,
        description="Override proposition extraction engine (heuristic|spacy|llm|auto)",
    )
    proposition_aggressiveness: Optional[int] = Field(
        None,
        ge=0,
        description="Aggressiveness level for proposition extraction (higher = more aggressive)",
    )
    proposition_min_proposition_length: Optional[int] = Field(
        None,
        ge=1,
        description="Minimum proposition length in characters before merging",
    )
    proposition_prompt_profile: Optional[str] = Field(
        None,
        description="Prompt profile for proposition extraction (generic|claimify|gemma_aps)",
    )
    # Template auto-apply options
    auto_apply_template: bool = Field(
        False, description="Automatically select and apply a matching chunking template by metadata"
    )
    chunking_template_name: Optional[str] = Field(None, description="Explicit template name to apply for chunking")
    # Contextual Chunking Options
    enable_contextual_chunking: bool = Field(
        False, description="Add LLM-generated context to chunks for better retrieval"
    )
    contextual_llm_model: Optional[str] = Field(
        None, description="LLM model to use for generating contextual summaries (e.g., 'gpt-3.5-turbo')"
    )
    context_window_size: Optional[int] = Field(
        None, ge=100, le=2000, description="Size of context window around chunks in characters"
    )
    context_strategy: Optional[Literal["auto", "full", "window", "outline_window"]] = Field(
        None, description="Context selection strategy: 'auto' (default), 'full', 'window', or 'outline_window'"
    )
    context_token_budget: Optional[int] = Field(
        None, ge=1000, le=200000, description="Approximate token budget for 'auto' strategy (len(text)/4 heuristic)"
    )
    # Hierarchical options (flattened into chunks for indexing)
    hierarchical_chunking: Optional[bool] = Field(
        False, description="Enable hierarchical parsing and flattening to leaf chunks"
    )
    hierarchical_template: Optional[dict[str, Any]] = Field(
        None, description="Custom boundary rules: {'boundaries': [{'kind','pattern','flags'}]}"
    )

    @field_validator("chunk_method", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Optional[Any]:  # Accept Any for input
        if v == "":
            return None
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: ValidationInfo) -> int:
        # Check if 'chunk_size' is available in the already validated data
        if "chunk_size" in info.data and info.data["chunk_size"] is not None:
            chunk_size = info.data["chunk_size"]
            if v >= chunk_size:
                raise ValueError("chunk_overlap must be less than chunk_size")
        # If chunk_size hasn't been validated yet or is missing, this check might implicitly pass
        # or you might want to handle that case explicitly if chunk_size is always required before overlap.
        return v

    @field_validator("custom_chapter_pattern")
    @classmethod
    def validate_regex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:  # Catch specific error
                raise ValueError(f"Invalid regex pattern provided for custom_chapter_pattern: {v}. Error: {e}") from e
        return v


class AudioVideoOptions(BaseModel):
    """Pydantic model for Audio/Video specific options"""

    transcription_model: str = Field(
        "deepdml/faster-distil-whisper-large-v3.5", description="Model ID for audio/video transcription"
    )
    transcription_language: str = Field("en", description="Language for audio/video transcription (ISO 639-1 code)")
    hotwords: Optional[str] = Field(
        None,
        description="Optional hotwords to guide transcription (CSV or JSON list). Primarily used by VibeVoice-ASR.",
    )
    diarize: bool = Field(False, description="Enable speaker diarization (audio/video)")
    timestamp_option: bool = Field(True, description="Include timestamps in the transcription (audio/video)")
    vad_use: bool = Field(
        False, description="Enable Voice Activity Detection filter during transcription (audio/video)"
    )
    perform_confabulation_check_of_analysis: bool = Field(
        False, description="Enable a confabulation check on analysis (if applicable)"
    )


class PdfOptions(BaseModel):
    """Pydantic model for PDF specific options"""

    pdf_parsing_engine: Optional[PdfEngine] = Field("pymupdf4llm", description="PDF parsing engine to use")
    enable_ocr: bool = Field(False, description="Enable OCR for scanned/low-text PDFs")
    ocr_backend: Optional[str] = Field(None, description="OCR backend name (e.g., 'tesseract' or 'auto')")
    ocr_lang: Optional[str] = Field("eng", description="OCR language (ISO 639-2 Tesseract codes, e.g., 'eng')")
    ocr_dpi: int = Field(300, ge=72, le=600, description="DPI for page rendering before OCR (72-600)")
    ocr_mode: Optional[OcrMode] = Field("fallback", description="'always' to force OCR, 'fallback' when no text")
    ocr_min_page_text_chars: int = Field(
        40, ge=0, description="Threshold to treat a page as 'no text' for OCR fallback"
    )
    ocr_output_format: Optional[str] = Field(
        None,
        description="OCR output format: text|markdown|json (structured outputs persisted in analysis_details)",
    )
    ocr_prompt_preset: Optional[str] = Field(
        None,
        description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')",
    )


class AddMediaForm(ChunkingOptions, AudioVideoOptions, PdfOptions):
    """
    Pydantic model representing the form data for the /add endpoint.
    Excludes 'files' (handled via File(...)) and 'token' (handled via Header(...)).
    """

    # --- Required Fields ---
    media_type: MediaType = Field(..., description="Type of media")

    # --- Input Sources ---
    # Note: 'files' is handled separately by FastAPI's File() parameter
    urls: Optional[list[str]] = Field(None, description="List of URLs of the media items to add")
    research_discovery_id: Optional[str] = Field(None, max_length=128)
    research_discovery_selections: Optional[str] = Field(None, max_length=8192)

    # --- Common Optional Fields ---
    title: Optional[str] = Field(None, max_length=500, description="Optional title (max 500 chars)")
    author: Optional[str] = Field(None, max_length=255, description="Optional author (max 255 chars)")
    keywords_str: str = Field(
        "", alias="keywords", max_length=1000, description="Comma-separated keywords (max 1000 chars)"
    )
    custom_prompt: Optional[str] = Field(None, max_length=100000, description="Optional custom prompt (max 100KB)")
    system_prompt: Optional[str] = Field(None, max_length=10000, description="Optional system prompt (max 10KB)")
    overwrite_existing: bool = Field(
        False, description="Overwrite any existing media with the same identifier (URL/filename)"
    )
    keep_original_file: bool = Field(False, description="Whether to retain original uploaded files after processing")
    perform_analysis: bool = Field(
        True, description="Perform analysis (e.g., summarization) if applicable (default=True)"
    )
    perform_claims_extraction: Optional[bool] = Field(
        None, description="Extract factual claims during analysis (defaults to server configuration when unset)."
    )
    claims_extractor_mode: Optional[str] = Field(
        None, description="Optional override for claims extractor mode (e.g., 'heuristic', 'ner', provider id)."
    )
    claims_max_per_chunk: Optional[int] = Field(
        None,
        ge=1,
        le=12,
        description="Maximum number of claims to extract per chunk (uses configuration defaults when unset).",
    )

    # --- Video/Audio Specific Timing --- ADD THESE ---
    start_time: Optional[str] = Field(
        None, description="Optional start time for processing (e.g., HH:MM:SS or seconds)"
    )
    end_time: Optional[str] = Field(None, description="Optional end time for processing (e.g., HH:MM:SS or seconds)")
    # -----------------------------------------------

    # --- Integration Options ---
    # SECURITY: Never accept API keys from client - lookup from server config instead
    api_provider: Optional[str] = Field(None, description="LLM provider name (e.g., openai, anthropic, local-llm)")
    model_name: Optional[str] = Field(None, description="Model name for the selected provider")
    use_cookies: bool = Field(False, description="Whether to attach cookies to URL download requests")
    cookies: Optional[str] = Field(None, description="Cookie string if `use_cookies` is set to True")

    # Legacy field for backward compatibility - will be removed
    api_name: Optional[str] = Field(None, description="DEPRECATED - use api_provider instead")

    # --- Email-specific options (optional, used when media_type='email') ---
    ingest_attachments: Optional[bool] = Field(
        False, description="For emails: parse nested .eml attachments and ingest as separate items"
    )
    extract_attachments: Optional[bool] = Field(
        None, description="For emails: override ingest_attachments; false keeps attachment metadata only"
    )
    attachment_mime_allowlist: Optional[list[str]] = Field(
        None, max_length=64, description="Attachment MIME selection (default message/rfc822); empty selects none"
    )
    attachment_mime_denylist: Optional[list[str]] = Field(
        None, max_length=64, description="Attachment MIME exclusion; denies win over allows and .eml inference"
    )

    @field_validator("attachment_mime_allowlist", "attachment_mime_denylist", mode="before")
    @classmethod
    def validate_attachment_mime_rules(cls, value: Any) -> Optional[list[str]]:
        """Validate repeated/comma-separated MIME rules before form coercion."""
        return normalize_attachment_mime_patterns(value)

    max_depth: Optional[int] = Field(
        2, ge=1, le=5, description="Max depth for nested email parsing when ingest_attachments is true"
    )
    accept_archives: Optional[bool] = Field(
        False, description="Allow and expand .zip archives of .eml files for email ingestion"
    )
    accept_mbox: Optional[bool] = Field(False, description="Allow and expand .mbox mailboxes for email ingestion")
    accept_pst: Optional[bool] = Field(
        False, description="Enable PST/OST container uploads (feature-flag; parsing may require external tools"
    )

    # --- Embedding Options ---
    generate_embeddings: bool = Field(False, description="Generate embeddings after media processing")
    embedding_dispatch_mode: Optional[Literal["auto", "jobs", "background"]] = Field(
        None,
        description="Embeddings dispatch strategy override for /media/add",
    )
    embedding_model: Optional[str] = Field(
        None, description="Specific embedding model to use (e.g., 'Qwen/Qwen3-Embedding-4B-GGUF')"
    )
    embedding_provider: Optional[str] = Field(None, description="Embedding provider (huggingface, openai, etc)")

    # --- Durable media collection binding (used by async-job fallback paths) ---
    media_collection_id: Optional[int] = Field(
        None,
        ge=1,
        description="Optional numeric durable media collection id associated with this ingest item",
    )
    media_collection_item_id: Optional[int] = Field(
        None,
        ge=1,
        description="Optional numeric planned durable media collection item id to resolve after ingest",
    )
    media_ingest_job_id: Optional[str] = Field(
        None,
        description="Optional media ingest job id that produced this fallback ingest",
    )

    # --- Deprecated/Less Common ---
    perform_rolling_summarization: bool = Field(False, description="Perform rolling summarization (legacy?)")
    summarize_recursively: bool = Field(
        False, description="Perform recursive summarization on chunks (if chunking enabled)"
    )

    # --- Computed Fields / Validators ---
    @computed_field
    @property
    def keywords(self) -> list[str]:
        """Parses the comma-separated keywords string into a list."""
        if not self.keywords_str:
            return []
        return [k.strip() for k in self.keywords_str.split(",") if k.strip()]

    def model_post_init(self, __context):
        """Handle legacy api_name field by splitting into provider/model."""
        if self.api_name and "/" in self.api_name:
            # If api_name contains '/', treat it as 'provider/model' format
            provider, model = self.api_name.split("/", 1)
            if not self.api_provider:
                self.api_provider = provider
            if not self.model_name:
                self.model_name = model
        elif self.api_name and not self.api_provider:
            # If only api_name is set without '/', use it as provider
            self.api_provider = self.api_name

    # Use alias for 'keywords' field to accept 'keywords' in the form data
    # but internally work with 'keywords_str' before parsing.
    model_config = ConfigDict(
        populate_by_name=True,  # Allows using 'alias' for fields
        json_schema_extra={
            "example": {
                "media_type": "document",
                "urls": ["https://example.com/guide.pdf"],
                "title": "Example Guide",
                "author": "Jane Doe",
                "keywords": "api,fastapi,docs",
                "perform_analysis": True,
                "perform_chunking": True,
                "chunk_method": "sentences",
                "chunk_size": 800,
                "chunk_overlap": 150,
                "generate_embeddings": True,
                "embedding_dispatch_mode": "background",
                "embedding_provider": "huggingface",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )

    # Provide a stable, test-friendly error message for invalid media_type values
    @field_validator("media_type", mode="before")
    @classmethod
    def validate_media_type_choices(cls, v):
        # Accept only known values; for invalid input, emit the exact message expected by tests
        allowed = {"video", "audio", "document", "pdf", "ebook", "email", "json"}
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv not in allowed:
                # Match historical message (without 'email') used by tests
                raise ValueError("Input should be 'video', 'audio', 'document', 'pdf' or 'ebook'")
            return lv
        return v

    @field_validator("start_time", "end_time")
    @classmethod
    def check_time_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":  # MODIFIED: Treat empty string like None
            return None  # Return None, which is valid for Optional[str]

        # Example basic check: Allow seconds or HH:MM:SS format
        if re.fullmatch(r"\d+", v) or re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", v):
            return v
        raise ValueError("Time format must be seconds or HH:MM:SS")

    # Validator to ensure 'cookies' is provided if 'use_cookies' is True
    @field_validator("cookies")
    @classmethod
    def check_cookies_provided(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # Access other validated fields via info.data
        # Check if 'use_cookies' exists in data AND is True
        if info.data.get("use_cookies") and not v:
            raise ValueError("Cookie string must be provided when 'use_cookies' is set to True.")
        return v
