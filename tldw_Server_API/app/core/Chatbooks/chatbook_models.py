# chatbook_models.py
# Description: Data models for chatbook/knowledge pack structures with multi-user support
# Adapted from single-user to multi-user architecture
#
"""
Chatbook Models for Multi-User Environment
------------------------------------------

Defines the data structures for chatbooks including manifest,
content organization, and metadata with user isolation.

Key Adaptations from Single-User:
- User-specific exports with access control
- Sanitized content to prevent cross-user data leakage
- Job-based export/import for large operations
- Temporary storage with automatic cleanup
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class ChatbookVersion(str, Enum):
    """
    Chatbook format versions.

    The chatbook format uses semantic versioning (MAJOR.MINOR.PATCH) to track
    compatibility and feature sets:

    - V1 (1.0.0): Initial stable format with basic content types and metadata
    - V2 (2.0.0): Future version with enhanced features and extended metadata

    Note: Both "1.0" and "1.0.0" are accepted for V1 compatibility, but "1.0.0"
    is the canonical format following semantic versioning conventions.
    """
    V1 = "1.0.0"  # Primary V1 format using semantic versioning
    V1_LEGACY = "1.0"  # Legacy format for backward compatibility
    V2 = "2.0.0"  # Future version with enhanced features


class ContentType(str, Enum):
    """Types of content that can be included in a chatbook."""
    CONVERSATION = "conversation"
    NOTE = "note"
    CHARACTER = "character"
    MEDIA = "media"
    EMBEDDING = "embedding"
    PROMPT = "prompt"
    EVALUATION = "evaluation"
    WORLD_BOOK = "world_book"
    DICTIONARY = "dictionary"
    GENERATED_DOCUMENT = "generated_document"
    EXPLAINER_SESSION = "explainer_session"


class ExportStatus(str, Enum):
    """Status of export job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ImportStatus(str, Enum):
    """Status of import job."""
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConflictResolution(str, Enum):
    """How to handle conflicts during import (full set including future strategies)."""
    SKIP = "skip"          # Skip conflicting items
    OVERWRITE = "overwrite"  # Overwrite existing items
    RENAME = "rename"      # Rename imported items
    MERGE = "merge"        # Merge with existing (where applicable)
    ASK = "ask"           # Ask user for each conflict (not for API)


@dataclass
class ImportStatusData:
    """Import status tracking data."""
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "conflicts": self.conflicts,
            "warnings": self.warnings
        }


@dataclass
class ContentItem:
    """Individual content item in a chatbook."""
    id: str
    type: ContentType
    title: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None  # Relative path within chatbook
    checksum: Optional[str] = None  # For integrity verification

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value if hasattr(self.type, 'value') else str(self.type),
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "checksum": self.checksum
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ContentItem':
        """Create ContentItem from dictionary."""
        return cls(
            id=data["id"],
            type=ContentType(data["type"]),
            title=data["title"],
            description=data.get("description"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            file_path=data.get("file_path"),
            checksum=data.get("checksum")
        )


@dataclass
class Relationship:
    """Relationship between content items."""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "references", "parent_of", "requires", "uses_character"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Relationship':
        """Create Relationship from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ChatbookManifest:
    """Manifest file containing chatbook metadata and contents listing."""
    version: ChatbookVersion
    name: str
    description: str
    author: Optional[str] = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    export_id: Optional[str] = None  # Unique export identifier
    exported_at: Optional[str] = None  # For compatibility with tests
    user_id: Optional[str] = None  # User who created the export

    # Content summary
    content_items: list[ContentItem] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    content_summary: Optional[dict[str, int]] = field(default_factory=dict)  # For compatibility
    metadata: Optional[dict[str, Any]] = field(default_factory=dict)  # For compatibility

    # Configuration
    include_media: bool = False
    include_embeddings: bool = False
    include_generated_content: bool = True
    media_quality: str = "compressed"  # thumbnail, compressed, original
    max_file_size_mb: int = 100  # Maximum chatbook size

    # Statistics
    total_conversations: int = 0
    total_notes: int = 0
    total_characters: int = 0
    total_media_items: int = 0
    total_prompts: int = 0
    total_evaluations: int = 0
    total_embeddings: int = 0
    total_world_books: int = 0
    total_dictionaries: int = 0
    total_documents: int = 0
    total_explainer_sessions: int = 0
    total_size_bytes: int = 0

    # Metadata
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    language: str = "en"
    license: Optional[str] = None
    binary_limits: dict[str, int] = field(default_factory=dict)
    truncation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        metadata: dict[str, Any] = {
            "tags": self.tags,
            "categories": self.categories,
            "language": self.language,
            "license": self.license
        }
        extra_metadata = self.metadata or {}
        for key, value in extra_metadata.items():
            if key not in metadata:
                metadata[key] = value
        if self.binary_limits:
            metadata["binary_limits"] = self.binary_limits

        payload = {
            "version": self.version.value if hasattr(self.version, 'value') else str(self.version),
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "export_id": self.export_id,
            "content_items": [item.to_dict() for item in self.content_items],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "configuration": {
                "include_media": self.include_media,
                "include_embeddings": self.include_embeddings,
                "include_generated_content": self.include_generated_content,
                "media_quality": self.media_quality,
                "max_file_size_mb": self.max_file_size_mb
            },
            "statistics": {
                "total_conversations": self.total_conversations,
                "total_notes": self.total_notes,
                "total_characters": self.total_characters,
                "total_media_items": self.total_media_items,
                "total_prompts": self.total_prompts,
                "total_evaluations": self.total_evaluations,
                "total_embeddings": self.total_embeddings,
                "total_world_books": self.total_world_books,
                "total_dictionaries": self.total_dictionaries,
                "total_documents": self.total_documents,
                "total_explainer_sessions": self.total_explainer_sessions,
                "total_size_bytes": self.total_size_bytes
            },
            "metadata": metadata,
            "user_info": {
                "user_id": self.user_id
            }
        }
        if self.truncation:
            payload["truncation"] = self.truncation
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatbookManifest':
        """Create ChatbookManifest from dictionary."""
        config = data.get("configuration", {})
        stats = data.get("statistics", {})
        meta = data.get("metadata", {}) or {}
        binary_limits = meta.get("binary_limits", {})
        extra_metadata = {
            key: value
            for key, value in meta.items()
            if key not in {"tags", "categories", "language", "license", "binary_limits"}
        }
        user = data.get("user_info", {})
        truncation = data.get("truncation", {}) or {}

        return cls(
            version=ChatbookVersion(data["version"]),
            name=data["name"],
            description=data["description"],
            author=data.get("author"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else _utc_now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else _utc_now(),
            export_id=data.get("export_id"),
            content_items=[ContentItem.from_dict(item) for item in data.get("content_items", [])],
            relationships=[Relationship.from_dict(rel) for rel in data.get("relationships", [])],
            include_media=config.get("include_media", False),
            include_embeddings=config.get("include_embeddings", False),
            include_generated_content=config.get("include_generated_content", True),
            media_quality=config.get("media_quality", "compressed"),
            max_file_size_mb=config.get("max_file_size_mb", 100),
            total_conversations=stats.get("total_conversations", 0),
            total_notes=stats.get("total_notes", 0),
            total_characters=stats.get("total_characters", 0),
            total_media_items=stats.get("total_media_items", 0),
            total_prompts=stats.get("total_prompts", 0),
            total_evaluations=stats.get("total_evaluations", 0),
            total_embeddings=stats.get("total_embeddings", 0),
            total_world_books=stats.get("total_world_books", 0),
            total_dictionaries=stats.get("total_dictionaries", 0),
            total_documents=stats.get("total_documents", 0),
            total_explainer_sessions=stats.get("total_explainer_sessions", 0),
            total_size_bytes=stats.get("total_size_bytes", 0),
            tags=meta.get("tags", []),
            categories=meta.get("categories", []),
            language=meta.get("language", "en"),
            license=meta.get("license"),
            metadata=extra_metadata,
            binary_limits=binary_limits,
            truncation=truncation,
            user_id=user.get("user_id")
        )


@dataclass
class ChatbookContent:
    """Container for all content in a chatbook."""
    conversations: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    characters: dict[str, Any] = field(default_factory=dict)
    media: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, Any] = field(default_factory=dict)
    prompts: dict[str, Any] = field(default_factory=dict)
    evaluations: dict[str, Any] = field(default_factory=dict)
    world_books: dict[str, Any] = field(default_factory=dict)
    dictionaries: dict[str, Any] = field(default_factory=dict)
    generated_documents: dict[str, Any] = field(default_factory=dict)
    explainer_sessions: dict[str, Any] = field(default_factory=dict)

    def get_all_ids(self) -> set[str]:
        """Get all content IDs."""
        all_ids = set()
        for content_dict in [
            self.conversations, self.notes, self.characters,
            self.media, self.embeddings, self.prompts,
            self.evaluations, self.world_books, self.dictionaries,
            self.generated_documents, self.explainer_sessions
        ]:
            all_ids.update(content_dict.keys())
        return all_ids


@dataclass
class ExportJob:
    """Track export job status."""
    job_id: str
    user_id: str
    status: ExportStatus
    chatbook_name: str
    output_path: Optional[str] = None
    created_at: datetime = field(default_factory=_utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = field(default_factory=dict)  # For storing additional data

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "chatbook_name": self.chatbook_name,
            "output_path": self.output_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress_percentage": self.progress_percentage,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "file_size_bytes": self.file_size_bytes,
            "download_url": self.download_url,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }

    # Provide dict-like access for test compatibility
    def __getitem__(self, key: str):  # type: ignore[override]
        val = getattr(self, key)
        if isinstance(val, Enum):
            return val.value
        return val


@dataclass
class ImportJob:
    """Track import job status."""
    job_id: str
    user_id: str
    status: ImportStatus
    chatbook_path: str
    created_at: datetime = field(default_factory=_utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "chatbook_path": self.chatbook_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress_percentage": self.progress_percentage,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "conflicts": self.conflicts,
            "warnings": self.warnings
        }

    # Provide dict-like access for test compatibility
    def __getitem__(self, key: str):  # type: ignore[override]
        val = getattr(self, key)
        if isinstance(val, Enum):
            return val.value
        return val


@dataclass
class ImportConflict:
    """Represents a conflict during import."""
    item_id: str
    item_type: ContentType
    item_title: str
    existing_id: str
    existing_title: str
    suggested_resolution: ConflictResolution
    user_resolution: Optional[ConflictResolution] = None
    new_title: Optional[str] = None  # For rename resolution

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "item_title": self.item_title,
            "existing_id": self.existing_id,
            "existing_title": self.existing_title,
            "suggested_resolution": self.suggested_resolution.value,
            "user_resolution": self.user_resolution.value if self.user_resolution else None,
            "new_title": self.new_title
        }


# Export main classes
__all__ = [
    'ChatbookVersion',
    'ContentType',
    'ExportStatus',
    'ImportStatus',
    'ConflictResolution',
    'ContentItem',
    'Relationship',
    'ChatbookManifest',
    'ChatbookContent',
    'ExportJob',
    'ImportJob',
    'ImportConflict'
]
