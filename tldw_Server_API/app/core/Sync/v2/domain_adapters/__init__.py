from __future__ import annotations

"""Domain-specific Sync v2 adapters."""

from .chat import ChatDomainAdapter
from .media import MediaCompatibilityAdapter, MediaMetadataAdapter, legacy_media_sync_log_to_envelope
from .notes import NotesDomainAdapter
from .notes_link import NotesLinkDomainAdapter
from .notes_organization import NotesOrganizationDomainAdapter
from .notes_task import NotesTaskDomainAdapter
from .notes_task_activity import NotesTaskActivityDomainAdapter
from .personal_context import PersonalContextDomainAdapter
from .source_cache import SourceCacheAdapter
from .workspaces import WorkspacesDomainAdapter

__all__ = [
    "ChatDomainAdapter",
    "MediaCompatibilityAdapter",
    "MediaMetadataAdapter",
    "NotesDomainAdapter",
    "NotesLinkDomainAdapter",
    "NotesTaskDomainAdapter",
    "NotesTaskActivityDomainAdapter",
    "PersonalContextDomainAdapter",
    "NotesOrganizationDomainAdapter",
    "SourceCacheAdapter",
    "WorkspacesDomainAdapter",
    "legacy_media_sync_log_to_envelope",
]
