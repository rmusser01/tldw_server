from __future__ import annotations

"""Sync v2 service composition helpers shared by HTTP and non-HTTP entrypoints."""

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    _resolve_user_id_for_storage,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SYNC_DB_FILENAME, SyncDatabase
from tldw_Server_API.app.core.Utils.path_utils import safe_join

from .adapters import StaticSyncAdapter, SyncAdapterRegistry
from .blob_store import LocalSyncBlobStore
from .domain_adapters.attachment_refs import AttachmentRefDomainAdapter
from .domain_adapters.media import MediaMetadataAdapter
from .domain_adapters.notes import NotesDomainAdapter
from .domain_adapters.notes_link import NotesLinkDomainAdapter
from .domain_adapters.notes_organization import NotesOrganizationDomainAdapter
from .domain_adapters.notes_task import NotesTaskDomainAdapter
from .domain_adapters.notes_task_activity import NotesTaskActivityDomainAdapter
from .domain_adapters.source_cache import SourceCacheAdapter
from .domain_adapters.workspaces import WorkspacesDomainAdapter
from .materializers import (
    AttachmentRefMaterializer,
    ChatConversationMaterializer,
    ChatMessageMaterializer,
    MediaMetadataMaterializer,
    NotesLinkMaterializer,
    NotesMaterializer,
    NotesOrganizationMaterializer,
    NotesTaskActivityMaterializer,
    NotesTaskMaterializer,
    SourceCacheMaterializer,
    SyncMaterializer,
)
from .models import (
    M1_SYNC_DOMAINS,
    MEDIA_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SOURCE_CACHE_SYNC_DOMAINS,
    WORKSPACE_SYNC_DOMAINS,
    SyncDomain,
)
from .notes_attachment_bootstrap import NotesAttachmentBootstrapper
from .notes_link_bootstrap import NotesLinkBootstrapper
from .notes_organization_bootstrap import NotesOrganizationBootstrapper
from .notes_task_activity_bootstrap import NotesTaskActivityBootstrapper
from .notes_task_bootstrap import NotesTaskBootstrapper
from .security import server_trusted_encryption_status_from_env
from .service import (
    SyncV2Service,
    SyncV2Settings,
    personal_context_sync_capabilities_from_env,
)
from .store import SyncV2Store


@lru_cache(maxsize=1)
def default_sync_v2_registry() -> SyncAdapterRegistry:
    """Return the default Sync v2 domain adapter registry."""

    return SyncAdapterRegistry(
        [
            (
                AttachmentRefDomainAdapter(
                    v2_writes_enabled=_sync_v2_bool_env(
                        "SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC",
                        default=False,
                    )
                )
                if domain == "attachment.ref"
                else NotesDomainAdapter()
                if domain == "notes.note"
                else StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
            )
            for domain in M1_SYNC_DOMAINS
        ]
        + [WorkspacesDomainAdapter(domain=domain) for domain in WORKSPACE_SYNC_DOMAINS]
        + [SourceCacheAdapter(domain=domain) for domain in SOURCE_CACHE_SYNC_DOMAINS]
        + [MediaMetadataAdapter(domain=domain) for domain in MEDIA_SYNC_DOMAINS]
        + [NotesOrganizationDomainAdapter(domain=domain) for domain in NOTES_ORGANIZATION_DOMAINS]
        + [NotesLinkDomainAdapter()]
        + [NotesTaskDomainAdapter()]
        + [NotesTaskActivityDomainAdapter()]
    )


def sync_v2_service_for_user(user_id: str) -> SyncV2Service:
    """Build a Sync v2 service for a user using cached storage wiring."""

    store = _sync_v2_store_for_user(
        user_id,
        os.getenv("SYNC_V2_DATABASE_URL", "").strip(),
        os.getenv("SYNC_V2_SQLITE_PATH", "").strip(),
    )
    note_db = _chacha_notes_db_for_user(user_id)
    adapters = default_sync_v2_registry()
    settings = _sync_v2_settings_from_env()
    materializers: dict[SyncDomain, SyncMaterializer] = {
        "attachment.ref": AttachmentRefMaterializer(note_db),
        "chat.conversation": ChatConversationMaterializer(note_db),
        "chat.message": ChatMessageMaterializer(note_db),
        "notes.note": NotesMaterializer(note_db),
        "notes.link": NotesLinkMaterializer(note_db),
        "notes.task": NotesTaskMaterializer(note_db),
        "notes.task_activity": NotesTaskActivityMaterializer(note_db),
        "source_cache.entry": SourceCacheMaterializer(),
        "media.item": MediaMetadataMaterializer(domain="media.item"),
        "media.keyword": MediaMetadataMaterializer(domain="media.keyword"),
        "media.keyword_link": MediaMetadataMaterializer(domain="media.keyword_link"),
        **{
            domain: NotesOrganizationMaterializer(note_db, domain)
            for domain in NOTES_ORGANIZATION_DOMAINS
        },
    }
    _validate_notes_organization_components(
        adapters=adapters,
        materializers=materializers,
        advertised_domains=settings.supported_domains,
    )
    _validate_notes_link_components(
        adapters=adapters,
        materializers=materializers,
        advertised_domains=settings.supported_domains,
    )
    _validate_notes_task_components(
        adapters=adapters,
        materializers=materializers,
    )
    return SyncV2Service(
        store=store,
        adapters=adapters,
        materializers=materializers,
        blob_store=_sync_v2_blob_store_for_user(user_id),
        settings=settings,
        workspace_access_checker=_workspace_access_checker,
        dataset_bootstrapper=NotesOrganizationBootstrapper(note_db),
        notes_link_bootstrapper=NotesLinkBootstrapper(note_db),
        notes_attachment_bootstrapper=NotesAttachmentBootstrapper(note_db),
        notes_task_bootstrapper=NotesTaskBootstrapper(note_db),
        notes_task_activity_bootstrapper=NotesTaskActivityBootstrapper(note_db),
    )


def _validate_notes_organization_components(
    *,
    adapters: SyncAdapterRegistry,
    materializers: Mapping[SyncDomain, SyncMaterializer],
    advertised_domains: list[SyncDomain],
) -> None:
    """Fail closed when an advertised organization domain is not fully wired."""

    for domain in NOTES_ORGANIZATION_DOMAINS:
        if domain not in advertised_domains:
            continue
        if not adapters.has_domain(domain) or not isinstance(
            adapters.get(domain), NotesOrganizationDomainAdapter
        ):
            raise RuntimeError(f"Advertised Sync domain has no strict adapter: {domain}")
        materializer = materializers.get(domain)
        if not isinstance(materializer, NotesOrganizationMaterializer) or materializer.domain != domain:
            raise RuntimeError(f"Advertised Sync domain has no user-bound materializer: {domain}")


def _validate_notes_link_components(
    *,
    adapters: SyncAdapterRegistry,
    materializers: Mapping[SyncDomain, SyncMaterializer],
    advertised_domains: list[SyncDomain],
) -> None:
    """Fail closed when advertised notes.link support is only partially wired."""

    if "notes.link" not in advertised_domains:
        return
    if not adapters.has_domain("notes.link") or not isinstance(
        adapters.get("notes.link"), NotesLinkDomainAdapter
    ):
        raise RuntimeError("Advertised Sync domain has no strict adapter: notes.link")
    if not isinstance(materializers.get("notes.link"), NotesLinkMaterializer):
        raise RuntimeError(
            "Advertised Sync domain has no user-bound materializer: notes.link"
        )


def _validate_notes_task_components(
    *,
    adapters: SyncAdapterRegistry,
    materializers: Mapping[SyncDomain, SyncMaterializer],
) -> None:
    """Fail closed when the private dormant task lifecycle is partially wired."""

    if not isinstance(adapters.get("notes.task"), NotesTaskDomainAdapter):
        raise RuntimeError("Private Sync domain has no strict adapter: notes.task")
    if not isinstance(materializers.get("notes.task"), NotesTaskMaterializer):
        raise RuntimeError(
            "Private Sync domain has no user-bound materializer: notes.task"
        )
    if not isinstance(
        adapters.get("notes.task_activity"), NotesTaskActivityDomainAdapter
    ):
        raise RuntimeError(
            "Private Sync domain has no strict adapter: notes.task_activity"
        )
    if not isinstance(
        materializers.get("notes.task_activity"), NotesTaskActivityMaterializer
    ):
        raise RuntimeError(
            "Private Sync domain has no user-bound materializer: notes.task_activity"
        )


def sync_v2_storage_exists_for_user(user_id: str) -> bool:
    """Return whether durable Sync v2 storage already exists for a user."""

    database_url = os.getenv("SYNC_V2_DATABASE_URL", "").strip()
    sqlite_path = os.getenv("SYNC_V2_SQLITE_PATH", "").strip()
    default_path = _default_sync_v2_path_for_user(user_id)
    if database_url:
        return _sync_v2_database_url_exists(database_url, default_path)
    if sqlite_path:
        return _sync_v2_sqlite_path_exists(sqlite_path)
    # lgtm[py/path-injection]: default_path is built from a normalized Sync v2 user ID.
    return default_path.exists()


@lru_cache(maxsize=256)
def _sync_v2_store_for_user(
    user_id: str,
    database_url: str,
    sqlite_path: str,
) -> SyncV2Store:
    del database_url, sqlite_path
    return SyncV2Store(SyncDatabase(user_id=user_id))


@lru_cache(maxsize=256)
def _chacha_notes_db_for_user(user_id: str) -> CharactersRAGDB:
    db_path = DatabasePaths.get_chacha_db_path(user_id)
    # lgtm[py/path-injection]: db_path is resolved through DatabasePaths user storage normalization.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return CharactersRAGDB(db_path=str(db_path), client_id=str(user_id))


@lru_cache(maxsize=256)
def _sync_v2_blob_store_for_user(user_id: str) -> LocalSyncBlobStore:
    del user_id
    configured_path = os.getenv("SYNC_V2_BLOB_STORE_PATH", "").strip()
    if configured_path:
        return LocalSyncBlobStore(configured_path)
    base_dir = DatabasePaths.resolve_user_db_base_dir()
    return LocalSyncBlobStore(base_dir / "_sync_v2_blobs")


def _sync_v2_settings_from_env() -> SyncV2Settings:
    return SyncV2Settings(
        supports_attachments=_sync_v2_bool_env("SYNC_V2_ENABLE_BLOB_TRANSFER", default=False),
        max_blob_bytes=_sync_v2_optional_positive_int_env("SYNC_V2_MAX_BLOB_BYTES"),
        max_chunk_bytes=_sync_v2_positive_int_env("SYNC_V2_MAX_CHUNK_BYTES", default=4_194_304),
        max_active_blob_uploads=_sync_v2_positive_int_env("SYNC_V2_MAX_ACTIVE_BLOB_UPLOADS", default=8),
        user_blob_quota_bytes=_sync_v2_optional_positive_int_env("SYNC_V2_USER_BLOB_QUOTA_BYTES"),
        server_trusted_encryption=server_trusted_encryption_status_from_env(),
        personal_context=personal_context_sync_capabilities_from_env(),
    )


def _sync_v2_bool_env(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sync_v2_positive_int_env(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return _sync_v2_parse_positive_int_env(name, value)


def _sync_v2_optional_positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return _sync_v2_parse_positive_int_env(name, value)


def _sync_v2_parse_positive_int_env(name: str, value: str) -> int:
    try:
        parsed = int(value.strip())
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _workspace_access_checker(user_id: str, workspace_id: str, permission: str) -> bool:
    if permission != "sync":
        return False
    return _chacha_notes_db_for_user(user_id).get_workspace(workspace_id) is not None


def _default_sync_v2_path_for_user(user_id: str) -> Path:
    base_dir = DatabasePaths.resolve_user_db_base_dir()
    safe_user_id = _resolve_user_id_for_storage(user_id)
    user_dir = safe_join(
        str(base_dir),
        safe_user_id,
        error_factory=lambda _exc: ValueError("Sync v2 user path escapes base directory"),
    )
    if user_dir is None:
        raise ValueError("Sync v2 user path escapes base directory")
    db_path = safe_join(
        user_dir,
        SYNC_DB_FILENAME,
        error_factory=lambda _exc: ValueError("Sync v2 DB path escapes user directory"),
    )
    if db_path is None:
        raise ValueError("Sync v2 DB path escapes user directory")
    return Path(db_path)


def _sync_v2_database_url_exists(database_url: str, default_path: Path) -> bool:
    parsed = urlparse(database_url)
    scheme = (parsed.scheme or "").lower().split("+", 1)[0]
    if scheme in {"postgres", "postgresql"}:
        return True
    if scheme in {"sqlite", "file", ""}:
        # lgtm[py/path-injection]: SQLite URL paths are resolved under default_path.parent unless absolute/admin configured.
        return _sync_v2_sqlite_url_path(database_url, default_path).exists()
    return True


def _sync_v2_sqlite_path_exists(sqlite_path: str) -> bool:
    if sqlite_path == ":memory:":
        return True
    # lgtm[py/path-injection] SYNC_V2_SQLITE_PATH is an administrator-controlled configuration path.
    return Path(sqlite_path).expanduser().exists()


def _sync_v2_sqlite_url_path(database_url: str, default_path: Path) -> Path:
    parsed = urlparse(database_url)
    raw_path = parsed.path or ""
    if raw_path in {"/:memory:", ":memory:"}:
        return Path(":memory:")
    if raw_path.startswith("/./"):
        raw_path = raw_path[1:]
    if raw_path.startswith("/") and raw_path != "/:memory:":
        return Path(raw_path)
    resolved = safe_join(
        str(default_path.parent),
        raw_path or default_path.name,
        error_factory=lambda _exc: ValueError("Sync v2 SQLite URL path escapes default directory"),
    )
    if resolved is None:
        raise ValueError("Sync v2 SQLite URL path escapes default directory")
    return Path(resolved)


__all__ = [
    "default_sync_v2_registry",
    "sync_v2_service_for_user",
    "sync_v2_storage_exists_for_user",
]
