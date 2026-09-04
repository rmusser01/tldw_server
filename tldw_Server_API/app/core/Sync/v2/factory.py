from __future__ import annotations

"""Sync v2 service composition helpers shared by HTTP and non-HTTP entrypoints."""

import hashlib
import hmac
import os
from base64 import urlsafe_b64encode
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    _resolve_user_id_for_storage,
)
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SYNC_DB_FILENAME, SyncDatabase
from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_existing_companion_storage_user_id,
)
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PersonalContextPublicationRelayStore,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
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
from .domain_adapters.personal_context import PersonalContextDomainAdapter
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
    PersonalContextMaterializer,
    SourceCacheMaterializer,
    SyncMaterializer,
)
from .models import (
    M1_SYNC_DOMAINS,
    MEDIA_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SOURCE_CACHE_SYNC_DOMAINS,
    WORKSPACE_SYNC_DOMAINS,
    SyncDomain,
)
from .notes_attachment_bootstrap import NotesAttachmentBootstrapper
from .notes_link_bootstrap import NotesLinkBootstrapper
from .notes_organization_bootstrap import NotesOrganizationBootstrapper
from .notes_task_activity_bootstrap import NotesTaskActivityBootstrapper
from .notes_task_bootstrap import NotesTaskBootstrapper
from .personal_context_relay import PersonalContextRelay
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
        + [
            PersonalContextDomainAdapter(
                domain=domain,
                integrity_key_resolver=_personal_context_integrity_key,
                encryption_key_resolver=_personal_context_encryption_key,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        ]
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
        **{
            domain: PersonalContextMaterializer(
                domain=domain,
                service_resolver=_personal_context_service_for_user,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
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
    _validate_personal_context_components(
        adapters=adapters,
        materializers=materializers,
    )
    service = SyncV2Service(
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
        personal_context_service_resolver=_personal_context_service_for_user,
        personal_context_key_wrapper=_wrap_personal_context_integrity_key,
        personal_context_key_fingerprint=_personal_context_wrapping_key_fingerprint,
        personal_context_authority_id=os.getenv(
            "SYNC_V2_PERSONAL_CONTEXT_AUTHORITY_ID", "tldw-server"
        ),
    )
    publication_service = _personal_context_service_for_user(user_id)
    service.personal_context_relay = PersonalContextRelay(
        publications=PersonalContextPublicationRelayStore(
            publication_service._repository.database
        ),
        stage_authority=service.stage_personal_context_authority,
        finalize_authority=service.finalize_personal_context_authority,
        cancel_authority=service.cancel_personal_context_authority,
    )
    def relay_after_commit(profile_id: str) -> None:
        """Best-effort recovery hook; durable journal debt survives all failures."""

        for dataset in store.list_datasets_for_user(user_id):
            state = dataset.metadata.get("personal_context")
            if not isinstance(state, Mapping) or state.get("profile_id") != profile_id:
                continue
            service.personal_context_relay.relay_profile(
                user_id=user_id,
                profile_id=profile_id,
                dataset_id=dataset.dataset_id,
                after_server_cursor=None,
            )
            return

    publication_service.set_after_commit_relay(relay_after_commit)

    def purge_cleanup_after_commit(intent: object) -> None:
        """Scrub all Sync datasets bound to an already-authorized direct purge."""

        for dataset in store.list_datasets_for_user(user_id, include_archived=True):
            state = dataset.metadata.get("personal_context")
            if not isinstance(state, Mapping) or state.get("profile_id") != getattr(
                intent, "profile_id", None
            ):
                continue
            service.shred_authorized_personal_context_history(
                intent,
                user_id=user_id,
                dataset_id=dataset.dataset_id,
            )

    publication_service.set_after_commit_purge_cleanup(purge_cleanup_after_commit)
    return service


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


def _validate_personal_context_components(
    *,
    adapters: SyncAdapterRegistry,
    materializers: Mapping[SyncDomain, SyncMaterializer],
) -> None:
    """Fail closed when a Personal Context domain is only partially wired."""

    for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
        adapter = adapters.get(domain)
        materializer = materializers.get(domain)
        if not isinstance(adapter, PersonalContextDomainAdapter):
            raise RuntimeError(
                f"Personal Context Sync domain has no strict adapter: {domain}"
            )
        if (
            not isinstance(materializer, PersonalContextMaterializer)
            or materializer.domain != domain
        ):
            raise RuntimeError(
                f"Personal Context Sync domain has no service materializer: {domain}"
            )


def _wrap_personal_context_integrity_key(
    *, device: object, integrity_key: bytes, integrity_key_id: str
) -> str:
    """Encrypt the integrity key to the registered device RSA public key."""

    capabilities = getattr(device, "capabilities", {})
    public_key_pem = (
        capabilities.get("personal_context_wrapping_public_key")
        if isinstance(capabilities, Mapping)
        else None
    )
    if not isinstance(public_key_pem, str) or not public_key_pem.strip():
        raise ValueError("personal_context_device_key_unavailable")
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    if not isinstance(public_key, rsa.RSAPublicKey) or public_key.key_size < 2048:
        raise ValueError("personal_context_device_key_invalid")
    label = f"personal-context:{integrity_key_id}".encode()
    ciphertext = public_key.encrypt(
        integrity_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=label,
        ),
    )
    return "rsa-oaep-sha256:" + urlsafe_b64encode(ciphertext).decode("ascii")


def _personal_context_wrapping_key_fingerprint(*, device: object) -> str:
    """Return the SHA-256 fingerprint of the registered device wrapping key."""

    capabilities = getattr(device, "capabilities", {})
    value = capabilities.get("personal_context_wrapping_public_key") if isinstance(capabilities, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("personal_context_device_key_unavailable")
    public_key = serialization.load_pem_public_key(value.encode("utf-8"))
    encoded = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(encoded).hexdigest()


def _personal_context_integrity_key(dataset: object, key_id: str) -> bytes:
    """Resolve the enrolled profile's actual canonical integrity key."""

    service, profile_id = _personal_context_key_service(dataset)
    actual_key_id, key = service.sync_integrity_key(profile_id)
    if not hmac.compare_digest(actual_key_id, key_id):
        raise RuntimeError("Personal Context integrity key is unavailable")
    return key


def _personal_context_encryption_key(dataset: object) -> tuple[bytes, int]:
    """Resolve the enrolled profile's actual canonical encryption key."""

    service, profile_id = _personal_context_key_service(dataset)
    return service.sync_encryption_key(profile_id)


def _personal_context_key_service(
    dataset: object,
) -> tuple[PersonalContextService, str]:
    owner_user_id = str(getattr(dataset, "owner_user_id", "")).strip()
    metadata = getattr(dataset, "metadata", {})
    state = metadata.get("personal_context") if isinstance(metadata, Mapping) else None
    profile_id = state.get("profile_id") if isinstance(state, Mapping) else None
    if not owner_user_id or not isinstance(profile_id, str) or not profile_id:
        raise RuntimeError("Personal Context key custody is unavailable")
    return _personal_context_service_for_user(owner_user_id), profile_id


@lru_cache(maxsize=256)
def _personal_context_service_for_user(user_id: str) -> PersonalContextService:
    """Return the canonical service bound to one authenticated Sync owner."""

    storage_user_id = resolve_existing_companion_storage_user_id(user_id)
    database = PersonalizationDB.for_user(storage_user_id)
    return PersonalContextService(PersonalContextRepository(database))


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
