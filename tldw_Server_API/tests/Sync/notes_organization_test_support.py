"""Focused real-store support for Notes organization capture regressions."""

from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import NotesOrganizationDomainAdapter
from tldw_Server_API.app.core.Sync.v2.materializers.chat import ChatConversationMaterializer
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import NotesOrganizationMaterializer
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, NOTES_ORGANIZATION_DOMAINS
from tldw_Server_API.app.core.Sync.v2.notes_organization_bootstrap import NotesOrganizationBootstrapper
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def build_ready_notes_sync_stack(
    tmp_path: Path,
    *,
    user_id: str = "user-1",
) -> tuple[CharactersRAGDB, SyncV2Store, SyncV2Service]:
    """Build a ready personal Notes Sync stack backed by real SQLite stores."""

    notes_db = CharactersRAGDB(db_path=str(tmp_path / "notes.sqlite"), client_id=user_id)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    adapters = [
        NotesDomainAdapter()
        if domain == "notes.note"
        else StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
        for domain in M1_SYNC_DOMAINS
    ]
    adapters.extend(NotesOrganizationDomainAdapter(domain=domain) for domain in NOTES_ORGANIZATION_DOMAINS)
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(adapters),
        materializers={
            "chat.conversation": ChatConversationMaterializer(notes_db),
            "notes.note": NotesMaterializer(notes_db),
            **{domain: NotesOrganizationMaterializer(notes_db, domain) for domain in NOTES_ORGANIZATION_DOMAINS},
        },
        dataset_bootstrapper=NotesOrganizationBootstrapper(notes_db),
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
        clock=lambda: "2026-08-09T08:00:00+00:00",
        id_factory=lambda prefix: f"{prefix}-test",
    )
    profile = service.bootstrap_profile(
        user_id=user_id,
        mode="server_frontend",
        device_id="frontend-device",
        requested_domains=[*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS],
    )
    assert profile.active_dataset_id is not None
    return notes_db, sync_store, service
