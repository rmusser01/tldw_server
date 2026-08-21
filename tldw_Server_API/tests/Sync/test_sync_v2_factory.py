from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Sync.v2 import factory
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import (
    NotesLinkDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
    NotesOrganizationDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.factory import _sync_v2_settings_from_env
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import NotesLinkMaterializer
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import NOTES_ORGANIZATION_DOMAINS


def test_sync_v2_factory_keeps_blob_transfer_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SYNC_V2_ENABLE_BLOB_TRANSFER", raising=False)

    settings = _sync_v2_settings_from_env()

    assert settings.supports_attachments is False


def test_sync_v2_factory_enables_blob_transfer_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SYNC_V2_ENABLE_BLOB_TRANSFER", "true")
    monkeypatch.setenv("SYNC_V2_MAX_BLOB_BYTES", "4096")
    monkeypatch.setenv("SYNC_V2_MAX_CHUNK_BYTES", "1024")
    monkeypatch.setenv("SYNC_V2_MAX_ACTIVE_BLOB_UPLOADS", "3")
    monkeypatch.setenv("SYNC_V2_USER_BLOB_QUOTA_BYTES", "8192")

    settings = _sync_v2_settings_from_env()

    assert settings.supports_attachments is True
    assert settings.max_blob_bytes == 4096
    assert settings.max_chunk_bytes == 1024
    assert settings.max_active_blob_uploads == 3
    assert settings.user_blob_quota_bytes == 8192


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SYNC_V2_MAX_BLOB_BYTES", "not-a-number"),
        ("SYNC_V2_MAX_BLOB_BYTES", "0"),
        ("SYNC_V2_MAX_CHUNK_BYTES", "not-a-number"),
        ("SYNC_V2_MAX_CHUNK_BYTES", "0"),
        ("SYNC_V2_MAX_ACTIVE_BLOB_UPLOADS", "not-a-number"),
        ("SYNC_V2_MAX_ACTIVE_BLOB_UPLOADS", "0"),
        ("SYNC_V2_USER_BLOB_QUOTA_BYTES", "not-a-number"),
        ("SYNC_V2_USER_BLOB_QUOTA_BYTES", "0"),
    ],
)
def test_sync_v2_factory_rejects_invalid_positive_integer_env(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        _sync_v2_settings_from_env()


def test_sync_v2_factory_registers_each_notes_organization_domain_as_a_complete_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db = SimpleNamespace(task_store=object())
    monkeypatch.setattr(factory, "_sync_v2_store_for_user", lambda *args: object())
    monkeypatch.setattr(factory, "_chacha_notes_db_for_user", lambda user_id: note_db)
    monkeypatch.setattr(factory, "_sync_v2_blob_store_for_user", lambda user_id: None)

    service = factory.sync_v2_service_for_user("owner-1")

    for domain in NOTES_ORGANIZATION_DOMAINS:
        adapter = service.adapters.get(domain)
        materializer = service.materializers[domain]
        assert isinstance(adapter, NotesOrganizationDomainAdapter)
        assert isinstance(materializer, NotesOrganizationMaterializer)
        assert materializer.domain == domain
        assert materializer.note_db is note_db


def test_sync_v2_factory_fails_closed_when_an_advertised_notes_organization_pair_is_incomplete() -> None:
    domain = NOTES_ORGANIZATION_DOMAINS[0]
    adapters = SyncAdapterRegistry([NotesOrganizationDomainAdapter(domain=domain)])

    with pytest.raises(RuntimeError, match="materializer"):
        factory._validate_notes_organization_components(
            adapters=adapters,
            materializers={},
            advertised_domains=[domain],
        )

    with pytest.raises(RuntimeError, match="adapter"):
        factory._validate_notes_organization_components(
            adapters=SyncAdapterRegistry(),
            materializers={domain: NotesOrganizationMaterializer(object(), domain)},
            advertised_domains=[domain],
        )

    with pytest.raises(RuntimeError, match="strict adapter"):
        factory._validate_notes_organization_components(
            adapters=SyncAdapterRegistry([StaticSyncAdapter(domain=domain)]),
            materializers={domain: NotesOrganizationMaterializer(object(), domain)},
            advertised_domains=[domain],
        )


def test_sync_v2_factory_registers_and_validates_notes_link_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db = SimpleNamespace(task_store=object())
    monkeypatch.setattr(factory, "_sync_v2_store_for_user", lambda *args: object())
    monkeypatch.setattr(factory, "_chacha_notes_db_for_user", lambda user_id: note_db)
    monkeypatch.setattr(factory, "_sync_v2_blob_store_for_user", lambda user_id: None)

    service = factory.sync_v2_service_for_user("owner-1")

    assert isinstance(service.adapters.get("notes.link"), NotesLinkDomainAdapter)
    materializer = service.materializers["notes.link"]
    assert isinstance(materializer, NotesLinkMaterializer)
    assert materializer.note_db is note_db


def test_sync_v2_factory_fails_closed_for_incomplete_notes_link_pair() -> None:
    adapter = NotesLinkDomainAdapter()
    materializer = NotesLinkMaterializer(object())

    with pytest.raises(RuntimeError, match="materializer"):
        factory._validate_notes_link_components(
            adapters=SyncAdapterRegistry([adapter]),
            materializers={},
            advertised_domains=["notes.link"],
        )
    with pytest.raises(RuntimeError, match="adapter"):
        factory._validate_notes_link_components(
            adapters=SyncAdapterRegistry(),
            materializers={"notes.link": materializer},
            advertised_domains=["notes.link"],
        )
    with pytest.raises(RuntimeError, match="strict adapter"):
        factory._validate_notes_link_components(
            adapters=SyncAdapterRegistry([StaticSyncAdapter(domain="notes.link")]),
            materializers={"notes.link": materializer},
            advertised_domains=["notes.link"],
        )
