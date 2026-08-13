"""Boundary contracts for canonical Notes attachment Sync APIs."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.api.v1.schemas.notes_attachments import (
    NotesAttachmentFromUploadRequest,
    NotesAttachmentItem,
    NotesAttachmentMutationResponse,
    NotesAttachmentPage,
    NotesAttachmentReasonRequest,
    NotesAttachmentRenameRequest,
    format_notes_attachment_etag,
    parse_notes_attachment_if_match,
    validate_notes_attachment_idempotency_key,
    validate_notes_attachment_keyset_cursor,
)
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    SyncBlobUploadCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.domain_adapters.attachment_refs import (
    AttachmentRefDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers import AttachmentRefMaterializer
from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

ATTACHMENT_ID = "2c4cb609-c4db-44f9-8e35-f078bd36d6b2"
NOTE_ID = "a1677eb1-1f41-4c86-a8dd-1eaa14b014e2"
OBJECT_HASH = "sha256:" + "a" * 64
BLOB_HASH = "sha256:" + "b" * 64
OWNER = "user-1"
DATASET_ID = "dataset-1"


class _NoopRateLimiter:
    async def check_user_rate_limit(
        self,
        user_id: int,
        endpoint: str,
        role: str = "user",
    ) -> tuple[bool, dict[str, object]]:
        return True, {}


@pytest.fixture
def canonical_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, CharactersRAGDB, SyncV2Service]:
    note_db = CharactersRAGDB(str(tmp_path / "notes.sqlite"), client_id=OWNER)
    note_db.add_note("Parent", "Body", note_id=NOTE_ID)
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id=OWNER,
            domains=["notes.note", "attachment.ref"],
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_attachment_v2": {"state": "ready"},
            },
        )
    )
    service = SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry(
            [AttachmentRefDomainAdapter(v2_writes_enabled=True)]
        ),
        materializers={"attachment.ref": AttachmentRefMaterializer(note_db)},
        blob_store=LocalSyncBlobStore(tmp_path / "blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=1024 * 1024,
            max_chunk_bytes=64 * 1024,
            pull_token_signing_secret="attachment-api-tests",
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
        clock=lambda: "2026-08-11T20:30:00+00:00",
    )
    app = FastAPI()
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    async def _db_override() -> CharactersRAGDB:
        return note_db

    async def _user_override() -> User:
        return User(id=OWNER, username=OWNER, is_admin=True)

    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[notes_endpoint.get_request_user] = _user_override
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = (
        lambda: _NoopRateLimiter()
    )
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: service,
    )
    yield TestClient(app), note_db, service
    note_db.close_all_connections()


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _complete_create_upload(service: SyncV2Service, payload: bytes) -> str:
    session = service.create_blob_upload_session(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        device_id=None,
        domain="attachment.ref",
        entity_id=ATTACHMENT_ID,
        attachment_id=ATTACHMENT_ID,
        content_type="application/pdf",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=len(payload),
        chunk_count=1,
        idempotency_key="upload-1",
        metadata={
            "notes_attachment_intent": {
                "intent": "create",
                "note_id": NOTE_ID,
                "attachment_id": ATTACHMENT_ID,
                "file_name": "Report.pdf",
            }
        },
    )
    service.upload_blob_chunk(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )
    service.complete_blob_upload(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        upload_id=session.upload_id,
    )
    return session.upload_id


def _complete_replace_upload(service: SyncV2Service, payload: bytes) -> str:
    head = service.store.get_current_head(
        DATASET_ID,
        "attachment.ref",
        ATTACHMENT_ID,
    )
    assert head is not None
    session = service.create_blob_upload_session(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        device_id=None,
        domain="attachment.ref",
        entity_id=ATTACHMENT_ID,
        attachment_id=ATTACHMENT_ID,
        content_type="application/pdf",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=len(payload),
        chunk_count=1,
        idempotency_key="upload-replace-1",
        metadata={
            "notes_attachment_intent": {
                "intent": "replace",
                "note_id": NOTE_ID,
                "attachment_id": ATTACHMENT_ID,
                "base_server_cursor": head.server_cursor,
                "base_object_revision": head.object_revision,
                "base_object_hash": head.payload_hash,
            }
        },
    )
    service.upload_blob_chunk(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )
    service.complete_blob_upload(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        upload_id=session.upload_id,
    )
    return session.upload_id


def _item_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": "dataset-1",
        "note_id": NOTE_ID,
        "attachment_id": ATTACHMENT_ID,
        "file_name": "report.pdf",
        "original_file_name": "report.pdf",
        "content_type": "application/pdf",
        "size_bytes": 42,
        "blob_hash": BLOB_HASH,
        "version": 3,
        "object_hash": OBJECT_HASH,
        "state": "live",
        "deleted_at": None,
        "delete_reason": None,
        "created_at": "2026-08-11T12:00:00.000000Z",
        "last_modified": "2026-08-11T12:01:00.000000Z",
        "created_by": "device-1",
        "source_kind": "upload",
        "availability": "available",
        "etag": format_notes_attachment_etag(ATTACHMENT_ID, 3, OBJECT_HASH),
    }
    payload.update(overrides)
    return payload


@pytest.mark.integration
def test_notes_attachment_schema_accepts_canonical_item_page_and_mutation_response():
    item = NotesAttachmentItem.model_validate(_item_payload())
    page = NotesAttachmentPage.model_validate(
        {"items": [item.model_dump()], "next_cursor": "cursor-1", "has_more": True}
    )
    mutation = NotesAttachmentMutationResponse.model_validate(
        {**item.model_dump(), "idempotent_replay": False}
    )

    assert item.etag == f'"att-{ATTACHMENT_ID}-v3-{OBJECT_HASH.removeprefix("sha256:")}"'
    assert page.items == [item]
    assert mutation.idempotent_replay is False


@pytest.mark.parametrize(
    ("payload", "model"),
    [
        ({"file_name": "renamed.pdf", "content_type": "application/pdf"}, NotesAttachmentRenameRequest),
        ({"upload_id": "upload-1", "attachment_id": ATTACHMENT_ID}, NotesAttachmentFromUploadRequest),
        ({"reason": "ok", "extra": True}, NotesAttachmentReasonRequest),
        ({**_item_payload(), "storage_key": "secret"}, NotesAttachmentItem),
    ],
)
@pytest.mark.integration
def test_notes_attachment_schema_rejects_extra_fields(payload: dict[str, object], model: type):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.integration
def test_notes_attachment_schema_enforces_rename_only_and_reason_bounds():
    rename = NotesAttachmentRenameRequest.model_validate({"file_name": " Quarterly Report.PDF "})
    assert rename.file_name == "Quarterly_Report.pdf"

    with pytest.raises(ValidationError):
        NotesAttachmentReasonRequest.model_validate({"reason": "x" * 257})

    with pytest.raises(ValidationError):
        NotesAttachmentRenameRequest.model_validate({"file_name": "payload.exe"})


@pytest.mark.integration
def test_notes_attachment_schema_validates_from_upload_identifier():
    request = NotesAttachmentFromUploadRequest.model_validate({"upload_id": "upload-1"})
    assert request.upload_id == "upload-1"

    for upload_id in ("", "x" * 129, "bad\nvalue", "café"):
        with pytest.raises(ValidationError):
            NotesAttachmentFromUploadRequest.model_validate({"upload_id": upload_id})


@pytest.mark.integration
def test_attachment_blob_upload_request_requires_one_strict_intent() -> None:
    base = {
        "dataset_id": "dataset-1",
        "domain": "attachment.ref",
        "object_id": ATTACHMENT_ID,
        "attachment_id": ATTACHMENT_ID,
        "content_type": "application/pdf",
        "size_bytes": 42,
        "payload_hash": BLOB_HASH,
        "chunk_size": 42,
        "chunk_count": 1,
    }
    request = SyncBlobUploadCreateRequest.model_validate(
        {
            **base,
            "metadata": {
                "notes_attachment_intent": {
                    "intent": "create",
                    "note_id": NOTE_ID,
                    "attachment_id": ATTACHMENT_ID,
                    "file_name": " Report.PDF ",
                }
            },
        }
    )

    assert request.metadata["notes_attachment_intent"]["file_name"] == "Report.pdf"

    for metadata in (
        {},
        {"notes_attachment_intent": {"intent": "create"}},
        {
            "notes_attachment_intent": {
                "intent": "create",
                "note_id": NOTE_ID,
                "attachment_id": ATTACHMENT_ID,
                "file_name": "report.pdf",
                "extra": True,
            }
        },
    ):
        with pytest.raises(ValidationError):
            SyncBlobUploadCreateRequest.model_validate({**base, "metadata": metadata})

    generic = SyncBlobUploadCreateRequest.model_validate(
        {
            **base,
            "domain": "notes.note",
            "object_id": NOTE_ID,
            "metadata": {},
        }
    )
    assert generic.metadata == {}


@pytest.mark.integration
def test_notes_attachment_etag_grammar_is_exact():
    etag = format_notes_attachment_etag(ATTACHMENT_ID, 3, OBJECT_HASH)
    parsed = parse_notes_attachment_if_match(etag)
    assert parsed == (ATTACHMENT_ID, 3, OBJECT_HASH)

    for value in (
        None,
        "*",
        f'W/{etag}',
        f"{etag}, {etag}",
        f'"att-{ATTACHMENT_ID}-v0-{OBJECT_HASH.removeprefix("sha256:")}"',
        f'"att-{ATTACHMENT_ID.upper()}-v3-{OBJECT_HASH.removeprefix("sha256:")}"',
        f'"att-{ATTACHMENT_ID}-v3-{OBJECT_HASH.removeprefix("sha256:").upper()}"',
    ):
        with pytest.raises(ValueError):
            parse_notes_attachment_if_match(value)


@pytest.mark.integration
def test_notes_attachment_idempotency_key_and_keyset_cursor_boundaries():
    assert validate_notes_attachment_idempotency_key("request-1") == "request-1"
    assert validate_notes_attachment_keyset_cursor("opaque.cursor") == "opaque.cursor"

    for value in ("", "x" * 129, "bad\nkey", "café"):
        with pytest.raises(ValueError):
            validate_notes_attachment_idempotency_key(value)
    for value in ("", "x" * 513, "bad\ncursor", "café"):
        with pytest.raises(ValueError):
            validate_notes_attachment_keyset_cursor(value)


@pytest.mark.integration
def test_notes_attachment_item_rejects_inconsistent_lifecycle_and_etag():
    with pytest.raises(ValidationError):
        NotesAttachmentItem.model_validate(_item_payload(state="tombstoned"))
    with pytest.raises(ValidationError):
        NotesAttachmentItem.model_validate(_item_payload(etag=f'"att-{ATTACHMENT_ID}-v2-{OBJECT_HASH}"'))


@pytest.mark.parametrize(
    "overrides",
    [
        {"size_bytes": "42"},
        {"size_bytes": True},
        {"version": "3"},
        {"dataset_id": " dataset-1"},
        {"created_by": " device-1"},
    ],
)
@pytest.mark.integration
def test_notes_attachment_item_rejects_coerced_or_unnormalized_authority_fields(overrides):
    with pytest.raises(ValidationError):
        NotesAttachmentItem.model_validate(_item_payload(**overrides))


@pytest.mark.integration
def test_canonical_static_route_precedes_legacy_filename_and_lists_keyset_page(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, _, _ = canonical_api

    response = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID, "limit": 1},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"items": [], "next_cursor": None, "has_more": False}


@pytest.mark.integration
def test_from_upload_requires_headers_and_supports_exact_lifecycle_replay(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, _, service = canonical_api
    upload_id = _complete_create_upload(service, b"attachment payload")
    path = f"/api/v1/notes/{NOTE_ID}/attachments/from-upload"

    missing_key = client.post(
        path,
        params={"dataset_id": DATASET_ID},
        json={"upload_id": upload_id},
    )
    assert missing_key.status_code == 428, missing_key.text

    created = client.post(
        path,
        params={"dataset_id": DATASET_ID},
        headers={"Idempotency-Key": "attach-upload-1"},
        json={"upload_id": upload_id},
    )
    replay = client.post(
        path,
        params={"dataset_id": DATASET_ID},
        headers={"Idempotency-Key": "attach-upload-1"},
        json={"upload_id": upload_id},
    )

    assert created.status_code == replay.status_code == 201, created.text
    assert created.json()["attachment_id"] == ATTACHMENT_ID
    assert created.json()["file_name"] == "Report.pdf"
    assert created.json()["availability"] == "available"
    assert created.json()["source_kind"] == "upload"
    assert created.json()["idempotent_replay"] is False
    assert replay.json()["idempotent_replay"] is True

    detail = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
    )
    assert detail.status_code == 200, detail.text
    assert detail.headers["etag"] == created.json()["etag"]

    missing_match = client.patch(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={"Idempotency-Key": "rename-1"},
        json={"file_name": "Renamed.pdf"},
    )
    assert missing_match.status_code == 428, missing_match.text

    stale = client.patch(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "rename-1",
            "If-Match": format_notes_attachment_etag(
                ATTACHMENT_ID,
                99,
                created.json()["object_hash"],
            ),
        },
        json={"file_name": "Renamed.pdf"},
    )
    assert stale.status_code == 409, stale.text

    renamed = client.patch(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "rename-1",
            "If-Match": created.json()["etag"],
        },
        json={"file_name": "Renamed.pdf"},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["file_name"] == "Renamed.pdf"
    rename_replay = client.patch(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "rename-1",
            "If-Match": created.json()["etag"],
        },
        json={"file_name": "Renamed.pdf"},
    )
    assert rename_replay.status_code == 200, rename_replay.text
    assert rename_replay.json()["idempotent_replay"] is True
    assert rename_replay.json()["etag"] == renamed.json()["etag"]

    replacement_upload_id = _complete_replace_upload(service, b"new payload")
    replaced = client.post(
        path,
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "replace-1",
            "If-Match": renamed.json()["etag"],
        },
        json={"upload_id": replacement_upload_id},
    )
    replace_replay = client.post(
        path,
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "replace-1",
            "If-Match": renamed.json()["etag"],
        },
        json={"upload_id": replacement_upload_id},
    )
    assert replaced.status_code == 201, replaced.text
    assert replaced.json()["file_name"] == "Renamed.pdf"
    assert replaced.json()["blob_hash"] == _sha256(b"new payload")
    assert replace_replay.status_code == 201, replace_replay.text
    assert replace_replay.json()["idempotent_replay"] is True
    assert replace_replay.json()["etag"] == replaced.json()["etag"]

    deleted = client.request(
        "DELETE",
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "delete-1",
            "If-Match": replaced.json()["etag"],
        },
        json={"reason": "removed"},
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["state"] == "tombstoned"
    delete_replay = client.request(
        "DELETE",
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "delete-1",
            "If-Match": replaced.json()["etag"],
        },
        json={"reason": "removed"},
    )
    assert delete_replay.status_code == 200, delete_replay.text
    assert delete_replay.json()["idempotent_replay"] is True
    assert delete_replay.json()["etag"] == deleted.json()["etag"]

    hidden_content = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}/content",
        params={"dataset_id": DATASET_ID},
    )
    assert hidden_content.status_code == 404

    restored = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}/restore",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "restore-1",
            "If-Match": deleted.json()["etag"],
        },
        json={"reason": "undo"},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["state"] == "live"
    restore_replay = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}/restore",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "restore-1",
            "If-Match": deleted.json()["etag"],
        },
        json={"reason": "undo"},
    )
    assert restore_replay.status_code == 200, restore_replay.text
    assert restore_replay.json()["idempotent_replay"] is True
    assert restore_replay.json()["etag"] == restored.json()["etag"]


@pytest.mark.integration
def test_canonical_content_supports_conditionals_and_single_byte_ranges(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, _, service = canonical_api
    payload = b"0123456789"
    upload_id = _complete_create_upload(service, payload)
    created = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments/from-upload",
        params={"dataset_id": DATASET_ID},
        headers={"Idempotency-Key": "attach-content-1"},
        json={"upload_id": upload_id},
    ).json()
    path = f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}/content"

    full = client.get(path, params={"dataset_id": DATASET_ID})
    partial = client.get(
        path,
        params={"dataset_id": DATASET_ID},
        headers={"Range": "bytes=2-5", "If-Range": created["etag"]},
    )
    stale_if_range = client.get(
        path,
        params={"dataset_id": DATASET_ID},
        headers={
            "Range": "bytes=2-5",
            "If-Range": format_notes_attachment_etag(
                ATTACHMENT_ID,
                99,
                created["object_hash"],
            ),
        },
    )
    not_modified = client.get(
        path,
        params={"dataset_id": DATASET_ID},
        headers={"If-None-Match": created["etag"], "Range": "bytes=0-1"},
    )
    unsatisfied = client.get(
        path,
        params={"dataset_id": DATASET_ID},
        headers={"Range": "bytes=99-100"},
    )
    malformed = client.get(
        path,
        params={"dataset_id": DATASET_ID},
        headers={"Range": "bytes=0-1,3-4"},
    )
    service.store.db.execute(
        """
        UPDATE sync_attachment_revision_bindings
           SET resolved_blob_id = NULL
         WHERE dataset_id = ? AND attachment_id = ? AND attachment_revision = ?
        """,
        (DATASET_ID, ATTACHMENT_ID, created["version"]),
    )
    unbound = client.get(path, params={"dataset_id": DATASET_ID})

    assert full.status_code == 200 and full.content == payload
    assert full.headers["accept-ranges"] == "bytes"
    assert partial.status_code == 206 and partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert partial.headers["content-length"] == "4"
    assert stale_if_range.status_code == 200 and stale_if_range.content == payload
    assert not_modified.status_code == 304 and not not_modified.content
    assert unsatisfied.status_code == 416
    assert unsatisfied.headers["content-range"] == "bytes */10"
    assert malformed.status_code == 400
    assert unbound.status_code == 404


@pytest.mark.integration
def test_active_legacy_filename_routes_are_canonical_compatibility_aliases(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, note_db, _ = canonical_api
    monkeypatch.setattr(
        notes_endpoint.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: tmp_path / "legacy-files-must-stay-empty",
    )

    created = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        headers={"Idempotency-Key": "legacy-create-1"},
        files={
            "file": (
                "Legacy Report.PDF",
                b"%PDF-1.7\nlegacy payload\n%%EOF\n",
                "application/pdf",
            )
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["file_name"] == "Legacy_Report.pdf"
    registry = note_db.note_attachment_store.get_live_by_name(
        DATASET_ID,
        NOTE_ID,
        "Legacy_Report.pdf",
    )
    assert registry is not None
    assert registry.original_file_name == "Legacy Report.PDF"
    assert not (tmp_path / "legacy-files-must-stay-empty").exists()

    listed = client.get(f"/api/v1/notes/{NOTE_ID}/attachments")
    assert listed.status_code == 200, listed.text
    assert listed.json()["count"] == 1
    assert listed.json()["attachments"][0]["file_name"] == "Legacy_Report.pdf"

    downloaded = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/Legacy_Report.pdf"
    )
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == b"%PDF-1.7\nlegacy payload\n%%EOF\n"

    deleted = client.delete(
        f"/api/v1/notes/{NOTE_ID}/attachments/Legacy_Report.pdf",
        headers={"Idempotency-Key": "legacy-delete-1"},
    )
    delete_replay = client.delete(
        f"/api/v1/notes/{NOTE_ID}/attachments/Legacy_Report.pdf",
        headers={"Idempotency-Key": "legacy-delete-1"},
    )
    delete_drift = client.delete(
        f"/api/v1/notes/{NOTE_ID}/attachments/Other.pdf",
        headers={"Idempotency-Key": "legacy-delete-1"},
    )
    assert deleted.status_code == 200, deleted.text
    assert delete_replay.status_code == 200, delete_replay.text
    assert delete_drift.status_code == 409, delete_drift.text
    tombstone = note_db.note_attachment_store.get(DATASET_ID, registry.attachment_id)
    assert tombstone is not None and tombstone.deleted is True


@pytest.mark.integration
def test_active_legacy_attachment_routes_map_invalid_idempotency_keys_to_400(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, note_db, _ = canonical_api

    rejected_upload = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        headers={"Idempotency-Key": "invalid key"},
        files={"file": ("report.txt", b"payload", "text/plain")},
    )
    rejected_delete = client.delete(
        f"/api/v1/notes/{NOTE_ID}/attachments/report.txt",
        headers={"Idempotency-Key": "invalid key"},
    )

    assert rejected_upload.status_code == 400, rejected_upload.text
    assert rejected_delete.status_code == 400, rejected_delete.text
    assert not note_db.note_attachment_store.list_page(
        DATASET_ID,
        NOTE_ID,
        after_attachment_id=None,
        limit=10,
        state="all",
    )


@pytest.mark.parametrize(
    ("payload", "declared_content_type"),
    [
        (b"plain text disguised as a PDF", "application/pdf"),
        (b"%PDF-1.7\nvalid bytes\n%%EOF\n", "text/plain"),
    ],
)
@pytest.mark.integration
def test_active_one_shot_upload_rejects_mismatched_content_before_blob_work(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
    payload: bytes,
    declared_content_type: str,
) -> None:
    client, note_db, _ = canonical_api

    rejected = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        headers={"Idempotency-Key": "mismatched-media-1"},
        files={"file": ("report.pdf", payload, declared_content_type)},
    )

    assert rejected.status_code == 400, rejected.text
    assert not note_db.note_attachment_store.list_page(
        DATASET_ID,
        NOTE_ID,
        after_attachment_id=None,
        limit=10,
        state="all",
    )


@pytest.mark.integration
def test_active_one_shot_upload_enforces_the_effective_sync_blob_limit(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, note_db, service = canonical_api
    service.settings = replace(service.settings, max_blob_bytes=4)

    rejected = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        headers={"Idempotency-Key": "too-large-1"},
        files={"file": ("large.txt", b"12345", "text/plain")},
    )

    assert rejected.status_code == 413, rejected.text
    assert not note_db.note_attachment_store.list_page(
        DATASET_ID,
        NOTE_ID,
        after_attachment_id=None,
        limit=10,
        state="all",
    )


@pytest.mark.integration
def test_active_one_shot_upload_rejects_noncanonical_media_type_before_blob_work(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, note_db, _ = canonical_api

    rejected = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        headers={"Idempotency-Key": "bad-media-type-1"},
        files={"file": ("report.pdf", b"payload", "Application/PDF")},
    )

    assert rejected.status_code == 400, rejected.text
    assert not note_db.note_attachment_store.list_page(
        DATASET_ID,
        NOTE_ID,
        after_attachment_id=None,
        limit=10,
        state="all",
    )


@pytest.mark.integration
def test_canonical_keyset_pages_batch_availability_and_reject_bad_cursors(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, service = canonical_api
    for index in range(2):
        created = client.post(
            f"/api/v1/notes/{NOTE_ID}/attachments",
            headers={"Idempotency-Key": f"page-create-{index}"},
            files={
                "file": (
                    f"page-{index}.txt",
                    f"payload-{index}".encode(),
                    "text/plain",
                )
            },
        )
        assert created.status_code == 201, created.text

    original = service.store.list_blob_availability_by_hashes
    availability_queries = 0

    def _availability(*args, **kwargs):
        nonlocal availability_queries
        availability_queries += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service.store, "list_blob_availability_by_hashes", _availability)
    monkeypatch.setattr(
        service.store,
        "get_blob_object",
        lambda *args, **kwargs: pytest.fail("canonical list performed an N+1 blob lookup"),
    )

    first = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID, "limit": 1},
    )
    assert first.status_code == 200, first.text
    assert first.json()["has_more"] is True
    assert len(first.json()["items"]) == 1
    cursor = first.json()["next_cursor"]

    second = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID, "limit": 1, "cursor": cursor},
    )
    assert second.status_code == 200, second.text
    assert second.json()["has_more"] is False
    assert len(second.json()["items"]) == 1
    assert availability_queries == 2

    tampered = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID, "cursor": cursor[:-1] + "x"},
    )
    oversized = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID, "cursor": "x" * 513},
    )
    assert tampered.status_code == 400
    assert oversized.status_code == 413


@pytest.mark.integration
def test_canonical_routes_hide_deleted_parent_and_fail_closed_when_writes_are_disabled(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, note_db, service = canonical_api
    upload_id = _complete_create_upload(service, b"attachment payload")
    created = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments/from-upload",
        params={"dataset_id": DATASET_ID},
        headers={"Idempotency-Key": "gate-create-1"},
        json={"upload_id": upload_id},
    )
    assert created.status_code == 201, created.text

    adapter = service.adapters.get("attachment.ref")
    assert isinstance(adapter, AttachmentRefDomainAdapter)
    adapter.v2_writes_enabled = False
    blocked = client.patch(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "gate-rename-1",
            "If-Match": created.json()["etag"],
        },
        json={"file_name": "Blocked.pdf"},
    )
    readable = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
    )
    unavailable_content = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}/content",
        params={"dataset_id": DATASET_ID},
    )
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["detail"]["error_code"] == "notes_attachment_sync_not_ready"
    assert readable.status_code == 200, readable.text
    assert unavailable_content.status_code == 404

    assert note_db.note_store.delete_note(NOTE_ID, expected_version=1) is True
    hidden_list = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID},
    )
    hidden_detail = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}",
        params={"dataset_id": DATASET_ID},
    )
    assert hidden_list.status_code == 404
    assert hidden_detail.status_code == 404


@pytest.mark.integration
def test_rollout_inactive_preserves_legacy_filesystem_routes_and_rejects_canonical_api(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = canonical_api
    legacy_root = tmp_path / "inactive-legacy-files"
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: None,
    )
    monkeypatch.setattr(
        notes_endpoint.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: legacy_root,
    )

    canonical = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID},
    )
    inactive_dataset_alias = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        params={"dataset_id": DATASET_ID},
        files={"file": ("ignored.txt", b"ignored", "text/plain")},
    )
    created = client.post(
        f"/api/v1/notes/{NOTE_ID}/attachments",
        files={"file": ("legacy.txt", b"legacy bytes", "text/plain")},
    )
    listed = client.get(f"/api/v1/notes/{NOTE_ID}/attachments")
    downloaded = client.get(f"/api/v1/notes/{NOTE_ID}/attachments/legacy.txt")
    deleted = client.delete(f"/api/v1/notes/{NOTE_ID}/attachments/legacy.txt")

    assert canonical.status_code == 409
    assert canonical.json()["detail"]["error_code"] == "notes_attachment_sync_inactive"
    assert inactive_dataset_alias.status_code == 409
    assert inactive_dataset_alias.json()["detail"]["error_code"] == (
        "notes_attachment_sync_inactive"
    )
    assert created.status_code == 201, created.text
    assert listed.status_code == 200 and listed.json()["count"] == 1
    assert downloaded.status_code == 200 and downloaded.content == b"legacy bytes"
    assert deleted.status_code == 200, deleted.text


@pytest.mark.parametrize("state", ["initializing", "failed"])
@pytest.mark.integration
def test_canonical_routes_fail_closed_before_attachment_initialization_is_ready(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
    state: str,
) -> None:
    client, _, service = canonical_api
    service.store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        (
            '{"client_family":"chatbook","default_personal":true,'
            f'"notes_attachment_v2":{{"state":"{state}"}}}}',
            DATASET_ID,
        ),
    )

    listed = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID},
    )

    assert listed.status_code == 409, listed.text
    assert listed.json()["detail"]["error_code"] == (
        "notes_attachment_dataset_unavailable"
    )


@pytest.mark.integration
def test_canonical_routes_require_attachment_domain_enrollment(
    canonical_api: tuple[TestClient, CharactersRAGDB, SyncV2Service],
) -> None:
    client, _, service = canonical_api
    service.store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
        ('["notes.note"]', DATASET_ID),
    )

    listed = client.get(
        f"/api/v1/notes/{NOTE_ID}/attachments/canonical",
        params={"dataset_id": DATASET_ID},
    )

    assert listed.status_code == 409, listed.text
    assert listed.json()["detail"]["error_code"] == (
        "notes_attachment_dataset_unavailable"
    )
