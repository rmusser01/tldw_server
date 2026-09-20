"""End-to-end Notes attachment lifecycle and restore proof."""

from __future__ import annotations

import hashlib

import pytest

from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.tests.Notes.test_notes_attachment_sync_api import (
    ATTACHMENT_ID,
    DATASET_ID,
    NOTE_ID,
    OWNER,
    _complete_create_upload,
    _complete_replace_upload,
)
from tldw_Server_API.tests.Notes.test_notes_attachment_sync_api import (
    canonical_api as _canonical_api,
)


def test_notes_attachment_full_lifecycle_restores_on_a_new_device(
    request: pytest.FixtureRequest,
) -> None:
    canonical_api = request.getfixturevalue("_" + _canonical_api.__name__)
    client, note_db, service = canonical_api
    note_payload = {"title": "Parent", "content": "Body"}
    service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id="e2e-parent-note",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            device_id="device-1",
            client_sequence=1,
            object_revision=1,
            payload=note_payload,
            payload_hash="sha256:" + hashlib.sha256(b'{"content":"Body","title":"Parent"}').hexdigest(),
            payload_size_bytes=len(str(note_payload).encode("utf-8")),
            created_at_client="2026-08-11T20:00:00+00:00",
            encryption_metadata={"policy": "server_trusted_v1"},
            stable_key=f"note:{NOTE_ID}",
            apply_status="applied",
        )
    )
    upload_id = _complete_create_upload(service, b"first attachment payload")
    collection_path = f"/api/v1/notes/{NOTE_ID}/attachments/from-upload"
    item_path = f"/api/v1/notes/{NOTE_ID}/attachments/by-id/{ATTACHMENT_ID}"

    created = client.post(
        collection_path,
        params={"dataset_id": DATASET_ID},
        headers={"Idempotency-Key": "e2e-create"},
        json={"upload_id": upload_id},
    )
    assert created.status_code == 201, created.text

    renamed = client.patch(
        item_path,
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "e2e-rename",
            "If-Match": created.json()["etag"],
        },
        json={"file_name": "Lifecycle.pdf"},
    )
    assert renamed.status_code == 200, renamed.text

    replacement_upload_id = _complete_replace_upload(
        service,
        b"replacement attachment payload",
    )
    replaced = client.post(
        collection_path,
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "e2e-replace",
            "If-Match": renamed.json()["etag"],
        },
        json={"upload_id": replacement_upload_id},
    )
    assert replaced.status_code == 201, replaced.text

    assert note_db.note_store.delete_note(NOTE_ID, expected_version=1) is True
    hidden = client.get(item_path, params={"dataset_id": DATASET_ID})
    assert hidden.status_code == 404
    assert note_db.note_store.restore_note(NOTE_ID, expected_version=2) is True
    visible = client.get(item_path, params={"dataset_id": DATASET_ID})
    assert visible.status_code == 200, visible.text
    assert visible.json()["etag"] == replaced.json()["etag"]

    deleted = client.request(
        "DELETE",
        item_path,
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "e2e-delete",
            "If-Match": replaced.json()["etag"],
        },
        json={"reason": "e2e lifecycle"},
    )
    assert deleted.status_code == 200, deleted.text
    restored = client.post(
        f"{item_path}/restore",
        params={"dataset_id": DATASET_ID},
        headers={
            "Idempotency-Key": "e2e-restore",
            "If-Match": deleted.json()["etag"],
        },
        json={"reason": "undo e2e lifecycle"},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["state"] == "live"

    service.register_device(
        user_id=OWNER,
        device_id="attachment-restore-device",
        display_name="Fresh restore device",
        client_type="chatbook",
        capabilities={
            "requested_domains": ["notes.note", "attachment.ref"],
            "supported_adapter_versions": {"attachment.ref": [2]},
        },
    )
    pulled = service.pull(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        device_id="attachment-restore-device",
        domains=["notes.note", "attachment.ref"],
        include_own_changes=True,
    )
    pulled_attachment = next(envelope for envelope in pulled.envelopes if envelope.domain == "attachment.ref")
    assert pulled_attachment.object_id == ATTACHMENT_ID
    assert pulled_attachment.operation == "upsert"

    preview = service.restore_preview(
        user_id=OWNER,
        device_id="attachment-restore-device",
        dataset_ids=[DATASET_ID],
        domains=["notes.note", "attachment.ref"],
        selected_object_ids=[NOTE_ID],
        selected_attachment_ids=[ATTACHMENT_ID],
        local_inventory=[],
    )
    assert preview.restore_status == "content_complete"
    assert [item.attachment_id for item in preview.attachment_refs] == [ATTACHMENT_ID]

    before = service.store.db.execute("SELECT * FROM sync_blob_objects ORDER BY blob_id").rows
    retention = service.retention_dry_run(
        user_id=OWNER,
        dataset_id=DATASET_ID,
        domains=["attachment.ref"],
        audit_mode=True,
        limit=100,
    )
    assert retention.audit_mode is True
    assert all(candidate.blockers for candidate in retention.candidates)
    assert service.store.db.execute("SELECT * FROM sync_blob_objects ORDER BY blob_id").rows == before
