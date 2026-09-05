"""Verify active Personal Context exchange proofs and atomic endpoint gating."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import (
    PostgreSQLBackend,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import service as service_module
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncConflictCreate,
    SyncDatasetCreate,
    SyncDeviceCursor,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
    PersonalContextExchangeProof,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    preference_record,
)
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_transport import (
    DATASET_ID,
    DOMAIN,
    EXCHANGE,
    _envelope,
    _service,
)
from tldw_Server_API.tests.Sync.test_sync_v2_service import _OutcomeMaterializer

pytestmark = pytest.mark.unit

Operation = Literal["push", "pull", "conflict_list", "conflict_resolve"]
ProofState = Literal[
    "exact",
    "missing",
    "stale_epoch",
    "stale_token",
    "wrong_device",
    "incomplete_link",
    "tampered_stored",
    "version_zero",
]
DeviceState = Literal["active", "paused", "revoked", "unknown"]


def _client(service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = lambda: User(
        id="user-a", username="user-a"
    )
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    return TestClient(app)


def _set_personal_context_state(
    service: SyncV2Service,
    **updates: object,
) -> None:
    dataset = service.store.get_dataset(DATASET_ID)
    assert dataset is not None
    metadata = dict(dataset.metadata)
    state = dict(metadata["personal_context"])
    state.update(updates)
    metadata["personal_context"] = state
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(metadata, sort_keys=True), DATASET_ID),
            connection=connection,
        )


def _remove_link_receipt(service: SyncV2Service, device_id: str) -> None:
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            """DELETE FROM sync_personal_context_link_receipts
                WHERE user_id = ? AND dataset_id = ? AND device_id = ?""",
            ("user-a", DATASET_ID, device_id),
            connection=connection,
        )


def _set_device_state(service: SyncV2Service, state: DeviceState) -> str:
    device_id = "device-a"
    if state == "revoked":
        service.store.revoke_device(
            user_id="user-a",
            device_id=device_id,
            reason="review-test",
        )
    elif state == "paused":
        with service.store.db.backend.transaction() as connection:
            service.store.db.execute(
                "UPDATE sync_devices SET status = ? WHERE user_id = ? AND device_id = ?",
                ("paused", "user-a", device_id),
                connection=connection,
            )
    elif state == "unknown":
        device_id = "device-unknown"
        service.store.complete_personal_context_link_receipt(
            user_id="user-a",
            dataset_id=DATASET_ID,
            device_id=device_id,
            profile_id="profile-a",
            integrity_key_id="personal-context-integrity-v1",
            purge_generation=0,
            bootstrap_cursor="fixture-cursor",
        )
    return device_id


def _proof_for_state(state: ProofState) -> PersonalContextExchangeProof | None:
    if state == "missing":
        return None
    if state == "stale_epoch":
        return EXCHANGE.model_copy(
            update={"activation_epoch": "stale_epoch_0123456789abcdef"}
        )
    if state == "stale_token":
        return EXCHANGE.model_copy(
            update={"continuity_token": "stale_token_0123456789abcdef"}
        )
    return EXCHANGE


def _prepare_state(
    service: SyncV2Service,
    state: ProofState,
) -> tuple[str, PersonalContextExchangeProof | None]:
    device_id = "device-a"
    if state == "wrong_device":
        device_id = "device-b"
        _remove_link_receipt(service, device_id)
    elif state == "incomplete_link":
        _set_personal_context_state(service, link_state="bootstrap_pending")
    elif state == "tampered_stored":
        _set_personal_context_state(
            service, continuity_token="stored_tamper_0123456789abcdef"
        )
    elif state == "version_zero":
        _set_personal_context_state(service, ongoing_sync_version=0)
    return device_id, _proof_for_state(state)


def _proof_query(proof: PersonalContextExchangeProof | None) -> dict[str, str]:
    if proof is None:
        return {}
    return {
        "personal_context_activation_epoch": proof.activation_epoch,
        "personal_context_continuity_token": proof.continuity_token,
    }


def _insert_conflict(
    service: SyncV2Service,
    *,
    conflict_id: str,
    domain: Literal["notes.note", "personal_context.record"],
    with_source: bool = True,
) -> None:
    if domain == DOMAIN:
        payload = preference_record(
            record_id=f"record-{conflict_id}",
            version_id=f"version-{conflict_id}",
        ).model_dump(mode="json")
        envelope = _envelope(
            payload,
            client_envelope_id=f"client-envelope-{conflict_id}",
        )
    else:
        envelope = SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id=f"client-envelope-{conflict_id}",
            device_id="device-a",
            domain="notes.note",
            operation="upsert",
            object_id=f"note-{conflict_id}",
            payload={"title": "Note", "content": "body"},
            payload_hash=f"sha256:{conflict_id}",
        )
    stored = (
        service.store.insert_envelope(replace(envelope, apply_status="conflict"))
        if with_source
        else None
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id=conflict_id,
            dataset_id=DATASET_ID,
            domain=domain,
            object_id=envelope.object_id,
            conflict_type="revision_mismatch",
            local_envelope_id=(stored.client_envelope_id if stored is not None else None),
            remote_envelope_id=(
                f"remote-envelope-{conflict_id}" if stored is not None else None
            ),
            server_sequence=(stored.server_cursor if stored is not None else None),
            metadata={"private_marker": f"secret-{conflict_id}"},
        )
    )


def _insert_mixed_conflicts_with_sources(
    service: SyncV2Service,
    note_conflict_id: str,
    personal_conflict_id: str,
) -> None:
    personal_payload = preference_record(
        record_id=f"record-{personal_conflict_id}",
        version_id=f"version-{personal_conflict_id}",
    ).model_dump(mode="json")
    envelopes = [
        SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id=f"client-envelope-{note_conflict_id}",
            device_id="device-a",
            domain="notes.note",
            operation="upsert",
            object_id=f"note-{note_conflict_id}",
            payload={"title": "Note", "content": "body"},
            payload_hash=f"sha256:{note_conflict_id}",
        ),
        _envelope(
            personal_payload,
            client_envelope_id=f"client-envelope-{personal_conflict_id}",
        ),
    ]
    stored = [
        service.store.insert_envelope(replace(item, apply_status="conflict"))
        for item in envelopes
    ]
    for conflict_id, envelope in zip(
        (note_conflict_id, personal_conflict_id),
        stored,
        strict=True,
    ):
        service.store.insert_conflict(
            SyncConflictCreate(
                conflict_id=conflict_id,
                dataset_id=DATASET_ID,
                domain=envelope.domain,
                object_id=envelope.object_id,
                conflict_type="revision_mismatch",
                local_envelope_id=envelope.client_envelope_id,
                remote_envelope_id=f"remote-envelope-{conflict_id}",
                server_sequence=envelope.server_cursor,
                metadata={"private_marker": f"secret-{conflict_id}"},
            )
        )


def _insert_conflict_set_with_sources(
    service: SyncV2Service,
    conflicts: list[
        tuple[str, Literal["notes.note", "personal_context.record"]]
    ],
) -> None:
    envelopes: list[SyncEnvelopeCreate] = []
    for conflict_id, domain in conflicts:
        if domain == DOMAIN:
            payload = preference_record(
                record_id=f"record-{conflict_id}",
                version_id=f"version-{conflict_id}",
            ).model_dump(mode="json")
            envelopes.append(
                _envelope(
                    payload,
                    client_envelope_id=f"client-envelope-{conflict_id}",
                )
            )
        else:
            envelopes.append(
                SyncEnvelopeCreate(
                    dataset_id=DATASET_ID,
                    client_envelope_id=f"client-envelope-{conflict_id}",
                    device_id="device-a",
                    domain="notes.note",
                    operation="upsert",
                    object_id=f"note-{conflict_id}",
                    payload={"title": "Note", "content": "body"},
                    payload_hash=f"sha256:{conflict_id}",
                )
            )
    stored = [
        service.store.insert_envelope(replace(item, apply_status="conflict"))
        for item in envelopes
    ]
    for (conflict_id, _domain), envelope in zip(
        conflicts,
        stored,
        strict=True,
    ):
        service.store.insert_conflict(
            SyncConflictCreate(
                conflict_id=conflict_id,
                dataset_id=DATASET_ID,
                domain=envelope.domain,
                object_id=envelope.object_id,
                conflict_type="revision_mismatch",
                local_envelope_id=envelope.client_envelope_id,
                remote_envelope_id=f"remote-envelope-{conflict_id}",
                server_sequence=envelope.server_cursor,
                metadata={"private_marker": f"secret-{conflict_id}"},
            )
        )


def _notes_resolution(
    conflict_id: str,
    action: Literal["skip", "overwrite", "duplicate_rename"],
) -> dict[str, object]:
    resolution: dict[str, object] = {
        "conflict_id": conflict_id,
        "action": action,
    }
    if action != "skip":
        object_id = (
            f"renamed-note-{conflict_id}"
            if action == "duplicate_rename"
            else f"note-{conflict_id}"
        )
        resolution["resolution_envelope"] = {
            "dataset_id": DATASET_ID,
            "client_envelope_id": f"resolution-envelope-{conflict_id}",
            "device_id": "device-a",
            "domain": "notes.note",
            "operation": "upsert",
            "object_id": object_id,
            "payload": {"title": "Resolved", "content": action},
            "payload_hash": f"sha256:resolution-{conflict_id}",
        }
    return resolution


def _request_operation(
    client: TestClient,
    service: SyncV2Service,
    *,
    operation: Operation,
    state: ProofState,
) -> tuple[object, str | None]:
    device_id, proof = _prepare_state(service, state)
    conflict_id: str | None = None
    if operation == "push":
        payload = preference_record(
            record_id="record-gate-push",
            version_id="version-gate-push",
        ).model_dump(mode="json")
        body: dict[str, object] = {
            "dataset_id": DATASET_ID,
            "device_id": device_id,
            "envelopes": [
                asdict(
                    replace(
                        _envelope(
                            payload,
                            client_envelope_id="client-envelope-gate-push",
                        ),
                        device_id=device_id,
                    )
                )
            ],
        }
        if proof is not None:
            body["personal_context_exchange"] = proof.model_dump(mode="json")
        return client.post("/api/v1/sync/push", json=body), None
    if operation == "pull":
        return client.get(
            "/api/v1/sync/pull",
            params={
                "dataset_id": DATASET_ID,
                "device_id": device_id,
                "domain": DOMAIN,
                **_proof_query(proof),
            },
        ), None

    conflict_id = f"conflict-{operation}-{state}"
    _insert_conflict(
        service,
        conflict_id=conflict_id,
        domain=DOMAIN,
        with_source=operation == "conflict_resolve",
    )
    if operation == "conflict_list":
        return client.get(
            "/api/v1/sync/conflicts",
            params={
                "dataset_id": DATASET_ID,
                "device_id": device_id,
                "domain": DOMAIN,
                **_proof_query(proof),
            },
        ), conflict_id

    resolution: dict[str, object] = {
        "conflict_id": conflict_id,
        "action": "skip",
    }
    body = {
        "dataset_id": DATASET_ID,
        "device_id": device_id,
        "resolutions": [resolution],
    }
    if proof is not None:
        body["personal_context_exchange"] = proof.model_dump(mode="json")
        resolution.update(
            {
                "expected_local_envelope_id": f"client-envelope-{conflict_id}",
                "expected_remote_envelope_id": f"remote-envelope-{conflict_id}",
                "idempotency_key": f"idempotency-key-{conflict_id}",
            }
        )
    return client.post("/api/v1/sync/conflicts/resolve", json=body), conflict_id


@pytest.mark.parametrize(
    "operation",
    ["push", "pull", "conflict_list", "conflict_resolve"],
)
@pytest.mark.parametrize(
    "state",
    [
        "exact",
        "missing",
        "stale_epoch",
        "stale_token",
        "wrong_device",
        "incomplete_link",
        "tampered_stored",
        "version_zero",
    ],
)
def test_personal_context_exchange_gate_matrix(
    tmp_path: Path,
    operation: Operation,
    state: ProofState,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    client = _client(service)
    before_cursors = {
        device_id: service.store.get_device_cursor(DATASET_ID, device_id, DOMAIN)
        for device_id in ("device-a", "device-b")
    }

    response, conflict_id = _request_operation(
        client,
        service,
        operation=operation,
        state=state,
    )

    if state == "exact":
        assert response.status_code == 200, response.text
        assert response.json()["personal_context_exchange"] == EXCHANGE.model_dump(
            mode="json"
        )
        return

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "error_code": "personal_context_activation_required",
        "message": "An active Personal Context exchange is required.",
    }
    assert "private_marker" not in response.text
    assert "secret-" not in response.text
    assert service.store.get_envelope_by_client_id(
        DATASET_ID, "client-envelope-gate-push"
    ) is None
    if conflict_id is not None:
        conflict = service.store.get_conflict(conflict_id)
        assert conflict is not None
        assert conflict.status == "unresolved"
    assert {
        device_id: service.store.get_device_cursor(DATASET_ID, device_id, DOMAIN)
        for device_id in ("device-a", "device-b")
    } == before_cursors


@pytest.mark.parametrize(
    "operation",
    ["push", "pull", "conflict_list", "conflict_resolve"],
)
@pytest.mark.parametrize("device_state", ["active", "paused", "revoked", "unknown"])
def test_personal_context_exchange_requires_current_active_device_registration(
    tmp_path: Path,
    operation: Operation,
    device_state: DeviceState,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    device_id = _set_device_state(service, device_state)
    conflict_id = f"device-state-{operation}-{device_state}"
    before_cursors = {
        known_device: service.store.get_device_cursor(DATASET_ID, known_device, DOMAIN)
        for known_device in ("device-a", "device-b")
    }
    if operation == "push":
        payload = preference_record(
            record_id="device-state-record",
            version_id="device-state-version",
        ).model_dump(mode="json")
        response = _client(service).post(
            "/api/v1/sync/push",
            json={
                "dataset_id": DATASET_ID,
                "device_id": device_id,
                "envelopes": [
                    asdict(
                        replace(
                            _envelope(
                                payload,
                                client_envelope_id="device-state-push",
                            ),
                            device_id=device_id,
                        )
                    )
                ],
                "personal_context_exchange": EXCHANGE.model_dump(mode="json"),
            },
        )
    elif operation == "pull":
        response = _client(service).get(
            "/api/v1/sync/pull",
            params={
                "dataset_id": DATASET_ID,
                "device_id": device_id,
                "domain": DOMAIN,
                **_proof_query(EXCHANGE),
            },
        )
    else:
        _insert_conflict(
            service,
            conflict_id=conflict_id,
            domain=DOMAIN,
            with_source=operation == "conflict_resolve",
        )
        if operation == "conflict_list":
            response = _client(service).get(
                "/api/v1/sync/conflicts",
                params={
                    "dataset_id": DATASET_ID,
                    "device_id": device_id,
                    "domain": DOMAIN,
                    **_proof_query(EXCHANGE),
                },
            )
        else:
            response = _client(service).post(
                "/api/v1/sync/conflicts/resolve",
                json={
                    "dataset_id": DATASET_ID,
                    "device_id": device_id,
                    "personal_context_exchange": EXCHANGE.model_dump(mode="json"),
                    "resolutions": [
                        {
                            "conflict_id": conflict_id,
                            "action": "skip",
                            "expected_local_envelope_id": (
                                f"client-envelope-{conflict_id}"
                            ),
                            "expected_remote_envelope_id": (
                                f"remote-envelope-{conflict_id}"
                            ),
                            "idempotency_key": f"idempotency-key-{conflict_id}",
                        }
                    ],
                },
            )

    if device_state == "active":
        assert response.status_code == 200, response.text
        return
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "error_code": "personal_context_activation_required",
        "message": "An active Personal Context exchange is required.",
    }
    assert conflict_id not in response.text
    assert "private_marker" not in response.text
    assert service.store.get_envelope_by_client_id(DATASET_ID, "device-state-push") is None
    if operation == "conflict_resolve":
        conflict = service.store.get_conflict(conflict_id)
        assert conflict is not None
        assert conflict.status == "unresolved"
    assert {
        known_device: service.store.get_device_cursor(DATASET_ID, known_device, DOMAIN)
        for known_device in ("device-a", "device-b")
    } == before_cursors


@pytest.mark.parametrize("state", ["missing", "version_zero"])
def test_pull_rejects_before_personal_context_recovery_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: Literal["missing", "version_zero"],
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    device_id, proof = _prepare_state(service, state)
    called = False

    def fail_if_called(**_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("recovery coordinator ran before proof validation")

    monkeypatch.setattr(
        service,
        "_coordinate_personal_context_recovery",
        fail_if_called,
    )

    response = _client(service).get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": DATASET_ID,
            "device_id": device_id,
            "domain": DOMAIN,
            **_proof_query(proof),
        },
    )

    assert response.status_code == 409
    assert called is False


def test_mixed_dataset_lists_notes_conflict_without_personal_context_proof(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(
        service,
        conflict_id="a-notes-conflict",
        domain="notes.note",
        with_source=False,
    )
    _insert_conflict(
        service,
        conflict_id="z-personal-conflict",
        domain=DOMAIN,
        with_source=False,
    )

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": DATASET_ID, "domain": "notes.note"},
    )

    assert response.status_code == 200, response.text
    assert [item["domain"] for item in response.json()] == ["notes.note"]


def test_bounded_notes_page_does_not_gate_for_later_personal_context_conflict(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(
        service,
        conflict_id="a-notes-conflict",
        domain="notes.note",
        with_source=False,
    )
    _insert_conflict(
        service,
        conflict_id="z-personal-conflict",
        domain=DOMAIN,
        with_source=False,
    )

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": DATASET_ID, "limit": 1},
    )

    assert response.status_code == 200, response.text
    assert [item["conflict_id"] for item in response.json()] == ["a-notes-conflict"]


def test_mixed_dataset_resolves_selected_notes_conflict_without_exchange(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(service, conflict_id="notes-selected", domain="notes.note")
    _insert_conflict(
        service,
        conflict_id="personal-not-selected",
        domain=DOMAIN,
        with_source=False,
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "resolutions": [
                {"conflict_id": "notes-selected", "action": "skip"}
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["resolved"][0]["conflict_id"] == "notes-selected"
    assert service.store.get_conflict("notes-selected").status == "dismissed"
    assert service.store.get_conflict("personal-not-selected").status == "unresolved"


def test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    conflict_ids = ["mixed-exact-note", "mixed-exact-personal"]
    _insert_mixed_conflicts_with_sources(service, *conflict_ids)

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "personal_context_exchange": EXCHANGE.model_dump(mode="json"),
            "resolutions": [
                {
                    "conflict_id": conflict_ids[0],
                    "action": "skip",
                },
                {
                    "conflict_id": conflict_ids[1],
                    "action": "skip",
                    "expected_local_envelope_id": (
                        f"client-envelope-{conflict_ids[1]}"
                    ),
                    "expected_remote_envelope_id": (
                        f"remote-envelope-{conflict_ids[1]}"
                    ),
                    "idempotency_key": f"idempotency-{conflict_ids[1]}",
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["conflict_id"] for item in response.json()["resolved"]] == conflict_ids
    assert response.json()["rejected"] == []
    assert response.json()["personal_context_exchange"] == EXCHANGE.model_dump(
        mode="json"
    )
    assert [service.store.get_conflict(item).status for item in conflict_ids] == [
        "dismissed",
        "dismissed",
    ]


@pytest.mark.parametrize("notes_action", ["overwrite", "duplicate_rename"])
def test_mixed_exact_proof_preserves_native_notes_resolution_actions(
    tmp_path: Path,
    notes_action: Literal["overwrite", "duplicate_rename"],
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    service.materializers["notes.note"] = _OutcomeMaterializer()
    note_id = f"mixed-{notes_action}-note"
    personal_id = f"mixed-{notes_action}-personal"
    _insert_conflict_set_with_sources(
        service,
        [(note_id, "notes.note"), (personal_id, DOMAIN)],
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "personal_context_exchange": EXCHANGE.model_dump(mode="json"),
            "resolutions": [
                _notes_resolution(note_id, notes_action),
                {
                    "conflict_id": personal_id,
                    "action": "skip",
                    "expected_local_envelope_id": (
                        f"client-envelope-{personal_id}"
                    ),
                    "expected_remote_envelope_id": (
                        f"remote-envelope-{personal_id}"
                    ),
                    "idempotency_key": f"idempotency-{personal_id}",
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["conflict_id"] for item in response.json()["resolved"]] == [
        note_id,
        personal_id,
    ]
    assert response.json()["rejected"] == []
    assert [service.store.get_conflict(item).status for item in (note_id, personal_id)] == [
        "resolved",
        "dismissed",
    ]
    resolved_note = service.store.get_conflict(note_id)
    stored_resolution = service.store.get_envelope_by_client_id(
        DATASET_ID,
        f"resolution-envelope-{note_id}",
    )
    assert resolved_note is not None
    assert stored_resolution is not None
    assert resolved_note.resolved_by_envelope_id == stored_resolution.envelope_id
    assert stored_resolution.apply_status == "applied"
    assert stored_resolution.object_id == (
        f"renamed-note-{note_id}"
        if notes_action == "duplicate_rename"
        else f"note-{note_id}"
    )


def test_bad_personal_context_item_shape_is_rejected_before_resolution_loop(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    personal_id = "bad-shape-personal"
    note_id = "bad-shape-note"
    _insert_conflict_set_with_sources(
        service,
        [(note_id, "notes.note"), (personal_id, DOMAIN)],
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "personal_context_exchange": EXCHANGE.model_dump(mode="json"),
            "resolutions": [
                {"conflict_id": personal_id, "action": "skip"},
                _notes_resolution(note_id, "skip"),
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["conflict_id"] for item in response.json()["rejected"]] == [
        personal_id
    ]
    assert [item["conflict_id"] for item in response.json()["resolved"]] == [note_id]
    assert service.store.get_conflict(personal_id).status == "unresolved"
    assert service.store.get_conflict(note_id).status == "dismissed"


def test_mixed_selected_conflicts_with_stale_proof_mutate_nothing(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    conflict_ids = ["mixed-stale-note", "mixed-stale-personal"]
    _insert_mixed_conflicts_with_sources(service, *conflict_ids)
    stale = EXCHANGE.model_copy(
        update={"continuity_token": "stale_token_0123456789abcdef"}
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "personal_context_exchange": stale.model_dump(mode="json"),
            "resolutions": [
                {
                    "conflict_id": conflict_ids[0],
                    "action": "skip",
                },
                {
                    "conflict_id": conflict_ids[1],
                    "action": "skip",
                    "expected_local_envelope_id": (
                        f"client-envelope-{conflict_ids[1]}"
                    ),
                    "expected_remote_envelope_id": (
                        f"remote-envelope-{conflict_ids[1]}"
                    ),
                    "idempotency_key": f"idempotency-{conflict_ids[1]}",
                },
            ],
        },
    )

    assert response.status_code == 409
    assert [service.store.get_conflict(item).status for item in conflict_ids] == [
        "unresolved",
        "unresolved",
    ]


def test_conflict_batch_stale_preflight_cannot_mutate_preceding_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    conflict_ids = ["toctou-note", "toctou-personal"]
    _insert_mixed_conflicts_with_sources(service, *conflict_ids)
    monkeypatch.setattr(
        service,
        "require_active_exchange_for_conflicts",
        lambda **_kwargs: None,
        raising=False,
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "resolutions": [
                {"conflict_id": conflict_id, "action": "skip"}
                for conflict_id in conflict_ids
            ],
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "personal_context_activation_required"
    )
    assert [service.store.get_conflict(item).status for item in conflict_ids] == [
        "unresolved",
        "unresolved",
    ]
    assert "private_marker" not in response.text


def test_conflict_batch_duplicate_id_replays_in_request_order(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    conflict_id = "duplicate-note-resolution"
    _insert_conflict(service, conflict_id=conflict_id, domain="notes.note")

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "resolutions": [
                {"conflict_id": conflict_id, "action": "skip"},
                {"conflict_id": conflict_id, "action": "skip"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["conflict_id"] for item in response.json()["resolved"]] == [
        conflict_id,
        conflict_id,
    ]
    assert response.json()["rejected"] == []
    assert service.store.get_conflict(conflict_id).status == "dismissed"


def test_conflict_batch_savepoint_rolls_back_failed_item_and_continues(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(
        service,
        conflict_id="savepoint-valid",
        domain="notes.note",
    )
    _insert_conflict(
        service,
        conflict_id="savepoint-missing-source",
        domain="notes.note",
        with_source=False,
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "resolutions": [
                {"conflict_id": "savepoint-missing-source", "action": "skip"},
                {"conflict_id": "savepoint-valid", "action": "skip"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["conflict_id"] for item in response.json()["rejected"]] == [
        "savepoint-missing-source"
    ]
    assert [item["conflict_id"] for item in response.json()["resolved"]] == [
        "savepoint-valid"
    ]
    assert service.store.get_conflict("savepoint-missing-source").status == (
        "unresolved"
    )
    assert service.store.get_conflict("savepoint-valid").status == "dismissed"


def test_conflict_batch_hides_missing_and_foreign_ids_then_resolves_valid(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(service, conflict_id="selected-valid", domain="notes.note")
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-foreign",
            owner_user_id="user-a",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
        )
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="selected-foreign",
            dataset_id="dataset-foreign",
            domain="notes.note",
            object_id="foreign-note",
            conflict_type="revision_mismatch",
        )
    )

    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "resolutions": [
                {"conflict_id": "selected-foreign", "action": "skip"},
                {"conflict_id": "selected-missing", "action": "skip"},
                {"conflict_id": "selected-valid", "action": "skip"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    rejected = response.json()["rejected"]
    assert [item["conflict_id"] for item in rejected] == [
        "selected-foreign",
        "selected-missing",
    ]
    assert len({(item["error_code"], item["message"]) for item in rejected}) == 1
    assert [item["conflict_id"] for item in response.json()["resolved"]] == [
        "selected-valid"
    ]
    assert service.store.get_conflict("selected-foreign").status == "unresolved"
    assert service.store.get_conflict("selected-valid").status == "dismissed"


@pytest.mark.parametrize(
    ("fails", "expected_queries"),
    [
        (
            False,
            [
                "SAVEPOINT sync_conflict_resolution_item",
                "RELEASE SAVEPOINT sync_conflict_resolution_item",
            ],
        ),
        (
            True,
            [
                "SAVEPOINT sync_conflict_resolution_item",
                "ROLLBACK TO SAVEPOINT sync_conflict_resolution_item",
                "RELEASE SAVEPOINT sync_conflict_resolution_item",
            ],
        ),
    ],
)
def test_conflict_resolution_savepoint_sql_is_sqlite_and_postgres_portable(
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
    expected_queries: list[str],
) -> None:
    database = cast(SyncDatabase, object.__new__(SyncDatabase))
    calls: list[str] = []

    def record(query: str, *_args: object, **_kwargs: object) -> None:
        calls.append(query)

    monkeypatch.setattr(database, "execute", record)
    if fails:
        with pytest.raises(RuntimeError, match="ordinary item failure"):
            with database.conflict_resolution_savepoint(connection=object()):
                raise RuntimeError("ordinary item failure")
    else:
        with database.conflict_resolution_savepoint(connection=object()):
            pass

    assert calls == expected_queries
    for query in calls:
        prepared, params = PostgreSQLBackend._prepare_query(
            cast(Any, database),
            query,
            None,
        )
        assert prepared == query
        assert params is None


def test_postgres_conflict_row_lock_is_scoped_to_the_guarded_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = cast(SyncDatabase, object.__new__(SyncDatabase))
    database.backend = SimpleNamespace(
        config=SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    )
    calls: list[tuple[str, tuple[object, ...]]] = []

    def record(
        query: str,
        params: tuple[object, ...],
        **_kwargs: object,
    ) -> SimpleNamespace:
        calls.append((" ".join(query.split()), params))
        return SimpleNamespace(rows=[])

    monkeypatch.setattr(database, "execute", record)

    assert (
        database.get_conflict(
            "foreign-conflict",
            dataset_id=DATASET_ID,
            for_update=True,
        )
        is None
    )
    assert calls == [
        (
            "SELECT * FROM sync_conflicts WHERE conflict_id = ? "
            "AND dataset_id = ? FOR UPDATE",
            ("foreign-conflict", DATASET_ID),
        )
    ]
    prepared, params = PostgreSQLBackend._prepare_query(
        cast(Any, database),
        calls[0][0],
        calls[0][1],
    )
    assert prepared.endswith("conflict_id = %s AND dataset_id = %s FOR UPDATE")
    assert params == ("foreign-conflict", DATASET_ID)


def test_conflict_batch_locks_unique_selected_ids_in_sorted_dataset_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    calls: list[tuple[str, str | None, bool]] = []

    def record(
        _store: SyncV2Store,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        for_update: bool = False,
    ) -> None:
        calls.append((conflict_id, dataset_id, for_update))
        return None

    monkeypatch.setattr(SyncV2Store, "get_conflict", record)
    response = _client(service).post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "resolutions": [
                {"conflict_id": "lock-b", "action": "skip"},
                {"conflict_id": "lock-a", "action": "skip"},
                {"conflict_id": "lock-b", "action": "skip"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert calls == [
        ("lock-a", DATASET_ID, True),
        ("lock-b", DATASET_ID, True),
    ]
    assert [item["conflict_id"] for item in response.json()["rejected"]] == [
        "lock-b",
        "lock-a",
        "lock-b",
    ]


@pytest.mark.parametrize(
    ("operation", "expected_status"),
    [
        ("push", 404),
        ("pull", 404),
        ("conflict_list", 200),
        ("conflict_resolve", 200),
    ],
)
def test_notes_only_unknown_device_preserves_legacy_behavior(
    tmp_path: Path,
    operation: Operation,
    expected_status: int,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    client = _client(service)
    conflict_id = f"unknown-device-{operation}"
    if operation == "push":
        response = client.post(
            "/api/v1/sync/push",
            json={
                "dataset_id": DATASET_ID,
                "device_id": "device-unknown",
                "envelopes": [
                    {
                        "dataset_id": DATASET_ID,
                        "client_envelope_id": "unknown-device-note",
                        "device_id": "device-unknown",
                        "domain": "notes.note",
                        "operation": "upsert",
                        "object_id": "unknown-device-note",
                        "payload": {"title": "Note", "content": "body"},
                        "payload_hash": "sha256:unknown-device-note",
                    }
                ],
            },
        )
    elif operation == "pull":
        response = client.get(
            "/api/v1/sync/pull",
            params={
                "dataset_id": DATASET_ID,
                "device_id": "device-unknown",
                "domain": "notes.note",
            },
        )
    else:
        _insert_conflict(
            service,
            conflict_id=conflict_id,
            domain="notes.note",
            with_source=operation == "conflict_resolve",
        )
        if operation == "conflict_list":
            response = client.get(
                "/api/v1/sync/conflicts",
                params={
                    "dataset_id": DATASET_ID,
                    "device_id": "device-unknown",
                    "domain": "notes.note",
                },
            )
        else:
            response = client.post(
                "/api/v1/sync/conflicts/resolve",
                json={
                    "dataset_id": DATASET_ID,
                    "device_id": "device-unknown",
                    "resolutions": [
                        {"conflict_id": conflict_id, "action": "skip"}
                    ],
                },
            )

    assert response.status_code == expected_status, response.text
    assert "personal_context_activation_required" not in response.text
    if operation == "conflict_resolve":
        assert response.json()["resolved"] == []
        assert response.json()["rejected"][0]["conflict_id"] == conflict_id
        assert service.store.get_conflict(conflict_id).status == "unresolved"


def test_mixed_notes_push_and_pull_remain_legacy_compatible(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    client = _client(service)
    pushed = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "envelopes": [
                {
                    "dataset_id": DATASET_ID,
                    "client_envelope_id": "legacy-notes-envelope",
                    "device_id": "device-a",
                    "domain": "notes.note",
                    "operation": "upsert",
                    "object_id": "legacy-note",
                    "payload": {"title": "Legacy", "content": "still works"},
                    "payload_hash": "sha256:legacy-note",
                }
            ],
        },
    )
    pulled = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "domain": "notes.note",
            "include_same_device_echoes": True,
        },
    )

    assert pushed.status_code == 200, pushed.text
    assert pushed.json()["personal_context_exchange"] is None
    assert pulled.status_code == 200, pulled.text
    assert [item["object_id"] for item in pulled.json()["envelopes"]] == [
        "legacy-note"
    ]
    assert pulled.json()["personal_context_exchange"] is None


def test_conflict_list_optional_query_parameters_preserve_legacy_shape(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(
        service,
        conflict_id="notes-legacy-shape",
        domain="notes.note",
        with_source=False,
    )

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "domain": "notes.note",
        },
    )

    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)
    assert response.json()[0]["conflict_id"] == "notes-legacy-shape"


def test_failed_personal_context_pull_does_not_advance_existing_cursor(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    service.store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id=DATASET_ID,
            device_id="device-a",
            domain=DOMAIN,
            last_pulled_sequence=7,
            max_delivered_sequence=7,
        )
    )

    response = _client(service).get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "domain": DOMAIN,
        },
    )

    assert response.status_code == 409
    cursor = service.store.get_device_cursor(DATASET_ID, "device-a", DOMAIN)
    assert cursor is not None
    assert cursor.last_pulled_sequence == 7
    assert cursor.max_delivered_sequence == 7


@pytest.mark.parametrize(
    "stored_update",
    [
        {"ongoing_sync_version": True},
        {"purge_generation": False},
        {"continuity_token": "stored_tampér_0123456789abcdef"},
        {"continuity_token": "stored_\ud800_0123456789abcdef"},
    ],
)
def test_tampered_non_integer_activation_state_fails_closed(
    tmp_path: Path,
    stored_update: dict[str, object],
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _set_personal_context_state(service, **stored_update)
    _insert_conflict(
        service,
        conflict_id="strict-stored-state",
        domain=DOMAIN,
        with_source=False,
    )

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "domain": DOMAIN,
            **_proof_query(EXCHANGE),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == (
        "personal_context_activation_required"
    )


def test_tampered_non_mapping_activation_state_fails_closed(tmp_path: Path) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    dataset = service.store.get_dataset(DATASET_ID)
    assert dataset is not None
    metadata = dict(dataset.metadata)
    metadata["personal_context"] = "tampered-non-mapping-state"
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(metadata, sort_keys=True), DATASET_ID),
            connection=connection,
        )
    _insert_conflict(
        service,
        conflict_id="non-mapping-stored-state",
        domain=DOMAIN,
        with_source=False,
    )

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "domain": DOMAIN,
            **_proof_query(EXCHANGE),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "error_code": "personal_context_activation_required",
        "message": "An active Personal Context exchange is required.",
    }
    assert "tampered-non-mapping-state" not in response.text


@pytest.mark.parametrize("field", ["activation_epoch", "continuity_token"])
@pytest.mark.parametrize(
    "malformed_value",
    ["malformed_tampér_0123456789abcdef", "malformed_\ud800_0123456789abcdef"],
)
def test_exact_malformed_stored_and_supplied_proof_fails_closed(
    tmp_path: Path,
    field: Literal["activation_epoch", "continuity_token"],
    malformed_value: str,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _set_personal_context_state(service, **{field: malformed_value})
    supplied_values = EXCHANGE.model_dump(mode="python")
    supplied_values[field] = malformed_value
    supplied = PersonalContextExchangeProof.model_construct(**supplied_values)
    dataset = service.store.get_dataset(DATASET_ID)
    assert dataset is not None

    with pytest.raises(
        SyncStoreError,
        match="^personal_context_activation_required$",
    ):
        service.require_active_exchange(
            dataset=dataset,
            user_id="user-a",
            device_id="device-a",
            exchange=supplied,
        )


def test_versioned_pull_uses_only_the_verified_exchange_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    device = service._require_registered_device("user-a", "device-a")
    cursor = service._encode_pull_token(
        dataset_id=DATASET_ID,
        device_id="device-a",
        version_set=service._pull_version_set(device),
        watermarks={(DOMAIN, 1): 0},
    )

    def fail_duplicate_validator(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("versioned pull revalidated untrusted request fields")

    monkeypatch.setattr(
        service_module,
        "_personal_context_exchange_is_active",
        fail_duplicate_validator,
        raising=False,
    )
    response = _client(service).get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": DATASET_ID,
            "device_id": "device-a",
            "domain": DOMAIN,
            "cursor": cursor,
            **_proof_query(EXCHANGE),
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["personal_context_exchange"] == EXCHANGE.model_dump(
        mode="json"
    )


@pytest.mark.parametrize("operation", ["push", "pull", "conflict_resolve"])
def test_mixed_selected_personal_context_work_gates_before_notes_effects(
    tmp_path: Path,
    operation: Literal["push", "pull", "conflict_resolve"],
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    client = _client(service)
    if operation == "push":
        response = client.post(
            "/api/v1/sync/push",
            json={
                "dataset_id": DATASET_ID,
                "device_id": "device-a",
                "envelopes": [
                    {
                        "dataset_id": DATASET_ID,
                        "client_envelope_id": "mixed-note",
                        "device_id": "device-a",
                        "domain": "notes.note",
                        "operation": "upsert",
                        "object_id": "mixed-note",
                        "payload": {"title": "Mixed", "content": "must not apply"},
                        "payload_hash": "sha256:mixed-note",
                    },
                    asdict(
                        _envelope(
                            preference_record(
                                record_id="mixed-record",
                                version_id="mixed-record-v1",
                            ).model_dump(mode="json"),
                            client_envelope_id="mixed-personal-context",
                        )
                    ),
                ],
            },
        )
        assert service.store.get_envelope_by_client_id(DATASET_ID, "mixed-note") is None
    elif operation == "pull":
        service.store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=DATASET_ID,
                client_envelope_id="mixed-note-existing",
                device_id="device-b",
                domain="notes.note",
                operation="upsert",
                object_id="mixed-note-existing",
                payload={"title": "Mixed", "content": "must not deliver"},
                payload_hash="sha256:mixed-note-existing",
            )
        )
        response = client.get(
            "/api/v1/sync/pull",
            params=[
                ("dataset_id", DATASET_ID),
                ("device_id", "device-a"),
                ("domain", "notes.note"),
                ("domain", DOMAIN),
            ],
        )
        assert service.store.get_device_cursor(DATASET_ID, "device-a", "notes.note") is None
    else:
        _insert_conflict(
            service,
            conflict_id="mixed-resolve-note",
            domain="notes.note",
            with_source=False,
        )
        _insert_conflict(
            service,
            conflict_id="mixed-resolve-personal",
            domain=DOMAIN,
            with_source=False,
        )
        response = client.post(
            "/api/v1/sync/conflicts/resolve",
            json={
                "dataset_id": DATASET_ID,
                "device_id": "device-a",
                "resolutions": [
                    {"conflict_id": "mixed-resolve-note", "action": "skip"},
                    {"conflict_id": "mixed-resolve-personal", "action": "skip"},
                ],
            },
        )
        assert service.store.get_conflict("mixed-resolve-note").status == "unresolved"

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "personal_context_activation_required"
    )


def test_mixed_conflict_page_with_personal_context_is_gated_without_leakage(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    _insert_conflict(
        service,
        conflict_id="mixed-page-note",
        domain="notes.note",
        with_source=False,
    )
    _insert_conflict(
        service,
        conflict_id="mixed-page-personal",
        domain=DOMAIN,
        with_source=False,
    )

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": DATASET_ID},
    )

    assert response.status_code == 409
    assert "mixed-page-personal" not in response.text
    assert "private_marker" not in response.text


def test_conflict_offset_checks_proof_for_the_selected_page(tmp_path: Path) -> None:
    """Paging past public notes never bypasses the Personal Context proof gate."""

    service, _target, _sqlite_path = _service(tmp_path)
    for conflict_id, domain in (("a-note", "notes.note"), ("b-personal", DOMAIN)):
        _insert_conflict(
            service, conflict_id=conflict_id, domain=domain, with_source=False
        )
    client = _client(service)
    first = client.get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": DATASET_ID, "limit": 1, "offset": 0},
    )
    second = client.get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": DATASET_ID, "limit": 1, "offset": 1},
    )

    assert first.status_code == 200
    assert [item["conflict_id"] for item in first.json()] == ["a-note"]
    assert second.status_code == 409
    assert "b-personal" not in second.text
    assert "private_marker" not in second.text


def test_conflict_domain_query_rejects_unknown_domain_at_boundary(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)

    response = _client(service).get(
        "/api/v1/sync/conflicts",
        params={"dataset_id": DATASET_ID, "domain": "not.a.sync.domain"},
    )

    assert response.status_code == 422


def test_conflict_store_filter_order_limit_and_offset_are_honest(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    for conflict_id, domain in (
        ("a-note", "notes.note"),
        ("b-personal", DOMAIN),
        ("c-note", "notes.note"),
    ):
        _insert_conflict(
            service,
            conflict_id=conflict_id,
            domain=domain,
            with_source=False,
        )

    page = service.store.list_conflicts(
        DATASET_ID,
        domain="notes.note",
        limit=1,
        offset=1,
    )

    assert [conflict.conflict_id for conflict in page] == ["c-note"]
    with pytest.raises(SyncStoreError, match="offset requires a limit"):
        service.store.list_conflicts(DATASET_ID, offset=1)
