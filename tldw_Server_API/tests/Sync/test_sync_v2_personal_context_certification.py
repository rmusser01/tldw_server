from __future__ import annotations

import base64
import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import (
    get_personal_context_service,
    personal_context_service_for_user,
)
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.api.v1.endpoints.personal_context import (
    router as personal_context_router,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    reset_managed_sqlite_backends,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import factory as sync_v2_factory
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDatasetCreate,
)
from tldw_Server_API.app.core.Sync.v2.profile import PersonalContextBootstrapError
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    preference_record,
)

pytestmark = pytest.mark.unit

_USER_ID = "13172"
_DEVICE_ID = "certification-device"
_AUTHORITY_ERROR = "personal_context_authority_mismatch"
_PLAINTEXT_CANARY = "TASK13172-PLAIN-7e9b86d4"
_INGRESS_CANARY = "TASK13172-INGRESS-3c67a895"
_KEY_CANARY = b"TASK13172-KEY-55db8893-BOUNDARY!"
_DIAGNOSTIC_CANARY = "TASK13172-DIAGNOSTIC-47a169be"
_EXCHANGE = {
    "ongoing_sync_version": 1,
    "activation_epoch": "epoch_13172certification",
    "continuity_token": "continuity_13172certification",
}


def _clear_factory_caches() -> None:
    for cached_factory in (
        sync_v2_factory._sync_v2_store_for_user,
        sync_v2_factory._chacha_notes_db_for_user,
        sync_v2_factory._sync_v2_blob_store_for_user,
        sync_v2_factory._personal_context_service_for_user,
    ):
        cached_factory.cache_clear()


@pytest.fixture()
def production_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Use the production factories over isolated, durable per-user files."""

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        base64.b64encode(_KEY_CANARY).decode("ascii"),
    )
    monkeypatch.setenv("SYNC_V2_AT_REST_ENCRYPTION_MODE", "managed_storage")
    monkeypatch.setenv("SYNC_V2_SERVER_TRUSTED_ENABLED", "true")
    monkeypatch.setenv(
        "SYNC_V2_PULL_TOKEN_SIGNING_SECRET",
        "task-13172-certification-signing-secret",
    )
    monkeypatch.setenv("SYNC_V2_BLOB_STORE_PATH", str(tmp_path / "sync_blobs"))
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    _clear_factory_caches()
    try:
        yield (
            personal_context_service_for_user(_USER_ID),
            sync_v2_factory.sync_v2_service_for_user(_USER_ID),
        )
    finally:
        _clear_factory_caches()


def _production_client() -> TestClient:
    """Compose both public routers with authenticated production factories."""

    app = FastAPI()
    app.include_router(personal_context_router, prefix="/api/v1/personal-context")
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = lambda: User(
        id=_USER_ID, username="task-13172-certification"
    )
    app.dependency_overrides[get_personal_context_service] = (
        lambda: personal_context_service_for_user(_USER_ID)
    )
    def sync_service():
        service = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
        service._recovery_clock_ns = lambda: 0
        return service

    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = sync_service
    return TestClient(app)


def _device_payload(public_key: rsa.RSAPublicKey) -> dict[str, object]:
    return {
        "device_id": _DEVICE_ID,
        "display_name": "TASK-13172 certification device",
        "client_type": "chatbook",
        "supported_domains": list(PERSONAL_CONTEXT_SYNC_DOMAINS),
        "supported_adapter_versions": {
            domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        },
        "capabilities": {
            "personal_context_wrapping_public_key": public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii")
        },
    }


def _seed_exchange(service, dataset_id: str) -> None:
    dataset = service.store.get_dataset(dataset_id, owner_user_id=_USER_ID)
    assert dataset is not None
    metadata = dict(dataset.metadata)
    metadata["personal_context"] = {
        **metadata["personal_context"],
        **_EXCHANGE,
    }
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(metadata, sort_keys=True), dataset_id),
            connection=connection,
        )


def _record_body(scope_id: str, value: str) -> dict[str, object]:
    return {
        "scope_id": scope_id,
        "payload": {
            "schema_version": 1,
            "kind": "preference",
            "subject": "response.certification",
            "polarity": "like",
            "value": value,
        },
        "semantic_key": {
            "namespace": "preference",
            "subject": "response.certification",
        },
        "controls": {
            "sync_mode": "syncable",
            "agent_visibility": "agent_visible",
        },
    }


def _publication_state(canonical) -> list[dict[str, object]]:
    with canonical._repository.database.transaction() as connection:
        return [
            dict(row)
            for row in connection.execute(
                """SELECT b.profile_publication_sequence, b.status,
                          r.batch_ordinal, r.role, r.row_state,
                          r.sync_server_cursor
                     FROM personal_context_publication_batches AS b
                     JOIN personal_context_publication_rows AS r
                       USING (profile_id, profile_publication_sequence)
                    ORDER BY b.profile_publication_sequence, r.batch_ordinal"""
            ).fetchall()
        ]


def _pull_params(dataset_id: str, cursor: str | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "dataset_id": dataset_id,
        "device_id": _DEVICE_ID,
        "domain": "personal_context.record",
        "personal_context_activation_epoch": _EXCHANGE["activation_epoch"],
        "personal_context_continuity_token": _EXCHANGE["continuity_token"],
    }
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _dataset_digest(service) -> str:
    rows = service.store.db.execute(
        """SELECT dataset_id, owner_user_id, domain_set_json, metadata_json, archived_at
             FROM sync_datasets
            ORDER BY dataset_id"""
    ).rows
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _transport_counts(service) -> dict[str, int]:
    return {
        table: int(
            service.store.db.execute(f"SELECT COUNT(*) AS count FROM {table}").rows[0][
                "count"
            ]
        )
        for table in (
            "sync_key_records",
            "sync_envelopes",
            "sync_device_cursors",
            "sync_personal_context_link_receipts",
        )
    }


def _new_dataset(
    service,
    dataset_id: str,
    *,
    default_personal: bool = False,
):
    metadata = (
        {"default_personal": True, "client_family": "chatbook"}
        if default_personal
        else {}
    )
    return service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset_id,
            owner_user_id=_USER_ID,
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
            metadata=metadata,
        )
    )


def _bind_dataset(service, canonical, dataset_id: str):
    manifest = canonical.get_manifest()
    integrity_key_id, _integrity_key = canonical.sync_integrity_key(
        manifest.profile_id
    )
    dataset = service.store.get_dataset(dataset_id, owner_user_id=_USER_ID)
    assert dataset is not None
    existing = dataset.metadata.get("personal_context")
    return service.store.bind_personal_context_dataset(
        dataset_id=dataset_id,
        user_id=_USER_ID,
        expected_binding=existing if isinstance(existing, dict) else None,
        profile_id=manifest.profile_id,
        authority_id="tldw-server",
        integrity_key_id=integrity_key_id,
        purge_generation=manifest.purge_generation,
        link_state="bootstrap_pending",
    )


def _binding_values(canonical) -> dict[str, object]:
    manifest = canonical.get_manifest()
    integrity_key_id, _integrity_key = canonical.sync_integrity_key(
        manifest.profile_id
    )
    return {
        "user_id": _USER_ID,
        "expected_binding": None,
        "profile_id": manifest.profile_id,
        "authority_id": "tldw-server",
        "integrity_key_id": integrity_key_id,
        "purge_generation": manifest.purge_generation,
        "link_state": "bootstrap_pending",
    }


def _register_device(service) -> None:
    public_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    ).public_key()
    service.register_device(
        user_id=_USER_ID,
        device_id=_DEVICE_ID,
        display_name="Certification device",
        client_type="chatbook",
        capabilities={
            "requested_domains": list(PERSONAL_CONTEXT_SYNC_DOMAINS),
            "supported_domains": list(PERSONAL_CONTEXT_SYNC_DOMAINS),
            "supported_adapter_versions": {
                domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            },
            "personal_context_wrapping_public_key": public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("ascii"),
        },
    )


def test_bootstrap_rejects_second_authoritative_dataset_before_side_effects(
    production_factories,
) -> None:
    """One profile cannot be bound into a second active Sync dataset."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    _new_dataset(service, "authoritative-dataset-a")
    first = _bind_dataset(service, canonical, "authoritative-dataset-a")

    same_before = _dataset_digest(service)
    repeated = _bind_dataset(service, canonical, first.dataset_id)
    assert repeated == first
    assert _dataset_digest(service) == same_before

    _new_dataset(service, "authoritative-dataset-b", default_personal=True)
    _register_device(service)
    counts_before = _transport_counts(service)
    datasets_before = _dataset_digest(service)
    wrapped = 0
    actual_wrapper = service.personal_context_key_wrapper
    assert actual_wrapper is not None

    def record_wrap(**kwargs: object) -> str:
        nonlocal wrapped
        wrapped += 1
        return actual_wrapper(**kwargs)

    service.personal_context_key_wrapper = record_wrap
    reason_code = None
    try:
        service.bootstrap_personal_context(
            user_id=_USER_ID,
            device_id=_DEVICE_ID,
            required_schema_version=1,
        )
    except PersonalContextBootstrapError as exc:
        reason_code = exc.reason_code

    assert reason_code == _AUTHORITY_ERROR
    assert wrapped == 0
    assert _transport_counts(service) == counts_before
    assert _dataset_digest(service) == datasets_before


def test_runtime_lookup_fails_closed_on_legacy_multiple_authoritative_datasets(
    production_factories,
) -> None:
    """Legacy duplicate bindings cannot be resolved by arbitrary row order."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    first = _new_dataset(service, "legacy-authority-a", default_personal=True)
    bound = _bind_dataset(service, canonical, first.dataset_id)
    second = _new_dataset(service, "legacy-authority-b", default_personal=True)
    corrupted_metadata = {**second.metadata, "personal_context": bound.metadata["personal_context"]}
    corrupted_domains = list(
        dict.fromkeys([*second.domains, *PERSONAL_CONTEXT_SYNC_DOMAINS])
    )
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            """UPDATE sync_datasets
                  SET domain_set_json = ?, metadata_json = ?
                WHERE dataset_id = ? AND owner_user_id = ?""",
            (
                json.dumps(corrupted_domains, sort_keys=True),
                json.dumps(corrupted_metadata, sort_keys=True),
                second.dataset_id,
                _USER_ID,
            ),
            connection=connection,
        )
    before = _dataset_digest(service)

    _clear_factory_caches()
    restarted = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
    selected = None
    reason_code = None
    try:
        selected = restarted.store.personal_context_dataset_for_profile(
            user_id=_USER_ID,
            profile_id=canonical.get_manifest().profile_id,
        )
    except SyncStoreError as exc:
        reason_code = str(exc)

    assert selected is None
    assert reason_code == _AUTHORITY_ERROR
    assert _dataset_digest(restarted) == before


def test_concurrent_sqlite_binds_choose_only_one_existing_dataset(
    production_factories,
) -> None:
    """Both candidate rows exist before racing into the authoritative DB fence."""

    canonical, first_service = production_factories
    canonical.create_profile(runtime_enabled=False)
    _new_dataset(first_service, "concurrent-authority-a")
    _new_dataset(first_service, "concurrent-authority-b")
    _clear_factory_caches()
    second_service = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
    barrier = Barrier(2)
    binding = _binding_values(canonical)

    def bind(service, dataset_id: str) -> str:
        barrier.wait()
        try:
            service.store.bind_personal_context_dataset(
                dataset_id=dataset_id,
                **binding,
            )
        except SyncStoreError as exc:
            return str(exc)
        return "bound"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(
            (
                executor.submit(bind, first_service, "concurrent-authority-a"),
                executor.submit(bind, second_service, "concurrent-authority-b"),
            ),
            key=id,
        )
        results = sorted(future.result() for future in outcomes)

    assert results == sorted([_AUTHORITY_ERROR, "bound"])
    active = [
        dataset
        for dataset in first_service.store.list_datasets_for_user(_USER_ID)
        if dataset.metadata.get("personal_context") is not None
    ]
    assert len(active) == 1


class _PostgresSiblingBindingBackend:
    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None, Any]] = []
        self.rows = [
            self._row("pg-authority-a", bound=True),
            self._row("pg-authority-b", bound=False),
        ]

    @staticmethod
    def _row(dataset_id: str, *, bound: bool) -> dict[str, object]:
        metadata: dict[str, object] = {}
        if bound:
            metadata["personal_context"] = {
                "profile_id": "profile-1",
                "authority_id": "tldw-server",
                "integrity_key_id": "integrity-key-1",
                "purge_generation": 0,
                "link_state": "bootstrap_pending",
            }
        return {
            "dataset_id": dataset_id,
            "owner_user_id": "user-1",
            "scope_type": "personal",
            "encryption_policy": "server_trusted_v1",
            "domain_set_json": json.dumps(["notes.note"]),
            "metadata_json": json.dumps(metadata),
            "workspace_id": None,
            "created_at": "2026-09-04T00:00:00+00:00",
            "updated_at": "2026-09-04T00:00:00+00:00",
            "archived_at": None,
        }

    @contextmanager
    def transaction(self, connection=None):
        yield connection or object()

    def execute(
        self,
        statement: str,
        params: tuple[Any, ...] | None = None,
        connection: Any = None,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params, connection))
        if normalized.startswith("SELECT * FROM sync_datasets"):
            if params == ("user-1",):
                return QueryResult(rows=[dict(row) for row in self.rows], rowcount=2)
            return QueryResult(rows=[dict(self.rows[1])], rowcount=1)
        if normalized.startswith("UPDATE sync_datasets"):
            return QueryResult(rows=[], rowcount=1)
        return QueryResult(rows=[], rowcount=1)


def test_postgres_bind_locks_all_existing_owner_rows_before_rejecting_sibling() -> None:
    """The parameterized PG lock observes an interleaved sibling before mutation."""

    backend = _PostgresSiblingBindingBackend()
    database = SyncDatabase.__new__(SyncDatabase)
    database.backend = cast(Any, backend)

    with pytest.raises(SyncStoreError, match=_AUTHORITY_ERROR):
        database.bind_personal_context_dataset(
            dataset_id="pg-authority-b",
            user_id="user-1",
            expected_binding=None,
            profile_id="profile-1",
            authority_id="tldw-server",
            integrity_key_id="integrity-key-1",
            purge_generation=0,
            link_state="bootstrap_pending",
        )

    statements = [statement for statement, _params, _connection in backend.calls]
    lock = next(
        (statement, params)
        for statement, params, _connection in backend.calls
        if statement.startswith("SELECT * FROM sync_datasets")
    )
    assert lock[0].endswith("ORDER BY dataset_id FOR UPDATE")
    assert lock[1] == ("user-1",)
    assert not any(statement.startswith("UPDATE sync_datasets") for statement in statements)


def test_production_http_relay_debt_survives_restart_and_recovers_on_push_and_pull(
    production_factories,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Certify the canonical HTTP -> durable journal -> Sync egress lifecycle."""

    initial_canonical, initial_sync = production_factories
    initial_backend = initial_sync.store.db.backend
    log_messages: list[str] = []
    sink_id = logger.add(lambda message: log_messages.append(str(message)))
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    responses: list[str] = []

    try:
        with _production_client() as client:
            capabilities = client.get("/api/v1/sync/capabilities")
            responses.append(capabilities.text)
            assert capabilities.status_code == 200
            assert capabilities.json()["personal_context"]["ongoing_sync_version"] == 0
            assert client.post(
                "/api/v1/sync/devices/register",
                json=_device_payload(private_key.public_key()),
            ).status_code == 200
            bootstrap = client.post(
                "/api/v1/sync/personal-context/bootstrap",
                json={"device_id": _DEVICE_ID, "required_schema_version": 1},
            )
            responses.append(bootstrap.text)
            assert bootstrap.status_code == 200, bootstrap.text
            boot = bootstrap.json()
            wrapped_blob = boot["wrapped_key_blob"]
            dataset_id = boot["dataset_id"]
            profile_id = boot["manifest"]["profile_id"]
            integrity_key = initial_canonical._repository.sync_integrity_key(profile_id)[1]
            wrapped = base64.urlsafe_b64decode(wrapped_blob.split(":", 1)[1])
            assert private_key.decrypt(
                wrapped,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=f"personal-context:{boot['integrity_key_id']}".encode(),
                ),
            ) == integrity_key
            complete = client.post(
                "/api/v1/sync/personal-context/complete",
                json={
                    "device_id": _DEVICE_ID,
                    "dataset_id": dataset_id,
                    "bootstrap_cursor": boot["cursor"],
                },
            )
            assert complete.status_code == 204, complete.text
            _seed_exchange(initial_sync, dataset_id)
            assert client.get("/api/v1/sync/capabilities").json()["personal_context"][
                "ongoing_sync_version"
            ] == 0

            scope_id = client.get("/api/v1/personal-context/scopes").json()["items"][0][
                "scope_id"
            ]
            before_direct = _publication_state(initial_canonical)
            created_response = client.post(
                "/api/v1/personal-context/records",
                json=_record_body(scope_id, _PLAINTEXT_CANARY),
            )
            responses.append(created_response.text)
            assert created_response.status_code == 201, created_response.text
            created = created_response.json()
            updated_response = client.patch(
                f"/api/v1/personal-context/records/{created['record_id']}",
                json={
                    "expected_version_id": created["version_id"],
                    "payload": _record_body(scope_id, _PLAINTEXT_CANARY + "-updated")[
                        "payload"
                    ],
                },
            )
            responses.append(updated_response.text)
            assert updated_response.status_code == 200, updated_response.text
            updated = updated_response.json()
            direct_rows = _publication_state(initial_canonical)[len(before_direct) :]
            direct_sequences = sorted({row["profile_publication_sequence"] for row in direct_rows})
            assert len(direct_sequences) == 2
            for sequence in direct_sequences:
                rows = [row for row in direct_rows if row["profile_publication_sequence"] == sequence]
                assert [(row["batch_ordinal"], row["role"]) for row in rows] == [
                    (0, "semantic"),
                    (1, "manifest"),
                ]
                assert len({row["status"] for row in rows}) == 1

            relay_type = type(initial_sync.personal_context_relay)
            original_relay = relay_type.relay_profile

            def deterministic_relay(relay, **kwargs):
                relay.clock_ns = lambda: 0
                return original_relay(relay, **kwargs)

            monkeypatch.setattr(relay_type, "relay_profile", deterministic_relay)
            drained = initial_sync.personal_context_relay.relay_profile(
                user_id=_USER_ID,
                profile_id=profile_id,
                dataset_id=dataset_id,
                after_server_cursor=None,
                wall_time_ms=10_000,
            )
            assert drained.continuation == "complete"
            assert drained.inspected_rows <= 100
            assert all(
                row["status"] == "complete"
                for row in _publication_state(initial_canonical)
                if row["profile_publication_sequence"] in direct_sequences
            )

            def fail_after_commit(_relay, **_kwargs):
                raise RuntimeError(_DIAGNOSTIC_CANARY)

            with monkeypatch.context() as fault:
                fault.setattr(relay_type, "relay_profile", fail_after_commit)
                accepted = client.patch(
                    f"/api/v1/personal-context/records/{created['record_id']}",
                    json={
                        "expected_version_id": updated["version_id"],
                        "payload": _record_body(scope_id, _PLAINTEXT_CANARY + "-debt")[
                            "payload"
                        ],
                    },
                )
            responses.append(accepted.text)
            assert accepted.status_code == 200, accepted.text
            pending = _publication_state(initial_canonical)
            pending_sequence = max(int(row["profile_publication_sequence"]) for row in pending)
            pending_rows = [
                row for row in pending if row["profile_publication_sequence"] == pending_sequence
            ]
            assert [(row["role"], row["row_state"]) for row in pending_rows] == [
                ("semantic", "pending"),
                ("manifest", "pending"),
            ]
            assert {row["status"] for row in pending_rows} == {"pending"}
            assert {row["sync_server_cursor"] for row in pending_rows} == {None}
            assert not initial_sync.store.db.execute(
                """SELECT server_sequence FROM sync_envelopes
                    WHERE routing_metadata_json LIKE ?""",
                (f'%"profile_publication_sequence":{pending_sequence}%',),
            ).rows

        reset_managed_sqlite_backends(backends=[initial_backend])
        _clear_factory_caches()
        restarted_canonical = personal_context_service_for_user(_USER_ID)
        restarted_sync = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
        assert restarted_canonical is not initial_canonical
        assert restarted_sync is not initial_sync
        assert restarted_sync.store.db.backend is not initial_backend
        assert _publication_state(restarted_canonical) == pending

        with _production_client() as restarted_client:
            ingress_payload = {
                **preference_record(
                    profile_id,
                    record_id="task-13172-ingress-record",
                    version_id="task-13172-ingress-v1",
                    value=_INGRESS_CANARY,
                ).model_dump(mode="json"),
                "scope_id": scope_id,
            }
            canonical_bytes = canonical_json_bytes(ingress_payload)
            tag = hmac.new(integrity_key, canonical_bytes, hashlib.sha256).hexdigest()
            push = restarted_client.post(
                "/api/v1/sync/push",
                json={
                    "dataset_id": dataset_id,
                    "device_id": _DEVICE_ID,
                    "personal_context_exchange": _EXCHANGE,
                    "envelopes": [
                        {
                            "dataset_id": dataset_id,
                            "client_envelope_id": "task-13172-client-ingress",
                            "device_id": _DEVICE_ID,
                            "domain": "personal_context.record",
                            "operation": "upsert",
                            "object_id": ingress_payload["record_id"],
                            "parent_id": scope_id,
                            "adapter_version": 1,
                            "schema_version": 1,
                            "payload": ingress_payload,
                            "payload_hash": f"hmac-sha256-v1:{tag}",
                            "payload_size_bytes": len(canonical_bytes),
                            "entity_version": ingress_payload["version_id"],
                            "routing_metadata": {
                                "integrity_key_id": boot["integrity_key_id"],
                                "profile_id": profile_id,
                                "purge_generation": boot["purge_generation"],
                            },
                            "encryption_metadata": {"policy": "server_trusted_v1"},
                        }
                    ],
                },
            )
            responses.append(push.text)
            assert push.status_code == 200, push.text
            assert [item["client_envelope_id"] for item in push.json()["accepted"]] == [
                "task-13172-client-ingress"
            ]
            assert not push.json()["rejected"]
            after_push = _publication_state(restarted_canonical)
            recovered_pending = [
                row
                for row in after_push
                if row["profile_publication_sequence"] == pending_sequence
            ]
            assert {row["status"] for row in recovered_pending} == {"complete"}
            assert {row["row_state"] for row in recovered_pending} == {"acknowledged"}

            ingress_sync = restarted_sync.store.db.execute(
                """SELECT e.server_sequence, e.apply_status, e.routing_metadata_json,
                          e.payload_json, e.payload_clear_json, e.payload_ciphertext,
                          r.*
                     FROM sync_envelopes AS e
                     JOIN sync_personal_context_ingress_receipts AS r
                       ON r.server_sequence = e.server_sequence
                    WHERE e.dataset_id = ? AND e.client_envelope_id = ?""",
                (dataset_id, "task-13172-client-ingress"),
            ).rows
            assert len(ingress_sync) == 1
            ingress_row = ingress_sync[0]
            assert ingress_row["apply_status"] == "applied"
            assert json.loads(ingress_row["routing_metadata_json"])[
                "personal_context_authority"
            ] == {"role": "client_ingress"}
            assert ingress_row["payload_json"] == "{}"
            assert ingress_row["payload_clear_json"] == "{}"
            assert ingress_row["payload_ciphertext"]
            with restarted_canonical._repository.database.transaction() as connection:
                canonical_receipt = dict(
                    connection.execute(
                        """SELECT * FROM personal_context_ingress_receipts
                            WHERE dataset_id = ? AND device_id = ?
                              AND client_envelope_id = ?""",
                        (dataset_id, _DEVICE_ID, "task-13172-client-ingress"),
                    ).fetchone()
                )
            for canonical_name, sync_name in (
                ("canonical_payload_digest", "canonical_payload_digest"),
                ("resulting_object_id", "resulting_object_id"),
                ("publication_batch_id", "publication_batch_id"),
                ("profile_publication_sequence", "profile_publication_sequence"),
                ("receipt_id", "receipt_id"),
                ("wire_entity_version", "wire_entity_version"),
            ):
                assert canonical_receipt[canonical_name] == ingress_row[sync_name]

            current_direct = restarted_client.get(
                f"/api/v1/personal-context/records/{created['record_id']}"
            ).json()
            with monkeypatch.context() as fault:
                fault.setattr(relay_type, "relay_profile", fail_after_commit)
                second_debt = restarted_client.patch(
                    f"/api/v1/personal-context/records/{created['record_id']}",
                    json={
                        "expected_version_id": current_direct["version_id"],
                        "payload": _record_body(
                            scope_id, _PLAINTEXT_CANARY + "-pull-debt"
                        )["payload"],
                    },
                )
            assert second_debt.status_code == 200, second_debt.text
            second_pending = max(
                int(row["profile_publication_sequence"])
                for row in _publication_state(restarted_canonical)
            )

        second_backend = restarted_sync.store.db.backend
        reset_managed_sqlite_backends(backends=[second_backend])
        _clear_factory_caches()
        pull_canonical = personal_context_service_for_user(_USER_ID)
        pull_sync = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
        assert pull_canonical is not restarted_canonical
        assert pull_sync is not restarted_sync
        assert pull_sync.store.db.backend is not second_backend

        with _production_client() as pull_client:
            assert pull_client.get(
                "/api/v1/sync/pull",
                params={**_pull_params(dataset_id), "page_size": 0},
            ).status_code == 422
            pulled = pull_client.get("/api/v1/sync/pull", params=_pull_params(dataset_id))
            responses.append(pulled.text)
            assert pulled.status_code == 200, pulled.text
            pull_body = pulled.json()
            assert pull_body["personal_context_relay"]["state"] == "complete"
            assert {
                row["status"]
                for row in _publication_state(pull_canonical)
                if row["profile_publication_sequence"] == second_pending
            } == {"complete"}
            ingress_egress = [
                envelope
                for envelope in pull_body["envelopes"]
                if envelope["object_id"] == ingress_payload["record_id"]
            ]
            assert len(ingress_egress) == 1
            authority = ingress_egress[0]["routing_metadata"][
                "personal_context_authority"
            ]
            assert authority["role"] == "home_authority"
            assert ingress_egress[0]["payload"] == ingress_payload
            assert ingress_egress[0]["client_envelope_id"] != "task-13172-client-ingress"
            assert all(
                envelope["routing_metadata"]["personal_context_authority"]["role"]
                == "home_authority"
                for envelope in pull_body["envelopes"]
            )
            authority_rows = pull_sync.store.db.execute(
                """SELECT server_sequence, apply_status, routing_metadata_json
                     FROM sync_envelopes
                    WHERE dataset_id = ? AND domain = 'personal_context.record'
                      AND entity_id = ?
                    ORDER BY server_sequence""",
                (dataset_id, ingress_payload["record_id"]),
            ).rows
            home_rows = [
                row
                for row in authority_rows
                if json.loads(row["routing_metadata_json"])
                .get("personal_context_authority", {})
                .get("role")
                == "home_authority"
            ]
            assert len(home_rows) == 1
            assert home_rows[0]["apply_status"] == "applied"

            repeated = pull_client.get(
                "/api/v1/sync/pull",
                params=_pull_params(dataset_id, pull_body["next_cursor"]),
            )
            responses.append(repeated.text)
            assert repeated.status_code == 200, repeated.text
            assert repeated.json()["envelopes"] == []
            cursor = pull_sync.store.db.execute(
                """SELECT last_pulled_sequence FROM sync_device_cursors
                    WHERE dataset_id = ? AND device_id = ?
                      AND domain = 'personal_context.record'""",
                (dataset_id, _DEVICE_ID),
            ).rows[0]["last_pulled_sequence"]
            assert int(cursor) == int(repeated.json()["next_cursor"])

        unsafe_response_text = "\n".join(responses)
        assert _DIAGNOSTIC_CANARY not in unsafe_response_text
        assert _KEY_CANARY.decode() not in unsafe_response_text
        assert _DIAGNOSTIC_CANARY not in "".join(log_messages)
        assert _PLAINTEXT_CANARY not in "".join(log_messages)
        assert _INGRESS_CANARY not in "".join(log_messages)
        assert _KEY_CANARY.decode() not in "".join(log_messages)
        artifact_paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
        assert {"Personalization.db", "Sync_v2.db", "ChaChaNotes.db"}.issubset(
            {path.name for path in artifact_paths}
        )
        forbidden = (
            _PLAINTEXT_CANARY.encode(),
            _INGRESS_CANARY.encode(),
            _DIAGNOSTIC_CANARY.encode(),
            _KEY_CANARY,
        )
        for artifact in artifact_paths:
            contents = artifact.read_bytes()
            assert all(canary not in contents for canary in forbidden), artifact
            if not artifact.name.startswith("Sync_v2.db"):
                assert wrapped_blob.encode() not in contents, artifact
        assert any(
            wrapped_blob.encode() in artifact.read_bytes()
            for artifact in artifact_paths
            if artifact.name.startswith("Sync_v2.db")
        )
        assert relay_type.relay_profile is deterministic_relay
    finally:
        logger.remove(sink_id)
