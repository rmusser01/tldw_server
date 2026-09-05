"""Certify Personal Context using production factories and durable isolated databases.

Exercise authority races and restart recovery, including real PostgreSQL connections.
A child pytest process forces a diagnostic failure and checks that protected canaries
are absent from its captured output; fixture teardown retires managed database handles.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3

# Trusted child pytest process using an explicit argument list, never a shell.
import subprocess  # nosec B404
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient
from loguru import logger
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.api.v1.API_Deps import personal_context_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import (
    get_personal_context_service,
    personal_context_service_for_user,
)
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
    reset_managed_sqlite_backends,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
from tldw_Server_API.app.core.Sync.v2 import factory as sync_v2_factory
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDatasetCreate,
)
from tldw_Server_API.app.core.Sync.v2.profile import PersonalContextBootstrapError
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
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


def _content_digest(value: object) -> str:
    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    elif isinstance(value, (set, frozenset)):
        encoded = json.dumps(
            sorted(_content_digest(item) for item in value),
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, diagnostic: str) -> None:
    if not condition:
        pytest.fail(diagnostic, pytrace=False)


def _require_digest_equal(actual: object, expected: object, diagnostic: str) -> None:
    _require(
        hmac.compare_digest(_content_digest(actual), _content_digest(expected)),
        diagnostic,
    )


def _require_status(response: object, expected: int, diagnostic: str) -> None:
    _require(getattr(response, "status_code", None) == expected, diagnostic)


def _artifact_category(path: Path) -> str:
    name = path.name
    database_categories = {
        "Personalization.db": "personalization-db",
        "Sync_v2.db": "sync-db",
        "ChaChaNotes.db": "notes-db",
    }
    if name in database_categories:
        return database_categories[name]
    for database_name, prefix in (
        ("Personalization.db", "personalization"),
        ("Sync_v2.db", "sync"),
        ("ChaChaNotes.db", "notes"),
    ):
        if name == f"{database_name}-wal":
            return f"{prefix}-wal"
        if name == f"{database_name}-shm":
            return f"{prefix}-shm"
    return {
        "application.log": "application-log",
        "diagnostic.json": "diagnostic",
        "migration-snapshot.json": "migration-snapshot",
        "application-backup.db": "application-backup",
    }.get(name, "application-other")


def _scan_application_artifacts(
    root: Path,
    *,
    phase: str,
    wrapped_blob: str,
    records: list[dict[str, object]],
) -> None:
    """Scan application-custody artifacts without retaining protected contents."""

    forbidden = (
        _PLAINTEXT_CANARY.encode(),
        _INGRESS_CANARY.encode(),
        _DIAGNOSTIC_CANARY.encode(),
        _KEY_CANARY,
    )
    wrapped_bytes = wrapped_blob.encode()
    wrapped_allowed = {
        "sync-db",
        "sync-wal",
        "sync-shm",
        "application-backup",
    }
    for artifact in sorted(path for path in root.rglob("*") if path.is_file()):
        contents = artifact.read_bytes()
        category = _artifact_category(artifact)
        _require(
            all(canary not in contents for canary in forbidden),
            "protected canary escaped into an application artifact",
        )
        wrapped_present = wrapped_bytes in contents
        _require(
            not wrapped_present or category in wrapped_allowed,
            "wrapped key escaped its authorized database boundary",
        )
        records.append(
            {
                "phase": phase,
                "path": artifact.relative_to(root).as_posix(),
                "category": category,
                "custody": "application-owned-test-boundary",
                "size_bytes": len(contents),
                "plaintext": "absent",
                "client_ingress": "absent",
                "diagnostic_marker": "absent",
                "raw_key": "absent",
                "wrapped_key": (
                    "present-authorized-encrypted" if wrapped_present else "absent"
                ),
            }
        )


def _create_sqlite_backup(source: Path, destination: Path) -> None:
    """Create a controlled application-owned backup from an active SQLite DB."""

    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(destination) as destination_connection:
            source_connection.backup(destination_connection)


def _forced_unsafe_diagnostic_mismatch() -> None:
    protected = {
        "plaintext": os.environ["TASK13172_DIAGNOSTIC_PLAINTEXT"],
        "ingress": os.environ["TASK13172_DIAGNOSTIC_INGRESS"],
        "key": os.environ["TASK13172_DIAGNOSTIC_KEY"],
        "wrapped": os.environ["TASK13172_DIAGNOSTIC_WRAPPED"],
    }
    _require_digest_equal(
        protected,
        {"expected": False},
        "forced protected-value digest mismatch",
    )


def test_certification_failure_diagnostics_are_content_free(tmp_path: Path) -> None:
    """Even a forced mismatch must expose only a fixed diagnostic."""

    if os.environ.get("TASK13172_FORCE_DIAGNOSTIC_FAILURE") == "1":
        _forced_unsafe_diagnostic_mismatch()
        return
    wrapped_canary = "TASK13172-WRAPPED-82f49f75"
    environment = {
        **os.environ,
        "TASK13172_FORCE_DIAGNOSTIC_FAILURE": "1",
        "TASK13172_DIAGNOSTIC_PLAINTEXT": _PLAINTEXT_CANARY,
        "TASK13172_DIAGNOSTIC_INGRESS": _INGRESS_CANARY,
        "TASK13172_DIAGNOSTIC_KEY": _KEY_CANARY.decode("ascii"),
        "TASK13172_DIAGNOSTIC_WRAPPED": wrapped_canary,
    }
    # Current interpreter and fixed pytest argv; paths are trusted test fixtures.
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{__file__}::test_certification_failure_diagnostics_are_content_free",
            "--tb=short",
            f"--basetemp={tmp_path / 'child'}",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        pytest.fail("forced diagnostic mismatch unexpectedly passed", pytrace=False)
    output = result.stdout + result.stderr
    for canary in (
        _PLAINTEXT_CANARY,
        _INGRESS_CANARY,
        _KEY_CANARY.decode("ascii"),
        wrapped_canary,
    ):
        if canary in output:
            pytest.fail("certification diagnostic leaked protected value", pytrace=False)


def _clear_factory_caches() -> None:
    """Remove cached production dependencies between isolated certification runs."""

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
) -> Iterator[tuple[PersonalContextService, SyncV2Service]]:
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
    from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
    from tldw_Server_API.app.api.v1.router_registry import register_router_specs
    from tldw_Server_API.app.main import app as production_app

    sync_specs = [
        spec for spec in iter_core_router_specs() if spec.route_key == "sync"
    ]
    register_router_specs(production_app, sync_specs)

    trace = {"personal_context_factory": 0, "sync_factory": 0}
    managed_sync_backends: list[object] = []
    actual_personal_context_factory = (
        personal_context_deps.personal_context_service_for_user
    )
    actual_sync_factory = sync_v2_factory.sync_v2_service_for_user

    def track_sync_backend(service: SyncV2Service) -> SyncV2Service:
        """Track the managed backend for teardown and make recovery time deterministic."""

        backend = service.store.db.backend
        if not any(candidate is backend for candidate in managed_sync_backends):
            managed_sync_backends.append(backend)
        service._recovery_clock_ns = lambda: 0
        return service

    def traced_personal_context_factory(
        *args: object, **kwargs: object
    ) -> PersonalContextService:
        """Count calls while exercising the actual Personal Context factory."""

        trace["personal_context_factory"] += 1
        return actual_personal_context_factory(*args, **kwargs)

    def traced_sync_factory(user_id: str) -> SyncV2Service:
        """Count calls and track resources from the actual Sync factory."""

        trace["sync_factory"] += 1
        return track_sync_backend(actual_sync_factory(user_id))

    monkeypatch.setattr(
        personal_context_deps,
        "personal_context_service_for_user",
        traced_personal_context_factory,
    )
    monkeypatch.setattr(
        sync_v2_factory,
        "sync_v2_service_for_user",
        traced_sync_factory,
    )
    monkeypatch.setattr(sync_endpoint, "sync_v2_service_for_user", traced_sync_factory)
    try:
        canonical = personal_context_service_for_user(_USER_ID)
        sync = track_sync_backend(actual_sync_factory(_USER_ID))
        sync._certification_production_app = production_app
        sync._certification_factory_trace = trace
        yield canonical, sync
    finally:
        reset_managed_sqlite_backends(backends=managed_sync_backends)
        _require(
            all(getattr(backend, "_retired", False) for backend in managed_sync_backends),
            "managed Sync backend cleanup was incomplete",
        )
        _clear_factory_caches()
        _require(
            all(
                cached_factory.cache_info().currsize == 0
                for cached_factory in (
                    sync_v2_factory._sync_v2_store_for_user,
                    sync_v2_factory._chacha_notes_db_for_user,
                    sync_v2_factory._sync_v2_blob_store_for_user,
                    sync_v2_factory._personal_context_service_for_user,
                )
            ),
            "managed Sync factory cache cleanup was incomplete",
        )


@contextmanager
def _production_client(app: object):
    """Run the production app with only its authentication dependency replaced."""

    previous_auth = app.dependency_overrides.get(get_request_user)
    app.dependency_overrides[get_request_user] = lambda: User(
        id=int(_USER_ID), username="task-13172-certification"
    )
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        if previous_auth is None:
            app.dependency_overrides.pop(get_request_user, None)
        else:
            app.dependency_overrides[get_request_user] = previous_auth


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


def _seed_exchange(service: SyncV2Service, dataset_id: str) -> dict[str, object]:
    """Install and acknowledge through the real journals before test-only rollout."""

    baseline = service.prepare_personal_context_activation(user_id=_USER_ID, device_id=_DEVICE_ID)
    _receipt, proof = service.acknowledge_personal_context_activation(
        user_id=_USER_ID,
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        activation_id=baseline.activation.activation_id,
        baseline_digest=baseline.activation.baseline_digest,
        local_receipt_id="certification-local-install-0123456789",
        exchange=baseline.personal_context_exchange,
    )
    dataset = service.store.get_dataset(dataset_id, owner_user_id=_USER_ID)
    _require(dataset is not None, "seed dataset was not found")
    metadata = dict(dataset.metadata)
    metadata["personal_context"] = {
        **metadata["personal_context"],
        **proof.model_dump(),
    }
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
            (json.dumps(metadata, sort_keys=True), dataset_id),
            connection=connection,
        )
    return proof.model_dump()


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


def _pull_params(dataset_id: str, cursor: str | None = None, *, exchange: dict[str, object]) -> dict[str, object]:
    params: dict[str, object] = {
        "dataset_id": dataset_id,
        "device_id": _DEVICE_ID,
        "domain": "personal_context.record",
        "personal_context_activation_epoch": exchange["activation_epoch"],
        "personal_context_continuity_token": exchange["continuity_token"],
    }
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _dataset_digest(service) -> str:
    rows = service.store.db.execute(
        """SELECT * FROM sync_datasets
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


def _domain_state_digest(service) -> str:
    rows = service.store.db.execute(
        """SELECT * FROM sync_domain_state
            ORDER BY dataset_id, domain, adapter_version"""
    ).rows
    return _content_digest(rows)


def _corrupt_authority_target(service, dataset_id: str, defect: str) -> None:
    """Install one legacy-invalid target shape without exercising enrollment guards."""

    row = service.store.db.execute(
        "SELECT * FROM sync_datasets WHERE dataset_id = ?",
        (dataset_id,),
    ).rows[0]
    workspace_id = row["workspace_id"]
    scope_type = row["scope_type"]
    encryption_policy = row["encryption_policy"]
    metadata_json = row["metadata_json"]
    archived_at = row["archived_at"]
    if defect == "workspace":
        workspace_id = "workspace-collision"
        scope_type = "workspace"
    elif defect == "archived":
        archived_at = "2026-09-04T00:00:00+00:00"
    elif defect == "policy":
        encryption_policy = "client_managed_v1"
    elif defect == "default-marker":
        metadata_json = json.dumps(
            {"default_personal": True, "client_family": "not-chatbook"}
        )
    elif defect == "generation":
        metadata_json = json.dumps(
            {
                "default_personal": True,
                "client_family": "chatbook",
                "personal_context": {
                    "profile_id": "legacy-profile",
                    "authority_id": "tldw-server",
                    "integrity_key_id": "legacy-key",
                    "purge_generation": "invalid",
                    "link_state": "bootstrap_pending",
                },
            }
        )
    else:
        pytest.fail("unsupported certification target defect", pytrace=False)
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            """UPDATE sync_datasets
                  SET workspace_id = ?, scope_type = ?, encryption_policy = ?,
                      metadata_json = ?, archived_at = ?
                WHERE dataset_id = ?""",
            (
                workspace_id,
                scope_type,
                encryption_policy,
                metadata_json,
                archived_at,
                dataset_id,
            ),
            connection=connection,
        )


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
    _require(dataset is not None, "binding dataset was not found")
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


def test_bind_rejects_second_authoritative_dataset_before_side_effects(
    production_factories,
) -> None:
    """One profile cannot be bound into a second active Sync dataset."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    _new_dataset(service, "authoritative-dataset-a")
    first = _bind_dataset(service, canonical, "authoritative-dataset-a")

    same_before = _dataset_digest(service)
    repeated = _bind_dataset(service, canonical, first.dataset_id)
    _require_digest_equal(repeated, first, "idempotent binding changed")
    _require_digest_equal(
        _dataset_digest(service), same_before, "idempotent dataset state changed"
    )

    _new_dataset(service, "authoritative-dataset-b", default_personal=True)
    counts_before = _transport_counts(service)
    datasets_before = _dataset_digest(service)
    reason_code = None
    try:
        _bind_dataset(service, canonical, "authoritative-dataset-b")
    except SyncStoreError as exc:
        reason_code = str(exc)

    _require(reason_code == _AUTHORITY_ERROR, "second binding was not rejected")
    _require_digest_equal(
        _transport_counts(service), counts_before, "rejected binding changed transport"
    )
    _require_digest_equal(
        _dataset_digest(service), datasets_before, "rejected binding changed dataset"
    )


def test_bootstrap_reuses_existing_nondefault_authority_without_creating_default(
    production_factories,
) -> None:
    """Bootstrap resolves the sole bound authority before default selection."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    authority = _new_dataset(service, "existing-nondefault-authority")
    _bind_dataset(service, canonical, authority.dataset_id)
    _register_device(service)

    bootstrap = service.bootstrap_personal_context(
        user_id=_USER_ID,
        device_id=_DEVICE_ID,
        required_schema_version=1,
    )

    _require(
        bootstrap.dataset_id == authority.dataset_id,
        "bootstrap did not reuse the existing authority",
    )
    datasets = service.store.list_datasets_for_user(_USER_ID)
    _require_digest_equal(
        [dataset.dataset_id for dataset in datasets],
        [authority.dataset_id],
        "bootstrap created an unexpected dataset",
    )
    _require(
        not any(dataset.metadata.get("default_personal") for dataset in datasets),
        "bootstrap created an unexpected default",
    )


def test_bootstrap_rejects_ambiguous_unbound_defaults_before_side_effects(
    production_factories: tuple[Any, Any],
) -> None:
    """Duplicate default markers cannot silently choose a new authority."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    _new_dataset(service, "ambiguous-default-a", default_personal=True)
    _new_dataset(service, "ambiguous-default-b", default_personal=True)
    _register_device(service)
    datasets_before = _dataset_digest(service)
    domains_before = _domain_state_digest(service)
    transport_before = _transport_counts(service)
    wrapper = service.personal_context_key_wrapper
    wrapped = 0

    def record_wrap(**kwargs: object) -> str:
        nonlocal wrapped
        wrapped += 1
        return wrapper(**kwargs)

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

    _require(reason_code == _AUTHORITY_ERROR, "ambiguous defaults were not rejected")
    _require(wrapped == 0, "ambiguous defaults wrapped key material")
    _require_digest_equal(
        _dataset_digest(service), datasets_before, "ambiguous defaults changed datasets"
    )
    _require_digest_equal(
        _domain_state_digest(service), domains_before, "ambiguous defaults changed domains"
    )
    _require_digest_equal(
        _transport_counts(service), transport_before, "ambiguous defaults changed transport"
    )
    _bind_dataset(service, canonical, "ambiguous-default-b")
    bootstrap = service.bootstrap_personal_context(
        user_id=_USER_ID, device_id=_DEVICE_ID, required_schema_version=1
    )
    _require(
        bootstrap.dataset_id == "ambiguous-default-b",
        "sole established authority did not take precedence over defaults",
    )


@pytest.mark.parametrize(
    "defect",
    ("workspace", "archived", "policy", "default-marker", "generation"),
)
def test_bootstrap_rejects_invalid_deterministic_default_before_side_effects(
    production_factories,
    defect: str,
) -> None:
    """A colliding deterministic ID is never repaired into an authority target."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    dataset_id = f"ds_personal_{_USER_ID}"
    _new_dataset(service, dataset_id, default_personal=True)
    _corrupt_authority_target(service, dataset_id, defect)
    _register_device(service)
    datasets_before = _dataset_digest(service)
    domains_before = _domain_state_digest(service)
    transport_before = _transport_counts(service)
    actual_wrapper = service.personal_context_key_wrapper
    _require(actual_wrapper is not None, "key wrapper was unavailable")
    wrapped = 0

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

    _require(reason_code == _AUTHORITY_ERROR, "invalid default target was not rejected")
    _require(wrapped == 0, "invalid default target wrapped key material")
    _require_digest_equal(
        _dataset_digest(service), datasets_before, "invalid default changed dataset state"
    )
    _require_digest_equal(
        _domain_state_digest(service), domains_before, "invalid default changed domains"
    )
    _require_digest_equal(
        _transport_counts(service), transport_before, "invalid default changed transport"
    )


@pytest.mark.parametrize("defect", ("workspace", "archived", "policy", "generation"))
def test_direct_bind_rejects_invalid_authority_target_before_side_effects(
    production_factories,
    defect: str,
) -> None:
    """Direct binding applies the same fail-closed authority-target contract."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    dataset_id = "invalid-direct-authority"
    _new_dataset(service, dataset_id)
    _corrupt_authority_target(service, dataset_id, defect)
    datasets_before = _dataset_digest(service)
    domains_before = _domain_state_digest(service)
    transport_before = _transport_counts(service)
    reason_code = None
    try:
        _bind_dataset(service, canonical, dataset_id)
    except SyncStoreError as exc:
        reason_code = str(exc)

    _require(reason_code == _AUTHORITY_ERROR, "invalid direct target was not rejected")
    _require_digest_equal(
        _dataset_digest(service), datasets_before, "invalid bind changed dataset state"
    )
    _require_digest_equal(
        _domain_state_digest(service), domains_before, "invalid bind changed domains"
    )
    _require_digest_equal(
        _transport_counts(service), transport_before, "invalid bind changed transport"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("profile_id", 13172),
        ("authority_id", 13172),
        ("integrity_key_id", 13172),
        ("link_state", b"complete"),
        ("purge_generation", True),
        ("purge_generation", 0.5),
        ("purge_generation", "0"),
    ),
)
def test_direct_bind_rejects_malformed_values_before_transaction(
    production_factories,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Malformed binding scalars fail before transaction or durable mutation."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    dataset_id = "malformed-direct-binding"
    _new_dataset(service, dataset_id)
    datasets_before = _dataset_digest(service)
    domains_before = _domain_state_digest(service)
    transport_before = _transport_counts(service)
    binding = _binding_values(canonical)
    binding[field] = value

    @contextmanager
    def unexpected_transaction(*_args: object, **_kwargs: object):
        pytest.fail("malformed bind entered database transaction", pytrace=False)
        yield

    monkeypatch.setattr(
        service.store.db.backend,
        "transaction",
        unexpected_transaction,
    )
    reason_code = None
    try:
        service.store.bind_personal_context_dataset(
            dataset_id=dataset_id,
            **binding,
        )
    except SyncStoreError as exc:
        reason_code = str(exc)

    _require(reason_code == _AUTHORITY_ERROR, "malformed bind value was not rejected")
    _require_digest_equal(
        _dataset_digest(service), datasets_before, "malformed bind changed dataset state"
    )
    _require_digest_equal(
        _domain_state_digest(service), domains_before, "malformed bind changed domains"
    )
    _require_digest_equal(
        _transport_counts(service), transport_before, "malformed bind changed transport"
    )


def test_bootstrap_rolls_back_default_and_domains_after_interleaved_bind_rejection(
    production_factories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-selection sibling bind cannot strand bootstrap transport effects."""

    canonical, service = production_factories
    canonical.create_profile(runtime_enabled=False)
    sibling = _new_dataset(service, "interleaved-authority")
    _register_device(service)
    datasets_before = _dataset_digest(service)
    counts_before = _transport_counts(service)
    domain_states_before = service.store.db.execute(
        "SELECT * FROM sync_domain_state ORDER BY dataset_id, domain, adapter_version"
    ).rows
    bootstrap_canonical = service._personal_context_service_for_user(_USER_ID)
    actual_plan = bootstrap_canonical.plan_sync_bootstrap
    actual_wrapper = service.personal_context_key_wrapper
    _require(actual_wrapper is not None, "key wrapper was unavailable")
    wrapped = 0

    def interleaved_plan():
        snapshot = actual_plan()
        _bind_dataset(service, canonical, sibling.dataset_id)
        return snapshot

    def record_wrap(**kwargs: object) -> str:
        nonlocal wrapped
        wrapped += 1
        return actual_wrapper(**kwargs)

    monkeypatch.setattr(bootstrap_canonical, "plan_sync_bootstrap", interleaved_plan)
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

    _require(reason_code == _AUTHORITY_ERROR, "interleaved binding was not rejected")
    _require(wrapped == 0, "rejected bootstrap wrapped key material")
    _require_digest_equal(
        _dataset_digest(service), datasets_before, "rejected bootstrap changed datasets"
    )
    _require_digest_equal(
        _transport_counts(service), counts_before, "rejected bootstrap changed transport"
    )
    _require_digest_equal(
        service.store.db.execute(
            "SELECT * FROM sync_domain_state ORDER BY dataset_id, domain, adapter_version"
        ).rows,
        domain_states_before,
        "rejected bootstrap changed domain state",
    )


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

    _require(selected is None, "corrupt authority lookup selected a dataset")
    _require(reason_code == _AUTHORITY_ERROR, "corrupt authority lookup did not fail")
    _require_digest_equal(
        _dataset_digest(restarted), before, "corrupt lookup changed dataset state"
    )


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

    _require_digest_equal(
        results,
        sorted([_AUTHORITY_ERROR, "bound"]),
        "SQLite bind race did not choose exactly one authority",
    )
    active = [
        dataset
        for dataset in first_service.store.list_datasets_for_user(_USER_ID)
        if dataset.metadata.get("personal_context") is not None
    ]
    _require(len(active) == 1, "SQLite bind race persisted multiple authorities")


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
    _require(
        lock[0].endswith("ORDER BY dataset_id FOR UPDATE"),
        "PostgreSQL binding query omitted deterministic row locks",
    )
    _require_digest_equal(lock[1], ("user-1",), "PostgreSQL lock parameters changed")
    _require(
        not any(statement.startswith("UPDATE sync_datasets") for statement in statements),
        "PostgreSQL sibling rejection mutated dataset state",
    )


@pytest.mark.integration
def test_postgres_two_connections_choose_exactly_one_existing_authority(
    pg_database_config: DatabaseConfig,
) -> None:
    """Two committed PostgreSQL transactions cannot bind different datasets."""

    pg_database_config.pool_size = 2
    pg_database_config.max_overflow = 0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    database = SyncDatabase(backend=backend)
    owner = "13172-postgres-owner"
    for dataset_id in ("postgres-authority-a", "postgres-authority-b"):
        database.enroll_dataset(
            SyncDatasetCreate(
                dataset_id=dataset_id,
                owner_user_id=owner,
                scope_type="personal",
                encryption_policy="server_trusted_v1",
                domains=["notes.note"],
            )
        )
    pool = backend.get_pool()
    first_connection = pool.get_connection()
    second_connection = pool.get_connection()
    _require(first_connection is not second_connection, "PostgreSQL pool reused a checkout")
    barrier = Barrier(2)

    def bind(connection: object, dataset_id: str) -> str:
        barrier.wait()
        try:
            database.bind_personal_context_dataset(
                dataset_id=dataset_id,
                user_id=owner,
                expected_binding=None,
                profile_id="postgres-profile",
                authority_id="tldw-server",
                integrity_key_id="postgres-integrity-key",
                purge_generation=0,
                link_state="bootstrap_pending",
                connection=connection,
            )
        except SyncStoreError as exc:
            return str(exc)
        return "bound"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(bind, first_connection, "postgres-authority-a"),
                executor.submit(bind, second_connection, "postgres-authority-b"),
            )
            outcomes = sorted(future.result(timeout=30) for future in futures)
        _require_digest_equal(
            outcomes,
            sorted([_AUTHORITY_ERROR, "bound"]),
            "PostgreSQL race did not choose exactly one authority",
        )
        active = [
            dataset
            for dataset in database.list_datasets_for_user(owner)
            if dataset.metadata.get("personal_context") is not None
        ]
        _require(len(active) == 1, "PostgreSQL race persisted multiple authorities")
    finally:
        pool.return_connection(first_connection)
        pool.return_connection(second_connection)
        pool.close_all()


def test_production_http_relay_debt_survives_restart_and_recovers_on_push_and_pull(
    production_factories,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Certify the canonical HTTP -> durable journal -> Sync egress lifecycle."""

    initial_canonical, initial_sync = production_factories
    production_app = initial_sync._certification_production_app
    factory_trace = initial_sync._certification_factory_trace
    trace_before = dict(factory_trace)
    store_cache_before = sync_v2_factory._sync_v2_store_for_user.cache_info()
    initial_backend = initial_sync.store.db.backend
    evidence_dir = tmp_path / "certification-evidence"
    evidence_dir.mkdir()
    (evidence_dir / "diagnostic.json").write_text(
        json.dumps({"status": "content-free", "source": "certification-fixture"}),
        encoding="utf-8",
    )
    (evidence_dir / "migration-snapshot.json").write_text(
        json.dumps({"status": "observed", "schema_change": False}),
        encoding="utf-8",
    )
    artifact_records: list[dict[str, object]] = []
    log_messages: list[str] = []
    sink_id = logger.add(lambda message: log_messages.append(str(message)))
    file_sink_id = logger.add(evidence_dir / "application.log")
    logger.info("TASK-13172 content-free application logger fixture active")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    responses: list[str] = []

    try:
        with _production_client(production_app) as client:
            _require(client.app is production_app, "production application was bypassed")
            _require(
                get_personal_context_service not in client.app.dependency_overrides,
                "Personal Context service dependency was overridden",
            )
            _require(
                sync_endpoint.get_sync_v2_service not in client.app.dependency_overrides,
                "Sync service dependency was overridden",
            )
            capabilities = client.get("/api/v1/sync/capabilities")
            responses.append(capabilities.text)
            _require_status(capabilities, 200, "capabilities request failed")
            _require(
                capabilities.json()["personal_context"]["ongoing_sync_version"] == 0,
                "ongoing sync protocol version changed",
            )
            registration = client.post(
                "/api/v1/sync/devices/register",
                json=_device_payload(private_key.public_key()),
            )
            _require_status(registration, 200, "device registration failed")
            bootstrap = client.post(
                "/api/v1/sync/personal-context/bootstrap",
                json={"device_id": _DEVICE_ID, "required_schema_version": 1},
            )
            responses.append(bootstrap.text)
            _require_status(bootstrap, 200, "Personal Context bootstrap failed")
            boot = bootstrap.json()
            wrapped_blob = boot["wrapped_key_blob"]
            dataset_id = boot["dataset_id"]
            profile_id = boot["manifest"]["profile_id"]
            integrity_key = initial_canonical._repository.sync_integrity_key(profile_id)[1]
            wrapped = base64.urlsafe_b64decode(wrapped_blob.split(":", 1)[1])
            decrypted_integrity_key = private_key.decrypt(
                wrapped,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=f"personal-context:{boot['integrity_key_id']}".encode(),
                ),
            )
            _require_digest_equal(
                decrypted_integrity_key,
                integrity_key,
                "wrapped integrity key did not decrypt to the canonical key",
            )
            complete = client.post(
                "/api/v1/sync/personal-context/complete",
                json={
                    "device_id": _DEVICE_ID,
                    "dataset_id": dataset_id,
                    "bootstrap_cursor": boot["cursor"],
                },
            )
            _require_status(complete, 204, "Personal Context link completion failed")
            exchange = _seed_exchange(initial_sync, dataset_id)
            _require(
                client.get("/api/v1/sync/capabilities").json()["personal_context"][
                    "ongoing_sync_version"
                ]
                == 0,
                "ongoing sync protocol version activated",
            )

            scope_id = client.get("/api/v1/personal-context/scopes").json()["items"][0][
                "scope_id"
            ]
            before_direct = _publication_state(initial_canonical)
            created_response = client.post(
                "/api/v1/personal-context/records",
                json=_record_body(scope_id, _PLAINTEXT_CANARY),
            )
            responses.append(created_response.text)
            _require_status(created_response, 201, "canonical record creation failed")
            _require(
                factory_trace["personal_context_factory"]
                > trace_before["personal_context_factory"],
                "production Personal Context dependency factory was not traversed",
            )
            _require(
                factory_trace["sync_factory"] > trace_before["sync_factory"],
                "production Sync dependency factory was not traversed",
            )
            store_cache_after = sync_v2_factory._sync_v2_store_for_user.cache_info()
            _require(
                store_cache_after.hits > store_cache_before.hits,
                "production Sync store cache was not traversed",
            )
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
            _require_status(updated_response, 200, "canonical record update failed")
            updated = updated_response.json()
            direct_rows = _publication_state(initial_canonical)[len(before_direct) :]
            direct_sequences = sorted({row["profile_publication_sequence"] for row in direct_rows})
            _require(len(direct_sequences) == 2, "canonical batch count changed")
            for sequence in direct_sequences:
                rows = [row for row in direct_rows if row["profile_publication_sequence"] == sequence]
                _require_digest_equal(
                    [(row["batch_ordinal"], row["role"]) for row in rows],
                    [(0, "semantic"), (1, "manifest")],
                    "canonical publication ordering changed",
                )
                _require(
                    len({row["status"] for row in rows}) == 1,
                    "canonical batch status became non-atomic",
                )

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
            _require(drained.continuation == "complete", "initial relay did not drain")
            _require(drained.inspected_rows <= 100, "relay exceeded row budget")
            _require(
                all(
                    row["status"] == "complete"
                    for row in _publication_state(initial_canonical)
                    if row["profile_publication_sequence"] in direct_sequences
                ),
                "canonical direct batches did not complete",
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
            _require_status(accepted, 200, "after-commit relay failure changed HTTP result")
            pending = _publication_state(initial_canonical)
            pending_sequence = max(int(row["profile_publication_sequence"]) for row in pending)
            pending_rows = [
                row for row in pending if row["profile_publication_sequence"] == pending_sequence
            ]
            _require_digest_equal(
                [(row["role"], row["row_state"]) for row in pending_rows],
                [("semantic", "pending"), ("manifest", "pending")],
                "failed relay did not preserve ordered pending rows",
            )
            _require_digest_equal(
                {row["status"] for row in pending_rows},
                {"pending"},
                "failed relay changed durable batch status",
            )
            _require_digest_equal(
                {row["sync_server_cursor"] for row in pending_rows},
                {None},
                "failed relay assigned a Sync cursor",
            )
            _require(
                not initial_sync.store.db.execute(
                    """SELECT server_sequence FROM sync_envelopes
                        WHERE routing_metadata_json LIKE ?""",
                    (f'%"profile_publication_sequence":{pending_sequence}%',),
                ).rows,
                "failed relay created authority envelopes",
            )

        _create_sqlite_backup(
            Path(cast(str, initial_backend.config.sqlite_path)),
            evidence_dir / "application-backup.db",
        )
        _scan_application_artifacts(
            tmp_path,
            phase="before-first-backend-reset",
            wrapped_blob=wrapped_blob,
            records=artifact_records,
        )
        reset_managed_sqlite_backends(backends=[initial_backend])
        _clear_factory_caches()
        restarted_canonical = personal_context_service_for_user(_USER_ID)
        restarted_sync = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
        _require(restarted_canonical is not initial_canonical, "canonical service was reused")
        _require(restarted_sync is not initial_sync, "Sync service was reused")
        _require(
            restarted_sync.store.db.backend is not initial_backend,
            "Sync backend was reused",
        )
        _require_digest_equal(
            _publication_state(restarted_canonical),
            pending,
            "durable journal changed across restart",
        )

        with _production_client(production_app) as restarted_client:
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
                    "personal_context_exchange": exchange,
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
            _require_status(push, 200, "client ingress push failed")
            _require_digest_equal(
                [item["client_envelope_id"] for item in push.json()["accepted"]],
                ["task-13172-client-ingress"],
                "client ingress acceptance identity changed",
            )
            _require(not push.json()["rejected"], "client ingress was rejected")
            after_push = _publication_state(restarted_canonical)
            recovered_pending = [
                row
                for row in after_push
                if row["profile_publication_sequence"] == pending_sequence
            ]
            _require_digest_equal(
                {row["status"] for row in recovered_pending},
                {"complete"},
                "push recovery did not complete pending publication",
            )
            _require_digest_equal(
                {row["row_state"] for row in recovered_pending},
                {"acknowledged"},
                "push recovery did not acknowledge publication rows",
            )

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
            _require(len(ingress_sync) == 1, "client ingress receipt cardinality changed")
            ingress_row = ingress_sync[0]
            _require(ingress_row["apply_status"] == "applied", "ingress was not applied")
            _require_digest_equal(
                json.loads(ingress_row["routing_metadata_json"])[
                    "personal_context_authority"
                ],
                {"role": "client_ingress"},
                "ingress authority role changed",
            )
            _require_digest_equal(
                ingress_row["payload_json"], "{}", "ingress plaintext payload persisted"
            )
            _require_digest_equal(
                ingress_row["payload_clear_json"],
                "{}",
                "ingress clear payload persisted",
            )
            _require(bool(ingress_row["payload_ciphertext"]), "ingress ciphertext missing")
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
                _require_digest_equal(
                    canonical_receipt[canonical_name],
                    ingress_row[sync_name],
                    "canonical and Sync receipt identity diverged",
                )

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
            _require_status(second_debt, 200, "pull-recovery debt mutation failed")
            second_pending = max(
                int(row["profile_publication_sequence"])
                for row in _publication_state(restarted_canonical)
            )

        second_backend = restarted_sync.store.db.backend
        _scan_application_artifacts(
            tmp_path,
            phase="before-second-backend-reset",
            wrapped_blob=wrapped_blob,
            records=artifact_records,
        )
        reset_managed_sqlite_backends(backends=[second_backend])
        _clear_factory_caches()
        pull_canonical = personal_context_service_for_user(_USER_ID)
        pull_sync = sync_v2_factory.sync_v2_service_for_user(_USER_ID)
        _require(pull_canonical is not restarted_canonical, "canonical restart was reused")
        _require(pull_sync is not restarted_sync, "second Sync restart was reused")
        _require(
            pull_sync.store.db.backend is not second_backend,
            "second Sync backend was reused",
        )

        with _production_client(production_app) as pull_client:
            zero_limit = pull_client.get(
                "/api/v1/sync/pull",
                params={**_pull_params(dataset_id, exchange=exchange), "page_size": 0},
            )
            _require_status(zero_limit, 422, "zero pull limit was accepted")
            pulled = pull_client.get("/api/v1/sync/pull", params=_pull_params(dataset_id, exchange=exchange))
            responses.append(pulled.text)
            _require_status(pulled, 200, "authority pull failed")
            pull_body = pulled.json()
            _require(
                pull_body["personal_context_relay"]["state"] == "complete",
                "pull recovery did not drain relay debt",
            )
            _require_digest_equal(
                {
                    row["status"]
                    for row in _publication_state(pull_canonical)
                    if row["profile_publication_sequence"] == second_pending
                },
                {"complete"},
                "pull recovery did not complete durable journal",
            )
            ingress_egress = [
                envelope
                for envelope in pull_body["envelopes"]
                if envelope["object_id"] == ingress_payload["record_id"]
            ]
            _require(len(ingress_egress) == 1, "authority egress cardinality changed")
            authority = ingress_egress[0]["routing_metadata"][
                "personal_context_authority"
            ]
            _require(authority["role"] == "home_authority", "egress authority role changed")
            _require_digest_equal(
                ingress_egress[0]["payload"],
                ingress_payload,
                "authority egress payload did not match canonical contract",
            )
            _require(
                ingress_egress[0]["client_envelope_id"]
                != "task-13172-client-ingress",
                "client ingress envelope escaped into egress",
            )
            _require(
                all(
                    envelope["routing_metadata"]["personal_context_authority"]["role"]
                    == "home_authority"
                    for envelope in pull_body["envelopes"]
                ),
                "non-authority envelope escaped into egress",
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
            _require(len(home_rows) == 1, "authority row cardinality changed")
            _require(home_rows[0]["apply_status"] == "applied", "authority row not applied")

            repeated = pull_client.get(
                "/api/v1/sync/pull",
                params=_pull_params(dataset_id, pull_body["next_cursor"], exchange=exchange),
            )
            responses.append(repeated.text)
            _require_status(repeated, 200, "repeat pull failed")
            _require(
                not repeated.json()["envelopes"],
                "repeat pull returned a duplicate envelope",
            )
            cursor = pull_sync.store.db.execute(
                """SELECT last_pulled_sequence FROM sync_device_cursors
                    WHERE dataset_id = ? AND device_id = ?
                      AND domain = 'personal_context.record'""",
                (dataset_id, _DEVICE_ID),
            ).rows[0]["last_pulled_sequence"]
            _require(
                int(cursor) == int(repeated.json()["next_cursor"]),
                "delivery and durable cursors diverged",
            )

        unsafe_response_text = "\n".join(responses)
        _require(
            _DIAGNOSTIC_CANARY not in unsafe_response_text,
            "diagnostic marker escaped into an HTTP response",
        )
        _require(
            _KEY_CANARY.decode() not in unsafe_response_text,
            "integrity key escaped into an HTTP response",
        )
        joined_logs = "".join(log_messages)
        for diagnostic, label in (
            (_DIAGNOSTIC_CANARY, "diagnostic marker"),
            (_PLAINTEXT_CANARY, "profile plaintext"),
            (_INGRESS_CANARY, "client ingress"),
            (_KEY_CANARY.decode(), "integrity key"),
        ):
            _require(diagnostic not in joined_logs, f"{label} escaped into logs")
        _scan_application_artifacts(
            tmp_path,
            phase="final-active-backend",
            wrapped_blob=wrapped_blob,
            records=artifact_records,
        )
        artifact_paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
        _require(
            {"Personalization.db", "Sync_v2.db", "ChaChaNotes.db"}.issubset(
                {path.name for path in artifact_paths}
            ),
            "expected application database artifact was not produced",
        )
        _require(
            any(
                wrapped_blob.encode() in artifact.read_bytes()
                for artifact in artifact_paths
                if _artifact_category(artifact)
                in {"sync-db", "sync-wal", "sync-shm", "application-backup"}
            ),
            "wrapped key was not stored in its authorized database boundary",
        )
        _require(
            relay_type.relay_profile is deterministic_relay,
            "deterministic relay seam was not restored",
        )
        categories = {record["category"] for record in artifact_records}
        _require(
            {
                "personalization-db",
                "sync-db",
                "notes-db",
                "application-log",
                "diagnostic",
                "migration-snapshot",
                "application-backup",
            }.issubset(categories),
            "required artifact category was not observed",
        )
        observed_phase_categories = {
            (record["phase"], record["category"]) for record in artifact_records
        }
        _require(
            {
                (phase, category)
                for phase in (
                    "before-first-backend-reset",
                    "before-second-backend-reset",
                    "final-active-backend",
                )
                for category in ("sync-wal", "sync-shm", "notes-wal", "notes-shm")
            }.issubset(observed_phase_categories),
            "required active WAL/SHM phase coverage was not observed",
        )
        _require_digest_equal(
            {record["phase"] for record in artifact_records},
            {
                "before-first-backend-reset",
                "before-second-backend-reset",
                "final-active-backend",
            },
            "artifact lifecycle phase coverage changed",
        )
        inventory_path = evidence_dir / "artifact-inventory.json"
        inventory_path.write_text(
            json.dumps(
                {
                    "records": artifact_records,
                    "custody_limit": (
                        "Scanned application-owned certification artifacts only; no "
                        "claim is made about physical deletion outside application custody."
                    ),
                    "excluded": [
                        "external-or-operator-managed-backups",
                        "exported-recovery-bundles",
                        "prior-process-memory",
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        inventory_contents = inventory_path.read_bytes()
        _require(
            all(
                canary not in inventory_contents
                for canary in (
                    _PLAINTEXT_CANARY.encode(),
                    _INGRESS_CANARY.encode(),
                    _DIAGNOSTIC_CANARY.encode(),
                    _KEY_CANARY,
                    wrapped_blob.encode(),
                )
            ),
            "protected canary escaped into artifact inventory",
        )
        _require(
            inventory_path.exists(),
            "phase-aware artifact inventory was not retained",
        )
    finally:
        logger.remove(sink_id)
        logger.remove(file_sink_id)
