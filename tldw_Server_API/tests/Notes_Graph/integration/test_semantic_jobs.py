"""Jobs and durable lifecycle contracts for the Notes semantic index."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDesiredState,
    SemanticDimensionState,
    SemanticGenerationState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.semantic_api import (
    SemanticAPIError,
    SemanticIndexAPI,
    SemanticStatusFacts,
    derive_semantic_state,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_capabilities import (
    SemanticCapabilityContract,
    build_semantic_capabilities,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_jobs import (
    JOB_DOMAIN,
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    JOB_TYPE,
    SemanticJobCancelled,
    SemanticJobCommand,
    SemanticJobCoordinator,
    SemanticJobHandler,
    SemanticJobsError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import SemanticVectorCleanup
from tldw_Server_API.app.services import notes_semantic_index_worker as semantic_worker
from tldw_Server_API.app.services.notes_semantic_index_worker import (
    ProductionSemanticRuntime,
    build_production_runtime,
)

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def jobs(tmp_path, monkeypatch) -> JobManager:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    monkeypatch.setenv("JOBS_TEST_NOW_EPOCH", str(NOW.timestamp()))
    return JobManager(tmp_path / "semantic-jobs.sqlite")


def _command(**overrides: object) -> SemanticJobCommand:
    values: dict[str, object] = {
        "dataset_id": "dataset-a",
        "configuration_revision": 7,
        "mode": "rebuild",
        "generation_id": None,
    }
    values.update(overrides)
    return SemanticJobCommand(**values)  # type: ignore[arg-type]


def _semantic_api(db: CharactersRAGDB, jobs: JobManager) -> SemanticIndexAPI:
    capabilities = _capabilities()
    return SemanticIndexAPI(
        note_db=db,
        jobs=jobs,
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        capability_resolver=lambda: capabilities,
        clock=lambda: NOW,
    )


def _capabilities(
    *,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    dimensions: int | None = 1536,
    vector_backend: str = "chromadb",
    endpoint_url: str = "https://api.openai.com",
):
    return build_semantic_capabilities(
        SemanticCapabilityContract(
            provider=provider,
            model=model,
            endpoint_url=endpoint_url,
            execution_boundary="external",
            vector_backend=vector_backend,
            storage_boundary="local",
            resolved_dimensions=dimensions,
            normalization_version="normalization-v1",
            chunker_version="chunker-v1",
            credential_source="durable",
            provider_healthy=True,
            vector_storage_available=True,
        )
    )


@pytest.mark.parametrize(
    ("has_active_generation", "expected_state"),
    [(False, "preparing"), (True, "updating")],
)
def test_active_rebuild_projects_progress_before_transient_capability_health(
    has_active_generation: bool,
    expected_state: str,
) -> None:
    state, detail = derive_semantic_state(
        SemanticStatusFacts(
            desired_state="enabled",
            has_active_generation=has_active_generation,
            active_generation_usable=False,
            has_active_job=True,
            active_job_failed=False,
            pending_notes=0,
            failed_notes=0,
            cleanup_pending=False,
            indexing_available=False,
            configuration_stale=True,
        )
    )

    assert state == expected_state
    assert detail == "building"


@pytest.mark.parametrize(
    ("failure_code", "status_code"),
    [
        ("notes_semantic_jobs_unavailable", 503),
        ("notes_semantic_quota_exceeded", 429),
        ("notes_semantic_writer_conflict", 409),
    ],
)
def test_renewal_admission_gap_is_rebuild_required_and_fresh_run_recovers(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
    status_code: int,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / f"semantic-renew-gap-{failure_code}.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db)
        capabilities = _capabilities(
            model="text-embedding-3-large",
            dimensions=3_072,
        )
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )
        original_admit = api._admit
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SemanticAPIError(status_code, failure_code)
            return original_admit(*args, **kwargs)

        monkeypatch.setattr(api, "_admit", fail_once)
        with pytest.raises(SemanticAPIError) as failed:
            api.enable(
                expected_revision=active.configuration_revision,
                capability_revision=capabilities.capability_revision,
                idempotency_key=f"renew-{failure_code}",
            )
        assert failed.value.code == failure_code

        committed = db.note_semantic_store.get_configuration("dataset-a")
        assert committed is not None
        assert committed.configuration_revision == active.configuration_revision + 1
        gap_status = api.status()
        assert gap_status["state"] == "needs_attention"
        assert gap_status["detail_reason"] == "rebuild_required"
        assert gap_status["active_generation_usable"] is False

        recovered = api.create_run(
            mode="rebuild",
            expected_revision=committed.configuration_revision,
            idempotency_key=f"fresh-{failure_code}",
        )
        assert recovered["status"] == "queued"
        assert recovered["revision"] == committed.configuration_revision
        assert api.status()["state"] == "updating"
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    ("persisted_backend", "current_backend"),
    [("chromadb", "pgvector"), ("pgvector", "chromadb")],
)
def test_backend_change_requires_delete_without_mutating_configuration(
    tmp_path,
    jobs: JobManager,
    persisted_backend: str,
    current_backend: str,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / f"semantic-backend-{persisted_backend}.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db, vector_backend=persisted_backend)
        capabilities = _capabilities(vector_backend=current_backend)
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        disclosure = api.capabilities()
        assert disclosure["renewal_requires_delete"] is True
        with pytest.raises(SemanticAPIError) as blocked:
            api.enable(
                expected_revision=active.configuration_revision,
                capability_revision=capabilities.capability_revision,
                idempotency_key=f"backend-{persisted_backend}-{current_backend}",
            )
        assert blocked.value.code == "notes_semantic_backend_change_requires_delete"
        unchanged = db.note_semantic_store.get_configuration("dataset-a")
        assert unchanged == active
        assert unchanged.vector_backend == persisted_backend
        with db.transaction() as conn:
            receipt_count = conn.execute(
                "SELECT COUNT(*) AS count FROM note_semantic_operation_receipts "
                "WHERE owner_user_id=? AND dataset_id=?",
                ("owner-a", "dataset-a"),
            ).fetchone()
        assert receipt_count is not None
        assert int(receipt_count["count"]) == 0
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    "capability_overrides",
    [
        {"provider": "cohere"},
        {"model": "text-embedding-3-large", "dimensions": 3_072},
        {"endpoint_url": "https://embedding-proxy.example.test/v1"},
    ],
)
def test_same_backend_identity_changes_remain_renewable(
    tmp_path,
    jobs: JobManager,
    capability_overrides: dict[str, object],
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-same-backend-renewal.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db, vector_backend="chromadb")
        capabilities = _capabilities(**capability_overrides)  # type: ignore[arg-type]
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        result = api.enable(
            expected_revision=active.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key=f"same-backend-{next(iter(capability_overrides))}",
        )

        renewed = db.note_semantic_store.get_configuration("dataset-a")
        assert renewed is not None
        assert renewed.vector_backend == "chromadb"
        assert renewed.capability_revision == capabilities.capability_revision
        assert result["run"]["mode"] == "rebuild"
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persisted_backend", "current_backend"),
    [("chromadb", "pgvector"), ("pgvector", "chromadb")],
)
async def test_backend_change_delete_uses_persisted_store_and_confirms_absence(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    persisted_backend: str,
    current_backend: str,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / f"semantic-backend-delete-{persisted_backend}.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db, vector_backend=persisted_backend)
        generation_id = active.active_generation_id
        assert generation_id is not None
        capabilities = _capabilities(vector_backend=current_backend)
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )
        with pytest.raises(SemanticAPIError):
            api.enable(
                expected_revision=active.configuration_revision,
                capability_revision=capabilities.capability_revision,
                idempotency_key=f"blocked-{persisted_backend}-{current_backend}",
            )

        selected_backends: list[str] = []
        physical_generations = {generation_id}

        class PhysicalStore:
            async def delete_ids(self, _dataset_id, _generation_id, _vector_ids):
                return SemanticVectorCleanup(confirmed_absent=True)

            async def delete_generation(self, _dataset_id, deleted_generation_id):
                physical_generations.discard(deleted_generation_id)
                return SemanticVectorCleanup(confirmed_absent=True)

        async def build_store(*, backend_name: str, **_kwargs):
            selected_backends.append(backend_name)
            return PhysicalStore()

        monkeypatch.setattr(semantic_worker, "_build_vector_store", build_store)
        deletion = api.disable(
            expected_revision=active.configuration_revision,
            idempotency_key=f"delete-{persisted_backend}-{current_backend}",
        )
        disabled = db.note_semantic_store.get_configuration("dataset-a")
        assert disabled is not None
        runtime = semantic_worker.ProductionSemanticRuntime(
            db=db,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            configuration_revision=disabled.configuration_revision,
            generation_id=None,
            root_job_id=deletion["run"]["run_id"],
            settings=SemanticIndexSettings(),
        )

        async def not_cancelled() -> bool:
            return False

        result = await runtime.execute(
            mode="delete",
            cancellation_requested=not_cancelled,
        )

        assert selected_backends == [persisted_backend]
        assert physical_generations == set()
        assert result["cleanup_complete"] is True
        deleted_generation = db.note_semantic_store.get_generation(
            "dataset-a", generation_id
        )
        assert deleted_generation is not None
        assert deleted_generation.deleted_at is not None
    finally:
        db.close_all_connections()


def test_probe_eligible_renewal_persists_pending_identity_and_admits_rebuild(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-pending-renewal.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db)
        capabilities = _capabilities(model="org/custom-model", dimensions=None)
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        result = api.enable(
            expected_revision=active.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key="pending-renewal",
        )

        renewed = db.note_semantic_store.get_configuration("dataset-a")
        assert renewed is not None
        assert renewed.dimension_state is SemanticDimensionState.PENDING
        assert renewed.dimensions is None
        assert renewed.compatibility_hash is None
        assert result["run"]["mode"] == "rebuild"
    finally:
        db.close_all_connections()


def test_probe_eligible_initial_consent_persists_pending_identity_and_admits_build(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-pending-initial.sqlite"),
        client_id="owner-a",
    )
    try:
        capabilities = _capabilities(model="org/custom-model", dimensions=None)
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        result = api.enable(
            expected_revision=0,
            capability_revision=capabilities.capability_revision,
            idempotency_key="pending-initial",
        )

        enabled = db.note_semantic_store.get_configuration("dataset-a")
        assert enabled is not None
        assert enabled.dimension_state is SemanticDimensionState.PENDING
        assert enabled.dimensions is None
        assert enabled.compatibility_hash is None
        assert result["run"]["mode"] == "build"
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    "binding_drift",
    ["state", "configuration_revision", "compatibility_hash", "dimensions", "model_revision"],
)
def test_status_requires_complete_active_generation_binding(
    tmp_path,
    jobs: JobManager,
    binding_drift: str,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / f"semantic-binding-{binding_drift}.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db)
        assert active.active_generation_id is not None
        capabilities = replace(
            _capabilities(),
            capability_revision=active.capability_revision or "",
            compatibility_hash=active.compatibility_hash,
        )
        with db.transaction() as conn:
            if binding_drift == "state":
                conn.execute(
                    "UPDATE note_semantic_generations SET state='retired' "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    ("owner-a", "dataset-a", active.active_generation_id),
                )
            elif binding_drift == "configuration_revision":
                conn.execute(
                    "UPDATE note_semantic_generations "
                    "SET configuration_revision=configuration_revision-1 "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    ("owner-a", "dataset-a", active.active_generation_id),
                )
            elif binding_drift == "compatibility_hash":
                conn.execute(
                    "UPDATE note_semantic_generations SET compatibility_hash='other' "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    ("owner-a", "dataset-a", active.active_generation_id),
                )
            elif binding_drift == "dimensions":
                conn.execute(
                    "UPDATE note_semantic_generations SET dimensions=dimensions+1 "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    ("owner-a", "dataset-a", active.active_generation_id),
                )
            else:
                conn.execute(
                    "UPDATE note_semantic_generations SET model_revision='other' "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    ("owner-a", "dataset-a", active.active_generation_id),
                )
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        projected = api.status()

        assert projected["state"] == "needs_attention"
        assert projected["detail_reason"] == "rebuild_required"
        assert projected["active_generation_usable"] is False
    finally:
        db.close_all_connections()


def test_failed_renewal_admission_replays_without_a_second_consent_mutation(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-renew-admission-replay.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db)
        capabilities = _capabilities(model="text-embedding-3-large", dimensions=3_072)
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )
        original_admit = api._admit
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SemanticAPIError(503, "notes_semantic_jobs_unavailable")
            return original_admit(*args, **kwargs)

        monkeypatch.setattr(api, "_admit", fail_once)
        with pytest.raises(SemanticAPIError):
            api.enable(
                expected_revision=active.configuration_revision,
                capability_revision=capabilities.capability_revision,
                idempotency_key="renew-replay",
            )

        replayed = api.enable(
            expected_revision=active.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key="renew-replay",
        )
        replayed_again = api.enable(
            expected_revision=active.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key="renew-replay",
        )

        committed = db.note_semantic_store.get_configuration("dataset-a")
        assert committed is not None
        assert committed.configuration_revision == active.configuration_revision + 1
        assert replayed_again == replayed
        assert replayed["run"]["status"] == "queued"
    finally:
        db.close_all_connections()


def test_enable_uses_validated_backend_snapshot_after_environment_drift(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-backend-snapshot.sqlite"),
        client_id="owner-a",
    )
    try:
        capabilities = _capabilities(vector_backend="chromadb")
        monkeypatch.setenv("NOTES_SEMANTIC_VECTOR_BACKEND", "pgvector")
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        api.enable(
            expected_revision=0,
            capability_revision=capabilities.capability_revision,
            idempotency_key="backend-snapshot",
        )

        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        assert config.vector_backend == "chromadb"
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persisted_backend", "current_backend"),
    [("chromadb", "pgvector"), ("pgvector", "chromadb")],
)
async def test_backend_change_cleans_old_backend_before_rebinding_disabled_config(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    persisted_backend: str,
    current_backend: str,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / f"semantic-backend-cleanup-{persisted_backend}.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db, vector_backend=persisted_backend)
        generation_id = active.active_generation_id
        assert generation_id is not None
        physical = {
            persisted_backend: {generation_id},
            current_backend: set(),
        }
        opened_backends: list[str] = []

        class BackendVectors:
            def __init__(self, backend: str) -> None:
                self.backend = backend

            async def delete_ids(self, _dataset_id, generation, vector_ids):
                for vector_id in vector_ids:
                    physical[self.backend].discard((generation, vector_id))
                return SemanticVectorCleanup(confirmed_absent=True)

            async def delete_generation(self, _dataset_id, generation):
                physical[self.backend].discard(generation)
                return SemanticVectorCleanup(confirmed_absent=True)

        async def build_vectors(**kwargs):
            backend = str(kwargs["backend_name"])
            opened_backends.append(backend)
            return BackendVectors(backend)

        monkeypatch.setattr(semantic_worker, "_build_vector_store", build_vectors)
        disabled = db.note_semantic_store.disable_and_schedule_cleanup(
            dataset_id="dataset-a",
            expected_configuration_revision=active.configuration_revision,
            now=NOW,
        )
        assert disabled is not None
        runtime = ProductionSemanticRuntime(
            db=db,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            configuration_revision=disabled.configuration_revision,
            generation_id=generation_id,
            root_job_id="cleanup-old-backend",
            settings=SemanticIndexSettings(),
        )

        async def not_cancelled() -> bool:
            return False

        cleanup = await runtime.execute(
            mode="delete",
            cancellation_requested=not_cancelled,
        )

        assert opened_backends == [persisted_backend]
        assert cleanup["cleanup_complete"] is True
        assert physical[persisted_backend] == set()
        assert physical[current_backend] == set()
        capabilities = _capabilities(vector_backend=current_backend)
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )
        assert api.capabilities()["renewal_requires_delete"] is False

        result = api.enable(
            expected_revision=disabled.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key=f"setup-{current_backend}",
        )

        rebound = db.note_semantic_store.get_configuration("dataset-a")
        assert rebound is not None
        assert rebound.vector_backend == current_backend
        assert result["run"]["mode"] == "build"
    finally:
        db.close_all_connections()


def test_backend_rebind_stays_blocked_when_cleanup_work_is_missing_but_generation_is_live(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-backend-missing-cleanup.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db, vector_backend="chromadb")
        disabled = db.note_semantic_store.disable_and_schedule_cleanup(
            dataset_id="dataset-a",
            expected_configuration_revision=active.configuration_revision,
            now=NOW,
        )
        assert disabled is not None
        with db.transaction() as conn:
            conn.execute(
                "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=?",
                ("owner-a", "dataset-a"),
            )
        capabilities = _capabilities(vector_backend="pgvector")
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        assert api.capabilities()["renewal_requires_delete"] is True
        with pytest.raises(SemanticAPIError) as blocked:
            api.enable(
                expected_revision=disabled.configuration_revision,
                capability_revision=capabilities.capability_revision,
                idempotency_key="unsafe-backend-rebind",
            )

        assert blocked.value.code == "notes_semantic_backend_change_requires_delete"
        unchanged = db.note_semantic_store.get_configuration("dataset-a")
        assert unchanged == disabled
    finally:
        db.close_all_connections()


def test_receipt_admission_is_content_free_owner_authoritative_and_opaque(
    jobs: JobManager,
) -> None:
    coordinator = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-a",
        clock=lambda: NOW,
    )

    admitted = coordinator.admit(_command(), idempotency_key="enable-once")

    UUID(admitted.run_id)
    assert admitted.job["uuid"] == admitted.run_id
    assert admitted.job["owner_user_id"] == "owner-a"
    assert admitted.job["domain"] == JOB_DOMAIN == "notes"
    assert admitted.job["queue"] == JOB_QUEUE
    assert admitted.job["job_type"] == JOB_TYPE
    assert set(admitted.job["payload"]) == JOB_PAYLOAD_KEYS
    assert admitted.job["payload"] == {
        "schema_version": 1,
        "dataset_id": "dataset-a",
        "configuration_revision": 7,
        "generation_id": None,
        "mode": "rebuild",
    }
    serialized = repr(admitted.job["payload"]).lower()
    assert "owner-a" not in serialized
    assert "provider" not in serialized
    assert "model" not in serialized
    assert "content" not in serialized


def test_one_active_writer_converges_or_conflicts_by_owner_dataset_and_revision(
    jobs: JobManager,
) -> None:
    coordinator = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-a",
        clock=lambda: NOW,
    )
    first = coordinator.admit(_command(), idempotency_key="writer-one")
    converged = coordinator.admit(_command(), idempotency_key="writer-two")

    assert converged.run_id == first.run_id
    assert converged.disposition == "converged"
    with pytest.raises(SemanticJobsError) as exc_info:
        coordinator.admit(
            _command(mode="retry_failed", generation_id="generation-active"),
            idempotency_key="different-active-operation",
        )
    assert exc_info.value.code == "notes_semantic_writer_conflict"

    other_owner = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-b",
        clock=lambda: NOW,
    ).admit(_command(), idempotency_key="writer-one")
    assert other_owner.run_id != first.run_id


def test_active_writer_scope_survives_internal_dimension_revision_advance(
    jobs: JobManager,
) -> None:
    coordinator = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-a",
        clock=lambda: NOW,
    )
    first = coordinator.admit(
        _command(configuration_revision=7, mode="build"),
        idempotency_key="dimension-writer",
    )

    with pytest.raises(SemanticJobsError) as exc_info:
        coordinator.admit(
            _command(configuration_revision=8, mode="rebuild"),
            idempotency_key="revision-advanced-writer",
        )

    assert first.job["status"] == "queued"
    assert exc_info.value.code == "notes_semantic_writer_conflict"


def test_receipt_replay_survives_exact_retry_and_rejects_key_reuse(
    jobs: JobManager,
) -> None:
    coordinator = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-a",
        clock=lambda: NOW,
    )
    first = coordinator.admit(_command(), idempotency_key="stable-key")
    replay = coordinator.admit(_command(), idempotency_key="stable-key")

    assert replay.run_id == first.run_id
    assert replay.disposition == "replayed"
    with pytest.raises(SemanticJobsError) as exc_info:
        coordinator.admit(
            _command(configuration_revision=8),
            idempotency_key="stable-key",
        )
    assert exc_info.value.code == "notes_semantic_idempotency_conflict"


def test_job_quota_rejection_is_stable_and_sanitized() -> None:
    class QuotaJobs:
        def admit_idempotent_operation(self, _command):
            raise ValueError("Quota exceeded for internal owner details")

    coordinator = SemanticJobCoordinator(
        jobs=QuotaJobs(),  # type: ignore[arg-type]
        owner_user_id="owner-a",
        clock=lambda: NOW,
    )

    with pytest.raises(SemanticJobsError) as exc_info:
        coordinator.admit(_command(), idempotency_key="quota-key")

    assert exc_info.value.code == "notes_semantic_quota_exceeded"
    assert "owner" not in str(exc_info.value)


def test_run_lookup_and_cancel_use_jobs_owner_column_as_authority(
    jobs: JobManager,
) -> None:
    owner = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-a",
        clock=lambda: NOW,
    )
    foreign = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-b",
        clock=lambda: NOW,
    )
    admitted = owner.admit(_command(), idempotency_key="cancel-me")

    assert foreign.get_job_for_run(admitted.run_id) is None
    with pytest.raises(SemanticJobsError) as exc_info:
        owner.cancel(admitted.run_id, expected_revision=6)
    assert exc_info.value.code == "notes_semantic_run_revision_conflict"

    cancelled = owner.cancel(admitted.run_id, expected_revision=7)
    assert cancelled["status"] == "cancelled"


class _Runtime:
    def __init__(self, *, recovered: dict[str, object] | None = None) -> None:
        self.pinned_provider = "provider-pinned-at-enable"
        self.recovered = recovered
        self.executions: list[dict[str, object]] = []

    async def recover(self, **kwargs):
        self.executions.append({"phase": "recover", **kwargs})
        return self.recovered

    async def execute(self, **kwargs):
        self.executions.append({"phase": "execute", **kwargs})
        return {
            "state": "completed",
            "indexed_notes": 4,
            "excluded_notes": 1,
            "failed_notes": 0,
            "published_chunks": 9,
            "cleanup_complete": True,
            "error_code": None,
        }


@pytest.mark.asyncio
async def test_handler_bounds_retries_and_resolves_pinned_provider_outside_payload() -> None:
    runtime = _Runtime()
    factory_kwargs: dict[str, object] = {}

    def runtime_factory(**kwargs):
        factory_kwargs.update(kwargs)
        return runtime

    handler = SemanticJobHandler(
        runtime_factory=runtime_factory,
        settings=SemanticIndexSettings(max_retries=2),
    )
    job = {
        "uuid": "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        "owner_user_id": "owner-a",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "payload": {
            "schema_version": 1,
            "dataset_id": "dataset-a",
            "configuration_revision": 7,
            "generation_id": None,
            "mode": "rebuild",
        },
    }

    result = await handler.handle(job, cancellation_requested=lambda: False)

    execute = runtime.executions[-1]
    assert execute["max_batch_retries"] == 2
    assert execute["mode"] == "rebuild"
    assert factory_kwargs["mode"] == "rebuild"
    assert runtime.pinned_provider == "provider-pinned-at-enable"
    assert "provider" not in job["payload"]
    assert result["indexed_notes"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_state", ["failed", "cancelled", "running"])
async def test_handler_rejects_non_completed_bounded_results(
    invalid_state: str,
) -> None:
    class InvalidStateRuntime(_Runtime):
        async def execute(self, **kwargs):
            result = await super().execute(**kwargs)
            result["state"] = invalid_state
            return result

    handler = SemanticJobHandler(
        runtime_factory=lambda **_kwargs: InvalidStateRuntime(),
    )
    job = {
        "uuid": "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        "owner_user_id": "owner-a",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "payload": _command().payload(),
    }

    with pytest.raises(SemanticJobsError) as exc_info:
        await handler.handle(job, cancellation_requested=lambda: False)

    assert exc_info.value.code == "notes_semantic_job_result_invalid"


@pytest.mark.asyncio
async def test_handler_fences_cancellation_before_provider_work() -> None:
    runtime = _Runtime()
    handler = SemanticJobHandler(runtime_factory=lambda **_kwargs: runtime)
    job = {
        "uuid": "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        "owner_user_id": "owner-a",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "payload": {
            "schema_version": 1,
            "dataset_id": "dataset-a",
            "configuration_revision": 7,
            "generation_id": None,
            "mode": "rebuild",
        },
    }

    with pytest.raises(SemanticJobCancelled):
        await handler.handle(job, cancellation_requested=lambda: True)
    assert runtime.executions == []


@pytest.mark.asyncio
async def test_handler_replays_published_receipt_after_crash_without_provider_work() -> None:
    recovered = {
        "state": "completed",
        "indexed_notes": 3,
        "excluded_notes": 0,
        "failed_notes": 0,
        "published_chunks": 6,
        "cleanup_complete": True,
        "error_code": None,
    }
    runtime = _Runtime(recovered=recovered)
    handler = SemanticJobHandler(runtime_factory=lambda **_kwargs: runtime)
    payload = {
        "schema_version": 1,
        "dataset_id": "dataset-a",
        "configuration_revision": 7,
        "generation_id": None,
        "mode": "rebuild",
    }

    result = await handler.handle(
        {
            "uuid": "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            "owner_user_id": "owner-a",
            "domain": JOB_DOMAIN,
            "queue": JOB_QUEUE,
            "job_type": JOB_TYPE,
            "payload": payload,
        },
        cancellation_requested=lambda: False,
    )

    assert result == recovered
    assert [item["phase"] for item in runtime.executions] == ["recover"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["recover", "execute"])
async def test_handler_sanitizes_non_allowlisted_typed_runtime_failures(stage: str) -> None:
    secret = "postgresql://user:password@private/db?token=super-secret"  # nosec B105

    class FailingRuntime(_Runtime):
        async def recover(self, **kwargs):
            if stage == "recover":
                raise SemanticJobsError(secret)
            return await super().recover(**kwargs)

        async def execute(self, **kwargs):
            if stage == "execute":
                raise SemanticJobsError(secret)
            return await super().execute(**kwargs)

    handler = SemanticJobHandler(runtime_factory=lambda **_kwargs: FailingRuntime())
    job = {
        "uuid": "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        "owner_user_id": "owner-a",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "payload": {
            "schema_version": 1,
            "dataset_id": "dataset-a",
            "configuration_revision": 7,
            "generation_id": None,
            "mode": "rebuild",
        },
    }

    with pytest.raises(SemanticJobsError) as exc_info:
        await handler.handle(job, cancellation_requested=lambda: False)

    assert exc_info.value.code == "notes_semantic_worker_runtime_failed"
    assert secret not in str(exc_info.value)


def _configuration(
    db: CharactersRAGDB,
    *,
    vector_backend: str = "chromadb",
):
    created = db.note_semantic_store.create_configuration(
        dataset_id="dataset-a",
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="external",
        vector_backend=vector_backend,
        storage_boundary="local",
        storage_label="local-vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id="dataset-a",
        expected_configuration_revision=created.configuration_revision,
        capability_revision=created.capability_revision or "",
        now=NOW,
    )
    assert enabled is not None
    return enabled


def _active_configuration(
    db: CharactersRAGDB,
    *,
    vector_backend: str = "chromadb",
):
    enabled = _configuration(db, vector_backend=vector_backend)
    pending = db.note_semantic_store.create_generation(
        dataset_id="dataset-a",
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id="dataset-a",
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=1536,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    resolved_config = db.note_semantic_store.get_configuration("dataset-a")
    assert resolved_config is not None
    active = db.note_semantic_store.activate_generation(
        dataset_id="dataset-a",
        generation_id=resolved.id,
        expected_configuration_revision=resolved_config.configuration_revision,
        publication_receipt="receipt-active",
        now=NOW,
    )
    assert active is not None
    return active


def test_enabled_stale_configuration_renews_consent_rebuilds_and_replays(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-renew-consent.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db)
        active_generation_id = active.active_generation_id
        capabilities = _capabilities(
            model="text-embedding-3-large",
            dimensions=3072,
        )
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW + timedelta(minutes=10),
        )
        assert api.status()["detail_reason"] == "stale_configuration"

        first = api.enable(
            expected_revision=active.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key="renew-consent",
        )

        assert first["resource"]["state"] == "updating"
        assert first["resource"]["detail_reason"] == "building"
        assert first["run"]["mode"] == "rebuild"
        assert first["run"]["revision"] == active.configuration_revision + 1
        renewed = db.note_semantic_store.get_configuration("dataset-a")
        assert renewed is not None
        assert renewed.configuration_revision == active.configuration_revision + 1
        assert renewed.active_generation_id == active_generation_id
        assert renewed.capability_revision == capabilities.capability_revision
        assert renewed.disclosure_hash == capabilities.disclosure_hash
        assert renewed.compatibility_hash == capabilities.compatibility_hash
        assert renewed.provider == "openai"
        assert renewed.model == "text-embedding-3-large"
        assert renewed.dimensions == 3072
        assert renewed.consented_at == (NOW + timedelta(minutes=10)).isoformat()

        replay = api.enable(
            expected_revision=active.configuration_revision,
            capability_revision=capabilities.capability_revision,
            idempotency_key="renew-consent",
        )
        assert replay == first
        assert len(api._jobs_for_dataset()) == 1
    finally:
        db.close_all_connections()


def test_renewed_consent_rejects_capability_mismatch_and_store_cas_loss(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-renew-conflicts.sqlite"),
        client_id="owner-a",
    )
    try:
        active = _active_configuration(db)
        capabilities = _capabilities(
            model="text-embedding-3-large",
            dimensions=3072,
        )
        api = SemanticIndexAPI(
            note_db=db,
            jobs=jobs,
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            capability_resolver=lambda: capabilities,
            clock=lambda: NOW,
        )

        with pytest.raises(SemanticAPIError) as mismatch:
            api.enable(
                expected_revision=active.configuration_revision,
                capability_revision="stale-capability-revision",
                idempotency_key="renew-capability-mismatch",
            )
        assert mismatch.value.code == "notes_semantic_capability_revision_conflict"

        renew_calls = 0

        def lose_renewal_cas(**_kwargs):
            nonlocal renew_calls
            renew_calls += 1
            return None

        monkeypatch.setattr(
            db.note_semantic_store,
            "renew_configuration_consent",
            lose_renewal_cas,
            raising=False,
        )
        with pytest.raises(SemanticAPIError) as cas_loss:
            api.enable(
                expected_revision=active.configuration_revision,
                capability_revision=capabilities.capability_revision,
                idempotency_key="renew-cas-loss",
            )
        assert cas_loss.value.code == "notes_semantic_configuration_revision_conflict"
        assert renew_calls == 1
        unchanged = db.note_semantic_store.get_configuration("dataset-a")
        assert unchanged is not None
        assert unchanged.configuration_revision == active.configuration_revision
        assert unchanged.capability_revision == active.capability_revision
    finally:
        db.close_all_connections()


def test_delete_run_cancellation_fails_before_receipt_or_jobs_side_effects(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-delete-cancel.sqlite"),
        client_id="owner-a",
    )
    api = _semantic_api(db, jobs)
    admitted = SemanticJobCoordinator(
        jobs=jobs,
        owner_user_id="owner-a",
        clock=lambda: NOW,
    ).admit(
        _command(mode="delete", configuration_revision=7),
        idempotency_key="delete-run",
    )
    receipt_calls = 0
    cancel_calls = 0
    original_receipt = db.note_semantic_store.begin_operation_receipt
    original_cancel = jobs.cancel_job

    def record_receipt(**kwargs):
        nonlocal receipt_calls
        receipt_calls += 1
        return original_receipt(**kwargs)

    def record_cancel(job_id, **kwargs):
        nonlocal cancel_calls
        cancel_calls += 1
        return original_cancel(job_id, **kwargs)

    monkeypatch.setattr(db.note_semantic_store, "begin_operation_receipt", record_receipt)
    monkeypatch.setattr(jobs, "cancel_job", record_cancel)
    try:
        with pytest.raises(SemanticAPIError) as exc_info:
            api.cancel_run(
                run_id=UUID(admitted.run_id),
                expected_revision=7,
                idempotency_key="cancel-delete-run",
            )
        assert exc_info.value.status_code == 422
        assert exc_info.value.code == "notes_semantic_invalid_request"
        assert receipt_calls == 0
        assert cancel_calls == 0
        current = jobs.get_job_or_archived_by_uuid(
            admitted.run_id,
            domain=JOB_DOMAIN,
            owner_user_id="owner-a",
        )
        assert current is not None
        assert current["status"] == "queued"
    finally:
        db.close_all_connections()


def test_generation_root_job_recovery_and_disable_cleanup_are_durable(tmp_path) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-store.sqlite"), client_id="owner-a")
    try:
        enabled = _configuration(db)
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            now=NOW,
        )

        recovered = db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        )
        assert recovered == generation

        disabled = db.note_semantic_store.disable_and_schedule_cleanup(
            dataset_id="dataset-a",
            expected_configuration_revision=enabled.configuration_revision,
            now=NOW,
        )
        assert disabled is not None
        assert disabled.active_generation_id is None
        assert db.note_semantic_store.get_generation(
            "dataset-a", generation.id
        ).state is SemanticGenerationState.RETIRED
        cleanup = db.note_semantic_store.claim_work(
            dataset_id="dataset-a",
            now=NOW,
        )
        assert cleanup is not None
        assert cleanup.kind.value == "delete_generation"
        assert cleanup.generation_id == generation.id
    finally:
        db.close_all_connections()


def test_exhausted_cleanup_projects_attention_and_can_be_rearmed(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cleanup-rearm.sqlite"), client_id="owner-a")
    try:
        enabled = _configuration(db)
        db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            now=NOW,
        )
        disabled = db.note_semantic_store.disable_and_schedule_cleanup(
            dataset_id="dataset-a",
            expected_configuration_revision=enabled.configuration_revision,
            now=NOW,
        )
        assert disabled is not None
        moment = NOW
        for _attempt in range(5):
            claim = db.note_semantic_store.claim_generation_cleanup_batch(
                dataset_id="dataset-a",
                limit=1,
                now=moment,
            )[0]
            retried = db.note_semantic_store.retry_work(
                dataset_id="dataset-a",
                work_id=claim.id,
                expected_claim_token=claim.claim_token,
                error_code="notes_semantic_cleanup_failed",
                retry_at=moment + timedelta(seconds=1),
                now=moment,
            )
            assert retried is not None
            moment += timedelta(seconds=2)

        assert db.note_semantic_store.claim_generation_cleanup_batch(
            dataset_id="dataset-a",
            limit=1,
            now=moment,
        ) == ()
        assert db.note_semantic_store.has_stalled_cleanup(
            "dataset-a",
            expired_before=moment,
        ) is True
        assert _semantic_api(db, jobs).status()["state"] == "needs_attention"

        assert db.note_semantic_store.rearm_exhausted_generation_cleanup(
            dataset_id="dataset-a",
            limit=1,
            now=moment,
        ) == 1
        rearmed = db.note_semantic_store.claim_generation_cleanup_batch(
            dataset_id="dataset-a",
            limit=1,
            now=moment,
        )
        assert len(rearmed) == 1
        assert rearmed[0].attempt_count == 0
    finally:
        db.close_all_connections()


def test_repeated_cleanup_lease_expiry_can_be_rearmed(
    tmp_path,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cleanup-lease.sqlite"), client_id="owner-a")
    try:
        enabled = _configuration(db)
        db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            now=NOW,
        )
        disabled = db.note_semantic_store.disable_and_schedule_cleanup(
            dataset_id="dataset-a",
            expected_configuration_revision=enabled.configuration_revision,
            now=NOW,
        )
        assert disabled is not None
        moment = NOW
        for _attempt in range(5):
            claims = db.note_semantic_store.claim_generation_cleanup_batch(
                dataset_id="dataset-a",
                limit=1,
                now=moment,
            )
            assert len(claims) == 1
            reclaimed = db.note_semantic_store.reclaim_expired_dataset_work(
                dataset_id="dataset-a",
                expired_before=moment + timedelta(seconds=1),
                limit=1,
                now=moment + timedelta(seconds=2),
            )
            assert reclaimed == 1
            moment += timedelta(seconds=3)

        assert db.note_semantic_store.rearm_exhausted_generation_cleanup(
            dataset_id="dataset-a",
            limit=1,
            now=moment,
        ) == 1
        assert len(
            db.note_semantic_store.claim_generation_cleanup_batch(
                dataset_id="dataset-a",
                limit=1,
                now=moment,
            )
        ) == 1
    finally:
        db.close_all_connections()


def test_http_admission_does_not_create_a_post_admission_ghost_generation(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-worker-owned-generation.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        admitted = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="worker-owned-generation",
        )

        assert db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            admitted["run"]["run_id"],
        ) is None
    finally:
        db.close_all_connections()


def test_enable_exact_retry_resumes_jobs_admission_after_committed_config_gap(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-enable-gap.sqlite"), client_id="owner-a")
    api = _semantic_api(db, jobs)
    original_admit = jobs.admit_idempotent_operation
    calls = 0

    def fail_once(command):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("jobs unavailable at /private/secret?token=abc")
        return original_admit(command)

    monkeypatch.setattr(jobs, "admit_idempotent_operation", fail_once)
    capability_revision = api.capabilities()["capability_revision"]
    try:
        with pytest.raises(SemanticAPIError) as exc_info:
            api.enable(
                expected_revision=0,
                capability_revision=capability_revision,
                idempotency_key="resume-enable",
            )
        assert exc_info.value.code == "notes_semantic_jobs_unavailable"
        committed = db.note_semantic_store.get_configuration("dataset-a")
        assert committed is not None
        assert committed.desired_state is SemanticDesiredState.ENABLED

        api._capability_resolver = lambda: (_ for _ in ()).throw(
            RuntimeError("provider changed at /private/secret?token=abc")
        )

        resumed = api.enable(
            expected_revision=0,
            capability_revision=capability_revision,
            idempotency_key="resume-enable",
        )
        assert resumed["run"]["status"] == "queued"

        with pytest.raises(SemanticAPIError) as reused:
            api.enable(
                expected_revision=0,
                capability_revision=capability_revision,
                idempotency_key="different-enable-key",
            )
        assert reused.value.code in {
            "notes_semantic_configuration_revision_conflict",
            "notes_semantic_idempotency_conflict",
        }
    finally:
        db.close_all_connections()


def test_resolved_model_revision_is_persisted_on_config_and_generation(tmp_path) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-model-revision.sqlite"), client_id="owner-a")
    try:
        created = db.note_semantic_store.create_configuration(
            dataset_id="dataset-a",
            capability_revision="capability-v1",
            disclosure_hash="disclosure-v1",
            provider="provider-a",
            model="model-a",
            model_revision=None,
            endpoint_origin_revision="origin-v1",
            endpoint_origin_display="https://api.example.test",
            data_boundary="external",
            vector_backend="chromadb",
            storage_boundary="local",
            storage_label="local-vectors",
            normalization_version="normalization-v1",
            chunker_version="chunker-v1",
            now=NOW,
        )
        enabled = db.note_semantic_store.enable_configuration(
            dataset_id="dataset-a",
            expected_configuration_revision=created.configuration_revision,
            capability_revision="capability-v1",
            now=NOW,
        )
        assert enabled is not None
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            model_revision=None,
            now=NOW,
        )

        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=enabled.configuration_revision,
            dimensions=1536,
            compatibility_hash="compatibility-v1",
            model_revision="model-digest-a",
            now=NOW,
        )

        assert resolved is not None
        assert resolved.model_revision == "model-digest-a"
        assert db.note_semantic_store.get_configuration("dataset-a").model_revision == "model-digest-a"
    finally:
        db.close_all_connections()


def test_local_note_lifecycle_targets_the_bound_enabled_canonical_dataset(tmp_path) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-local-lifecycle.sqlite"), client_id="owner-a")
    try:
        enabled = _configuration(db)
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            now=NOW,
        )
        note_id = "11111111-1111-4111-8111-111111111111"

        assert db.note_store.add_note("Title", "Body", note_id=note_id) == note_id
        created = db.note_semantic_store.get_note_state("dataset-a", generation.id, note_id)
        assert created is not None and created.content_version == 1

        assert db.note_store.update_note(note_id, {"content": "Edited"}, expected_version=1)
        edited = db.note_semantic_store.get_note_state("dataset-a", generation.id, note_id)
        assert edited is not None and edited.content_version == 2

        assert db.note_store.soft_delete_note(note_id, expected_version=2)
        trashed = db.note_semantic_store.get_note_state("dataset-a", generation.id, note_id)
        assert trashed is not None and trashed.state.value == "tombstoned"

        assert db.note_store.restore_note(note_id, expected_version=3)
        restored = db.note_semantic_store.get_note_state("dataset-a", generation.id, note_id)
        assert restored is not None and restored.state.value == "pending"
        assert restored.content_version == 4
    finally:
        db.close_all_connections()


def test_disable_fences_notes_authority_before_jobs_cancellation_and_cleanup_admission(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-disable-order.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-before-disable",
        )
        expected_revision = enabled["resource"]["configuration_revision"]
        events: list[str] = []
        original_disable = db.note_semantic_store.disable_and_schedule_cleanup
        original_cancel = jobs.cancel_job
        original_admit = jobs.admit_idempotent_operation

        def disable_and_schedule_cleanup(**kwargs):
            events.append("disable")
            return original_disable(**kwargs)

        def cancel_job(job_id, **kwargs):
            config = db.note_semantic_store.get_configuration("dataset-a")
            assert config is not None
            assert config.desired_state is SemanticDesiredState.DISABLED
            events.append("cancel")
            return original_cancel(job_id, **kwargs)

        def admit_idempotent_operation(command):
            config = db.note_semantic_store.get_configuration("dataset-a")
            assert config is not None
            assert config.desired_state is SemanticDesiredState.DISABLED
            events.append("admit")
            return original_admit(command)

        monkeypatch.setattr(
            db.note_semantic_store,
            "disable_and_schedule_cleanup",
            disable_and_schedule_cleanup,
        )
        monkeypatch.setattr(jobs, "cancel_job", cancel_job)
        monkeypatch.setattr(jobs, "admit_idempotent_operation", admit_idempotent_operation)

        first = api.disable(
            expected_revision=expected_revision,
            idempotency_key="disable-in-order",
        )
        replay = api.disable(
            expected_revision=expected_revision,
            idempotency_key="disable-in-order",
        )

        assert events == ["disable", "cancel", "admit"]
        assert replay["run"]["run_id"] == first["run"]["run_id"]
    finally:
        db.close_all_connections()


def test_enable_receipt_replays_after_dimension_revision_advances(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-replay.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        capability_revision = api.capabilities()["capability_revision"]
        first = api.enable(
            expected_revision=0,
            capability_revision=capability_revision,
            idempotency_key="enable-replay",
        )
        generation = db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            first["run"]["run_id"],
        )
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        assert generation is None
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=config.configuration_revision,
            compatibility_hash=config.compatibility_hash,
            dimension_state=config.dimension_state,
            dimensions=config.dimensions,
            root_job_id=first["run"]["run_id"],
            model_revision=config.model_revision,
            now=NOW,
        )
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=config.configuration_revision,
            dimensions=1536,
            compatibility_hash=api._capabilities().compatibility_hash or "",
            model_revision=config.model_revision,
            now=NOW,
        )
        assert resolved is not None

        replay = api.enable(
            expected_revision=0,
            capability_revision=capability_revision,
            idempotency_key="enable-replay",
        )

        assert replay["run"]["run_id"] == first["run"]["run_id"]
    finally:
        db.close_all_connections()


def test_cancelled_staging_generation_is_failed_and_queued_for_cleanup(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cancel.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-cancel",
        )
        run_id = enabled["run"]["run_id"]
        generation = db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            run_id,
        )
        assert generation is None
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=config.configuration_revision,
            compatibility_hash=config.compatibility_hash,
            dimension_state=config.dimension_state,
            dimensions=config.dimensions,
            root_job_id=run_id,
            model_revision=config.model_revision,
            now=NOW,
        )

        api.cancel_run(
            run_id=UUID(run_id),
            expected_revision=generation.configuration_revision,
            idempotency_key="cancel-staging",
        )

        cancelled = db.note_semantic_store.get_generation("dataset-a", generation.id)
        assert cancelled is not None
        assert cancelled.state is SemanticGenerationState.FAILED
        assert db.note_semantic_store.has_pending_cleanup("dataset-a") is True
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
async def test_committed_cancel_intent_blocks_late_worker_generation_creation(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-cancel-before-create.sqlite"),
        client_id="owner-a",
    )
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-cancel-before-create",
        )
        run_id = enabled["run"]["run_id"]
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        assert db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            run_id,
        ) is None

        cancelled = api.cancel_run(
            run_id=UUID(run_id),
            expected_revision=enabled["run"]["revision"],
            idempotency_key="cancel-before-worker-create",
        )
        assert cancelled["run"]["status"] == "cancelled"

        with pytest.raises(SemanticJobCancelled):
            await build_production_runtime(
                db=db,
                settings=SemanticIndexSettings(),
                owner_user_id="owner-a",
                dataset_id="dataset-a",
                configuration_revision=config.configuration_revision,
                generation_id=None,
                root_job_id=run_id,
                mode="build",
            )

        assert db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            run_id,
        ) is None
        current = db.note_semantic_store.get_configuration("dataset-a")
        assert current is not None
        assert current.active_generation_id is None
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
async def test_stale_cancel_revision_does_not_create_worker_cancellation_intent(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "semantic-stale-cancel-before-create.sqlite"),
        client_id="owner-a",
    )
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-before-stale-cancel",
        )
        run_id = enabled["run"]["run_id"]
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None

        with pytest.raises(SemanticAPIError) as exc_info:
            api.cancel_run(
                run_id=UUID(run_id),
                expected_revision=config.configuration_revision + 1,
                idempotency_key="stale-cancel-before-worker-create",
            )
        assert exc_info.value.code == "notes_semantic_run_revision_conflict"

        await build_production_runtime(
            db=db,
            settings=SemanticIndexSettings(),
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            configuration_revision=config.configuration_revision,
            generation_id=None,
            root_job_id=run_id,
            mode="build",
        )

        generation = db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            run_id,
        )
        assert generation is not None
        assert generation.state is SemanticGenerationState.STAGING
    finally:
        db.close_all_connections()


def test_root_cancellation_fence_precedes_activation_and_jobs_cancellation(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cancel-order.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-cancel-order",
        )
        run_id = enabled["run"]["run_id"]
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=config.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id=run_id,
            model_revision=None,
            now=NOW,
        )
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=config.configuration_revision,
            dimensions=1536,
            compatibility_hash=api._capabilities().compatibility_hash or "",
            model_revision=None,
            now=NOW,
        )
        assert resolved is not None
        original_cancel = jobs.cancel_job
        activation_wins: list[bool] = []

        def activate_then_cancel(job_id, **kwargs):
            activated = db.note_semantic_store.activate_generation(
                dataset_id="dataset-a",
                generation_id=resolved.id,
                expected_configuration_revision=resolved.configuration_revision,
                publication_receipt="publication-a",
                now=NOW + timedelta(seconds=1),
            )
            activation_wins.append(activated is not None)
            return original_cancel(job_id, **kwargs)

        monkeypatch.setattr(jobs, "cancel_job", activate_then_cancel)

        cancelled = api.cancel_run(
            run_id=UUID(run_id),
            expected_revision=enabled["run"]["revision"],
            idempotency_key="cancel-before-activate",
        )

        terminal = db.note_semantic_store.get_generation("dataset-a", resolved.id)
        assert activation_wins == [False]
        assert terminal is not None
        assert terminal.state is SemanticGenerationState.FAILED
        assert terminal.terminal_error_code == "notes_semantic_run_cancelled"
        assert cancelled["run"]["status"] == "cancelled"
    finally:
        db.close_all_connections()


def test_root_cancellation_conflicts_when_activation_already_committed(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cancel-active.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-before-activation",
        )
        run_id = enabled["run"]["run_id"]
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=config.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id=run_id,
            model_revision=None,
            now=NOW,
        )
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=config.configuration_revision,
            dimensions=1536,
            compatibility_hash=api._capabilities().compatibility_hash or "",
            model_revision=None,
            now=NOW,
        )
        assert resolved is not None
        activated = db.note_semantic_store.activate_generation(
            dataset_id="dataset-a",
            generation_id=resolved.id,
            expected_configuration_revision=resolved.configuration_revision,
            publication_receipt="publication-a",
            now=NOW + timedelta(seconds=1),
        )
        assert activated is not None

        with pytest.raises(SemanticAPIError) as exc_info:
            api.cancel_run(
                run_id=UUID(run_id),
                expected_revision=enabled["run"]["revision"],
                idempotency_key="cancel-after-activation",
            )

        assert exc_info.value.code == "notes_semantic_run_revision_conflict"
        assert SemanticJobCoordinator(
            jobs=jobs,
            owner_user_id="owner-a",
            clock=lambda: NOW,
        ).get_job_for_run(run_id)["status"] == "queued"
    finally:
        db.close_all_connections()


def test_cancel_retry_finishes_jobs_after_notes_fence_was_committed(
    tmp_path,
    jobs: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cancel-crash.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-cancel-crash",
        )
        run_id = enabled["run"]["run_id"]
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=config.configuration_revision,
            compatibility_hash=config.compatibility_hash,
            dimension_state=config.dimension_state,
            dimensions=config.dimensions,
            root_job_id=run_id,
            model_revision=config.model_revision,
            now=NOW,
        )
        original_cancel = jobs.cancel_job
        calls = 0

        def fail_once(job_id, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SemanticJobsError("notes_semantic_jobs_unavailable")
            return original_cancel(job_id, **kwargs)

        monkeypatch.setattr(jobs, "cancel_job", fail_once)
        with pytest.raises(SemanticAPIError):
            api.cancel_run(
                run_id=UUID(run_id),
                expected_revision=enabled["run"]["revision"],
                idempotency_key="cancel-crash-retry",
            )
        fenced = db.note_semantic_store.get_generation("dataset-a", generation.id)
        assert fenced is not None
        assert fenced.state is SemanticGenerationState.FAILED
        assert fenced.terminal_error_code == "notes_semantic_run_cancelled"

        retried = api.cancel_run(
            run_id=UUID(run_id),
            expected_revision=enabled["run"]["revision"],
            idempotency_key="cancel-crash-retry",
        )

        assert calls == 2
        assert retried["run"]["status"] == "cancelled"
    finally:
        db.close_all_connections()


def test_cancel_idempotency_replays_bounded_response_and_rejects_key_reuse(
    tmp_path,
    jobs: JobManager,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "semantic-cancel-receipt.sqlite"), client_id="owner-a")
    try:
        api = _semantic_api(db, jobs)
        enabled = api.enable(
            expected_revision=0,
            capability_revision=api.capabilities()["capability_revision"],
            idempotency_key="enable-for-cancel-receipt",
        )
        run_id = enabled["run"]["run_id"]
        config = db.note_semantic_store.get_configuration("dataset-a")
        assert config is not None
        generation = db.note_semantic_store.get_generation_by_root_job_id(
            "dataset-a",
            run_id,
        )
        if generation is None:
            generation = db.note_semantic_store.create_generation(
                dataset_id="dataset-a",
                configuration_revision=config.configuration_revision,
                compatibility_hash=config.compatibility_hash,
                dimension_state=config.dimension_state,
                dimensions=config.dimensions,
                root_job_id=run_id,
                model_revision=config.model_revision,
                now=NOW,
            )

        first = api.cancel_run(
            run_id=UUID(run_id),
            expected_revision=enabled["run"]["revision"],
            idempotency_key="cancel-operation-key",
        )
        disabled = db.note_semantic_store.disable_and_schedule_cleanup(
            dataset_id="dataset-a",
            expected_configuration_revision=config.configuration_revision,
            now=NOW,
        )
        assert disabled is not None

        replay = api.cancel_run(
            run_id=UUID(run_id),
            expected_revision=enabled["run"]["revision"],
            idempotency_key="cancel-operation-key",
        )
        assert replay == first

        next_job = SemanticJobCoordinator(
            jobs=jobs,
            owner_user_id="owner-a",
            clock=lambda: NOW,
        ).admit(
            SemanticJobCommand(
                dataset_id="dataset-a",
                configuration_revision=disabled.configuration_revision,
                mode="rebuild",
            ),
            idempotency_key="next-cancellable-job",
        )
        with pytest.raises(SemanticAPIError) as reused:
            api.cancel_run(
                run_id=UUID(next_job.run_id),
                expected_revision=disabled.configuration_revision,
                idempotency_key="cancel-operation-key",
            )
        assert reused.value.code == "notes_semantic_idempotency_conflict"
    finally:
        db.close_all_connections()
