"""Jobs and durable lifecycle contracts for the Notes semantic index."""

from __future__ import annotations

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
from tldw_Server_API.app.core.Notes_Graph.semantic_api import SemanticAPIError, SemanticIndexAPI
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

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def jobs(tmp_path, monkeypatch) -> JobManager:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
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
    capabilities = build_semantic_capabilities(
        SemanticCapabilityContract(
            provider="openai",
            model="text-embedding-3-small",
            endpoint_url="https://api.openai.com",
            execution_boundary="external",
            vector_backend="chromadb",
            storage_boundary="local",
            resolved_dimensions=1536,
            normalization_version="normalization-v1",
            chunker_version="chunker-v1",
            credential_source="durable",
            provider_healthy=True,
            vector_storage_available=True,
        )
    )
    return SemanticIndexAPI(
        note_db=db,
        jobs=jobs,
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        capability_resolver=lambda: capabilities,
        clock=lambda: NOW,
    )


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


def _configuration(db: CharactersRAGDB):
    created = db.note_semantic_store.create_configuration(
        dataset_id="dataset-a",
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
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
        capability_revision=created.capability_revision or "",
        now=NOW,
    )
    assert enabled is not None
    return enabled


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
                mode="delete",
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
