from __future__ import annotations

pytest_plugins = [
    "tldw_Server_API.tests._plugins.postgres",
]

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.config import clear_config_cache
from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.docker_runner import DockerRunner
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.store import InMemoryStore, PostgresStore, SandboxStore


def _configure_background_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configure services safely before injecting the adapter under test."""
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("SANDBOX_ROOT_DIR", str(tmp_path / "sandbox_root"))
    monkeypatch.setenv("SANDBOX_SNAPSHOT_PATH", str(tmp_path / "snapshots"))
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("SANDBOX_RUN_CLAIM_LEASE_SEC", "30")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    if hasattr(app_settings, "SANDBOX_STORE_BACKEND"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_BACKEND", "memory")
    if hasattr(app_settings, "SANDBOX_ROOT_DIR"):
        monkeypatch.setattr(app_settings, "SANDBOX_ROOT_DIR", str(tmp_path / "sandbox_root"))
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_PATH", str(tmp_path / "snapshots"))
    clear_config_cache()


def _force_docker_preflight_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def _preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        del self, network_policy
        return {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                execution_mode="mocked",
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _preflights)


@pytest.fixture(
    params=[
        pytest.param("memory", marks=pytest.mark.unit),
        pytest.param(
            "postgres",
            marks=[pytest.mark.integration, pytest.mark.postgres],
        ),
    ]
)
def shared_store_pair(
    request: pytest.FixtureRequest,
) -> tuple[SandboxStore, SandboxStore]:
    """Return stores sharing one logical backend across two service nodes."""
    if request.param == "memory":
        store = InMemoryStore()
        return store, store

    pg_temp_db = request.getfixturevalue("pg_temp_db")
    dsn = str(pg_temp_db["dsn"])
    return PostgresStore(dsn=dsn), PostgresStore(dsn=dsn)


def _services_with_stores(
    stores: tuple[SandboxStore, SandboxStore],
) -> tuple[SandboxService, SandboxService]:
    primary = SandboxService()
    competitor = SandboxService()
    # Direct injection keeps the service boundary real while selecting an
    # isolated adapter pair instead of relying on process-global configuration.
    primary._orch._store = stores[0]
    competitor._orch._store = stores[1]
    return primary, competitor


def _docker_spec(command: str) -> RunSpec:
    return RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["echo", command],
    )


def test_background_claim_is_deferred_until_dispatch_across_store_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shared_store_pair: tuple[SandboxStore, SandboxStore],
) -> None:
    _configure_background_execution(monkeypatch, tmp_path)
    _force_docker_preflight_available(monkeypatch)
    submitted: list[Callable[[], None]] = []
    executed: list[str] = []

    def _fake_start_run(
        self: DockerRunner,
        run_id: str,
        spec: RunSpec,
        workspace_path: str | None,
    ) -> RunStatus:
        del self, workspace_path
        executed.append(run_id)
        now = datetime.now(timezone.utc)
        return RunStatus(
            id=run_id,
            phase=RunPhase.completed,
            runtime=RuntimeType.docker,
            base_image=spec.base_image,
            exit_code=0,
            started_at=now,
            finished_at=now,
            message="ok",
        )

    monkeypatch.setattr(DockerRunner, "start_run", _fake_start_run)
    primary, competitor = _services_with_stores(shared_store_pair)
    monkeypatch.setattr(primary, "_submit_background_worker", submitted.append)
    try:
        run = primary.start_run_scaffold(
            user_id="adapter-deferred-claim",
            spec=_docker_spec("deferred"),
            spec_version="1.0",
            idem_key=None,
            raw_body={"command": ["echo", "deferred"]},
        )

        queued = competitor.get_run(run.id)
        assert queued is not None
        assert queued.phase == RunPhase.queued
        assert queued.claim_owner is None
        assert len(submitted) == 1

        submitted[0]()

        completed = competitor.get_run(run.id)
        assert completed is not None
        assert completed.phase == RunPhase.completed
        assert executed == [run.id]
        assert run.id not in primary._background_pending_run_ids
    finally:
        primary.shutdown()
        competitor.shutdown()


def test_queued_policy_hash_is_durable_across_store_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shared_store_pair: tuple[SandboxStore, SandboxStore],
) -> None:
    _configure_background_execution(monkeypatch, tmp_path)
    _force_docker_preflight_available(monkeypatch)
    submitted: list[Callable[[], None]] = []
    observed_policy_hashes: list[str | None] = []
    primary, competitor = _services_with_stores(shared_store_pair)
    run_ids: list[str] = []
    original_enqueue = primary._orch.enqueue_run

    def _tracking_enqueue(*args: Any, **kwargs: Any) -> RunStatus:
        queued = original_enqueue(*args, **kwargs)
        run_ids.append(queued.id)
        return queued

    def _submit(worker: Callable[[], None]) -> None:
        persisted_before_dispatch = competitor.get_run(run_ids[-1])
        assert persisted_before_dispatch is not None
        observed_policy_hashes.append(persisted_before_dispatch.policy_hash)
        submitted.append(worker)

    monkeypatch.setattr(primary._orch, "enqueue_run", _tracking_enqueue)
    monkeypatch.setattr(primary, "_submit_background_worker", _submit)
    try:
        run = primary.start_run_scaffold(
            user_id="adapter-policy-hash",
            spec=_docker_spec("durable"),
            spec_version="1.0",
            idem_key=None,
            raw_body={"command": ["echo", "durable"]},
        )

        persisted = competitor.get_run(run.id)
        assert run.policy_hash
        assert persisted is not None
        assert persisted.policy_hash == run.policy_hash
        assert observed_policy_hashes == [run.policy_hash]
        assert len(submitted) == 1
    finally:
        primary.shutdown()
        competitor.shutdown()


def test_concurrent_queued_idempotent_replays_submit_once_across_store_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shared_store_pair: tuple[SandboxStore, SandboxStore],
) -> None:
    _configure_background_execution(monkeypatch, tmp_path)
    _force_docker_preflight_available(monkeypatch)
    submitted: list[Callable[[], None]] = []
    primary, competitor = _services_with_stores(shared_store_pair)
    monkeypatch.setattr(primary, "_submit_background_worker", submitted.append)
    raw_body = {"command": ["echo", "idempotent"]}
    try:
        first = primary.start_run_scaffold(
            user_id="adapter-idempotent",
            spec=_docker_spec("idempotent"),
            spec_version="1.0",
            idem_key="same-run",
            raw_body=raw_body,
        )
        replay_count = 4
        replay_barrier = threading.Barrier(replay_count)

        def _replay(_: int) -> RunStatus:
            replay_barrier.wait(timeout=5)
            return primary.start_run_scaffold(
                user_id="adapter-idempotent",
                spec=_docker_spec("idempotent"),
                spec_version="1.0",
                idem_key="same-run",
                raw_body=raw_body,
            )

        with ThreadPoolExecutor(max_workers=replay_count) as executor:
            replays = list(executor.map(_replay, range(replay_count)))

        assert {replayed.id for replayed in replays} == {first.id}
        assert {replayed.phase for replayed in replays} == {RunPhase.queued}
        assert competitor.get_run(first.id) is not None
        assert len(submitted) == 1
        assert first.id in primary._background_pending_run_ids

        primary.shutdown()

        assert first.id not in primary._background_pending_run_ids
    finally:
        primary.shutdown()
        competitor.shutdown()


def test_background_callback_is_fenced_by_competing_admission_across_store_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shared_store_pair: tuple[SandboxStore, SandboxStore],
) -> None:
    _configure_background_execution(monkeypatch, tmp_path)
    _force_docker_preflight_available(monkeypatch)
    submitted: list[Callable[[], None]] = []
    executed: list[str] = []
    monkeypatch.setattr(
        DockerRunner,
        "start_run",
        lambda self, run_id, spec, workspace_path: executed.append(run_id),
    )
    primary, competitor = _services_with_stores(shared_store_pair)
    monkeypatch.setattr(primary, "_submit_background_worker", submitted.append)
    try:
        run = primary.start_run_scaffold(
            user_id="adapter-competing-admission",
            spec=_docker_spec("fenced"),
            spec_version="1.0",
            idem_key=None,
            raw_body={"command": ["echo", "fenced"]},
        )
        claimed = competitor._orch.try_claim_run(
            run.id,
            worker_id=competitor._claim_worker_id,
            lease_seconds=30,
        )
        assert claimed is not None
        admitted = competitor._orch.try_admit_run_start(
            run.id,
            worker_id=competitor._claim_worker_id,
            max_active_runs=1,
            lease_seconds=30,
        )
        assert admitted is not None
        assert admitted.phase == RunPhase.starting

        submitted[0]()

        fenced = primary.get_run(run.id)
        assert fenced is not None
        assert fenced.phase == RunPhase.starting
        assert fenced.claim_owner == competitor._claim_worker_id
        assert executed == []
    finally:
        primary.shutdown()
        competitor.shutdown()


def test_submission_failure_does_not_regress_competing_admission_across_store_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shared_store_pair: tuple[SandboxStore, SandboxStore],
) -> None:
    _configure_background_execution(monkeypatch, tmp_path)
    _force_docker_preflight_available(monkeypatch)
    primary, competitor = _services_with_stores(shared_store_pair)
    run_ids: list[str] = []
    original_enqueue = primary._orch.enqueue_run

    def _tracking_enqueue(*args: Any, **kwargs: Any) -> RunStatus:
        queued = original_enqueue(*args, **kwargs)
        run_ids.append(queued.id)
        return queued

    def _failing_submit(worker: Callable[[], None]) -> None:
        del worker
        run_id = run_ids[-1]
        claimed = competitor._orch.try_claim_run(
            run_id,
            worker_id=competitor._claim_worker_id,
            lease_seconds=30,
        )
        assert claimed is not None
        admitted = competitor._orch.try_admit_run_start(
            run_id,
            worker_id=competitor._claim_worker_id,
            max_active_runs=1,
            lease_seconds=30,
        )
        assert admitted is not None
        assert admitted.phase == RunPhase.starting
        raise RuntimeError("synthetic submission failure")

    monkeypatch.setattr(primary._orch, "enqueue_run", _tracking_enqueue)
    monkeypatch.setattr(primary, "_submit_background_worker", _failing_submit)
    try:
        returned = primary.start_run_scaffold(
            user_id="adapter-submission-failure",
            spec=_docker_spec("racing"),
            spec_version="1.0",
            idem_key=None,
            raw_body={"command": ["echo", "racing"]},
        )

        persisted = primary.get_run(returned.id)
        assert persisted is not None
        assert returned.phase == RunPhase.starting
        assert persisted.phase == RunPhase.starting
        assert persisted.claim_owner == competitor._claim_worker_id
        assert returned.id not in primary._background_pending_run_ids
    finally:
        primary.shutdown()
        competitor.shutdown()
