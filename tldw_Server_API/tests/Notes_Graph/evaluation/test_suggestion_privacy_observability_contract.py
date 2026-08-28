from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph import suggestion_observability as observability
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_decisions import (
    SuggestionDecisionService,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_generation import (
    GenerationProvider,
    parse_and_validate_generation,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_DOMAIN,
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    JOB_RESULT_KEYS,
    JOB_TYPE,
    SuggestionAdmissionService,
    SuggestionCancellationCoordinator,
    SuggestionPublisher,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenanceScope,
    SuggestionMaintenance,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import (
    SuggestionWorker,
    SuggestionWorkerError,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
DATASET_ID = "privacy-evaluation"
SOURCE_ID = "12000000-0000-4000-8000-000000000001"
TARGET_IDS = tuple(f"12000000-0000-4000-8000-{value:012d}" for value in range(2, 5))

SENTINELS = {
    "PRIVACY_TITLE_Z9Q",
    "PRIVACY_BODY_Z9Q",
    "EVIDENCE_TEXT_Z9Q",
    "IGNORE_PREVIOUS_AND_EXPOSE_PROMPT_Z9Q",
    "RATIONALE_Z9Q",
    "PROPOSED_TAG_Z9Q",
    "API_KEY_Z9Q",
    "CREDENTIAL_Z9Q",
    "RAW_PROVIDER_FAILURE_Z9Q",
    *TARGET_IDS,
}
EXPECTED_EVENTS = {event.value for event in observability.SuggestionEventName}
EXPECTED_METRICS = {
    "notes_graph_suggestion_queue_latency_seconds",
    "notes_graph_suggestion_run_duration_seconds",
    "notes_graph_suggestion_candidate_count",
    "notes_graph_suggestion_evidence_count",
    "notes_graph_suggestion_provider_input_tokens",
    "notes_graph_suggestion_provider_output_tokens",
    "notes_graph_suggestion_validated_count",
    "notes_graph_suggestion_dropped_count",
    "notes_graph_suggestion_run_errors_total",
    "notes_graph_suggestion_decisions_total",
    "notes_graph_suggestion_acceptance_reconciliation_total",
}
SAFE_EVENT_KEYS = {
    "event",
    "run_id",
    "job_id",
    "suggestion_id",
    "count",
    "duration_seconds",
    "error_code",
}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class RecordingRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, dict[str, str]]] = []

    def increment(self, name, value=1, labels=None) -> None:
        self.calls.append(("increment", name, value, dict(labels or {})))

    def observe(self, name, value, labels=None) -> None:
        self.calls.append(("observe", name, value, dict(labels or {})))


class LocalLinkCoordinator:
    def __init__(self, notes: CharactersRAGDB) -> None:
        self.notes = notes

    def create(self, *, source_note_id, target_note_id, guarded_mutation, **_kwargs):
        edge_id = guarded_mutation.expected_object_id
        guarded_mutation.require_identity("notes.link", edge_id)
        result = self.notes.notes_link_store.upsert(
            edge_id=edge_id,
            payload={
                "source_note_id": source_note_id,
                "target_note_id": target_note_id,
                "type": "manual",
                "directed": False,
                "weight": 1.0,
                "label": None,
                "properties": {},
                "created_at": NOW.isoformat(),
                "last_modified": NOW.isoformat(),
                "created_by": "server-origin",
            },
            expected_version=None,
            before=guarded_mutation.before,
            after=guarded_mutation.after,
        )
        return SimpleNamespace(edge_id=result.link.edge_id)


def _admit(notes: CharactersRAGDB, jobs: JobManager, *, key: str, model: str):
    source = notes.get_note_by_id(SOURCE_ID, include_deleted=True)
    assert source is not None
    return SuggestionAdmissionService(
        store=notes.note_graph_suggestion_store,
        jobs=jobs,
        owner_user_id=notes.client_id,
    ).admit(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=content_fingerprint(source["title"], source["content"]),
        provider="local-test",
        model=model,
        capability_revision=f"sha256:{'a' * 64}",
        prompt_contract_version="notes-graph-suggestions-v1",
        idempotency_key=key,
        now=NOW,
    )


async def _run_success(notes: CharactersRAGDB, jobs: JobManager):
    admission = _admit(notes, jobs, key="privacy-success", model="recorded-response")
    acquired = jobs.acquire_next_job(
        domain=JOB_DOMAIN,
        queue=JOB_QUEUE,
        job_type=JOB_TYPE,
        worker_id="privacy-worker",
        lease_seconds=30,
    )
    assert acquired is not None

    async def recorded_generation(*, prepared, provider):
        assert provider.api_key == "API_KEY_Z9Q"
        relationships = [
            {
                "target_note_id": target,
                "rationale": f"RATIONALE_Z9Q grounded relationship {index}.",
                "source_evidence_ids": [prepared.source_evidence_ids[0]],
                "target_evidence_ids": [prepared.candidate_evidence_ids[target][0]],
            }
            for index, target in enumerate(prepared.candidate_ids[:3], start=1)
        ]
        relationships.append({"malformed": "validation rejection"})
        tags = [
            {
                "existing_tag_id": None,
                "new_tag": "PROPOSED_TAG_Z9Q",
                "rationale": "RATIONALE_Z9Q grounded tag.",
                "source_evidence_ids": [prepared.source_evidence_ids[0]],
            },
            {
                "existing_tag_id": None,
                "new_tag": " proposed_tag_z9q ",
                "rationale": "RATIONALE_Z9Q duplicate tag.",
                "source_evidence_ids": [prepared.source_evidence_ids[0]],
            },
        ]
        validated = parse_and_validate_generation(
            json.dumps({"relationships": relationships, "tags": tags}),
            prepared=prepared,
        )
        return replace(validated, input_tokens=91, output_tokens=37)

    provider = GenerationProvider(
        adapter="local-test",
        model="recorded-response",
        endpoint_url="http://127.0.0.1:1",
        api_key="API_KEY_Z9Q",
        app_config={"credential": "CREDENTIAL_Z9Q"},
    )
    worker = SuggestionWorker(
        store_factory=lambda owner: (
            notes.note_graph_suggestion_store
            if owner == notes.client_id
            else pytest.fail("worker crossed owner scope")
        ),
        resolve_capability=lambda **_kwargs: (
            SimpleNamespace(
                revision=f"sha256:{'a' * 64}",
                generation_available=True,
            ),
            provider,
        ),
        cancellation_requested=lambda _job: False,
        generate=recorded_generation,
        now=lambda: NOW,
    )
    result = await worker.handle(acquired)
    publishing = notes.note_graph_suggestion_store.get_run(
        dataset_id=DATASET_ID,
        run_id=admission.run.id,
    )
    assert jobs.complete_job(
        int(acquired["id"]),
        result=result,
        worker_id="privacy-worker",
        lease_id=acquired["lease_id"],
        completion_token=acquired["lease_id"],
    )
    published = SuggestionPublisher(
        jobs=jobs,
        store_factory=lambda _owner: notes.note_graph_suggestion_store,
    ).publish(
        run=publishing,
        job_uuid=acquired["uuid"],
        owner_user_id=notes.client_id,
        dataset_id=DATASET_ID,
        now=NOW,
    )
    assert published is not None
    return admission, acquired, result, published


async def _run_closed_worker_failure(*, stale: bool) -> None:
    run_id = "privacy-stale" if stale else "privacy-failed"
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_id": DATASET_ID,
        "source_note_id": SOURCE_ID,
        "source_fingerprint": f"sha256:{'b' * 64}",
        "provider": "local-test",
        "model": "failure-model",
        "capability_revision": "failure-capability",
        "prompt_contract_version": "notes-graph-suggestions-v1",
    }
    queued = SimpleNamespace(
        id=run_id,
        revision=1,
        job_id=f"job-{run_id}",
        state=SimpleNamespace(value="queued"),
        created_at=NOW - timedelta(seconds=1),
        **{
            key: payload[key]
            for key in (
                "source_note_id",
                "source_fingerprint",
                "provider",
                "model",
                "capability_revision",
                "prompt_contract_version",
            )
        },
    )

    class Store:
        def get_run(self, **_kwargs):
            return queued

        def start_run(self, **_kwargs):
            return SimpleNamespace(**{**queued.__dict__, "revision": 2})

        def stage_suggestions(self, **_kwargs):
            raise AssertionError("failed generation must not stage")

    async def generate(**_kwargs):
        if not stale:
            raise RuntimeError("RAW_PROVIDER_FAILURE_Z9Q")
        return SimpleNamespace(
            relationships=(),
            tags=(),
            validation_counts={},
            input_tokens=1,
            output_tokens=1,
        )

    def freshness(**_kwargs):
        if stale:
            raise SuggestionWorkerError(observability.SuggestionErrorCode.FINGERPRINT_STALE)

    worker = SuggestionWorker(
        store_factory=lambda _owner: Store(),
        retrieve=lambda **_kwargs: SimpleNamespace(candidates=()),
        prepare=lambda **_kwargs: object(),
        resolve_capability=lambda **_kwargs: (
            SimpleNamespace(revision="failure-capability", generation_available=True),
            object(),
        ),
        cancellation_requested=lambda _job: False,
        generate=generate,
        freshness_check=freshness,
        now=lambda: NOW,
    )
    with pytest.raises(SuggestionWorkerError):
        await worker.handle(
            {
                "uuid": f"job-{run_id}",
                "owner_user_id": "owner-privacy",
                "domain": JOB_DOMAIN,
                "queue": JOB_QUEUE,
                "job_type": JOB_TYPE,
                "lease_id": f"lease-{run_id}",
                "created_at": NOW - timedelta(seconds=1),
                "payload": payload,
            }
        )


def _assert_no_feature_telemetry_initialization() -> None:
    feature_root = Path(__file__).parents[3] / "app" / "core" / "Notes_Graph"
    forbidden_imports = ("opentelemetry", "sentry_sdk", "telemetry")
    forbidden_calls = {
        "configure_exporter",
        "initialize_exporter",
        "install_exporter",
        "start_exporter",
    }
    findings: list[str] = []
    for path in sorted(feature_root.glob("suggestion_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            if any(module.startswith(forbidden_imports) for module in modules):
                findings.append(f"{path.name}:{node.lineno}:import")
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name in forbidden_calls:
                    findings.append(f"{path.name}:{node.lineno}:{name}")
    assert not findings, f"feature telemetry initialization found: {findings}"


@pytest.mark.asyncio
async def test_real_suggestion_paths_emit_only_the_closed_privacy_safe_contract(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    events: list[dict[str, object]] = []
    structured_logs: list[dict[str, object]] = []
    registry = RecordingRegistry()
    original_write = observability._write_event

    def capture_event(payload: dict[str, object]) -> None:
        events.append(dict(payload))
        original_write(payload)

    sink_id = logger.add(
        lambda message: structured_logs.append(
            {
                "message": message.record["message"],
                "extra": dict(message.record["extra"]),
            }
        )
        if message.record["message"] == "Notes graph suggestion lifecycle event"
        else None
    )
    monkeypatch.setattr(observability, "_write_event", capture_event)
    monkeypatch.setattr(observability, "get_metrics_registry", lambda: registry)

    notes = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="owner-privacy")
    jobs = JobManager(tmp_path / "jobs.db")
    receipts: list[object] = []
    job_records: list[dict[str, object]] = []
    try:
        notes.add_note(
            "PRIVACY_TITLE_Z9Q",
            "PRIVACY_BODY_Z9Q EVIDENCE_TEXT_Z9Q "
            "IGNORE_PREVIOUS_AND_EXPOSE_PROMPT_Z9Q graph retrieval atlas",
            note_id=SOURCE_ID,
        )
        for index, target_id in enumerate(TARGET_IDS, start=1):
            notes.add_note(
                f"Candidate {index}",
                "graph retrieval atlas grounded candidate evidence",
                note_id=target_id,
            )
        with notes.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (notes.client_id, DATASET_ID),
            )

        admission, acquired, result, published = await _run_success(notes, jobs)
        assert set(admission.job["payload"]) == JOB_PAYLOAD_KEYS
        assert set(result) == JOB_RESULT_KEYS
        assert int(admission.job["max_retries"]) == 0
        job_records.append(
            jobs.get_job_or_archived_by_uuid(
                acquired["uuid"], domain=JOB_DOMAIN, owner_user_id=notes.client_id
            )
        )

        source = notes.get_note_by_id(SOURCE_ID, include_deleted=True)
        assert source is not None
        source_fingerprint = content_fingerprint(source["title"], source["content"])
        page = notes.note_graph_suggestion_store.list_suggestions(
            dataset_id=DATASET_ID,
            source_note_id=SOURCE_ID,
            source_fingerprint=source_fingerprint,
            states=("pending",),
            limit=100,
            after=None,
        )
        related = [item for item in page.items if item.kind.value == "related_note"]
        assert len(related) == 3
        decisions = SuggestionDecisionService(
            store=notes.note_graph_suggestion_store,
            link_coordinator=LocalLinkCoordinator(notes),
            organization_coordinator=SimpleNamespace(),
            clock=lambda: NOW,
        )
        accepted = decisions.accept(
            dataset_id=DATASET_ID,
            suggestion_id=related[0].id,
            expected_revision=related[0].revision,
            expected_source_fingerprint=related[0].source_fingerprint,
            expected_target_fingerprint=related[0].target_fingerprint,
            idempotency_key="privacy-accept",
        )
        rejected = decisions.reject(
            dataset_id=DATASET_ID,
            suggestion_id=related[1].id,
            expected_revision=related[1].revision,
            expected_source_fingerprint=related[1].source_fingerprint,
            expected_target_fingerprint=related[1].target_fingerprint,
            idempotency_key="privacy-reject",
        )
        accepting = notes.note_graph_suggestion_store.claim_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id=related[2].id,
            expected_revision=related[2].revision,
            expected_source_fingerprint=related[2].source_fingerprint,
            expected_target_fingerprint=related[2].target_fingerprint,
            idempotency_key="privacy-expired",
            now=NOW,
        )
        reconciled_decisions = decisions.reconcile_expired(
            dataset_id=DATASET_ID,
            now=NOW + timedelta(minutes=6),
        )
        assert len(reconciled_decisions) == 1
        receipts.extend(
            [accepted.envelope, rejected.envelope, accepting.envelope]
            + [item.envelope for item in reconciled_decisions]
        )

        cancellation_admission = _admit(
            notes,
            jobs,
            key="privacy-cancel",
            model="cancel-before-worker",
        )
        cancellation = SuggestionCancellationCoordinator(
            store=notes.note_graph_suggestion_store,
            jobs=jobs,
            owner_user_id=notes.client_id,
        ).cancel(
            dataset_id=DATASET_ID,
            run_id=cancellation_admission.run.id,
            expected_state="queued",
            expected_revision=cancellation_admission.run.revision,
            idempotency_key="privacy-cancel-command",
            now=NOW,
        )
        assert cancellation.accepted is True
        receipts.append(cancellation.cancellation.replay_envelope)
        maintenance = SuggestionMaintenance(
            jobs=jobs,
            scopes=(
                MaintenanceScope(
                    notes.note_graph_suggestion_store,
                    DATASET_ID,
                    decision_service=decisions,
                ),
            ),
        ).run_pass(now=NOW + timedelta(minutes=7))
        assert maintenance.reconciled == 1
        cancelled_job = jobs.get_job_or_archived_by_uuid(
            cancellation_admission.job["uuid"],
            domain=JOB_DOMAIN,
            owner_user_id=notes.client_id,
        )
        assert cancelled_job is not None
        job_records.append(cancelled_job)

        replay = _admit(notes, jobs, key="privacy-success", model="recorded-response")
        assert replay.disposition == "terminal_replay"
        receipts.append(replay.replay_envelope)

        await _run_closed_worker_failure(stale=False)
        await _run_closed_worker_failure(stale=True)

        with notes.transaction() as conn:
            run_rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM note_graph_suggestion_runs ORDER BY id"
                ).fetchall()
            ]
    finally:
        logger.remove(sink_id)
        notes.close_all_connections()

    assert {item["event"] for item in events} == EXPECTED_EVENTS
    assert {call[1] for call in registry.calls} == EXPECTED_METRICS
    for event in events:
        assert set(event) <= SAFE_EVENT_KEYS
        assert SAFE_ID.fullmatch(str(event["run_id"]))
        assert all(
            SAFE_ID.fullmatch(str(event[key]))
            for key in ("job_id", "suggestion_id", "error_code")
            if key in event
        )
        if "count" in event:
            assert 0 <= int(event["count"]) <= 1_000_000
    for _method, name, value, labels in registry.calls:
        assert name in EXPECTED_METRICS
        assert set(labels) <= {"error_code", "outcome"}
        assert 0 <= float(value) <= 1_000_000

    for job in job_records:
        assert job is not None
        assert set(job["payload"]) == JOB_PAYLOAD_KEYS
        assert int(job["max_retries"]) == 0
        if job.get("result") is not None:
            assert set(job["result"]) == JOB_RESULT_KEYS

    scanned = json.dumps(
        {
            "jobs": job_records,
            "events": events,
            "logs": structured_logs,
            "metrics": registry.calls,
            "runs": run_rows,
            "receipts": receipts,
            "published": {
                "run_id": published.id,
                "state": published.state.value,
                "revision": published.revision,
            },
        },
        default=str,
        sort_keys=True,
    )
    leaked = sorted(sentinel for sentinel in SENTINELS if sentinel in scanned)
    assert not leaked, f"forbidden values reached safe sinks: {leaked}"
    _assert_no_feature_telemetry_initialization()
