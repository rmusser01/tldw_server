"""Owned run responses derive missing progress without changing execution results."""

from collections import OrderedDict

import pytest
import pytest_asyncio
from fastapi import HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_crud
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunProgress
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService


@pytest_asyncio.fixture
async def real_owned_run(tmp_path, monkeypatch):
    """Execute the real no-model evaluator and retain actual owned DB/service reads."""
    db_path = str(tmp_path / "progress.db")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", db_path)
    service = UnifiedEvaluationService(db_path, enable_webhooks=False)
    await service.initialize()
    owner = User(id="progress-owner", username="owner")
    foreign = User(id="another-owner", username="other")
    # Supply actual services through the existing per-user content cache; both
    # users share this controlled DB so real created_by filtering is exercised.
    monkeypatch.setattr(unified_evaluation_service, "_service_instances_by_user",
                        OrderedDict([(owner.id_str, service), (foreign.id_str, service)]))
    dataset = service.db.create_dataset(
        name="Two literal outputs", created_by=owner.id_str,
        samples=[{"input": {"output": "alpha"}, "expected": "alpha"},
                 {"input": {"output": "beta"}, "expected": "gamma"}],
    )
    evaluation = service.db.create_evaluation(
        name="Exact match progress", eval_type="exact_match", dataset_id=dataset,
        eval_spec={"metrics": ["exact_match"], "case_sensitive": True}, created_by=owner.id_str,
    )
    run = service.db.create_run(evaluation)
    try:
        await service.runner.run_evaluation(
            run, evaluation, {"config": {"batch_size": 1}}, background=False,
        )
        yield service, owner, foreign, evaluation, run
    finally:
        await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["detail", "list"])
async def test_completed_exact_match_progress_is_100_without_changing_results(real_owned_run, read):
    service, owner, _foreign, evaluation, run = real_owned_run
    saved = service.db.get_run(run, created_by=owner.id_str)
    assert saved["progress"]["completed_samples"] == 2  # nosec B101 - pytest real runner count assertion.
    assert saved["progress"]["total_samples"] == 2  # nosec B101 - pytest real source sample assertion.
    assert "percent_complete" not in saved["progress"]  # nosec B101 - pytest legacy persisted shape assertion.
    if read == "detail":
        response = await evaluations_crud.get_run(run, current_user=owner)
    else:
        response = (await evaluations_crud.list_runs(evaluation, limit=20, after=None, run_status=None,
                                                     current_user=owner)).data[0]
    payload = response.model_dump(mode="json")
    assert payload["progress"]["percent_complete"] == 100.0  # nosec B101 - pytest completed response assertion.
    assert payload["progress"]["current_sample"] is None  # nosec B101 - pytest preserved nullable field assertion.
    assert payload["status"] == "completed"  # nosec B101 - pytest retained terminal state assertion.
    assert [sample["scores"]["exact_match"] for sample in payload["results"]["sample_results"]] == [1.0, 0.0]  # nosec B101 - pytest unchanged scores assertion.
    assert [sample["passed"] for sample in payload["results"]["sample_results"]] == [True, False]  # nosec B101 - pytest unchanged sample verdict assertion.
    assert payload["results"]["aggregate"]["mean_score"] == 0.5  # nosec B101 - pytest unchanged mean assertion.
    assert payload["results"]["aggregate"]["pass_rate"] == 0.5  # nosec B101 - pytest unchanged pass rate assertion.
    assert payload["usage"]["total_tokens"] == 0  # nosec B101 - pytest no-model usage assertion.
    assert service.db.get_run(run, created_by=owner.id_str) == saved  # nosec B101 - pytest no read mutation assertion.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_progress_read_preserves_foreign_owner_denial(real_owned_run):
    _service, _owner, foreign, evaluation, run = real_owned_run
    with pytest.raises(HTTPException) as rejected:
        await evaluations_crud.get_run(run, current_user=foreign)
    assert rejected.value.status_code == 404  # nosec B101 - pytest real foreign run denial assertion.
    response = await evaluations_crud.list_runs(evaluation, limit=20, after=None, run_status=None,
                                                current_user=foreign)
    assert response.data == []  # nosec B101 - pytest foreign run list exclusion assertion.


@pytest.mark.unit
@pytest.mark.parametrize("counts,want", [
    ({}, 0.0),
    ({"completed_samples": 0, "total_samples": 2}, 0.0),
    ({"completed_samples": 1, "total_samples": 2}, 50.0),
    ({"completed_samples": 2, "total_samples": 3}, 200.0 / 3),
    ({"completed_samples": 2, "total_samples": 2}, 100.0),
    ({"completed_samples": 3, "total_samples": 2}, 100.0),
    ({"completed_samples": 2, "total_samples": 0}, 0.0),
    ({"completed_samples": "1", "total_samples": "2"}, 50.0),
    ({"completed_samples": 10**400, "total_samples": 10**400}, 100.0),
])
def test_missing_percentage_uses_validated_counts_and_stays_bounded(counts, want):
    progress = RunProgress(**counts)
    assert progress.percent_complete == pytest.approx(want)  # nosec B101 - pytest independently calculated percentage assertion.


@pytest.mark.unit
@pytest.mark.parametrize("percent", [0.0, 12.5, 100.0])
def test_explicit_valid_percentage_is_preserved(percent):
    progress = RunProgress(completed_samples=2, total_samples=2, percent_complete=percent, current_sample=1)
    assert progress.percent_complete == percent  # nosec B101 - pytest supplied percentage compatibility assertion.
    assert progress.current_sample == 1  # nosec B101 - pytest supplied current sample assertion.


@pytest.mark.unit
@pytest.mark.parametrize("invalid", [
    {"completed_samples": -1}, {"total_samples": -1},
    {"completed_samples": "invalid"}, {"total_samples": "invalid"},
    {"percent_complete": -1}, {"percent_complete": 101}, {"percent_complete": None},
])
def test_progress_retains_count_type_and_percentage_validation(invalid):
    with pytest.raises(ValidationError):
        RunProgress(**invalid)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pending_run_without_progress_remains_null(real_owned_run):
    service, owner, _foreign, evaluation, _run = real_owned_run
    pending = service.db.create_run(evaluation)
    response = await evaluations_crud.get_run(pending, current_user=owner)
    assert response.progress is None  # nosec B101 - pytest absent progress compatibility assertion.
    assert response.status.value == "pending"  # nosec B101 - pytest pending state compatibility assertion.

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["detail", "list"])
async def test_saved_run_keeps_executed_dataset_and_recipe_after_source_changes(real_owned_run, read):
    from tldw_Server_API.app.core.Evaluations.recipes.dataset_snapshot import build_dataset_content_hash

    service, owner, _foreign, evaluation, run = real_owned_run
    source = service.db.get_evaluation(evaluation, created_by=owner.id_str)
    dataset_id = source["dataset_id"]
    samples = service.db.get_dataset(dataset_id, created_by=owner.id_str)["samples"]
    recipe_hash = build_dataset_content_hash({"eval_type": source["eval_type"], "eval_spec": source["eval_spec"]})
    service.db.update_evaluation(evaluation, {"eval_spec": {"case_sensitive": False}}, created_by=owner.id_str)
    service.db.delete_dataset(dataset_id, created_by=owner.id_str)
    if read == "detail":
        response = await evaluations_crud.get_run(run, current_user=owner)
    else:
        response = (await evaluations_crud.list_runs(evaluation, limit=20, after=None, run_status=None,
                                                     current_user=owner)).data[0]
    saved = response.model_dump(mode="json")
    assert saved["results"]["provenance"] == {  # nosec B101 - pytest executed snapshot identity assertion.
        "evaluation_id": evaluation, "evaluation_type": "exact_match",
        "evaluation_content_hash": recipe_hash,
        "dataset_id": dataset_id, "dataset_source": "dataset",
        "dataset_content_hash": build_dataset_content_hash(samples),
    }
    assert [(row["input"], row["expected"]) for row in saved["results"]["sample_results"]] == [  # nosec B101 - pytest actual retained sample assertion.
        ({"output": "alpha"}, "alpha"), ({"output": "beta"}, "gamma"),
    ]
    assert [row["passed"] for row in saved["results"]["sample_results"]] == [True, False]  # nosec B101 - pytest retained actual verdict assertion.
    assert saved["results"]["aggregate"]["mean_score"] == 0.5  # nosec B101 - pytest retained aggregate assertion.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_override_snapshot_does_not_claim_evaluations_original_dataset(real_owned_run):
    from tldw_Server_API.app.core.Evaluations.recipes.dataset_snapshot import build_dataset_content_hash

    service, owner, _foreign, evaluation, _original_run = real_owned_run
    samples = [{"input": {"output": "override"}, "expected": "override"}]
    run = service.db.create_run(evaluation, target_model="precomputed")
    await service.runner.run_evaluation(run, evaluation, {
        "created_by": owner.id_str, "dataset_override": {"samples": samples},
    }, background=False)
    response = await evaluations_crud.get_run(run, current_user=owner)
    provenance = response.results["provenance"]
    assert provenance["dataset_id"] is None  # nosec B101 - pytest truthful override attribution assertion.
    assert provenance["dataset_source"] == "override"  # nosec B101 - pytest actual source assertion.
    assert provenance["dataset_content_hash"] == build_dataset_content_hash(samples)  # nosec B101 - pytest actual override fingerprint assertion.
    assert response.results["sample_results"][0]["input"] == {"output": "override"}  # nosec B101 - pytest persisted override assertion.


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_batch_keeps_sample_before_evaluator_mutation(real_owned_run):
    from copy import deepcopy

    service, _owner, _foreign, _evaluation, _run = real_owned_run
    sample = {"input": {"output": "original"}, "expected": "wanted", "sample_id": "source-row"}
    original = deepcopy(sample)

    async def fail_after_mutation(sample, **_kwargs):
        sample["input"]["output"] = "changed by evaluator"
        raise ValueError("evaluation rejected")

    rows = await service.runner._process_batch(
        [sample], fail_after_mutation, {}, {}, max_workers=1,
    )
    assert rows[0]["input"] == original["input"]  # nosec B101 - pytest pre-execution actual input assertion.
    assert rows[0]["expected"] == original["expected"]  # nosec B101 - pytest failed example expected value assertion.
    assert rows[0]["dataset_sample_id"] == "source-row"  # nosec B101 - pytest original row identity assertion.
    assert rows[0]["error"] == "evaluation rejected"  # nosec B101 - pytest error contract preservation assertion.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancellation_retains_completed_batch_and_owner_fence(tmp_path, monkeypatch):
    """Cancel a real run while its second evaluator is pending, without timing races."""
    import asyncio

    path = str(tmp_path / "cancel-partial.db")
    service = UnifiedEvaluationService(path, enable_webhooks=False)
    await service.initialize()
    second_entered = asyncio.Event()
    release_second = asyncio.Event()
    task = None
    original = service.runner._eval_exact_match

    async def gate_second(sample, eval_spec, config, sample_id):
        if sample_id == "sample_000001":
            second_entered.set()
            await release_second.wait()
        return await original(sample, eval_spec, config, sample_id)

    monkeypatch.setattr(service.runner, "_get_evaluation_function", lambda *_args: gate_second)
    evaluation = await service.create_evaluation(
        name="cancel_partial", eval_type="exact_match", eval_spec={"metrics": ["exact_match"]},
        dataset=[{"input": {"output": "ORBIT-742"}, "expected": "ORBIT-742"},
                 {"input": {"output": "ORBIT-999"}, "expected": "ORBIT-742"}], created_by="owner",
    )
    try:
        run = await service.create_run(evaluation["id"], "precomputed", config={"batch_size": 1}, created_by="owner")
        await asyncio.wait_for(second_entered.wait(), timeout=2)
        task = service.runner.running_tasks[run["id"]]
        assert await service.cancel_run(run["id"], cancelled_by="owner", created_by="owner")  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        with pytest.raises(asyncio.CancelledError):
            await task
        saved = await service.get_run(run["id"], created_by="owner")
        assert saved["status"] == "cancelled"  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert saved["progress"]["completed_samples"] == 1  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert saved["progress"]["total_samples"] == 2  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert saved["results"] is not None  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert [row["sample_id"] for row in saved["results"]["sample_results"]] == ["sample_000000"]  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert saved["results"]["sample_results"][0]["input"] == {"output": "ORBIT-742"}  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert saved["results"]["sample_results"][0]["passed"] is True  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert saved["results"]["aggregate"]["total_samples"] == 1  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert await service.get_run(run["id"], created_by="another-owner") is None  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
    finally:
        release_second.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_checkpoint_cannot_overwrite_completed_run(real_owned_run):
    service, owner, _foreign, _evaluation, run = real_owned_run
    before = await service.get_run(run, created_by=owner.id_str)
    assert service.db.update_run_progress(run, {"completed_samples": 999}, results={"wrong": True}) is False  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
    assert await service.get_run(run, created_by=owner.id_str) == before  # nosec B101 - Pytest C03 causal behavior/ownership assertion.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runner_stops_when_another_worker_cancelled_before_checkpoint(tmp_path, monkeypatch):
    """A rejected checkpoint must not admit more samples or overwrite cancellation."""
    import asyncio

    service = UnifiedEvaluationService(str(tmp_path / "retired-worker.db"), enable_webhooks=False)
    await service.initialize()
    original = service.runner._eval_exact_match
    called = []
    run_id = None

    async def external_cancel_after_evaluation(sample, eval_spec, config, sample_id):
        called.append(sample_id)
        result = await original(sample, eval_spec, config, sample_id)
        service.db.update_run_status(run_id, "cancelled")
        return result

    monkeypatch.setattr(service.runner, "_get_evaluation_function", lambda *_args: external_cancel_after_evaluation)
    evaluation = await service.create_evaluation(
        name="external_cancel", eval_type="exact_match", eval_spec={},
        dataset=[{"input": {"output": "one"}, "expected": "one"},
                 {"input": {"output": "two"}, "expected": "two"}], created_by="owner",
    )
    run_id = service.db.create_run(evaluation["id"])
    try:
        with pytest.raises(asyncio.CancelledError):
            await service.runner.run_evaluation(run_id, evaluation["id"], {
                "created_by": "owner", "config": {"batch_size": 1},
            }, background=False)
        assert called == ["sample_000000"]  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
        assert service.db.get_run(run_id, created_by="owner")["status"] == "cancelled"  # nosec B101 - Pytest C03 causal behavior/ownership assertion.
    finally:
        await service.shutdown()


@pytest.mark.integration
def test_restricted_postgres_checkpoint_preserves_owner_and_cancelled_output(pg_restricted_backend, tmp_path):
    """The production checkpoint transaction must work through the restricted PG fixture."""
    from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

    db = EvaluationsDatabase(str(tmp_path / "pg-checkpoint.db"), backend=pg_restricted_backend)
    dataset = db.create_dataset(name="checkpoint", samples=[{"input": {"output": "kept"}}], created_by="owner")
    evaluation = db.create_evaluation(name="checkpoint", eval_type="exact_match", eval_spec={}, dataset_id=dataset, created_by="owner")
    run = db.create_run(evaluation)
    db.update_run_status(run, "running")
    progress = {"completed_samples": 1, "total_samples": 2}
    results = {"sample_results": [{"input": {"output": "kept"}, "passed": True}]}
    assert db.update_run_progress(run, progress, results=results, usage={"total_tokens": 0})  # nosec B101 - Real restricted fixture checkpoint.
    saved = db.get_run(run, created_by="owner")
    assert saved["status"] == "running"  # nosec B101 - Checkpoint is not terminal.
    assert saved["progress"] == progress  # nosec B101 - Atomic count retention.
    assert saved["results"] == results  # nosec B101 - Actual output retention.
    db.update_run_status(run, "cancelled")
    assert db.update_run_progress(run, {"completed_samples": 2}, results={}) is False  # nosec B101 - Retired writer refused.
    assert db.get_run(run, created_by="owner")["results"] == results  # nosec B101 - Preserve cancelled output.
    assert db.get_run(run, created_by="owner")["status"] == "cancelled"  # nosec B101 - Preserve cancellation.
    assert db.get_run(run, created_by="foreign") is None  # nosec B101 - Actual owner-filtered read.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_respects_each_sample_budget_beyond_thirty_seconds(tmp_path, monkeypatch):
    """Two individually bounded examples must outlive the former whole-run cap."""
    import asyncio

    db_path = str(tmp_path / "service-deadline.db")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", db_path)
    service = UnifiedEvaluationService(db_path, enable_webhooks=False)
    await service.initialize()
    dataset = service.db.create_dataset(
        name="Two bounded examples", created_by="deadline-owner",
        samples=[{"input": {"output": "alpha"}, "expected": "alpha"},
                 {"input": {"output": "beta"}, "expected": "gamma"}],
    )
    evaluation = service.db.create_evaluation(
        name="Per-example deadline", eval_type="exact_match", dataset_id=dataset,
        eval_spec={"metrics": ["exact_match"]}, created_by="deadline-owner",
    )
    run = service.db.create_run(evaluation)
    exact_match = service.runner._eval_exact_match

    async def bounded_example(*args, **kwargs):
        await asyncio.sleep(15.2)
        return await exact_match(*args, **kwargs)

    monkeypatch.setattr(service.runner, "_eval_exact_match", bounded_example)
    try:
        await asyncio.wait_for(service._run_evaluation_async(
            run, evaluation,
            {"config": {"batch_size": 1, "max_workers": 1, "timeout_seconds": 20}},
            "deadline-owner",
        ), timeout=40)
        saved = service.db.get_run(run, created_by="deadline-owner")
        assert saved["status"] == "completed"  # nosec B101 - pytest real service terminal state.
        assert [row["passed"] for row in saved["results"]["sample_results"]] == [True, False]  # nosec B101 - pytest both real evaluator outputs.
        assert service.circuit_breaker.stats.successful_calls == 1  # nosec B101 - pytest retained breaker accounting.
        assert run not in service.runner.running_tasks  # nosec B101 - pytest task ownership cleanup.
    finally:
        await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_still_enforces_the_example_timeout(tmp_path, monkeypatch):
    """Removing a whole-batch deadline must not permit unbounded examples."""
    import asyncio

    db_path = str(tmp_path / "example-deadline.db")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", db_path)
    service = UnifiedEvaluationService(db_path, enable_webhooks=False)
    await service.initialize()
    evaluation = service.db.create_evaluation(
        name="Example deadline", eval_type="exact_match",
        eval_spec={"metrics": ["exact_match"]}, created_by="deadline-owner",
    )
    run = service.db.create_run(evaluation)
    stopped = asyncio.Event()

    async def slow_example(*_args, **_kwargs):
        try:
            await asyncio.sleep(5)
        finally:
            stopped.set()

    monkeypatch.setattr(service.runner, "_eval_exact_match", slow_example)
    try:
        await asyncio.wait_for(service._run_evaluation_async(
            run, evaluation, {
                "config": {"timeout_seconds": 1},
                "dataset_override": {"samples": [{"input": {"output": "alpha"}, "expected": "alpha"}]},
            }, "deadline-owner",
        ), timeout=5)
        saved = service.db.get_run(run, created_by="deadline-owner")
        assert saved["status"] == "completed"  # nosec B101 - pytest terminal batch with explicit failed example.
        assert saved["results"]["aggregate"]["failed_samples"] == 1  # nosec B101 - pytest timeout counted once.
        assert saved["results"]["sample_results"][0]["error"] == "Timeout after 1.0s"  # nosec B101 - pytest retained per-example deadline.
        assert stopped.is_set()  # nosec B101 - pytest timed-out coroutine cancellation.
        assert run not in service.runner.running_tasks  # nosec B101 - pytest task ownership cleanup.
    finally:
        await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_pipeline_keeps_its_existing_whole_run_deadline(tmp_path, monkeypatch):
    """RAG orchestration has no ordinary per-example guard and keeps its old cap."""
    import asyncio

    db_path = str(tmp_path / "rag-deadline.db")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", db_path)
    service = UnifiedEvaluationService(db_path, enable_webhooks=False)
    await service.initialize()
    evaluation = service.db.create_evaluation(
        name="RAG deadline", eval_type="model_graded",
        eval_spec={"sub_type": "rag_pipeline"}, created_by="deadline-owner",
    )
    run = service.db.create_run(evaluation)
    stopped = asyncio.Event()

    async def slow_pipeline(**_kwargs):
        try:
            await asyncio.sleep(30.2)
            return {"sample_results": []}, {}
        finally:
            stopped.set()

    monkeypatch.setattr(service.runner, "_execute_rag_pipeline_run", slow_pipeline)
    try:
        await asyncio.wait_for(service._run_evaluation_async(
            run, evaluation, {"dataset_override": {"samples": [{"input": "query"}]}},
            "deadline-owner",
        ), timeout=40)
        saved = service.db.get_run(run, created_by="deadline-owner")
        assert saved["status"] == "failed"  # nosec B101 - pytest preserved exceptional-branch deadline.
        assert "RAG pipeline timed out after 30s" in saved["error_message"]  # nosec B101 - pytest useful timeout diagnosis.
        assert stopped.is_set()  # nosec B101 - pytest deadline cancelled pipeline coroutine.
        assert service.circuit_breaker.stats.failed_calls == 1  # nosec B101 - pytest retained breaker failure accounting.
        assert run not in service.runner.running_tasks  # nosec B101 - pytest task ownership cleanup.
    finally:
        await service.shutdown()


@pytest.fixture
def owned_run_webhook(real_owned_run, monkeypatch):
    """Isolate the external webhook sink while cancellation and persistence stay real."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    service = real_owned_run[0]
    webhook = AsyncMock()
    monkeypatch.setattr(service, "enable_webhooks", True)
    monkeypatch.setattr(service, "webhook_manager", SimpleNamespace(send_webhook=webhook))
    return webhook


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_cancel_cleanup_keeps_first_terminal_record(real_owned_run, monkeypatch, owned_run_webhook):
    """Once API cancellation is persisted, worker retirement must not write it again."""
    import asyncio

    service, owner, _foreign, evaluation, _completed_run = real_owned_run
    entered = asyncio.Event()
    release = asyncio.Event()
    original_update = type(service.db).update_run_status
    cancelled_records = []
    webhook = owned_run_webhook

    async def pending_example(**_kwargs):
        entered.set()
        await release.wait()

    def record_update(database, run_id, status, error_message=None):
        result = original_update(database, run_id, status, error_message)
        if status == "cancelled":
            cancelled_records.append(service.db.get_run(run_id, created_by=owner.id_str))
        return result

    monkeypatch.setattr(service.runner, "_get_evaluation_function", lambda *_args: pending_example)
    monkeypatch.setattr(type(service.db), "update_run_status", record_update)
    run = service.db.create_run(evaluation)
    task = asyncio.create_task(
        service._run_evaluation_async(
            run,
            evaluation,
            {"config": {"batch_size": 1, "timeout_seconds": 5}, "webhook_url": "https://example.com/webhook"},
            owner.id_str,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert await service.cancel_run(run, cancelled_by=owner.id_str, created_by=owner.id_str)  # nosec B101 - Real owner cancellation admission.
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(cancelled_records) == 1  # nosec B101 - One durable terminal event, not a second cleanup write.
        assert service.db.get_run(run, created_by=owner.id_str) == cancelled_records[0]  # nosec B101 - Retain the entire first terminal record.
        assert run not in service.runner.running_tasks  # nosec B101 - Real task ownership is released.
        webhook.assert_awaited_once_with(
            user_id=owner.id_str,
            event=unified_evaluation_service.WebhookEvent.EVALUATION_CANCELLED,
            evaluation_id=run,
            data={"run_id": run},
        )
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_failure", [True, False])
async def test_provider_failure_retires_run_after_preserving_batch(tmp_path, monkeypatch, provider_failure):
    """A provider outage stops later work; ordinary bad examples remain retryable rows."""
    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import SummaryProviderError

    db_path = str(tmp_path / "provider-failure.db")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", db_path)
    service = UnifiedEvaluationService(db_path, enable_webhooks=False)
    await service.initialize()
    samples = [
        {"input": {"prompt": "Name the director", "response": value}, "expected": "Mira Vale"}
        for value in ("good", "bad", "later")
    ]
    dataset = service.db.create_dataset(name="Provider failure boundaries", samples=samples, created_by="owner")
    evaluation = service.db.create_evaluation(
        name="Provider failure",
        eval_type="model_graded",
        dataset_id=dataset,
        eval_spec={"sub_type": "response_quality", "model": "ollama:existing-model", "metrics": ["accuracy"]},
        created_by="owner",
    )
    run = service.db.create_run(evaluation)
    calls, failures = [], []

    async def grade(**kwargs):
        calls.append(kwargs["response"])
        if kwargs["response"] == "bad":
            if provider_failure:
                raise SummaryProviderError(code="provider_failure", provider="ollama")
            raise ValueError("Invalid example")
        return {"metrics": {"accuracy": {"score": 1.0}}, "overall_quality": 1.0}

    original_update = type(service.db).update_run_status

    def record_terminal(database, run_id, status, error_message=None):
        result = original_update(database, run_id, status, error_message)
        if run_id == run and status == "failed":
            failures.append(database.get_run(run_id, created_by="owner"))
        return result

    monkeypatch.setattr(service.runner.quality_evaluator, "evaluate", grade)
    monkeypatch.setattr(type(service.db), "update_run_status", record_terminal)
    try:
        await service._run_evaluation_async(
            run,
            evaluation,
            {"config": {"batch_size": 2, "max_workers": 1, "timeout_seconds": 5}},
            "owner",
        )
        saved = service.db.get_run(run, created_by="owner")
        assert saved["status"] == ("failed" if provider_failure else "completed")  # nosec B101 - Real provider-vs-example terminal contract.
        assert calls == (["good", "bad"] if provider_failure else ["good", "bad", "later"])  # nosec B101 - Unattempted later batches must stay unattempted.
        assert saved["progress"]["completed_samples"] == (1 if provider_failure else 2)  # nosec B101 - Actual successful work retained.
        assert saved["progress"]["failed_samples"] == 1  # nosec B101 - Failed example counted exactly once.
        assert saved["progress"]["total_samples"] == 3  # nosec B101 - Original denominator retained.
        rows = saved["results"]["sample_results"]
        assert [row["input"] for row in rows] == [row["input"] for row in samples[: len(calls)]]  # nosec B101 - Real input snapshots survive failure.
        assert rows[0]["passed"] is True  # nosec B101 - Completed grade survives later provider failure.
        assert saved["results"]["provenance"]["dataset_id"] == dataset  # nosec B101 - Actual dataset provenance retained.
        assert saved["results"]["aggregate"]["failed_samples"] == 1  # nosec B101 - Aggregate matches retained failed row.
        assert run not in service.runner.running_tasks  # nosec B101 - Owned worker retired.
        if provider_failure:
            assert rows[1]["provider_error"] == "provider_failure"  # nosec B101 - Typed provider failure remains attributable.
            assert "ollama" in saved["error_message"] and "provider" in saved["error_message"]  # nosec B101 - Sanitized actionable provider identity.
            assert len(failures) == 1  # nosec B101 - Service cleanup cannot rewrite the terminal record.
            assert saved == failures[0]  # nosec B101 - Full first terminal record is stable.
        else:
            assert failures == []  # nosec B101 - Ordinary example failure does not fail the whole run.
    finally:
        await service.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_webhook_failure_is_visible_without_rewriting_outcome(real_owned_run, monkeypatch, status):
    """Only delivery metadata may change after an evaluation has finished."""
    from tldw_Server_API.app.core.Evaluations import eval_runner

    service, owner, foreign, evaluation, run = real_owned_run
    if status == "failed":
        service.db.update_run_status(run, "failed", error_message="Original grading failure")
    before = service.db.get_run(run, created_by=owner.id_str)

    async def unavailable(_url):
        raise OSError("private-token-must-not-be-persisted")

    monkeypatch.setattr(eval_runner.webhook_validator, "resolve_safe_delivery_target_async", unavailable)
    await service.runner._send_webhook("https://unavailable.invalid/hook?token=secret", run, evaluation, status, {})
    after = service.db.get_run(run, created_by=owner.id_str)
    warning = after["error_message"] or ""
    assert "Webhook delivery failed" in warning  # nosec B101 - User-visible delivery failure contract.
    assert "Check the webhook URL and receiver" in warning  # nosec B101 - Actionable sanitized recovery.
    assert "private-token" not in warning and "secret" not in warning  # nosec B101 - Receiver credentials never enter the persisted warning.
    assert {**after, "error_message": before["error_message"]} == before  # nosec B101 - Full terminal record except notification warning retained.
    if status == "failed":
        assert warning.startswith("Original grading failure")  # nosec B101 - Notification cannot erase the execution error.
    await service.runner._send_webhook("https://unavailable.invalid/hook?token=secret", run, evaluation, status, {})
    assert service.db.get_run(run, created_by=owner.id_str) == after  # nosec B101 - Repeated delivery failure is idempotent.
    assert service.db.get_run(run, created_by=foreign.id_str) is None  # nosec B101 - Existing owner isolation remains intact.


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["success", "failure", "warning_write_failure"])
async def test_real_run_retains_grades_through_webhook_delivery(real_owned_run, monkeypatch, delivery):
    """The shared completion path uses the warning writer without changing grading."""
    from types import SimpleNamespace

    from loguru import logger

    from tldw_Server_API.app.core.Evaluations import eval_runner

    service, owner, _foreign, evaluation, _old_run = real_owned_run
    run = service.db.create_run(evaluation)
    sent = []

    async def target(url):
        return url, {}

    async def deliver(**kwargs):
        sent.append(kwargs["json"])
        if delivery != "success":
            raise OSError("delivery-secret-must-not-leak")
        return SimpleNamespace(status_code=204)

    monkeypatch.setattr(eval_runner.webhook_validator, "resolve_safe_delivery_target_async", target)
    monkeypatch.setattr(eval_runner, "afetch", deliver)
    if delivery == "warning_write_failure":

        def broken_write(*_args):
            raise RuntimeError("bookkeeping-secret-must-not-leak")

        monkeypatch.setattr(service.runner.db, "record_webhook_failure", broken_write)
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        await service.runner.run_evaluation(
            run,
            evaluation,
            {
                "config": {"batch_size": 1},
                "webhook_url": "https://receiver.invalid/hook?token=url-secret-must-not-leak",
            },
            background=False,
        )
    finally:
        logger.remove(sink)
    assert not any("secret-must-not-leak" in message for message in messages)  # nosec B101 - Neither delivery nor bookkeeping exceptions may leak receiver secrets.
    saved = service.db.get_run(run, created_by=owner.id_str)
    assert saved["status"] == "completed"  # nosec B101 - Delivery is independent of grading.
    assert [row["passed"] for row in saved["results"]["sample_results"]] == [True, False]  # nosec B101 - Actual evaluator outputs retained.
    assert len(sent) == 1 and sent[0]["run_id"] == run  # nosec B101 - Shared direct delivery is invoked once.
    assert bool(saved["error_message"]) == (delivery == "failure")  # nosec B101 - Warning only on persisted delivery failure.
    assert run not in service.runner.running_tasks  # nosec B101 - Failed notification cannot strand owned work.


def _check_webhook_warning_boundaries(db):
    evaluation = db.create_evaluation(
        name="Webhook boundary", eval_type="exact_match", eval_spec={}, created_by="owner"
    )
    other = db.create_evaluation(name="Other owner", eval_type="exact_match", eval_spec={}, created_by="foreign")
    for status in ("pending", "running", "completed", "failed", "cancelled"):
        run = db.create_run(evaluation)
        db.update_run_status(run, status)
        before = db.get_run(run, created_by="owner")
        assert db.record_webhook_failure(run, other) is False  # nosec B101 - Evaluation identity must match.
        assert db.get_run(run, created_by="owner") == before  # nosec B101 - Cross-evaluation write refused.
        changed = db.record_webhook_failure(run, evaluation)
        after = db.get_run(run, created_by="owner")
        assert changed is (status in ("completed", "failed", "cancelled"))  # nosec B101 - Only terminal notification warnings are admitted.
        assert {**after, "error_message": before["error_message"]} == before  # nosec B101 - No outcome or timestamp write.
        assert db.record_webhook_failure(run, evaluation) is False  # nosec B101 - Single atomic append is idempotent.
        assert db.get_run(run, created_by="foreign") is None  # nosec B101 - Owner-filtered read is unchanged.


@pytest.mark.integration
def test_sqlite_webhook_warning_boundaries(tmp_path):
    from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

    _check_webhook_warning_boundaries(EvaluationsDatabase(str(tmp_path / "webhook.db")))


@pytest.mark.integration
def test_restricted_postgres_webhook_warning_boundaries(pg_restricted_backend, tmp_path):
    from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

    _check_webhook_warning_boundaries(
        EvaluationsDatabase(str(tmp_path / "pg-webhook.db"), backend=pg_restricted_backend)
    )
