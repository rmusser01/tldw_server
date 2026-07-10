from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
    AudioBriefingTriggerResult,
)
from tldw_Server_API.app.core.Watchlists.briefing_fulfillment import (
    FulfillmentResult,
    audio_request_id_for_occurrence,
    fulfill_watchlist_briefing,
    no_material_updates_markdown,
    retry_briefing_stage,
)
from tldw_Server_API.app.core.Watchlists.pipeline import _fulfillment_stats_projection

pytestmark = pytest.mark.unit


def test_read_stages_preserves_terminal_delivery_adapter_history():
    from tldw_Server_API.app.core.Watchlists.briefing_fulfillment import _read_stages

    occurrence = SimpleNamespace(
        stages_json=json.dumps(
            {
                "persist_text": {"status": "failed"},
                "deliver:email": {"status": "ready", "outcome": "successful", "attempt_count": 1},
                "deliver:chatbook": {"status": "failed", "outcome": "unknown", "attempt_count": 2},
            }
        )
    )
    run = SimpleNamespace(started_at=None, finished_at=None)

    stages = _read_stages(
        occurrence,
        run=run,
        audio_enabled=False,
        delivery_configured=True,
    )

    assert stages["deliver:email"]["outcome"] == "successful"
    assert stages["deliver:chatbook"]["outcome"] == "unknown"


def _contract(*, audio: bool = False, limit: int = 100) -> dict[str, Any]:
    return {
        "version": 1,
        "selection": {"mode": "automatic", "max_items": limit},
        "editorial": {
            "program_format": "host_discussion" if audio else "concise_briefing",
            "outcome_noun": "episode" if audio else "briefing",
            "show_name": "Tracked Weekly" if audio else "",
            "premise": "Verified developments from tracked sources.",
        },
        "text": {
            "enabled": True,
            "type": "briefing_markdown",
            "format": "md",
            "template_name": "",
            "show_notes": audio,
        },
        "audio": {
            "enabled": audio,
            "target_minutes": 20,
            "language": "en",
            "cast": {
                "speaker_count": 2,
                "speakers": [
                    {"id": "host", "label": "Host", "voice": "alloy"},
                    {"id": "analyst", "label": "Analyst", "voice": "nova"},
                ],
            },
        },
        "delivery": {
            "reports": {"enabled": True},
            "email": {"enabled": False, "recipients": []},
            "chatbook": {"enabled": False},
        },
        "test": {"external_delivery": False, "audio_sample_seconds": 60},
    }


def _job(*, audio: bool = False, limit: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        name="Tracked Weekly",
        schedule_expr="0 8 * * *",
        schedule_timezone="UTC",
        next_run_at="2026-07-11T08:00:00+00:00",
        output_prefs_json=json.dumps({"briefing_pipeline": _contract(audio=audio, limit=limit)}),
    )


def _run(*, run_id: int = 11, zero_items: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        job_id=7,
        status="succeeded",
        started_at="2026-07-10T08:00:00+00:00",
        finished_at="2026-07-10T08:01:00+00:00",
        stats_json=json.dumps(
            {
                "items_ingested": 0 if zero_items else 3,
                "source_statuses": [
                    {"source_id": 1, "status": "ok"},
                    {"source_id": 2, "status": "error:fetch"},
                    {"source_id": 3, "status": "deferred"},
                ],
            }
        ),
    )


def _item(item_id: int, published_at: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        media_id=None,
        title=f"Item {item_id}",
        url=f"https://example.com/{item_id}",
        summary=f"Summary {item_id}",
        published_at=published_at,
        created_at=published_at,
        tags_json="[]",
    )


def make_items(count: int) -> list[SimpleNamespace]:
    return [_item(index, f"2026-07-{1 + (index % 9):02d}T00:00:00+00:00") for index in range(1, count + 1)]


class FakeWatchlistsDB:
    def __init__(self, *, job: Any, run: Any, items: list[Any]) -> None:
        self.job = job
        self.run = run
        self.items = items
        self.occurrence: SimpleNamespace | None = None
        self.list_calls: list[dict[str, Any]] = []
        self.stage_snapshots: list[dict[str, Any]] = []

    def create_or_get_briefing_occurrence(
        self,
        *,
        run_id: int,
        occurrence_key: str,
        contract_json: str,
    ) -> SimpleNamespace:
        if self.occurrence is None:
            self.occurrence = SimpleNamespace(
                id=31,
                user_id="1",
                job_id=self.job.id,
                run_id=run_id,
                occurrence_key=occurrence_key,
                contract_json=contract_json,
                stages_json="{}",
                artifact_status="running",
                delivery_status="waiting_for_artifacts",
                output_id=None,
                audio_task_id=None,
                delivery_task_id=None,
                selected_count=0,
                omitted_count=0,
                created_at="2026-07-10T08:01:00+00:00",
                updated_at="2026-07-10T08:01:00+00:00",
            )
        return self.occurrence

    def update_briefing_occurrence(self, occurrence_id: int, **patch: Any) -> SimpleNamespace:
        assert self.occurrence is not None and occurrence_id == self.occurrence.id
        if "stages" in patch:
            self.stage_snapshots.append(deepcopy(patch["stages"]))
            self.occurrence.stages_json = json.dumps(patch.pop("stages"))
        for key, value in patch.items():
            setattr(self.occurrence, key, value)
        return self.occurrence

    def get_briefing_occurrence(self, occurrence_id: int) -> SimpleNamespace:
        assert self.occurrence is not None and occurrence_id == self.occurrence.id
        return self.occurrence

    def get_job(self, job_id: int) -> Any:
        assert job_id == self.job.id
        return self.job

    def get_run(self, run_id: int) -> Any:
        assert run_id == self.run.id
        return self.run

    def list_items(self, **kwargs: Any) -> tuple[list[Any], int]:
        self.list_calls.append(kwargs)
        ordered = sorted(
            self.items,
            key=lambda item: (item.published_at or item.created_at or "", int(item.id)),
            reverse=True,
        )
        return ordered[: int(kwargs["limit"])], len(ordered)

    def get_item(self, item_id: int) -> Any:
        return next(item for item in self.items if int(item.id) == int(item_id))


class FakeCollectionsDB:
    def __init__(self) -> None:
        self.outputs: dict[int, SimpleNamespace] = {}
        self.create_calls = 0
        self._idempotency: dict[str, int] = {}
        self._lock = threading.Lock()

    def create_output_artifact(self, **kwargs: Any) -> SimpleNamespace:
        idempotency_key = kwargs.get("idempotency_key")
        with self._lock:
            if idempotency_key in self._idempotency:
                return self.outputs[self._idempotency[idempotency_key]]
            self.create_calls += 1
            output_id = 100 + self.create_calls
            row = SimpleNamespace(
                id=output_id,
                metadata_json=kwargs["metadata_json"],
                storage_path=kwargs["storage_path"],
                **{key: value for key, value in kwargs.items() if key not in {"metadata_json", "storage_path"}},
            )
            self.outputs[output_id] = row
            if idempotency_key:
                self._idempotency[str(idempotency_key)] = output_id
            return row

    def get_output_artifact(self, output_id: int) -> SimpleNamespace:
        if output_id not in self.outputs:
            raise KeyError("output_not_found")
        return self.outputs[output_id]

    def get_output_artifact_by_idempotency_key(self, idempotency_key: str) -> SimpleNamespace:
        try:
            return self.outputs[self._idempotency[idempotency_key]]
        except KeyError as exc:
            raise KeyError("output_not_found") from exc

    def update_output_artifact_metadata(self, output_id: int, *, metadata_json: str) -> SimpleNamespace:
        row = self.get_output_artifact(output_id)
        row.metadata_json = metadata_json
        return row


@pytest.fixture()
def output_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._outputs_dir_for_user",
        lambda _user_id: tmp_path,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._resolve_output_path_for_user",
        lambda _user_id, filename: tmp_path / filename,
    )
    return tmp_path


@pytest.mark.asyncio
async def test_zero_items_persists_text_and_requests_short_audio(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run(zero_items=True)
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=[])
    collections_db = FakeCollectionsDB()
    audio_requests: list[dict[str, Any]] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        audio_requests.append(kwargs)
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id="audio-task-1",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )

    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert result.artifact_status == "running"
    assert result.selected_count == 0
    assert result.output_id is not None
    saved = collections_db.get_output_artifact(result.output_id)
    metadata = json.loads(saved.metadata_json)
    assert metadata["no_material_updates"] is True
    assert metadata["source_counts"] == {"succeeded": 1, "failed": 1, "deferred": 1}
    assert audio_requests[0]["items"][0]["status_kind"] == "no_material_updates"
    assert audio_requests[0]["items"][0]["summary"] == (
        "No qualifying new material was found. "
        "Sources succeeded: 1. Sources failed: 1. Sources deferred: 1. "
        "Checked: 2026-07-10T08:01:00+00:00. "
        "Next run: 2026-07-11T08:00:00+00:00."
    )
    assert audio_requests[0]["status_audio"] is True
    assert audio_requests[0]["occurrence_id"] == result.occurrence_id
    assert audio_requests[0]["output_id"] == result.output_id
    assert "No qualifying new material was found" in (output_paths / saved.storage_path).read_text()


@pytest.mark.asyncio
async def test_text_fulfillment_schedules_post_artifact_delivery(output_paths: Path) -> None:
    job = _job()
    contract = _contract()
    contract["delivery"]["email"] = {
        "enabled": True,
        "recipients": ["digest@example.com"],
    }
    job.output_prefs_json = json.dumps({"briefing_pipeline": contract})
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    scheduler = MagicMock()
    scheduler.submit = AsyncMock(return_value="delivery-task-text")

    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=FakeCollectionsDB(),
        scheduler=scheduler,
    )

    assert result.artifact_status == "ready"
    assert scheduler.submit.await_args.args[0] == "watchlists_deliver_briefing"
    assert scheduler.submit.await_args.kwargs["depends_on"] is None
    assert watchlists_db.occurrence.delivery_task_id == "delivery-task-text"


@pytest.mark.asyncio
async def test_audio_fulfillment_schedules_delivery_after_audio_dependency(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    contract = _contract(audio=True)
    contract["delivery"]["email"] = {
        "enabled": True,
        "recipients": ["digest@example.com"],
    }
    job.output_prefs_json = json.dumps({"briefing_pipeline": contract})
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    scheduler = MagicMock()
    scheduler.submit = AsyncMock(return_value="delivery-task-audio")

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id="audio-task-7",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )

    await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=FakeCollectionsDB(),
        scheduler=scheduler,
    )

    assert scheduler.submit.await_args.kwargs["depends_on"] == ["audio-task-7"]


@pytest.mark.asyncio
async def test_text_failure_is_persisted_and_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()

    async def fail_persist(**_kwargs: Any) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._persist_text_output",
        fail_persist,
    )

    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert result.artifact_status == "failed"
    assert result.stages["persist_text"]["status"] == "failed"
    assert result.stages["persist_text"]["code"] == "text_persist_failed"
    assert result.stages["persist_text"]["retryable"] is True
    assert result.stages["persist_text"]["finished_at"]
    assert any(snapshot["persist_text"]["status"] == "running" for snapshot in watchlists_db.stage_snapshots)
    assert json.loads(watchlists_db.occurrence.stages_json)["persist_text"]["status"] == "failed"


@pytest.mark.asyncio
async def test_text_and_audio_share_selection_cap(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True, limit=100)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(137))
    collections_db = FakeCollectionsDB()
    render_item_ids: list[int] = []
    audio_item_ids: list[int] = []

    def render(**kwargs: Any) -> str:
        render_item_ids.extend(item["id"] for item in kwargs["items"])
        return "# rendered"

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        audio_item_ids.extend(item["id"] for item in kwargs["items"])
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id="audio-task-cap",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._render_briefing_text",
        render,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )

    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert result.selected_count == 100
    assert result.omitted_count == 37
    assert render_item_ids == audio_item_ids
    assert watchlists_db.list_calls == [
        {"run_id": run.id, "status": "ingested", "sort": "published_desc", "limit": 100, "offset": 0}
    ]


@pytest.mark.asyncio
async def test_repeated_fulfillment_reuses_logical_output_and_audio_request(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()
    audio_requests: list[str] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        audio_requests.append(kwargs["audio_request_id"])
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id="audio-task-repeat",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )

    first = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    second = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert second.occurrence_id == first.occurrence_id
    assert second.output_id == first.output_id
    assert collections_db.create_calls == 1
    assert watchlists_db.list_calls == [watchlists_db.list_calls[0]]
    assert audio_requests == [audio_requests[0]]


def test_audio_request_id_is_stable_sha256_derivative() -> None:
    occurrence_key = "user:1:job:7:run:11:v1"

    request_id = audio_request_id_for_occurrence(occurrence_key)

    assert request_id == f"wla_{hashlib.sha256(occurrence_key.encode()).hexdigest()[:32]}"
    assert audio_request_id_for_occurrence(occurrence_key) == request_id
    assert audio_request_id_for_occurrence(occurrence_key, output_version=2) != request_id


@pytest.mark.asyncio
async def test_selection_orders_timestamp_descending_then_stable_id(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(limit=4)
    run = _run()
    items = [
        _item(4, "2026-07-09T12:00:00+00:00"),
        _item(2, "2026-07-10T12:00:00+00:00"),
        _item(5, "2026-07-10T12:00:00+00:00"),
        _item(1, "2026-07-08T12:00:00+00:00"),
    ]
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=items)
    collections_db = FakeCollectionsDB()
    selected_ids: list[int] = []

    def render(**kwargs: Any) -> str:
        selected_ids.extend(item["id"] for item in kwargs["items"])
        return "# ordered"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._render_briefing_text",
        render,
    )

    await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert selected_ids == [5, 2, 4, 1]


def test_no_material_markdown_is_deterministic_and_complete() -> None:
    kwargs = {
        "title": "Tracked Weekly",
        "checked_at": "2026-07-10T08:01:00+00:00",
        "next_run_at": "2026-07-11T08:00:00+00:00",
        "source_counts": {"succeeded": 2, "failed": 1, "deferred": 3},
    }

    first = no_material_updates_markdown(**kwargs)

    assert first == no_material_updates_markdown(**kwargs)
    assert "No qualifying new material was found" in first
    assert "Sources succeeded: 2" in first
    assert "Sources failed: 1" in first
    assert "Sources deferred: 3" in first
    assert "Checked: 2026-07-10T08:01:00+00:00" in first
    assert "Next run: 2026-07-11T08:00:00+00:00" in first


def test_run_stats_projection_is_compact_and_preserves_legacy_keys() -> None:
    result = FulfillmentResult(
        occurrence_id=31,
        output_id=101,
        audio_task_id="audio-task-1",
        artifact_status="running",
        delivery_status="waiting_for_artifacts",
        selected_count=8,
        omitted_count=2,
        stages={"select": {"status": "ready"}},
    )

    projection = _fulfillment_stats_projection(result)

    assert projection == {
        "briefing_occurrence": {
            "id": 31,
            "artifact_status": "running",
            "delivery_status": "waiting_for_artifacts",
            "output_id": 101,
            "audio_task_id": "audio-task-1",
            "selected_count": 8,
            "omitted_count": 2,
        },
        "auto_output_id": 101,
        "audio_briefing_task_id": "audio-task-1",
        "audio_briefing_status": "queued",
    }
    assert "stages" not in projection["briefing_occurrence"]


@pytest.mark.asyncio
async def test_failed_audio_stage_retries_without_recreating_ready_text(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()
    request_ids: list[str] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        request_ids.append(kwargs["audio_request_id"])
        if len(request_ids) == 1:
            return AudioBriefingTriggerResult(
                status="enqueue_failed",
                audio_request_id=kwargs["audio_request_id"],
                reason="scheduler_submit_failed",
            )
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id="audio-task-retry",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )

    failed = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    retried = await retry_briefing_stage(
        user_id=1,
        occurrence_id=failed.occurrence_id,
        stage="compose_audio_script",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert failed.artifact_status == "failed"
    assert failed.stages["compose_audio_script"]["status"] == "failed"
    assert failed.stages["compose_audio_script"]["code"] == "scheduler_submit_failed"
    assert retried.artifact_status == "running"
    assert retried.stages["compose_audio_script"]["status"] == "queued"
    assert retried.audio_task_id == "audio-task-retry"
    assert request_ids == [request_ids[0], request_ids[0]]
    assert collections_db.create_calls == 1
    assert len(watchlists_db.list_calls) == 1


@pytest.mark.asyncio
async def test_downstream_audio_stage_retry_resumes_same_audio_attempt(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    collections_db = FakeCollectionsDB()
    request_ids: list[str] = []
    trigger_calls: list[dict[str, Any]] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        trigger_calls.append(kwargs)
        request_ids.append(kwargs["audio_request_id"])
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id=f"audio-task-{len(request_ids)}",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )
    first = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    stages = json.loads(watchlists_db.occurrence.stages_json)
    stages["compose_audio_script"] = {"status": "ready"}
    stages["persist_audio_script"] = {"status": "ready"}
    stages["generate_audio"] = {"status": "failed", "retryable": True}
    watchlists_db.update_briefing_occurrence(
        first.occurrence_id,
        stages=stages,
        artifact_status="failed",
        audio_task_id=None,
    )

    retried = await retry_briefing_stage(
        user_id=1,
        occurrence_id=first.occurrence_id,
        stage="generate_audio",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert retried.audio_task_id == "audio-task-2"
    assert request_ids == [request_ids[0], request_ids[0]]
    assert trigger_calls[1]["requested_stage"] == "generate_audio"
    assert collections_db.create_calls == 1


@pytest.mark.asyncio
async def test_failed_text_persistence_retries_same_occurrence(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    from tldw_Server_API.app.core.Watchlists import briefing_fulfillment as module

    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(2))
    collections_db = FakeCollectionsDB()
    persist = module._persist_text_output
    attempts = 0

    async def fail_once(**kwargs: Any) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("disk full")
        return await persist(**kwargs)

    monkeypatch.setattr(module, "_persist_text_output", fail_once)

    failed = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    retried = await retry_briefing_stage(
        user_id=1,
        occurrence_id=failed.occurrence_id,
        stage="persist_text",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert failed.stages["persist_text"]["status"] == "failed"
    assert retried.occurrence_id == failed.occurrence_id
    assert retried.artifact_status == "ready"
    assert retried.stages["persist_text"]["status"] == "ready"
    assert retried.output_id is not None
    assert collections_db.create_calls == 1


@pytest.mark.asyncio
async def test_text_retry_schedules_delivery_after_artifact_recovers(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    from tldw_Server_API.app.core.Watchlists import briefing_fulfillment as module

    job = _job()
    contract = _contract()
    contract["delivery"]["email"] = {
        "enabled": True,
        "recipients": ["digest@example.com"],
    }
    job.output_prefs_json = json.dumps({"briefing_pipeline": contract})
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    collections_db = FakeCollectionsDB()
    scheduler = MagicMock()
    scheduler.submit = AsyncMock(return_value="delivery-task-after-text-retry")
    persist = module._persist_text_output
    attempts = 0

    async def fail_once(**kwargs: Any) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("disk full")
        return await persist(**kwargs)

    monkeypatch.setattr(module, "_persist_text_output", fail_once)

    failed = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        scheduler=scheduler,
    )
    retried = await retry_briefing_stage(
        user_id=1,
        occurrence_id=failed.occurrence_id,
        stage="persist_text",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        scheduler=scheduler,
    )

    assert retried.output_id is not None
    assert retried.delivery_status == "waiting_for_artifacts"
    assert scheduler.submit.await_count == 1
    assert scheduler.submit.await_args.kwargs["depends_on"] is None


@pytest.mark.asyncio
async def test_audio_retry_replaces_delivery_dependency(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    contract = _contract(audio=True)
    contract["delivery"]["email"] = {
        "enabled": True,
        "recipients": ["digest@example.com"],
    }
    job.output_prefs_json = json.dumps({"briefing_pipeline": contract})
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    collections_db = FakeCollectionsDB()
    scheduler = MagicMock()
    scheduler.submit = AsyncMock(side_effect=["delivery-task-old", "delivery-task-new"])
    request_ids: list[str] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        request_ids.append(kwargs["audio_request_id"])
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id=f"audio-task-{len(request_ids)}",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )
    first = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        scheduler=scheduler,
    )
    stages = json.loads(watchlists_db.occurrence.stages_json)
    stages["generate_audio"] = {"status": "failed", "retryable": True}
    watchlists_db.update_briefing_occurrence(
        first.occurrence_id,
        stages=stages,
        artifact_status="failed",
        audio_task_id=None,
    )

    await retry_briefing_stage(
        user_id=1,
        occurrence_id=first.occurrence_id,
        stage="generate_audio",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        scheduler=scheduler,
    )

    assert scheduler.submit.await_count == 2
    assert scheduler.submit.await_args.kwargs["depends_on"] == ["audio-task-2"]
    assert scheduler.submit.await_args.kwargs["idempotency_key"] == (
        "watchlists-briefing-delivery:1:31:audio:audio-task-2"
    )


@pytest.mark.asyncio
async def test_persist_retry_reuses_durable_render_without_rerendering(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    from tldw_Server_API.app.core.Watchlists import briefing_fulfillment as module

    job = _job()
    contract = _contract()
    contract["text"]["template_name"] = "mutable-template"
    job.output_prefs_json = json.dumps({"briefing_pipeline": contract})
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(2))
    collections_db = FakeCollectionsDB()
    template = SimpleNamespace(content="# Original render")
    render_calls = 0

    def load_template(_name: str) -> SimpleNamespace:
        nonlocal render_calls
        render_calls += 1
        return template

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.template_store.load_template",
        load_template,
    )
    persist = module._persist_text_output
    persist_calls = 0

    async def fail_once(**kwargs: Any) -> int:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise OSError("disk full")
        return await persist(**kwargs)

    monkeypatch.setattr(module, "_persist_text_output", fail_once)
    failed = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    render_stage_before = deepcopy(failed.stages["render_text"])
    template.content = "# Changed render"
    job.next_run_at = "2026-07-12T08:00:00+00:00"

    retried = await retry_briefing_stage(
        user_id=1,
        occurrence_id=failed.occurrence_id,
        stage="persist_text",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    row = collections_db.get_output_artifact(retried.output_id)
    metadata = json.loads(row.metadata_json)
    assert render_calls == 1
    assert retried.stages["render_text"] == render_stage_before
    assert metadata["next_run_at"] == "2026-07-11T08:00:00+00:00"
    assert (output_paths / row.storage_path).read_text() == "# Original render"


@pytest.mark.asyncio
async def test_explicit_audio_regeneration_versions_request_not_occurrence(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(2))
    collections_db = FakeCollectionsDB()
    request_ids: list[str] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        request_ids.append(kwargs["audio_request_id"])
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id=f"audio-task-{len(request_ids)}",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )

    first = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    regenerated = await retry_briefing_stage(
        user_id=1,
        occurrence_id=first.occurrence_id,
        stage="compose_audio_script",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        regenerate=True,
    )

    assert regenerated.occurrence_id == first.occurrence_id
    assert request_ids[1] != request_ids[0]
    assert request_ids[1] == audio_request_id_for_occurrence(
        watchlists_db.occurrence.occurrence_key,
        output_version=2,
    )
    assert collections_db.create_calls == 1


class CrashAfterArtifactDB(FakeWatchlistsDB):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.crash_output_link_once = False
        self.crash_audio_link_once = False

    def update_briefing_occurrence(self, occurrence_id: int, **patch: Any) -> SimpleNamespace:
        if self.crash_output_link_once and patch.get("output_id") is not None:
            self.crash_output_link_once = False
            raise RuntimeError("crash_after_output_create")
        if self.crash_audio_link_once and patch.get("audio_task_id") is not None:
            self.crash_audio_link_once = False
            raise RuntimeError("crash_after_audio_metadata")
        return super().update_briefing_occurrence(occurrence_id, **patch)


@pytest.mark.asyncio
async def test_replay_recovers_output_when_link_save_crashes(
    output_paths: Path,
) -> None:
    job = _job()
    run = _run()
    watchlists_db = CrashAfterArtifactDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()
    watchlists_db.crash_output_link_once = True

    with pytest.raises(RuntimeError, match="crash_after_output_create"):
        await fulfill_watchlist_briefing(
            user_id=1,
            job=job,
            run=run,
            watchlists_db=watchlists_db,
            collections_db=collections_db,
        )

    replay = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert replay.output_id == 101
    assert replay.stages["persist_text"]["status"] == "ready"
    assert collections_db.create_calls == 1


@pytest.mark.asyncio
async def test_stale_occurrence_output_id_recovers_by_idempotency_key(
    output_paths: Path,
) -> None:
    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()
    first = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    watchlists_db.occurrence.output_id = 999_999

    replay = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert replay.output_id == first.output_id
    assert collections_db.create_calls == 1
    assert len(watchlists_db.list_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_fulfillment_creates_one_logical_output(
    output_paths: Path,
) -> None:
    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()

    async def fulfill() -> FulfillmentResult:
        return await fulfill_watchlist_briefing(
            user_id=1,
            job=job,
            run=run,
            watchlists_db=watchlists_db,
            collections_db=collections_db,
        )

    results = await asyncio.gather(fulfill(), fulfill())

    assert results[0].output_id == results[1].output_id
    assert collections_db.create_calls == 1


@pytest.mark.asyncio
async def test_replay_recovers_submitted_audio_from_output_metadata(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = CrashAfterArtifactDB(job=job, run=run, items=make_items(2))
    collections_db = FakeCollectionsDB()
    requests: list[str] = []

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        requests.append(kwargs["audio_request_id"])
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id="audio-task-durable",
            audio_request_id=kwargs["audio_request_id"],
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )
    watchlists_db.crash_audio_link_once = True

    with pytest.raises(RuntimeError, match="crash_after_audio_metadata"):
        await fulfill_watchlist_briefing(
            user_id=1,
            job=job,
            run=run,
            watchlists_db=watchlists_db,
            collections_db=collections_db,
        )
    replay = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert replay.audio_task_id == "audio-task-durable"
    assert replay.stages["compose_audio_script"]["status"] == "queued"
    assert requests == [requests[0]]
    assert collections_db.create_calls == 1
    metadata = json.loads(collections_db.get_output_artifact(replay.output_id).metadata_json)
    assert metadata["ai_generated_speech"] is False
    assert metadata["speech_disclosure"] == "Synthetic speech generation pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_status", ["delivered", "unknown"])
async def test_replay_preserves_delivery_truth(
    output_paths: Path,
    delivery_status: str,
) -> None:
    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    collections_db = FakeCollectionsDB()
    await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    watchlists_db.occurrence.delivery_status = delivery_status

    replay = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert replay.delivery_status == delivery_status


@pytest.mark.asyncio
async def test_render_retry_reuses_durable_selection_order(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(limit=2)
    run = _run()
    original = [_item(1, "2026-07-10T00:00:00+00:00"), _item(2, "2026-07-09T00:00:00+00:00")]
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=original)
    collections_db = FakeCollectionsDB()
    rendered_ids: list[list[int]] = []

    def render(**kwargs: Any) -> str:
        rendered_ids.append([int(item["id"]) for item in kwargs["items"]])
        if len(rendered_ids) == 1:
            raise RuntimeError("render failed")
        return "# recovered"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._render_briefing_text",
        render,
    )
    failed = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    watchlists_db.items = [*original, _item(99, "2026-07-11T00:00:00+00:00")]

    retried = await retry_briefing_stage(
        user_id=1,
        occurrence_id=failed.occurrence_id,
        stage="render_text",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert rendered_ids == [[1, 2], [1, 2]]
    assert retried.stages["select"]["selected_item_ids"] == [1, 2]
    assert any(snapshot["persist_text"]["status"] == "running" for snapshot in watchlists_db.stage_snapshots)


@pytest.mark.asyncio
async def test_zero_selection_is_explicitly_durable(output_paths: Path) -> None:
    job = _job()
    run = _run(zero_items=True)
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=[])

    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=FakeCollectionsDB(),
    )

    assert result.stages["select"]["selected_item_ids"] == []
    assert result.stages["select"]["candidate_count"] == 0


@pytest.mark.asyncio
async def test_default_markdown_and_template_context_disclose_omissions(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(limit=2)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(3))
    collections_db = FakeCollectionsDB()

    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    row = collections_db.get_output_artifact(result.output_id)
    assert "Included 2 of 3 candidate items; 1 omitted by the selection cap." in (
        output_paths / row.storage_path
    ).read_text()

    contexts: list[dict[str, Any]] = []
    from tldw_Server_API.app.core.Watchlists import briefing_fulfillment as module

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.template_store.load_template",
        lambda _name: SimpleNamespace(content="ignored"),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "render_output_template",
        lambda _body, context: contexts.append(context) or "# template",
    )
    contract = _contract(limit=2)
    contract["text"]["template_name"] = "test-template"
    job.output_prefs_json = json.dumps({"briefing_pipeline": contract})
    job.id = 8
    run.id = 12
    run.job_id = 8
    watchlists_db.job = job
    watchlists_db.run = run
    watchlists_db.occurrence = None
    await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert contexts[0]["candidate_count"] == 3
    assert contexts[0]["included_count"] == 2
    assert contexts[0]["omitted_count"] == 1
    assert contexts[0]["selection_cap"] == 2


@pytest.mark.asyncio
async def test_enqueue_failure_keeps_speech_truth_pending_and_uses_canonical_premise(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))
    collections_db = FakeCollectionsDB()

    async def trigger(**kwargs: Any) -> AudioBriefingTriggerResult:
        return AudioBriefingTriggerResult(
            status="enqueue_failed",
            audio_request_id=kwargs["audio_request_id"],
            reason="scheduler_submit_failed",
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        trigger,
    )
    result = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    metadata = json.loads(collections_db.get_output_artifact(result.output_id).metadata_json)

    assert metadata["show_identity"]["premise"] == "Verified developments from tracked sources."
    assert metadata["audio_selected"] is True
    assert metadata["ai_generated_speech"] is False
    assert metadata["speech_disclosure"] == "Synthetic speech generation pending"


@pytest.mark.asyncio
async def test_explicit_render_regeneration_uses_new_content_version_and_key(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(2))
    collections_db = FakeCollectionsDB()
    rendered_content = "# Version one"
    render_calls = 0

    def render(**_kwargs: Any) -> str:
        nonlocal render_calls
        render_calls += 1
        return rendered_content

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment._render_briefing_text",
        render,
    )
    first = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    rendered_content = "# Version two"

    regenerated = await retry_briefing_stage(
        user_id=1,
        occurrence_id=first.occurrence_id,
        stage="render_text",
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        regenerate=True,
    )

    first_row = collections_db.get_output_artifact(first.output_id)
    regenerated_row = collections_db.get_output_artifact(regenerated.output_id)
    assert regenerated.output_id != first.output_id
    assert regenerated.stages["persist_text"]["output_version"] == 2
    assert json.loads(regenerated_row.metadata_json)["output_version"] == 2
    assert regenerated_row.idempotency_key != first_row.idempotency_key
    assert render_calls == 2
    assert (output_paths / first_row.storage_path).read_text() == "# Version one"
    assert (output_paths / regenerated_row.storage_path).read_text() == "# Version two"


@pytest.mark.asyncio
async def test_select_retry_is_rejected_without_mutating_ready_occurrence(
    output_paths: Path,
) -> None:
    job = _job(limit=2)
    run = _run()
    original = [_item(1, "2026-07-10T00:00:00+00:00"), _item(2, "2026-07-09T00:00:00+00:00")]
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=original)
    collections_db = FakeCollectionsDB()

    ready = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    row = collections_db.get_output_artifact(ready.output_id)
    path = output_paths / row.storage_path
    before = {
        "occurrence": deepcopy(vars(watchlists_db.occurrence)),
        "create_calls": collections_db.create_calls,
        "list_calls": deepcopy(watchlists_db.list_calls),
        "stage_snapshots": deepcopy(watchlists_db.stage_snapshots),
        "content": path.read_text(),
    }
    watchlists_db.items = [*original, _item(99, "2026-07-11T00:00:00+00:00")]

    with pytest.raises(ValueError, match="unsupported_briefing_retry_stage"):
        await retry_briefing_stage(
            user_id=1,
            occurrence_id=ready.occurrence_id,
            stage="select",
            watchlists_db=watchlists_db,
            collections_db=collections_db,
        )

    assert vars(watchlists_db.occurrence) == before["occurrence"]
    assert collections_db.create_calls == before["create_calls"]
    assert watchlists_db.list_calls == before["list_calls"]
    assert watchlists_db.stage_snapshots == before["stage_snapshots"]
    assert path.read_text() == before["content"]


@pytest.mark.asyncio
async def test_failed_initial_selection_recovers_via_ordinary_fulfillment_replay(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    from tldw_Server_API.app.core.Watchlists import briefing_fulfillment as module

    job = _job()
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(2))
    collections_db = FakeCollectionsDB()
    load_selection = module._load_selection
    attempts = 0

    async def fail_once(*args: Any, **kwargs: Any):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("selection unavailable")
        return await load_selection(*args, **kwargs)

    monkeypatch.setattr(module, "_load_selection", fail_once)
    failed = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )
    replay = await fulfill_watchlist_briefing(
        user_id=1,
        job=job,
        run=run,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert failed.stages["select"]["status"] == "failed"
    assert replay.artifact_status == "ready"
    assert replay.stages["select"]["selected_item_ids"]
    assert collections_db.create_calls == 1


@pytest.mark.asyncio
async def test_audio_cancellation_persists_occurrence_state(
    monkeypatch: pytest.MonkeyPatch,
    output_paths: Path,
) -> None:
    job = _job(audio=True)
    run = _run()
    watchlists_db = FakeWatchlistsDB(job=job, run=run, items=make_items(1))

    async def cancelled(**_kwargs: Any) -> AudioBriefingTriggerResult:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.briefing_fulfillment.trigger_audio_briefing",
        cancelled,
    )
    with pytest.raises(asyncio.CancelledError):
        await fulfill_watchlist_briefing(
            user_id=1,
            job=job,
            run=run,
            watchlists_db=watchlists_db,
            collections_db=FakeCollectionsDB(),
        )

    assert watchlists_db.occurrence.artifact_status == "cancelled"
    stages = json.loads(watchlists_db.occurrence.stages_json)
    assert stages["compose_audio_script"]["status"] == "cancelled"
