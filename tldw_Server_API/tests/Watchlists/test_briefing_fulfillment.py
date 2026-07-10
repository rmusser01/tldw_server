from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def _contract(*, audio: bool = False, limit: int = 100) -> dict[str, Any]:
    return {
        "version": 1,
        "selection": {"mode": "automatic", "max_items": limit},
        "editorial": {
            "program_format": "host_discussion" if audio else "concise_briefing",
            "outcome_noun": "episode" if audio else "briefing",
            "show_name": "Tracked Weekly" if audio else "",
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


class FakeCollectionsDB:
    def __init__(self) -> None:
        self.outputs: dict[int, SimpleNamespace] = {}
        self.create_calls = 0

    def create_output_artifact(self, **kwargs: Any) -> SimpleNamespace:
        self.create_calls += 1
        output_id = 100 + self.create_calls
        row = SimpleNamespace(
            id=output_id,
            metadata_json=kwargs["metadata_json"],
            storage_path=kwargs["storage_path"],
            **{key: value for key, value in kwargs.items() if key not in {"metadata_json", "storage_path"}},
        )
        self.outputs[output_id] = row
        return row

    def get_output_artifact(self, output_id: int) -> SimpleNamespace:
        if output_id not in self.outputs:
            raise KeyError("output_not_found")
        return self.outputs[output_id]

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
    assert audio_requests[0]["status_audio"] is True
    assert audio_requests[0]["occurrence_id"] == result.occurrence_id
    assert audio_requests[0]["output_id"] == result.output_id
    assert "No qualifying new material was found" in (output_paths / saved.storage_path).read_text()


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
