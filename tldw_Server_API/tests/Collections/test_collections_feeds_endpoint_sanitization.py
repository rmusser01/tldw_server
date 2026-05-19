from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from fastapi import BackgroundTasks, HTTPException

from tldw_Server_API.app.api.v1.endpoints import collections_feeds
from tldw_Server_API.app.api.v1.schemas.collections_feeds_schemas import (
    CollectionsFeedCreateRequest,
    CollectionsFeedUpdateRequest,
)

pytestmark = pytest.mark.unit


class _FakeFeedDb:
    def __init__(self) -> None:
        self.schedule_ids: list[tuple[int, str]] = []

    def set_job_schedule_id(self, job_id: int, schedule_id: str) -> None:
        self.schedule_ids.append((job_id, schedule_id))


class _FailingScheduler:
    def create(self, **kwargs):
        raise RuntimeError("scheduler backend exploded at /private/scheduler.db")


class _FailingUpdateScheduler(_FailingScheduler):
    def update(self, schedule_id: str, payload: dict):
        assert schedule_id == "schedule-private"
        raise RuntimeError("scheduler update exploded at /private/scheduler-update.db")


def _job_row() -> SimpleNamespace:
    return SimpleNamespace(
        id=9,
        name="Private Feed",
        schedule_expr="0 * * * *",
        schedule_timezone="UTC",
        active=True,
    )


def _scheduled_job_row() -> SimpleNamespace:
    job = _job_row()
    job.wf_schedule_id = "schedule-private"
    return job


def _source_row(*, settings: dict | None = None, tags: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        name="Private Feed",
        url="https://example.com/feed.xml",
        source_type="rss",
        active=True,
        settings_json=collections_feeds.json.dumps(settings or {"collections_origin": collections_feeds.FEED_ORIGIN}),
        tags=tags or [],
        last_scraped_at=None,
        etag=None,
        last_modified=None,
        defer_until=None,
        status=None,
        consec_not_modified=0,
        consec_errors=0,
        created_at=None,
        updated_at=None,
    )


class _ListingFeedDb:
    def list_sources(self, **kwargs):
        assert kwargs["limit"] == 100
        assert kwargs["offset"] == 0
        rows = [_source_row() for _ in range(3)]
        for index, row in enumerate(rows, start=1):
            row.id = index
        return rows, 3


async def test_list_feed_subscriptions_includes_canonical_page_pagination() -> None:
    response = await collections_feeds.list_feed_subscriptions(
        q=None,
        page=2,
        size=1,
        current_user=SimpleNamespace(id=42),
        db=_ListingFeedDb(),
    )

    assert response.total == 3
    assert [item.id for item in response.items] == [2]
    assert response.pagination.model_dump(mode="json") == {
        "mode": "page",
        "page": 2,
        "per_page": 1,
        "total": 3,
        "total_pages": 3,
        "has_more": True,
    }


def test_register_schedule_sanitizes_scheduler_failure_log(monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as wfdb_module
    from tldw_Server_API.app.services import workflows_scheduler

    class _FallbackSchedulerDb:
        def __init__(self, *, user_id: int) -> None:
            assert user_id == 42

        def create_schedule(self, **kwargs) -> None:
            return None

    fake_db = _FakeFeedDb()
    fake_logger = MagicMock()
    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", lambda: _FailingScheduler())
    monkeypatch.setattr(wfdb_module, "WorkflowsSchedulerDB", _FallbackSchedulerDb)
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    collections_feeds._register_schedule(
        fake_db,
        _job_row(),
        current_user=SimpleNamespace(id=42),
    )

    fake_logger.debug.assert_called_once_with("Collections feeds schedule registration failed")
    assert fake_db.schedule_ids


def test_register_schedule_sanitizes_db_fallback_failure_log(monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as wfdb_module
    from tldw_Server_API.app.services import workflows_scheduler

    class _FailingFallbackSchedulerDb:
        def __init__(self, *, user_id: int) -> None:
            assert user_id == 42

        def create_schedule(self, **kwargs) -> None:
            raise RuntimeError("scheduler DB fallback exploded at /private/scheduler-fallback.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", lambda: _FailingScheduler())
    monkeypatch.setattr(wfdb_module, "WorkflowsSchedulerDB", _FailingFallbackSchedulerDb)
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    collections_feeds._register_schedule(
        _FakeFeedDb(),
        _job_row(),
        current_user=SimpleNamespace(id=42),
    )

    assert fake_logger.debug.call_args_list == [
        call("Collections feeds schedule registration failed"),
        call("Collections feeds schedule DB fallback failed"),
    ]


def test_sync_job_schedule_sanitizes_scheduler_update_failure_log(monkeypatch):
    from tldw_Server_API.app.services import workflows_scheduler

    class _FakeFeedDbWithJob(_FakeFeedDb):
        def get_job(self, job_id: int) -> SimpleNamespace:
            assert job_id == 9
            return SimpleNamespace(id=job_id)

    fake_logger = MagicMock()
    monkeypatch.setattr(workflows_scheduler, "get_workflows_scheduler", lambda: _FailingUpdateScheduler())
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    result = collections_feeds._sync_job_schedule(
        _FakeFeedDbWithJob(),
        _scheduled_job_row(),
        current_user=SimpleNamespace(id=42),
    )

    assert result.id == 9
    fake_logger.debug.assert_called_once_with("Collections feeds schedule update failed")


async def test_create_feed_subscription_source_failure_log_is_sanitized(monkeypatch):
    class _FailingCreateSourceDb:
        def create_source(self, **_kwargs):
            raise RuntimeError("collections source backend exploded at /private/feeds-source.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await collections_feeds.create_feed_subscription(
            payload=CollectionsFeedCreateRequest(url="https://example.com/feed.xml", active=False),
            background_tasks=BackgroundTasks(),
            current_user=SimpleNamespace(id=42),
            db=_FailingCreateSourceDb(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "feed_create_failed"
    fake_logger.error.assert_called_once_with("collections_feeds_create_source_failed")
    rendered = " ".join(str(part) for call_args in fake_logger.error.call_args_list for part in call_args.args)
    assert "/private/feeds-source.db" not in rendered
    assert "exploded" not in rendered


async def test_create_feed_subscription_job_failure_log_is_sanitized(monkeypatch):
    class _FailingCreateJobDb:
        def __init__(self) -> None:
            self.deleted_sources: list[int] = []

        def create_source(self, **kwargs):
            return SimpleNamespace(
                id=7,
                name=kwargs["name"],
                url=kwargs["url"],
                source_type=kwargs["source_type"],
                active=kwargs["active"],
            )

        def create_job(self, **_kwargs):
            raise RuntimeError("collections job backend exploded at /private/feeds-job.db")

        def delete_source(self, source_id: int) -> None:
            self.deleted_sources.append(source_id)

    fake_logger = MagicMock()
    fake_db = _FailingCreateJobDb()
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await collections_feeds.create_feed_subscription(
            payload=CollectionsFeedCreateRequest(url="https://example.com/feed.xml", active=False),
            background_tasks=BackgroundTasks(),
            current_user=SimpleNamespace(id=42),
            db=fake_db,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "feed_create_failed"
    assert fake_db.deleted_sources == [7]
    fake_logger.error.assert_called_once_with("collections_feeds_create_job_failed")
    rendered = " ".join(str(part) for call_args in fake_logger.error.call_args_list for part in call_args.args)
    assert "/private/feeds-job.db" not in rendered
    assert "exploded" not in rendered


async def test_update_feed_subscription_source_failure_log_is_sanitized(monkeypatch):
    class _FailingUpdateSourceDb:
        def get_source(self, feed_id: int):
            assert feed_id == 7
            return _source_row()

        def update_source(self, feed_id: int, _patch: dict):
            assert feed_id == 7
            raise RuntimeError("collections update backend exploded at /private/feeds-update.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await collections_feeds.update_feed_subscription(
            feed_id=7,
            payload=CollectionsFeedUpdateRequest(name="Updated Feed"),
            current_user=SimpleNamespace(id=42),
            db=_FailingUpdateSourceDb(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "feed_update_failed"
    fake_logger.error.assert_called_once_with("collections_feeds_update_source_failed")
    rendered = " ".join(str(part) for call_args in fake_logger.error.call_args_list for part in call_args.args)
    assert "/private/feeds-update.db" not in rendered
    assert "exploded" not in rendered


async def test_update_feed_subscription_tags_failure_log_is_sanitized(monkeypatch):
    class _FailingTagsDb:
        def get_source(self, feed_id: int):
            assert feed_id == 7
            return _source_row()

        def update_source(self, feed_id: int, _patch: dict):
            assert feed_id == 7
            return _source_row()

        def set_source_tags(self, feed_id: int, _tags: list[str]):
            assert feed_id == 7
            raise RuntimeError("collections tags backend exploded at /private/feeds-tags.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_updated", lambda **_kwargs: None)

    response = await collections_feeds.update_feed_subscription(
        feed_id=7,
        payload=CollectionsFeedUpdateRequest(tags=["private"]),
        current_user=SimpleNamespace(id=42),
        db=_FailingTagsDb(),
    )

    assert response.id == 7
    fake_logger.error.assert_called_once_with("collections_feeds_update_tags_failed")
    rendered = " ".join(str(part) for call_args in fake_logger.error.call_args_list for part in call_args.args)
    assert "/private/feeds-tags.db" not in rendered
    assert "exploded" not in rendered


async def test_update_feed_subscription_job_failure_log_is_sanitized(monkeypatch):
    settings = {
        "collections_origin": collections_feeds.FEED_ORIGIN,
        "collections_feed_job_id": 9,
    }

    class _FailingJobDb:
        def get_source(self, feed_id: int):
            assert feed_id == 7
            return _source_row(settings=settings)

        def update_source(self, feed_id: int, _patch: dict):
            assert feed_id == 7
            return _source_row(settings=settings)

        def get_job(self, job_id: int):
            assert job_id == 9
            return _job_row()

        def update_job(self, job_id: int, _patch: dict):
            assert job_id == 9
            raise RuntimeError("collections job update backend exploded at /private/feeds-job-update.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_updated", lambda **_kwargs: None)

    response = await collections_feeds.update_feed_subscription(
        feed_id=7,
        payload=CollectionsFeedUpdateRequest(schedule_expr="*/15 * * * *"),
        current_user=SimpleNamespace(id=42),
        db=_FailingJobDb(),
    )

    assert response.id == 7
    assert response.job_id == 9
    fake_logger.error.assert_called_once_with("collections_feeds_update_job_failed")
    rendered = " ".join(str(part) for call_args in fake_logger.error.call_args_list for part in call_args.args)
    assert "/private/feeds-job-update.db" not in rendered
    assert "exploded" not in rendered


async def test_create_feed_subscription_first_run_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Watchlists import pipeline as pipeline_module

    class _CreateFeedDb:
        def __init__(self) -> None:
            self.source = _source_row(settings={"collections_origin": collections_feeds.FEED_ORIGIN})
            self.job = _job_row()

        def create_source(self, **kwargs):
            self.source = _source_row(
                settings=collections_feeds.json.loads(kwargs["settings_json"]),
                tags=kwargs["tags"],
            )
            return self.source

        def create_job(self, **_kwargs):
            return self.job

        def update_source(self, _source_id: int, patch: dict):
            self.source = _source_row(settings=collections_feeds.json.loads(patch["settings_json"]))
            return self.source

        def get_source(self, _source_id: int):
            return self.source

        def get_job(self, _job_id: int):
            return self.job

    async def _raise_first_run(*_args, **_kwargs):
        raise RuntimeError("watchlist pipeline exploded at /private/feeds-first-run.db")

    fake_logger = MagicMock()
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(collections_feeds, "logger", fake_logger)
    monkeypatch.setattr(collections_feeds, "_compute_next_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collections_feeds, "_register_schedule", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collections_feeds, "record_watchlist_source_created", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline_module, "run_watchlist_job", _raise_first_run)

    response = await collections_feeds.create_feed_subscription(
        payload=CollectionsFeedCreateRequest(url="https://example.com/feed.xml", active=True),
        background_tasks=background_tasks,
        current_user=SimpleNamespace(id=42),
        db=_CreateFeedDb(),
    )
    await background_tasks()

    assert response.id == 7
    fake_logger.debug.assert_called_once_with("collections_feeds_first_run_failed")
    rendered = " ".join(str(part) for call_args in fake_logger.debug.call_args_list for part in call_args.args)
    assert "/private/feeds-first-run.db" not in rendered
    assert "exploded" not in rendered
    assert "9" not in rendered
