from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _FixedClock:
    def now_utc(self) -> datetime:
        return NOW


class _OwnerMediaDB:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.lookup_calls: list[list[str]] = []

    def get_media_by_urls(self, urls, **_kwargs):
        self.lookup_calls.append(list(urls))
        return list(self.rows)

    def close_connection(self) -> None:
        return None


@pytest.fixture()
def service_context(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "playlist-service-jobs.db", clock=_FixedClock())
    media_db = _OwnerMediaDB()
    service = PlaylistIngestService(manager)
    service._media_db_factory = lambda _owner: media_db
    return service, PlaylistIngestStore(manager), manager, media_db


def _seed_materialized_video(store, normalized_source_id: str = "youtube:video:abc"):
    preflight = store.create_preflight(
        "owner-1",
        source_url="https://www.youtube.com/playlist?list=PL123",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "owner-1",
        preflight.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "occ-materialized",
                "ordinal": 1,
                "source_url": "https://www.youtube.com/watch?v=abc",
                "normalized_source_id": normalized_source_id,
                "source_kind": "youtube_video",
                "availability": "available",
                "duplicate_status": "new",
                "selected_by_default": True,
                "display_metadata": {"title": "Authoritative title"},
            }
        ],
    )
    materialization = store.create_materialization(
        "owner-1",
        preflight_id=preflight.preflight_id,
        occurrence_ids=["occ-materialized"],
        expires_at=NOW + timedelta(hours=1),
    )
    return {
        "input_kind": "materialized_playlist_item",
        "materialization_id": materialization.materialization_id,
        "occurrence_id": "occ-materialized",
    }


def _table_count(manager, table: str) -> int:
    connection = manager._connect()
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # nosec B608
    finally:
        connection.close()


def _direct_input(occurrence_id: str, url: str) -> dict:
    return {
        "input_kind": "direct_url",
        "occurrence_id": occurrence_id,
        "url": url,
        "source_kind": "video",
        "display_metadata": {"title": f"Title {occurrence_id}"},
    }


def _file_input(occurrence_id: str = "occ-file") -> dict:
    return {
        "input_kind": "file_stub",
        "occurrence_id": occurrence_id,
        "name": "episode.mp3",
        "content_type": "audio/mpeg",
        "size_bytes": 1234,
        "display_metadata": {"title": "Local episode"},
    }


def test_create_run_fresh_duplicate_requires_review_without_side_effects(service_context):
    service, store, manager, media_db = service_context
    materialized = _seed_materialized_video(store)
    media_db.rows = [{"id": 91, "url": "https://www.youtube.com/watch?v=abc"}]

    assert hasattr(service, "create_run"), "run creation is absent"

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as exc_info:
        service.create_run("owner-1", inputs=[materialized], review_overrides={})

    assert exc_info.value.items[0].occurrence_id == materialized["occurrence_id"]
    assert _table_count(manager, "media_ingest_runs") == 0
    assert _table_count(manager, "media_ingest_run_events") == 0
    assert _table_count(manager, "jobs") == 0


def test_create_run_mixed_inputs_preserve_identity_order_and_initial_state(service_context):
    service, store, manager, media_db = service_context
    materialized = _seed_materialized_video(store)
    direct = _direct_input("occ-direct", "https://Example.com/video/?b=2&a=1")
    file_stub = _file_input()

    created = service.create_run(
        "owner-1",
        inputs=[materialized, direct, file_stub],
        review_overrides={},
    )

    from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
        normalize_media_dedupe_url,
    )

    items = list(store.list_run_items("owner-1", created.run_id, limit=10))
    events = list(store.list_run_events("owner-1", created.run_id, limit=10))
    assert [item.occurrence_id for item in items] == [
        "occ-materialized",
        "occ-direct",
        "occ-file",
    ]
    assert [item.input_kind for item in items] == [
        "materialized_playlist_item",
        "direct_url",
        "file_stub",
    ]
    assert [item.state for item in items] == ["staged", "staged", "awaiting_upload"]
    assert [item.action for item in items] == ["ingest", "ingest", "ingest"]
    assert [item.attempt for item in items] == [1, 1, 1]
    assert items[0].source_url == "https://www.youtube.com/watch?v=abc"
    assert items[0].display_metadata == {"title": "Authoritative title"}
    assert items[1].source_url == normalize_media_dedupe_url(direct["url"])
    assert items[2].source_url is None
    assert items[2].display_metadata == {
        "content_type": "audio/mpeg",
        "name": "episode.mp3",
        "size_bytes": 1234,
        "title": "Local episode",
    }
    assert media_db.lookup_calls == [["https://www.youtube.com/watch?v=abc", normalize_media_dedupe_url(direct["url"])]]
    assert len(events) == 3
    assert [event.occurrence_id for event in events] == [item.occurrence_id for item in items]
    assert [event.attrs["action"] for event in events] == ["ingest", "ingest", "ingest"]
    assert store.get_run("owner-1", created.run_id).version == 2
    assert _table_count(manager, "jobs") == 0


def test_create_run_file_only_skips_library_lookup(service_context):
    service, store, _manager, media_db = service_context

    created = service.create_run(
        "owner-1",
        inputs=[_file_input()],
        review_overrides={},
    )

    assert media_db.lookup_calls == []
    assert store.list_run_items("owner-1", created.run_id)[0].state == "awaiting_upload"


def test_create_run_owner_library_open_failure_is_safe_and_has_no_side_effects(service_context):
    service, _store, manager, _media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    def fail_open(_owner):
        raise RuntimeError("/private/owner/token=do-not-leak")

    service._media_db_factory = fail_open
    with pytest.raises(PlaylistRunValidationError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-direct", "https://example.com/video")],
            review_overrides={},
        )

    assert str(exc_info.value) == "library_lookup_failed"
    assert "private" not in str(exc_info.value)
    assert _table_count(manager, "media_ingest_runs") == 0


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/watch?v=abc&list=PL123",
    ],
)
def test_create_run_direct_playlist_requires_preflight_without_side_effects(service_context, url):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistPreflightRequiredError,
    )

    with pytest.raises(PlaylistPreflightRequiredError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-playlist", url)],
            review_overrides={},
        )

    assert str(exc_info.value) == "playlist_preflight_required"
    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/video",
        "https://user:password@example.com/video",
        "https://example.com/video#fragment",
        "not-a-url",
    ],
)
def test_create_run_rejects_malformed_direct_url_before_lookup(service_context, url):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-url", url)],
            review_overrides={},
        )

    assert str(exc_info.value) == "invalid_direct_url"
    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


def test_create_run_expired_or_cross_owner_materialization_is_generic(service_context):
    service, store, manager, media_db = service_context
    materialized = _seed_materialized_video(store)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        service.create_run("owner-2", inputs=[materialized], review_overrides={})

    connection = manager._connect()
    try:
        connection.execute(
            "UPDATE playlist_materializations SET expires_at = ? WHERE materialization_id = ?",
            ((NOW - timedelta(seconds=1)).isoformat(), materialized["materialization_id"]),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        service.create_run("owner-1", inputs=[materialized], review_overrides={})

    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


def test_run_request_union_is_strict_bounded_and_occurrence_unique():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import (
        PlaylistIngestRunCreateRequest,
    )

    valid = PlaylistIngestRunCreateRequest.model_validate(
        {"inputs": [_direct_input("occ-1", "https://example.com/1")], "review_overrides": {}}
    )
    assert valid.inputs[0].input_kind == "direct_url"

    invalid_payloads = [
        {
            "inputs": [
                _direct_input("same", "https://example.com/1"),
                _file_input("same"),
            ],
            "review_overrides": {},
        },
        {"inputs": [{**_file_input(), "occurrence_id": 7}], "review_overrides": {}},
        {"inputs": [{**_file_input(), "path": "/private/file"}], "review_overrides": {}},
        {"inputs": [{**_file_input(), "size_bytes": -1}], "review_overrides": {}},
        {"inputs": [_file_input(f"occ-{index}") for index in range(501)], "review_overrides": {}},
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            PlaylistIngestRunCreateRequest.model_validate(payload)


def test_create_run_validates_missing_extra_and_stale_review_overrides(service_context):
    service, _store, manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as missing:
        service.create_run("owner-1", inputs=[direct], review_overrides={})
    assert [(item.occurrence_id, item.reason) for item in missing.value.items] == [
        ("occ-direct", "duplicate_action_required")
    ]

    with pytest.raises(PlaylistRunValidationError, match="unknown_review_override"):
        service.create_run(
            "owner-1",
            inputs=[direct],
            review_overrides={"unknown": {"duplicate_policy": "skip", "existing_media_id": 17}},
        )

    media_db.rows = []
    with pytest.raises(ReviewRequiredError) as stale:
        service.create_run(
            "owner-1",
            inputs=[direct],
            review_overrides={"occ-direct": {"duplicate_policy": "skip", "existing_media_id": 17}},
        )
    assert stale.value.items[0].reason == "duplicate_no_longer_present"
    assert _table_count(manager, "media_ingest_runs") == 0


def test_create_run_accepts_current_library_override_and_persists_patch(service_context):
    service, store, _manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    created = service.create_run(
        "owner-1",
        inputs=[direct],
        review_overrides={
            "occ-direct": {
                "duplicate_policy": "update_metadata_only",
                "existing_media_id": 17,
                "metadata_patch": {"title": "Reviewed title"},
            }
        },
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    assert item.action == "update_metadata_only"
    assert item.duplicate_policy == "update_metadata_only"
    assert item.metadata_patch == {"title": "Reviewed title"}
    assert item.media_id is None


def test_create_run_in_run_repeat_requires_occurrence_bound_override(service_context):
    service, store, manager, _media_db = service_context
    inputs = [
        _direct_input("occ-first", "https://youtu.be/abc"),
        _direct_input("occ-repeat", "https://www.youtube.com/watch?v=abc"),
    ]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as missing:
        service.create_run("owner-1", inputs=inputs, review_overrides={})
    assert len(missing.value.items) == 1
    assert missing.value.items[0].occurrence_id == "occ-repeat"
    assert missing.value.items[0].evidence.duplicate_of_occurrence_id == "occ-first"
    assert _table_count(manager, "media_ingest_runs") == 0

    created = service.create_run(
        "owner-1",
        inputs=inputs,
        review_overrides={
            "occ-repeat": {
                "duplicate_policy": "overwrite",
                "duplicate_of_occurrence_id": "occ-first",
            }
        },
    )
    items = list(store.list_run_items("owner-1", created.run_id, limit=10))
    assert [item.action for item in items] == ["ingest", "overwrite"]


def test_create_run_in_run_repeat_uses_existing_generic_url_normalization(service_context):
    service, _store, manager, _media_db = service_context
    inputs = [
        _direct_input("occ-first", "https://EXAMPLE.com/video/?utm_source=review&a=1"),
        _direct_input("occ-repeat", "https://example.com/video?a=1"),
    ]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as exc_info:
        service.create_run("owner-1", inputs=inputs, review_overrides={})

    assert [item.occurrence_id for item in exc_info.value.items] == ["occ-repeat"]
    assert exc_info.value.items[0].evidence.duplicate_of_occurrence_id == "occ-first"
    assert _table_count(manager, "media_ingest_runs") == 0


def test_create_run_manifest_failure_rolls_back_all_rows(service_context, monkeypatch):
    service, _store, manager, _media_db = service_context
    original_query = service._store._query
    event_inserts = 0

    def fail_second_event(db, sql, params=()):
        nonlocal event_inserts
        if "INSERT INTO media_ingest_run_events" in sql:
            event_inserts += 1
            if event_inserts == 2:
                raise RuntimeError("injected manifest failure")
        return original_query(db, sql, params)

    monkeypatch.setattr(service._store, "_query", fail_second_event)

    with pytest.raises(RuntimeError, match="injected manifest failure"):
        service.create_run(
            "owner-1",
            inputs=[_file_input("occ-1"), _file_input("occ-2")],
            review_overrides={},
        )

    assert _table_count(manager, "media_ingest_runs") == 0
    assert _table_count(manager, "media_ingest_run_items") == 0
    assert _table_count(manager, "media_ingest_run_events") == 0
