from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
        self.before_lookup = None
        self.metadata_calls: list[dict] = []
        self.metadata_error: Exception | None = None
        self.metadata_commit_then_error_once: Exception | None = None
        self.metadata_mutations = 0
        self._applied_metadata_patches: set[str] = set()

    def get_media_by_urls(self, urls, **_kwargs):
        self.lookup_calls.append(list(urls))
        if self.before_lookup is not None:
            self.before_lookup()
        return list(self.rows)

    def apply_media_metadata_patch(self, media_id, **patch):
        self.metadata_calls.append({"media_id": media_id, **patch})
        if self.metadata_error is not None:
            raise self.metadata_error
        fingerprint = repr((media_id, sorted(patch.items())))
        if fingerprint not in self._applied_metadata_patches:
            self._applied_metadata_patches.add(fingerprint)
            self.metadata_mutations += 1
        if self.metadata_commit_then_error_once is not None:
            error = self.metadata_commit_then_error_once
            self.metadata_commit_then_error_once = None
            raise error
        return {"media_id": media_id, "new_media_version": 2}

    def close_connection(self) -> None:
        return None


class _OwnerCollectionsDB:
    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.resolve_calls: list[dict] = []
        self.discard_calls: list[int] = []
        self.create_error: Exception | None = None
        self.create_commit_then_error_once: Exception | None = None
        self.resolve_error: Exception | None = None
        self.get_error: Exception | None = None
        self.restore_error: Exception | None = None
        self.discard_error: Exception | None = None
        self.restore_calls: list[dict] = []
        self.resolve_commit_then_error_once: Exception | None = None
        self.items: dict[int, SimpleNamespace] = {}
        self.last_collection = None

    def create_media_collection_with_items(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        items = [SimpleNamespace(id=701 + index, ordinal=item["ordinal"]) for index, item in enumerate(kwargs["items"])]
        self.items = {
            item.id: SimpleNamespace(
                id=item.id,
                status="planned",
                media_id=None,
                content_item_id=None,
                latest_job_id=None,
                latest_run_id=None,
                updated_at="2026-07-12T12:00:00+00:00",
            )
            for item in items
        }
        self.last_collection = SimpleNamespace(id=700, items=items)
        if self.create_commit_then_error_once is not None:
            error = self.create_commit_then_error_once
            self.create_commit_then_error_once = None
            raise error
        return self.last_collection

    def get_playlist_ingest_collection_for_run(self, _run_id):
        if self.last_collection is None:
            raise KeyError("media_collection_not_found")
        return self.last_collection

    def resolve_media_collection_item(self, item_id, **kwargs):
        self.resolve_calls.append({"item_id": item_id, **kwargs})
        if self.resolve_error is not None:
            raise self.resolve_error
        resolved = SimpleNamespace(
            id=item_id,
            status=kwargs["status"],
            media_id=kwargs["media_id"],
            content_item_id=None,
            latest_job_id=None,
            latest_run_id=None,
            updated_at="2026-07-12T12:00:01+00:00",
        )
        self.items[item_id] = resolved
        if self.resolve_commit_then_error_once is not None:
            error = self.resolve_commit_then_error_once
            self.resolve_commit_then_error_once = None
            raise error
        return resolved

    def get_media_collection_item(self, item_id):
        if self.get_error is not None:
            raise self.get_error
        return self.items[item_id]

    def restore_media_collection_item_plan(self, item_id, **kwargs):
        self.restore_calls.append({"item_id": item_id, **kwargs})
        if self.restore_error is not None:
            raise self.restore_error
        restored = SimpleNamespace(
            id=item_id,
            status="planned",
            media_id=None,
            content_item_id=None,
            latest_job_id=None,
            latest_run_id=None,
            updated_at="2026-07-12T12:00:02+00:00",
        )
        self.items[item_id] = restored
        return restored

    def discard_media_collection(self, collection_id, *, expected_item_ids):
        self.discard_calls.append(collection_id)
        if self.discard_error is not None:
            raise self.discard_error
        return True

    def close(self) -> None:
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
    collections_db = _OwnerCollectionsDB()
    service = PlaylistIngestService(manager)
    service._media_db_factory = lambda _owner: media_db
    service._collections_db_factory = lambda _owner: collections_db
    service.test_collections_db = collections_db
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


def test_preflight_materialization_and_run_mutations_invoke_bounded_cleanup(
    service_context,
    monkeypatch,
):
    service, store, _manager, _media_db = service_context
    monkeypatch.setenv("PLAYLIST_INGEST_CLEANUP_LIMIT", "3")
    cleanup_calls: list[tuple[str, int]] = []

    def record_cleanup(owner_user_id, *, limit, now=None):  # noqa: ARG001
        cleanup_calls.append((owner_user_id, limit))
        return {"preflights": 0, "materializations": 0, "runs": 0}

    monkeypatch.setattr(service._store, "cleanup_expired_resources", record_cleanup)
    created = service.create_preflight(
        "owner-1",
        url="https://www.youtube.com/playlist?list=PLcleanupseams",
        max_items=10,
        timeout_seconds=30,
    )
    store.replace_preflight_snapshot(
        "owner-1",
        created.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "cleanup-seam-occ",
                "ordinal": 1,
                "source_url": "https://www.youtube.com/watch?v=cleanupseam",
                "normalized_source_id": "youtube:video:cleanupseam",
                "source_kind": "youtube_video",
                "availability": "available",
                "duplicate_status": "new",
                "selected_by_default": True,
                "display_metadata": {"title": "Cleanup seam"},
            }
        ],
    )
    service.create_materialization(
        "owner-1",
        created.preflight_id,
        ["cleanup-seam-occ"],
    )
    service.create_run(
        "owner-1",
        inputs=[_direct_input("cleanup-direct", "https://example.com/cleanup-direct")],
        review_overrides={},
    )

    assert cleanup_calls == [("owner-1", 3), ("owner-1", 3), ("owner-1", 3)]


def test_cleanup_failure_is_sanitized_and_does_not_break_run_mutation(
    service_context,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_service

    service, _store, _manager, _media_db = service_context

    class _LoggerStub:
        def __init__(self) -> None:
            self.bindings: list[dict] = []
            self.messages: list[str] = []

        def bind(self, **kwargs):
            self.bindings.append(dict(kwargs))
            return self

        def warning(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

    logger_stub = _LoggerStub()

    def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError("https://youtube.com/playlist?list=private&token=secret")

    monkeypatch.setattr(service._store, "cleanup_expired_resources", fail_cleanup)
    monkeypatch.setattr(playlist_ingest_service, "logger", logger_stub)

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("cleanup-safe", "https://example.com/cleanup-safe")],
        review_overrides={},
    )

    assert created.run_id
    assert logger_stub.bindings == [{"error_type": "RuntimeError"}]
    assert "secret" not in repr((logger_stub.bindings, logger_stub.messages))


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
    assert media_db.lookup_calls == [
        [
            "https://www.youtube.com/watch?v=abc",
            direct["url"],
            normalize_media_dedupe_url(direct["url"]),
        ]
    ]
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
        "https://www.youtube.com/watch?v=abc%26list%3DPL123",
        "https://www.youtube.com/watch?v=abc%3Flist%3DPL123",
        "https://www.youtube.com/watch?v=abc%2526list%253DPL123",
        "https://www.youtube.com/watch?v=abc%3Blist%3DPL123",
    ],
)
def test_create_run_encoded_youtube_playlist_injection_requires_preflight(service_context, url):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistPreflightRequiredError,
    )

    with pytest.raises(PlaylistPreflightRequiredError, match="playlist_preflight_required"):
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-encoded-playlist", url)],
            review_overrides={},
        )

    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


def test_create_run_normal_youtube_video_control_is_staged(service_context):
    service, store, _manager, media_db = service_context

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-video", "https://youtu.be/abc_123-Z")],
        review_overrides={},
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    assert item.source_url == "https://www.youtube.com/watch?v=abc_123-Z"
    assert item.normalized_source_id == "youtube:video:abc_123-Z"
    assert item.action == "ingest"
    assert media_db.lookup_calls == [
        [
            "https://youtu.be/abc_123-Z",
            "https://www.youtube.com/watch?v=abc_123-Z",
        ]
    ]


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


def test_run_request_new_collection_is_bounded_and_rejects_existing_collection_id():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import (
        PlaylistIngestRunCreateRequest,
    )

    valid = PlaylistIngestRunCreateRequest.model_validate(
        {
            "inputs": [_direct_input("occ-1", "https://example.com/1")],
            "review_overrides": {},
            "new_collection": {
                "name": "  Playlist research  ",
                "description": "Selected videos",
                "source_url": "https://www.youtube.com/playlist?list=PL123",
                "default_tags": [" research ", "video"],
            },
        }
    )
    assert valid.new_collection.name == "Playlist research"
    assert valid.new_collection.default_tags == ["research", "video"]

    invalid_collections = [
        {"name": "x" * 256},
        {"name": "Playlist", "description": "x" * 2001},
        {"name": "Playlist", "source_url": "x" * 2049},
        {"name": "Playlist", "default_tags": ["tag"] * 51},
        {"name": "Playlist", "default_tags": ["x" * 101]},
        {"name": "Playlist", "source_url": "https://user:secret@example.com/list"},
        {"name": "Playlist", "source_url": "https://example.com/list?token=secret"},
        {"name": "Playlist", "source_url": "https://example.com/list#section"},
        {"name": "Playlist", "source_url": "https://example.com/list#access_token=secret"},
        {"name": "Playlist", "metadata": {"secret": "not accepted"}},
    ]
    for collection in invalid_collections:
        with pytest.raises(ValidationError):
            PlaylistIngestRunCreateRequest.model_validate(
                {
                    "inputs": [_direct_input("occ-1", "https://example.com/1")],
                    "review_overrides": {},
                    "new_collection": collection,
                }
            )

    for payload in (
        {"collection_id": 7},
        {"collection_id": 7, "new_collection": {"name": "Playlist"}},
    ):
        with pytest.raises(ValidationError):
            PlaylistIngestRunCreateRequest.model_validate(
                {
                    "inputs": [_direct_input("occ-1", "https://example.com/1")],
                    "review_overrides": {},
                    **payload,
                }
            )


@pytest.mark.parametrize(
    "collection_id",
    [
        pytest.param(41, id="foreign"),
        pytest.param(42, id="missing"),
        pytest.param(43, id="deleted"),
        pytest.param(44, id="non-null"),
    ],
)
def test_create_run_rejects_existing_collection_association_without_side_effects(
    service_context,
    collection_id,
):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError, match="invalid_run_request"):
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-direct", "https://example.com/video")],
            review_overrides={},
            collection_id=collection_id,
        )

    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0
    assert _table_count(manager, "jobs") == 0


def test_create_run_validates_missing_extra_and_stale_review_overrides(service_context):
    service, _store, manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as missing:
        service.create_run("owner-1", inputs=[direct], review_overrides={})
    assert [(item.occurrence_id, item.reason) for item in missing.value.items] == [
        ("occ-direct", "duplicate_action_required")
    ]

    with pytest.raises(ReviewRequiredError) as extra:
        service.create_run(
            "owner-1",
            inputs=[direct],
            review_overrides={"unknown": {"duplicate_policy": "skip", "existing_media_id": 17}},
        )
    assert [(item.occurrence_id, item.reason) for item in extra.value.items] == [
        ("occ-direct", "duplicate_action_required"),
        ("unknown", "unknown_review_override"),
    ]

    media_db.rows = []
    with pytest.raises(ReviewRequiredError) as stale:
        service.create_run(
            "owner-1",
            inputs=[direct],
            review_overrides={"occ-direct": {"duplicate_policy": "skip", "existing_media_id": 17}},
        )
    assert stale.value.items[0].reason == "duplicate_no_longer_present"
    assert _table_count(manager, "media_ingest_runs") == 0


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"duplicate_policy": "launch_missiles", "existing_media_id": 17}, "invalid_duplicate_override"),
        ({"duplicate_policy": "update_metadata_only", "existing_media_id": 17}, "invalid_duplicate_override"),
        (
            {
                "duplicate_policy": "update_metadata_only",
                "existing_media_id": 17,
                "metadata_patch": {},
            },
            "invalid_duplicate_override",
        ),
        (
            {
                "duplicate_policy": "update_metadata_only",
                "existing_media_id": 17,
                "metadata_patch": {"content": "forbidden"},
            },
            "invalid_duplicate_override",
        ),
        (
            {
                "duplicate_policy": "skip",
                "existing_media_id": 17,
                "metadata_patch": {"title": "not allowed"},
            },
            "invalid_duplicate_override",
        ),
        (
            {
                "duplicate_policy": "include_existing",
                "existing_media_id": 17,
                "metadata_patch": {"title": "not allowed"},
            },
            "invalid_duplicate_override",
        ),
        (
            {
                "duplicate_policy": "overwrite",
                "existing_media_id": 17,
                "metadata_patch": {"title": 42},
            },
            "invalid_duplicate_override",
        ),
    ],
)
def test_create_run_duplicate_override_semantics_are_reviewed_after_fresh_lookup(
    service_context,
    override,
    reason,
):
    service, _store, manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[direct],
            review_overrides={"occ-direct": override},
        )

    assert [(item.occurrence_id, item.reason) for item in exc_info.value.items] == [("occ-direct", reason)]
    assert [action.value for action in exc_info.value.items[0].allowed_actions] == [
        "skip",
        "include_existing",
        "update_metadata_only",
        "overwrite",
    ]
    assert media_db.lookup_calls == [["https://example.com/existing"]]
    assert _table_count(manager, "media_ingest_runs") == 0
    assert _table_count(manager, "media_ingest_run_items") == 0
    assert _table_count(manager, "media_ingest_run_events") == 0
    assert _table_count(manager, "jobs") == 0


def test_create_run_overwrite_with_valid_patch_control(service_context):
    service, store, _manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    created = service.create_run(
        "owner-1",
        inputs=[direct],
        review_overrides={
            "occ-direct": {
                "duplicate_policy": "overwrite",
                "existing_media_id": 17,
                "metadata_patch": {"title": "Reviewed overwrite"},
            }
        },
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    assert item.action == "overwrite"
    assert item.metadata_patch == {"title": "Reviewed overwrite"}
    assert media_db.lookup_calls == [["https://example.com/existing"]]


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
    assert item.state == "terminal"
    assert item.outcome == "metadata_updated"
    assert item.media_id == 17
    assert media_db.metadata_calls == [{"media_id": 17, "title": "Reviewed title"}]


@pytest.mark.parametrize(
    ("policy", "patch", "expected_outcome"),
    [
        ("skip", None, "skipped_existing"),
        ("include_existing", None, "included_existing"),
        (
            "update_metadata_only",
            {"title": "Reviewed", "author": "Speaker", "keywords_add": ["tag"]},
            "metadata_updated",
        ),
    ],
)
def test_create_run_resolves_nonprocessing_duplicate_actions_without_media_jobs(
    service_context,
    policy,
    patch,
    expected_outcome,
):
    service, store, manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    override = {
        "duplicate_policy": policy,
        "existing_media_id": 17,
    }
    if patch is not None:
        override["metadata_patch"] = patch

    created = service.create_run(
        "owner-1",
        inputs=[direct],
        review_overrides={"occ-direct": override},
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    events = list(store.list_run_events("owner-1", created.run_id))
    assert item.state == "terminal"
    assert item.outcome == expected_outcome
    assert item.media_id == 17
    assert events[-1].event_type == "duplicate_action_resolved"
    assert events[-1].outcome == expected_outcome
    assert _table_count(manager, "jobs") == 0
    assert media_db.metadata_calls == ([{"media_id": 17, **patch}] if patch is not None else [])


def test_create_run_metadata_conflict_becomes_terminal_failure_without_job(service_context):
    service, store, manager, media_db = service_context
    direct = _direct_input("occ-direct", "https://example.com/existing")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError

    media_db.metadata_error = ConflictError("Media", 17)

    created = service.create_run(
        "owner-1",
        inputs=[direct],
        review_overrides={
            "occ-direct": {
                "duplicate_policy": "update_metadata_only",
                "existing_media_id": 17,
                "metadata_patch": {"title": "Reviewed"},
            }
        },
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    events = list(store.list_run_events("owner-1", created.run_id))
    assert item.state == "terminal"
    assert item.outcome == "metadata_update_failed"
    assert item.media_id == 17
    assert events[-1].outcome == "metadata_update_failed"
    assert _table_count(manager, "jobs") == 0


def test_create_run_metadata_commit_then_error_retries_idempotently(service_context):
    service, store, manager, media_db = service_context
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    media_db.metadata_commit_then_error_once = RuntimeError("private post-commit detail")

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-direct", "https://example.com/existing")],
        review_overrides={
            "occ-direct": {
                "duplicate_policy": "update_metadata_only",
                "existing_media_id": 17,
                "metadata_patch": {"title": "Reviewed"},
            }
        },
    )

    item = store.get_run_item("owner-1", created.run_id, "occ-direct")
    assert item.state == "terminal"
    assert item.outcome == "metadata_updated"
    assert len(media_db.metadata_calls) == 2
    assert media_db.metadata_mutations == 1
    assert _table_count(manager, "jobs") == 0


def test_create_run_ambiguous_metadata_raises_pending_and_later_reconciles_same_run(
    service_context,
):
    service, store, manager, media_db = service_context
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    media_db.metadata_error = RuntimeError("private unknown write state")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-direct", "https://example.com/existing")],
            review_overrides={
                "occ-direct": {
                    "duplicate_policy": "update_metadata_only",
                    "existing_media_id": 17,
                    "metadata_patch": {"title": "Reviewed"},
                }
            },
        )

    run_id = pending.value.run_id
    assert str(pending.value) == "duplicate_action_pending"
    item = store.get_run_item("owner-1", run_id, "occ-direct")
    assert item.state == "preparing"
    assert item.outcome is None
    assert len(media_db.metadata_calls) == 2
    assert _table_count(manager, "jobs") == 0

    with pytest.raises(PlaylistIngestNotFoundError):
        service.reconcile_nonprocessing_actions("owner-2", run_id)
    assert len(media_db.metadata_calls) == 2

    media_db.metadata_error = None
    reconciled = service.reconcile_nonprocessing_actions("owner-1", run_id)
    assert reconciled.run_id == run_id
    assert reconciled.status == "completed"
    item = store.get_run_item("owner-1", run_id, "occ-direct")
    assert item.state == "terminal"
    assert item.outcome == "metadata_updated"
    assert media_db.metadata_mutations == 1
    assert len(media_db.metadata_calls) == 3

    replayed = service.reconcile_nonprocessing_actions("owner-1", run_id)
    assert replayed == reconciled
    assert len(media_db.metadata_calls) == 3
    assert [event.event_type for event in store.list_run_events("owner-1", run_id)].count(
        "duplicate_action_resolved"
    ) == 1


def test_reconcile_initial_get_run_failure_is_pending_without_side_effects(
    service_context,
    monkeypatch,
):
    service, store, manager, media_db = service_context
    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-direct", "https://example.com/video")],
        review_overrides={},
    )
    original_get_run = service._store.get_run
    original_list_run_items = service._store.list_run_items
    before_run = original_get_run("owner-1", created.run_id)
    before_items = tuple(original_list_run_items("owner-1", created.run_id))
    before_events = tuple(store.list_run_events("owner-1", created.run_id))
    monkeypatch.setattr(
        service._store,
        "get_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private get-run detail")),
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.reconcile_nonprocessing_actions("owner-1", created.run_id)

    assert pending.value.run_id == created.run_id
    assert str(pending.value) == "duplicate_action_pending"
    assert "private" not in str(pending.value)
    assert original_get_run("owner-1", created.run_id) == before_run
    assert tuple(original_list_run_items("owner-1", created.run_id)) == before_items
    assert tuple(store.list_run_events("owner-1", created.run_id)) == before_events
    assert media_db.metadata_calls == []
    assert _table_count(manager, "jobs") == 0


def test_reconcile_initial_list_items_failure_is_pending_without_side_effects(
    service_context,
    monkeypatch,
):
    service, store, manager, media_db = service_context
    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-direct", "https://example.com/video")],
        review_overrides={},
    )
    original_get_run = service._store.get_run
    original_list_run_items = service._store.list_run_items
    before_run = original_get_run("owner-1", created.run_id)
    before_items = tuple(original_list_run_items("owner-1", created.run_id))
    before_events = tuple(store.list_run_events("owner-1", created.run_id))
    monkeypatch.setattr(
        service._store,
        "list_run_items",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private item-read detail")),
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.reconcile_nonprocessing_actions("owner-1", created.run_id)

    assert pending.value.run_id == created.run_id
    assert str(pending.value) == "duplicate_action_pending"
    assert "private" not in str(pending.value)
    assert original_get_run("owner-1", created.run_id) == before_run
    assert tuple(original_list_run_items("owner-1", created.run_id)) == before_items
    assert tuple(store.list_run_events("owner-1", created.run_id)) == before_events
    assert media_db.metadata_calls == []
    assert _table_count(manager, "jobs") == 0


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


def test_create_run_in_run_skip_is_terminal_without_media_id_or_job(service_context):
    service, store, manager, _media_db = service_context
    inputs = [
        _direct_input("occ-first", "https://youtu.be/abc"),
        _direct_input("occ-repeat", "https://www.youtube.com/watch?v=abc"),
    ]

    created = service.create_run(
        "owner-1",
        inputs=inputs,
        review_overrides={
            "occ-repeat": {
                "duplicate_policy": "skip",
                "duplicate_of_occurrence_id": "occ-first",
            }
        },
    )

    repeated = store.list_run_items("owner-1", created.run_id, limit=10)[1]
    assert repeated.state == "terminal"
    assert repeated.outcome == "skipped_existing"
    assert repeated.media_id is None
    assert _table_count(manager, "jobs") == 0


@pytest.mark.parametrize(
    ("policy", "patch"),
    [
        ("include_existing", None),
        ("update_metadata_only", {"title": "Cannot target an in-run duplicate"}),
    ],
)
def test_create_run_rejects_in_run_reuse_without_media_id_before_side_effects(
    service_context,
    policy,
    patch,
):
    service, _store, manager, media_db = service_context
    collections_db = service.test_collections_db
    override = {
        "duplicate_policy": policy,
        "duplicate_of_occurrence_id": "occ-first",
    }
    if patch is not None:
        override["metadata_patch"] = patch

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[
                _direct_input("occ-first", "https://youtu.be/abc"),
                _direct_input("occ-repeat", "https://www.youtube.com/watch?v=abc"),
            ],
            review_overrides={"occ-repeat": override},
            new_collection={"name": "Must not be created"},
        )

    item = exc_info.value.items[0]
    assert item.occurrence_id == "occ-repeat"
    assert item.reason == "in_run_duplicate_requires_processing_or_skip"
    assert [action.value for action in item.allowed_actions] == ["skip", "overwrite"]
    assert media_db.metadata_calls == []
    assert collections_db.create_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0
    assert _table_count(manager, "jobs") == 0


def test_create_run_plans_non_skip_collection_items_and_resolves_existing_membership(service_context):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    media_db.rows = [
        {"id": 11, "url": "https://example.com/skip"},
        {"id": 12, "url": "https://example.com/include"},
    ]

    created = service.create_run(
        "owner-1",
        inputs=[
            _direct_input("occ-skip", "https://example.com/skip"),
            _direct_input("occ-include", "https://example.com/include"),
            _direct_input("occ-new", "https://example.com/new"),
        ],
        review_overrides={
            "occ-skip": {"duplicate_policy": "skip", "existing_media_id": 11},
            "occ-include": {"duplicate_policy": "include_existing", "existing_media_id": 12},
        },
        new_collection={
            "name": "Playlist research",
            "description": "Selected videos",
            "default_tags": ["research"],
        },
    )

    items = list(store.list_run_items("owner-1", created.run_id, limit=10))
    assert created.collection_id == 700
    assert [item.planned_collection_item_id for item in items] == [None, 701, 702]
    assert [item["source_url"] for item in collections_db.create_calls[0]["items"]] == [
        "https://example.com/include",
        "https://example.com/new",
    ]
    assert collections_db.resolve_calls == [{"item_id": 701, "media_id": 12, "status": "skipped_existing"}]
    assert _table_count(manager, "jobs") == 0


def test_create_run_resolves_successful_metadata_update_collection_membership(service_context):
    service, store, _manager, media_db = service_context
    collections_db = service.test_collections_db
    media_db.rows = [{"id": 17, "url": "https://example.com/update"}]

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-update", "https://example.com/update")],
        review_overrides={
            "occ-update": {
                "duplicate_policy": "update_metadata_only",
                "existing_media_id": 17,
                "metadata_patch": {"title": "Reviewed"},
            }
        },
        new_collection={"name": "Metadata updates"},
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    assert item.planned_collection_item_id == 701
    assert collections_db.resolve_calls == [{"item_id": 701, "media_id": 17, "status": "completed"}]


@pytest.mark.parametrize(
    ("policy", "metadata_patch", "expected_metadata_calls"),
    [
        ("include_existing", None, []),
        ("update_metadata_only", {"title": "Reviewed"}, [{"media_id": 17, "title": "Reviewed"}]),
    ],
)
def test_create_run_collection_membership_failure_terminalizes_complete_action_without_job(
    service_context,
    policy,
    metadata_patch,
    expected_metadata_calls,
):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    collections_db.resolve_error = RuntimeError("private membership detail")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    override = {"duplicate_policy": policy, "existing_media_id": 17}
    if metadata_patch is not None:
        override["metadata_patch"] = metadata_patch

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-existing", "https://example.com/existing")],
        review_overrides={"occ-existing": override},
        new_collection={"name": "Membership failure"},
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    events = list(store.list_run_events("owner-1", created.run_id))
    assert created.status == "completed"
    assert item.state == "terminal"
    assert item.outcome == "metadata_update_failed"
    assert events[-1].event_type == "duplicate_action_resolved"
    assert events[-1].outcome == "metadata_update_failed"
    assert media_db.metadata_calls == expected_metadata_calls
    assert collections_db.resolve_calls == [
        {
            "item_id": 701,
            "media_id": 17,
            "status": "skipped_existing" if policy == "include_existing" else "completed",
        }
    ]
    assert collections_db.restore_calls == []
    assert _table_count(manager, "jobs") == 0


def test_create_run_ambiguous_membership_readback_raises_pending_without_job(service_context):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    collections_db.resolve_error = RuntimeError("private membership write state")
    collections_db.get_error = RuntimeError("private membership read state")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-existing", "https://example.com/existing")],
            review_overrides={
                "occ-existing": {
                    "duplicate_policy": "include_existing",
                    "existing_media_id": 17,
                }
            },
            new_collection={"name": "Ambiguous membership readback"},
        )

    item = store.get_run_item("owner-1", pending.value.run_id, "occ-existing")
    assert item.state == "preparing"
    assert item.outcome is None
    assert _table_count(manager, "jobs") == 0


def test_create_run_prepare_store_failure_raises_sanitized_pending_identity(
    service_context,
    monkeypatch,
):
    service, store, manager, media_db = service_context
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    monkeypatch.setattr(
        service._store,
        "prepare_nonprocessing_run_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private store detail")),
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-existing", "https://example.com/existing")],
            review_overrides={
                "occ-existing": {
                    "duplicate_policy": "include_existing",
                    "existing_media_id": 17,
                }
            },
        )

    assert str(pending.value) == "duplicate_action_pending"
    assert "private" not in str(pending.value)
    assert store.get_run_item("owner-1", pending.value.run_id, "occ-existing").state == "staged"
    assert _table_count(manager, "jobs") == 0


def test_create_run_membership_commit_then_error_reconciles_without_restore(service_context):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    collections_db.resolve_commit_then_error_once = RuntimeError("private post-commit detail")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-existing", "https://example.com/existing")],
        review_overrides={
            "occ-existing": {
                "duplicate_policy": "include_existing",
                "existing_media_id": 17,
            }
        },
        new_collection={"name": "Ambiguous membership"},
    )

    item = store.get_run_item("owner-1", created.run_id, "occ-existing")
    assert item.state == "terminal"
    assert item.outcome == "included_existing"
    assert collections_db.restore_calls == []
    assert _table_count(manager, "jobs") == 0


def test_create_run_finalization_failure_restores_exact_resolved_membership(service_context, monkeypatch):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    monkeypatch.setattr(
        service._store,
        "resolve_nonprocessing_run_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private finalization detail")),
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-existing", "https://example.com/existing")],
            review_overrides={"occ-existing": {"duplicate_policy": "include_existing", "existing_media_id": 17}},
            new_collection={"name": "Finalization failure"},
        )

    item = store.list_run_items("owner-1", pending.value.run_id)[0]
    assert item.state == "preparing"
    assert collections_db.resolve_calls == [{"item_id": 701, "media_id": 17, "status": "skipped_existing"}]
    assert collections_db.restore_calls == [
        {
            "item_id": 701,
            "expected_media_id": 17,
            "expected_status": "skipped_existing",
            "expected_updated_at": "2026-07-12T12:00:01+00:00",
        }
    ]
    assert _table_count(manager, "jobs") == 0

    monkeypatch.undo()
    reconciled = service.reconcile_nonprocessing_actions("owner-1", pending.value.run_id)
    item = store.list_run_items("owner-1", pending.value.run_id)[0]
    assert reconciled.run_id == pending.value.run_id
    assert reconciled.status == "completed"
    assert item.state == "terminal"
    assert item.outcome == "included_existing"
    assert collections_db.resolve_calls == [
        {"item_id": 701, "media_id": 17, "status": "skipped_existing"},
        {"item_id": 701, "media_id": 17, "status": "skipped_existing"},
    ]
    assert _table_count(manager, "jobs") == 0


def test_create_run_finalization_commit_then_error_never_restores_terminal_membership(
    service_context,
    monkeypatch,
):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    original_resolve = service._store.resolve_nonprocessing_run_item

    def resolve_then_raise(*args, **kwargs):
        original_resolve(*args, **kwargs)
        raise RuntimeError("private post-commit detail")

    monkeypatch.setattr(service._store, "resolve_nonprocessing_run_item", resolve_then_raise)

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-existing", "https://example.com/existing")],
        review_overrides={"occ-existing": {"duplicate_policy": "include_existing", "existing_media_id": 17}},
        new_collection={"name": "Ambiguous finalization"},
    )

    item = store.get_run_item("owner-1", created.run_id, "occ-existing")
    assert item.state == "terminal"
    assert item.outcome == "included_existing"
    assert collections_db.restore_calls == []
    assert _table_count(manager, "jobs") == 0


def test_create_run_finalization_readback_failure_is_pending_and_reconciles_same_run(
    service_context,
    monkeypatch,
):
    service, store, manager, media_db = service_context
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    monkeypatch.setattr(
        service._store,
        "resolve_nonprocessing_run_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private finalization detail")),
    )
    monkeypatch.setattr(
        service._store,
        "get_run_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private readback detail")),
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-existing", "https://example.com/existing")],
            review_overrides={"occ-existing": {"duplicate_policy": "include_existing", "existing_media_id": 17}},
            new_collection={"name": "Finalization readback failure"},
        )

    assert str(pending.value) == "duplicate_action_pending"
    assert "private" not in str(pending.value)
    assert store.get_run_item("owner-1", pending.value.run_id, "occ-existing").state == "preparing"
    assert _table_count(manager, "jobs") == 0

    monkeypatch.undo()
    reconciled = service.reconcile_nonprocessing_actions("owner-1", pending.value.run_id)
    item = store.get_run_item("owner-1", pending.value.run_id, "occ-existing")
    assert reconciled.run_id == pending.value.run_id
    assert reconciled.status == "completed"
    assert item.state == "terminal"
    assert item.outcome == "included_existing"
    assert _table_count(manager, "jobs") == 0


def test_create_run_finalization_cleanup_failure_is_pending_and_reconciles_same_run(
    service_context,
    monkeypatch,
):
    service, store, manager, media_db = service_context
    collections_db = service.test_collections_db
    collections_db.restore_error = RuntimeError("private cleanup detail")
    media_db.rows = [{"id": 17, "url": "https://example.com/existing"}]
    monkeypatch.setattr(
        service._store,
        "resolve_nonprocessing_run_item",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private finalization detail")),
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunPendingError,
    )

    with pytest.raises(PlaylistRunPendingError) as pending:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-existing", "https://example.com/existing")],
            review_overrides={"occ-existing": {"duplicate_policy": "include_existing", "existing_media_id": 17}},
            new_collection={"name": "Cleanup failure"},
        )

    assert str(pending.value) == "duplicate_action_pending"
    assert "private" not in str(pending.value)
    assert store.list_run_items("owner-1", pending.value.run_id)[0].state == "preparing"
    with pytest.raises(PlaylistRunPendingError) as repeated:
        service.reconcile_nonprocessing_actions("owner-1", pending.value.run_id)
    assert repeated.value.run_id == pending.value.run_id
    assert len(collections_db.restore_calls) == 2
    assert _table_count(manager, "jobs") == 0

    collections_db.restore_error = None
    monkeypatch.undo()
    reconciled = service.reconcile_nonprocessing_actions("owner-1", pending.value.run_id)
    item = store.list_run_items("owner-1", pending.value.run_id)[0]
    assert reconciled.run_id == pending.value.run_id
    assert reconciled.status == "completed"
    assert item.state == "terminal"
    assert item.outcome == "included_existing"
    assert _table_count(manager, "jobs") == 0


def test_create_run_collection_attachment_failure_discards_plan_and_keeps_run_unsubmitted(
    service_context,
    monkeypatch,
):
    service, store, manager, _media_db = service_context
    collections_db = service.test_collections_db
    monkeypatch.setattr(
        service._store,
        "attach_collection_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("attach exploded")),
        raising=False,
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError, match="collection_planning_failed"):
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-new", "https://example.com/new")],
            review_overrides={},
            new_collection={"name": "Discard me"},
        )

    connection = manager._connect()
    try:
        run_id = connection.execute("SELECT run_id FROM media_ingest_runs").fetchone()[0]
    finally:
        connection.close()
    run = store.get_run("owner-1", run_id)
    item = store.list_run_items("owner-1", run_id)[0]
    assert run.status == "staged"
    assert run.collection_id is None
    assert item.planned_collection_item_id is None
    assert collections_db.discard_calls == [700]
    assert _table_count(manager, "jobs") == 0


def test_create_run_attachment_commit_then_exception_reconciles_without_discard(
    service_context,
    monkeypatch,
):
    service, store, manager, _media_db = service_context
    collections_db = service.test_collections_db
    original_attach = service._store.attach_collection_plan

    def attach_then_raise(*args, **kwargs):
        original_attach(*args, **kwargs)
        raise RuntimeError("private post-commit detail")

    monkeypatch.setattr(service._store, "attach_collection_plan", attach_then_raise)

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-new", "https://example.com/new")],
        review_overrides={},
        new_collection={"name": "Keep attached"},
    )

    item = store.list_run_items("owner-1", created.run_id)[0]
    assert created.collection_id == 700
    assert item.planned_collection_item_id == 701
    assert collections_db.discard_calls == []
    assert _table_count(manager, "jobs") == 0


def test_create_run_collection_creation_commit_then_exception_reconciles(
    service_context,
):
    service, store, manager, _media_db = service_context
    collections_db = service.test_collections_db
    collections_db.create_commit_then_error_once = RuntimeError("private post-commit detail")

    created = service.create_run(
        "owner-1",
        inputs=[_direct_input("occ-new", "https://example.com/new")],
        review_overrides={},
        new_collection={"name": "Keep committed plan"},
    )

    item = store.get_run_item("owner-1", created.run_id, "occ-new")
    assert created.collection_id == 700
    assert item.planned_collection_item_id == 701
    assert collections_db.discard_calls == []
    assert collections_db.create_calls[0]["metadata"] == {"playlist_ingest_run_id": created.run_id}
    assert _table_count(manager, "jobs") == 0


def test_create_run_collection_creation_failure_keeps_staged_run_without_side_effects(service_context):
    service, store, manager, _media_db = service_context
    collections_db = service.test_collections_db
    collections_db.create_error = RuntimeError("private create detail")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError, match="collection_planning_failed"):
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-new", "https://example.com/new")],
            review_overrides={},
            new_collection={"name": "Create fails"},
        )

    connection = manager._connect()
    try:
        run_id = connection.execute("SELECT run_id FROM media_ingest_runs").fetchone()[0]
    finally:
        connection.close()
    run = store.get_run("owner-1", run_id)
    assert run.status == "staged"
    assert run.collection_id is None
    assert collections_db.discard_calls == []
    assert _table_count(manager, "jobs") == 0


def test_create_run_collection_factory_failure_is_safe_and_keeps_staged_run(service_context):
    service, store, manager, _media_db = service_context
    service._collections_db_factory = lambda _owner: (_ for _ in ()).throw(RuntimeError("private factory detail"))

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-new", "https://example.com/new")],
            review_overrides={},
            new_collection={"name": "Factory fails"},
        )

    assert str(exc_info.value) == "collection_planning_failed"
    assert "private" not in str(exc_info.value)
    connection = manager._connect()
    try:
        run_id = connection.execute("SELECT run_id FROM media_ingest_runs").fetchone()[0]
    finally:
        connection.close()
    run = store.get_run("owner-1", run_id)
    assert run.status == "staged"
    assert run.collection_id is None
    assert _table_count(manager, "jobs") == 0


def test_create_run_collection_cleanup_failure_is_safe_and_explicit(service_context, monkeypatch):
    service, _store, manager, _media_db = service_context
    collections_db = service.test_collections_db
    collections_db.discard_error = RuntimeError("private cleanup detail")
    monkeypatch.setattr(
        service._store,
        "attach_collection_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private attach detail")),
        raising=False,
    )

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-new", "https://example.com/new")],
            review_overrides={},
            new_collection={"name": "Cleanup fails"},
        )

    assert str(exc_info.value) == "collection_planning_cleanup_failed"
    assert "private" not in str(exc_info.value)
    assert _table_count(manager, "jobs") == 0


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


def test_create_run_direct_duplicate_lookup_preserves_raw_and_canonical_candidates(service_context):
    service, store, _manager, media_db = service_context
    raw_url = "https://example.com/video?utm_source=review&a=1"
    canonical_url = "https://example.com/video?a=1"
    direct = _direct_input("occ-direct", raw_url)
    media_db.rows = [{"id": 23, "url": raw_url}]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        ReviewRequiredError,
    )

    with pytest.raises(ReviewRequiredError) as exc_info:
        service.create_run("owner-1", inputs=[direct], review_overrides={})

    assert exc_info.value.items[0].evidence.existing_media_id == 23
    assert media_db.lookup_calls == [[raw_url, canonical_url]]

    created = service.create_run(
        "owner-1",
        inputs=[direct],
        review_overrides={"occ-direct": {"duplicate_policy": "skip", "existing_media_id": 23}},
    )
    item = store.list_run_items("owner-1", created.run_id)[0]
    events = list(store.list_run_events("owner-1", created.run_id))
    assert item.source_url == canonical_url
    assert raw_url not in repr(item)
    assert raw_url not in repr(events)


@pytest.mark.parametrize(
    "query_key",
    [
        "token",
        "SECRET",
        "signature",
        "Key",
        "password",
        "authorization",
        "cookie_value",
        "credential_id",
    ],
)
def test_create_run_rejects_credential_query_keys_before_lookup(service_context, query_key):
    service, _store, manager, media_db = service_context
    secret_value = "do-not-leak-value"

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError) as exc_info:
        service.create_run(
            "owner-1",
            inputs=[
                _direct_input(
                    "occ-secret",
                    f"https://example.com/video?{query_key}={secret_value}",
                )
            ],
            review_overrides={},
        )

    assert str(exc_info.value) == "invalid_direct_url"
    assert secret_value not in str(exc_info.value)
    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


@pytest.mark.parametrize(
    ("inputs", "overrides", "kwargs"),
    [
        ([{**_file_input(), "size_bytes": "12"}], {}, {}),
        ([{**_file_input(), "size_bytes": True}], {}, {}),
        (
            [
                {
                    **_direct_input("occ-direct", "https://example.com/video"),
                    "display_metadata": {"duration_seconds": "12"},
                }
            ],
            {},
            {},
        ),
        (
            [
                {
                    **_direct_input("occ-direct", "https://example.com/video"),
                    "display_metadata": {"duration_seconds": True},
                }
            ],
            {},
            {},
        ),
        ([_file_input()], {}, {"collection_id": "12"}),
        ([_file_input()], {}, {"collection_id": True}),
        (
            [_direct_input("occ-direct", "https://example.com/video")],
            {"occ-direct": {"duplicate_policy": "skip", "existing_media_id": "17"}},
            {},
        ),
        (
            [_direct_input("occ-direct", "https://example.com/video")],
            {"occ-direct": {"duplicate_policy": "skip", "existing_media_id": True}},
            {},
        ),
    ],
)
def test_create_run_rejects_coerced_run_integers_before_lookup(
    service_context,
    inputs,
    overrides,
    kwargs,
):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError, match="invalid_run_request"):
        service.create_run(
            "owner-1",
            inputs=inputs,
            review_overrides=overrides,
            **kwargs,
        )

    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"processing_options": {"temperature": float("nan")}},
        {"processing_options": {"temperature": float("inf")}},
        {"processing_options": {"temperature": float("-inf")}},
        {"processing_options": {"nested": {"not_json": object()}}},
        {"playlist_summaries": [{"score": float("nan")}]},
    ],
)
def test_create_run_rejects_nonfinite_or_nonjson_options_before_lookup(service_context, kwargs):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError, match="invalid_run_request"):
        service.create_run(
            "owner-1",
            inputs=[_file_input()],
            review_overrides={},
            **kwargs,
        )

    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


def test_create_run_rejects_nonfinite_raw_patch_before_lookup(service_context):
    service, _store, manager, media_db = service_context

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistRunValidationError,
    )

    with pytest.raises(PlaylistRunValidationError, match="invalid_run_request"):
        service.create_run(
            "owner-1",
            inputs=[_direct_input("occ-direct", "https://example.com/video")],
            review_overrides={
                "occ-direct": {
                    "duplicate_policy": "overwrite",
                    "existing_media_id": 17,
                    "metadata_patch": {"title": float("nan")},
                }
            },
        )

    assert media_db.lookup_calls == []
    assert _table_count(manager, "media_ingest_runs") == 0


def test_create_run_revalidates_materialized_authority_after_duplicate_lookup(service_context):
    service, store, manager, media_db = service_context
    materialized = _seed_materialized_video(store)

    def expire_materialization():
        connection = manager._connect()
        try:
            connection.execute(
                "UPDATE playlist_materializations SET status = 'expired' WHERE materialization_id = ?",
                (materialized["materialization_id"],),
            )
            connection.commit()
        finally:
            connection.close()

    media_db.before_lookup = expire_materialization

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        service.create_run("owner-1", inputs=[materialized], review_overrides={})

    assert _table_count(manager, "media_ingest_runs") == 0
    assert _table_count(manager, "media_ingest_run_items") == 0
    assert _table_count(manager, "media_ingest_run_events") == 0


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
