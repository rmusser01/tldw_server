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
        self.before_lookup = None

    def get_media_by_urls(self, urls, **_kwargs):
        self.lookup_calls.append(list(urls))
        if self.before_lookup is not None:
            self.before_lookup()
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
