import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _FixedClock:
    def __init__(self) -> None:
        self.current = NOW

    def now_utc(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    clock = _FixedClock()
    manager = JobManager(db_path=tmp_path / "playlist-jobs.db", clock=clock)
    playlist_store = PlaylistIngestStore(manager)
    playlist_store.test_clock = clock
    return playlist_store


def _preflight_item(
    occurrence_id: str,
    ordinal: int,
    *,
    display_metadata: dict | None = None,
) -> dict:
    return {
        "occurrence_id": occurrence_id,
        "ordinal": ordinal,
        "occurrence_index_for_source": 1,
        "source_url": f"https://www.youtube.com/watch?v={ordinal}",
        "normalized_source_id": f"youtube:{ordinal}",
        "source_kind": "youtube",
        "availability": "available",
        "duplicate_status": "not_found",
        "selected_by_default": True,
        "display_metadata": display_metadata or {"title": f"Video {ordinal}"},
    }


def _seed_preflight(store, *, owner_id: str = "1", status: str = "ready", item_count: int = 2):
    preflight = store.create_preflight(
        owner_id,
        source_url="https://www.youtube.com/playlist?list=PL123",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    items = [_preflight_item(f"occ-{index}", index) for index in range(1, item_count + 1)]
    store.replace_preflight_snapshot(
        owner_id,
        preflight.preflight_id,
        status=status,
        items=items,
        summary={"item_count": item_count},
    )
    return store.get_preflight(owner_id, preflight.preflight_id)


def _seed_materialization(store, *, owner_id: str = "1", item_count: int = 2):
    ready = _seed_preflight(store, owner_id=owner_id, item_count=item_count)
    return store.create_materialization(
        owner_id,
        preflight_id=ready.preflight_id,
        occurrence_ids=[f"occ-{index}" for index in range(1, item_count + 1)],
        expires_at=NOW + timedelta(hours=1),
    )


def test_ready_snapshot_guard_rejects_cancelled_linked_job_atomically(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistPreflightLeaseLostError,
    )

    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload={"preflight_id": "guarded"},
        owner_user_id="1",
    )
    preflight = store.create_preflight(
        "1",
        source_url="https://example.com/guarded",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
        job_id=int(job["id"]),
    )
    claimed = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="guard-test",
    )
    assert claimed is not None
    assert store._jobs.cancel_job(int(job["id"]), reason="race") is True

    with pytest.raises(PlaylistPreflightLeaseLostError) as exc_info:
        store.replace_preflight_snapshot(
            "1",
            preflight.preflight_id,
            status="ready",
            items=[_preflight_item("must-not-persist", 1)],
            expected_job_id=int(claimed["id"]),
            expected_lease_id=str(claimed["lease_id"]),
            expected_worker_id=str(claimed["worker_id"]),
        )

    assert exc_info.value.cancelled is True
    assert store.get_preflight("1", preflight.preflight_id).status == "pending"
    assert list(store.list_preflight_items("1", preflight.preflight_id, limit=10)) == []


def test_stale_reclaimed_worker_cannot_write_snapshot_without_lease_identity(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
        PlaylistPreflightLeaseLostError,
    )

    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload={"preflight_id": "stale-guard"},
        owner_user_id="1",
    )
    preflight = store.create_preflight(
        "1",
        source_url="https://example.com/stale-guard",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
        job_id=int(job["id"]),
    )
    stale_claim = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="worker-a",
    )
    assert stale_claim is not None
    connection = store._jobs._connect()
    try:
        connection.execute(
            "UPDATE jobs SET leased_until = DATETIME('now', '-1 second') WHERE id = ?",
            (int(job["id"]),),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(PlaylistPreflightLeaseLostError):
        store.replace_preflight_snapshot(
            "1",
            preflight.preflight_id,
            status="running",
            items=[_preflight_item("expired-item", 1)],
            expected_job_id=int(stale_claim["id"]),
            expected_lease_id=str(stale_claim["lease_id"]),
            expected_worker_id=str(stale_claim["worker_id"]),
        )

    active_claim = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="worker-b",
    )
    assert active_claim is not None
    assert active_claim["lease_id"] != stale_claim["lease_id"]

    with pytest.raises(PlaylistIngestConflictError):
        store.replace_preflight_snapshot(
            "1",
            preflight.preflight_id,
            status="running",
            items=[_preflight_item("stale-item", 1)],
            expected_job_id=int(stale_claim["id"]),
        )

    assert store.get_preflight("1", preflight.preflight_id).status == "pending"
    assert list(store.list_preflight_items("1", preflight.preflight_id, limit=10)) == []

    store.replace_preflight_snapshot(
        "1",
        preflight.preflight_id,
        status="ready",
        items=[_preflight_item("active-item", 1)],
    )
    assert store.get_preflight("1", preflight.preflight_id).status == "ready"
    assert store.list_preflight_items("1", preflight.preflight_id)[0].occurrence_id == "active-item"


def test_snapshot_guard_fences_stale_lease_and_allows_active_lease(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistPreflightLeaseLostError,
    )

    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload={"preflight_id": "lease-guard"},
        owner_user_id="1",
    )
    preflight = store.create_preflight(
        "1",
        source_url="https://example.com/lease-guard",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
        job_id=int(job["id"]),
    )
    stale_claim = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="worker-a",
    )
    assert stale_claim is not None
    connection = store._jobs._connect()
    try:
        connection.execute(
            "UPDATE jobs SET leased_until = DATETIME('now', '-1 second') WHERE id = ?",
            (int(job["id"]),),
        )
        connection.commit()
    finally:
        connection.close()
    active_claim = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        lease_seconds=120,
        worker_id="worker-b",
    )
    assert active_claim is not None

    with pytest.raises(PlaylistPreflightLeaseLostError):
        store.replace_preflight_snapshot(
            "1",
            preflight.preflight_id,
            status="running",
            items=[_preflight_item("stale-item", 1)],
            expected_job_id=int(stale_claim["id"]),
            expected_lease_id=str(stale_claim["lease_id"]),
            expected_worker_id=str(stale_claim["worker_id"]),
        )

    assert store.get_preflight("1", preflight.preflight_id).status == "pending"
    assert list(store.list_preflight_items("1", preflight.preflight_id, limit=10)) == []

    store.replace_preflight_snapshot(
        "1",
        preflight.preflight_id,
        status="running",
        items=[],
        expected_job_id=int(active_claim["id"]),
        expected_lease_id=str(active_claim["lease_id"]),
        expected_worker_id=str(active_claim["worker_id"]),
    )
    store.replace_preflight_snapshot(
        "1",
        preflight.preflight_id,
        status="ready",
        items=[_preflight_item("active-item", 1)],
        expected_job_id=int(active_claim["id"]),
        expected_lease_id=str(active_claim["lease_id"]),
        expected_worker_id=str(active_claim["worker_id"]),
    )
    assert store.get_preflight("1", preflight.preflight_id).status == "ready"
    assert store.list_preflight_items("1", preflight.preflight_id)[0].occurrence_id == "active-item"


def test_duplicate_policy_choices_are_explicit():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import DuplicatePolicy

    assert {policy.value for policy in DuplicatePolicy} == {
        "skip",
        "include_existing",
        "update_metadata_only",
        "overwrite",
    }


def test_review_override_requires_explicit_duplicate_policy():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride()


def test_run_state_rejects_client_only_file_reattach_state():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="occ-1",
            ordinal=1,
            state="file_reattach_required",
        )


@pytest.mark.parametrize(
    ("state", "outcome"),
    [
        pytest.param("terminal", None, id="terminal-without-outcome"),
        pytest.param("running", "completed", id="nonterminal-with-outcome"),
    ],
)
def test_run_snapshot_requires_outcome_exactly_when_terminal(state, outcome):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="occ-1",
            ordinal=1,
            state=state,
            outcome=outcome,
        )


@pytest.mark.parametrize("duplicate_policy", ["update_metadata_only", "overwrite"])
def test_mutating_policies_accept_and_normalize_review_patch(duplicate_policy):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    override = ReviewOverride(
        duplicate_policy=duplicate_policy,
        metadata_patch={
            "title": "  Reviewed title  ",
            "author": "  Reviewed author  ",
            "keywords_add": ["  alpha  ", "beta"],
        },
    )

    assert override.model_dump()["metadata_patch"] == {
        "title": "Reviewed title",
        "author": "Reviewed author",
        "keywords_add": ["alpha", "beta"],
    }


@pytest.mark.parametrize(
    "metadata_patch",
    [
        pytest.param({}, id="empty"),
        pytest.param({"content": "forbidden"}, id="forbidden-content"),
        pytest.param({"title": "   "}, id="blank-title"),
        pytest.param({"author": "   "}, id="blank-author"),
        pytest.param({"keywords_add": []}, id="empty-keywords"),
        pytest.param({"keywords_add": ["   "]}, id="blank-keyword"),
        pytest.param({"title": "x" * 501}, id="oversize-title"),
        pytest.param({"author": "x" * 501}, id="oversize-author"),
        pytest.param({"keywords_add": ["x" * 129]}, id="oversize-keyword"),
        pytest.param({"keywords_add": ["tag"] * 101}, id="too-many-keywords"),
        pytest.param({"title": {"nested": "value"}}, id="deep-title"),
        pytest.param({"keywords_add": ("   ",)}, id="tuple-blank-keyword"),
        pytest.param({"keywords_add": ("x" * 129,)}, id="tuple-oversize-keyword"),
        pytest.param({"keywords_add": ("alpha",)}, id="tuple-keywords"),
        pytest.param({"title": b"   "}, id="bytes-title"),
        pytest.param({"author": b"Reviewed author"}, id="bytes-author"),
    ],
)
def test_review_patch_rejects_invalid_shape_and_values(metadata_patch):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride(
            duplicate_policy="overwrite",
            metadata_patch=metadata_patch,
        )


def test_update_metadata_policy_requires_patch():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride(duplicate_policy="update_metadata_only")


@pytest.mark.parametrize("duplicate_policy", ["skip", "include_existing"])
def test_nonmutating_duplicate_policies_reject_patch(duplicate_policy):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride(
            duplicate_policy=duplicate_policy,
            metadata_patch={"title": "Reviewed title"},
        )


def test_run_snapshot_normalizes_occurrence_id():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    snapshot = RunItemSnapshot(
        occurrence_id="  occ-1  ",
        ordinal=1,
        state="running",
    )

    assert snapshot.occurrence_id == "occ-1"


def test_run_snapshot_rejects_blank_occurrence_id():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="   ",
            ordinal=1,
            state="running",
        )


@pytest.mark.parametrize("occurrence_id", [b"   ", b"occ-1"])
def test_run_snapshot_rejects_bytes_occurrence_id(occurrence_id):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id=occurrence_id,
            ordinal=1,
            state="running",
        )


@pytest.mark.parametrize(("field", "value"), [("job_id", 0), ("media_id", -1)])
def test_run_snapshot_requires_positive_persisted_ids(field, value):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="occ-1",
            ordinal=1,
            state="running",
            **{field: value},
        )


def test_preflight_creation_is_owner_scoped(store):
    preflight = store.create_preflight(
        "1",
        source_url="https://www.youtube.com/playlist?list=PL-create",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )

    assert preflight.owner_user_id == "1"
    assert preflight.status == "pending"
    assert store.get_preflight("1", preflight.preflight_id) == preflight

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        store.get_preflight("2", preflight.preflight_id)


def test_snapshot_replacement_is_atomic(store):
    preflight = store.create_preflight(
        "1",
        source_url="https://www.youtube.com/playlist?list=PL-atomic",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    duplicate_occurrences = [
        _preflight_item("occ-duplicate", 1),
        _preflight_item("occ-duplicate", 2),
    ]

    with pytest.raises(ValueError, match="occurrence_id"):
        store.replace_preflight_snapshot(
            "1",
            preflight.preflight_id,
            status="ready",
            items=duplicate_occurrences,
        )

    persisted = store.get_preflight("1", preflight.preflight_id)
    assert persisted.status == "pending"
    assert list(store.list_preflight_items("1", preflight.preflight_id)) == []


@pytest.mark.parametrize("status", ["pending", "running", "cancelled", "failed"])
def test_materialization_requires_ready_snapshot(store, status):
    preflight = _seed_preflight(store, status=status)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        store.create_materialization(
            "1",
            preflight_id=preflight.preflight_id,
            occurrence_ids=["occ-1"],
        )


def test_materialization_rejects_expired_snapshot(store):
    preflight = store.create_preflight(
        "1",
        source_url="https://www.youtube.com/playlist?list=PL-expired",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "1",
        preflight.preflight_id,
        status="ready",
        items=[_preflight_item("occ-expired", 1)],
    )
    store.test_clock.advance(timedelta(hours=2))

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        store.create_materialization(
            "1",
            preflight_id=preflight.preflight_id,
            occurrence_ids=["occ-expired"],
        )


def test_materialization_keeps_only_compact_display_metadata(store):
    ready = store.create_preflight(
        "1",
        source_url="https://www.youtube.com/playlist?list=PL-policy",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "1",
        ready.preflight_id,
        status="ready",
        items=[
            _preflight_item("occ-1", 1),
            _preflight_item(
                "occ-2",
                2,
                display_metadata={
                    "title": "Video 2",
                    "channel_or_uploader": "Channel",
                    "duration_seconds": 123,
                    "published_at": "2026-01-02T03:04:05Z",
                    "thumbnail_url": "https://example.com/thumb.jpg",
                    "playlist_id": "PL-policy",
                    "playlist_title": "Policy playlist",
                    "duplicate_policy": "overwrite",
                    "metadata_patch": {"title": "Changed"},
                    "duplicate_evidence": {"matched_by": "url"},
                    "review_required": True,
                    "library_match_id": 42,
                    "media_id": 42,
                    "arbitrary": "must not survive",
                },
            ),
        ],
    )

    materialized = store.create_materialization(
        "1",
        preflight_id=ready.preflight_id,
        occurrence_ids=["occ-2"],
    )
    item = store.list_materialization_items("1", materialized.id)[0]

    assert item.occurrence_id == "occ-2"
    assert item.source_url.endswith("v=2")
    assert item.display_metadata == {
        "title": "Video 2",
        "channel_or_uploader": "Channel",
        "duration_seconds": 123,
        "published_at": "2026-01-02T03:04:05Z",
        "thumbnail_url": "https://example.com/thumb.jpg",
        "playlist_id": "PL-policy",
        "playlist_title": "Policy playlist",
    }


def test_materialization_rejects_duplicate_or_unknown_selection(store):
    ready = _seed_preflight(store)

    with pytest.raises(ValueError, match="duplicate occurrence_id"):
        store.create_materialization(
            "1",
            preflight_id=ready.preflight_id,
            occurrence_ids=["occ-1", "occ-1"],
        )
    with pytest.raises(ValueError, match="selected occurrence_id"):
        store.create_materialization(
            "1",
            preflight_id=ready.preflight_id,
            occurrence_ids=["occ-missing"],
        )


def test_materialization_uses_snapshot_order_not_selection_order(store):
    ready = _seed_preflight(store, item_count=3)
    materialized = store.create_materialization(
        "1",
        preflight_id=ready.preflight_id,
        occurrence_ids=["occ-3", "occ-1", "occ-2"],
    )

    items = list(store.list_materialization_items("1", materialized.id))

    assert [(item.ordinal, item.occurrence_id) for item in items] == [
        (1, "occ-1"),
        (2, "occ-2"),
        (3, "occ-3"),
    ]


def test_ready_snapshot_order_is_immutable(store):
    ready = _seed_preflight(store)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    with pytest.raises(PlaylistIngestConflictError, match="immutable"):
        store.replace_preflight_snapshot(
            "1",
            ready.preflight_id,
            status="ready",
            items=[_preflight_item("occ-new", 1)],
        )


def test_cursor_is_signed_and_bound_to_owner_resource_and_order(store):
    ready = _seed_preflight(store, item_count=3)
    first_page = store.list_preflight_items("1", ready.preflight_id, limit=1)
    assert first_page.next_cursor

    tampered = first_page.next_cursor[:-1] + ("A" if first_page.next_cursor[-1] != "A" else "B")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    attempts = [
        ("1", ready.preflight_id, tampered),
        ("2", ready.preflight_id, first_page.next_cursor),
        ("1", "missing", first_page.next_cursor),
    ]
    for owner_id, preflight_id, cursor in attempts:
        with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
            store.list_preflight_items(owner_id, preflight_id, limit=1, cursor=cursor)


@pytest.mark.parametrize("cursor", ["!.*", "x" * 4097])
def test_malformed_or_oversized_cursor_fails_closed(store, cursor):
    ready = _seed_preflight(store)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
        store.list_preflight_items("1", ready.preflight_id, limit=1, cursor=cursor)


def test_cursor_rejects_noncanonical_base64_and_wrong_signature_length(store):
    ready = _seed_preflight(store, item_count=2)
    cursor = store.list_preflight_items("1", ready.preflight_id, limit=1).next_cursor
    assert cursor is not None
    payload_segment, signature_segment = cursor.split(".")

    def decoded(segment: str) -> bytes:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))

    signature_bytes = decoded(signature_segment)
    alias = next(
        signature_segment[:-1] + character
        for character in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        if character != signature_segment[-1] and decoded(signature_segment[:-1] + character) == signature_bytes
    )
    short_signature = base64.urlsafe_b64encode(signature_bytes[:-1]).decode("ascii").rstrip("=")
    invalid = [
        f"{payload_segment}=.{signature_segment}",
        f"{payload_segment}.{signature_segment}=",
        f"{payload_segment}.{alias}",
        f"{payload_segment}.{short_signature}",
    ]

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    for candidate in invalid:
        with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
            store.list_preflight_items("1", ready.preflight_id, limit=1, cursor=candidate)


def test_run_creation_copies_materialized_items_in_one_manifest(store):
    materialized = _seed_materialization(store)

    run = store.create_run(
        "1",
        materialization_ids=[materialized.id],
        processing_options={"chunk_method": "semantic"},
        expires_at=NOW + timedelta(days=7),
    )
    items = list(store.list_run_items("1", run.run_id))

    assert run.version == 1
    assert [item.occurrence_id for item in items] == ["occ-1", "occ-2"]
    assert all(item.state == "staged" for item in items)


def test_event_replay_and_version_bump_are_atomic(store):
    materialized = _seed_materialization(store, item_count=1)
    run = store.create_run("1", materialization_ids=[materialized.id])

    event = store.append_run_event(
        "1",
        run.run_id,
        event_type="item_running",
        occurrence_id="occ-1",
        state="running",
        attrs={"worker": "w1"},
        expected_version=1,
    )

    replay = list(store.list_run_events("1", run.run_id, after_event_id=0))
    assert [item.event_id for item in replay] == [event.event_id]
    assert replay[0].attrs == {"worker": "w1"}
    assert store.get_run("1", run.run_id).version == 2

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    with pytest.raises(PlaylistIngestConflictError, match="version"):
        store.append_run_event(
            "1",
            run.run_id,
            event_type="stale",
            expected_version=1,
        )
    assert len(store.list_run_events("1", run.run_id)) == 1


def test_expired_resource_reads_pages_events_and_old_cursor_fail_closed(store):
    ready = _seed_preflight(store, item_count=2)
    cursor = store.list_preflight_items("1", ready.preflight_id, limit=1).next_cursor
    materialized = store.create_materialization(
        "1",
        preflight_id=ready.preflight_id,
        occurrence_ids=["occ-1", "occ-2"],
        expires_at=NOW + timedelta(hours=1),
    )
    run = store.create_run(
        "1",
        materialization_ids=[materialized.id],
        expires_at=NOW + timedelta(hours=1),
    )
    store.append_run_event("1", run.run_id, event_type="created", expected_version=1)
    store.test_clock.advance(timedelta(hours=2))

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    operations = [
        lambda: store.get_preflight("1", ready.preflight_id),
        lambda: store.list_preflight_items("1", ready.preflight_id),
        lambda: store.list_preflight_items("1", ready.preflight_id, limit=1, cursor=cursor),
        lambda: store.get_materialization("1", materialized.id),
        lambda: store.list_materialization_items("1", materialized.id),
        lambda: store.get_run("1", run.run_id),
        lambda: store.list_run_items("1", run.run_id),
        lambda: store.list_run_events("1", run.run_id),
    ]
    for operation in operations:
        with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
            operation()


def test_expired_resources_reject_snapshot_event_and_cas_mutations(store):
    pending = store.create_preflight(
        "1",
        source_url="https://example.com/pending",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    materialized = _seed_materialization(store, item_count=1)
    run = store.create_run(
        "1",
        materialization_ids=[materialized.id],
        expires_at=NOW + timedelta(hours=1),
    )
    store.test_clock.advance(timedelta(hours=2))

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    operations = [
        lambda: store.replace_preflight_snapshot(
            "1",
            pending.preflight_id,
            status="ready",
            items=[_preflight_item("late-occ", 1)],
        ),
        lambda: store.append_run_event("1", run.run_id, event_type="late", expected_version=1),
        lambda: store.compare_and_set_run_item_state(
            "1",
            run.run_id,
            "occ-1",
            expected_state="staged",
            new_state="running",
        ),
    ]
    for operation in operations:
        with pytest.raises(PlaylistIngestNotFoundError, match="playlist resource not found"):
            operation()


def test_create_preflight_rejects_nonfuture_expiry(store):
    with pytest.raises(ValueError, match="expires_at must be in the future"):
        store.create_preflight(
            "1",
            source_url="https://example.com/expired-preflight",
            source_kind="playlist",
            expires_at=NOW,
        )


def test_create_materialization_rejects_nonfuture_expiry(store):
    ready = _seed_preflight(store, item_count=1)

    with pytest.raises(ValueError, match="expires_at must be in the future"):
        store.create_materialization(
            "1",
            preflight_id=ready.preflight_id,
            occurrence_ids=["occ-1"],
            expires_at=NOW,
        )


def test_create_run_rejects_nonfuture_expiry(store):
    materialized = _seed_materialization(store, item_count=1)

    with pytest.raises(ValueError, match="expires_at must be in the future"):
        store.create_run(
            "1",
            materialization_ids=[materialized.id],
            expires_at=NOW,
        )


def test_compare_and_set_run_item_transition_rejects_stale_state(store):
    materialized = _seed_materialization(store, item_count=1)
    run = store.create_run("1", materialization_ids=[materialized.id])

    assert store.compare_and_set_run_item_state(
        "1",
        run.run_id,
        "occ-1",
        expected_state="staged",
        new_state="running",
    )
    assert not store.compare_and_set_run_item_state(
        "1",
        run.run_id,
        "occ-1",
        expected_state="staged",
        new_state="queued",
    )
    assert store.list_run_items("1", run.run_id)[0].state == "running"


def test_postgres_snapshot_replacement_locks_mutable_unexpired_parent(store, monkeypatch):
    queries: list[str] = []

    class _Result:
        rowcount = 1

        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    @contextmanager
    def fake_connection(*, write):
        assert write is True
        yield object()

    def fake_query(_db, sql, _params=()):
        queries.append(sql)
        if "SELECT status" in sql:
            return _Result({"status": "pending", "expires_at": NOW + timedelta(hours=1)})
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_preflight", lambda *_args: object())

    store.replace_preflight_snapshot(
        "1",
        "preflight-id",
        status="ready",
        items=[],
    )

    locked_read = next(query for query in queries if "SELECT status" in query)
    assert "expires_at" in locked_read
    assert locked_read.rstrip().endswith("FOR UPDATE")


def test_postgres_preflight_admission_uses_transaction_advisory_lock_before_capacity_counts(
    store,
    monkeypatch,
):
    queries: list[tuple[str, tuple]] = []

    class _Result:
        rowcount = 1

        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    @contextmanager
    def fake_connection(*, write):
        assert write is True
        yield object()

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        if "COUNT(*) AS active_count" in sql:
            return _Result({"active_count": 0})
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_preflight", lambda *_args: object())

    store.reserve_preflight(
        "owner-1",
        source_url="https://www.youtube.com/playlist?list=PLlock",
        source_kind="youtube_playlist",
        playlist_id="PLlock",
        expires_at=NOW + timedelta(hours=1),
        global_capacity=2,
        owner_capacity=1,
    )

    advisory_index = next(index for index, (sql, _params) in enumerate(queries) if "pg_advisory_xact_lock" in sql)
    count_indexes = [index for index, (sql, _params) in enumerate(queries) if "COUNT(*) AS active_count" in sql]
    insert_index = next(
        index for index, (sql, _params) in enumerate(queries) if "INSERT INTO playlist_preflights" in sql
    )
    assert len(count_indexes) == 2
    assert advisory_index < min(count_indexes) < insert_index
    assert queries[advisory_index][1] == (store._jobs._pg_advisory_key("playlist_preflight_admission"),)


def test_postgres_ready_guard_locks_preflight_then_exact_active_lease(store, monkeypatch):
    queries: list[tuple[str, tuple]] = []

    class _Result:
        rowcount = 1

        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    @contextmanager
    def fake_connection(*, write):
        assert write is True
        yield object()

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        if "SELECT status" in sql:
            return _Result({"status": "running", "expires_at": NOW + timedelta(hours=1), "job_id": 42})
        if "SELECT owner_user_id" in sql:
            return _Result(
                {
                    "owner_user_id": "1",
                    "status": "processing",
                    "lease_id": "lease-42",
                    "worker_id": "worker-42",
                    "lease_active": True,
                }
            )
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_preflight", lambda *_args: object())

    store.replace_preflight_snapshot(
        "1",
        "preflight-id",
        status="ready",
        items=[],
        expected_job_id=42,
        expected_lease_id="lease-42",
        expected_worker_id="worker-42",
    )

    preflight_lock = next(index for index, (query, _) in enumerate(queries) if "SELECT status" in query)
    job_lock = next(index for index, (query, _) in enumerate(queries) if "SELECT owner_user_id" in query)
    assert preflight_lock < job_lock
    assert queries[preflight_lock][0].rstrip().endswith("FOR UPDATE")
    assert "lease_id" in queries[job_lock][0]
    assert "worker_id" in queries[job_lock][0]
    assert "leased_until" in queries[job_lock][0]
    assert "WHERE id = ?" in queries[job_lock][0]
    assert queries[job_lock][0].rstrip().endswith("FOR UPDATE")
    assert queries[job_lock][1] == (42,)


def test_postgres_cleanup_locks_all_expired_parents_before_child_deletes(store, monkeypatch):
    queries: list[tuple[str, tuple]] = []

    class _Result:
        rowcount = 1

        def __init__(self, rows=()):
            self.rows = rows

        def fetchall(self):
            return list(self.rows)

    @contextmanager
    def fake_connection(*, write):
        assert write is True
        yield object()

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        if "SELECT preflight_id" in sql:
            return _Result([{"preflight_id": "pf-1"}])
        if "SELECT materialization_id" in sql:
            return _Result([{"materialization_id": "mat-1"}])
        if "SELECT run_id" in sql:
            return _Result([{"run_id": "run-1"}])
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)

    store.cleanup_expired("owner-1", now=NOW)

    locked = [index for index, (sql, _params) in enumerate(queries) if "FOR UPDATE" in sql]
    first_delete = next(index for index, (sql, _params) in enumerate(queries) if "DELETE FROM" in sql)
    assert len(locked) == 3
    assert max(locked) < first_delete
    assert all("owner-1" not in sql for sql, _params in queries)


def test_postgres_event_append_locks_run_before_versioning(store, monkeypatch):
    queries: list[str] = []

    class _Result:
        rowcount = 1
        lastrowid = 1

        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    @contextmanager
    def fake_connection(*, write):
        assert write is True
        yield object()

    def fake_query(_db, sql, _params=()):
        queries.append(sql)
        if "SELECT version" in sql:
            return _Result({"version": 1})
        if "RETURNING event_id" in sql:
            return _Result({"event_id": 1})
        if "SELECT * FROM media_ingest_run_events" in sql:
            return _Result({"event_id": 1})
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "_event_record", lambda row: row)

    store.append_run_event("owner-1", "run-1", event_type="progress")

    locked_read = next(query for query in queries if "SELECT version" in query)
    assert locked_read.rstrip().endswith("FOR UPDATE")


def test_no_cas_event_appends_both_persist_with_monotonic_versions(store):
    materialized = _seed_materialization(store, item_count=1)
    run = store.create_run("1", materialization_ids=[materialized.id])

    def append(index: int):
        return store.append_run_event("1", run.run_id, event_type=f"event-{index}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        events = list(pool.map(append, range(2)))

    replay = store.list_run_events("1", run.run_id)
    assert len(events) == 2
    assert [event.event_id for event in replay] == sorted(event.event_id for event in events)
    assert store.get_run("1", run.run_id).version == 3


def test_cleanup_expired_is_owner_safe(store):
    materialized = _seed_materialization(store, item_count=1)
    run = store.create_run(
        "1",
        materialization_ids=[materialized.id],
        expires_at=NOW + timedelta(hours=1),
    )
    store.append_run_event("1", run.run_id, event_type="before-expiry")
    expired = store.create_preflight(
        "1",
        source_url="https://example.com/expired",
        source_kind="url",
        expires_at=NOW + timedelta(hours=1),
    )
    store.create_preflight(
        "2",
        source_url="https://example.com/other",
        source_kind="url",
        expires_at=NOW + timedelta(hours=1),
    )
    store.test_clock.advance(timedelta(hours=2))

    deleted = store.cleanup_expired("1", now=store.test_clock.now_utc())

    assert deleted == {"preflights": 2, "materializations": 1, "runs": 1}
    connection = store._jobs._connect()
    try:
        for table in (
            "playlist_preflight_items",
            "playlist_materialization_items",
            "media_ingest_run_items",
            "media_ingest_run_events",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # nosec B608
    finally:
        connection.close()
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    with pytest.raises(PlaylistIngestNotFoundError):
        store.get_preflight("1", expired.preflight_id)
    assert store.cleanup_expired("2", now=store.test_clock.now_utc())["preflights"] == 1
