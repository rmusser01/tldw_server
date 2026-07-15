import base64
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

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


def _validated_direct_record(**overrides) -> dict:
    record = {
        "occurrence_id": "occ-direct",
        "input_kind": "direct_url",
        "materialization_id": None,
        "source_url": "https://example.com/video",
        "normalized_source_id": "url:https://example.com/video",
        "source_kind": "generic_url",
        "display_metadata": {"title": "Video"},
        "state": "staged",
        "action": "ingest",
        "metadata_patch": None,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("postgres", "shredder"),
    [(False, "json_each(?)"), (True, "jsonb_to_recordset(?::jsonb)")],
)
def test_bulk_materialization_resolution_uses_one_fixed_shape_collection_bind(
    store,
    monkeypatch,
    postgres,
    shredder,
):
    pairs = [(f"mat-{index}", f"occ-{index}") for index in range(500)]
    calls = []

    class _Rows:
        def fetchall(self):
            return [
                {
                    "request_ordinal": index + 1,
                    "materialization_id": materialization_id,
                    "occurrence_id": occurrence_id,
                    "source_url": f"https://example.com/{index}",
                    "normalized_source_id": f"url:https://example.com/{index}",
                    "source_kind": "generic_url",
                    "display_metadata_json": {"title": str(index)} if postgres else json.dumps({"title": str(index)}),
                }
                for index, (materialization_id, occurrence_id) in enumerate(pairs)
            ]

    @contextmanager
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "1"
        assert write is False
        yield object()

    def fake_query(_db, sql, params=()):
        calls.append((sql, params))
        return _Rows()

    store._postgres = postgres
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)

    resolved = store.resolve_materialization_occurrences("1", pairs)

    assert len(resolved) == 500
    assert len(calls) == 1
    sql, params = calls[0]
    assert shredder in sql
    if postgres:
        assert "FROM ROWS FROM (" in sql
        assert ") WITH ORDINALITY AS requested(" in sql
    assert len(params) == 3
    assert len(json.loads(params[0])) == 500
    assert sql.count("?") == 3


def test_postgres_authority_revalidation_locks_parent_and_item_rows(store, monkeypatch):
    calls = []

    class _Rows:
        def fetchall(self):
            return [
                {
                    "request_ordinal": 1,
                    "materialization_id": "mat-1",
                    "occurrence_id": "occ-1",
                    "source_url": "https://example.com/1",
                    "normalized_source_id": "url:https://example.com/1",
                    "source_kind": "generic_url",
                    "display_metadata_json": {"title": "One"},
                }
            ]

    def fake_query(_db, sql, params=()):
        calls.append((sql, params))
        return _Rows()

    store._postgres = True
    monkeypatch.setattr(store, "_query", fake_query)

    resolved = store._resolve_materialization_occurrences_in_connection(
        object(),
        "1",
        [("mat-1", "occ-1")],
        now=NOW,
        lock=True,
    )

    assert len(resolved) == 1
    assert len(calls) == 1
    assert "FOR SHARE OF m, mi" in calls[0][0]
    assert len(calls[0][1]) == 3


@pytest.mark.parametrize(
    ("items", "processing_options"),
    [
        ([_validated_direct_record(display_metadata={"score": float("nan")})], None),
        ([_validated_direct_record(display_metadata={"opaque": object()})], None),
        ([_validated_direct_record()], {"nested": {"opaque": object()}}),
    ],
)
def test_validated_manifest_rejects_internal_json_before_write_lock(
    store,
    monkeypatch,
    items,
    processing_options,
):
    entered = False

    @contextmanager
    def fail_if_entered(*, owner_user_id, write):
        nonlocal entered
        entered = True
        raise AssertionError(f"connection entered for owner={owner_user_id} with write={write}")
        yield

    monkeypatch.setattr(store, "_connection", fail_if_entered)

    with pytest.raises(ValueError):
        store.create_validated_run(
            "1",
            items=items,
            processing_options=processing_options,
        )

    assert entered is False


def test_idempotent_validated_run_insert_has_one_concurrent_initializer_and_manifest(store):
    def create(token: str):
        return store.create_validated_run(
            "owner-1",
            items=[_validated_direct_record(occurrence_id="occ-idempotent")],
            client_request_id="concurrent-create",
            request_fingerprint="f" * 64,
            initialization_token=token,
            initialization_lease_seconds=30,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(pool.map(create, ("initializer-a", "initializer-b")))

    assert created[0].run_id == created[1].run_id
    assert {record.initialization_token for record in created} <= {"initializer-a", "initializer-b"}
    assert len({record.initialization_token for record in created}) == 1
    assert [item.occurrence_id for item in store.list_run_items("owner-1", created[0].run_id)] == [
        "occ-idempotent"
    ]
    assert len(store.list_run_events("owner-1", created[0].run_id)) == 1


def test_run_initialization_lease_claim_and_renew_are_owner_run_token_cas(store):
    created = store.create_validated_run(
        "owner-1",
        items=[_validated_direct_record(occurrence_id="occ-lease-init")],
        client_request_id="abandoned-lease",
        request_fingerprint="a" * 64,
        initialization_token="initializer-old",
        initialization_lease_seconds=5,
    )

    live = store.claim_run_initialization(
        "owner-1",
        client_request_id="abandoned-lease",
        request_fingerprint="a" * 64,
        initialization_token="initializer-new",
        initialization_lease_seconds=5,
    )
    assert live.initialization_token == "initializer-old"

    store.test_clock.advance(timedelta(seconds=6))
    claimed = store.claim_run_initialization(
        "owner-1",
        client_request_id="abandoned-lease",
        request_fingerprint="a" * 64,
        initialization_token="initializer-new",
        initialization_lease_seconds=5,
    )
    assert claimed.run_id == created.run_id
    assert claimed.initialization_token == "initializer-new"
    assert store.renew_run_initialization(
        "owner-2",
        created.run_id,
        initialization_token="initializer-new",
        initialization_lease_seconds=5,
    ) is False
    assert store.renew_run_initialization(
        "owner-1",
        created.run_id,
        initialization_token="initializer-old",
        initialization_lease_seconds=5,
    ) is False
    assert store.renew_run_initialization(
        "owner-1",
        created.run_id,
        initialization_token="initializer-new",
        initialization_lease_seconds=5,
    ) is True


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


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("owner_user_id", "wrong-owner"),
        ("domain", "wrong-domain"),
        ("queue", "wrong-queue"),
        ("job_type", "wrong-type"),
        ("status", "processing"),
        ("available_at", "published"),
        ("available_at", "ordinary-schedule"),
        ("payload_preflight_id", "wrong-preflight"),
        ("payload_max_items", 21),
        ("payload_timeout_seconds", 11),
    ],
)
def test_preflight_bind_rejects_every_mismatched_job_attribute_without_publication(
    tmp_path,
    monkeypatch,
    mutation,
    value,
):
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / f"bind-{mutation}-{value}.db")
    playlist_store = PlaylistIngestStore(manager)
    owner = "bind-owner"
    preflight = playlist_store.create_preflight(
        owner,
        source_url="https://www.youtube.com/playlist?list=PLbind",
        source_kind="youtube_playlist",
        playlist_id="PLbind",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    payload = {
        "preflight_id": preflight.preflight_id,
        "max_items": 20,
        "timeout_seconds": 10,
    }
    sentinel = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload=payload,
        owner_user_id=owner,
        priority=5,
        max_retries=0,
        available_at=sentinel,
    )

    with manager._connect() as connection:
        if mutation.startswith("payload_"):
            changed_payload = dict(payload)
            changed_payload[mutation.removeprefix("payload_")] = value
            connection.execute(
                "UPDATE jobs SET payload = ? WHERE id = ?",
                (json.dumps(changed_payload), int(job["id"])),
            )
        elif mutation == "available_at" and value == "published":
            connection.execute("UPDATE jobs SET available_at = DATETIME('now') WHERE id = ?", (int(job["id"]),))
        elif mutation == "available_at":
            connection.execute(
                "UPDATE jobs SET available_at = DATETIME('now', '+1 day') WHERE id = ?",
                (int(job["id"]),),
            )
        else:
            connection.execute(f"UPDATE jobs SET {mutation} = ? WHERE id = ?", (value, int(job["id"])))  # nosec B608
        connection.commit()
    before = manager.get_job(int(job["id"]))

    with pytest.raises(PlaylistIngestConflictError, match="preflight job is unavailable"):
        playlist_store.bind_preflight_job(
            owner,
            preflight.preflight_id,
            int(job["id"]),
            expected_queue="default",
            expected_payload=payload,
        )

    assert playlist_store.get_preflight(owner, preflight.preflight_id).job_id is None
    after = manager.get_job(int(job["id"]))
    assert after["status"] == before["status"]
    assert after["available_at"] == before["available_at"]


def test_preflight_bind_decrypts_and_matches_the_exact_expected_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("JOBS_ENCRYPT_MEDIA_INGEST", "true")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "encrypted-bind.db")
    playlist_store = PlaylistIngestStore(manager)
    preflight = playlist_store.create_preflight(
        "encrypted-owner",
        source_url="https://www.youtube.com/playlist?list=PLencrypted",
        source_kind="youtube_playlist",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    payload = {"preflight_id": preflight.preflight_id, "max_items": 20, "timeout_seconds": 10}
    job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="playlist_preflight",
        payload=payload,
        owner_user_id="encrypted-owner",
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )

    bound = playlist_store.bind_preflight_job(
        "encrypted-owner",
        preflight.preflight_id,
        int(job["id"]),
        expected_queue="default",
        expected_payload=payload,
    )

    assert bound.job_id == int(job["id"])


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
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "1"
        assert write is True
        yield object()

    def fake_query(_db, sql, _params=()):
        queries.append(sql)
        if "SELECT status" in sql:
            return _Result({"status": "pending", "expires_at": NOW + timedelta(hours=1)})
        if "SELECT * FROM playlist_preflights" in sql:
            return _Result(
                {
                    "preflight_id": "preflight-id",
                    "owner_user_id": "1",
                    "status": "ready",
                    "source_url": "https://www.youtube.com/playlist?list=PLtest",
                    "source_kind": "youtube_playlist",
                    "created_at": NOW,
                    "updated_at": NOW,
                    "expires_at": NOW + timedelta(hours=1),
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
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "owner-1"
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
        ttl_seconds=3600,
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
    assert all("NOW()" in queries[index][0] for index in count_indexes)
    assert all(len(queries[index][1]) <= 1 for index in count_indexes)


def test_postgres_preflight_bind_uses_database_now_and_exact_publication_constraints(store, monkeypatch):
    queries: list[tuple[str, tuple]] = []
    payload = {"preflight_id": "preflight-id", "max_items": 20, "timeout_seconds": 10}

    class _Result:
        rowcount = 1

        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    @contextmanager
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "owner-1"
        assert write is True
        yield object()

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        if "SELECT status, job_id" in sql:
            return _Result({"status": "pending", "job_id": None})
        if "SELECT id, owner_user_id" in sql:
            return _Result(
                {
                    "id": 42,
                    "owner_user_id": "owner-1",
                    "domain": "media_ingest",
                    "queue": "default",
                    "job_type": "playlist_preflight",
                    "status": "queued",
                    "available_at": datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                    "payload": payload,
                }
            )
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_preflight", lambda *_args: object())

    store.bind_preflight_job(
        "owner-1",
        "preflight-id",
        42,
        expected_queue="default",
        expected_payload=payload,
    )

    preflight_lock = next(index for index, (sql, _) in enumerate(queries) if "SELECT status, job_id" in sql)
    job_lock = next(index for index, (sql, _) in enumerate(queries) if "SELECT id, owner_user_id" in sql)
    publication = next((sql, params) for sql, params in queries if "UPDATE jobs SET available_at" in sql)
    assert preflight_lock < job_lock
    assert "expires_at > NOW()" in queries[preflight_lock][0]
    assert "available_at = NOW()" in publication[0]
    assert "updated_at = NOW()" in publication[0]
    for constraint in ("owner_user_id", "domain", "queue", "job_type", "status", "available_at"):
        assert constraint in publication[0]


def test_postgres_ready_guard_locks_preflight_then_exact_active_lease(store, monkeypatch):
    queries: list[tuple[str, tuple]] = []

    class _Result:
        rowcount = 1

        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    @contextmanager
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "1"
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
        if "SELECT * FROM playlist_preflights" in sql:
            return _Result(
                {
                    "preflight_id": "preflight-id",
                    "owner_user_id": "1",
                    "status": "ready",
                    "source_url": "https://www.youtube.com/playlist?list=PLtest",
                    "source_kind": "youtube_playlist",
                    "job_id": 42,
                    "created_at": NOW,
                    "updated_at": NOW,
                    "expires_at": NOW + timedelta(hours=1),
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
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "owner-1"
        assert write is True
        yield object()

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        if "UNION ALL" in sql:
            return _Result(
                [
                    {"resource_type": "preflight", "resource_id": "pf-1"},
                    {"resource_type": "materialization", "resource_id": "mat-1"},
                    {"resource_type": "run", "resource_id": "run-1"},
                ]
            )
        if "SELECT preflight_id AS resource_id" in sql:
            return _Result([{"resource_id": "pf-1"}])
        if "SELECT materialization_id AS resource_id" in sql:
            return _Result([{"resource_id": "mat-1"}])
        if "SELECT run_id AS resource_id" in sql:
            return _Result([{"resource_id": "run-1"}])
        if "SELECT run_id FROM media_ingest_runs" in sql:
            return _Result([{"run_id": "run-1"}])
        return _Result()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)

    store.cleanup_expired("owner-1", now=NOW)

    parent_locks = [
        index for index, (sql, _params) in enumerate(queries) if "AS resource_id" in sql and "FOR UPDATE" in sql
    ]
    first_delete = next(index for index, (sql, _params) in enumerate(queries) if "DELETE FROM" in sql)
    assert len(parent_locks) == 6
    assert max(parent_locks) < first_delete
    assert any("FROM media_ingest_run_items" in sql and "FOR UPDATE" in sql for sql, _params in queries)
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
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "owner-1"
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


def test_postgres_attach_collection_plan_locks_run_and_items_before_updates(store, monkeypatch):
    queries: list[str] = []

    class _Result:
        rowcount = 1

        def __init__(self, rows=()):
            self.rows = list(rows)

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            return self.rows

    @contextmanager
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "owner-1"
        assert write is True
        yield object()

    def fake_query(_db, sql, _params=()):
        queries.append(sql)
        if "SELECT version" in sql:
            return _Result([{"version": 1}])
        if "SELECT occurrence_id, duplicate_policy" in sql:
            return _Result([{"occurrence_id": "occ-one", "duplicate_policy": "overwrite"}])
        if "SELECT * FROM media_ingest_runs" in sql:
            return _Result([{"run_id": "run-1"}])
        return _Result()

    sentinel = object()
    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "_run_record", lambda _row: sentinel)

    attached = store.attach_collection_plan(
        "owner-1",
        "run-1",
        collection_id=55,
        planned_item_ids={"occ-one": 101},
    )

    run_lock = next(index for index, query in enumerate(queries) if "SELECT version" in query)
    items_lock = next(index for index, query in enumerate(queries) if "SELECT occurrence_id" in query)
    first_update = next(index for index, query in enumerate(queries) if "UPDATE media_ingest" in query)
    assert attached is sentinel
    assert run_lock < items_lock < first_update
    assert queries[run_lock].rstrip().endswith("FOR UPDATE")
    assert queries[items_lock].rstrip().endswith("FOR UPDATE")


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


def test_cleanup_expired_resources_is_bounded_across_parent_and_event_rows(store):
    for index in range(2):
        occurrence_id = f"cleanup-occ-{index}"
        preflight = store.create_preflight(
            "1",
            source_url=f"https://www.youtube.com/playlist?list=PLcleanup{index}",
            source_kind="youtube_playlist",
            expires_at=NOW + timedelta(hours=1),
        )
        store.replace_preflight_snapshot(
            "1",
            preflight.preflight_id,
            status="ready",
            items=[
                {
                    **_preflight_item(occurrence_id, index + 1),
                    "normalized_source_id": f"youtube:cleanup:{index}",
                }
            ],
        )
        materialized = store.create_materialization(
            "1",
            preflight_id=preflight.preflight_id,
            occurrence_ids=[occurrence_id],
            expires_at=NOW + timedelta(hours=1),
        )
        run = store.create_run(
            "1",
            materialization_ids=[materialized.id],
            expires_at=NOW + timedelta(hours=1),
        )
        store.append_run_event("1", run.run_id, event_type="before-expiry")
    store.test_clock.advance(timedelta(hours=2))

    deleted = store.cleanup_expired_resources(
        "1",
        now=store.test_clock.now_utc(),
        limit=1,
    )

    assert sum(deleted.values()) == 1
    connection = store._jobs._connect()
    try:
        remaining_parents = sum(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # nosec B608
            for table in (
                "playlist_preflights",
                "playlist_materializations",
                "media_ingest_runs",
            )
        )
        assert remaining_parents == 5
        orphan_events = connection.execute(
            """
            SELECT COUNT(*)
            FROM media_ingest_run_events AS event
            LEFT JOIN media_ingest_runs AS run
              ON run.owner_user_id = event.owner_user_id AND run.run_id = event.run_id
            WHERE event.owner_user_id = '1' AND run.run_id IS NULL
            """
        ).fetchone()[0]
        assert orphan_events == 0
    finally:
        connection.close()

    second = store.cleanup_expired_resources(
        "1",
        now=store.test_clock.now_utc(),
        limit=1,
    )
    assert sum(second.values()) == 1


def test_cleanup_global_budget_selects_oldest_parent_across_resource_types(store):
    occurrence_id = "cleanup-global-budget"
    preflight = store.create_preflight(
        "1",
        source_url="https://www.youtube.com/playlist?list=PLglobalbudget",
        source_kind="youtube_playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "1",
        preflight.preflight_id,
        status="ready",
        items=[_preflight_item(occurrence_id, 1)],
    )
    materialized = store.create_materialization(
        "1",
        preflight_id=preflight.preflight_id,
        occurrence_ids=[occurrence_id],
        expires_at=NOW + timedelta(hours=1),
    )
    run = store.create_run(
        "1",
        materialization_ids=[materialized.id],
        expires_at=NOW + timedelta(hours=1),
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE playlist_preflights SET expires_at = ? WHERE preflight_id = ?",
            (store._db_datetime(NOW - timedelta(hours=1)), preflight.preflight_id),
        )
        store._query(
            db,
            "UPDATE playlist_materializations SET expires_at = ? WHERE materialization_id = ?",
            (store._db_datetime(NOW - timedelta(hours=3)), materialized.id),
        )
        store._query(
            db,
            "UPDATE media_ingest_runs SET expires_at = ? WHERE run_id = ?",
            (store._db_datetime(NOW - timedelta(hours=2)), run.run_id),
        )

    first = store.cleanup_expired_resources("1", now=NOW, limit=1)

    assert first == {"preflights": 0, "materializations": 1, "runs": 0}
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM playlist_preflights WHERE preflight_id = ?",
                (preflight.preflight_id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()[0]
            == 1
        )

    second = store.cleanup_expired_resources("1", now=NOW, limit=1)

    assert second == {"preflights": 0, "materializations": 0, "runs": 1}


@pytest.mark.parametrize(
    ("later_resource_type", "expected"),
    [
        ("preflight", {"preflights": 1, "materializations": 0, "runs": 0}),
        ("materialization", {"preflights": 0, "materializations": 1, "runs": 0}),
        ("run", {"preflights": 0, "materializations": 0, "runs": 1}),
    ],
)
def test_cleanup_global_budget_backfills_after_blocked_oldest_run(
    store,
    later_resource_type,
    expected,
):
    blocked, reserved, job = _seed_held_run_job(store, occurrence_id="occ-backfill-blocked")
    store.bind_run_item_job(
        "1",
        blocked.run_id,
        "occ-backfill-blocked",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    if later_resource_type == "preflight":
        later_id = store.create_preflight(
            "1",
            source_url="https://example.com/backfill-preflight",
            source_kind="url",
            expires_at=NOW + timedelta(hours=1),
        ).preflight_id
        table, id_column = "playlist_preflights", "preflight_id"
    elif later_resource_type == "materialization":
        later_id = _seed_materialization(store, item_count=1).id
        table, id_column = "playlist_materializations", "materialization_id"
    else:
        later_id = store.create_validated_run(
            "1",
            items=[_validated_direct_record(occurrence_id="occ-backfill-later-run")],
        ).run_id
        table, id_column = "media_ingest_runs", "run_id"
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE media_ingest_runs SET expires_at = ? WHERE run_id = ?",
            (store._db_datetime(NOW - timedelta(hours=3)), blocked.run_id),
        )
        store._query(
            db,
            f"UPDATE {table} SET expires_at = ? WHERE {id_column} = ?",  # nosec B608
            (store._db_datetime(NOW - timedelta(hours=2)), later_id),
        )

    deleted = store.cleanup_expired_resources("1", now=NOW, limit=1)

    assert deleted == expected
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                (blocked.run_id,),
            ).fetchone()[0]
            == 1
        )
    assert store._jobs.get_job(int(job["id"]))["status"] == "queued"


def test_cleanup_global_budget_backfills_after_retry_retained_staging(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    blocked, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-backfill-staging-blocked",
    )
    later = store.create_preflight(
        "1",
        source_url="https://example.com/backfill-after-staging",
        source_kind="url",
        expires_at=NOW + timedelta(hours=1),
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE media_ingest_runs SET expires_at = ? WHERE run_id = ?",
            (store._db_datetime(NOW - timedelta(hours=3)), blocked.run_id),
        )
        store._query(
            db,
            "UPDATE playlist_preflights SET expires_at = ? WHERE preflight_id = ?",
            (store._db_datetime(NOW - timedelta(hours=2)), later.preflight_id),
        )
    monkeypatch.setattr(
        playlist_ingest_store,
        "cleanup_exact_run_file_staging",
        lambda **_kwargs: "failed",
    )

    deleted = store.cleanup_expired_resources("1", now=NOW, limit=1)

    assert deleted == {"preflights": 1, "materializations": 0, "runs": 0}
    _assert_retained_run_staging_authority(store, blocked.run_id, staging_dir)
    assert staging_dir.exists()
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"


def test_cleanup_global_budget_backfills_after_partial_staging_batch(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(playlist_ingest_store, "_STAGING_CLEANUP_CANDIDATE_LIMIT", 1)
    blocked, jobs, staging_dirs = _seed_file_run_jobs(
        store,
        tmp_path,
        occurrence_ids=("occ-partial-a", "occ-partial-b"),
    )
    later = store.create_preflight(
        "1",
        source_url="https://example.com/backfill-after-partial-staging",
        source_kind="url",
        expires_at=NOW + timedelta(hours=1),
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE media_ingest_runs SET expires_at = ? WHERE run_id = ?",
            (store._db_datetime(NOW - timedelta(hours=3)), blocked.run_id),
        )
        store._query(
            db,
            "UPDATE playlist_preflights SET expires_at = ? WHERE preflight_id = ?",
            (store._db_datetime(NOW - timedelta(hours=2)), later.preflight_id),
        )

    deleted = store.cleanup_expired_resources("1", now=NOW, limit=1)

    assert deleted == {"preflights": 1, "materializations": 0, "runs": 0}
    assert sum(deleted.values()) == 1
    with store._jobs._connect() as connection:
        remaining_staging = connection.execute(
            """
            SELECT COUNT(*) FROM media_ingest_run_items
            WHERE run_id = ? AND staging_temp_dir IS NOT NULL
            """,
            (blocked.run_id,),
        ).fetchone()[0]
        remaining_run = connection.execute(
            "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
            (blocked.run_id,),
        ).fetchone()[0]
    assert remaining_staging == 1
    assert remaining_run == 1
    assert sum(path.exists() for path in staging_dirs) == 1
    assert [store._jobs.get_job(int(job["id"]))["status"] for job in jobs] == [
        "cancelled",
        "cancelled",
    ]


def test_postgres_expired_parent_scan_preserves_global_order_and_locks_exact_ids(store, monkeypatch):
    queries: list[tuple[str, tuple]] = []

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        if "UNION ALL" in sql:
            return _Rows(
                [
                    {"resource_type": "run", "resource_id": "run-blocked"},
                    {"resource_type": "preflight", "resource_id": "preflight-next"},
                ]
            )
        if "FROM media_ingest_runs" in sql:
            return _Rows([{"resource_id": "run-blocked"}])
        if "FROM playlist_preflights" in sql:
            return _Rows([{"resource_id": "preflight-next"}])
        return _Rows([])

    store._postgres = True
    monkeypatch.setattr(store, "_query", fake_query)

    selected = store._select_expired_parent_ids(
        object(),
        owner_user_id="owner-1",
        cutoff=NOW,
        limit=500,
    )

    assert selected == [("run", "run-blocked"), ("preflight", "preflight-next")]
    candidate_query = next((sql, params) for sql, params in queries if "UNION ALL" in sql)
    assert candidate_query[1][-1] == 500
    lock_queries = [sql for sql, _params in queries if "FOR UPDATE" in sql]
    assert len(lock_queries) == 2
    assert all("owner_user_id = ?" in sql and "= ANY(?)" in sql for sql in lock_queries)


def test_expired_parent_scan_cap_logs_only_aggregate_and_does_not_paginate(store, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    queries: list[tuple[str, tuple]] = []

    class _Rows:
        def fetchall(self):
            return [{"resource_type": "preflight", "resource_id": f"private-parent-{index}"} for index in range(500)]

    class _LoggerStub:
        def __init__(self):
            self.bindings: list[dict] = []
            self.messages: list[str] = []

        def bind(self, **kwargs):
            self.bindings.append(dict(kwargs))
            return self

        def warning(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        return _Rows()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(playlist_ingest_store, "logger", logger_stub)

    selected = store._select_expired_parent_ids(
        object(),
        owner_user_id="private-owner",
        cutoff=NOW,
        limit=500,
    )

    assert len(selected) == 500
    assert len(queries) == 1
    assert queries[0][1][-1] == 500
    assert logger_stub.bindings == [
        {
            "error_code": "playlist_expired_parent_scan_cap_reached",
            "candidate_count": 500,
            "scan_limit": 500,
        }
    ]
    assert "private" not in repr((logger_stub.bindings, logger_stub.messages))


def test_postgres_cleanup_uses_one_bounded_oldest_first_parent_scan(store, monkeypatch):
    queries: list[tuple[str, tuple]] = []

    class _Rows:
        rowcount = 0

        def fetchall(self):
            return []

    @contextmanager
    def fake_connection(*, owner_user_id, write):
        assert owner_user_id == "owner-1"
        assert write is True
        yield object()

    def fake_query(_db, sql, params=()):
        queries.append((sql, tuple(params)))
        return _Rows()

    store._postgres = True
    monkeypatch.setattr(store, "_connection", fake_connection)
    monkeypatch.setattr(store, "_query", fake_query)

    assert store.cleanup_expired_resources("owner-1", now=NOW, limit=2) == {
        "preflights": 0,
        "materializations": 0,
        "runs": 0,
    }
    candidate_queries = [sql for sql, _params in queries if "UNION ALL" in sql]
    assert len(candidate_queries) == 1
    assert "ORDER BY expires_at, resource_type, resource_id" in candidate_queries[0]
    assert any(params[-1] == 500 for sql, params in queries if "UNION ALL" in sql)


def test_cleanup_expired_resources_cancels_exact_held_job_and_releases_scheduled_counter(
    store,
    monkeypatch,
):
    from tldw_Server_API.app.core.exceptions import JobSubmissionLimitError

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_MEDIA_INGEST_USER_1", "1")
    with store._jobs._connect() as connection:
        row = connection.execute(
            """
            SELECT scheduled_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default'
              AND job_type = 'media_ingest_item'
            """
        ).fetchone()
    prior_scheduled_count = int(row[0]) if row is not None else 0
    run, _reserved, job = _seed_held_run_job(store, occurrence_id="occ-cleanup-held")
    with store._jobs._connect() as connection:
        created_counter = connection.execute(
            """
            SELECT scheduled_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default'
              AND job_type = 'media_ingest_item'
            """
        ).fetchone()
    assert created_counter is not None
    assert int(created_counter[0]) == prior_scheduled_count + 1
    with pytest.raises(JobSubmissionLimitError, match="max queued"):
        store._jobs.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={"source": "https://example.com/quota-before-cleanup"},
            owner_user_id="1",
        )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"
    with store._jobs._connect() as connection:
        row = connection.execute(
            """
            SELECT ready_count, scheduled_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default'
              AND job_type = 'media_ingest_item'
            """
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 0
    assert int(row[1]) == prior_scheduled_count
    admitted = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={"source": "https://example.com/quota-after-cleanup"},
        owner_user_id="1",
    )
    assert admitted["status"] == "queued"


def test_cleanup_emits_cancelled_job_lifecycle_once_after_commit(store, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    run, _reserved, job = _seed_held_run_job(store, occurrence_id="occ-cleanup-lifecycle")
    gauge_calls: list[dict] = []
    metric_calls: list[dict] = []
    event_calls: list[tuple[str, dict, dict]] = []
    cascade_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(store._jobs, "_update_gauges", lambda **kwargs: gauge_calls.append(kwargs))
    monkeypatch.setattr(
        playlist_ingest_store,
        "increment_cancelled",
        lambda value: metric_calls.append(value),
        raising=False,
    )
    monkeypatch.setattr(
        playlist_ingest_store,
        "emit_job_event",
        lambda event, *, job, attrs: event_calls.append((event, job, attrs)),
        raising=False,
    )
    monkeypatch.setattr(
        store._jobs,
        "_cancel_dependent_jobs",
        lambda job_uuid, *, reason: cascade_calls.append((job_uuid, reason)),
    )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    expected_job = {
        "id": int(job["id"]),
        "uuid": job["uuid"],
        "owner_user_id": "1",
        "domain": "media_ingest",
        "queue": "default",
        "job_type": "media_ingest_item",
    }
    assert gauge_calls == [{"domain": "media_ingest", "queue": "default", "job_type": "media_ingest_item"}]
    assert metric_calls == [expected_job]
    assert event_calls == [
        (
            "job.cancelled",
            expected_job,
            {"reason": "expired_playlist_ingest_run", "terminal": True},
        )
    ]
    assert cascade_calls == [(job["uuid"], "expired_playlist_ingest_run")]
    with store._jobs._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?", (run.run_id,)).fetchone()[0]
            == 0
        )


def test_cleanup_lifecycle_failures_are_best_effort_and_sanitized(store, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    run, _reserved, job = _seed_held_run_job(store, occurrence_id="occ-cleanup-lifecycle-failure")

    class _LoggerStub:
        def __init__(self):
            self.bindings: list[dict] = []
            self.messages: list[str] = []

        def bind(self, **kwargs):
            self.bindings.append(dict(kwargs))
            return self

        def warning(self, message, *_args, **_kwargs):
            self.messages.append(str(message))

    logger_stub = _LoggerStub()

    def fail_lifecycle(*_args, **_kwargs):
        raise RuntimeError("https://example.com/private?token=secret")

    monkeypatch.setattr(store._jobs, "_update_gauges", fail_lifecycle)
    monkeypatch.setattr(store._jobs, "_cancel_dependent_jobs", fail_lifecycle)
    monkeypatch.setattr(playlist_ingest_store, "increment_cancelled", fail_lifecycle, raising=False)
    monkeypatch.setattr(playlist_ingest_store, "emit_job_event", fail_lifecycle, raising=False)
    monkeypatch.setattr(playlist_ingest_store, "logger", logger_stub)
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"
    assert "secret" not in repr((logger_stub.bindings, logger_stub.messages))
    with store._jobs._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?", (run.run_id,)).fetchone()[0]
            == 0
        )


def test_cleanup_expired_resources_preserves_published_job(store):
    run, reserved, job = _seed_held_run_job(store, occurrence_id="occ-cleanup-published")
    store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-cleanup-published",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 0
    assert store._jobs.get_job(int(job["id"]))["status"] == "queued"
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE owner_user_id = ? AND run_id = ?",
                ("1", run.run_id),
            ).fetchone()[0]
            == 1
        )


def test_cleanup_expired_resources_preserves_processing_job(store):
    job_status = "processing"
    occurrence_id = f"occ-cleanup-{job_status}"
    run, reserved, job = _seed_held_run_job(store, occurrence_id=occurrence_id)
    store.bind_run_item_job(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE jobs SET status = ? WHERE id = ?",
            (job_status, int(job["id"])),
        )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 0
    assert store._jobs.get_job(int(job["id"]))["status"] == job_status
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE owner_user_id = ? AND run_id = ?",
                ("1", run.run_id),
            ).fetchone()[0]
            == 1
        )


def test_cleanup_expired_resources_preserves_currently_claimed_job(store):
    occurrence_id = "occ-cleanup-claimed"
    run, reserved, job = _seed_held_run_job(store, occurrence_id=occurrence_id)
    store.bind_run_item_job(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    claimed = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="cleanup-active-worker",
        lease_seconds=30,
        owner_user_id="1",
        job_type="media_ingest_item",
    )
    assert claimed is not None
    assert int(claimed["id"]) == int(job["id"])
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 0
    assert store._jobs.get_job(int(job["id"]))["status"] == "processing"
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE owner_user_id = ? AND run_id = ?",
                ("1", run.run_id),
            ).fetchone()[0]
            == 1
        )
    assert (
        store._jobs.acquire_next_job(
            domain="media_ingest",
            queue="default",
            worker_id="cleanup-duplicate-worker",
            lease_seconds=30,
            owner_user_id="1",
            job_type="media_ingest_item",
        )
        is None
    )


def test_cleanup_expired_resources_preserves_claimed_job_after_item_starts_running(store):
    occurrence_id = "occ-cleanup-running-item"
    run, reserved, job = _seed_held_run_job(store, occurrence_id=occurrence_id)
    store.bind_run_item_job(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    claimed = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="cleanup-running-item-worker",
        lease_seconds=30,
        owner_user_id="1",
        job_type="media_ingest_item",
    )
    assert claimed is not None
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            """
            UPDATE media_ingest_run_items SET state = 'running'
            WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
            """,
            ("1", run.run_id, occurrence_id),
        )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 0
    assert store._jobs.get_job(int(job["id"]))["status"] == "processing"
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE owner_user_id = ? AND run_id = ?",
                ("1", run.run_id),
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("job_status", ["completed", "failed", "cancelled", "quarantined"])
def test_cleanup_expired_resources_allows_terminal_jobs_to_survive_run_deletion(store, job_status):
    occurrence_id = f"occ-cleanup-{job_status}"
    run, reserved, job = _seed_held_run_job(store, occurrence_id=occurrence_id)
    store.bind_run_item_job(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE jobs SET status = ? WHERE id = ?",
            (job_status, int(job["id"])),
        )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    assert store._jobs.get_job(int(job["id"]))["status"] == job_status


def test_postgres_expired_run_job_proof_locks_rows_and_blocks_processing(
    store,
    monkeypatch,
):
    queries: list[tuple[str, tuple]] = []

    class _Rows:
        def fetchall(self):
            return [
                {
                    "run_id": "run-active",
                    "occurrence_id": "occ-active",
                    "attempt": 1,
                    "job_id": 11,
                    "batch_id": "batch-active",
                    "idempotency_identity": "playlist-ingest-v1:active",
                    "submission_queue": "default",
                    "id": 11,
                    "uuid": "job-active",
                    "owner_user_id": "1",
                    "domain": "media_ingest",
                    "queue": "default",
                    "job_type": "media_ingest_item",
                    "status": "processing",
                    "available_at": NOW,
                    "batch_group": "batch-active",
                    "idempotency_key": "playlist-ingest-v1:active",
                    "created_at": NOW,
                    "payload": {
                        "run_id": "run-active",
                        "occurrence_id": "occ-active",
                        "attempt": 1,
                    },
                }
            ]

    def fake_query(_db, sql, params=()):
        queries.append((sql, params))
        return _Rows()

    store._postgres = True
    monkeypatch.setattr(store, "_query", fake_query)

    proof = store._cancel_expired_run_held_jobs(
        object(),
        owner_user_id="1",
        cutoff=NOW,
        run_ids=["run-active", "run-terminal"],
    )

    assert proof.deletable_run_ids == ("run-terminal",)
    assert proof.cancelled_jobs == ()
    assert len(queries) == 1
    assert "run.run_id = ANY(?)" in queries[0][0]
    assert queries[0][0].rstrip().endswith("FOR UPDATE OF item, job")


def test_cleanup_job_eligibility_proves_every_row_before_cancelling_any_job(store):
    run, _reservations, jobs = _seed_held_run_jobs(
        store,
        occurrence_ids=("occ-proof-first", "occ-proof-ambiguous"),
    )
    with store._jobs._connect() as connection, connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM jobs WHERE id = ?",
                (int(jobs[1]["id"]),),
            ).fetchone()[0]
        )
        payload["occurrence_id"] = "occ-proof-mismatch"
        connection.execute(
            "UPDATE jobs SET payload = ? WHERE id = ?",
            (json.dumps(payload), int(jobs[1]["id"])),
        )

    with store._connection(owner_user_id="1", write=True) as db:
        proof = store._cancel_expired_run_held_jobs(
            db,
            owner_user_id="1",
            cutoff=NOW + timedelta(days=8),
            run_ids=[run.run_id],
        )

    assert proof.deletable_run_ids == ()
    assert proof.cancelled_jobs == ()
    assert [store._jobs.get_job(int(job["id"]))["status"] for job in jobs] == ["queued", "queued"]


def _assert_retained_run_staging_authority(store, run_id: str, staging_dir: Path) -> None:
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE owner_user_id = ? AND run_id = ?",
                ("1", run_id),
            ).fetchone()[0]
            == 1
        )
        item = connection.execute(
            """
            SELECT staging_temp_dir FROM media_ingest_run_items
            WHERE owner_user_id = ? AND run_id = ?
            """,
            ("1", run_id),
        ).fetchone()
        assert item is not None
        assert item[0] == str(staging_dir)
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_run_events WHERE owner_user_id = ? AND run_id = ?",
                ("1", run_id),
            ).fetchone()[0]
            > 0
        )


def test_cleanup_decrypt_failure_retains_only_affected_run_and_held_quota(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.exceptions import JobSubmissionLimitError

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_MEDIA_INGEST_USER_1", "1")
    monkeypatch.setenv("JOBS_ENCRYPT_MEDIA_INGEST", "true")
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"playlist-cleanup-encryption-key!"[:32]).decode(),
    )
    retained, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-decrypt-failure",
    )
    store.append_run_event("1", retained.run_id, event_type="decrypt-failure-authority")
    eligible = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-cleanup-decrypt-eligible")],
    )
    store.test_clock.advance(timedelta(days=8))
    monkeypatch.setattr(store._jobs, "_maybe_decrypt_json", lambda payload: payload)

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    _assert_retained_run_staging_authority(store, retained.run_id, staging_dir)
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                (eligible.run_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT status FROM jobs WHERE id = ?",
                (int(job["id"]),),
            ).fetchone()[0]
            == "queued"
        )
        counter = connection.execute(
            """
            SELECT scheduled_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default'
              AND job_type = 'media_ingest_item'
            """
        ).fetchone()
    assert counter is not None
    assert int(counter[0]) == 1
    with pytest.raises(JobSubmissionLimitError, match="max queued"):
        store._jobs.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={"source": "https://example.com/decrypt-quota"},
            owner_user_id="1",
        )


def test_cleanup_binding_mismatch_retains_only_affected_run_and_held_quota(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.exceptions import JobSubmissionLimitError

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_MEDIA_INGEST_USER_1", "1")
    retained, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-binding-mismatch",
    )
    store.append_run_event("1", retained.run_id, event_type="binding-mismatch-authority")
    eligible = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-cleanup-mismatch-eligible")],
    )
    with store._jobs._connect() as connection, connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM jobs WHERE id = ?",
                (int(job["id"]),),
            ).fetchone()[0]
        )
        payload["occurrence_id"] = "occ-client-mismatch"
        connection.execute(
            "UPDATE jobs SET payload = ? WHERE id = ?",
            (json.dumps(payload), int(job["id"])),
        )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    _assert_retained_run_staging_authority(store, retained.run_id, staging_dir)
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                (eligible.run_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT status FROM jobs WHERE id = ?",
                (int(job["id"]),),
            ).fetchone()[0]
            == "queued"
        )
    with pytest.raises(JobSubmissionLimitError, match="max queued"):
        store._jobs.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={"source": "https://example.com/mismatch-quota"},
            owner_user_id="1",
        )


def test_cleanup_normalization_exception_retains_only_affected_run(
    store,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    retained, _reserved, _job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-normalization-error",
    )
    eligible = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-cleanup-normalization-eligible")],
    )
    store.test_clock.advance(timedelta(days=8))

    def fail_normalization(_job, *, owner_user_id):  # noqa: ARG001
        raise RuntimeError("synthetic decrypt adapter failure")

    monkeypatch.setattr(store._jobs, "normalize_job_binding_view", fail_normalization)

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    _assert_retained_run_staging_authority(store, retained.run_id, staging_dir)
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                (eligible.run_id,),
            ).fetchone()[0]
            == 0
        )


def test_cleanup_expired_resources_rolls_back_no_staging_cancellation_on_delete_failure(
    store,
    monkeypatch,
):
    run, _reserved, job = _seed_held_run_job(store, occurrence_id="occ-cleanup-rollback")
    lifecycle_calls: list[tuple[dict, ...]] = []
    monkeypatch.setattr(
        store,
        "_emit_expired_job_cancellation_lifecycle",
        lambda jobs: lifecycle_calls.append(tuple(jobs)),
    )
    store.test_clock.advance(timedelta(days=8))
    original_query = store._query

    def fail_after_cancel(db, sql, params=()):
        if "DELETE FROM media_ingest_run_events" in sql:
            raise RuntimeError("synthetic cleanup delete failure")
        return original_query(db, sql, params)

    monkeypatch.setattr(store, "_query", fail_after_cancel)

    with pytest.raises(RuntimeError, match="synthetic cleanup delete failure"):
        store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert store._jobs.get_job(int(job["id"]))["status"] == "queued"
    with store._jobs._connect() as connection:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]
    assert remaining == 1
    assert lifecycle_calls == []

    monkeypatch.setattr(store, "_query", original_query)

    assert store.cleanup_expired_resources("1", now=store.test_clock.current)["runs"] == 1
    assert len(lifecycle_calls) == 1
    assert [cancelled["id"] for cancelled in lifecycle_calls[0]] == [int(job["id"])]


def test_cleanup_expired_resources_retires_unreferenced_staging_after_commit(
    store,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-file-held",
    )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 1
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"
    assert not staging_dir.exists()
    with store._jobs._connect() as connection:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]
    assert remaining == 0


@pytest.mark.parametrize("cleanup_outcome", ["invalid", "protected", "failed"])
def test_cleanup_staging_retirement_outcome_retains_authority(
    store,
    tmp_path,
    monkeypatch,
    cleanup_outcome,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id=f"occ-cleanup-file-{cleanup_outcome}",
    )
    store.test_clock.advance(timedelta(days=8))
    monkeypatch.setattr(
        playlist_ingest_store,
        "cleanup_exact_run_file_staging",
        lambda **_kwargs: cleanup_outcome,
    )

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 0
    _assert_retained_run_staging_authority(store, run.run_id, staging_dir)
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"
    assert staging_dir.exists()


def test_cleanup_failed_staging_retirement_retries_before_run_deletion(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-file-retry",
    )
    store.test_clock.advance(timedelta(days=8))
    original_cleanup = playlist_ingest_store.cleanup_exact_run_file_staging
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return "failed"
        return original_cleanup(**kwargs)

    monkeypatch.setattr(playlist_ingest_store, "cleanup_exact_run_file_staging", fail_once)

    first = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert first["runs"] == 0
    _assert_retained_run_staging_authority(store, run.run_id, staging_dir)
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"

    second = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert second["runs"] == 1
    assert attempts == 2
    assert not staging_dir.exists()
    with store._jobs._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_run_items WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()[0]
            == 0
        )


def test_cleanup_staging_scan_cap_ambiguity_retries_before_run_deletion(
    store,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-file-scan-cap",
    )
    store.test_clock.advance(timedelta(days=8))
    monkeypatch.setattr(store, "_has_live_job_staging_reference", lambda *_args, **_kwargs: True)

    first = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert first["runs"] == 0
    _assert_retained_run_staging_authority(store, run.run_id, staging_dir)
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"

    monkeypatch.setattr(store, "_has_live_job_staging_reference", lambda *_args, **_kwargs: False)
    second = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert second["runs"] == 1
    assert not staging_dir.exists()


@pytest.mark.parametrize("missing_payload_field", ["attempt", "batch_id", "temp_dir"])
def test_staging_reference_check_fails_closed_when_held_payload_binding_is_incomplete(
    store,
    tmp_path,
    monkeypatch,
    missing_payload_field,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        _RunStagingCleanupCandidate,
    )

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    occurrence_id = f"occ-cleanup-incomplete-{missing_payload_field}"
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id=occurrence_id,
    )
    with store._jobs._connect() as connection, connection:
        payload = json.loads(connection.execute("SELECT payload FROM jobs WHERE id = ?", (job["id"],)).fetchone()[0])
        payload.pop(missing_payload_field)
        connection.execute(
            "UPDATE jobs SET payload = ? WHERE id = ?",
            (json.dumps(payload), int(job["id"])),
        )
    candidate = _RunStagingCleanupCandidate(
        run_id=run.run_id,
        occurrence_id=occurrence_id,
        batch_id=f"batch-{occurrence_id}",
        idempotency_identity=f"playlist-ingest-v1:{occurrence_id}",
        temp_dir=str(staging_dir),
    )

    assert store._has_live_job_staging_reference("1", candidate) is True


def test_cleanup_publish_wins_supported_bind_race_and_preserves_staging(
    store,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-publish-wins",
    )
    publish_entered = Event()
    allow_publish = Event()
    cleanup_started = Event()
    original_publish = store._publish_run_bound_job

    def blocked_publish(*args, **kwargs):
        publish_entered.set()
        assert allow_publish.wait(timeout=5)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(store, "_publish_run_bound_job", blocked_publish)

    def bind():
        return store.bind_run_item_job(
            "1",
            run.run_id,
            "occ-cleanup-publish-wins",
            attempt=1,
            job_id=int(job["id"]),
            batch_id="batch-occ-cleanup-publish-wins",
            idempotency_identity="playlist-ingest-v1:occ-cleanup-publish-wins",
            submission_lease_token=reserved.submission_lease_token,
        )

    def cleanup():
        cleanup_started.set()
        return store.cleanup_expired_resources("1", now=NOW + timedelta(days=8))

    with ThreadPoolExecutor(max_workers=2) as executor:
        bind_future = executor.submit(bind)
        assert publish_entered.wait(timeout=5)
        cleanup_future = executor.submit(cleanup)
        assert cleanup_started.wait(timeout=5)
        assert cleanup_future.done() is False
        allow_publish.set()
        assert bind_future.result(timeout=5).state == "queued"
        deleted = cleanup_future.result(timeout=5)

    assert deleted["runs"] == 0
    assert staging_dir.exists()
    assert store._jobs.get_job(int(job["id"]))["status"] == "queued"
    _assert_retained_run_staging_authority(store, run.run_id, staging_dir)


def test_cleanup_cancellation_wins_supported_bind_race_before_staging_retirement(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-cancel-wins",
    )
    lifecycle_entered = Event()
    allow_retirement = Event()
    lifecycle_calls: list[tuple[dict, ...]] = []

    def pause_after_cancellation(jobs):
        lifecycle_calls.append(tuple(jobs))
        lifecycle_entered.set()
        assert allow_retirement.wait(timeout=5)

    monkeypatch.setattr(store, "_emit_expired_job_cancellation_lifecycle", pause_after_cancellation)

    def bind():
        return store.bind_run_item_job(
            "1",
            run.run_id,
            "occ-cleanup-cancel-wins",
            attempt=1,
            job_id=int(job["id"]),
            batch_id="batch-occ-cleanup-cancel-wins",
            idempotency_identity="playlist-ingest-v1:occ-cleanup-cancel-wins",
            submission_lease_token=reserved.submission_lease_token,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        cleanup_future = executor.submit(
            store.cleanup_expired_resources,
            "1",
            now=NOW + timedelta(days=8),
        )
        assert lifecycle_entered.wait(timeout=5)
        assert staging_dir.exists()
        assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"
        with store._jobs._connect() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?",
                    (run.run_id,),
                ).fetchone()[0]
                == 1
            )
        bind_future = executor.submit(bind)
        with pytest.raises(PlaylistIngestConflictError):
            bind_future.result(timeout=5)
        allow_retirement.set()
        assert cleanup_future.result(timeout=5)["runs"] == 1

    assert len(lifecycle_calls) == 1
    assert not staging_dir.exists()


def test_cleanup_final_transaction_rollback_preserves_authority_after_staging_retirement(
    store,
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-file-rollback",
    )
    with store._jobs._connect() as connection:
        scheduled_before = connection.execute(
            """
            SELECT scheduled_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default'
              AND job_type = 'media_ingest_item'
            """
        ).fetchone()[0]
    store.test_clock.advance(timedelta(days=8))
    original_query = store._query

    def fail_after_cancel(db, sql, params=()):
        if "DELETE FROM media_ingest_run_events" in sql:
            raise RuntimeError("synthetic cleanup delete failure")
        return original_query(db, sql, params)

    monkeypatch.setattr(store, "_query", fail_after_cancel)

    with pytest.raises(RuntimeError, match="synthetic cleanup delete failure"):
        store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert not staging_dir.exists()
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"
    with store._jobs._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM media_ingest_runs WHERE run_id = ?", (run.run_id,)).fetchone()[0]
            == 1
        )
        item = connection.execute(
            """
            SELECT staging_temp_dir FROM media_ingest_run_items
            WHERE run_id = ? AND occurrence_id = ?
            """,
            (run.run_id, "occ-cleanup-file-rollback"),
        ).fetchone()
        assert item is not None
        assert item[0] is None
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM media_ingest_run_events WHERE run_id = ?", (run.run_id,)
            ).fetchone()[0]
            >= 1
        )
        assert (
            connection.execute(
                """
            SELECT scheduled_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default'
              AND job_type = 'media_ingest_item'
            """
            ).fetchone()[0]
            == max(scheduled_before - 1, 0)
        )

    monkeypatch.setattr(store, "_query", original_query)

    def unexpected_second_retirement(**_kwargs):
        raise AssertionError("confirmed staging retirement must not be retried")

    monkeypatch.setattr(
        playlist_ingest_store,
        "cleanup_exact_run_file_staging",
        unexpected_second_retirement,
    )

    assert store.cleanup_expired_resources("1", now=store.test_clock.current)["runs"] == 1
    assert store._jobs.get_job(int(job["id"]))["status"] == "cancelled"


def test_cleanup_expired_resources_preserves_published_staging_reference(
    store,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    run, _reserved, job, staging_dir = _seed_file_run_job(
        store,
        tmp_path,
        occurrence_id="occ-cleanup-file-published",
        publish=True,
    )
    store.test_clock.advance(timedelta(days=8))

    deleted = store.cleanup_expired_resources("1", now=store.test_clock.current)

    assert deleted["runs"] == 0
    assert store._jobs.get_job(int(job["id"]))["status"] == "queued"
    assert staging_dir.exists()
    _assert_retained_run_staging_authority(store, run.run_id, staging_dir)


def test_staging_reference_scan_finds_encrypted_job_after_first_page(store, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_staging import (
        run_file_staging_prefix,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        _RunStagingCleanupCandidate,
    )

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("JOBS_ENCRYPT_MEDIA_INGEST", "true")
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44],
    )
    batch_id = "batch-reference-page-two"
    identity = "playlist-ingest-v1:reference-page-two"
    staging_dir = tmp_path / f"{run_file_staging_prefix(batch_id=batch_id, idempotency_identity=identity)}candidate"
    staging_dir.mkdir()
    candidate = _RunStagingCleanupCandidate(
        run_id="run-reference-page-two",
        occurrence_id="occ-reference-page-two",
        batch_id=batch_id,
        idempotency_identity=identity,
        temp_dir=str(staging_dir),
    )
    store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        owner_user_id="1",
        batch_group=batch_id,
        idempotency_key=identity,
        payload={"temp_dir": str(staging_dir)},
    )
    for index in range(100):
        store._jobs.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            owner_user_id="1",
            batch_group=batch_id,
            idempotency_key=f"playlist-ingest-v1:unrelated-{index}",
            payload={"source": f"https://example.com/{index}"},
        )
    list_calls: list[dict] = []
    original_list_jobs = store._jobs.list_jobs

    def tracking_list_jobs(**kwargs):
        list_calls.append(dict(kwargs))
        return original_list_jobs(**kwargs)

    monkeypatch.setattr(store._jobs, "list_jobs", tracking_list_jobs)

    assert store._has_live_job_staging_reference("1", candidate) is True
    assert len(list_calls) == 2


def test_staging_reference_scan_fails_closed_at_total_bound(store, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_staging import (
        run_file_staging_prefix,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        _RunStagingCleanupCandidate,
    )

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    batch_id = "batch-total-scan-bound"
    identity = "playlist-ingest-v1:total-scan-bound"
    staging_dir = tmp_path / f"{run_file_staging_prefix(batch_id=batch_id, idempotency_identity=identity)}candidate"
    staging_dir.mkdir()
    candidate = _RunStagingCleanupCandidate(
        run_id="run-total-scan-bound",
        occurrence_id="occ-total-scan-bound",
        batch_id=batch_id,
        idempotency_identity=identity,
        temp_dir=str(staging_dir),
    )
    list_calls: list[dict] = []

    def full_page(**kwargs):
        list_calls.append(dict(kwargs))
        if len(list_calls) > 5:
            raise AssertionError("staging reference scan exceeded total bound")
        page = len(list_calls)
        return [
            {
                "id": 1000 - (page * 100) - index,
                "created_at": NOW - timedelta(minutes=page),
                "status": "queued",
                "batch_group": batch_id,
                "idempotency_key": f"playlist-ingest-v1:other-{page}-{index}",
                "payload": {},
            }
            for index in range(100)
        ]

    monkeypatch.setattr(store._jobs, "list_jobs", full_page)
    monkeypatch.setattr(
        store._jobs,
        "normalize_job_binding_view",
        lambda job, **_kwargs: job,
    )
    monkeypatch.setattr(store, "has_live_run_item_staging_reference", lambda *_args, **_kwargs: False)
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        playlist_ingest_store,
        "cleanup_exact_run_file_staging",
        lambda **kwargs: cleanup_calls.append(str(kwargs["temp_dir"])),
    )

    store._retire_expired_run_staging("1", [candidate])

    assert cleanup_calls == []
    assert len(list_calls) == 5
    assert all(call["limit"] == 100 for call in list_calls)


def test_staging_retirement_failure_log_excludes_client_identifiers(store, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_store
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        _RunStagingCleanupCandidate,
    )

    sentinel = "client-sentinel-must-not-leak"
    candidate = _RunStagingCleanupCandidate(
        run_id=f"run-{sentinel}",
        occurrence_id=f"occ-{sentinel}",
        batch_id="batch-safe-log",
        idempotency_identity="playlist-ingest-v1:safe-log",
        temp_dir=sentinel,
    )

    class _LoggerStub:
        def __init__(self) -> None:
            self.bindings: list[dict] = []
            self.messages: list[str] = []

        def bind(self, **kwargs):
            self.bindings.append(dict(kwargs))
            return self

        def warning(self, message, *args):
            self.messages.append(str(message).format(*args))

    logger_stub = _LoggerStub()

    def fail_nested(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(store, "has_live_run_item_staging_reference", fail_nested)
    monkeypatch.setattr(playlist_ingest_store, "logger", logger_stub)

    store._retire_expired_run_staging("1", [candidate])

    assert logger_stub.bindings == [
        {
            "error_code": "playlist_staging_retirement_failed",
            "error_type": "RuntimeError",
            "failure_count": 1,
        }
    ]
    assert sentinel not in repr((logger_stub.bindings, logger_stub.messages))


def test_cleanup_keeps_active_db_time_preflight_linked_to_acquirable_job(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_TTL_SECONDS", "600")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "cleanup-active-db-time.db")
    service = PlaylistIngestService(manager)
    created = service.create_preflight(
        "cleanup-owner",
        url="https://www.youtube.com/playlist?list=PLcleanupactive",
        max_items=20,
        timeout_seconds=10,
    )
    store = PlaylistIngestStore(manager)
    cutoff = datetime.now(timezone.utc)

    deleted = store.cleanup_expired("cleanup-owner", now=cutoff)

    assert deleted["preflights"] == 0
    remaining = store.get_preflight("cleanup-owner", created.preflight_id)
    assert remaining.job_id == created.record.job_id
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="cleanup-active-worker",
        lease_seconds=120,
        job_type="playlist_preflight",
    )
    assert claimed is not None
    assert int(claimed["id"]) == remaining.job_id


@pytest.mark.parametrize("storage_format", ["iso", "database"])
def test_cleanup_deletes_historical_expiry_formats(tmp_path, monkeypatch, storage_format):
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / f"cleanup-expired-{storage_format}.db")
    store = PlaylistIngestStore(manager)
    now = datetime.now(timezone.utc)
    preflight = store.create_preflight(
        "cleanup-owner",
        source_url="https://example.com/historical-expiry",
        source_kind="url",
        expires_at=now + timedelta(hours=1),
    )
    expired = now - timedelta(hours=1)
    stored_expiry = expired.isoformat() if storage_format == "iso" else expired.strftime("%Y-%m-%d %H:%M:%S")
    with manager._connect() as connection:
        connection.execute(
            "UPDATE playlist_preflights SET expires_at = ? WHERE preflight_id = ?",
            (stored_expiry, preflight.preflight_id),
        )
        connection.commit()

    deleted = store.cleanup_expired("cleanup-owner", now=now)

    assert deleted["preflights"] == 1


def test_attach_collection_plan_persists_run_and_per_occurrence_mapping(store):
    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(occurrence_id="occ-one"),
            _validated_direct_record(
                occurrence_id="occ-skip",
                source_url="https://example.com/skip",
                normalized_source_id="url:https://example.com/skip",
                action="skip",
                media_id=9,
            ),
            _validated_direct_record(
                occurrence_id="occ-two",
                source_url="https://example.com/two",
                normalized_source_id="url:https://example.com/two",
            ),
        ],
    )

    attached = store.attach_collection_plan(
        "1",
        run.run_id,
        collection_id=55,
        planned_item_ids={"occ-one": 101, "occ-two": 102},
    )

    items = list(store.list_run_items("1", run.run_id, limit=10))
    events = list(store.list_run_events("1", run.run_id))
    assert attached.collection_id == 55
    assert [item.planned_collection_item_id for item in items] == [101, None, 102]
    assert events[-1].event_type == "collection_plan_attached"
    assert events[-1].attrs == {"collection_id": 55, "planned_item_count": 2}


def test_attach_collection_plan_reads_attached_run_before_commit(store, monkeypatch):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-one")],
    )
    monkeypatch.setattr(
        store,
        "get_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("post-commit readback")),
    )

    attached = store.attach_collection_plan(
        "1",
        run.run_id,
        collection_id=55,
        planned_item_ids={"occ-one": 101},
    )

    assert attached.collection_id == 55
    assert attached.version == run.version + 1


def test_nonprocessing_action_preparation_is_durable_and_idempotent(store):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(action="skip")],
    )

    prepared = store.prepare_nonprocessing_run_item("1", run.run_id, "occ-direct")
    repeated = store.prepare_nonprocessing_run_item("1", run.run_id, "occ-direct")

    loaded = store.get_run("1", run.run_id)
    events = list(store.list_run_events("1", run.run_id))
    assert prepared.state == "preparing"
    assert repeated.state == "preparing"
    assert loaded.version == run.version + 1
    assert [event.event_type for event in events].count("duplicate_action_preparing") == 1


@pytest.mark.parametrize(
    ("action", "outcome", "media_id"),
    [
        ("ingest", "included_existing", 17),
        ("overwrite", "included_existing", 17),
        ("skip", "included_existing", 17),
        ("include_existing", "skipped_existing", 17),
        ("include_existing", "included_existing", None),
        ("update_metadata_only", "included_existing", 17),
        ("update_metadata_only", "metadata_updated", None),
    ],
)
def test_nonprocessing_action_outcome_mapping_is_enforced(
    store,
    action,
    outcome,
    media_id,
):
    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(
                action=action,
                metadata_patch=({"title": "Reviewed"} if action in {"overwrite", "update_metadata_only"} else None),
            )
        ],
    )
    if action in {"skip", "include_existing", "update_metadata_only"}:
        store.prepare_nonprocessing_run_item("1", run.run_id, "occ-direct")

    with pytest.raises(ValueError, match="invalid non-processing action outcome"):
        store.resolve_nonprocessing_run_item(
            "1",
            run.run_id,
            "occ-direct",
            outcome=outcome,
            media_id=media_id,
        )
    assert store.get_run_item("1", run.run_id, "occ-direct").state != "terminal"


def test_include_existing_can_terminalize_with_safe_action_failure(store):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(action="include_existing")],
    )
    store.prepare_nonprocessing_run_item("1", run.run_id, "occ-direct")

    resolved = store.resolve_nonprocessing_run_item(
        "1",
        run.run_id,
        "occ-direct",
        outcome="metadata_update_failed",
        media_id=17,
    )

    assert resolved.state == "terminal"
    assert resolved.outcome == "metadata_update_failed"
    assert store.get_run("1", run.run_id).status == "completed"


def test_nonprocessing_action_requires_preparing_and_exact_terminal_retry_is_idempotent(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(action="include_existing")],
    )
    with pytest.raises(PlaylistIngestConflictError, match="no longer preparing"):
        store.resolve_nonprocessing_run_item(
            "1",
            run.run_id,
            "occ-direct",
            outcome="included_existing",
            media_id=17,
        )

    store.prepare_nonprocessing_run_item("1", run.run_id, "occ-direct")
    resolved = store.resolve_nonprocessing_run_item(
        "1",
        run.run_id,
        "occ-direct",
        outcome="included_existing",
        media_id=17,
    )
    version = store.get_run("1", run.run_id).version
    repeated = store.resolve_nonprocessing_run_item(
        "1",
        run.run_id,
        "occ-direct",
        outcome="included_existing",
        media_id=17,
    )

    assert resolved.state == repeated.state == "terminal"
    assert store.get_run("1", run.run_id).version == version
    assert [event.event_type for event in store.list_run_events("1", run.run_id)].count(
        "duplicate_action_resolved"
    ) == 1


def test_run_completes_only_after_every_item_is_terminal(store):
    mixed = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(occurrence_id="occ-skip", action="skip"),
            _validated_direct_record(
                occurrence_id="occ-ingest",
                source_url="https://example.com/new",
                normalized_source_id="url:https://example.com/new",
            ),
        ],
    )
    store.prepare_nonprocessing_run_item("1", mixed.run_id, "occ-skip")
    store.resolve_nonprocessing_run_item("1", mixed.run_id, "occ-skip", outcome="skipped_existing", media_id=None)
    assert store.get_run("1", mixed.run_id).status == "staged"

    complete = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(occurrence_id="occ-one", action="skip"),
            _validated_direct_record(
                occurrence_id="occ-two",
                source_url="https://example.com/two",
                normalized_source_id="url:https://example.com/two",
                action="include_existing",
            ),
        ],
    )
    store.prepare_nonprocessing_run_item("1", complete.run_id, "occ-one")
    store.resolve_nonprocessing_run_item("1", complete.run_id, "occ-one", outcome="skipped_existing", media_id=None)
    assert store.get_run("1", complete.run_id).status == "staged"
    store.prepare_nonprocessing_run_item("1", complete.run_id, "occ-two")
    store.resolve_nonprocessing_run_item("1", complete.run_id, "occ-two", outcome="included_existing", media_id=17)
    assert store.get_run("1", complete.run_id).status == "completed"


def test_run_returns_to_staged_when_only_terminal_and_unsent_items_remain(store):
    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(occurrence_id="occ-bound"),
            _validated_direct_record(
                occurrence_id="occ-unsent",
                source_url="https://example.com/unsent",
                normalized_source_id="url:https://example.com/unsent",
            ),
        ],
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            """
            UPDATE media_ingest_run_items
            SET state = 'queued', job_id = 101
            WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
            """,
            ("1", run.run_id, "occ-bound"),
        )
        store._query(
            db,
            "UPDATE media_ingest_runs SET status = 'running' WHERE owner_user_id = ? AND run_id = ?",
            ("1", run.run_id),
        )

    store.reconcile_run_item_job(
        "1",
        run.run_id,
        "occ-bound",
        expected_job_id=101,
        expected_attempt=1,
        state="terminal",
        outcome="completed",
        progress_percent=100.0,
        progress_message="done",
        retryable=False,
        media_id=55,
    )

    assert store.get_run("1", run.run_id).status == "staged"
    assert {item.state for item in store.list_run_items("1", run.run_id, limit=10)} == {
        "staged",
        "terminal",
    }


def test_attach_collection_plan_rolls_back_every_mapping_on_failure(store, monkeypatch):
    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(occurrence_id="occ-one"),
            _validated_direct_record(
                occurrence_id="occ-two",
                source_url="https://example.com/two",
                normalized_source_id="url:https://example.com/two",
            ),
        ],
    )
    original_query = store._query
    mapping_updates = 0

    def fail_second_mapping(db, sql, params=()):
        nonlocal mapping_updates
        if "SET planned_collection_item_id = ?" in " ".join(sql.split()):
            mapping_updates += 1
            if mapping_updates == 2:
                raise RuntimeError("synthetic mapping failure")
        return original_query(db, sql, params)

    monkeypatch.setattr(store, "_query", fail_second_mapping)

    with pytest.raises(RuntimeError, match="synthetic mapping failure"):
        store.attach_collection_plan(
            "1",
            run.run_id,
            collection_id=55,
            planned_item_ids={"occ-one": 101, "occ-two": 102},
        )

    loaded = store.get_run("1", run.run_id)
    items = list(store.list_run_items("1", run.run_id, limit=10))
    assert loaded.collection_id is None
    assert loaded.status == "staged"
    assert loaded.version == run.version
    assert [item.planned_collection_item_id for item in items] == [None, None]


def test_run_item_job_submission_reservation_and_binding_are_idempotent(store):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-job")],
    )

    prepared = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-original",
        idempotency_identity="playlist-ingest-v1:derived",
        source_kind="url",
        planned_item_id=None,
    )
    repeated = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-new-request",
        idempotency_identity="playlist-ingest-v1:derived",
        source_kind="url",
        planned_item_id=None,
    )

    assert prepared.state == repeated.state == "submit_pending"
    assert prepared.batch_id == repeated.batch_id == "batch-original"
    assert store.get_run("1", run.run_id).version == run.version + 1

    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": "occ-job",
            "attempt": 1,
            "batch_id": "batch-original",
            "idempotency_key": "playlist-ingest-v1:derived",
            "source": prepared.source_url,
            "source_kind": "url",
            "options": {"overwrite_existing": False},
        },
        batch_group="batch-original",
        owner_user_id="1",
        idempotency_key="playlist-ingest-v1:derived",
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )
    bound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-original",
        idempotency_identity="playlist-ingest-v1:derived",
        submission_lease_token=prepared.submission_lease_token,
    )
    rebound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-original",
        idempotency_identity="playlist-ingest-v1:derived",
    )

    assert bound.state == rebound.state == "queued"
    assert bound.job_id == rebound.job_id == int(job["id"])
    assert store.get_run("1", run.run_id).batch_ids == ["batch-original"]
    events = list(store.list_run_events("1", run.run_id))
    assert [event.event_type for event in events].count("occurrence_submit_pending") == 1
    assert [event.event_type for event in events].count("occurrence_job_accepted") == 1


def test_submission_lease_waits_then_atomically_takes_over_with_stable_identity(store):
    run = store.create_validated_run("1", items=[_validated_direct_record(occurrence_id="occ-lease")])

    first = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-lease",
        attempt=1,
        batch_id="batch-original",
        idempotency_identity="playlist-ingest-v1:stable",
        submission_queue="heavy",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="lease-owner-a",
        submission_lease_seconds=10,
    )
    waiting = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-lease",
        attempt=1,
        batch_id="batch-retry",
        idempotency_identity="playlist-ingest-v1:rotated",
        submission_queue="default",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="lease-owner-b",
        submission_lease_seconds=10,
    )

    assert first.submission_lease_token == waiting.submission_lease_token == "lease-owner-a"
    assert first.submission_lease_generation == waiting.submission_lease_generation == 1
    assert first.submission_lease_expires_at == NOW + timedelta(seconds=10)

    store.test_clock.advance(timedelta(seconds=11))
    taken = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-lease",
        attempt=1,
        batch_id="batch-retry",
        idempotency_identity="playlist-ingest-v1:rotated",
        submission_queue="default",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="lease-owner-b",
        submission_lease_seconds=20,
    )

    assert taken.submission_lease_token == "lease-owner-b"
    assert taken.submission_lease_generation == 2
    assert taken.submission_lease_expires_at == store.test_clock.current + timedelta(seconds=20)
    assert taken.batch_id == "batch-original"
    assert taken.idempotency_identity == "playlist-ingest-v1:stable"
    assert taken.submission_queue == "heavy"


def test_submission_lease_takeover_race_has_one_new_owner(store):
    run = store.create_validated_run("1", items=[_validated_direct_record(occurrence_id="occ-race")])
    store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-race",
        attempt=1,
        batch_id="batch-race",
        idempotency_identity="playlist-ingest-v1:race",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="lease-crashed",
        submission_lease_seconds=5,
    )
    store.test_clock.advance(timedelta(seconds=6))

    def take_over(token: str):
        record = store.prepare_run_item_job_submission(
            "1",
            run.run_id,
            "occ-race",
            attempt=1,
            batch_id="batch-ignored",
            idempotency_identity="playlist-ingest-v1:ignored",
            source_kind="url",
            planned_item_id=None,
            submission_lease_token=token,
            submission_lease_seconds=30,
        )
        return token, record.submission_lease_token, record.submission_lease_generation

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(take_over, ["lease-racer-a", "lease-racer-b"]))

    assert sum(proposed == stored for proposed, stored, _generation in results) == 1
    assert {generation for _proposed, _stored, generation in results} == {2}


def test_url_submission_owner_can_release_lease_without_rotating_identity(store):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-url-release")],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-url-release",
        attempt=1,
        batch_id="batch-url-release",
        idempotency_identity="playlist-ingest-v1:url-release",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="lease-url-owner",
    )

    assert not store.release_run_item_url_submission_lease(
        "1",
        run.run_id,
        "occ-url-release",
        attempt=1,
        batch_id="batch-url-release",
        idempotency_identity="playlist-ingest-v1:url-release",
        submission_lease_token="lease-url-stale",
    )
    assert store.release_run_item_url_submission_lease(
        "1",
        run.run_id,
        "occ-url-release",
        attempt=1,
        batch_id="batch-url-release",
        idempotency_identity="playlist-ingest-v1:url-release",
        submission_lease_token="lease-url-owner",
    )
    released = store.get_run_item("1", run.run_id, "occ-url-release")
    assert released.submission_lease_token is None
    assert released.submission_lease_expires_at is None
    assert released.batch_id == reserved.batch_id
    assert released.idempotency_identity == reserved.idempotency_identity

    taken = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-url-release",
        attempt=1,
        batch_id="ignored-batch",
        idempotency_identity="playlist-ingest-v1:ignored",
        source_kind="url",
        planned_item_id=None,
        submission_lease_token="lease-url-retry",
    )
    assert taken.submission_lease_token == "lease-url-retry"
    assert taken.submission_lease_generation == reserved.submission_lease_generation + 1
    assert taken.batch_id == reserved.batch_id
    assert taken.idempotency_identity == reserved.idempotency_identity


def test_completed_file_staging_takeover_uses_generation_cas_and_preserves_pointer(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(
                occurrence_id="occ-completed-takeover",
                input_kind="file_stub",
                source_url=None,
                normalized_source_id=None,
                source_kind="file",
                state="awaiting_upload",
            )
        ],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-completed-takeover",
        attempt=1,
        batch_id="batch-completed-takeover",
        idempotency_identity="playlist-ingest-v1:completed-takeover",
        source_kind="file",
        planned_item_id=None,
        submission_lease_token="lease-original",
    )
    staging_dir = "/tmp/media_ingest_job_completed_takeover"
    store.record_run_item_staging(
        "1",
        run.run_id,
        "occ-completed-takeover",
        attempt=1,
        batch_id="batch-completed-takeover",
        idempotency_identity="playlist-ingest-v1:completed-takeover",
        submission_lease_token="lease-original",
        temp_dir=staging_dir,
    )

    taken = store.takeover_completed_run_item_submission_lease(
        "1",
        run.run_id,
        "occ-completed-takeover",
        attempt=1,
        batch_id="batch-completed-takeover",
        idempotency_identity="playlist-ingest-v1:completed-takeover",
        expected_submission_lease_token="lease-original",
        expected_submission_lease_generation=reserved.submission_lease_generation,
        submission_lease_token="lease-retry",
    )

    assert taken.submission_lease_token == "lease-retry"
    assert taken.submission_lease_generation == reserved.submission_lease_generation + 1
    assert taken.staging_temp_dir == staging_dir
    with pytest.raises(PlaylistIngestConflictError, match="lease"):
        store.takeover_completed_run_item_submission_lease(
            "1",
            run.run_id,
            "occ-completed-takeover",
            attempt=1,
            batch_id="batch-completed-takeover",
            idempotency_identity="playlist-ingest-v1:completed-takeover",
            expected_submission_lease_token="lease-original",
            expected_submission_lease_generation=reserved.submission_lease_generation,
            submission_lease_token="lease-racer",
        )
    assert not store.clear_run_item_staging(
        "1",
        run.run_id,
        "occ-completed-takeover",
        attempt=1,
        batch_id="batch-completed-takeover",
        idempotency_identity="playlist-ingest-v1:completed-takeover",
        submission_lease_token="lease-original",
        temp_dir=staging_dir,
    )
    assert store.clear_run_item_staging(
        "1",
        run.run_id,
        "occ-completed-takeover",
        attempt=1,
        batch_id="batch-completed-takeover",
        idempotency_identity="playlist-ingest-v1:completed-takeover",
        submission_lease_token="lease-retry",
        temp_dir=staging_dir,
    )


def test_submission_lease_token_gates_staging_heartbeat_and_reset(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(
                occurrence_id="occ-file-lease",
                input_kind="file_stub",
                source_url=None,
                normalized_source_id=None,
                source_kind="file",
                state="awaiting_upload",
            )
        ],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-file-lease",
        attempt=1,
        batch_id="batch-file-lease",
        idempotency_identity="playlist-ingest-v1:file-lease",
        source_kind="file",
        planned_item_id=None,
        submission_lease_token="lease-current",
        submission_lease_seconds=10,
    )
    store.test_clock.advance(timedelta(seconds=8))
    renewed = store.renew_run_item_submission_lease(
        "1",
        run.run_id,
        "occ-file-lease",
        attempt=1,
        batch_id="batch-file-lease",
        idempotency_identity="playlist-ingest-v1:file-lease",
        submission_lease_token="lease-current",
        submission_lease_seconds=20,
    )
    assert renewed.submission_lease_expires_at == store.test_clock.current + timedelta(seconds=20)
    assert renewed.submission_lease_generation == reserved.submission_lease_generation

    with pytest.raises(PlaylistIngestConflictError, match="lease"):
        store.record_run_item_staging(
            "1",
            run.run_id,
            "occ-file-lease",
            attempt=1,
            batch_id="batch-file-lease",
            idempotency_identity="playlist-ingest-v1:file-lease",
            submission_lease_token="lease-stale",
            temp_dir="/tmp/media_ingest_job_stale",
        )
    recorded = store.record_run_item_staging(
        "1",
        run.run_id,
        "occ-file-lease",
        attempt=1,
        batch_id="batch-file-lease",
        idempotency_identity="playlist-ingest-v1:file-lease",
        submission_lease_token="lease-current",
        temp_dir="/tmp/media_ingest_job_current",
    )
    assert recorded.staging_temp_dir == "/tmp/media_ingest_job_current"

    assert not store.clear_run_item_staging(
        "1",
        run.run_id,
        "occ-file-lease",
        attempt=1,
        batch_id="batch-file-lease",
        idempotency_identity="playlist-ingest-v1:file-lease",
        submission_lease_token="lease-stale",
        temp_dir="/tmp/media_ingest_job_current",
    )


def test_run_item_submission_reuses_stored_identity_and_queue_after_rotation(store):
    run = store.create_validated_run("1", items=[_validated_direct_record(occurrence_id="occ-job")])

    first = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-original",
        idempotency_identity="playlist-ingest-v1:old-secret",
        submission_queue="default",
        source_kind="url",
        planned_item_id=None,
    )
    retried = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-new",
        idempotency_identity="playlist-ingest-v1:new-secret",
        submission_queue="low",
        source_kind="url",
        planned_item_id=None,
    )

    assert first.idempotency_identity == retried.idempotency_identity == "playlist-ingest-v1:old-secret"
    assert first.submission_queue == retried.submission_queue == "default"
    assert retried.batch_id == "batch-original"


def test_upgraded_pending_submission_initializes_null_queue_once(store):
    run = store.create_validated_run("1", items=[_validated_direct_record(occurrence_id="occ-job")])
    store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-original",
        idempotency_identity="playlist-ingest-v1:stable",
        submission_queue="default",
        source_kind="url",
        planned_item_id=None,
    )
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE media_ingest_run_items SET submission_queue = NULL WHERE run_id = ? AND occurrence_id = ?",
            (run.run_id, "occ-job"),
        )

    initialized = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-new",
        idempotency_identity="playlist-ingest-v1:rotated",
        submission_queue="low",
        source_kind="url",
        planned_item_id=None,
    )
    repeated = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-newer",
        idempotency_identity="playlist-ingest-v1:rotated-again",
        submission_queue="high",
        source_kind="url",
        planned_item_id=None,
    )

    assert initialized.idempotency_identity == repeated.idempotency_identity == "playlist-ingest-v1:stable"
    assert initialized.submission_queue == repeated.submission_queue == "low"


def test_job_binding_view_decrypts_payload_and_fails_closed_for_other_owner(store, monkeypatch):
    monkeypatch.setenv("JOBS_ENCRYPT_MEDIA_INGEST", "true")
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44],
    )
    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={"run_id": "run-secret", "occurrence_id": "occ-secret", "attempt": 1},
        owner_user_id="1",
    )

    view = store._jobs.normalize_job_binding_view(job, owner_user_id="1")

    assert view is not None
    assert view["payload"] == {"run_id": "run-secret", "occurrence_id": "occ-secret", "attempt": 1}
    assert store._jobs.normalize_job_binding_view(job, owner_user_id="2") is None
    assert "_encrypted" not in view["payload"]


@pytest.mark.parametrize(
    "payload",
    [
        {"_encrypted": {"_enc": "aesgcm:v1", "nonce": "bad", "ciphertext": "bad"}},
        {"_enc": "aesgcm:v1", "nonce": "bad", "ciphertext": "bad"},
    ],
)
def test_job_binding_view_fails_closed_when_encrypted_payload_cannot_be_decrypted(
    store,
    monkeypatch,
    payload,
):
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    monkeypatch.setattr(jobs_manager, "decrypt_json_blob", lambda _envelope: None, raising=True)
    job = {
        "id": 17,
        "owner_user_id": "1",
        "domain": "media_ingest",
        "queue": "default",
        "job_type": "media_ingest_item",
        "payload": payload,
    }

    assert store._jobs.normalize_job_binding_view(job, owner_user_id="1") is None


def test_bind_run_item_job_rejects_overwrite_option_opposite_reserved_action(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id="occ-overwrite", action="overwrite")],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-overwrite",
        attempt=1,
        batch_id="batch-overwrite",
        idempotency_identity="playlist-ingest-v1:overwrite",
        submission_queue="default",
        source_kind="url",
        planned_item_id=None,
    )
    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": "occ-overwrite",
            "attempt": 1,
            "batch_id": "batch-overwrite",
            "idempotency_key": "playlist-ingest-v1:overwrite",
            "source": reserved.source_url,
            "source_kind": "url",
            "options": {"overwrite_existing": False},
        },
        batch_group="batch-overwrite",
        owner_user_id="1",
        idempotency_key="playlist-ingest-v1:overwrite",
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )

    with pytest.raises(PlaylistIngestConflictError, match="does not match occurrence"):
        store.bind_run_item_job(
            "1",
            run.run_id,
            "occ-overwrite",
            attempt=1,
            job_id=int(job["id"]),
            batch_id="batch-overwrite",
            idempotency_identity="playlist-ingest-v1:overwrite",
        )


def _seed_held_run_jobs(store, *, occurrence_ids: tuple[str, ...]):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id=occurrence_id) for occurrence_id in occurrence_ids],
    )
    reservations = []
    jobs = []
    for occurrence_id in occurrence_ids:
        batch_id = f"batch-{occurrence_id}"
        identity = f"playlist-ingest-v1:{occurrence_id}"
        reserved = store.prepare_run_item_job_submission(
            "1",
            run.run_id,
            occurrence_id,
            attempt=1,
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_queue="default",
            source_kind="url",
            planned_item_id=None,
        )
        job = store._jobs.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={
                "run_id": run.run_id,
                "occurrence_id": occurrence_id,
                "attempt": 1,
                "batch_id": batch_id,
                "idempotency_key": identity,
                "source": reserved.source_url,
                "source_kind": "url",
                "options": {"overwrite_existing": False},
            },
            owner_user_id="1",
            batch_group=batch_id,
            idempotency_key=identity,
            available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        )
        reservations.append(reserved)
        jobs.append(job)
    return run, reservations, jobs


def _seed_held_run_job(store, *, occurrence_id="occ-held"):
    run = store.create_validated_run(
        "1",
        items=[_validated_direct_record(occurrence_id=occurrence_id)],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_queue="default",
        source_kind="url",
        planned_item_id=None,
    )
    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": occurrence_id,
            "attempt": 1,
            "batch_id": "batch-held",
            "idempotency_key": "playlist-ingest-v1:held",
            "source": reserved.source_url,
            "source_kind": "url",
            "options": {"overwrite_existing": False},
        },
        owner_user_id="1",
        batch_group="batch-held",
        idempotency_key="playlist-ingest-v1:held",
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )
    return run, reserved, job


def _seed_file_run_jobs(store, tmp_path, *, occurrence_ids: tuple[str, ...]):
    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import _run_file_staging_prefix

    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(
                occurrence_id=occurrence_id,
                input_kind="file_stub",
                source_url=None,
                normalized_source_id=None,
                source_kind="file",
                state="awaiting_upload",
            )
            for occurrence_id in occurrence_ids
        ],
    )
    jobs = []
    staging_dirs = []
    for occurrence_id in occurrence_ids:
        batch_id = f"batch-{occurrence_id}"
        identity = f"playlist-ingest-v1:{occurrence_id}"
        reserved = store.prepare_run_item_job_submission(
            "1",
            run.run_id,
            occurrence_id,
            attempt=1,
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_queue="default",
            source_kind="file",
            planned_item_id=None,
        )
        prefix = _run_file_staging_prefix(
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
        )
        staging_dir = tmp_path / f"{prefix}cleanup"
        staging_dir.mkdir()
        source = staging_dir / "clip.mp3"
        source.write_bytes(b"test")
        store.record_run_item_staging(
            "1",
            run.run_id,
            occurrence_id,
            attempt=1,
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
            temp_dir=str(staging_dir),
        )
        jobs.append(
            store._jobs.create_job(
                domain="media_ingest",
                queue="default",
                job_type="media_ingest_item",
                payload={
                    "run_id": run.run_id,
                    "occurrence_id": occurrence_id,
                    "attempt": 1,
                    "batch_id": batch_id,
                    "idempotency_key": identity,
                    "source": str(source),
                    "source_kind": "file",
                    "temp_dir": str(staging_dir),
                    "cleanup_temp_dir": True,
                    "options": {"overwrite_existing": False},
                },
                owner_user_id="1",
                batch_group=batch_id,
                idempotency_key=identity,
                available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
        )
        staging_dirs.append(Path(staging_dir))
    return run, jobs, staging_dirs


def _seed_file_run_job(store, tmp_path, *, occurrence_id: str, publish: bool = False):
    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import _run_file_staging_prefix

    batch_id = f"batch-{occurrence_id}"
    identity = f"playlist-ingest-v1:{occurrence_id}"
    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(
                occurrence_id=occurrence_id,
                input_kind="file_stub",
                source_url=None,
                normalized_source_id=None,
                source_kind="file",
                state="awaiting_upload",
            )
        ],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        batch_id=batch_id,
        idempotency_identity=identity,
        submission_queue="default",
        source_kind="file",
        planned_item_id=None,
    )
    prefix = _run_file_staging_prefix(
        batch_id=batch_id,
        idempotency_identity=identity,
        submission_lease_token=reserved.submission_lease_token,
    )
    staging_dir = tmp_path / f"{prefix}cleanup"
    staging_dir.mkdir()
    source = staging_dir / "clip.mp3"
    source.write_bytes(b"test")
    store.record_run_item_staging(
        "1",
        run.run_id,
        occurrence_id,
        attempt=1,
        batch_id=batch_id,
        idempotency_identity=identity,
        submission_lease_token=reserved.submission_lease_token,
        temp_dir=str(staging_dir),
    )
    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": occurrence_id,
            "attempt": 1,
            "batch_id": batch_id,
            "idempotency_key": identity,
            "source": str(source),
            "source_kind": "file",
            "temp_dir": str(staging_dir),
            "cleanup_temp_dir": True,
            "options": {"overwrite_existing": False},
        },
        owner_user_id="1",
        batch_group=batch_id,
        idempotency_key=identity,
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )
    if publish:
        store.bind_run_item_job(
            "1",
            run.run_id,
            occurrence_id,
            attempt=1,
            job_id=int(job["id"]),
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
        )
    return run, reserved, job, Path(staging_dir)


def test_bind_run_item_job_publishes_held_job_atomically(store):
    run, reserved, job = _seed_held_run_job(store)

    assert (
        store._jobs.acquire_next_job(
            domain="media_ingest",
            queue="default",
            worker_id="before-bind",
            lease_seconds=30,
        )
        is None
    )

    bound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-held",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    acquired = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="after-bind",
        lease_seconds=30,
    )

    assert bound.state == "queued"
    assert acquired is not None
    assert acquired["id"] == job["id"]


def test_bind_run_item_job_rejects_nonaccepting_run_and_leaves_job_held(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run, reserved, job = _seed_held_run_job(store)
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE media_ingest_runs SET status = 'completed' WHERE run_id = ?",
            (run.run_id,),
        )

    with pytest.raises(PlaylistIngestConflictError, match="not accepting"):
        store.bind_run_item_job(
            "1",
            run.run_id,
            "occ-held",
            attempt=1,
            job_id=int(job["id"]),
            batch_id="batch-held",
            idempotency_identity="playlist-ingest-v1:held",
            submission_lease_token=reserved.submission_lease_token,
        )

    assert store.get_run_item("1", run.run_id, "occ-held").state == "submit_pending"
    assert (
        store._jobs.acquire_next_job(
            domain="media_ingest",
            queue="default",
            worker_id="after-reject",
            lease_seconds=30,
        )
        is None
    )


def test_bind_run_item_job_repairs_exact_bound_but_held_job(store):
    run, reserved, job = _seed_held_run_job(store)
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            """
            UPDATE media_ingest_run_items SET state = 'queued', job_id = ?
            WHERE owner_user_id = '1' AND run_id = ? AND occurrence_id = 'occ-held'
            """,
            (int(job["id"]), run.run_id),
        )

    repaired = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-held",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    acquired = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="repair-worker",
        lease_seconds=30,
    )

    assert repaired.job_id == job["id"]
    assert acquired is not None
    assert acquired["id"] == job["id"]


@pytest.mark.parametrize("complete", [False, True])
def test_bind_run_item_job_is_idempotent_after_published_job_advances_lifecycle(store, complete):
    run, reserved, job = _seed_held_run_job(store)
    bound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-held",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    acquired = store._jobs.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="lifecycle-worker",
        lease_seconds=30,
    )
    assert acquired is not None
    if complete:
        assert store._jobs.complete_job(int(job["id"]), enforce=False)

    rebound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-held",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
    )

    assert rebound.job_id == bound.job_id == job["id"]


def test_bind_run_item_job_accepts_already_bound_future_available_at(store):
    run, reserved, job = _seed_held_run_job(store, occurrence_id="occ-future-bound")
    bound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-future-bound",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
        submission_lease_token=reserved.submission_lease_token,
    )
    future_retry = store.test_clock.current + timedelta(minutes=15)
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            "UPDATE jobs SET status = 'queued', available_at = ? WHERE id = ?",
            (store._job_datetime(future_retry), int(job["id"])),
        )

    rebound = store.bind_run_item_job(
        "1",
        run.run_id,
        "occ-future-bound",
        attempt=1,
        job_id=int(job["id"]),
        batch_id="batch-held",
        idempotency_identity="playlist-ingest-v1:held",
    )

    assert bound.job_id == rebound.job_id == int(job["id"])


def test_persisted_file_staging_candidates_are_bounded_to_abandoned_pending_items(store):
    run = store.create_validated_run(
        "1",
        items=[
            {
                "occurrence_id": "occ-staging",
                "input_kind": "file_stub",
                "source_url": None,
                "normalized_source_id": None,
                "source_kind": "file",
                "display_metadata": {"name": "clip.mp3", "size_bytes": 4},
                "state": "awaiting_upload",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-staging",
        attempt=1,
        batch_id="batch-staging",
        idempotency_identity="playlist-ingest-v1:staging",
        submission_queue="default",
        source_kind="file",
        planned_item_id=None,
    )

    recorded = store.record_run_item_staging(
        "1",
        run.run_id,
        "occ-staging",
        attempt=1,
        batch_id="batch-staging",
        idempotency_identity="playlist-ingest-v1:staging",
        submission_lease_token=reserved.submission_lease_token,
        temp_dir="/tmp/media_ingest_job_staging",
    )

    assert recorded.staging_temp_dir == "/tmp/media_ingest_job_staging"
    assert (
        store.list_abandoned_run_item_staging(
            "1",
            older_than=NOW - timedelta(hours=1),
            limit=10,
        )
        == []
    )

    store.test_clock.advance(timedelta(days=8))
    candidates = store.list_abandoned_run_item_staging(
        "1",
        older_than=store.test_clock.current - timedelta(days=1),
        limit=1,
    )

    assert len(candidates) == 1
    assert candidates[0].staging_temp_dir == "/tmp/media_ingest_job_staging"
    with store._connection(owner_user_id="1", write=True) as db:
        store._query(
            db,
            """
            UPDATE media_ingest_run_items SET state = 'queued', job_id = 99
            WHERE owner_user_id = '1' AND run_id = ? AND occurrence_id = 'occ-staging'
            """,
            (run.run_id,),
        )
    assert (
        store.list_abandoned_run_item_staging(
            "1",
            older_than=store.test_clock.current,
            limit=10,
        )
        == []
    )


def test_file_job_binding_rejects_a_different_persisted_staging_directory(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run(
        "1",
        items=[
            _validated_direct_record(
                occurrence_id="occ-staging-mismatch",
                input_kind="file_stub",
                source_url=None,
                normalized_source_id=None,
                source_kind="file",
                state="awaiting_upload",
            )
        ],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-staging-mismatch",
        attempt=1,
        batch_id="batch-staging-mismatch",
        idempotency_identity="playlist-ingest-v1:staging-mismatch",
        submission_queue="default",
        source_kind="file",
        planned_item_id=None,
    )
    store.record_run_item_staging(
        "1",
        run.run_id,
        "occ-staging-mismatch",
        attempt=1,
        batch_id="batch-staging-mismatch",
        idempotency_identity="playlist-ingest-v1:staging-mismatch",
        submission_lease_token=reserved.submission_lease_token,
        temp_dir="/tmp/media_ingest_job_reserved",
    )
    job = store._jobs.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": run.run_id,
            "occurrence_id": "occ-staging-mismatch",
            "attempt": 1,
            "batch_id": "batch-staging-mismatch",
            "idempotency_key": "playlist-ingest-v1:staging-mismatch",
            "source": "/tmp/media_ingest_job_other/clip.mp3",
            "source_kind": "file",
            "temp_dir": "/tmp/media_ingest_job_other",
            "options": {"overwrite_existing": False},
        },
        batch_group="batch-staging-mismatch",
        owner_user_id="1",
        idempotency_key="playlist-ingest-v1:staging-mismatch",
        available_at=datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )

    with pytest.raises(PlaylistIngestConflictError, match="does not match occurrence"):
        store.bind_run_item_job(
            "1",
            run.run_id,
            "occ-staging-mismatch",
            attempt=1,
            job_id=int(job["id"]),
            batch_id="batch-staging-mismatch",
            idempotency_identity="playlist-ingest-v1:staging-mismatch",
            submission_lease_token=reserved.submission_lease_token,
        )


def test_live_staging_reference_check_is_owner_scoped_and_excludes_only_candidate(store):
    def file_item(occurrence_id: str) -> dict:
        return _validated_direct_record(
            occurrence_id=occurrence_id,
            input_kind="file_stub",
            source_url=None,
            normalized_source_id=None,
            source_kind="file",
            state="awaiting_upload",
        )

    expired_run = store.create_validated_run("1", items=[file_item("occ-expired")])
    expired_reserved = store.prepare_run_item_job_submission(
        "1",
        expired_run.run_id,
        "occ-expired",
        attempt=1,
        batch_id="batch-expired",
        idempotency_identity="playlist-ingest-v1:expired",
        submission_queue="default",
        source_kind="file",
        planned_item_id=None,
    )
    shared_path = "/tmp/media_ingest_job_shared_alias"
    store.record_run_item_staging(
        "1",
        expired_run.run_id,
        "occ-expired",
        attempt=1,
        batch_id="batch-expired",
        idempotency_identity="playlist-ingest-v1:expired",
        submission_lease_token=expired_reserved.submission_lease_token,
        temp_dir=shared_path,
    )
    store.test_clock.advance(timedelta(days=8))
    live_run = store.create_validated_run("1", items=[file_item("occ-live")])
    live_reserved = store.prepare_run_item_job_submission(
        "1",
        live_run.run_id,
        "occ-live",
        attempt=1,
        batch_id="batch-live",
        idempotency_identity="playlist-ingest-v1:live",
        submission_queue="default",
        source_kind="file",
        planned_item_id=None,
    )
    store.record_run_item_staging(
        "1",
        live_run.run_id,
        "occ-live",
        attempt=1,
        batch_id="batch-live",
        idempotency_identity="playlist-ingest-v1:live",
        submission_lease_token=live_reserved.submission_lease_token,
        temp_dir=shared_path,
    )

    assert store.has_live_run_item_staging_reference(
        "1",
        shared_path,
        excluding_run_id=expired_run.run_id,
        excluding_occurrence_id="occ-expired",
    )
    assert not store.has_live_run_item_staging_reference(
        "2",
        shared_path,
        excluding_run_id=expired_run.run_id,
        excluding_occurrence_id="occ-expired",
    )


@pytest.mark.parametrize(
    ("attempt", "error_type"),
    [(True, ValueError), (0, ValueError), (2, RuntimeError)],
)
def test_run_item_job_submission_rejects_nonexact_attempt_without_mutation(store, attempt, error_type):
    run = store.create_validated_run("1", items=[_validated_direct_record(occurrence_id="occ-job")])

    with pytest.raises(error_type, match="attempt"):
        store.prepare_run_item_job_submission(
            "1",
            run.run_id,
            "occ-job",
            attempt=attempt,
            batch_id="batch-1",
            idempotency_identity="playlist-ingest-v1:derived",
            source_kind="url",
            planned_item_id=None,
        )

    assert store.get_run_item("1", run.run_id, "occ-job").state == "staged"


def test_file_job_submission_failure_resets_exact_pending_reservation(store):
    run = store.create_validated_run(
        "1",
        items=[
            {
                "occurrence_id": "occ-file",
                "input_kind": "file_stub",
                "source_url": None,
                "normalized_source_id": None,
                "source_kind": "file",
                "display_metadata": {"name": "clip.mp3", "size_bytes": 4},
                "state": "awaiting_upload",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-file",
        attempt=1,
        batch_id="batch-file",
        idempotency_identity="playlist-ingest-v1:file-derived",
        source_kind="file",
        planned_item_id=None,
    )

    reset = store.reset_run_item_job_submission(
        "1",
        run.run_id,
        "occ-file",
        attempt=1,
        batch_id="batch-file",
        idempotency_identity="playlist-ingest-v1:file-derived",
        submission_lease_token=reserved.submission_lease_token,
    )

    assert reset.state == "awaiting_upload"
    assert reset.job_id is None
    assert reset.batch_id is None
    assert store.get_run("1", run.run_id).batch_ids == []


def test_pending_job_submission_reset_requires_exact_reservation_batch(store):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run("1", items=[_validated_direct_record(occurrence_id="occ-job")])
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-job",
        attempt=1,
        batch_id="batch-owner",
        idempotency_identity="playlist-ingest-v1:derived",
        source_kind="url",
        planned_item_id=None,
    )

    with pytest.raises(PlaylistIngestConflictError, match="reservation"):
        store.reset_run_item_job_submission(
            "1",
            run.run_id,
            "occ-job",
            attempt=1,
            batch_id="batch-other",
            idempotency_identity="playlist-ingest-v1:derived",
            submission_lease_token=reserved.submission_lease_token,
        )

    pending = store.get_run_item("1", run.run_id, "occ-job")
    assert pending.state == "submit_pending"
    assert pending.batch_id == "batch-owner"


def test_pending_file_submission_reset_requires_cleared_staging_pointer(store, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestConflictError,
    )

    run = store.create_validated_run(
        "1",
        items=[
            {
                "occurrence_id": "occ-file-reset",
                "input_kind": "file_stub",
                "source_url": None,
                "normalized_source_id": None,
                "source_kind": "file",
                "display_metadata": {"name": "clip.mp3", "size_bytes": 4},
                "state": "awaiting_upload",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    identity = "playlist-ingest-v1:file-reset"
    batch_id = "batch-file-reset"
    staging_dir = tmp_path / "staging"
    reserved = store.prepare_run_item_job_submission(
        "1",
        run.run_id,
        "occ-file-reset",
        attempt=1,
        batch_id=batch_id,
        idempotency_identity=identity,
        source_kind="file",
        planned_item_id=None,
    )
    store.record_run_item_staging(
        "1",
        run.run_id,
        "occ-file-reset",
        attempt=1,
        batch_id=batch_id,
        idempotency_identity=identity,
        submission_lease_token=reserved.submission_lease_token,
        temp_dir=str(staging_dir),
    )

    with pytest.raises(PlaylistIngestConflictError, match="reservation"):
        store.reset_run_item_job_submission(
            "1",
            run.run_id,
            "occ-file-reset",
            attempt=1,
            batch_id=batch_id,
            idempotency_identity=identity,
            submission_lease_token=reserved.submission_lease_token,
        )

    pending = store.get_run_item("1", run.run_id, "occ-file-reset")
    assert pending.state == "submit_pending"
    assert pending.staging_temp_dir == str(staging_dir)
