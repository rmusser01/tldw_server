from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _FixedClock:
    def now_utc(self) -> datetime:
        return NOW


def _new_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / f"property-{uuid4()}.db", clock=_FixedClock())
    return PlaylistIngestStore(manager)


@pytest.mark.property
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    occurrence_ids=st.lists(
        st.integers(min_value=0, max_value=1_000_000).map(lambda value: f"occ-{value}"),
        min_size=1,
        max_size=24,
        unique=True,
    ),
    page_size=st.integers(min_value=1, max_value=10),
)
def test_cursor_pages_reproduce_source_order_exactly_once(
    tmp_path,
    monkeypatch,
    occurrence_ids,
    page_size,
):
    store = _new_store(tmp_path, monkeypatch)
    preflight = store.create_preflight(
        "owner-1",
        source_url="https://example.com/playlist",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "owner-1",
        preflight.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": occurrence_id,
                "ordinal": ordinal,
                "occurrence_index_for_source": 1,
                "source_url": f"https://example.com/video/{ordinal}",
                "source_kind": "url",
                "availability": "available",
                "duplicate_status": "not_found",
                "display_metadata": {"title": occurrence_id},
            }
            for ordinal, occurrence_id in enumerate(occurrence_ids, start=1)
        ],
    )

    concatenated: list[str] = []
    cursor = None
    while True:
        page = store.list_preflight_items(
            "owner-1",
            preflight.preflight_id,
            limit=page_size,
            cursor=cursor,
        )
        concatenated.extend(item.occurrence_id for item in page)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert concatenated == occurrence_ids
    assert len(concatenated) == len(set(concatenated))


@pytest.mark.property
@settings(max_examples=12, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(other_owner=st.text(min_size=1, max_size=20).filter(lambda value: value.strip() not in {"", "owner-1"}))
def test_invalid_owner_cursor_pairs_use_the_same_not_found_contract(tmp_path, monkeypatch, other_owner):
    store = _new_store(tmp_path, monkeypatch)
    preflight = store.create_preflight(
        "owner-1",
        source_url="https://example.com/playlist",
        source_kind="playlist",
        expires_at=NOW + timedelta(hours=1),
    )
    store.replace_preflight_snapshot(
        "owner-1",
        preflight.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": f"occ-{ordinal}",
                "ordinal": ordinal,
                "occurrence_index_for_source": 1,
                "source_url": f"https://example.com/video/{ordinal}",
                "source_kind": "url",
                "availability": "available",
                "duplicate_status": "not_found",
            }
            for ordinal in range(1, 4)
        ],
    )
    cursor = store.list_preflight_items("owner-1", preflight.preflight_id, limit=1).next_cursor
    assert cursor is not None

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
    )

    messages = []
    for resource_id in (preflight.preflight_id, "resource-does-not-exist"):
        with pytest.raises(PlaylistIngestNotFoundError) as exc_info:
            store.list_preflight_items(other_owner, resource_id, limit=1, cursor=cursor)
        messages.append(str(exc_info.value))
    assert messages == ["playlist resource not found", "playlist resource not found"]


@pytest.mark.property
@settings(
    max_examples=12,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    occurrences=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1_000_000).map(lambda value: f"run-occ-{value}"),
            st.sampled_from(["ingest", "overwrite", "skip", "include_existing", "update_metadata_only"]),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda value: value[0],
    )
)
def test_validated_run_preserves_arbitrary_unique_order_and_actions(
    tmp_path,
    monkeypatch,
    occurrences,
):
    store = _new_store(tmp_path, monkeypatch)

    run = store.create_validated_run(
        "owner-1",
        items=[
            {
                "occurrence_id": occurrence_id,
                "input_kind": "direct_url",
                "source_url": f"https://example.com/{index}",
                "normalized_source_id": f"url:{index}",
                "source_kind": "generic_url",
                "display_metadata": {},
                "state": "staged",
                "action": action,
                "metadata_patch": {"title": "Reviewed"} if action == "update_metadata_only" else None,
            }
            for index, (occurrence_id, action) in enumerate(occurrences, start=1)
        ],
    )

    items = list(store.list_run_items("owner-1", run.run_id, limit=500))
    events = list(store.list_run_events("owner-1", run.run_id, limit=500))
    assert [(item.occurrence_id, item.action) for item in items] == occurrences
    assert [event.occurrence_id for event in events] == [value[0] for value in occurrences]
    assert [event.attrs["action"] for event in events] == [value[1] for value in occurrences]
    assert store.get_run("owner-1", run.run_id).version == 2


@pytest.mark.property
@settings(
    max_examples=12,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    owner_suffix=st.integers(min_value=1, max_value=1_000_000),
    occurrence_suffix=st.integers(min_value=1, max_value=1_000_000),
    repeats=st.integers(min_value=1, max_value=5),
)
def test_occurrence_attempt_accepts_at_most_one_job_under_repeated_submit(
    tmp_path,
    monkeypatch,
    owner_suffix,
    occurrence_suffix,
    repeats,
):
    store = _new_store(tmp_path, monkeypatch)
    owner = f"owner-{owner_suffix}"
    occurrence = f"occ-{occurrence_suffix}"
    run = store.create_validated_run(
        owner,
        items=[
            {
                "occurrence_id": occurrence,
                "input_kind": "direct_url",
                "source_url": "https://www.youtube.com/watch?v=alpha123456",
                "normalized_source_id": "youtube:video:alpha123456",
                "source_kind": "youtube_video",
                "display_metadata": {},
                "state": "staged",
                "action": "ingest",
                "metadata_patch": None,
            }
        ],
    )
    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import (
        _derive_run_occurrence_idempotency,
    )

    identity = _derive_run_occurrence_idempotency(
        key=b"property-test-key",
        owner_user_id=owner,
        run_id=run.run_id,
        occurrence_id=occurrence,
        attempt=1,
    )
    accepted_ids = []
    for request_index in range(repeats):
        reserved = store.prepare_run_item_job_submission(
            owner,
            run.run_id,
            occurrence,
            attempt=1,
            batch_id=f"batch-{request_index}",
            idempotency_identity=identity,
            source_kind="url",
            planned_item_id=None,
        )
        if reserved.job_id is None:
            job = store._jobs.create_job(
                domain="media_ingest",
                queue="default",
                job_type="media_ingest_item",
                payload={
                    "run_id": run.run_id,
                    "occurrence_id": occurrence,
                    "attempt": 1,
                    "batch_id": reserved.batch_id,
                    "idempotency_key": identity,
                    "source": reserved.source_url,
                    "source_kind": "url",
                },
                batch_group=reserved.batch_id,
                owner_user_id=owner,
                idempotency_key=identity,
            )
            reserved = store.bind_run_item_job(
                owner,
                run.run_id,
                occurrence,
                attempt=1,
                job_id=int(job["id"]),
                batch_id=str(reserved.batch_id),
                idempotency_identity=identity,
            )
        accepted_ids.append(reserved.job_id)

    jobs = store._jobs.list_jobs(domain="media_ingest", owner_user_id=owner, limit=10)
    assert len(jobs) == 1
    assert accepted_ids == [jobs[0]["id"]] * repeats
