"""TASK-13321: blob upload-session expiry was designed, schema'd and typed but never run.

The lifecycle exists on paper and nowhere else:

* `sync_blob_upload_sessions.expires_at` is a real column, written verbatim from
  `session.expires_at`, and `SyncBlobUploadSessionCreate.expires_at` defaults to `None` --
  set at **zero** call sites.
* The status `"expired"` is in the `SyncBlobUploadStatus` Literal and read as terminal in
  exactly one place. Written at **zero** call sites.
* `summarize_blob_quota` sums `WHERE status IN ('created','uploading')` with no expiry
  predicate, so an abandoned session counts against its user forever.

`max_active_blob_uploads` defaults to 8. A client that starts 8 uploads and then crashes or
loses network -- ordinary transient failure, not abuse -- permanently exhausts its own
limit, and every subsequent upload for that user+dataset fails with "Sync blob active
upload limit exceeded". Each abandoned session's `reserved_quota_bytes` also counts against
`user_blob_quota_bytes` forever, so abandoning one 1 GB upload permanently costs 1 GB.

Recovery today requires the client to have remembered all 8 `upload_id`s, or a manual DB
edit.

These tests pin expiry as a **read-time predicate** rather than a background reaper: an
expired session stops counting the moment it is past its deadline, with no sweeper needed to
make that true. Disk reclamation of staged chunks is deliberately out of scope here -- see
the task -- because orphaned bytes are a cost problem, while the lockout is a correctness
one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import (
    SyncBlobChunkCreate,
    SyncBlobUploadSessionCreate,
    SyncDatabase,
    SyncDatasetCreate,
    SyncDeviceUpsert,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    """A fresh SQLite-backed store per test."""
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2.db"))


@pytest.fixture()
def enrolled(sync_store: SyncV2Store) -> SyncV2Store:
    """A store with the one device and dataset these tests upload against."""
    sync_store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-1",
            user_id="user-1",
            display_name="Laptop",
            client_type="chatbook",
            client_version="0.1.0",
            capabilities={"domains": ["notes.note"]},
        )
    )
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["attachment.ref"],
            metadata={"label": "Expiry fixtures"},
        )
    )
    return sync_store


def _iso(moment: datetime) -> str:
    """Render an aware datetime as canonical UTC ISO text, as the store writes it."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


# Fixed instants far from any real clock, so the expired/live split cannot drift with
# system time or land on a boundary. The store compares against its own real "now", so
# these must bracket every plausible run date rather than sit an hour either side of it.
_PAST = "2000-01-01T00:00:00.000000+00:00"
_FUTURE = "2999-01-01T00:00:00.000000+00:00"


def _past() -> str:
    """A deadline that has certainly passed."""
    return _PAST


def _future() -> str:
    """A deadline that has certainly not passed."""
    return _FUTURE


def _session(**overrides: Any) -> SyncBlobUploadSessionCreate:
    """One upload session create payload, overridable per test."""
    payload = {
        "upload_id": "upload-1",
        "dataset_id": "dataset-1",
        "owner_user_id": "user-1",
        "device_id": "device-1",
        "attachment_id": "attachment-1",
        "domain": "attachment.ref",
        "object_id": "attachment-1",
        "content_type": "application/octet-stream",
        "size_bytes": 2048,
        "payload_hash": "sha256:" + "a" * 64,
        "chunk_size": 1024,
        "chunk_count": 2,
        "reserved_quota_bytes": 2048,
    }
    payload.update(overrides)
    return SyncBlobUploadSessionCreate(**payload)


# ---------------------------------------------------------------------------
# The quota predicate: an expired session must stop counting, with no reaper run.
# ---------------------------------------------------------------------------


def test_an_expired_session_stops_reserving_quota(enrolled: SyncV2Store) -> None:
    """The headline defect: abandoned uploads permanently consumed quota."""
    enrolled.create_blob_upload_session(_session(expires_at=_past()))

    quota = enrolled.summarize_blob_quota("user-1")

    assert quota.reserved_blob_bytes == 0, (
        "an expired upload session still reserves quota, so abandoning an upload costs "
        "its bytes permanently"
    )
    assert quota.active_upload_count == 0, (
        "an expired upload session still counts toward max_active_blob_uploads, so a "
        "client that crashes mid-upload locks itself out forever"
    )


def test_an_unexpired_session_still_reserves_quota(enrolled: SyncV2Store) -> None:
    """Control: the predicate must not release quota for live sessions."""
    enrolled.create_blob_upload_session(_session(expires_at=_future()))

    quota = enrolled.summarize_blob_quota("user-1")

    assert quota.reserved_blob_bytes == 2048
    assert quota.active_upload_count == 1


def test_a_session_with_no_deadline_still_reserves_quota(enrolled: SyncV2Store) -> None:
    """Control: `expires_at IS NULL` means no deadline, not already expired.

    Every session written before this change has a NULL here, so treating NULL as expired
    would silently free quota for live uploads on upgrade.
    """
    enrolled.create_blob_upload_session(_session())

    quota = enrolled.summarize_blob_quota("user-1")

    assert quota.reserved_blob_bytes == 2048
    assert quota.active_upload_count == 1


def test_the_dataset_scoped_quota_query_applies_the_same_predicate(
    enrolled: SyncV2Store,
) -> None:
    """summarize_blob_quota has two SQL branches; both need the predicate."""
    enrolled.create_blob_upload_session(_session(expires_at=_past()))

    scoped = enrolled.summarize_blob_quota("user-1", dataset_id="dataset-1")

    assert scoped.reserved_blob_bytes == 0, "the dataset-scoped branch ignores expiry"
    assert scoped.active_upload_count == 0, "the dataset-scoped branch ignores expiry"


# ---------------------------------------------------------------------------
# An expired session must not be usable, or its quota release is a double-spend.
# ---------------------------------------------------------------------------


def test_an_expired_session_cannot_accept_more_chunks(enrolled: SyncV2Store) -> None:
    """Releasing quota at read time is only sound if the session is also closed to writes.

    Otherwise a client could let a session expire -- releasing its reservation, letting
    another upload take the budget -- then resume it and exceed the quota.
    """
    session = enrolled.create_blob_upload_session(_session(expires_at=_past()))

    with pytest.raises(SyncStoreError, match="expired"):
        enrolled.record_blob_chunk(
            SyncBlobChunkCreate(
                upload_id=session.upload_id,
                dataset_id="dataset-1",
                chunk_index=0,
                offset_bytes=0,
                size_bytes=1024,
                chunk_hash="sha256:" + "b" * 64,
                storage_key=f"{session.upload_id}/0",
            )
        )


# ---------------------------------------------------------------------------
# The service must actually SET a deadline, or the predicate above never fires.
# ---------------------------------------------------------------------------


def test_the_service_stamps_a_deadline_on_new_sessions() -> None:
    """`expires_at` was set at zero call sites, so nothing could ever expire.

    The predicate in `summarize_blob_quota` is inert until the creating path stamps a
    deadline. This is the half that makes the fix reachable in production rather than only
    for a caller that passes `expires_at` by hand.
    """
    from tldw_Server_API.app.core.Sync.v2.models import blob_upload_expires_at

    deadline = blob_upload_expires_at(3600, now="2026-05-23T18:12:00+00:00")

    assert deadline is not None
    assert deadline > "2026-05-23T18:12:00.000000+00:00", (
        "a fresh session's deadline must be in the future or it expires immediately"
    )


def test_a_non_positive_ttl_means_no_deadline() -> None:
    """Control: an operator must be able to turn expiry off explicitly.

    Returning a past timestamp for ttl<=0 would expire every session the moment it is
    created, which is a far worse failure than the one this task fixes.
    """
    from tldw_Server_API.app.core.Sync.v2.models import blob_upload_expires_at

    assert blob_upload_expires_at(0) is None
    assert blob_upload_expires_at(-1) is None


def test_the_expiry_check_treats_null_as_live() -> None:
    """Control, stated directly: NULL is "no deadline", never "already expired"."""
    from tldw_Server_API.app.core.Sync.v2.models import (
        blob_upload_session_is_expired,
    )

    assert blob_upload_session_is_expired(None) is False
    assert blob_upload_session_is_expired("") is False
    assert blob_upload_session_is_expired(_past()) is True
    assert blob_upload_session_is_expired(_future()) is False


def test_the_default_ttl_enables_expiry() -> None:
    """A zero default would leave the predicate inert and the defect unfixed.

    `expires_at` was written at zero call sites, so the whole lifecycle was dead code. The
    setting has to default to something positive for the creation path to stamp a deadline
    without an operator opting in.
    """
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Settings

    assert SyncV2Settings().blob_upload_session_ttl_seconds > 0


def test_the_deadline_the_service_composes_is_exact() -> None:
    """Pin the composition `create_blob_upload_session` performs, deterministically.

    The service calls `blob_upload_expires_at(ttl, now=self.clock())`, and `clock` is
    injectable, so this is the same arithmetic with no wall-clock dependence.
    """
    from tldw_Server_API.app.core.Sync.v2.models import blob_upload_expires_at

    assert (
        blob_upload_expires_at(3600, now="2026-05-23T18:12:00+00:00")
        == "2026-05-23T19:12:00.000000+00:00"
    )
    assert (
        blob_upload_expires_at(86_400, now="2026-05-23T18:12:00+00:00")
        == "2026-05-24T18:12:00.000000+00:00"
    )


# ---------------------------------------------------------------------------
# AC#1: the user-visible scenario. Eight abandoned uploads must not be a life sentence.
# ---------------------------------------------------------------------------


def test_eight_abandoned_uploads_do_not_lock_the_user_out(enrolled: SyncV2Store) -> None:
    """The reported defect, end to end through the limit check.

    `max_active_blob_uploads` defaults to 8. A client that starts 8 uploads and then
    crashes or loses network used to exhaust its own limit permanently: every later
    attachment upload for that user+dataset raised "Sync blob active upload limit
    exceeded", forever, because the quota query counted `created`/`uploading` rows with no
    expiry predicate. Recovery needed the client to have remembered all 8 upload_ids, or a
    manual DB edit.
    """
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings

    for index in range(8):
        enrolled.create_blob_upload_session(
            _session(
                upload_id=f"abandoned-{index}",
                attachment_id=f"attachment-{index}",
                object_id=f"attachment-{index}",
                expires_at=_past(),
            )
        )

    # Before the ninth attempt, the limit check must see no live uploads.
    service = SyncV2Service.__new__(SyncV2Service)
    service.store = enrolled
    service.settings = SyncV2Settings()

    quota = enrolled.summarize_blob_quota("user-1", dataset_id="dataset-1")
    assert quota.active_upload_count == 0, (
        f"{quota.active_upload_count} abandoned sessions still count as active, so the "
        "ninth upload is refused and the user is locked out permanently"
    )

    # The ninth upload's limit check passes.
    service._validate_blob_limits(
        user_id="user-1",
        dataset_id="dataset-1",
        size_bytes=2048,
        chunk_size=1024,
        chunk_count=2,
    )


def test_eight_live_uploads_still_hit_the_limit(enrolled: SyncV2Store) -> None:
    """Control: the limit must still bite when the sessions really are live.

    Otherwise this change would have removed the protection rather than fixed its
    accounting.
    """
    from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncStoreError
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings

    for index in range(8):
        enrolled.create_blob_upload_session(
            _session(
                upload_id=f"live-{index}",
                attachment_id=f"attachment-{index}",
                object_id=f"attachment-{index}",
                expires_at=_future(),
            )
        )

    service = SyncV2Service.__new__(SyncV2Service)
    service.store = enrolled
    service.settings = SyncV2Settings()

    with pytest.raises(SyncStoreError, match="active upload limit"):
        service._validate_blob_limits(
            user_id="user-1",
            dataset_id="dataset-1",
            size_bytes=2048,
            chunk_size=1024,
            chunk_count=2,
        )


def test_the_service_create_path_stamps_the_deadline(enrolled: SyncV2Store) -> None:
    """The regression this task actually found: expires_at set at zero call sites.

    Deliberately exercises `SyncV2Service.create_blob_upload_session` rather than the
    helper it calls. Verified necessary by probing: with only the helper under test,
    deleting the `expires_at=` argument from the create path left every other test in this
    module passing -- the same false assurance that let the whole lifecycle sit unwired.
    """
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings

    service = SyncV2Service.__new__(SyncV2Service)
    service.store = enrolled
    service.settings = SyncV2Settings(
        supports_attachments=True,
        blob_upload_session_ttl_seconds=3600,
    )
    service.clock = lambda: "2026-05-23T18:12:00+00:00"
    service.id_factory = lambda prefix: f"{prefix}-stamped"
    service._require_blob_transfer = lambda: None
    service._require_blob_dataset = lambda **kwargs: None
    service._validate_blob_limits = lambda **kwargs: None
    # Stubbed so the assertion stays about expires_at, not the notes-intent contract.
    service._normalize_notes_attachment_upload_metadata = (
        lambda **kwargs: dict(kwargs.get("metadata") or {})
    )

    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id=None,
        domain="attachment.ref",
        entity_id="entity-1",
        attachment_id="attachment-stamped",
        content_type="application/octet-stream",
        size_bytes=2048,
        payload_hash="sha256:" + "c" * 64,
        chunk_size=1024,
        chunk_count=2,
    )

    stored = enrolled.get_blob_upload_session(session.upload_id)
    assert stored is not None
    assert datetime.fromisoformat(stored.expires_at) == datetime(2026, 5, 23, 19, 12, tzinfo=timezone.utc), (
        f"the create path stored expires_at={stored.expires_at!r}; a session with no "
        "deadline never expires, so its quota and upload slot are held forever"
    )


def test_the_ttl_is_configurable_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC#2 asks for a configurable setting, so pin that the env var is wired.

    Zero must be accepted, not rejected: it is how an operator turns expiry off on purpose.
    `_sync_v2_positive_int_env` cannot express that, which is why this uses a non-negative
    reader.
    """
    from tldw_Server_API.app.core.Sync.v2.factory import (
        _sync_v2_non_negative_int_env,
    )

    monkeypatch.setenv("SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", "600")
    assert _sync_v2_non_negative_int_env(
        "SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", default=86_400
    ) == 600

    monkeypatch.setenv("SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", "0")
    assert _sync_v2_non_negative_int_env(
        "SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", default=86_400
    ) == 0

    monkeypatch.delenv("SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", raising=False)
    assert _sync_v2_non_negative_int_env(
        "SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", default=86_400
    ) == 86_400

    monkeypatch.setenv("SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _sync_v2_non_negative_int_env(
            "SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS", default=86_400
        )


# ---------------------------------------------------------------------------
# Qodo review on #3006: three real defects in the first cut.
# ---------------------------------------------------------------------------


def test_an_expired_session_cannot_be_completed(enrolled: SyncV2Store) -> None:
    """Completion must honour the deadline, or the quota can be exceeded.

    summarize_blob_quota stops counting a session's reservation at its deadline, so a
    replacement upload may already hold that allowance. Committing the late blob would
    push committed usage past the quota.
    """
    from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncBlobObjectCreate

    session = enrolled.create_blob_upload_session(_session(expires_at=_past()))

    with pytest.raises(SyncStoreError, match="expired"):
        enrolled.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id="blob-late",
                dataset_id="dataset-1",
                owner_user_id="user-1",
                attachment_id=session.attachment_id,
                payload_hash=session.payload_hash,
                content_type=session.content_type,
                size_bytes=session.size_bytes,
                encryption_policy="server_trusted_v1",
                storage_backend="local_fs",
                storage_key="blobs/late",
                status="available",
            )
        )


@pytest.mark.parametrize(
    "clock_value",
    [
        "2026-05-23T18:12:00+00:00",
        "2026-05-23T20:12:00+02:00",
        "2026-05-23T13:12:00-05:00",
        "2026-05-23T18:12:00",  # naive: taken as UTC
    ],
)
def test_the_deadline_is_canonical_utc_whatever_the_clock_offset(clock_value: str) -> None:
    """Equivalent instants in any offset must yield the identical UTC deadline.

    The SQL predicate compares expires_at lexicographically with UTC text, so an
    offset-carrying deadline would release the reservation at the wrong instant.
    """
    from tldw_Server_API.app.core.Sync.v2.models import blob_upload_expires_at

    assert (
        blob_upload_expires_at(3600, now=clock_value)
        == "2026-05-23T19:12:00.000000+00:00"
    )


def test_expiry_compares_instants_not_text() -> None:
    """A +02:00 deadline one second ahead in real time is not yet expired."""
    from tldw_Server_API.app.core.Sync.v2.models import blob_upload_session_is_expired

    assert blob_upload_session_is_expired(
        "2026-05-23T20:12:01+02:00", now="2026-05-23T18:12:00+00:00"
    ) is False
    assert blob_upload_session_is_expired(
        "2026-05-23T20:11:59+02:00", now="2026-05-23T18:12:00+00:00"
    ) is True


class _RecordingBlobStore:
    """Blob-store double that records writes and discards."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []
        self.discarded: list[str] = []

    def write_upload_chunk(self, *, upload_id: str, chunk_index: int, payload: bytes,
                           expected_hash: str) -> str:
        """Record the write and return a storage key."""
        self.writes.append((upload_id, chunk_index))
        return f"_uploads/{upload_id}/{chunk_index}"

    def discard_upload(self, upload_id: str) -> None:
        """Record the discard."""
        self.discarded.append(upload_id)


def test_an_expired_chunk_upload_writes_nothing_to_disk(enrolled: SyncV2Store) -> None:
    """Refuse before touching disk, so an expired session leaves no staged bytes."""
    import hashlib

    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings

    payload = b"x" * 1024
    session = enrolled.create_blob_upload_session(_session(expires_at=_past()))
    store = _RecordingBlobStore()

    service = SyncV2Service.__new__(SyncV2Service)
    service.store = enrolled
    service.settings = SyncV2Settings(supports_attachments=True)
    service.clock = lambda: "2026-05-23T18:12:00+00:00"
    service._require_blob_transfer = lambda: store
    service.get_blob_upload_session = lambda **kwargs: session

    with pytest.raises(SyncStoreError, match="expired"):
        service.upload_blob_chunk(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=session.upload_id,
            chunk_index=0,
            offset_bytes=0,
            chunk_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
            chunk_payload=payload,
        )

    assert store.writes == [], "an expired session still wrote a chunk to disk"
    assert store.discarded == [session.upload_id], (
        "the expired session's staged chunks were not discarded"
    )
