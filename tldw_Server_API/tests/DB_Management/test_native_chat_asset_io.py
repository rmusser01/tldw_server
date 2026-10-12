"""Private native candidate bytes obey the original durable reservation fence."""

from __future__ import annotations

import hashlib
import importlib
import multiprocessing
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

pytestmark = pytest.mark.integration
PAYLOAD = b"native retained bytes"
NAMESPACE = "a" * 32


@pytest.fixture(params=["sqlite", "postgres"])
def native_fork_db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CharactersRAGDB]:
    kwargs: dict[str, Any] = {
        "db_path": str(tmp_path / "operations.sqlite"),
        "client_id": "1",
        "owner_user_id": "1",
    }
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(**kwargs)
    yield db
    db.close_connection()
    if request.param == "postgres":
        kwargs["backend"].get_pool().close_all()


def owner() -> AuthorizedNativeOwner:
    return AuthorizedNativeOwner("1", "native:1", NativeScopeV1(kind="global"))


def reserve(db: CharactersRAGDB, *, operation_id: str) -> None:
    with db.transaction() as conn:
        db.native_forks.reserve_operation(
            owner(), "native_fork_v1", operation_id, "sha256:one", {"title": "A"}, conn=conn
        )


def _reserve(
    db: CharactersRAGDB,
    *,
    candidate: str = "candidate-one",
    namespace: str = NAMESPACE,
    upload: str = "upload-one",
    operation: str = "fork-one",
    payload: bytes = PAYLOAD,
    confirmed: bool = True,
    mime_type: str = "text/plain",
    representation: str = "file_v1",
) -> int:
    reserve(db, operation_id=operation)
    now = datetime.now(timezone.utc)
    with db.transaction() as conn:
        attempt = db.native_forks.claim_attempt(owner(), "native_fork_v1", operation, "sha256:one", now=now, conn=conn)
        db.native_assets.reserve_candidate(
            owner(),
            "native_fork_v1",
            operation,
            "sha256:one",
            attempt.generation,
            candidate,
            namespace,
            upload,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            mime_type,
            representation,
            now=now,
            conn=conn,
        )
        if confirmed:
            conn.execute(
                "INSERT INTO native_chat_quota_intents "
                "(client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at) "
                "VALUES (?,?, 'reserve',?,?, 'confirmed',?,?)",
                ("1", candidate, owner().owner_key, len(payload), now.isoformat(), now.isoformat()),
            )
    return attempt.generation


def _io(db: CharactersRAGDB, root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    try:
        module = importlib.import_module("tldw_Server_API.app.core.Chat.native_chat_assets")
    except ModuleNotFoundError:
        pytest.fail("private native candidate writer/reclamation adapter is missing")
    monkeypatch.setattr(DatabasePaths, "resolve_user_base_directory", staticmethod(lambda user_id: root))
    return module.NativeChatAssetIO(db, user_id=1)


def _chunk(io: Any, generation: int, *, payload: bytes = PAYLOAD, index: int = 0) -> str:
    return io.write_chunk(
        owner(),
        "native_fork_v1",
        "fork-one",
        "sha256:one",
        generation,
        "candidate-one",
        chunk_index=index,
        payload=payload,
        expected_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def _publish(io: Any, generation: int, indexes: list[int] | None = None) -> str:
    return io.publish(
        owner(),
        "native_fork_v1",
        "fork-one",
        "sha256:one",
        generation,
        "candidate-one",
        chunk_indexes=[0] if indexes is None else indexes,
    )


def _expire(db: CharactersRAGDB) -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), "fork-one"),
        )


def _confirm_release(db: CharactersRAGDB) -> None:
    with db.transaction() as conn:
        changed = conn.execute(
            "UPDATE native_chat_quota_intents SET state = 'confirmed' "
            "WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'",
            ("1", "candidate-one"),
        ).rowcount
        assert changed == 1


def _candidate(db: CharactersRAGDB) -> dict[str, Any]:
    with db.transaction() as conn:
        return dict(
            conn.execute(
                "SELECT * FROM native_chat_asset_candidates WHERE client_id = ? AND candidate_id = ?",
                ("1", "candidate-one"),
            ).fetchone()
        )


@pytest.mark.parametrize("alias", ["namespace", "upload"])
def test_candidate_paths_cannot_be_reserved_by_another_operation(native_fork_db: CharactersRAGDB, alias: str) -> None:
    db = native_fork_db
    _reserve(db)
    with pytest.raises(ValueError, match="candidate_storage_identity_conflict"):
        _reserve(
            db,
            candidate="candidate-other",
            operation="fork-other",
            namespace=NAMESPACE if alias == "namespace" else "b" * 32,
            upload="upload-one" if alias == "upload" else "upload-other",
        )


def test_verified_publication_leaves_mime_admission_reserved(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    key = _publish(io, generation)
    assert (tmp_path / "chat_assets" / key).read_bytes() == PAYLOAD
    assert _candidate(db)["state"] == "reserved"
    assert _candidate(db)["storage_key"] is None


@pytest.mark.parametrize("reservation", ["missing", "pending", "wrong_owner", "wrong_size", "release_started"])
def test_unconfirmed_or_mismatched_quota_refuses_payload_bytes(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reservation: str
) -> None:
    db = native_fork_db
    generation = _reserve(db, confirmed=reservation != "missing")
    if reservation != "missing":
        with db.transaction() as conn:
            if reservation == "pending":
                conn.execute("UPDATE native_chat_quota_intents SET state = 'pending'")
            elif reservation == "wrong_owner":
                conn.execute("UPDATE native_chat_quota_intents SET owner_key = 'other'")
            elif reservation == "wrong_size":
                conn.execute("UPDATE native_chat_quota_intents SET size_bytes = size_bytes + 1")
            else:
                stamp = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO native_chat_quota_intents "
                    "(client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at) "
                    "VALUES (?,?, 'release',?,?, 'pending',?,?)",
                    ("1", "candidate-one", owner().owner_key, len(PAYLOAD), stamp, stamp),
                )
    io = _io(db, tmp_path, monkeypatch)
    with pytest.raises(
        ValueError, match="quota_reservation_unconfirmed|quota_intent_mismatch|candidate_release_started"
    ):
        _chunk(io, generation)
    assert not (tmp_path / "chat_assets" / "_uploads").exists()
    assert not (tmp_path / "chat_assets" / "blobs").exists()


@pytest.mark.parametrize("action", ["chunk", "publish", "reclaim"])
def test_outer_transaction_refuses_before_filesystem_side_effects(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    with db.transaction():
        with pytest.raises(ValueError, match="native_assets_outermost_required"):
            if action == "chunk":
                _chunk(io, generation)
            elif action == "publish":
                _publish(io, generation)
            else:
                io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    assert not (tmp_path / "chat_assets").exists()


def test_reclamation_removes_unacknowledged_final_and_all_crash_residue(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    key = _publish(io, generation)
    root = tmp_path / "chat_assets"
    namespace_path = root / "blobs" / "v2" / NAMESPACE
    (namespace_path / "assembly.crash.tmp").write_bytes(b"partial assembly")
    upload_path = root / "_uploads" / "upload-one"
    (upload_path / "0.crash.tmp").write_bytes(b"partial chunk")
    other = root / "blobs" / "v2" / ("b" * 32)
    other.mkdir()
    (other / "keep.blob").write_bytes(b"other candidate")
    _expire(db)
    with pytest.raises(ValueError, match="quota_release_unconfirmed"):
        io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    assert not (root / key).exists()
    assert not namespace_path.exists()
    assert not upload_path.exists()
    assert (other / "keep.blob").read_bytes() == b"other candidate"
    assert _candidate(db)["state"] == "reclaiming"
    with db.transaction() as conn:
        release = conn.execute(
            "SELECT state,size_bytes FROM native_chat_quota_intents "
            "WHERE candidate_id = ? AND intent_kind = 'release'",
            ("candidate-one",),
        ).fetchone()
        assert dict(release) == {"state": "pending", "size_bytes": len(PAYLOAD)}
    _confirm_release(db)
    assert io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one") is True
    assert _candidate(db)["state"] == "discarded"
    assert io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one") is False


@pytest.mark.parametrize("payload_state", ["wrong_hash", "wrong_size"])
def test_existing_final_content_is_verified_before_publication(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload_state: str
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore, SyncBlobStoreError

    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    key = LocalSyncBlobStore.namespace_storage_key(NAMESPACE, "sha256:" + hashlib.sha256(PAYLOAD).hexdigest())
    target = tmp_path / "chat_assets" / key
    target.parent.mkdir(parents=True)
    unexpected = b"x" * len(PAYLOAD) if payload_state == "wrong_hash" else b"bad"
    target.write_bytes(unexpected)
    with pytest.raises(SyncBlobStoreError, match="verification|size|hash|digest"):
        _publish(io, generation)
    assert target.read_bytes() == unexpected
    assert _candidate(db)["state"] == "reserved"


def test_missing_native_link_capability_refuses_payload_io(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError

    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    monkeypatch.setattr(
        os, "supports_dir_fd", {function for function in os.supports_dir_fd if function.__name__ != "link"}
    )
    with pytest.raises(SyncBlobStoreError, match="unavailable|unsupported"):
        _chunk(io, generation)
    assert not (tmp_path / "chat_assets").exists()


def test_root_change_before_removal_ack_keeps_quota_unreleased(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError

    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    _publish(io, generation)
    _expire(db)
    original = io._remove_directory

    def remove_then_swap(root: int, parents: tuple[str, ...], name: str) -> None:
        original(root, parents, name)
        if parents == ("_uploads",):
            (tmp_path / "chat_assets").rename(tmp_path / "retired-assets")
            (tmp_path / "chat_assets").mkdir()
            (tmp_path / "chat_assets" / "keep").write_bytes(b"replacement-root")

    monkeypatch.setattr(io, "_remove_directory", remove_then_swap)
    with pytest.raises(SyncBlobStoreError, match="root identity changed"):
        io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone() is None
    assert (tmp_path / "chat_assets" / "keep").read_bytes() == b"replacement-root"
    assert _candidate(db)["state"] == "reclaiming"


def test_zero_byte_candidate_refuses_publication_but_can_be_reclaimed(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = native_fork_db
    generation = _reserve(db, payload=b"")
    io = _io(db, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unsupported_zero_size_native_asset"):
        _chunk(io, generation, payload=b"")
    _expire(db)
    with pytest.raises(ValueError, match="quota_release_unconfirmed"):
        io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    _confirm_release(db)
    assert io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one") is True


@pytest.mark.parametrize("stage", ["chunk", "final"])
def test_reclamation_drains_publisher_and_refuses_waiting_stale_writer(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    entered, release, reclaiming = threading.Event(), threading.Event(), threading.Event()
    method = "_write_native_chunk" if stage == "chunk" else "_commit_native_upload"
    original = getattr(LocalSyncBlobStore, method)
    if stage == "final":
        _chunk(io, generation)

    def paused(**kwargs: Any) -> str:
        connection = db.get_connection()
        if hasattr(connection, "in_transaction"):
            assert not connection.in_transaction
        else:
            assert connection._connection.info.transaction_status.name == "IDLE"
            assert db._connection_state().tx_depth == 0
        entered.set()
        assert release.wait(8), "publisher was not released"
        return original(**kwargs)

    monkeypatch.setattr(LocalSyncBlobStore, method, staticmethod(paused))
    original_claim = db.native_assets.claim_reclamation

    def claimed(*args: Any, **kwargs: Any) -> Any:
        result = original_claim(*args, **kwargs)
        if result is not None:
            reclaiming.set()
        return result

    monkeypatch.setattr(db.native_assets, "claim_reclamation", claimed)

    def run(action: str) -> Any:
        try:
            if action == "publisher":
                return _chunk(io, generation) if stage == "chunk" else _publish(io, generation)
            if action == "waiting":
                return _chunk(io, generation)
            return io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=3) as workers:
        publisher = workers.submit(run, "publisher")
        try:
            assert entered.wait(5)
            _expire(db)
            cleanup = workers.submit(run, "cleanup")
            assert reclaiming.wait(5)
            assert _candidate(db)["state"] == "reclaiming"
            assert not cleanup.done()
            waiting = workers.submit(run, "waiting")
            release.set()
            assert publisher.result(timeout=8)
            with pytest.raises(ValueError, match="quota_release_unconfirmed"):
                cleanup.result(timeout=8)
            with pytest.raises(ValueError, match="stale_attempt|candidate_state_conflict"):
                waiting.result(timeout=8)
        finally:
            release.set()
    root = tmp_path / "chat_assets"
    assert not (root / "_uploads" / "upload-one").exists()
    assert not (root / "blobs" / "v2" / NAMESPACE).exists()
    _confirm_release(db)
    assert io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one") is True
    with db.transaction() as conn:
        replacement = db.native_forks.claim_attempt(
            owner(), "native_fork_v1", "fork-one", "sha256:one", now=datetime.now(timezone.utc), conn=conn
        )
        assert replacement.generation == generation + 1
    with pytest.raises(ValueError, match="stale_attempt"):
        _chunk(io, generation)
    assert not (root / "_uploads" / "upload-one").exists()


@pytest.mark.parametrize("index,payload", [(True, PAYLOAD), (-1, PAYLOAD), (1, PAYLOAD), (0, b"short")])
def test_chunk_layout_never_exceeds_reserved_bytes(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: Any, payload: bytes
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError

    generation = _reserve(native_fork_db)
    io = _io(native_fork_db, tmp_path, monkeypatch)
    with pytest.raises(SyncBlobStoreError, match="layout"):
        _chunk(io, generation, index=index, payload=payload)
    assert not (tmp_path / "chat_assets" / "_uploads").exists()


@pytest.mark.parametrize("indexes", [[True], [-1], [1], [0, 0], []])
def test_final_layout_is_exact_before_any_blob_directory(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indexes: list[int]
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError

    generation = _reserve(native_fork_db)
    io = _io(native_fork_db, tmp_path, monkeypatch)
    _chunk(io, generation)
    with pytest.raises(SyncBlobStoreError, match="layout"):
        _publish(io, generation, indexes)
    assert not (tmp_path / "chat_assets" / "blobs").exists()


@pytest.mark.parametrize("state", ["prepared", "adopted", "reclaiming", "discarded"])
def test_nonreserved_candidates_refuse_writes(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    with db.transaction() as conn:
        conn.execute("UPDATE native_chat_asset_candidates SET state = ?", (state,))
    with pytest.raises(ValueError, match="candidate_state_conflict"):
        _chunk(io, generation)
    assert not (tmp_path / "chat_assets" / "_uploads").exists()


def test_discarded_paths_remain_bound_to_the_original_candidate(native_fork_db: CharactersRAGDB) -> None:
    db = native_fork_db
    generation = _reserve(db, confirmed=False)
    _expire(db)
    with db.transaction() as conn:
        db.native_assets.claim_reclamation(
            owner(),
            "native_fork_v1",
            "fork-one",
            "sha256:one",
            "candidate-one",
            now=datetime.now(timezone.utc),
            conn=conn,
        )
        assert db.native_assets.finish_reclamation(
            owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one", generation, conn=conn
        )
    with pytest.raises(ValueError, match="candidate_storage_identity_conflict"):
        _reserve(db, candidate="other", operation="fork-other")


def test_concurrent_operations_cannot_share_candidate_paths(native_fork_db: CharactersRAGDB) -> None:
    db = native_fork_db
    barrier = threading.Barrier(2)

    def reserve_one(index: int) -> str:
        try:
            barrier.wait(timeout=5)
            _reserve(db, candidate=f"candidate-{index}", operation=f"fork-{index}")
            return "reserved"
        except ValueError as exc:
            return str(exc)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(reserve_one, [1, 2]))
    assert sorted(results) == ["candidate_storage_identity_conflict", "reserved"]


@pytest.mark.parametrize("path", ["chat_assets", "_uploads", "namespace", "guard"])
def test_symlinked_roots_and_candidate_paths_fail_without_touching_external_files(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import SyncBlobStoreError

    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    external = tmp_path / "outside"
    external.mkdir()
    (external / "keep").write_bytes(b"external marker")
    root = tmp_path / "chat_assets"
    if path == "chat_assets":
        root.symlink_to(external, target_is_directory=True)
    else:
        root.mkdir()
        if path == "_uploads":
            (root / "_uploads").symlink_to(external, target_is_directory=True)
        elif path == "namespace":
            _chunk(io, generation)
            parent = root / "blobs" / "v2"
            parent.mkdir(parents=True)
            (parent / NAMESPACE).symlink_to(external, target_is_directory=True)
        else:
            parent = root / "_locks" / "native-candidates"
            parent.mkdir(parents=True)
            name = hashlib.sha256(b'["1","candidate-one"]').hexdigest() + ".lock"
            (parent / name).symlink_to(external / "keep")
    with pytest.raises((SyncBlobStoreError, OSError)):
        _publish(io, generation) if path == "namespace" else _chunk(io, generation)
    assert (external / "keep").read_bytes() == b"external marker"
    assert sorted(p.name for p in external.iterdir()) == ["keep"]


def test_removal_failure_never_records_release_and_retries_exactly(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    key = _publish(io, generation)
    _expire(db)
    original = io._remove_directory

    def fail_remove(*args: Any) -> None:
        raise OSError("simulated removal failure")

    monkeypatch.setattr(io, "_remove_directory", fail_remove)
    with pytest.raises(OSError, match="simulated removal"):
        io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    assert (tmp_path / "chat_assets" / key).read_bytes() == PAYLOAD
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone() is None
    monkeypatch.setattr(io, "_remove_directory", original)
    with pytest.raises(ValueError, match="quota_release_unconfirmed"):
        io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    _confirm_release(db)
    assert io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one") is True


def test_reclaim_descriptor_changed_while_waiting_cannot_ack_removal(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.Chat.native_chat_assets")
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    key = _publish(io, generation)
    _expire(db)
    original_lock = module._lock_file_at

    @contextmanager
    def changed_lock(*args: Any, **kwargs: Any) -> Any:
        with original_lock(*args, **kwargs) as locked:
            with db.transaction() as conn:
                conn.execute("UPDATE native_chat_asset_candidates SET storage_namespace_id = ?", ("b" * 32,))
            yield locked

    monkeypatch.setattr(module, "_lock_file_at", changed_lock)
    with pytest.raises(ValueError, match="stale_reclamation"):
        io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
    assert (tmp_path / "chat_assets" / key).read_bytes() == PAYLOAD
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone() is None


def _child_physical_publisher(
    db_path: str, root: str, generation: int, stage: str, after_publish: bool, pg_config: Any, channel: Any
) -> None:
    from tldw_Server_API.app.core.Chat.native_chat_assets import NativeChatAssetIO
    from tldw_Server_API.app.core.Sync.v2 import blob_store

    kwargs: dict[str, Any] = {"db_path": db_path, "client_id": "1", "owner_user_id": "1"}
    if pg_config is not None:
        kwargs["backend"] = DatabaseBackendFactory.create_backend(pg_config)
    db = CharactersRAGDB(**kwargs)
    DatabasePaths.resolve_user_base_directory = staticmethod(lambda user_id: Path(root))
    io = NativeChatAssetIO(db, user_id=1)
    channel.send(("ready", None))
    original = blob_store._publish_native_file_at

    def paused(handle: Any, **kwargs: Any) -> None:
        selected = kwargs["name"].endswith(".blob") if stage == "final" else kwargs["name"].endswith(".part")
        if selected:
            if after_publish:
                original(handle, **kwargs)
            channel.send(("paused", kwargs["temporary"]))
            assert channel.recv() == "publish"
            if not after_publish:
                original(handle, **kwargs)
        else:
            original(handle, **kwargs)

    blob_store._publish_native_file_at = paused
    try:
        result = _publish(io, generation) if stage == "final" else _chunk(io, generation)
        channel.send(("completed", result))
    finally:
        db.close_connection()
        channel.close()


@pytest.mark.parametrize("stage,outcome", [("chunk", "drain"), ("final", "drain"), ("chunk", "die"), ("final", "die")])
def test_cross_process_publication_drain_and_death_preserve_the_quota_fence(
    native_fork_db: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    outcome: str,
    request: pytest.FixtureRequest,
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    if stage == "final":
        _chunk(io, generation)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    pg_config = request.getfixturevalue("pg_database_config") if db.backend_type == BackendType.POSTGRESQL else None
    process = context.Process(
        target=_child_physical_publisher,
        args=(
            db.db_path_str,
            str(tmp_path),
            generation,
            stage,
            outcome == "die" and stage == "final",
            pg_config,
            child,
        ),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(120), "child failed to initialize"
        assert parent.recv()[0] == "ready"
        assert parent.poll(15), "child failed to reach actual publication boundary"
        assert parent.recv()[0] == "paused"
        _expire(db)
        claimed = threading.Event()
        original_claim = db.native_assets.claim_reclamation

        def claim(*args: Any, **kwargs: Any) -> Any:
            candidate = original_claim(*args, **kwargs)
            claimed.set()
            return candidate

        monkeypatch.setattr(db.native_assets, "claim_reclamation", claim)

        def cleanup() -> bool:
            try:
                return io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
            finally:
                db.close_connection()

        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(cleanup)
            assert claimed.wait(5)
            assert _candidate(db)["state"] == "reclaiming"
            assert not future.done()
            if outcome == "die":
                process.terminate()
            else:
                parent.send("publish")
                assert parent.poll(8)
                assert parent.recv()[0] == "completed"
            process.join(8)
            assert not process.is_alive()
            if outcome == "drain":
                assert process.exitcode == 0
            with pytest.raises(ValueError, match="quota_release_unconfirmed"):
                future.result(timeout=8)
        root = tmp_path / "chat_assets"
        assert not (root / "_uploads" / "upload-one").exists()
        assert not (root / "blobs" / "v2" / NAMESPACE).exists()
        assert _candidate(db)["state"] == "reclaiming"
        _confirm_release(db)
        assert io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one") is True
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        parent.close()


def test_native_storage_cannot_select_another_users_root(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.Chat.native_chat_assets")
    resolved: list[int] = []

    def resolve(user_id: int) -> Path:
        resolved.append(user_id)
        return tmp_path

    monkeypatch.setattr(DatabasePaths, "resolve_user_base_directory", staticmethod(resolve))
    with pytest.raises(ValueError, match="native_asset_owner_mismatch"):
        module.NativeChatAssetIO(native_fork_db, user_id=2)
    assert resolved == []


@pytest.mark.parametrize("stage", ["chunk", "final", "reclaim"])
def test_named_user_directory_replacement_refuses_ack_and_quota_release(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from tldw_Server_API.app.core.Sync.v2 import blob_store

    db = native_fork_db
    generation = _reserve(db)
    physical = tmp_path / "user-1"
    physical.mkdir()
    io = _io(db, physical, monkeypatch)
    if stage != "chunk":
        _chunk(io, generation)
    if stage == "reclaim":
        _publish(io, generation)
        _expire(db)

    def replace_user() -> None:
        physical.rename(tmp_path / "detached-user-1")
        physical.mkdir()
        (physical / "keep").write_bytes(b"replacement marker")

    if stage == "reclaim":
        original_remove = io._remove_directory

        def remove(root: int, parents: tuple[str, ...], name: str) -> None:
            original_remove(root, parents, name)
            if parents == ("_uploads",):
                replace_user()

        monkeypatch.setattr(io, "_remove_directory", remove)
    else:
        original_publish = blob_store._publish_native_file_at

        def publish(*args: Any, **kwargs: Any) -> None:
            original_publish(*args, **kwargs)
            replace_user()

        monkeypatch.setattr(blob_store, "_publish_native_file_at", publish)
    with pytest.raises(blob_store.SyncBlobStoreError, match="user.*identity changed"):
        if stage == "reclaim":
            io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
        elif stage == "final":
            _publish(io, generation)
        else:
            _chunk(io, generation)
    assert (physical / "keep").read_bytes() == b"replacement marker"
    assert sorted(p.name for p in physical.iterdir()) == ["keep"]
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone() is None


@pytest.mark.parametrize("upload", ["UPLOAD-ONE", "üpload-one", "upload.one"])
def test_noncanonical_upload_token_cannot_touch_another_candidates_bytes(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload: str
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    _chunk(io, generation)
    other = _reserve(db, candidate="candidate-other", namespace="b" * 32, upload=upload, operation="fork-other")
    with pytest.raises(ValueError, match="invalid_native_upload_identity"):
        io.write_chunk(
            owner(),
            "native_fork_v1",
            "fork-other",
            "sha256:one",
            other,
            "candidate-other",
            chunk_index=0,
            payload=PAYLOAD,
            expected_hash="sha256:" + hashlib.sha256(PAYLOAD).hexdigest(),
        )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE native_chat_operations SET lease_expires_at = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),),
        )
    with pytest.raises(ValueError, match="invalid_native_upload_identity"):
        io.reclaim(owner(), "native_fork_v1", "fork-other", "sha256:one", "candidate-other")
    assert (tmp_path / "chat_assets" / "_uploads" / "upload-one" / "0.part").read_bytes() == PAYLOAD
    assert _candidate(db)["state"] == "reserved"


def test_writer_admission_rechecks_lease_after_candidate_lock(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    original = db.native_assets._candidate_row

    def expired_after_lock(*args: Any, **kwargs: Any) -> Any:
        row = original(*args, **kwargs)
        kwargs["conn"].execute(
            "UPDATE native_chat_operations SET lease_expires_at = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),),
        )
        return row

    monkeypatch.setattr(db.native_assets, "_candidate_row", expired_after_lock)
    with pytest.raises(ValueError, match="stale_attempt"):
        _chunk(io, generation)
    assert not (tmp_path / "chat_assets" / "_uploads").exists()


def test_postgres_operation_lock_wait_rejects_lease_expired_during_admission(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = native_fork_db
    if db.backend_type != BackendType.POSTGRESQL:
        pytest.skip("requires the official PostgreSQL fixture's row lock")
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    waiting = threading.Event()
    original_row = db.native_forks._row

    def waited_row(*args: Any, **kwargs: Any) -> Any:
        waiting.set()
        return original_row(*args, **kwargs)

    def writer() -> str:
        try:
            return _chunk(io, generation)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as workers:
        with db.transaction() as conn:
            original_row(owner(), "native_fork_v1", "fork-one", conn=conn)
            monkeypatch.setattr(db.native_forks, "_row", waited_row)
            future = workers.submit(writer)
            assert waiting.wait(5), "writer did not reach the locked operation"
            conn.execute(
                "UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?",
                (datetime.now(timezone.utc).isoformat(), "fork-one"),
            )
        with pytest.raises(ValueError, match="stale_attempt"):
            future.result(timeout=8)
    assert not (tmp_path / "chat_assets" / "_uploads").exists()


def test_native_storage_refuses_noncanonical_tenant_even_with_matching_db_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.Chat.native_chat_assets")
    db = CharactersRAGDB(db_path=str(tmp_path / "other-tenant.sqlite"), client_id="other-tenant", owner_user_id="1")
    resolved: list[int] = []

    def resolve(user_id: int) -> Path:
        resolved.append(user_id)
        return tmp_path

    monkeypatch.setattr(DatabasePaths, "resolve_user_base_directory", staticmethod(resolve))
    try:
        with pytest.raises(ValueError, match="native_asset_owner_mismatch"):
            module.NativeChatAssetIO(db, user_id=1)
        assert resolved == []
    finally:
        db.close_connection()


@pytest.mark.parametrize("stage", ["before", "chunk", "final", "reclaim"])
def test_native_storage_keeps_captured_owner_after_client_reassignment(
    native_fork_db: CharactersRAGDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from tldw_Server_API.app.core.Sync.v2 import blob_store

    db = native_fork_db
    generation = _reserve(db)
    io = _io(db, tmp_path, monkeypatch)
    if stage in {"final", "reclaim"}:
        _chunk(io, generation)
    if stage == "reclaim":
        _publish(io, generation)
        _expire(db)
        original_remove = io._remove_directory

        def remove(*args: Any) -> None:
            original_remove(*args)
            if args[1] == ("_uploads",):
                db.client_id = "2"

        monkeypatch.setattr(io, "_remove_directory", remove)
    elif stage in {"chunk", "final"}:
        original_publish = blob_store._publish_native_file_at

        def publish(*args: Any, **kwargs: Any) -> None:
            original_publish(*args, **kwargs)
            db.client_id = "2"

        monkeypatch.setattr(blob_store, "_publish_native_file_at", publish)
    else:
        db.client_id = "2"
    try:
        with pytest.raises(ValueError, match="operation_owner_mismatch"):
            if stage == "reclaim":
                io.reclaim(owner(), "native_fork_v1", "fork-one", "sha256:one", "candidate-one")
            elif stage == "final":
                _publish(io, generation)
            elif stage == "before":
                changed = AuthorizedNativeOwner("2", "native:2", NativeScopeV1(kind="global"))
                io.write_chunk(
                    changed,
                    "native_fork_v1",
                    "fork-one",
                    "sha256:one",
                    generation,
                    "candidate-one",
                    chunk_index=0,
                    payload=PAYLOAD,
                    expected_hash="sha256:" + hashlib.sha256(PAYLOAD).hexdigest(),
                )
            else:
                _chunk(io, generation)
    finally:
        db.client_id = "1"
    if stage == "before":
        assert not (tmp_path / "chat_assets").exists()
    with db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE intent_kind = 'release'").fetchone() is None


def test_retention_lease_renews_only_after_actual_empty_candidate_cleanup(native_fork_db, tmp_path, monkeypatch):
    """Reserved candidate has no bytes/charge; actual guarded cleanup still must settle it."""
    from tldw_Server_API.tests.DB_Management.test_native_retention_receipts import KIND, OPERATION, payload, request

    db = native_fork_db
    cid = db.add_conversation({"character_id": None, "title": "Retention preparation"})
    with db.transaction() as conn:
        token = db.native_forks.read_retention_source_revision(owner(), cid, conn=conn)
    original = request(payload(cid) | {"owner_key": owner().owner_key, "expected_asset_manifest_revision": token})
    with db.transaction() as conn:
        first = db.native_forks.claim_retention_attempt(owner(), original, conn=conn)
        db.native_assets.reserve_candidate(
            owner(),
            KIND,
            OPERATION,
            original.request_digest,
            first.generation,
            "retention-candidate",
            NAMESPACE,
            "retention-upload",
            "0" * 64,
            3,
            "image/png",
            "embedded_image_v1",
            now=datetime.now(timezone.utc),
            conn=conn,
        )
        conn.execute(
            "UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?",
            ("2000-01-01T00:00:00+00:00", OPERATION),
        )
    with db.transaction() as conn:
        assert db.native_forks.claim_retention_attempt(owner(), original, conn=conn).state == "reconciliation_required"
    root = tmp_path / "retention-root"
    root.mkdir()
    adapter = _io(db, root, monkeypatch)
    assert adapter.reclaim(owner(), KIND, OPERATION, original.request_digest, "retention-candidate")
    assert (
        db.execute_query(
            "SELECT state FROM native_chat_asset_candidates WHERE candidate_id = 'retention-candidate'"
        ).fetchone()["state"]
        == "discarded"
    )
    with db.transaction() as conn:
        second = db.native_forks.claim_retention_attempt(owner(), original, conn=conn)
        assert second.generation == first.generation + 1
        with pytest.raises(ValueError, match="stale_attempt"):
            db.native_assets.reserve_candidate(
                owner(),
                KIND,
                OPERATION,
                original.request_digest,
                first.generation,
                "old-worker",
                "b" * 32,
                "old-upload",
                "0" * 64,
                3,
                "image/png",
                "embedded_image_v1",
                now=datetime.now(timezone.utc),
                conn=conn,
            )
