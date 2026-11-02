"""Real SDK regressions for complete, coordinated storage access lifetimes."""

import hashlib
import importlib
import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError
from tldw_Server_API.app.core.Embeddings.chroma_storage import ChromaStorageError
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("CHROMADB_FORCE_STUB", raising=False)
    return {
        "USER_DB_BASE_DIR": str(tmp_path),
        "embedding_config": {},
        "chroma_client_settings": {"anonymized_telemetry": False},
    }


@pytest.fixture
def seeded(config):
    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection(
            "claims_for_1",
            embedding_function=None,
            configuration={"hnsw": {"batch_size": 2, "sync_threshold": 2}},
        )
        collection.upsert(ids=[str(i) for i in range(10)], embeddings=[[float(i + 1), 1.0] for i in range(10)])
        path = manager.user_chroma_path
    return path


def inventory(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()
    }


def open_reader(config):
    module = importlib.import_module("tldw_Server_API.app.core.Claims_Extraction.claims_cluster_vectors")
    return module.open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config)


def test_raw_collection_get_waits_for_writer_lifetime(config, seeded):
    with ChromaDBManager("1", config) as manager:
        collection = manager.get_collection("claims_for_1")
        started = threading.Event()

        def read():
            started.set()
            return collection.get(ids=["0"], include=["embeddings"])["ids"]

        with ThreadPoolExecutor(max_workers=1) as pool:
            with manager._lock:
                future = pool.submit(read)
                assert started.wait(5)
                try:
                    future.result(timeout=0.2)
                except TimeoutError:
                    pass
                else:
                    pytest.fail("raw collection escaped the writer's storage lock")
            assert future.result(timeout=10) == ["0"]


def test_existing_only_reads_real_vectors(config, seeded):
    with open_reader(config) as read:
        assert read is not None
        assert list(read(("0",))["0"]) == [1.0, 1.0]


def test_absent_owner_does_not_provision(config, tmp_path):
    before = inventory(tmp_path)
    with open_reader(config) as read:
        assert read is None
    assert inventory(tmp_path) == before
    assert not (tmp_path / "1").exists()


@pytest.mark.parametrize(
    "missing", ["index_metadata.pickle", "data_level0.bin", "header.bin", "length.bin", "link_lists.bin", "directory"]
)
def test_partial_index_rejected_before_sdk_without_inventory_changes(config, seeded, missing):
    import shutil

    (segment,) = [p for p in seeded.iterdir() if p.is_dir()]
    if missing == "directory":
        shutil.rmtree(segment)
    else:
        (segment / missing).unlink()
    before = inventory(seeded)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config):
            pytest.fail("incomplete index was accepted")
    assert not caught.value.retryable
    assert inventory(seeded) == before


@pytest.mark.parametrize("sql", ["DROP TABLE collections", "DELETE FROM databases", "DELETE FROM tenants"])
def test_strict_schema_and_default_records_rejected_without_repair(config, seeded, sql):
    with sqlite3.connect(seeded / "chroma.sqlite3") as connection:
        connection.execute(sql)
    before = inventory(seeded)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config):
            pytest.fail("malformed existing store accepted")
    assert not caught.value.retryable
    assert inventory(seeded) == before


def test_path_resolver_is_noncreating(config, tmp_path):
    result = DatabasePaths.resolve_user_chroma_directory("1", base_dir_override=tmp_path)
    assert result == tmp_path / "1" / "chroma_storage"
    assert not result.parent.exists()


def _spawn_writer(config, contended, outcome, action):
    """Instrument actual native-lock contention, without replacing locking."""
    from tldw_Server_API.app.core.Embeddings import chroma_storage

    acquire = chroma_storage._acquire_platform_file_lock

    def observed(fd):
        try:
            acquire(fd)
        except OSError:
            contended.set()
            raise

    chroma_storage._acquire_platform_file_lock = observed
    try:
        with ChromaDBManager("1", config) as manager:
            if action == "delete":
                manager.client.delete_collection(name="claims_for_1")
            else:
                collection = manager.client.get_collection(name="claims_for_1")
                collection.upsert(ids=["0"], embeddings=[[9.0, 1.0]])
        outcome.put("ok")
    except ChromaStorageError as exc:
        outcome.put((exc.code, exc.retryable))


@pytest.mark.parametrize("action", ["upsert", "delete"])
def test_spawned_writer_waits_between_validation_and_actual_vector_read(config, seeded, action):
    context = multiprocessing.get_context("spawn")
    contended, outcome = context.Event(), context.Queue()
    process = context.Process(target=_spawn_writer, args=(config, contended, outcome, action))
    try:
        with open_reader(config) as read:
            process.start()
            assert contended.wait(15), "spawned writer never reached the real native lock"
            assert process.is_alive()
            assert outcome.empty()
            assert list(read(("0",))["0"]) == [1.0, 1.0]
        process.join(15)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert outcome.get(timeout=2) == "ok"
        with open_reader(config) as read:
            if action == "delete":
                assert read is None
            else:
                assert list(read(("0",))["0"]) == [9.0, 1.0]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        outcome.close()


def test_spawned_contention_is_bounded_retryable_and_restores_lock(config, seeded):
    child_config = {**config, "chroma_client_settings": {"storage_lock_timeout": 0.2}}
    context = multiprocessing.get_context("spawn")
    contended, outcome = context.Event(), context.Queue()
    process = context.Process(target=_spawn_writer, args=(child_config, contended, outcome, "upsert"))
    try:
        with open_reader(config) as read:
            process.start()
            assert contended.wait(15)
            process.join(10)
            assert process.exitcode == 0
            assert outcome.get(timeout=2) == ("vector_busy", True)
            assert list(read(("0",))["0"]) == [1.0, 1.0]
        with ChromaDBManager("1", config) as manager:
            assert manager.get_collection("claims_for_1").count() == 10
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        outcome.close()


def test_nonclosing_manager_does_not_pin_sdk_or_process_lock(config, seeded):
    manager = ChromaDBManager("1", config)
    collection = manager.client.get_collection(name="claims_for_1")
    assert collection.count() == 10
    assert manager._lock.sdk is None
    context = multiprocessing.get_context("spawn")
    contended, outcome = context.Event(), context.Queue()
    process = context.Process(target=_spawn_writer, args=(config, contended, outcome, "upsert"))
    try:
        process.start()
        process.join(15)
        assert process.exitcode == 0
        assert outcome.get(timeout=2) == "ok"
        assert list(collection.get(ids=["0"], include=["embeddings"])["embeddings"][0]) == [9.0, 1.0]
        before = inventory(seeded)
        time.sleep(0.1)
        assert inventory(seeded) == before
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        outcome.close()
        manager.close()


def test_other_owner_proceeds_while_same_owner_waits(config, seeded):
    started = threading.Event()
    manager = ChromaDBManager("1", config)

    def same_owner():
        started.set()
        with ChromaDBManager("1", config):
            return "same"

    def other_owner():
        with ChromaDBManager("2", config):
            return "other"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            with manager.storage_access():
                same = pool.submit(same_owner)
                assert started.wait(5)
                other = pool.submit(other_owner)
                assert other.result(timeout=5) == "other"
                assert not same.done()
            assert same.result(timeout=5) == "same"
    finally:
        manager.close()


def test_raw_handle_after_last_manager_close_cannot_reopen_store(config, seeded):
    manager = ChromaDBManager("1", config)
    collection = manager.client.get_collection(name="claims_for_1")
    manager.close()
    before = inventory(seeded)
    with pytest.raises(ChromaStorageError, match="vector_closed"):
        collection.get(ids=["0"], include=["embeddings"])
    assert inventory(seeded) == before


def test_source_disappearance_before_manager_open_is_not_provisioned(config, seeded, monkeypatch):
    import shutil

    from tldw_Server_API.app.core.Claims_Extraction import claims_cluster_vectors

    def disappeared(*args, **kwargs):
        shutil.rmtree(seeded)
        return ChromaDBManager(*args, **kwargs)

    monkeypatch.setattr(claims_cluster_vectors, "ChromaDBManager", disappeared)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config):
            pytest.fail("disappearance became an absence skip")
    assert caught.value.retryable
    assert not seeded.exists()


def test_source_disappearance_immediately_before_sdk_open_does_not_recreate(config, seeded, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    original = ExistingClient._factory

    def disappeared(self):
        (seeded / "chroma.sqlite3").unlink()
        return original(self)

    monkeypatch.setattr(ExistingClient, "_factory", disappeared)
    with pytest.raises(ClusterRebuildStorageError):
        with open_reader(config) as read:
            read(("0",))
    assert not (seeded / "chroma.sqlite3").exists()


def test_existing_only_never_uses_forced_stub_or_fallback(config, seeded, monkeypatch):
    monkeypatch.setenv("CHROMADB_FORCE_STUB", "1")
    config["chroma_client_settings"].update(backend="stub", allow_stub_fallback=True)
    with open_reader(config) as read:
        assert list(read(("0",))["0"]) == [1.0, 1.0]


def test_access_errors_are_retryable_not_no_embeddings(config, seeded, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import chroma_read_validation

    def denied(*args, **kwargs):
        raise PermissionError("sensitive location")

    monkeypatch.setattr(chroma_read_validation.sqlite3, "connect", denied)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config):
            pytest.fail("access failure became no_embeddings")
    assert caught.value.retryable
    assert "sensitive" not in str(caught.value)


def test_no_collection_is_proven_absent_without_sdk_initialization(config, seeded, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    with ChromaDBManager("1", config) as manager:
        manager.delete_collection("claims_for_1")
    before = inventory(seeded)

    def forbidden(self):
        pytest.fail("SDK initialized to prove collection absence")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with open_reader(config) as read:
        assert read is None
    assert inventory(seeded) == before


def test_disappearance_under_real_coordination_fails_without_recreation(config, seeded, monkeypatch):
    import shutil

    from tldw_Server_API.app.core.Claims_Extraction import claims_cluster_vectors

    reached_open = threading.Event()

    def observed(*args, **kwargs):
        reached_open.set()
        return ChromaDBManager(*args, **kwargs)

    def reader():
        with open_reader(config):
            pytest.fail("disappearing storage was accepted")

    monkeypatch.setattr(claims_cluster_vectors, "ChromaDBManager", observed)
    with ChromaDBManager("1", config) as manager:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with manager.storage_access():
                future = pool.submit(reader)
                assert reached_open.wait(5)
                shutil.rmtree(seeded)
            with pytest.raises(ClusterRebuildStorageError):
                future.result(timeout=10)
    assert not seeded.exists()


def test_healthy_read_does_not_repair_or_overwrite_index_files(config, seeded):
    before = {key: digest for key, digest in inventory(seeded).items() if key != "chroma.sqlite3"}
    with open_reader(config) as read:
        assert read(("0",))
        assert read(("1",))
    assert {key: digest for key, digest in inventory(seeded).items() if key != "chroma.sqlite3"} == before


def test_closed_reader_callback_cannot_escape_session(config, seeded):
    with open_reader(config) as read:
        assert read(("0",))
    before = inventory(seeded)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_vector_closed"):
        read(("0",))
    assert inventory(seeded) == before


def test_empty_vector_request_does_not_expand_to_unbounded_sdk_get(config, seeded):
    with open_reader(config) as read:
        assert read(()) == {}


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE collections ADD COLUMN unexpected TEXT",
        "CREATE TRIGGER unexpected AFTER INSERT ON collections BEGIN DELETE FROM segments; END",
        "UPDATE migrations SET hash = 'bad' WHERE dir = 'sysdb' AND version = 1",
        "UPDATE embeddings_queue_config SET config_json_str = '{}'",
        "DELETE FROM segments WHERE scope = 'METADATA'",
    ],
)
def test_malformed_schema_or_collection_fails_before_any_sdk(config, seeded, monkeypatch, sql):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    with sqlite3.connect(seeded / "chroma.sqlite3") as db:
        db.execute(sql)
    before = inventory(seeded)

    def forbidden(self):
        pytest.fail("SDK was allowed to inspect malformed storage")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config):
            pytest.fail("malformed storage accepted")
    assert not caught.value.retryable
    assert inventory(seeded) == before


def test_unknown_sdk_profile_fails_without_provisioning(config, seeded, monkeypatch):
    import chromadb

    before = inventory(seeded)
    monkeypatch.setattr(chromadb, "__version__", "0.4.0")
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_vector_unsupported_sdk"):
        with open_reader(config):
            pytest.fail("unknown SDK profile accepted")
    assert inventory(seeded) == before


@pytest.mark.parametrize(
    "name", ["header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"]
)
def test_malformed_index_files_rejected_before_sdk(config, seeded, monkeypatch, name):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    (segment,) = [p for p in seeded.iterdir() if p.is_dir()]
    target = segment / name
    data = target.read_bytes()
    target.write_bytes(bytes(len(data)) if name == "header.bin" else data[:-1])
    before = inventory(seeded)

    def forbidden(self):
        pytest.fail("SDK opened a structurally incomplete index")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with pytest.raises(ClusterRebuildStorageError):
        with open_reader(config) as read:
            read(("0",))
    assert inventory(seeded) == before


def test_duplicate_ids_keep_sdk_rejection_as_classified_failure(config, seeded):
    from chromadb.errors import DuplicateIDError

    with ChromaDBManager("1", config) as manager:
        with pytest.raises(DuplicateIDError):
            manager.get_collection("claims_for_1").get(ids=["0", "0"], include=["embeddings"])
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config) as read:
            read(("0", "0"))
    assert not caught.value.retryable


def test_real_missing_ids_are_a_successful_partial_coverage_read(config, seeded):
    with open_reader(config) as read:
        assert read(("not-present",)) == {}


def test_persisted_overwide_vectors_rejected_before_sdk_allocation(config, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection(
            "claims_for_1",
            embedding_function=None,
            configuration={"hnsw": {"batch_size": 2, "sync_threshold": 2}},
        )
        collection.upsert(ids=[str(i) for i in range(4)], embeddings=[[1.0] * 4097 for _ in range(4)])

    def forbidden(self):
        pytest.fail("SDK was asked to allocate over-wide vectors")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        with open_reader(config) as read:
            read(("0",))


def test_missing_native_lock_backend_fails_closed(config, seeded, monkeypatch):
    from tldw_Server_API.app.core.Infrastructure import distributed_lock

    before = inventory(seeded)
    monkeypatch.setattr(distributed_lock, "_fcntl_mod", None)
    monkeypatch.setattr(distributed_lock, "_msvcrt_mod", None)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_reader(config):
            pytest.fail("thread-only fallback was used")
    assert not caught.value.retryable
    assert inventory(seeded) == before


def test_writer_preserves_public_embedding_adapter_without_retaining_sdk(config):
    import numpy as np
    from chromadb.api.types import EmbeddingFunction

    class FixedEmbedding(EmbeddingFunction):
        def __init__(self):
            pass

        @staticmethod
        def name():
            return "task6-fixed"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return FixedEmbedding()

        def __call__(self, input):
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in input]

    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection("documents", embedding_function=FixedEmbedding())
        collection.upsert(ids=["a"], documents=["text"])
        assert list(collection.get(ids=["a"], include=["embeddings"])["embeddings"][0]) == [1.0, 0.0]
        assert manager._lock.sdk is None


def _spawn_lock_only(path, outcome):
    from pathlib import Path

    from tldw_Server_API.app.core.Embeddings.chroma_storage import StorageSession

    try:
        with StorageSession(Path(path), timeout=0.2):
            outcome.put("acquired")
    except ChromaStorageError as exc:
        outcome.put((exc.code, exc.retryable))


def test_failed_writer_startup_stops_and_evicts_partially_started_sdk(config, monkeypatch):
    from chromadb.api.shared_system_client import SharedSystemClient
    from chromadb.config import System

    original = System.start
    systems = []

    def failed(self):
        original(self)
        systems.append(self)
        raise RuntimeError("failure after SDK runtime start")

    monkeypatch.setattr(System, "start", failed)
    try:
        with pytest.raises(RuntimeError):
            ChromaDBManager("1", config)
        assert systems
        assert not systems[0]._running
        assert systems[0] not in SharedSystemClient._identifier_to_system.values()
    finally:
        for system in systems:
            if system._running:
                system.stop()
            for key, value in list(SharedSystemClient._identifier_to_system.items()):
                if value is system:
                    SharedSystemClient._identifier_to_system.pop(key)


def test_failed_sdk_shutdown_cannot_release_store_to_another_process(config, seeded, monkeypatch):
    import os

    from tldw_Server_API.app.core.Embeddings import chroma_storage

    manager = ChromaDBManager("1", config)
    session = manager._lock
    stop = chroma_storage._stop_client
    retained = []

    def failed(client):
        retained.append(client)
        raise RuntimeError("simulated failed SDK shutdown")

    context = multiprocessing.get_context("spawn")
    outcome = context.Queue()
    process = context.Process(target=_spawn_lock_only, args=(str(seeded), outcome))
    try:
        monkeypatch.setattr(chroma_storage, "_stop_client", failed)
        with pytest.raises(RuntimeError):
            manager.get_collection("claims_for_1")
        process.start()
        process.join(15)
        assert process.exitcode == 0
        assert outcome.get(timeout=2) == ("vector_busy", True)
        with pytest.raises(ChromaStorageError, match="vector_shutdown"):
            manager.get_collection("claims_for_1")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        outcome.close()
        monkeypatch.setattr(chroma_storage, "_stop_client", stop)
        for client in retained:
            stop(client)
        # Only this fixture may release quarantine after proving actual SDK stop.
        if session._fd is not None:
            chroma_storage._release_platform_file_lock(session._fd)
            os.close(session._fd)
            session._fd = None
        session.sdk = None
        session._stop = None
        session._mode = None
        session._shutdown_failed = False
        getattr(chroma_storage, "_FAILED_SHUTDOWNS", {}).pop(str(seeded), None)
        manager.close()
