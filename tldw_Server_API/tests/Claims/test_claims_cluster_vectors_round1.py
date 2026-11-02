"""Accepted Task 6 review regressions: identity, fork, bounded/default access."""

import io
import os
import pickle
import select
import signal
import sqlite3
import tempfile
from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Claims_Extraction.claims_cluster_vectors import open_claim_cluster_vector_reader
from tldw_Server_API.app.core.DB_Management import chroma_existing
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError
from tldw_Server_API.app.core.Embeddings.chroma_storage import ChromaStorageError, StorageSession
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.tests.Claims.test_claims_cluster_vectors_existing_only import (
    config as config,
)
from tldw_Server_API.tests.Claims.test_claims_cluster_vectors_existing_only import (
    inventory,
)
from tldw_Server_API.tests.Claims.test_claims_cluster_vectors_existing_only import (
    seeded as seeded,
)


@pytest.mark.parametrize("dangling", [False, True])
def test_manager_and_reader_reject_storage_root_alias(config, seeded, dangling):
    root = seeded.parent.parent
    (root / "2").mkdir()
    alias = root / "2" / "chroma_storage"
    alias.symlink_to(seeded if not dangling else root / "missing", target_is_directory=True)
    before = inventory(root)
    with pytest.raises((ValueError, ChromaStorageError)):
        ChromaDBManager("2", config)
    with pytest.raises(ClusterRebuildStorageError):
        with open_claim_cluster_vector_reader(owner_user_id="2", embedding_config=config):
            pytest.fail("aliased/dangling storage became an absence skip")
    assert inventory(root) == before


def test_owner_directory_alias_is_not_a_second_owner_identity(config, seeded):
    (seeded.parent.parent / "2").symlink_to(seeded.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        DatabasePaths.resolve_user_chroma_directory("2", base_dir_override=seeded.parent.parent)


def _fork_result(action):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            action()
        except ChromaStorageError as exc:
            result = exc.code.encode()
        except RuntimeError:
            result = b"runtime_error"
        else:
            result = b"entered"
        os.write(write_fd, result)
        os._exit(0)
    os.close(write_fd)
    try:
        assert select.select([read_fd], [], [], 5)[0], "forked child waited on inherited state"
        return os.read(read_fd, 100).decode()
    finally:
        os.close(read_fd)
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if not waited:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires real POSIX fork")
def test_forked_held_session_cannot_enter_or_exit_parent_lock(config, seeded):
    session = StorageSession(seeded)
    with session:
        assert _fork_result(lambda: session.__enter__()) == "vector_forked_handle"
        assert _fork_result(lambda: session.__exit__(None, None, None)) == "vector_forked_handle"
        assert session._depth == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires real POSIX fork")
def test_forked_cached_facade_and_manager_fail_before_inherited_mutex(config, seeded):
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as library

    with ChromaDBManager("1", config) as manager:
        collection = manager.get_collection("claims_for_1")
        with manager.storage_access():
            with library._PERSISTENT_CLIENTS_LOCK:
                assert _fork_result(lambda: ChromaDBManager("1", config)) == "vector_forked_handle"
            assert _fork_result(lambda: collection.count()) == "vector_forked_handle"
            assert _fork_result(lambda: manager.close()) == "vector_forked_handle"
        assert collection.count() == 10


@pytest.mark.parametrize(
    "mode",
    [
        "empty",
        "empty_deleted",
        "empty_updated_vector",
        "buffered",
        "mutated",
        "all_deleted",
        "interleaved",
        "delete_readd",
    ],
)
def test_default_logical_vectors_match_real_sdk_without_any_file_changes(config, mode, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection("claims_for_1", embedding_function=None)
        if not mode.startswith("empty"):
            collection.upsert(ids=["a", "b", "c"], embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        if mode in ("mutated", "interleaved"):
            collection.update(ids=["a"], embeddings=[[2.0, 3.0]])
            collection.update(ids=["b"], metadatas=[{"x": 1}])
            collection.delete(ids=["c"])
            collection.add(ids=["a", "d"], embeddings=[[9.0, 9.0], [4.0, 5.0]])
            collection.upsert(ids=["d"], embeddings=[[6.0, 7.0]])
        if mode == "all_deleted":
            collection.delete(ids=["a", "b", "c"])
        if mode == "delete_readd":
            collection.delete(ids=["a"])
            collection.add(ids=["a"], embeddings=[[7.0, 8.0]])
        if mode == "empty_deleted":
            collection.delete(ids=["missing"])
            collection.update(ids=["missing"], metadatas=[{"x": 1}])
        if mode == "empty_updated_vector":
            collection.update(ids=["missing"], embeddings=[[2.0, 3.0]])
        if mode == "interleaved":
            other = manager.client.create_collection("other", embedding_function=None)
            other.upsert(ids=["x"], embeddings=[[7.0, 8.0]])
            collection.update(ids=["a"], embeddings=[[5.0, 6.0]])
        result = collection.get(include=["embeddings"])
        expected = {key: tuple(vector) for key, vector in zip(result["ids"], result["embeddings"], strict=True)}
        path = manager.user_chroma_path
    before = inventory(path)

    def forbidden(self):
        pytest.fail("native SDK opened a buffered/empty target")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
        assert dict(read(("a", "b", "c", "d", "missing"))) == expected
    assert inventory(path) == before


def test_materialized_target_allows_unrelated_default_empty_and_buffered_collections(config, seeded):
    with ChromaDBManager("1", config) as manager:
        manager.client.create_collection("empty", embedding_function=None)
        other = manager.client.create_collection("buffered", embedding_function=None)
        other.upsert(ids=["x"], embeddings=[[1.0, 2.0]])
    before = inventory(seeded)
    with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
        assert tuple(read(("0",))["0"]) == (1.0, 1.0)
    after = inventory(seeded)
    assert set(after) == set(before)
    assert {key: value for key, value in after.items() if key != "chroma.sqlite3"} == {
        key: value for key, value in before.items() if key != "chroma.sqlite3"
    }


@pytest.mark.parametrize(
    "field,value", [("collections", 1), ("elements", 1), ("metadata_bytes", 1), ("opcodes", 1), ("sql_steps", 1)]
)
def test_aggregate_preflight_budget_fails_before_sdk(config, seeded, field, value):
    limits_type = chroma_existing.ChromaPreflightLimits
    limits = replace(limits_type(), **{field: value})
    if field == "collections":
        with ChromaDBManager("1", config) as manager:
            manager.client.create_collection("unrelated", embedding_function=None)
    before = inventory(seeded)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config, preflight_limits=limits):
            pytest.fail("preflight ignored an aggregate budget")
    assert inventory(seeded) == before


def _dump_data(value):
    buffer = io.BytesIO()
    writer = pickle.Pickler(buffer, protocol=3)
    writer.fast = True
    writer.dump(value)
    return buffer.getvalue()


@pytest.mark.parametrize("kind", ["bare_mark", "wrong_type", "fake_keys", "underflow", "extra_stack"])
def test_metadata_must_be_a_well_typed_mapping_before_sdk(config, seeded, kind, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    (segment,) = [p for p in seeded.iterdir() if p.is_dir()]
    keys = ["dimensionality", "total_elements_added", "max_seq_id", "id_to_label", "label_to_id", "id_to_seq_id"]
    if kind == "bare_mark":
        data = b"\x80\x03(" + b"".join(_dump_data(k)[2:-1] for k in keys) + b"."
    else:
        value = dict.fromkeys(keys)
        value.update(total_elements_added="ten", id_to_label={}, label_to_id={}, id_to_seq_id={})
        if kind == "fake_keys":
            value = {"fake": dict.fromkeys(keys)}
        data = _dump_data(value)
        if kind == "underflow":
            data = data[:-1] + b"s."
        if kind == "extra_stack":
            data = data[:-1] + b"N."
    (segment / "index_metadata.pickle").write_bytes(data)
    before = inventory(seeded)

    def forbidden(self):
        pytest.fail("native SDK saw malformed metadata mapping")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config):
            pytest.fail("invalid metadata mapping accepted")
    assert not caught.value.retryable
    assert inventory(seeded) == before


@pytest.mark.parametrize("field", ["metadata_bytes", "elements"])
def test_preflight_limits_are_aggregate_across_segments(config, seeded, field):
    with ChromaDBManager("1", config) as manager:
        other = manager.client.create_collection(
            "other", embedding_function=None, configuration={"hnsw": {"batch_size": 2, "sync_threshold": 2}}
        )
        other.upsert(ids=[str(i) for i in range(10)], embeddings=[[1.0, 2.0] for _ in range(10)])
    maximum = (
        sum(p.stat().st_size for p in seeded.glob("*/index_metadata.pickle")) - 1 if field == "metadata_bytes" else 15
    )
    limits = replace(chroma_existing.ChromaPreflightLimits(), **{field: maximum})
    before = inventory(seeded)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config, preflight_limits=limits):
            pytest.fail("each segment got a fresh budget")
    assert inventory(seeded) == before


@pytest.mark.parametrize("field", ["queue_records", "queue_components", "returned_components"])
def test_log_and_result_budgets_fail_without_file_changes(config, field):
    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection("claims_for_1", embedding_function=None)
        collection.upsert(ids=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        path = manager.user_chroma_path
    limits = replace(chroma_existing.ChromaPreflightLimits(), **{field: 1})
    before = inventory(path)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        with open_claim_cluster_vector_reader(
            owner_user_id="1", embedding_config=config, preflight_limits=limits
        ) as read:
            read(("a", "b"))
    assert inventory(path) == before


@pytest.mark.parametrize("damage", ["prefix", "tail", "codec", "width"])
def test_pending_coverage_and_float_codec_fail_closed(config, damage, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection("claims_for_1", embedding_function=None)
        collection.upsert(ids=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        collection.update(ids=["a"], metadatas=[{"x": 1}])
        path = manager.user_chroma_path
    with sqlite3.connect(path / "chroma.sqlite3") as db:
        if damage == "prefix":
            db.execute("DELETE FROM embeddings_queue WHERE seq_id = (SELECT MIN(seq_id) FROM embeddings_queue)")
        elif damage == "tail":
            db.execute("DELETE FROM embeddings_queue WHERE seq_id = (SELECT MAX(seq_id) FROM embeddings_queue)")
        elif damage == "codec":
            db.execute("UPDATE embeddings_queue SET encoding = 'INT32' WHERE vector IS NOT NULL")
        else:
            db.execute("UPDATE embeddings_queue SET vector = ? WHERE vector IS NOT NULL", (b"\x00" * 4,))
    before = inventory(path)

    def forbidden(self):
        pytest.fail("SDK was used to repair missing pending history")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
            read(("a", "b"))
    assert not caught.value.retryable
    assert inventory(path) == before


def test_validated_snapshot_reused_for_batches_within_one_lock(config, seeded, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import chroma_existing_state

    original = chroma_existing_state.preflight_existing_chroma
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chroma_existing_state, "preflight_existing_chroma", observed)
    with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
        before = len(calls)
        assert read(("0",))
        assert read(("1",))
        assert len(calls) == before


def test_large_startup_marker_history_does_not_require_a_linear_scan(config, seeded):
    with sqlite3.connect(seeded / "chroma.sqlite3") as db:
        db.executemany("INSERT INTO acquire_write(lock_status) VALUES (?)", [(1,)] * 5000)
    limits = replace(chroma_existing.ChromaPreflightLimits(), sql_steps=10_000)
    with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config, preflight_limits=limits) as read:
        assert read(("0",))


@pytest.mark.parametrize("count", [100, 999, 1001])
@pytest.mark.parametrize("space", ["l2", "cosine", "ip"])
def test_real_default_batch_and_persistence_boundaries_match_sdk(config, count, space):
    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection(
            "claims_for_1", embedding_function=None, metadata={"hnsw:space": space}
        )
        collection.upsert(ids=[str(i) for i in range(count)], embeddings=[[2.0, 3.0] for _ in range(count)])
        result = collection.get(ids=["0", str(count - 1)], include=["embeddings"])
        expected = {key: tuple(vector) for key, vector in zip(result["ids"], result["embeddings"], strict=True)}
        path = manager.user_chroma_path
    before = inventory(path)
    with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
        assert dict(read(("0", str(count - 1)))) == expected
    after = inventory(path)
    assert set(after) == set(before)
    if count < 1000:
        assert after == before


def test_missing_materialized_metadata_never_replays_a_purged_tail(config, seeded, monkeypatch):
    from tldw_Server_API.app.core.Embeddings.chroma_storage import ExistingClient

    (metadata,) = seeded.glob("*/index_metadata.pickle")
    metadata.unlink()
    before = inventory(seeded)

    def forbidden(self):
        pytest.fail("SDK repaired materialized metadata from a partial tail")

    monkeypatch.setattr(ExistingClient, "_factory", forbidden)
    with pytest.raises(ClusterRebuildStorageError):
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config):
            pytest.fail("materialized purge provenance was ignored")
    assert inventory(seeded) == before


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["add", "upsert", "update", "metadata", "delete"]),
            st.sampled_from(["a", "b", "c"]),
            st.integers(1, 5),
        ),
        max_size=12,
    )
)
def test_pending_operation_sequences_match_real_sdk(config, operations):
    # config is read-only; every Hypothesis example owns a fresh SDK store.
    with tempfile.TemporaryDirectory(dir=config["USER_DB_BASE_DIR"]) as base:
        example = {**config, "USER_DB_BASE_DIR": base}
        with ChromaDBManager("1", example) as manager:
            collection = manager.client.create_collection("claims_for_1", embedding_function=None)
            collection.upsert(ids=["a"], embeddings=[[1.0, 0.0]])
            for operation, key, value in operations:
                if operation == "delete":
                    collection.delete(ids=[key])
                elif operation == "metadata":
                    collection.update(ids=[key], metadatas=[{"value": value}])
                else:
                    getattr(collection, operation)(ids=[key], embeddings=[[float(value), float(value + 1)]])
            result = collection.get(include=["embeddings"])
            expected = {key: tuple(vector) for key, vector in zip(result["ids"], result["embeddings"], strict=True)}
            path = manager.user_chroma_path
        before = inventory(path)
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=example) as read:
            assert dict(read(("a", "b", "c"))) == expected
        assert inventory(path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("dimensionality", "2"),
        ("total_elements_added", -1),
        ("max_seq_id", "wrong"),
        ("id_to_label", {1: "a"}),
        ("label_to_id", {1: "wrong"}),
        ("id_to_seq_id", {"unknown": 1}),
        ("id_to_seq_id", {"0": "wrong"}),
    ],
)
def test_each_metadata_field_is_validated_independently(config, seeded, field, value):
    valid = {
        "dimensionality": None,
        "total_elements_added": 10,
        "max_seq_id": None,
        "id_to_label": {str(i): i + 1 for i in range(10)},
        "label_to_id": {i + 1: str(i) for i in range(10)},
        "id_to_seq_id": {},
    }
    valid[field] = value
    (metadata,) = seeded.glob("*/index_metadata.pickle")
    metadata.write_bytes(_dump_data(valid))
    before = inventory(seeded)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config):
            pytest.fail("malformed metadata field accepted")
    assert not caught.value.retryable
    assert inventory(seeded) == before


def test_reentrant_writer_cannot_invalidate_cached_logical_snapshot(config):
    with ChromaDBManager("1", config) as manager:
        collection = manager.client.create_collection("claims_for_1", embedding_function=None)
        collection.upsert(ids=["a"], embeddings=[[1.0, 0.0]])
        with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
            assert tuple(read(("a",))["a"]) == (1.0, 0.0)
            with pytest.raises(ChromaStorageError, match="vector_mixed_access"):
                collection.upsert(ids=["a"], embeddings=[[9.0, 0.0]])
            assert tuple(read(("a",))["a"]) == (1.0, 0.0)
        collection.upsert(ids=["a"], embeddings=[[2.0, 0.0]])
    with open_claim_cluster_vector_reader(owner_user_id="1", embedding_config=config) as read:
        assert tuple(read(("a",))["a"]) == (2.0, 0.0)


@pytest.mark.parametrize(
    "field,value", [("collections", True), ("elements", 0), ("metadata_bytes", 33_554_433), ("sql_steps", 0)]
)
def test_preflight_limits_reject_bool_zero_and_over_hard_maximum(field, value):
    with pytest.raises(ValueError):
        replace(chroma_existing.ChromaPreflightLimits(), **{field: value})
