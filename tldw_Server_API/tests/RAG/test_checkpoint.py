"""
Tests for checkpoint-based resumption for batch RAG operations.

Covers:
- CheckpointData model properties (is_complete, progress_percent, remaining_items)
- CheckpointManager lifecycle: create, save_progress, save_batch_progress,
  load, load_by_id, exists, cleanup, list, and age-based cleanup
- Atomic save semantics (temp file + rename)
- Safe ID validation against path-traversal attempts
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.rag_service.checkpoint import (
    CheckpointData,
    CheckpointManager,
    _SAFE_ID_RE,
)


# ---------------------------------------------------------------------------
# CheckpointData model tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointDataProperties:
    """Tests for CheckpointData computed properties."""

    def test_is_complete_false_when_items_remain(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="ingestion",
            created_at=now,
            updated_at=now,
            total_items=10,
            completed_items=5,
        )
        assert cp.is_complete is False

    def test_is_complete_true_when_all_done(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="ingestion",
            created_at=now,
            updated_at=now,
            total_items=5,
            completed_items=5,
        )
        assert cp.is_complete is True

    def test_is_complete_true_when_exceeded(self) -> None:
        """completed_items > total_items should still count as complete."""
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="ingestion",
            created_at=now,
            updated_at=now,
            total_items=5,
            completed_items=7,
        )
        assert cp.is_complete is True

    def test_progress_percent_midway(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="embedding",
            created_at=now,
            updated_at=now,
            total_items=200,
            completed_items=50,
        )
        assert cp.progress_percent == pytest.approx(25.0)

    def test_progress_percent_zero_total(self) -> None:
        """Zero total_items should report 100% progress."""
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="embedding",
            created_at=now,
            updated_at=now,
            total_items=0,
            completed_items=0,
        )
        assert cp.progress_percent == pytest.approx(100.0)

    def test_progress_percent_complete(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="embedding",
            created_at=now,
            updated_at=now,
            total_items=10,
            completed_items=10,
        )
        assert cp.progress_percent == pytest.approx(100.0)

    def test_remaining_items(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="evaluation",
            created_at=now,
            updated_at=now,
            total_items=100,
            completed_items=30,
        )
        assert cp.remaining_items == 70

    def test_remaining_items_when_exceeded(self) -> None:
        """remaining_items should never go negative."""
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="evaluation",
            created_at=now,
            updated_at=now,
            total_items=5,
            completed_items=8,
        )
        assert cp.remaining_items == 0

    def test_remaining_items_zero(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="evaluation",
            created_at=now,
            updated_at=now,
            total_items=10,
            completed_items=10,
        )
        assert cp.remaining_items == 0

    def test_frozen_model_rejects_mutation(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cp = CheckpointData(
            checkpoint_id="test_abc",
            task_type="ingestion",
            created_at=now,
            updated_at=now,
            total_items=10,
            completed_items=0,
        )
        with pytest.raises(Exception):
            cp.completed_items = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Safe ID regex tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSafeIdRegex:
    """Tests for the _SAFE_ID_RE used to prevent path traversal."""

    @pytest.mark.parametrize("valid_id", [
        "ingestion_abcd1234",
        "embedding_DEADBEEF",
        "task.2024-01-01",
        "simple_id",
        "A",
    ])
    def test_safe_ids_match(self, valid_id: str) -> None:
        assert _SAFE_ID_RE.match(valid_id) is not None

    @pytest.mark.parametrize("bad_id", [
        "../../etc/passwd",
        "../sneaky",
        "id with spaces",
        "id/slash",
        "",
        "id\x00null",
    ])
    def test_unsafe_ids_rejected(self, bad_id: str) -> None:
        assert _SAFE_ID_RE.match(bad_id) is None


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointManagerCreate:
    """Tests for CheckpointManager.create()."""

    def test_create_checkpoint_fields(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=50, config={"chunk_size": 512})

        assert cp.task_type == "ingestion"
        assert cp.total_items == 50
        assert cp.completed_items == 0
        assert cp.results == []
        assert cp.config == {"chunk_size": 512}
        assert cp.metadata == {}
        assert cp.checkpoint_id.startswith("ingestion_")
        assert cp.is_complete is False
        assert cp.progress_percent == pytest.approx(0.0)

    def test_create_checkpoint_persists_file(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("embedding", total_items=10)

        path = tmp_path / f"{cp.checkpoint_id}.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["checkpoint_id"] == cp.checkpoint_id
        assert data["task_type"] == "embedding"
        assert data["total_items"] == 10

    def test_create_with_metadata(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create(
            "evaluation",
            total_items=100,
            metadata={"model": "gpt-4", "namespace": "test"},
        )
        assert cp.metadata == {"model": "gpt-4", "namespace": "test"}

    def test_create_auto_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "checkpoints"
        manager = CheckpointManager(checkpoint_dir=nested)
        cp = manager.create("ingestion", total_items=1)
        assert nested.is_dir()
        assert manager.exists(cp.checkpoint_id)


@pytest.mark.unit
class TestCheckpointManagerSaveProgress:
    """Tests for save_progress and save_batch_progress."""

    def test_save_progress_increments_count(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=3)

        cp2 = manager.save_progress(cp, {"doc_id": 1, "status": "ok"})

        assert cp2.completed_items == 1
        # In-memory results stay empty to avoid O(N^2) accumulation;
        # results are stored in the JSONL sidecar and reconstituted on load().
        assert cp2.results == []
        # Verify sidecar persistence via load()
        loaded = manager.load_by_id(cp2.checkpoint_id)
        assert len(loaded.results) == 1
        assert loaded.results[0] == {"doc_id": 1, "status": "ok"}
        # Original is immutable and unchanged
        assert cp.completed_items == 0
        assert cp.results == []

    def test_save_progress_returns_new_instance(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=5)
        cp2 = manager.save_progress(cp, {"item": 1})

        assert cp2 is not cp
        assert cp2.checkpoint_id == cp.checkpoint_id
        assert cp2.updated_at >= cp.created_at

    def test_save_progress_chain(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=3)

        cp = manager.save_progress(cp, {"i": 0})
        cp = manager.save_progress(cp, {"i": 1})
        cp = manager.save_progress(cp, {"i": 2})

        assert cp.completed_items == 3
        # In-memory results stay empty; verify via load()
        assert cp.results == []
        loaded = manager.load_by_id(cp.checkpoint_id)
        assert len(loaded.results) == 3
        assert cp.is_complete is True
        assert cp.progress_percent == pytest.approx(100.0)
        assert cp.remaining_items == 0

    def test_save_batch_progress_multiple_results(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("embedding", total_items=10)

        batch = [{"doc": i, "ok": True} for i in range(4)]
        cp2 = manager.save_batch_progress(cp, batch)

        assert cp2.completed_items == 4
        # In-memory results stay empty; verify via load()
        assert cp2.results == []
        loaded = manager.load_by_id(cp2.checkpoint_id)
        assert len(loaded.results) == 4
        assert cp2.progress_percent == pytest.approx(40.0)
        # Original unchanged
        assert cp.completed_items == 0

    def test_save_batch_progress_empty_list_returns_same(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("embedding", total_items=5)

        cp2 = manager.save_batch_progress(cp, [])
        # Should return the same checkpoint instance when no results
        assert cp2 is cp
        assert cp2.completed_items == 0

    def test_save_batch_progress_accumulates(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("embedding", total_items=10)

        cp = manager.save_batch_progress(cp, [{"i": 0}, {"i": 1}])
        cp = manager.save_batch_progress(cp, [{"i": 2}, {"i": 3}, {"i": 4}])

        assert cp.completed_items == 5
        # In-memory results stay empty; verify via load()
        assert cp.results == []
        loaded = manager.load_by_id(cp.checkpoint_id)
        assert len(loaded.results) == 5
        assert cp.progress_percent == pytest.approx(50.0)


@pytest.mark.unit
class TestCheckpointManagerLoadRoundtrip:
    """Tests for load() and load_by_id() roundtrip."""

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        original = manager.create(
            "ingestion",
            total_items=20,
            config={"model": "bge-small"},
            metadata={"user": "test"},
        )
        original = manager.save_progress(original, {"doc": 1})

        path = tmp_path / f"{original.checkpoint_id}.json"
        loaded = manager.load(path)

        assert loaded.checkpoint_id == original.checkpoint_id
        assert loaded.task_type == original.task_type
        assert loaded.created_at == original.created_at
        assert loaded.updated_at == original.updated_at
        assert loaded.total_items == original.total_items
        assert loaded.completed_items == original.completed_items
        # In-memory results are empty after save; load() reconstitutes from sidecar
        assert len(loaded.results) == 1
        assert loaded.results[0] == {"doc": 1}
        assert loaded.config == original.config
        assert loaded.metadata == original.metadata

    def test_load_by_id_roundtrip(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        original = manager.create("evaluation", total_items=5)
        original = manager.save_batch_progress(original, [{"r": i} for i in range(3)])

        loaded = manager.load_by_id(original.checkpoint_id)

        assert loaded.checkpoint_id == original.checkpoint_id
        assert loaded.completed_items == 3
        # In-memory results are empty after save; load() reconstitutes from sidecar
        assert len(loaded.results) == 3

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.load(tmp_path / "does_not_exist.json")

    def test_load_by_id_nonexistent_raises(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.load_by_id("nonexistent_checkpoint_id")

    def test_load_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad_checkpoint.json"
        bad_file.write_text("{not valid json!!")
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid checkpoint file"):
            manager.load(bad_file)

    def test_load_by_id_path_traversal_raises(self, tmp_path: Path) -> None:
        """Attempting path traversal via load_by_id should raise ValueError."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid checkpoint ID"):
            manager.load_by_id("../../etc/passwd")

    def test_load_by_id_slash_in_id_raises(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid checkpoint ID"):
            manager.load_by_id("some/path")

    def test_load_by_id_space_in_id_raises(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid checkpoint ID"):
            manager.load_by_id("id with spaces")


@pytest.mark.unit
class TestCheckpointManagerExists:
    """Tests for the exists() method."""

    def test_exists_true_for_created_checkpoint(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=10)
        assert manager.exists(cp.checkpoint_id) is True

    def test_exists_false_for_missing_checkpoint(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        assert manager.exists("nonexistent_abc12345") is False

    def test_exists_false_for_unsafe_id(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        assert manager.exists("../../etc/passwd") is False

    def test_exists_false_after_cleanup(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=1)
        assert manager.exists(cp.checkpoint_id) is True
        manager.cleanup(cp)
        assert manager.exists(cp.checkpoint_id) is False


@pytest.mark.unit
class TestCheckpointManagerCleanup:
    """Tests for cleanup, cleanup_by_id, cleanup_completed, and cleanup_older_than."""

    def test_cleanup_removes_file(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=5)

        path = tmp_path / f"{cp.checkpoint_id}.json"
        assert path.exists()

        manager.cleanup(cp)
        assert not path.exists()

    def test_cleanup_idempotent(self, tmp_path: Path) -> None:
        """Cleaning up a checkpoint that was already removed should not raise."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=1)
        manager.cleanup(cp)
        # Second cleanup should be a no-op, not raise
        manager.cleanup(cp)

    def test_cleanup_by_id_returns_true_when_exists(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=1)
        assert manager.cleanup_by_id(cp.checkpoint_id) is True
        assert not manager.exists(cp.checkpoint_id)

    def test_cleanup_by_id_returns_false_when_missing(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        assert manager.cleanup_by_id("nonexistent_abc12345") is False

    def test_cleanup_by_id_returns_false_for_unsafe_id(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        assert manager.cleanup_by_id("../../etc/passwd") is False

    def test_cleanup_completed_removes_only_finished(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create a completed checkpoint
        cp_done = manager.create("ingestion", total_items=2)
        cp_done = manager.save_batch_progress(cp_done, [{"r": 1}, {"r": 2}])
        assert cp_done.is_complete is True

        # Create an incomplete checkpoint
        cp_pending = manager.create("embedding", total_items=10)
        cp_pending = manager.save_progress(cp_pending, {"r": 1})
        assert cp_pending.is_complete is False

        removed = manager.cleanup_completed()

        assert removed == 1
        assert not manager.exists(cp_done.checkpoint_id)
        assert manager.exists(cp_pending.checkpoint_id)

    def test_cleanup_completed_returns_zero_when_none_complete(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        manager.create("ingestion", total_items=10)
        assert manager.cleanup_completed() == 0

    def test_cleanup_older_than(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create a checkpoint and manually backdate it on disk
        cp_old = manager.create("ingestion", total_items=5)
        old_path = tmp_path / f"{cp_old.checkpoint_id}.json"
        data = json.loads(old_path.read_text())
        old_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        data["updated_at"] = old_date
        old_path.write_text(json.dumps(data, indent=2))

        # Create a recent checkpoint
        cp_new = manager.create("ingestion", total_items=5)

        removed = manager.cleanup_older_than(days=7)

        assert removed == 1
        assert not manager.exists(cp_old.checkpoint_id)
        assert manager.exists(cp_new.checkpoint_id)

    def test_cleanup_older_than_zero_when_all_recent(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        manager.create("ingestion", total_items=1)
        assert manager.cleanup_older_than(days=1) == 0


@pytest.mark.unit
class TestCheckpointManagerListCheckpoints:
    """Tests for list_checkpoints with and without task_type filter."""

    def test_list_all_checkpoints(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp1 = manager.create("ingestion", total_items=10)
        cp2 = manager.create("embedding", total_items=20)
        cp3 = manager.create("evaluation", total_items=5)

        result = manager.list_checkpoints()
        ids = {c.checkpoint_id for c in result}

        assert len(result) == 3
        assert cp1.checkpoint_id in ids
        assert cp2.checkpoint_id in ids
        assert cp3.checkpoint_id in ids

    def test_list_checkpoints_filtered_by_task_type(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        manager.create("ingestion", total_items=10)
        manager.create("ingestion", total_items=20)
        manager.create("embedding", total_items=5)

        ingestion = manager.list_checkpoints(task_type="ingestion")
        embedding = manager.list_checkpoints(task_type="embedding")
        evaluation = manager.list_checkpoints(task_type="evaluation")

        assert len(ingestion) == 2
        assert len(embedding) == 1
        assert len(evaluation) == 0

    def test_list_checkpoints_empty_directory(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        assert manager.list_checkpoints() == []

    def test_list_checkpoints_sorted_by_updated_at(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp1 = manager.create("ingestion", total_items=10)
        cp2 = manager.create("ingestion", total_items=10)
        # Force cp1 to be newer without relying on platform clock granularity.
        newer_than_cp2 = datetime.fromisoformat(cp2.updated_at) + timedelta(seconds=1)
        cp1 = cp1.model_copy(update={"updated_at": newer_than_cp2.isoformat()})
        manager._save_atomic(cp1)

        result = manager.list_checkpoints()
        # Most recently updated should be first
        assert result[0].checkpoint_id == cp1.checkpoint_id

    def test_list_checkpoints_skips_invalid_json_files(self, tmp_path: Path) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        manager.create("ingestion", total_items=10)

        # Drop an invalid JSON file in the same directory
        (tmp_path / "corrupted.json").write_text("{bad json!!")

        result = manager.list_checkpoints()
        assert len(result) == 1


@pytest.mark.unit
class TestCheckpointManagerAtomicSave:
    """Tests verifying atomic save behaviour."""

    def test_checkpoint_file_is_valid_json(self, tmp_path: Path) -> None:
        """Saved checkpoint files should always be valid JSON."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=5)
        cp = manager.save_progress(cp, {"key": "value"})

        path = tmp_path / f"{cp.checkpoint_id}.json"
        data = json.loads(path.read_text())
        assert data["completed_items"] == 1
        # Results are stored in a JSONL sidecar to avoid O(n^2) accumulation;
        # the in-memory model keeps results empty during incremental saves.
        assert cp.results == []

        # Verify the sidecar exists and contains the result
        sidecar = tmp_path / f"{cp.checkpoint_id}.results.jsonl"
        assert sidecar.exists()
        lines = [json.loads(l) for l in sidecar.read_text().splitlines() if l.strip()]
        assert lines == [{"key": "value"}]

    def test_no_temp_files_left_behind(self, tmp_path: Path) -> None:
        """After a successful save, no .tmp files should remain."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        cp = manager.create("ingestion", total_items=3)
        for i in range(3):
            cp = manager.save_progress(cp, {"i": i})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []
