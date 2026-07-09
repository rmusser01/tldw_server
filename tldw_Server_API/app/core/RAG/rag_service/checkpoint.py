"""Checkpoint-based resumption for batch RAG operations.

Provides atomic checkpoint saves (temp file + rename) with progress tracking,
resume capability, and auto-cleanup. Useful for:
- Batch document ingestion
- Bulk re-embedding when switching models
- Batch chunking operations
- Long-running evaluation runs

Ported from RAGnarok-AI's checkpoint system, adapted for tldw_server2.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class CheckpointData(BaseModel):
    """Data stored in a checkpoint file.

    Immutable after creation -- updates create new instances.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        task_type: Type of task (e.g., "ingestion", "embedding", "evaluation").
        created_at: ISO timestamp when the checkpoint was created.
        updated_at: ISO timestamp when the checkpoint was last updated.
        total_items: Total number of items to process.
        completed_items: Number of items completed so far.
        results: Results collected so far (list of dicts).
        config: Configuration used for the task.
        metadata: Additional metadata (model name, namespace, etc.).
    """

    model_config = {"frozen": True}

    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    task_type: str = Field(..., description="Type of task")
    created_at: str = Field(..., description="Creation timestamp (ISO)")
    updated_at: str = Field(..., description="Last update timestamp (ISO)")
    total_items: int = Field(..., ge=0, description="Total items to process")
    completed_items: int = Field(default=0, ge=0, description="Items completed")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Results so far")
    config: dict[str, Any] = Field(default_factory=dict, description="Task configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def is_complete(self) -> bool:
        """Check if the task is complete."""
        return self.completed_items >= self.total_items

    @property
    def progress_percent(self) -> float:
        """Get progress as a percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100

    @property
    def remaining_items(self) -> int:
        """Number of items remaining."""
        return max(0, self.total_items - self.completed_items)


# Restrict checkpoint IDs to safe characters for filenames.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CheckpointManager:
    """Manages checkpoints for long-running batch operations.

    Provides atomic saves (write to temp file, rename), resume
    capabilities, and auto-cleanup of completed/aged checkpoints.

    Example::

        manager = CheckpointManager()
        checkpoint = manager.create("ingestion", total_items=1000)
        for i, result in enumerate(process_items()):
            checkpoint = manager.save_progress(checkpoint, result)
        manager.cleanup(checkpoint)
    """

    DEFAULT_CHECKPOINT_DIR = ".tldw/checkpoints"

    def __init__(self, checkpoint_dir: Path | str | None = None) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
                           Defaults to .tldw/checkpoints/
        """
        self.checkpoint_dir = Path(checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR).resolve(strict=False)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        task_type: str,
        total_items: int,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointData:
        """Create a new checkpoint.

        Args:
            task_type: Type of task (e.g., "ingestion", "embedding", "evaluation").
            total_items: Total number of items to process.
            config: Configuration for the task.
            metadata: Additional metadata.

        Returns:
            New CheckpointData instance.
        """
        now = datetime.now(timezone.utc).isoformat()
        safe_task_type = self._safe_checkpoint_component(task_type, fallback="task")
        checkpoint_id = f"{safe_task_type}_{uuid.uuid4().hex[:8]}"

        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            task_type=task_type,
            created_at=now,
            updated_at=now,
            total_items=total_items,
            completed_items=0,
            results=[],
            config=config or {},
            metadata=metadata or {},
        )

        self._save_atomic(checkpoint)
        logger.info(
            f"Created checkpoint {checkpoint_id} for {task_type} "
            f"({total_items} items)"
        )
        return checkpoint

    def save_progress(
        self,
        checkpoint: CheckpointData,
        result: dict[str, Any],
    ) -> CheckpointData:
        """Save progress with a new result (immutable update).

        New results are appended to a JSONL sidecar file so that each
        save is O(1) instead of copying the entire results list.
        The in-memory ``results`` list is kept empty during incremental
        saves to avoid O(N^2) memory; full results are reconstituted
        from the sidecar on ``load()``.

        Args:
            checkpoint: Current checkpoint data.
            result: New result to add.

        Returns:
            Updated CheckpointData instance (new object).
        """
        updated = CheckpointData(
            checkpoint_id=checkpoint.checkpoint_id,
            task_type=checkpoint.task_type,
            created_at=checkpoint.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            total_items=checkpoint.total_items,
            completed_items=checkpoint.completed_items + 1,
            results=[],
            config=checkpoint.config,
            metadata=checkpoint.metadata,
        )

        # Append only the new result to the sidecar; write the checkpoint
        # JSON without the full results list to keep it small on disk.
        self._append_results(checkpoint.checkpoint_id, [result])
        self._save_atomic(updated, exclude_results=True)
        return updated

    def save_batch_progress(
        self,
        checkpoint: CheckpointData,
        results: list[dict[str, Any]],
    ) -> CheckpointData:
        """Save progress with multiple results at once.

        Like ``save_progress``, keeps the in-memory ``results`` list
        empty to avoid quadratic memory growth; results are appended
        to the JSONL sidecar only.

        Args:
            checkpoint: Current checkpoint data.
            results: New results to add.

        Returns:
            Updated CheckpointData instance (new object).
        """
        if not results:
            return checkpoint

        updated = CheckpointData(
            checkpoint_id=checkpoint.checkpoint_id,
            task_type=checkpoint.task_type,
            created_at=checkpoint.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            total_items=checkpoint.total_items,
            completed_items=checkpoint.completed_items + len(results),
            results=[],
            config=checkpoint.config,
            metadata=checkpoint.metadata,
        )

        self._append_results(checkpoint.checkpoint_id, results)
        self._save_atomic(updated, exclude_results=True)
        return updated

    def load(self, checkpoint_path: Path | str) -> CheckpointData:
        """Load a checkpoint from file.

        If a JSONL results sidecar exists, results are reconstituted from
        it (the main JSON may omit results to save space).

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Loaded CheckpointData.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint file is invalid.
        """
        path = self._resolve_checkpoint_path(checkpoint_path)
        if not path.exists():
            msg = f"Checkpoint file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid checkpoint file: {e}"
            raise ValueError(msg) from e

        # Reconstitute results from sidecar if they were stripped from the
        # main checkpoint JSON to save space.
        checkpoint_id = data.get("checkpoint_id", "")
        if not data.get("results") and checkpoint_id:
            sidecar = self._get_results_path(checkpoint_id)
            if sidecar.exists():
                results: list[dict[str, Any]] = []
                for line in sidecar.read_text().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                data["results"] = results

        try:
            return CheckpointData(**data)
        except (TypeError, ValueError) as e:
            msg = f"Invalid checkpoint file: {e}"
            raise ValueError(msg) from e

    def load_by_id(self, checkpoint_id: str) -> CheckpointData:
        """Load a checkpoint by its ID.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            Loaded CheckpointData.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            ValueError: If checkpoint_id contains unsafe characters.
        """
        if not _SAFE_ID_RE.match(checkpoint_id):
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id}")
        path = self._get_checkpoint_path(checkpoint_id)
        return self.load(path)

    def exists(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint exists.
        """
        if not _SAFE_ID_RE.match(checkpoint_id):
            return False
        return self._get_checkpoint_path(checkpoint_id).exists()

    def cleanup(self, checkpoint: CheckpointData) -> None:
        """Remove a checkpoint file and its results sidecar.

        Args:
            checkpoint: The checkpoint to remove.
        """
        path = self._get_checkpoint_path(checkpoint.checkpoint_id)
        sidecar = self._get_results_path(checkpoint.checkpoint_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Cleaned up checkpoint {checkpoint.checkpoint_id}")
        if sidecar.exists():
            sidecar.unlink()

    def cleanup_by_id(self, checkpoint_id: str) -> bool:
        """Remove a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            True if checkpoint was removed, False if it didn't exist.
        """
        if not _SAFE_ID_RE.match(checkpoint_id):
            return False
        path = self._get_checkpoint_path(checkpoint_id)
        sidecar = self._get_results_path(checkpoint_id)
        if path.exists():
            path.unlink()
            if sidecar.exists():
                sidecar.unlink()
            return True
        return False

    def list_checkpoints(self, task_type: str | None = None) -> list[CheckpointData]:
        """List all checkpoints, optionally filtered by task type.

        Args:
            task_type: Optional filter by task type.

        Returns:
            List of checkpoint data, sorted by most recently updated.
        """
        checkpoints: list[CheckpointData] = []

        for path in self.checkpoint_dir.glob("*.json"):
            try:
                checkpoint = self.load(path)
                if task_type is None or checkpoint.task_type == task_type:
                    checkpoints.append(checkpoint)
            except (ValueError, FileNotFoundError):
                continue

        return sorted(checkpoints, key=lambda c: c.updated_at, reverse=True)

    def cleanup_completed(self) -> int:
        """Remove all completed checkpoints.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        for checkpoint in self.list_checkpoints():
            if checkpoint.is_complete:
                self.cleanup(checkpoint)
                removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} completed checkpoint(s)")
        return removed

    def cleanup_older_than(self, days: int) -> int:
        """Remove checkpoints older than specified days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        cutoff = datetime.now(timezone.utc)

        for checkpoint in self.list_checkpoints():
            updated = datetime.fromisoformat(checkpoint.updated_at)
            age = (cutoff - updated).days
            if age > days:
                self.cleanup(checkpoint)
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} checkpoint(s) older than {days} days")
        return removed

    def get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint (public accessor).

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            Path to the checkpoint file.
        """
        return self._get_checkpoint_path(checkpoint_id)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint ID."""
        return self._resolve_checkpoint_path(f"{checkpoint_id}.json")

    def _get_results_path(self, checkpoint_id: str) -> Path:
        """Get the JSONL sidecar path for checkpoint results."""
        return self._resolve_checkpoint_path(f"{checkpoint_id}.results.jsonl")

    @staticmethod
    def _safe_checkpoint_component(value: str, *, fallback: str) -> str:
        """Return a checkpoint file component containing only safe filename characters."""
        safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in str(value))
        safe = safe.strip("._")
        return safe or fallback

    def _resolve_checkpoint_path(self, path_value: Path | str) -> Path:
        """Resolve a checkpoint file path under the configured checkpoint directory."""
        raw_path = Path(path_value)
        if raw_path.is_absolute():
            path = raw_path.resolve(strict=False)
        else:
            path = (self.checkpoint_dir / raw_path).resolve(strict=False)
        try:
            path.relative_to(self.checkpoint_dir)
        except ValueError as exc:
            raise ValueError("checkpoint path escapes checkpoint directory") from exc
        return path

    def _append_results(
        self, checkpoint_id: str, results: list[dict[str, Any]]
    ) -> None:
        """Append results to the JSONL sidecar file (O(1) per call)."""
        sidecar = self._get_results_path(checkpoint_id)
        with open(sidecar, "a") as f:
            for r in results:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    def _save_atomic(
        self, checkpoint: CheckpointData, *, exclude_results: bool = False
    ) -> None:
        """Save checkpoint atomically using temp file + rename.

        This prevents corrupted checkpoint files from partial writes.

        Args:
            checkpoint: The checkpoint data to save.
            exclude_results: If True, omit the ``results`` list from the
                JSON file (results are stored in a JSONL sidecar instead).
        """
        target_path = self._get_checkpoint_path(checkpoint.checkpoint_id)

        data = checkpoint.model_dump()
        if exclude_results:
            data["results"] = []

        # Write to temp file in same directory (for atomic rename on same filesystem)
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="checkpoint_",
            dir=self.checkpoint_dir,
        )

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            Path(temp_path).replace(target_path)
        except Exception:
            # Clean up temp file on error
            temp_path_obj = Path(temp_path)
            if temp_path_obj.exists():
                temp_path_obj.unlink()
            raise


def get_default_checkpoint_manager() -> CheckpointManager:
    """Get a checkpoint manager with default settings.

    Returns:
        CheckpointManager instance using .tldw/checkpoints/.
    """
    return CheckpointManager()
