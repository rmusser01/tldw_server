"""Persistence helpers for first-run setup readiness state."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Setup.readiness_models import LANE_IDS, LANE_STATUSES, OVERLAY_IDS

CONFIG_ROOT = setup_manager.CONFIG_RELATIVE_PATH.parent
READINESS_FILENAME = "setup_readiness.json"
_STORE: SetupReadinessStore | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SetupReadinessRecord(BaseModel):
    """Persisted first-run setup readiness snapshot."""

    status: Literal[
        "not_started",
        "previewed",
        "provisioning",
        "ready",
        "ready_with_warnings",
        "failed",
        "blocked",
    ] = "not_started"
    selected_profile_id: str | None = None
    lanes: list[dict[str, Any]] = Field(default_factory=list)
    overlays: list[str] = Field(default_factory=list)
    last_preview: dict[str, Any] | None = None
    last_provision: dict[str, Any] | None = None
    last_verification: dict[str, Any] | None = None
    operation_id: str | None = None
    operation_status: Literal["queued", "running", "completed", "failed"] | None = None
    errors: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def validate_lanes_and_overlays(self) -> "SetupReadinessRecord":
        """Reject unknown lane/status/overlay values before persisting."""

        for lane in self.lanes:
            lane_id = lane.get("lane_id")
            lane_status = lane.get("status")
            if lane_id not in LANE_IDS:
                raise ValueError(f"Unsupported setup readiness lane: {lane_id}")
            if lane_status not in LANE_STATUSES:
                raise ValueError(f"Unsupported setup readiness lane status: {lane_status}")

        unknown_overlays = [overlay for overlay in self.overlays if overlay not in OVERLAY_IDS]
        if unknown_overlays:
            raise ValueError(f"Unsupported setup readiness overlay: {unknown_overlays[0]}")
        return self


def _candidate_readiness_files() -> list[Path]:
    candidates: list[Path] = []

    override_file = os.getenv("TLDW_SETUP_READINESS_FILE")
    if override_file:
        candidates.append(Path(override_file))

    override_dir = os.getenv("TLDW_INSTALL_STATE_DIR")
    if override_dir:
        candidates.append(Path(override_dir) / READINESS_FILENAME)

    candidates.append(CONFIG_ROOT / READINESS_FILENAME)

    try:
        home = Path.home()
    except Exception:  # noqa: BLE001
        home = None
    if home:
        candidates.append(home / ".cache" / "tldw_server" / READINESS_FILENAME)

    candidates.append(Path(tempfile.gettempdir()) / "tldw_server" / READINESS_FILENAME)
    return candidates


def _resolve_readiness_file() -> Path | None:
    for path in _candidate_readiness_files():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            probe = path.parent / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            with contextlib.suppress(FileNotFoundError):
                probe.unlink()
            return path
        except Exception:  # noqa: BLE001
            logger.debug("Setup readiness candidate path not writable")

    logger.warning("No writable location found for setup readiness persistence.")
    return None


class SetupReadinessStore:
    """Read and write the first-run setup readiness snapshot."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        default_record = SetupReadinessRecord()
        if not self.path or not self.path.is_file():
            return default_record.model_dump()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return SetupReadinessRecord.model_validate(data).model_dump()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read setup readiness")
            return default_record.model_dump()

    def save(self, readiness: dict[str, Any]) -> dict[str, Any]:
        payload = dict(readiness)
        payload["updated_at"] = _utc_now()
        record = SetupReadinessRecord.model_validate(payload)
        data = record.model_dump()

        if not self.path:
            return data

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = handle.name
            os.replace(tmp_path, self.path)
        except Exception:
            if tmp_path:
                with contextlib.suppress(FileNotFoundError):
                    Path(tmp_path).unlink()
            raise
        return data

    def update(self, **fields: Any) -> dict[str, Any]:
        current = self.load()
        current.update(fields)
        return self.save(current)

    def reset(self) -> dict[str, Any]:
        return self.save(SetupReadinessRecord().model_dump())


def get_setup_readiness_store() -> SetupReadinessStore:
    global _STORE
    if _STORE is None:
        _STORE = SetupReadinessStore(_resolve_readiness_file())
    return _STORE


def reset_setup_readiness_store() -> None:
    global _STORE
    _STORE = None


__all__ = [
    "READINESS_FILENAME",
    "SetupReadinessRecord",
    "SetupReadinessStore",
    "get_setup_readiness_store",
    "reset_setup_readiness_store",
]
