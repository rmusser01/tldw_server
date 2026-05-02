"""Persistence helpers for setup audio readiness state."""

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
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import (
    AUDIO_BUNDLE_CATALOG_VERSION,
    DEFAULT_AUDIO_RESOURCE_PROFILE,
    build_audio_selection_key,
    get_audio_bundle_catalog,
)

CONFIG_ROOT = setup_manager.CONFIG_RELATIVE_PATH.parent
READINESS_FILENAME = "setup_audio_readiness.json"
_STORE: AudioReadinessStore | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AudioReadinessRecord(BaseModel):
    """Persisted setup audio readiness snapshot."""

    status: Literal[
        "not_started",
        "provisioning",
        "partial",
        "ready",
        "ready_with_warnings",
        "failed",
    ] = "not_started"
    selected_bundle_id: str | None = None
    selected_resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE
    tts_choice: str | None = None
    catalog_version: str = AUDIO_BUNDLE_CATALOG_VERSION
    selection_key: str | None = None
    machine_profile: dict[str, Any] | None = None
    last_verification: dict[str, Any] | None = None
    installed_profiles: list[str] = Field(default_factory=list)
    installed_asset_manifests: list[dict[str, Any]] = Field(default_factory=list)
    imported_packs: list[dict[str, Any]] = Field(default_factory=list)
    remediation_items: list[Any] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_utc_now)

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_payload(cls, data: Any) -> Any:
        """Backfill newer profile-aware fields for legacy readiness files."""

        if not isinstance(data, dict):
            return data

        payload = dict(data)
        payload.setdefault("selected_resource_profile", DEFAULT_AUDIO_RESOURCE_PROFILE)
        payload.setdefault("tts_choice", None)
        payload.setdefault("catalog_version", AUDIO_BUNDLE_CATALOG_VERSION)
        payload.setdefault("installed_profiles", [])
        payload.setdefault("installed_asset_manifests", [])
        payload.setdefault("imported_packs", [])

        selected_bundle_id = payload.get("selected_bundle_id")
        if selected_bundle_id and not payload.get("selection_key"):
            payload["selection_key"] = build_audio_selection_key(
                selected_bundle_id,
                payload["selected_resource_profile"],
                payload["catalog_version"],
                payload.get("tts_choice"),
            )
        else:
            payload.setdefault("selection_key", None)

        return payload

    @model_validator(mode="after")
    def canonicalize_selection_identity(self) -> "AudioReadinessRecord":
        """Normalize persisted bundle/profile/TTS selection identity when the catalog knows it."""

        if not self.selected_bundle_id:
            self.tts_choice = None
            self.selection_key = None
            return self

        try:
            bundle = get_audio_bundle_catalog().bundle_by_id(self.selected_bundle_id)
            profile = bundle.profile_by_id(self.selected_resource_profile)
            canonical_tts_choice = profile.canonical_tts_choice(self.tts_choice)
        except KeyError:
            if not self.selection_key:
                self.selection_key = build_audio_selection_key(
                    self.selected_bundle_id,
                    self.selected_resource_profile,
                    self.catalog_version,
                    self.tts_choice,
                )
            return self

        self.tts_choice = canonical_tts_choice
        self.selection_key = build_audio_selection_key(
            self.selected_bundle_id,
            self.selected_resource_profile,
            self.catalog_version,
            canonical_tts_choice,
        )
        return self


def _candidate_readiness_files() -> list[Path]:
    candidates: list[Path] = []

    override_file = os.getenv("TLDW_AUDIO_READINESS_FILE")
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
            logger.debug("Audio readiness candidate path not writable")

    logger.warning("No writable location found for audio readiness persistence.")
    return None


class AudioReadinessStore:
    """Read and write the setup audio readiness snapshot."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        default_record = AudioReadinessRecord()
        if not self.path or not self.path.is_file():
            return default_record.model_dump()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return AudioReadinessRecord.model_validate(data).model_dump()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read audio readiness")
            return default_record.model_dump()

    def save(self, readiness: dict[str, Any]) -> dict[str, Any]:
        payload = dict(readiness)
        payload["updated_at"] = _utc_now()
        record = AudioReadinessRecord.model_validate(payload)
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
        return self.save(AudioReadinessRecord().model_dump())


def get_audio_readiness_store() -> AudioReadinessStore:
    global _STORE
    if _STORE is None:
        _STORE = AudioReadinessStore(_resolve_readiness_file())
    return _STORE


def reset_audio_readiness_store() -> None:
    global _STORE
    _STORE = None


__all__ = [
    "AudioReadinessRecord",
    "AudioReadinessStore",
    "READINESS_FILENAME",
    "get_audio_readiness_store",
    "reset_audio_readiness_store",
]
