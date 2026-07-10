"""Helpers for writing internal deep research artifacts."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import (
    ResearchArtifactRow,
    ResearchSessionsDB,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import normalize_output_storage_filename


class ResearchArtifactStore:
    """Write internal artifacts to disk and register them in the research manifest."""

    def __init__(self, *, base_dir: str | Path, db: ResearchSessionsDB):
        self.base_dir = Path(base_dir).resolve(strict=False)
        self.db = db

    @staticmethod
    def _safe_session_component(session_id: str) -> str:
        raw = str(session_id or "").strip()
        if not raw:
            raise ValueError("session_id must contain at least one safe character")
        return f"session_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"

    def _resolve_artifact_path(self, session_id: str, safe_name: str) -> Path:
        base = (self.base_dir / "research").resolve(strict=False)
        session_dir = (base / self._safe_session_component(session_id)).resolve(strict=False)
        try:
            session_dir.relative_to(base)
        except ValueError as exc:
            raise ValueError("artifact session path escapes research directory") from exc
        session_dir.mkdir(parents=True, exist_ok=True)
        path = (session_dir / safe_name).resolve(strict=False)
        try:
            path.relative_to(session_dir)
        except ValueError as exc:
            raise ValueError("artifact path escapes session directory") from exc
        return path

    @staticmethod
    def _safe_artifact_name(artifact_name: str) -> str:
        return normalize_output_storage_filename(
            artifact_name,
            allow_absolute=False,
            reject_relative_with_separators=True,
            expand_user=False,
        )

    @staticmethod
    def _storage_file_name(safe_name: str) -> str:
        suffix = "".join(Path(safe_name).suffixes)
        digest = hashlib.sha256(safe_name.encode("utf-8")).hexdigest()
        return f"artifact_{digest}{suffix}"

    def _artifact_path(self, session_id: str, artifact_name: str) -> Path:
        safe_name = self._safe_artifact_name(artifact_name)
        storage_name = self._storage_file_name(safe_name)
        return self._resolve_artifact_path(session_id, storage_name)

    def _versioned_artifact_path(self, session_id: str, artifact_name: str, artifact_version: int) -> Path:
        path = self._artifact_path(session_id, artifact_name)
        unique_suffix = uuid.uuid4().hex
        suffix = "".join(path.suffixes)
        if suffix:
            stem = path.name[: -len(suffix)]
            return path.with_name(f"{stem}.v{artifact_version}.{unique_suffix}{suffix}")
        return path.with_name(f"{path.name}.v{artifact_version}.{unique_suffix}")

    @staticmethod
    def _write_bytes_atomically(path: Path, encoded: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_bytes(encoded)
        tmp_path.replace(path)

    def _next_version(self, session_id: str, artifact_name: str) -> int:
        existing = [
            artifact.artifact_version
            for artifact in self.db.list_artifacts(session_id)
            if artifact.artifact_name == artifact_name
        ]
        return (max(existing) + 1) if existing else 1

    def _latest_artifact(self, session_id: str, artifact_name: str) -> ResearchArtifactRow | None:
        safe_name = self._safe_artifact_name(artifact_name)
        matches = [
            artifact
            for artifact in self.db.list_artifacts(session_id)
            if artifact.artifact_name == safe_name
        ]
        if not matches:
            return None
        return max(matches, key=lambda artifact: artifact.artifact_version)

    @staticmethod
    def _artifact_event_payload(
        *,
        artifact_name: str,
        artifact_version: int,
        content_type: str,
        phase: str,
        job_id: str | None,
    ) -> dict[str, Any]:
        return {
            "artifact_name": artifact_name,
            "artifact_version": artifact_version,
            "content_type": content_type,
            "phase": phase,
            "job_id": job_id,
        }

    def write_json(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        artifact_name: str,
        payload: dict[str, Any],
        phase: str,
        job_id: str | None,
    ) -> ResearchArtifactRow:
        owner_user_id = str(owner_user_id)
        safe_name = self._safe_artifact_name(artifact_name)
        next_version = self._next_version(session_id, safe_name)
        path = self._versioned_artifact_path(session_id, safe_name, next_version)
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self._write_bytes_atomically(path, encoded)
        self._write_bytes_atomically(self._artifact_path(session_id, safe_name), encoded)
        artifact, _ = self.db.record_artifact_with_event(
            owner_user_id=owner_user_id,
            session_id=session_id,
            artifact_name=safe_name,
            artifact_version=next_version,
            storage_path=str(path),
            content_type="application/json",
            byte_size=len(encoded),
            checksum=hashlib.sha256(encoded).hexdigest(),
            phase=phase,
            job_id=job_id,
            event_type="artifact",
            event_payload=self._artifact_event_payload(
                artifact_name=safe_name,
                artifact_version=next_version,
                content_type="application/json",
                phase=phase,
                job_id=job_id,
            ),
        )
        return artifact

    def write_jsonl(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        artifact_name: str,
        records: list[dict[str, Any]],
        phase: str,
        job_id: str | None,
    ) -> ResearchArtifactRow:
        owner_user_id = str(owner_user_id)
        safe_name = self._safe_artifact_name(artifact_name)
        next_version = self._next_version(session_id, safe_name)
        path = self._versioned_artifact_path(session_id, safe_name, next_version)
        encoded = "".join(
            f"{json.dumps(record, sort_keys=True)}\n"
            for record in records
        ).encode("utf-8")
        self._write_bytes_atomically(path, encoded)
        self._write_bytes_atomically(self._artifact_path(session_id, safe_name), encoded)
        artifact, _ = self.db.record_artifact_with_event(
            owner_user_id=owner_user_id,
            session_id=session_id,
            artifact_name=safe_name,
            artifact_version=next_version,
            storage_path=str(path),
            content_type="application/x-ndjson",
            byte_size=len(encoded),
            checksum=hashlib.sha256(encoded).hexdigest(),
            phase=phase,
            job_id=job_id,
            event_type="artifact",
            event_payload=self._artifact_event_payload(
                artifact_name=safe_name,
                artifact_version=next_version,
                content_type="application/x-ndjson",
                phase=phase,
                job_id=job_id,
            ),
        )
        return artifact

    def write_text(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        artifact_name: str,
        content: str,
        phase: str,
        job_id: str | None,
        content_type: str = "text/plain",
    ) -> ResearchArtifactRow:
        owner_user_id = str(owner_user_id)
        safe_name = self._safe_artifact_name(artifact_name)
        next_version = self._next_version(session_id, safe_name)
        path = self._versioned_artifact_path(session_id, safe_name, next_version)
        encoded = content.encode("utf-8")
        self._write_bytes_atomically(path, encoded)
        self._write_bytes_atomically(self._artifact_path(session_id, safe_name), encoded)
        artifact, _ = self.db.record_artifact_with_event(
            owner_user_id=owner_user_id,
            session_id=session_id,
            artifact_name=safe_name,
            artifact_version=next_version,
            storage_path=str(path),
            content_type=content_type,
            byte_size=len(encoded),
            checksum=hashlib.sha256(encoded).hexdigest(),
            phase=phase,
            job_id=job_id,
            event_type="artifact",
            event_payload=self._artifact_event_payload(
                artifact_name=safe_name,
                artifact_version=next_version,
                content_type=content_type,
                phase=phase,
                job_id=job_id,
            ),
        )
        return artifact

    def read_json(self, *, session_id: str, artifact_name: str) -> dict[str, Any] | None:
        artifact = self._latest_artifact(session_id, artifact_name)
        if artifact is None:
            return None
        path = Path(artifact.storage_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def read_jsonl(self, *, session_id: str, artifact_name: str) -> list[dict[str, Any]] | None:
        artifact = self._latest_artifact(session_id, artifact_name)
        if artifact is None:
            return None
        path = Path(artifact.storage_path)
        if not path.exists():
            return None
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
                else:
                    return None
        except json.JSONDecodeError:
            return None
        return records

    def read_text(self, *, session_id: str, artifact_name: str) -> str | None:
        artifact = self._latest_artifact(session_id, artifact_name)
        if artifact is None:
            return None
        path = Path(artifact.storage_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
