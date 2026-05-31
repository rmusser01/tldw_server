"""Durable state for the unified first-run setup flow."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.core.Setup.first_run_models import (
    FirstRunChatResult,
    FirstRunStateResponse,
    FirstRunStatus,
)

REQUIRED_FIRST_RUN_STEPS = (
    "setup_path",
    "privacy_security",
    "providers",
    "ingest_defaults",
    "audio_defaults",
    "optional_advanced",
)

TERMINAL_STATE_REASONS = {
    FirstRunStatus.BLOCKED: "state_blocked",
    FirstRunStatus.SKIPPED: "state_skipped",
    FirstRunStatus.COMPLETED: "setup_already_completed",
}
SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "bearer",
)
REDACTION_PLACEHOLDER = "********"


class InvalidFirstRunTransition(ValueError):
    """Raised when a setup state transition would violate first-run rules."""


class FirstRunState(FirstRunStateResponse):
    """Internal state model persisted by FirstRunStateStore."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _default_state() -> FirstRunState:
    now = _now()
    return FirstRunState(
        status=FirstRunStatus.NOT_STARTED,
        current_step=None,
        created_at=now,
        updated_at=now,
    )


def _blocked_recovery_state(*, reason: str, quarantined: bool) -> FirstRunState:
    now = _now()
    return FirstRunState(
        status=FirstRunStatus.BLOCKED,
        current_step="state_recovery",
        skip_reason="state_file_recovery",
        created_at=now,
        updated_at=now,
        step_data={
            "state_recovery": {
                "reason": reason,
                "quarantined": quarantined,
                "message": "First-run state could not be read and was moved aside for recovery.",
            }
        },
    )


def _ensure_mutable_state(state: FirstRunState) -> None:
    reason = TERMINAL_STATE_REASONS.get(state.status)
    if reason:
        raise InvalidFirstRunTransition(reason)


def _sync_step_completion(state: FirstRunState, step: str) -> None:
    is_required = step in REQUIRED_FIRST_RUN_STEPS
    is_acknowledged = state.step_data.get(step, {}).get("acknowledged") is True

    if is_required and not is_acknowledged:
        if step in state.completed_steps:
            state.completed_steps.remove(step)
        return

    if step not in state.completed_steps:
        state.completed_steps.append(step)


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _is_sensitive_key(key: object) -> bool:
    normalized_key = _normalized_key(key)
    return any(_normalized_key(marker) in normalized_key for marker in SECRET_KEY_MARKERS)


def _redact_secret_step_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: REDACTION_PLACEHOLDER if _is_sensitive_key(key) else _redact_secret_step_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_step_data(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secret_step_data(item) for item in value]
    return value


class FirstRunStateStore:
    """JSON-backed first-run setup state store."""

    def __init__(
        self,
        path: Path,
        *,
        lock_timeout_seconds: float = 10.0,
        stale_lock_seconds: float = 300.0,
    ):
        self.path = path
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds

    def load(self) -> FirstRunState:
        if not self.path.exists():
            return _default_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._recover_unreadable_state(reason="invalid_json")
        except UnicodeDecodeError:
            return self._recover_unreadable_state(reason="invalid_encoding")
        except OSError:
            return self._recover_unreadable_state(reason="read_error")

        try:
            return FirstRunState.model_validate(payload)
        except ValidationError:
            return self._recover_unreadable_state(reason="invalid_schema")

    def _save_unlocked(self, state: FirstRunState) -> FirstRunState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = _now()
        self._atomic_write(state.model_dump_json(indent=2))
        return state

    def _atomic_write(self, payload: str) -> None:
        tmp_path: str | None = None
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = handle.name

        try:
            os.replace(tmp_path, self.path)
        except Exception:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            raise

    def _recover_unreadable_state(self, *, reason: str) -> FirstRunState:
        quarantined = self._quarantine_bad_state()
        logger.warning("First-run state file unreadable; entering recovery state")
        state = _blocked_recovery_state(reason=reason, quarantined=quarantined)
        return self._save_unlocked(state)

    def _quarantine_bad_state(self) -> bool:
        quarantine_path = self.path.with_name(
            f"{self.path.name}.corrupt-{_now().strftime('%Y%m%d%H%M%S%f')}"
        )
        try:
            os.replace(self.path, quarantine_path)
        except OSError:
            logger.warning("Failed to quarantine unreadable first-run state file")
            return False
        return True

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        deadline = time.monotonic() + self.lock_timeout_seconds
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                if self._recover_stale_lock(lock_path):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("first_run_state_lock_timeout") from exc
                time.sleep(0.05)

        try:
            metadata = {
                "pid": os.getpid(),
                "created_at": _now().isoformat(),
            }
            os.write(fd, json.dumps(metadata).encode("ascii"))
            os.close(fd)
            fd = None
            yield
        finally:
            if fd is not None:
                os.close(fd)
            with suppress(FileNotFoundError):
                lock_path.unlink()

    def _recover_stale_lock(self, lock_path: Path) -> bool:
        if not self._should_recover_lock(lock_path):
            return False
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            logger.warning("Failed to remove stale first-run state lock file")
            return False
        logger.warning("Removed stale first-run state lock file")
        return True

    def _should_recover_lock(self, lock_path: Path) -> bool:
        metadata = self._read_lock_metadata(lock_path)
        return self._lock_is_too_old(metadata) or self._lock_owner_is_dead(metadata)

    def _read_lock_metadata(self, lock_path: Path) -> dict[str, Any]:
        try:
            raw_metadata = lock_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            with suppress(ValueError):
                return {"pid": int(raw_metadata)}
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _lock_is_too_old(self, metadata: dict[str, Any]) -> bool:
        created_at = metadata.get("created_at")
        if not isinstance(created_at, str):
            return False
        try:
            parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed_created_at.tzinfo is None:
            parsed_created_at = parsed_created_at.replace(tzinfo=timezone.utc)
        return (_now() - parsed_created_at).total_seconds() > self.stale_lock_seconds

    def _lock_owner_is_dead(self, metadata: dict[str, Any]) -> bool:
        if os.name != "posix":
            return False
        pid = metadata.get("pid")
        if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return True
        return False

    def _mutate_state(self, mutator: Callable[[FirstRunState], FirstRunState | None]) -> FirstRunState:
        with self._state_lock():
            state = self.load()
            result = mutator(state)
            state = result if result is not None else state
            return self._save_unlocked(state)

    def update_step(self, step: str, data: dict[str, Any] | None = None) -> FirstRunState:
        def _update(state: FirstRunState) -> None:
            _ensure_mutable_state(state)
            if state.status == FirstRunStatus.NOT_STARTED:
                state.status = FirstRunStatus.IN_PROGRESS
            state.current_step = step
            if data is not None:
                state.step_data[step] = _redact_secret_step_data(data)
                if data.get("acknowledged") is True and step not in state.acknowledged_steps:
                    state.acknowledged_steps.append(step)
                elif data.get("acknowledged") is not True and step in state.acknowledged_steps:
                    state.acknowledged_steps.remove(step)
            elif step in REQUIRED_FIRST_RUN_STEPS:
                state.step_data.setdefault(step, {})["acknowledged"] = False
                if step in state.acknowledged_steps:
                    state.acknowledged_steps.remove(step)
            _sync_step_completion(state, step)

        return self._mutate_state(_update)

    def record_first_chat_success(
        self,
        *,
        provider: str,
        model: str,
        response_id: str | None,
    ) -> FirstRunState:
        def _record(state: FirstRunState) -> None:
            _ensure_mutable_state(state)
            state.status = FirstRunStatus.FIRST_CHAT_COMPLETE
            state.first_chat = FirstRunChatResult(
                completed=True,
                provider=provider,
                model=model,
                response_id=response_id,
                completed_at=_now(),
            )
            _sync_step_completion(state, "first_chat")

        return self._mutate_state(_record)

    def mark_completed(self) -> FirstRunState:
        def _complete(state: FirstRunState) -> None:
            _ensure_mutable_state(state)
            if not state.first_chat.completed:
                raise InvalidFirstRunTransition("first_chat_required")
            missing_steps = [
                step
                for step in REQUIRED_FIRST_RUN_STEPS
                if state.step_data.get(step, {}).get("acknowledged") is not True
            ]
            if missing_steps:
                raise InvalidFirstRunTransition(
                    "required_steps_missing:" + ",".join(missing_steps)
                )
            state.status = FirstRunStatus.COMPLETED
            state.completed_at = _now()

        return self._mutate_state(_complete)

    def mark_skipped(self, *, reason: str | None = None) -> FirstRunState:
        def _skip(state: FirstRunState) -> None:
            if state.status == FirstRunStatus.SKIPPED:
                return
            _ensure_mutable_state(state)
            state.status = FirstRunStatus.SKIPPED
            state.skip_reason = reason

        return self._mutate_state(_skip)
