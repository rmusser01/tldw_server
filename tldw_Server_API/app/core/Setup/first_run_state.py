"""Durable state for the unified first-run setup flow."""

from __future__ import annotations

import json
import os
import tempfile
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


class FirstRunStateStore:
    """JSON-backed first-run setup state store."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> FirstRunState:
        if not self.path.exists():
            return _default_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._recover_unreadable_state(reason="invalid_json")

        try:
            return FirstRunState.model_validate(payload)
        except ValidationError:
            return self._recover_unreadable_state(reason="invalid_schema")

    def save(self, state: FirstRunState) -> FirstRunState:
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
        return self.save(state)

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

    def update_step(self, step: str, data: dict[str, Any] | None = None) -> FirstRunState:
        state = self.load()
        if state.status == FirstRunStatus.NOT_STARTED:
            state.status = FirstRunStatus.IN_PROGRESS
        state.current_step = step
        if data is not None:
            state.step_data[step] = data
            if data.get("acknowledged") is True and step not in state.acknowledged_steps:
                state.acknowledged_steps.append(step)
            elif data.get("acknowledged") is not True and step in state.acknowledged_steps:
                state.acknowledged_steps.remove(step)
        if step not in state.completed_steps:
            state.completed_steps.append(step)
        return self.save(state)

    def record_first_chat_success(
        self,
        *,
        provider: str,
        model: str,
        response_id: str | None,
    ) -> FirstRunState:
        state = self.load()
        state.status = FirstRunStatus.FIRST_CHAT_COMPLETE
        state.first_chat = FirstRunChatResult(
            completed=True,
            provider=provider,
            model=model,
            response_id=response_id,
            completed_at=_now(),
        )
        if "first_chat" not in state.completed_steps:
            state.completed_steps.append("first_chat")
        return self.save(state)

    def mark_completed(self) -> FirstRunState:
        state = self.load()
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
        return self.save(state)

    def mark_skipped(self, *, reason: str | None = None) -> FirstRunState:
        state = self.load()
        state.status = FirstRunStatus.SKIPPED
        state.skip_reason = reason
        return self.save(state)
