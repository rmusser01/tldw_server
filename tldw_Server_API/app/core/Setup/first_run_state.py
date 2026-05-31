"""Durable state for the unified first-run setup flow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.api.v1.schemas.setup_schemas import (
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


class FirstRunStateStore:
    """JSON-backed first-run setup state store."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> FirstRunState:
        if not self.path.exists():
            return _default_state()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return FirstRunState.model_validate(payload)

    def save(self, state: FirstRunState) -> FirstRunState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = _now()
        self.path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return state

    def update_step(self, step: str, data: dict[str, Any] | None = None) -> FirstRunState:
        state = self.load()
        if state.status == FirstRunStatus.NOT_STARTED:
            state.status = FirstRunStatus.IN_PROGRESS
        state.current_step = step
        if data is not None:
            state.step_data[step] = data
            if data.get("acknowledged") is True and step not in state.acknowledged_steps:
                state.acknowledged_steps.append(step)
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
            if step not in state.acknowledged_steps
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
