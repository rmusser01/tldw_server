"""Agent/Session state machine with Phase/Activity separation.

Inspired by Scion's agent state model. Phase tracks infrastructure
lifecycle; Activity tracks runtime behavior (only valid when Phase=running).

Sticky activities resist normal event updates until new work arrives.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class SessionPhase(str, Enum):
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class SessionActivity(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_FOR_INPUT = "waiting_for_input"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    LIMITS_EXCEEDED = "limits_exceeded"
    STALLED = "stalled"
    OFFLINE = "offline"


_STICKY_ACTIVITIES = frozenset({
    SessionActivity.WAITING_FOR_INPUT,
    SessionActivity.BLOCKED,
    SessionActivity.COMPLETED,
    SessionActivity.LIMITS_EXCEEDED,
})

_PLATFORM_SET_ACTIVITIES = frozenset({
    SessionActivity.STALLED,
    SessionActivity.OFFLINE,
})


def is_sticky(activity: SessionActivity) -> bool:
    return activity in _STICKY_ACTIVITIES


def is_platform_set(activity: SessionActivity) -> bool:
    return activity in _PLATFORM_SET_ACTIVITIES


def validate_activity(phase: SessionPhase, activity: SessionActivity | None) -> None:
    if activity is not None and phase != SessionPhase.RUNNING:
        raise ValueError(
            f"Activity '{activity}' is only valid when phase is 'running', "
            f"got phase='{phase}'"
        )


def should_update_activity(
    current: SessionActivity | None,
    proposed: SessionActivity,
    *,
    is_new_work: bool = False,
    is_tool_start: bool = False,
) -> bool:
    if is_new_work:
        return True
    if current is None:
        return True
    if is_tool_start and current == SessionActivity.WAITING_FOR_INPUT:
        return True
    if is_sticky(current):
        return False
    return True


@dataclass
class ActivityDetail:
    tool_name: str | None = None
    message: str | None = None
    task_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ActivityDetail:
        if not d:
            return cls()
        return cls(
            tool_name=d.get("tool_name"),
            message=d.get("message"),
            task_summary=d.get("task_summary"),
        )

    @classmethod
    def from_json(cls, s: str | None) -> ActivityDetail:
        if not s:
            return cls()
        return cls.from_dict(json.loads(s))


_STATUS_TO_PHASE = {
    "active": SessionPhase.RUNNING,
    "closed": SessionPhase.STOPPED,
    "error": SessionPhase.ERROR,
}

_STATUS_TO_ACTIVITY = {
    "active": SessionActivity.IDLE,
    "closed": None,
    "error": None,
}


def migrate_legacy_status(status: str) -> tuple[SessionPhase, SessionActivity | None]:
    phase = _STATUS_TO_PHASE.get(status, SessionPhase.ERROR)
    activity = _STATUS_TO_ACTIVITY.get(status)
    return phase, activity
