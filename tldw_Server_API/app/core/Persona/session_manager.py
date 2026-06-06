"""
Persona Session Manager (scaffold)

Tracks persona sessions: session_id, user, persona, recent turns.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

_VALID_PERSONA_PLAN_STEP_TYPES = frozenset({"mcp_tool", "skill", "rag_query", "final_answer"})
_TRUNCATION_SUFFIX = "... [truncated]"


def _truncate_text(value: Any, *, max_chars: int) -> tuple[str, bool, int]:
    text = str(value or "")
    original_length = len(text)
    safe_limit = max(1, int(max_chars))
    if original_length <= safe_limit:
        return text, False, original_length
    if safe_limit <= len(_TRUNCATION_SUFFIX):
        return text[:safe_limit], True, original_length
    retained = text[: safe_limit - len(_TRUNCATION_SUFFIX)]
    return f"{retained}{_TRUNCATION_SUFFIX}", True, original_length


def _normalize_turn_metadata(
    metadata: dict[str, Any] | None,
    *,
    max_chars: int,
) -> tuple[dict[str, Any], bool, int]:
    if isinstance(metadata, dict):
        payload: dict[str, Any] = dict(metadata)
    elif metadata is None:
        payload = {}
    else:
        payload = {"value": str(metadata)}

    safe_limit = max(1, int(max_chars))
    try:
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        payload = {"_unserializable_metadata": str(payload)}
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)

    original_length = len(serialized)
    if original_length <= safe_limit:
        return payload, False, original_length

    retained_keys = sorted(str(key) for key in payload.keys())[:32]
    return (
        {
            "_truncated": True,
            "key_count": len(payload),
            "retained_key_count": len(retained_keys),
            "retained_keys": retained_keys,
            "original_char_count": original_length,
        },
        True,
        original_length,
    )


def _infer_step_type_from_tool(tool_name: str) -> str:
    normalized = str(tool_name or "").strip().lower()
    if normalized == "rag_search":
        return "rag_query"
    if normalized == "summarize":
        return "final_answer"
    return "mcp_tool"


def _normalize_step_type(step_type: Any, *, tool_name: str) -> str:
    candidate = str(step_type or "").strip().lower()
    if candidate in _VALID_PERSONA_PLAN_STEP_TYPES:
        return candidate
    return _infer_step_type_from_tool(tool_name)


@dataclass
class PlanStep:
    idx: int
    tool: str
    step_type: str = "mcp_tool"
    args: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    why: str | None = None


@dataclass
class PendingPlan:
    plan_id: str
    session_id: str
    created_at: datetime
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    user_id: str
    persona_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    preferences: dict[str, Any] = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)
    pending_plans: dict[str, PendingPlan] = field(default_factory=dict)


class SessionManager:
    def __init__(
        self,
        *,
        max_turns_per_session: int = 200,
        max_pending_plans_per_session: int = 20,
        session_ttl_seconds: int = 86_400,
        max_sessions_per_user: int = 200,
        max_turn_content_chars: int = 8_192,
        max_turn_metadata_chars: int = 4_096,
    ):
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()
        self._max_turns_per_session = max(1, int(max_turns_per_session))
        self._max_pending_plans_per_session = max(1, int(max_pending_plans_per_session))
        self._session_ttl_seconds = max(1, int(session_ttl_seconds))
        self._max_sessions_per_user = max(1, int(max_sessions_per_user))
        self._max_turn_content_chars = max(16, int(max_turn_content_chars))
        self._max_turn_metadata_chars = max(32, int(max_turn_metadata_chars))

    @staticmethod
    def _touch_session(session: Session) -> None:
        session.updated_at = datetime.now(timezone.utc)

    def _prune_expired_sessions_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._session_ttl_seconds)
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.updated_at < cutoff
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _trim_turns_locked(self, session: Session) -> None:
        if len(session.turns) > self._max_turns_per_session:
            session.turns = session.turns[-self._max_turns_per_session :]

    def _trim_pending_plans_locked(self, session: Session) -> None:
        if len(session.pending_plans) <= self._max_pending_plans_per_session:
            return
        ordered_plan_ids = [
            plan_id
            for _, plan_id in sorted(
                enumerate(session.pending_plans),
                key=lambda item: (session.pending_plans[item[1]].created_at, item[0]),
                reverse=True,
            )
        ]
        keep_ids = set(ordered_plan_ids[: self._max_pending_plans_per_session])
        drop_ids = [plan_id for plan_id in session.pending_plans if plan_id not in keep_ids]
        for plan_id in drop_ids:
            session.pending_plans.pop(plan_id, None)

    def _enforce_user_session_cap_locked(self, user_id: str) -> None:
        user_sessions = [
            session
            for session in self._sessions.values()
            if session.user_id == user_id
        ]
        if len(user_sessions) <= self._max_sessions_per_user:
            return
        user_sessions.sort(key=lambda s: s.updated_at, reverse=True)
        keep_ids = {session.session_id for session in user_sessions[: self._max_sessions_per_user]}
        drop_ids = [
            session.session_id
            for session in user_sessions[self._max_sessions_per_user :]
            if session.session_id not in keep_ids
        ]
        for session_id in drop_ids:
            self._sessions.pop(session_id, None)

    def create(self, user_id: str, persona_id: str, resume_session_id: str | None = None) -> Session:
        with self._lock:
            self._prune_expired_sessions_locked()
            if resume_session_id and resume_session_id in self._sessions:
                existing = self._sessions[resume_session_id]
                if existing.user_id != user_id:
                    raise ValueError("session ownership mismatch")
                if existing.persona_id != persona_id:
                    raise ValueError("session persona mismatch")
                self._touch_session(existing)
                return existing
            sid = resume_session_id or str(uuid.uuid4())
            sess = Session(
                session_id=sid,
                user_id=user_id,
                persona_id=persona_id,
                preferences={"use_memory_context": True},
            )
            self._sessions[sid] = sess
            self._enforce_user_session_cap_locked(user_id)
            return self._sessions[sid]

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._prune_expired_sessions_locked()
            return self._sessions.get(session_id)

    def put_plan(
        self,
        *,
        session_id: str,
        user_id: str,
        persona_id: str,
        plan_id: str,
        steps: list[dict[str, Any]],
    ) -> PendingPlan:
        if not session_id:
            raise ValueError("session_id is required")
        if not plan_id:
            raise ValueError("plan_id is required")
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self.create(user_id=user_id, persona_id=persona_id, resume_session_id=session_id)
            normalized_steps: list[PlanStep] = []
            for raw in steps:
                if not isinstance(raw, dict):
                    continue
                try:
                    idx = int(raw.get("idx"))
                except (TypeError, ValueError):
                    continue
                tool = str(raw.get("tool") or "").strip()
                if not tool:
                    continue
                step_type = _normalize_step_type(raw.get("step_type"), tool_name=tool)
                args = raw.get("args")
                if not isinstance(args, dict):
                    args = {}
                description = raw.get("description")
                if description is not None:
                    description = str(description)
                why = raw.get("why")
                if why is not None:
                    why = str(why)
                normalized_steps.append(
                    PlanStep(
                        idx=idx,
                        tool=tool,
                        step_type=step_type,
                        args=args,
                        description=description,
                        why=why,
                    )
                )
            if not normalized_steps:
                raise ValueError("no valid plan steps")
            normalized_steps.sort(key=lambda step: step.idx)
            pending = PendingPlan(
                plan_id=plan_id,
                session_id=session_id,
                created_at=datetime.now(timezone.utc),
                steps=normalized_steps,
            )
            session.pending_plans[plan_id] = pending
            self._trim_pending_plans_locked(session)
            self._touch_session(session)
            return session.pending_plans[plan_id]

    def get_plan(
        self,
        *,
        session_id: str,
        plan_id: str,
        user_id: str | None = None,
        consume: bool = False,
    ) -> PendingPlan | None:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if user_id is not None and session.user_id != user_id:
                return None
            pending = session.pending_plans.get(plan_id)
            if pending is None:
                return None
            if consume:
                session.pending_plans.pop(plan_id, None)
                self._touch_session(session)
            return pending

    def clear_plans(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> int:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return 0
            if user_id is not None and session.user_id != user_id:
                return 0
            cleared = len(session.pending_plans)
            session.pending_plans.clear()
            if cleared:
                self._touch_session(session)
            return cleared

    def append_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        persona_id: str,
        role: str,
        content: str,
        turn_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self.create(user_id=user_id, persona_id=persona_id, resume_session_id=session_id)
            bounded_content, content_truncated, original_content_length = _truncate_text(
                content,
                max_chars=self._max_turn_content_chars,
            )
            bounded_metadata, metadata_truncated, original_metadata_length = _normalize_turn_metadata(
                metadata,
                max_chars=self._max_turn_metadata_chars,
            )
            if content_truncated or metadata_truncated:
                bounded_metadata = dict(bounded_metadata)
                retention_meta = bounded_metadata.get("_retention")
                if not isinstance(retention_meta, dict):
                    retention_meta = {}
                retention_meta.update(
                    {
                        "content_truncated": bool(content_truncated),
                        "content_original_length": int(original_content_length),
                        "content_retained_length": len(bounded_content),
                        "metadata_truncated": bool(metadata_truncated),
                        "metadata_original_char_count": int(original_metadata_length),
                        "metadata_max_char_count": int(self._max_turn_metadata_chars),
                    }
                )
                bounded_metadata["_retention"] = retention_meta
            turn: dict[str, Any] = {
                "turn_id": uuid.uuid4().hex,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": str(role or "unknown"),
                "type": str(turn_type or "text"),
                "content": bounded_content,
                "metadata": bounded_metadata,
            }
            session.turns.append(turn)
            self._trim_turns_locked(session)
            self._touch_session(session)
            return turn

    def list_turns(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            if user_id is not None and session.user_id != user_id:
                return []
            turns = list(session.turns)
            if limit is not None:
                try:
                    safe_limit = max(0, int(limit))
                except (TypeError, ValueError):
                    safe_limit = 0
                if safe_limit:
                    turns = turns[-safe_limit:]
                else:
                    turns = []
            return turns

    def update_preferences(
        self,
        *,
        session_id: str,
        user_id: str,
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None or session.user_id != user_id:
                raise ValueError("session not found or ownership mismatch")
            for key, value in (preferences or {}).items():
                if not isinstance(key, str) or not key.strip():
                    continue
                if value is None:
                    session.preferences.pop(key, None)
                else:
                    session.preferences[key] = value
            self._touch_session(session)
            return dict(session.preferences)

    def get_preferences(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return {}
            if user_id is not None and session.user_id != user_id:
                return {}
            return dict(session.preferences)

    def list_sessions(
        self,
        *,
        user_id: str,
        persona_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_expired_sessions_locked()
            sessions = [
                session
                for session in self._sessions.values()
                if session.user_id == user_id and (persona_id is None or session.persona_id == persona_id)
            ]
            sessions.sort(key=lambda s: s.updated_at, reverse=True)
            if limit is not None:
                try:
                    safe_limit = max(0, int(limit))
                except (TypeError, ValueError):
                    safe_limit = 0
                sessions = sessions[:safe_limit] if safe_limit else []
            return [
                {
                    "session_id": session.session_id,
                    "persona_id": session.persona_id,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "turn_count": len(session.turns),
                    "pending_plan_count": len(session.pending_plans),
                    "preferences": dict(session.preferences),
                }
                for session in sessions
            ]

    def get_session_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        limit_turns: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            self._prune_expired_sessions_locked()
            session = self._sessions.get(session_id)
            if session is None or session.user_id != user_id:
                return None
            turns = self.list_turns(session_id=session_id, user_id=user_id, limit=limit_turns)
            return {
                "session_id": session.session_id,
                "persona_id": session.persona_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "turn_count": len(session.turns),
                "pending_plan_count": len(session.pending_plans),
                "preferences": dict(session.preferences),
                "turns": turns,
            }


_singleton: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _singleton
    if _singleton is None:
        _singleton = SessionManager()
    return _singleton
