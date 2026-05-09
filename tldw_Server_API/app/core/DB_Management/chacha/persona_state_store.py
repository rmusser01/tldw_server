from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    logger,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.chacha import exemplar_normalization
from tldw_Server_API.app.core.Persona.buddy import resolve_persona_buddy_profile

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_UNSET = object()


class PersonaStateStore:
    """Focused persistence seam for persona state and analytics operations."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        """Coerce mixed persistence values into a boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _normalize_deleted_input(value: Any) -> bool:
        """Normalize soft-delete inputs for persona records."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0"}
        return bool(value)

    @staticmethod
    def _parse_version_input(value: Any) -> int:
        """Parse optimistic-lock version values and require positive integers."""
        try:
            version = int(value)
        except (TypeError, ValueError) as exc:
            raise InputError("version must be an integer >= 1.") from exc  # noqa: TRY003
        if version < 1:
            raise InputError("version must be >= 1.")  # noqa: TRY003
        return version

    def _normalize_persona_mode(self, value: Any) -> str:
        """Normalize persona mode into one of the allowed lowercase values."""
        mode = str(value or self._DEFAULT_PERSONA_MODE).strip().lower()
        if mode not in self._ALLOWED_PERSONA_MODES:
            allowed = ", ".join(self._ALLOWED_PERSONA_MODES)
            raise InputError(f"Invalid persona mode '{mode}'. Allowed: {allowed}.")  # noqa: TRY003
        return mode

    def _normalize_persona_scope_rule_type(self, value: Any) -> str:
        """Normalize scope rule types for persona rule persistence."""
        rule_type = str(value or "").strip().lower()
        if rule_type not in self._ALLOWED_PERSONA_SCOPE_RULE_TYPES:
            allowed = ", ".join(self._ALLOWED_PERSONA_SCOPE_RULE_TYPES)
            raise InputError(
                f"Invalid persona scope rule_type '{rule_type}'. Allowed: {allowed}."
            )  # noqa: TRY003
        return rule_type

    def _normalize_persona_policy_rule_kind(self, value: Any) -> str:
        """Normalize policy rule kinds for persona rule persistence."""
        rule_kind = str(value or "").strip().lower()
        if rule_kind not in self._ALLOWED_PERSONA_POLICY_RULE_KINDS:
            allowed = ", ".join(self._ALLOWED_PERSONA_POLICY_RULE_KINDS)
            raise InputError(
                f"Invalid persona policy rule_kind '{rule_kind}'. Allowed: {allowed}."
            )  # noqa: TRY003
        return rule_kind

    def _normalize_persona_session_status(self, value: Any) -> str:
        """Normalize persona session statuses into the allowed set."""
        status = str(value or "active").strip().lower()
        if status not in self._ALLOWED_PERSONA_SESSION_STATUSES:
            allowed = ", ".join(self._ALLOWED_PERSONA_SESSION_STATUSES)
            raise InputError(
                f"Invalid persona session status '{status}'. Allowed: {allowed}."
            )  # noqa: TRY003
        return status

    def _ensure_persona_live_voice_session_summaries_table(self) -> None:
        if self.backend_type == BackendType.SQLITE:
            self.execute_query(
                """
                CREATE TABLE IF NOT EXISTS persona_live_voice_session_summaries(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  persona_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  started_at TEXT,
                  ended_at TEXT,
                  auto_commit_enabled INTEGER,
                  vad_threshold REAL,
                  min_silence_ms INTEGER,
                  turn_stop_secs REAL,
                  min_utterance_secs REAL,
                  turn_detection_changed_during_session INTEGER NOT NULL DEFAULT 0,
                  total_committed_turns INTEGER NOT NULL DEFAULT 0,
                  vad_auto_commit_count INTEGER NOT NULL DEFAULT 0,
                  manual_commit_count INTEGER NOT NULL DEFAULT 0,
                  manual_mode_required_count INTEGER NOT NULL DEFAULT 0,
                  text_only_tts_count INTEGER NOT NULL DEFAULT 0,
                  listening_recovery_count INTEGER NOT NULL DEFAULT 0,
                  thinking_recovery_count INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(user_id, persona_id, session_id)
                )
                """,
                script=False,
                commit=True,
            )
            self.execute_query(
                """
                CREATE INDEX IF NOT EXISTS idx_persona_live_voice_session_summaries_persona_time
                ON persona_live_voice_session_summaries(persona_id, started_at, updated_at)
                """,
                script=False,
                commit=True,
            )
            return

        if self.backend_type == BackendType.POSTGRESQL:
            self.backend.execute(
                """
                CREATE TABLE IF NOT EXISTS persona_live_voice_session_summaries(
                  id BIGSERIAL PRIMARY KEY,
                  user_id BIGINT NOT NULL,
                  persona_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                  started_at TEXT,
                  ended_at TEXT,
                  auto_commit_enabled BOOLEAN,
                  vad_threshold DOUBLE PRECISION,
                  min_silence_ms INTEGER,
                  turn_stop_secs DOUBLE PRECISION,
                  min_utterance_secs DOUBLE PRECISION,
                  turn_detection_changed_during_session BOOLEAN NOT NULL DEFAULT FALSE,
                  total_committed_turns INTEGER NOT NULL DEFAULT 0,
                  vad_auto_commit_count INTEGER NOT NULL DEFAULT 0,
                  manual_commit_count INTEGER NOT NULL DEFAULT 0,
                  manual_mode_required_count INTEGER NOT NULL DEFAULT 0,
                  text_only_tts_count INTEGER NOT NULL DEFAULT 0,
                  listening_recovery_count INTEGER NOT NULL DEFAULT 0,
                  thinking_recovery_count INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(user_id, persona_id, session_id)
                )
                """
            )
            self.backend.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_persona_live_voice_session_summaries_persona_time
                ON persona_live_voice_session_summaries(persona_id, started_at, updated_at)
                """
            )
            return

        raise NotImplementedError(
            "persona_live_voice_session_summaries table creation not supported "
            f"for backend {self.backend_type.value}"
        )

    def upsert_persona_live_voice_session_summary(
        self,
        *,
        user_id: int,
        persona_id: str,
        session_id: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        auto_commit_enabled: bool | None = None,
        vad_threshold: float | None = None,
        min_silence_ms: int | None = None,
        turn_stop_secs: float | None = None,
        min_utterance_secs: float | None = None,
        commit_source: str | None = None,
        manual_mode_required_increment: int = 0,
        text_only_tts_increment: int = 0,
        listening_recovery_count: int | None = None,
        thinking_recovery_count: int | None = None,
        finalize: bool = False,
    ) -> bool:
        self._ensure_persona_live_voice_session_summaries_table()
        now = self._get_current_utc_timestamp_iso()

        def _bool_for_db(value: bool | None) -> bool | int | None:
            if value is None:
                return None
            if self.backend_type == BackendType.POSTGRESQL:
                return bool(value)
            return int(bool(value))

        snapshot_values = {
            "auto_commit_enabled": _bool_for_db(auto_commit_enabled),
            "vad_threshold": float(vad_threshold) if vad_threshold is not None else None,
            "min_silence_ms": int(min_silence_ms) if min_silence_ms is not None else None,
            "turn_stop_secs": float(turn_stop_secs) if turn_stop_secs is not None else None,
            "min_utterance_secs": (
                float(min_utterance_secs) if min_utterance_secs is not None else None
            ),
        }
        snapshot_provided = any(value is not None for value in snapshot_values.values())

        with self.transaction():
            row = self.execute_query(
                """
                SELECT *
                FROM persona_live_voice_session_summaries
                WHERE user_id = ? AND persona_id = ? AND session_id = ?
                """,
                (user_id, persona_id, session_id),
            ).fetchone()

            if row is None:
                total_committed_turns = 1 if commit_source in {"vad_auto", "manual"} else 0
                vad_auto_commit_count = 1 if commit_source == "vad_auto" else 0
                manual_commit_count = 1 if commit_source == "manual" else 0
                normalized_started_at = started_at or now
                normalized_ended_at = ended_at or (now if finalize else None)
                self.execute_query(
                    """
                    INSERT INTO persona_live_voice_session_summaries(
                        user_id, persona_id, session_id, created_at, updated_at,
                        started_at, ended_at, auto_commit_enabled, vad_threshold,
                        min_silence_ms, turn_stop_secs, min_utterance_secs,
                        turn_detection_changed_during_session, total_committed_turns,
                        vad_auto_commit_count, manual_commit_count,
                        manual_mode_required_count, text_only_tts_count,
                        listening_recovery_count, thinking_recovery_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        persona_id,
                        session_id,
                        now,
                        now,
                        normalized_started_at,
                        normalized_ended_at,
                        snapshot_values["auto_commit_enabled"],
                        snapshot_values["vad_threshold"],
                        snapshot_values["min_silence_ms"],
                        snapshot_values["turn_stop_secs"],
                        snapshot_values["min_utterance_secs"],
                        _bool_for_db(False),
                        total_committed_turns,
                        vad_auto_commit_count,
                        manual_commit_count,
                        max(0, int(manual_mode_required_increment or 0)),
                        max(0, int(text_only_tts_increment or 0)),
                        max(0, int(listening_recovery_count or 0)),
                        max(0, int(thinking_recovery_count or 0)),
                    ),
                )
                return True

            existing = dict(row)
            updates: list[str] = ["updated_at = ?"]
            params: list[Any] = [now]

            existing_snapshot_values = {
                "auto_commit_enabled": existing.get("auto_commit_enabled"),
                "vad_threshold": existing.get("vad_threshold"),
                "min_silence_ms": existing.get("min_silence_ms"),
                "turn_stop_secs": existing.get("turn_stop_secs"),
                "min_utterance_secs": existing.get("min_utterance_secs"),
            }
            existing_snapshot_missing = all(
                value is None for value in existing_snapshot_values.values()
            )
            if snapshot_provided and existing_snapshot_missing:
                for field_name, value in snapshot_values.items():
                    updates.append(f"{field_name} = ?")
                    params.append(value)
            elif snapshot_provided:
                snapshot_changed = any(
                    snapshot_values[field_name] is not None
                    and snapshot_values[field_name] != existing_snapshot_values[field_name]
                    for field_name in snapshot_values
                )
                if snapshot_changed and not bool(
                    existing.get("turn_detection_changed_during_session")
                ):
                    updates.append("turn_detection_changed_during_session = ?")
                    params.append(_bool_for_db(True))

            if started_at and not existing.get("started_at"):
                updates.append("started_at = ?")
                params.append(started_at)

            if commit_source == "vad_auto":
                updates.extend(
                    [
                        "total_committed_turns = total_committed_turns + 1",
                        "vad_auto_commit_count = vad_auto_commit_count + 1",
                    ]
                )
            elif commit_source == "manual":
                updates.extend(
                    [
                        "total_committed_turns = total_committed_turns + 1",
                        "manual_commit_count = manual_commit_count + 1",
                    ]
                )

            if manual_mode_required_increment:
                updates.append(
                    "manual_mode_required_count = manual_mode_required_count + ?"
                )
                params.append(max(0, int(manual_mode_required_increment)))

            if text_only_tts_increment:
                updates.append("text_only_tts_count = text_only_tts_count + ?")
                params.append(max(0, int(text_only_tts_increment)))

            if listening_recovery_count is not None:
                updates.append("listening_recovery_count = ?")
                params.append(max(0, int(listening_recovery_count)))

            if thinking_recovery_count is not None:
                updates.append("thinking_recovery_count = ?")
                params.append(max(0, int(thinking_recovery_count)))

            if ended_at:
                updates.append("ended_at = ?")
                params.append(ended_at)
            elif finalize:
                updates.append("ended_at = ?")
                params.append(now)

            params.extend([user_id, persona_id, session_id])
            self.execute_query(
                f"""
                UPDATE persona_live_voice_session_summaries
                SET {", ".join(updates)}
                WHERE user_id = ? AND persona_id = ? AND session_id = ?
                """,  # nosec B608
                tuple(params),
            )
            return True

    def list_persona_live_voice_session_summaries(
        self,
        *,
        user_id: int,
        persona_id: str,
        days: int = 7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self._ensure_persona_live_voice_session_summaries_table()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        cursor = self.execute_query(
            """
            SELECT *
            FROM persona_live_voice_session_summaries
            WHERE user_id = ? AND persona_id = ?
              AND COALESCE(started_at, created_at) >= ?
            ORDER BY COALESCE(started_at, created_at) DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, persona_id, cutoff, max(1, int(limit))),
        )
        return [dict(row) for row in cursor.fetchall() if row]

    def get_persona_live_voice_session_summary(
        self,
        *,
        user_id: int,
        persona_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        self._ensure_persona_live_voice_session_summaries_table()
        row = self.execute_query(
            """
            SELECT *
            FROM persona_live_voice_session_summaries
            WHERE user_id = ? AND persona_id = ? AND session_id = ?
            """,
            (user_id, persona_id, session_id),
        ).fetchone()
        return dict(row) if row else None

    def _ensure_persona_setup_events_table(self) -> None:
        if self.backend_type == BackendType.SQLITE:
            self.execute_query(
                """
                CREATE TABLE IF NOT EXISTS persona_setup_events(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_id TEXT NOT NULL UNIQUE,
                  user_id INTEGER NOT NULL,
                  persona_id TEXT NOT NULL,
                  run_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  event_key TEXT,
                  step TEXT,
                  completion_type TEXT,
                  detour_source TEXT,
                  action_target TEXT,
                  metadata_json TEXT,
                  created_at TEXT NOT NULL
                )
                """,
                script=False,
                commit=True,
            )
            self.execute_query(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_setup_events_event_key
                ON persona_setup_events(user_id, persona_id, run_id, event_key)
                WHERE event_key IS NOT NULL
                """,
                script=False,
                commit=True,
            )
            self.execute_query(
                """
                CREATE INDEX IF NOT EXISTS idx_persona_setup_events_persona_created
                ON persona_setup_events(user_id, persona_id, created_at, id)
                """,
                script=False,
                commit=True,
            )
            return

        if self.backend_type == BackendType.POSTGRESQL:
            self.backend.execute(
                """
                CREATE TABLE IF NOT EXISTS persona_setup_events(
                  id BIGSERIAL PRIMARY KEY,
                  event_id TEXT NOT NULL UNIQUE,
                  user_id BIGINT NOT NULL,
                  persona_id TEXT NOT NULL,
                  run_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  event_key TEXT,
                  step TEXT,
                  completion_type TEXT,
                  detour_source TEXT,
                  action_target TEXT,
                  metadata_json TEXT,
                  created_at TEXT NOT NULL
                )
                """
            )
            self.backend.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_setup_events_event_key
                ON persona_setup_events(user_id, persona_id, run_id, event_key)
                WHERE event_key IS NOT NULL
                """
            )
            self.backend.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_persona_setup_events_persona_created
                ON persona_setup_events(user_id, persona_id, created_at, id)
                """
            )
            return

        raise NotImplementedError(
            "persona_setup_events table creation not supported "
            f"for backend {self.backend_type.value}"
        )

    def _persona_setup_event_row_to_dict(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        event_label = str(item.get("event_id") or item.get("id") or "unknown")
        item["metadata"] = self._decode_persona_json_object(
            item.get("metadata_json"),
            field_name="metadata_json",
            context_label=f"persona setup event {event_label}",
        )
        return item

    def _decode_persona_json_object(
        self,
        raw_value: Any,
        *,
        field_name: str,
        context_label: str,
    ) -> dict[str, Any]:
        if isinstance(raw_value, dict):
            return raw_value
        if not isinstance(raw_value, str):
            return {}

        try:
            decoded_value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid JSON in {} for {}: {}",
                field_name,
                context_label,
                exc,
            )
            return {}

        if isinstance(decoded_value, dict):
            return decoded_value

        logger.warning(
            "Expected JSON object in {} for {}, got {}.",
            field_name,
            context_label,
            type(decoded_value).__name__,
        )
        return {}

    def record_persona_setup_event(
        self,
        *,
        user_id: int,
        persona_id: str,
        event_id: str,
        run_id: str,
        event_type: str,
        event_key: str | None = None,
        step: str | None = None,
        completion_type: str | None = None,
        detour_source: str | None = None,
        action_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_persona_setup_events_table()
        now = self._get_current_utc_timestamp_iso()
        metadata_json = self._ensure_json_string(metadata if isinstance(metadata, dict) else {})

        with self.transaction():
            row = self.execute_query(
                """
                SELECT *
                FROM persona_setup_events
                WHERE user_id = ? AND persona_id = ? AND event_id = ?
                """,
                (user_id, persona_id, event_id),
            ).fetchone()
            if row:
                item = self._persona_setup_event_row_to_dict(row)
                item["deduped"] = True
                return item

            if event_key:
                row = self.execute_query(
                    """
                    SELECT *
                    FROM persona_setup_events
                    WHERE user_id = ? AND persona_id = ? AND run_id = ? AND event_key = ?
                    """,
                    (user_id, persona_id, run_id, event_key),
                ).fetchone()
                if row:
                    item = self._persona_setup_event_row_to_dict(row)
                    item["deduped"] = True
                    return item

            self.execute_query(
                """
                INSERT INTO persona_setup_events(
                    event_id, user_id, persona_id, run_id, event_type, event_key,
                    step, completion_type, detour_source, action_target,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    user_id,
                    persona_id,
                    run_id,
                    event_type,
                    event_key,
                    step,
                    completion_type,
                    detour_source,
                    action_target,
                    metadata_json,
                    now,
                ),
            )
            row = self.execute_query(
                """
                SELECT *
                FROM persona_setup_events
                WHERE user_id = ? AND persona_id = ? AND event_id = ?
                """,
                (user_id, persona_id, event_id),
            ).fetchone()
            if not row:
                raise CharactersRAGDBError("Failed to load inserted persona setup event.")  # noqa: TRY003
            item = self._persona_setup_event_row_to_dict(row)
            item["deduped"] = False
            return item

    def list_persona_setup_events(
        self,
        *,
        user_id: int,
        persona_id: str,
        days: int = 30,
        run_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        self._ensure_persona_setup_events_table()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        params: list[Any] = [user_id, persona_id, cutoff]
        query = """
            SELECT *
            FROM persona_setup_events
            WHERE user_id = ? AND persona_id = ? AND created_at >= ?
        """
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))

        cursor = self.execute_query(query, tuple(params))
        return [self._persona_setup_event_row_to_dict(row) for row in cursor.fetchall() if row]

    def get_persona_setup_analytics_summary(
        self,
        *,
        user_id: int,
        persona_id: str,
        days: int = 30,
        recent_run_limit: int = 10,
    ) -> dict[str, Any]:
        events = self.list_persona_setup_events(
            user_id=user_id,
            persona_id=persona_id,
            days=days,
            limit=5000,
        )
        runs: dict[str, dict[str, Any]] = {}
        detour_started_counts: dict[str, int] = {}
        detour_returned_counts: dict[str, int] = {}

        for event in events:
            run_id = str(event.get("run_id") or "").strip()
            if not run_id:
                continue
            run = runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "started_at": None,
                    "completed_at": None,
                    "completion_type": None,
                    "terminal_step": None,
                    "handoff_clicked": False,
                    "handoff_target_reached": False,
                    "handoff_dismissed": False,
                    "first_post_setup_action": False,
                    "_reached_targets": set(),
                    "_latest_id": -1,
                    "_earliest_created_at": None,
                    "_latest_step_id": -1,
                },
            )
            event_id_value = int(event.get("id") or 0)
            created_at = str(event.get("created_at") or "").strip() or None
            event_type = str(event.get("event_type") or "").strip()
            step = str(event.get("step") or "").strip() or None
            detour_source = str(event.get("detour_source") or "").strip() or None

            if event_id_value > int(run["_latest_id"]):
                run["_latest_id"] = event_id_value
            if created_at and (
                run["_earliest_created_at"] is None
                or created_at < str(run["_earliest_created_at"])
            ):
                run["_earliest_created_at"] = created_at
            if event_type == "setup_started" and created_at and (
                run["started_at"] is None or created_at < str(run["started_at"])
            ):
                run["started_at"] = created_at
            if event_type == "setup_completed" and created_at and (
                run["completed_at"] is None or created_at > str(run["completed_at"])
            ):
                run["completed_at"] = created_at
                run["completion_type"] = (
                    str(event.get("completion_type") or "").strip() or None
                )
            if step and event_id_value >= int(run["_latest_step_id"]):
                run["_latest_step_id"] = event_id_value
                run["terminal_step"] = step
            if event_type == "handoff_action_clicked":
                run["handoff_clicked"] = True
            elif event_type == "handoff_target_reached":
                run["handoff_target_reached"] = True
                action_target = str(event.get("action_target") or "").strip()
                if action_target:
                    reached_targets = run["_reached_targets"]
                    if isinstance(reached_targets, set):
                        reached_targets.add(action_target)
            elif event_type == "handoff_dismissed":
                run["handoff_dismissed"] = True
            elif event_type == "first_post_setup_action":
                run["first_post_setup_action"] = True
            elif event_type == "detour_started" and detour_source:
                detour_started_counts[detour_source] = (
                    detour_started_counts.get(detour_source, 0) + 1
                )
            elif event_type == "detour_returned" and detour_source:
                detour_returned_counts[detour_source] = (
                    detour_returned_counts.get(detour_source, 0) + 1
                )

        recent_runs: list[dict[str, Any]] = []
        completed_runs = 0
        dry_run_completion_count = 0
        live_session_completion_count = 0
        handoff_clicked_runs = 0
        handoff_target_reached_runs = 0
        first_post_setup_action_runs = 0
        handoff_target_reached_counts: dict[str, int] = {}
        dropoff_counts: dict[str, int] = {}

        for run in sorted(
            runs.values(),
            key=lambda item: int(item["_latest_id"]),
            reverse=True,
        ):
            if run["started_at"] is None:
                run["started_at"] = run["_earliest_created_at"]
            completion_type = str(run.get("completion_type") or "").strip() or None
            if completion_type:
                completed_runs += 1
                if completion_type == "dry_run":
                    dry_run_completion_count += 1
                elif completion_type == "live_session":
                    live_session_completion_count += 1
            else:
                terminal_step = str(run.get("terminal_step") or "").strip() or None
                if terminal_step:
                    dropoff_counts[terminal_step] = dropoff_counts.get(terminal_step, 0) + 1

            if bool(run.get("handoff_clicked")):
                handoff_clicked_runs += 1
            if bool(run.get("handoff_target_reached")):
                handoff_target_reached_runs += 1
            if bool(run.get("first_post_setup_action")):
                first_post_setup_action_runs += 1
            reached_targets = run.get("_reached_targets")
            if isinstance(reached_targets, set):
                for action_target in reached_targets:
                    handoff_target_reached_counts[action_target] = (
                        handoff_target_reached_counts.get(action_target, 0) + 1
                    )

            recent_runs.append(
                {
                    "run_id": run["run_id"],
                    "started_at": run["started_at"],
                    "completed_at": run["completed_at"],
                    "completion_type": completion_type,
                    "terminal_step": run["terminal_step"],
                    "handoff_clicked": bool(run["handoff_clicked"]),
                    "handoff_target_reached": bool(run["handoff_target_reached"]),
                    "handoff_dismissed": bool(run["handoff_dismissed"]),
                    "first_post_setup_action": bool(run["first_post_setup_action"]),
                }
            )

        total_runs = len(runs)
        most_common_dropoff_step = None
        if dropoff_counts:
            most_common_dropoff_step = max(
                dropoff_counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]

        return {
            "summary": {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "completion_rate": (
                    float(completed_runs) / float(total_runs)
                    if total_runs
                    else 0.0
                ),
                "dry_run_completion_count": dry_run_completion_count,
                "live_session_completion_count": live_session_completion_count,
                "most_common_dropoff_step": most_common_dropoff_step,
                "handoff_click_rate": (
                    float(handoff_clicked_runs) / float(total_runs)
                    if total_runs
                    else 0.0
                ),
                "handoff_target_reach_rate": (
                    float(handoff_target_reached_runs) / float(handoff_clicked_runs)
                    if handoff_clicked_runs
                    else 0.0
                ),
                "first_post_setup_action_rate": (
                    float(first_post_setup_action_runs) / float(total_runs)
                    if total_runs
                    else 0.0
                ),
                "handoff_target_reached_counts": handoff_target_reached_counts,
                "detour_started_counts": detour_started_counts,
                "detour_returned_counts": detour_returned_counts,
            },
            "recent_runs": recent_runs[: max(1, int(recent_run_limit))],
        }

    def _persona_profile_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        persona_label = str(item.get("id") or item.get("name") or "unknown")
        item["voice_defaults"] = self._decode_persona_json_object(
            item.get("voice_defaults_json"),
            field_name="voice_defaults_json",
            context_label=f"persona profile {persona_label}",
        )
        item["setup"] = self._decode_persona_json_object(
            item.get("setup_json"),
            field_name="setup_json",
            context_label=f"persona profile {persona_label}",
        )
        item["is_active"] = self._as_bool(item.get("is_active"))
        item["use_persona_state_context_default"] = self._as_bool(
            item.get("use_persona_state_context_default", True)
        )
        item["deleted"] = self._as_bool(item.get("deleted"))
        return item

    def _persona_buddy_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        buddy_label = str(item.get("persona_id") or "unknown")
        item["derived_core"] = self._decode_persona_json_object(
            item.get("derived_core_json"),
            field_name="derived_core_json",
            context_label=f"persona buddy {buddy_label}",
        )
        item["overlay_preferences"] = self._decode_persona_json_object(
            item.get("overlay_preferences_json"),
            field_name="overlay_preferences_json",
            context_label=f"persona buddy {buddy_label}",
        )
        try:
            item["resolved_profile"] = resolve_persona_buddy_profile(
                derived_core=item["derived_core"],
                overlay_preferences=item["overlay_preferences"],
            )
        except (TypeError, KeyError, ValueError) as exc:
            logger.warning(
                "Unable to resolve persona buddy profile for persona_id={}: {}",
                buddy_label,
                exc,
            )
            raise CharactersRAGDBError(
                f"Unable to resolve persona buddy profile for persona_id={buddy_label}."
            ) from exc
        return item

    def _decode_persona_json_list(
        self,
        raw_value: Any,
        *,
        field_name: str,
        context_label: str,
    ) -> list[Any]:
        if isinstance(raw_value, list):
            return raw_value
        if isinstance(raw_value, str):
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode JSON list for field '{}' in {}. Falling back to empty list. Value preview: {}",
                    field_name,
                    context_label,
                    raw_value[:100] + ("..." if len(raw_value) > 100 else ""),
                )
                return []
            return decoded if isinstance(decoded, list) else []
        return []

    def _persona_visual_pack_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        pack_label = str(item.get("id") or "unknown")
        item["manifest"] = self._decode_persona_json_object(
            item.get("manifest_json"),
            field_name="manifest_json",
            context_label=f"persona visual pack {pack_label}",
        )
        item["deleted"] = self._as_bool(item.get("deleted"))
        return item

    def _persona_visual_asset_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["deleted"] = self._as_bool(item.get("deleted"))
        for field_name in ("byte_size", "width", "height", "duration_ms", "version"):
            value = item.get(field_name)
            if value is None:
                continue
            try:
                item[field_name] = int(value)
            except (TypeError, ValueError):
                item[field_name] = None
        return item

    def _persona_visual_candidate_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        candidate_label = str(item.get("id") or "unknown")
        item["proposed_manifest_patch"] = self._decode_persona_json_object(
            item.get("proposed_manifest_patch_json"),
            field_name="proposed_manifest_patch_json",
            context_label=f"persona visual candidate {candidate_label}",
        )
        item["generated_asset_ids"] = self._decode_persona_json_list(
            item.get("generated_asset_ids_json"),
            field_name="generated_asset_ids_json",
            context_label=f"persona visual candidate {candidate_label}",
        )
        item["deleted"] = self._as_bool(item.get("deleted"))
        return item

    def _persona_visual_library_item_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item_label = str(item.get("id") or "unknown")
        item["tags"] = self._decode_persona_json_list(
            item.get("tags_json"),
            field_name="tags_json",
            context_label=f"persona visual library item {item_label}",
        )
        item["deleted"] = self._as_bool(item.get("deleted"))
        for field_name in ("source_pack_version", "source_current_version", "version"):
            value = item.get(field_name)
            if value is None:
                continue
            try:
                item[field_name] = int(value)
            except (TypeError, ValueError):
                item[field_name] = None

        source_available = item.get("source_available")
        if source_available is None:
            source_available = bool(item.get("source_persona_id") and item.get("source_pack_id"))
        item["source_available"] = self._as_bool(source_available)

        source_changed = item.get("source_changed")
        if source_changed is None:
            current_version = item.get("source_current_version")
            stored_version = item.get("source_pack_version")
            source_changed = current_version is not None and stored_version is not None and current_version != stored_version
        item["source_changed"] = self._as_bool(source_changed)

        live_persona_name = item.pop("live_source_persona_name", None)
        live_pack_title = item.pop("live_source_pack_title", None)
        item["source_persona_name"] = live_persona_name or item.get("source_persona_name_snapshot")
        item["source_pack_title"] = live_pack_title or item.get("source_pack_title_snapshot")
        return item

    def _normalize_persona_visual_library_tags(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            raise InputError("persona visual library tags must be a list.")  # noqa: TRY003
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_tag in value:
            tag = str(raw_tag or "").strip().lower()
            if not tag or tag in seen:
                continue
            if len(tag) > 64:
                raise InputError("persona visual library tags must be 64 characters or fewer.")  # noqa: TRY003
            seen.add(tag)
            normalized.append(tag)
            if len(normalized) >= 20:
                break
        return normalized

    def _require_persona_visual_library_item_owner(
        self,
        conn: Any,
        *,
        item_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM persona_visual_library_items WHERE id = ? AND user_id = ? LIMIT 1",
            (item_id, user_id),
        ).fetchone()
        if not row:
            raise ConflictError(  # noqa: TRY003
                "Persona visual library item not found for user.",
                entity="persona_visual_library_items",
                entity_id=item_id,
            )
        item = dict(row)
        if not include_deleted and self._as_bool(item.get("deleted")):
            raise ConflictError(  # noqa: TRY003
                "Persona visual library item is soft-deleted.",
                entity="persona_visual_library_items",
                entity_id=item_id,
            )
        return item

    def _normalize_persona_visual_enum(
        self,
        value: Any,
        *,
        allowed: tuple[str, ...],
        field_name: str,
        default: str | None = None,
    ) -> str:
        normalized = str(value if value is not None else default or "").strip().lower()
        if normalized not in allowed:
            allowed_values = ", ".join(allowed)
            raise InputError(
                f"Invalid persona visual {field_name} '{normalized}'. Allowed: {allowed_values}."
            )  # noqa: TRY003
        return normalized

    def _require_persona_visual_pack_owner(
        self,
        conn: Any,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        row = conn.execute(
            """
            SELECT p.*
              FROM persona_visual_packs p
              JOIN persona_profiles pp
                ON pp.id = p.persona_id
               AND pp.user_id = p.user_id
             WHERE p.id = ?
               AND p.persona_id = ?
               AND p.user_id = ?
             LIMIT 1
            """,
            (pack_id, persona_id, user_id),
        ).fetchone()
        if not row:
            raise ConflictError(  # noqa: TRY003
                "Persona visual pack not found for user.",
                entity="persona_visual_packs",
                entity_id=pack_id,
            )
        item = dict(row)
        if not include_deleted and self._as_bool(item.get("deleted")):
            raise ConflictError(  # noqa: TRY003
                "Persona visual pack is soft-deleted.",
                entity="persona_visual_packs",
                entity_id=pack_id,
            )
        return item

    def _persona_scope_rule_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["include"] = self._as_bool(item.get("include"))
        item["deleted"] = self._as_bool(item.get("deleted"))
        return item

    def _persona_policy_rule_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["allowed"] = self._as_bool(item.get("allowed"))
        item["require_confirmation"] = self._as_bool(item.get("require_confirmation"))
        item["deleted"] = self._as_bool(item.get("deleted"))
        max_calls = item.get("max_calls_per_turn")
        if max_calls is not None:
            try:
                item["max_calls_per_turn"] = int(max_calls)
            except (TypeError, ValueError):
                item["max_calls_per_turn"] = None
        return item

    def _persona_session_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        row_id = item.get("id") or item.get("session_id") or item.get("uuid") or "N/A"
        item["reuse_allowed"] = self._as_bool(item.get("reuse_allowed"))
        item["deleted"] = self._as_bool(item.get("deleted"))
        item["activity_surface"] = self._normalize_persona_session_activity_surface(item.get("activity_surface"))
        raw_preferences = item.get("preferences_json")
        if isinstance(raw_preferences, str):
            try:
                decoded_preferences = json.loads(raw_preferences)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode JSON for field '{}' in persona session row {}. Falling back to empty object. Value preview: {}",
                    "preferences_json",
                    row_id,
                    raw_preferences[:100] + ("..." if len(raw_preferences) > 100 else ""),
                )
                decoded_preferences = {}
            item["preferences"] = decoded_preferences if isinstance(decoded_preferences, dict) else {}
        elif isinstance(raw_preferences, dict):
            item["preferences"] = raw_preferences
        else:
            item["preferences"] = {}
        raw_snapshot = item.get("scope_snapshot_json")
        if isinstance(raw_snapshot, str):
            try:
                item["scope_snapshot"] = json.loads(raw_snapshot)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode JSON for field '{}' in persona session row {}. Falling back to empty object. Value preview: {}",
                    "scope_snapshot_json",
                    row_id,
                    raw_snapshot[:100] + ("..." if len(raw_snapshot) > 100 else ""),
                )
                item["scope_snapshot"] = {}
        elif isinstance(raw_snapshot, dict):
            item["scope_snapshot"] = raw_snapshot
        else:
            item["scope_snapshot"] = {}
        return item

    @staticmethod
    def _normalize_persona_session_activity_surface(value: Any) -> str:
        from tldw_Server_API.app.core.Personalization.companion_activity import (
            normalize_persona_activity_surface,
        )

        return normalize_persona_activity_surface(value)

    def _persona_memory_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["archived"] = self._as_bool(item.get("archived"))
        item["deleted"] = self._as_bool(item.get("deleted"))
        salience = item.get("salience")
        if salience is not None:
            try:
                item["salience"] = float(salience)
            except (TypeError, ValueError):
                item["salience"] = 0.0
        return item

    def _persona_exemplar_row_to_dict(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        item = self._deserialize_row_fields(row, self._PERSONA_EXEMPLAR_JSON_FIELDS)
        if not item:
            return None
        item["enabled"] = self._as_bool(item.get("enabled", True))
        item["deleted"] = self._as_bool(item.get("deleted"))
        try:
            item["priority"] = int(item.get("priority", 0) or 0)
        except (TypeError, ValueError):
            item["priority"] = 0
        item["scenario_tags"] = self._normalize_persona_exemplar_tags(
            item.pop("scenario_tags_json", []),
            "scenario_tags",
        )
        item["capability_tags"] = self._normalize_persona_exemplar_tags(
            item.pop("capability_tags_json", []),
            "capability_tags",
        )
        return item

    def _require_active_persona_profile_owner(self, conn: Any, *, persona_id: str, user_id: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT id, user_id, mode, deleted FROM persona_profiles WHERE id = ? AND user_id = ?",
            (persona_id, user_id),
        ).fetchone()
        if not row:
            raise ConflictError(  # noqa: TRY003
                "Persona profile not found for user.",
                entity="persona_profiles",
                entity_id=persona_id,
            )
        item = dict(row)
        if self._as_bool(item.get("deleted")):
            raise ConflictError(  # noqa: TRY003
                "Persona profile is soft-deleted.",
                entity="persona_profiles",
                entity_id=persona_id,
            )
        return item

    def _normalize_persona_exemplar_tone(self, value: Any) -> str | None:
        tone = self._normalize_nullable_text(value)
        if tone is None:
            return None
        normalized = tone.strip().lower()
        return normalized or None

    def _normalize_exemplar_enum(
        self,
        value: Any,
        *,
        allowed: tuple[str, ...],
        field_name: str,
        default: str,
    ) -> str:
        """Normalize and validate enum-like persona exemplar fields."""
        return exemplar_normalization.normalize_exemplar_enum(
            value,
            allowed=allowed,
            field_name=field_name,
            default=default,
        )

    def _normalize_exemplar_string_list(self, value: Any, field_name: str) -> list[str]:
        """Normalize list-like persona exemplar metadata to a string list."""
        return exemplar_normalization.normalize_exemplar_string_list(value, field_name)

    def _normalize_persona_exemplar_tags(self, value: Any, field_name: str) -> list[str]:
        """Normalize free-form persona exemplar tags to lowercase unique strings."""
        raw_values = self._normalize_exemplar_string_list(value, field_name)
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            text = str(item).strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def create_persona_exemplar(self, exemplar_data: dict[str, Any]) -> str:
        persona_id = str(exemplar_data.get("persona_id") or "").strip()
        user_id = str(exemplar_data.get("user_id") or "").strip()
        if not persona_id:
            raise InputError("persona_id is required for persona exemplar creation.")  # noqa: TRY003
        if not user_id:
            raise InputError("user_id is required for persona exemplar creation.")  # noqa: TRY003

        kind = self._normalize_exemplar_enum(
            exemplar_data.get("kind"),
            allowed=self._ALLOWED_PERSONA_EXEMPLAR_KINDS,
            field_name="kind",
            default="style",
        )
        content = self._normalize_nullable_text(exemplar_data.get("content"))
        if not content:
            raise InputError("content is required for persona exemplar creation.")  # noqa: TRY003

        tone = self._normalize_persona_exemplar_tone(exemplar_data.get("tone"))
        scenario_tags = self._normalize_persona_exemplar_tags(
            exemplar_data.get("scenario_tags"),
            "scenario_tags",
        )
        capability_tags = self._normalize_persona_exemplar_tags(
            exemplar_data.get("capability_tags"),
            "capability_tags",
        )
        try:
            priority = int(exemplar_data.get("priority", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise InputError("priority must be an integer.") from exc  # noqa: TRY003
        enabled = self._as_bool(exemplar_data.get("enabled", True))
        source_type = self._normalize_exemplar_enum(
            exemplar_data.get("source_type"),
            allowed=self._ALLOWED_PERSONA_EXEMPLAR_SOURCE_TYPES,
            field_name="source_type",
            default="manual",
        )
        source_ref = self._normalize_nullable_text(exemplar_data.get("source_ref"))
        notes = self._normalize_nullable_text(exemplar_data.get("notes"))
        exemplar_id = str(exemplar_data.get("id") or self._generate_uuid()).strip()
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted = self._normalize_deleted_input(exemplar_data.get("deleted", False))
        version = self._parse_version_input(exemplar_data.get("version", 1))

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            query = (
                "INSERT INTO persona_exemplars("
                "id, persona_id, user_id, kind, content, tone, scenario_tags_json, capability_tags_json, "
                "priority, enabled, source_type, source_ref, notes, created_at, last_modified, deleted, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            params = (
                exemplar_id,
                persona_id,
                user_id,
                kind,
                content,
                tone,
                self._ensure_json_string(scenario_tags) or "[]",
                self._ensure_json_string(capability_tags) or "[]",
                priority,
                bool_cast(enabled),
                source_type,
                source_ref,
                notes,
                exemplar_data.get("created_at") or now,
                exemplar_data.get("last_modified") or now,
                bool_cast(deleted),
                version,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            conn.execute(prepared_query, prepared_params or ())
        return exemplar_id

    def get_persona_exemplar(
        self,
        *,
        exemplar_id: str,
        persona_id: str,
        user_id: str,
        include_disabled: bool = False,
        include_deleted: bool = False,
        include_deleted_personas: bool = False,
    ) -> dict[str, Any] | None:
        query = """
            SELECT pe.*
              FROM persona_exemplars pe
              JOIN persona_profiles pp
                ON pp.id = pe.persona_id
               AND pp.user_id = pe.user_id
             WHERE pe.id = ?
               AND pe.persona_id = ?
               AND pe.user_id = ?
               AND (? OR pe.enabled = 1)
               AND (? OR pe.deleted = 0)
               AND (? OR pp.deleted = FALSE)
             LIMIT 1
        """
        params = (
            exemplar_id,
            persona_id,
            user_id,
            bool(include_disabled),
            bool(include_deleted),
            bool(include_deleted_personas),
        )
        cursor = self.execute_query(query, params)
        return self._persona_exemplar_row_to_dict(cursor.fetchone())

    def list_persona_exemplars(
        self,
        *,
        user_id: str,
        persona_id: str | None = None,
        include_disabled: bool = False,
        include_deleted: bool = False,
        include_deleted_personas: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT pe.*
              FROM persona_exemplars pe
              JOIN persona_profiles pp
                ON pp.id = pe.persona_id
               AND pp.user_id = pe.user_id
             WHERE pe.user_id = ?
               AND (? IS NULL OR pe.persona_id = ?)
               AND (? OR pe.enabled = 1)
               AND (? OR pe.deleted = 0)
               AND (? OR pp.deleted = FALSE)
             ORDER BY pe.priority DESC, pe.last_modified DESC, pe.id ASC
             LIMIT ? OFFSET ?
        """
        params = (
            user_id,
            persona_id,
            persona_id,
            bool(include_disabled),
            bool(include_deleted),
            bool(include_deleted_personas),
            max(1, int(limit)),
            max(0, int(offset)),
        )
        cursor = self.execute_query(query, params)
        return [self._persona_exemplar_row_to_dict(row) for row in cursor.fetchall() if row]

    def update_persona_exemplar(
        self,
        *,
        exemplar_id: str,
        persona_id: str,
        user_id: str,
        update_data: dict[str, Any],
    ) -> bool:
        if not update_data:
            raise InputError("No exemplar fields provided for update.")  # noqa: TRY003

        allowed_fields = {
            "kind",
            "content",
            "tone",
            "scenario_tags",
            "capability_tags",
            "priority",
            "enabled",
            "source_type",
            "source_ref",
            "notes",
            "deleted",
        }
        normalized_updates: dict[str, Any] = {}
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int

        for key, value in update_data.items():
            if key not in allowed_fields:
                continue
            if key == "kind":
                normalized_updates["kind"] = self._normalize_exemplar_enum(
                    value,
                    allowed=self._ALLOWED_PERSONA_EXEMPLAR_KINDS,
                    field_name="kind",
                    default="style",
                )
            elif key == "content":
                content = self._normalize_nullable_text(value)
                if not content:
                    raise InputError("content cannot be empty.")  # noqa: TRY003
                normalized_updates["content"] = content
            elif key == "tone":
                normalized_updates["tone"] = self._normalize_persona_exemplar_tone(value)
            elif key == "scenario_tags":
                normalized_updates["scenario_tags_json"] = (
                    self._ensure_json_string(self._normalize_persona_exemplar_tags(value, "scenario_tags")) or "[]"
                )
            elif key == "capability_tags":
                normalized_updates["capability_tags_json"] = (
                    self._ensure_json_string(self._normalize_persona_exemplar_tags(value, "capability_tags")) or "[]"
                )
            elif key == "priority":
                try:
                    normalized_updates["priority"] = int(value)
                except (TypeError, ValueError) as exc:
                    raise InputError("priority must be an integer.") from exc  # noqa: TRY003
            elif key == "enabled":
                normalized_updates["enabled"] = bool_cast(self._as_bool(value))
            elif key == "source_type":
                normalized_updates["source_type"] = self._normalize_exemplar_enum(
                    value,
                    allowed=self._ALLOWED_PERSONA_EXEMPLAR_SOURCE_TYPES,
                    field_name="source_type",
                    default="manual",
                )
            elif key == "source_ref":
                normalized_updates["source_ref"] = self._normalize_nullable_text(value)
            elif key == "notes":
                normalized_updates["notes"] = self._normalize_nullable_text(value)
            elif key == "deleted":
                normalized_updates["deleted"] = bool_cast(self._normalize_deleted_input(value))

        if not normalized_updates:
            raise InputError("No valid exemplar fields provided for update.")  # noqa: TRY003

        now = self._get_current_utc_timestamp_iso()
        query = """
            UPDATE persona_exemplars
               SET kind = CASE WHEN ? THEN ? ELSE kind END,
                   content = CASE WHEN ? THEN ? ELSE content END,
                   tone = CASE WHEN ? THEN ? ELSE tone END,
                   scenario_tags_json = CASE WHEN ? THEN ? ELSE scenario_tags_json END,
                   capability_tags_json = CASE WHEN ? THEN ? ELSE capability_tags_json END,
                   priority = CASE WHEN ? THEN ? ELSE priority END,
                   enabled = CASE WHEN ? THEN ? ELSE enabled END,
                   source_type = CASE WHEN ? THEN ? ELSE source_type END,
                   source_ref = CASE WHEN ? THEN ? ELSE source_ref END,
                   notes = CASE WHEN ? THEN ? ELSE notes END,
                   deleted = CASE WHEN ? THEN ? ELSE deleted END,
                   last_modified = ?,
                   version = version + 1
             WHERE id = ? AND persona_id = ? AND user_id = ? AND deleted = 0
        """
        params = (
            "kind" in normalized_updates,
            normalized_updates.get("kind"),
            "content" in normalized_updates,
            normalized_updates.get("content"),
            "tone" in normalized_updates,
            normalized_updates.get("tone"),
            "scenario_tags_json" in normalized_updates,
            normalized_updates.get("scenario_tags_json"),
            "capability_tags_json" in normalized_updates,
            normalized_updates.get("capability_tags_json"),
            "priority" in normalized_updates,
            normalized_updates.get("priority"),
            "enabled" in normalized_updates,
            normalized_updates.get("enabled"),
            "source_type" in normalized_updates,
            normalized_updates.get("source_type"),
            "source_ref" in normalized_updates,
            normalized_updates.get("source_ref"),
            "notes" in normalized_updates,
            normalized_updates.get("notes"),
            "deleted" in normalized_updates,
            normalized_updates.get("deleted"),
            now,
            exemplar_id,
            persona_id,
            user_id,
        )

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def soft_delete_persona_exemplar(
        self,
        *,
        exemplar_id: str,
        persona_id: str,
        user_id: str,
    ) -> bool:
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            query = (
                "UPDATE persona_exemplars "
                "SET deleted = ?, enabled = ?, last_modified = ?, version = version + 1 "
                "WHERE id = ? AND persona_id = ? AND user_id = ? AND deleted = 0"
            )
            params = (
                bool_cast(True),
                bool_cast(False),
                now,
                exemplar_id,
                persona_id,
                user_id,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def create_persona_profile(self, profile_data: dict[str, Any]) -> str:
        user_id = str(profile_data.get("user_id") or "").strip()
        name = str(profile_data.get("name") or "").strip()
        if not user_id:
            raise InputError("user_id is required for persona profile creation.")  # noqa: TRY003
        if not name:
            raise InputError("name is required for persona profile creation.")  # noqa: TRY003

        persona_id = str(profile_data.get("id") or self._generate_uuid())
        mode = self._normalize_persona_mode(profile_data.get("mode"))
        system_prompt = profile_data.get("system_prompt")
        is_active = self._as_bool(profile_data.get("is_active", True))
        use_persona_state_context_default = self._as_bool(
            profile_data.get("use_persona_state_context_default", True)
        )
        voice_defaults_json = self._ensure_json_string(
            profile_data.get("voice_defaults")
            if isinstance(profile_data.get("voice_defaults"), dict)
            else None
        )
        setup_json = self._ensure_json_string(
            profile_data.get("setup") if isinstance(profile_data.get("setup"), dict) else None
        )
        now = self._get_current_utc_timestamp_iso()

        character_card_id = profile_data.get("character_card_id")
        if character_card_id is not None:
            try:
                character_card_id = int(character_card_id)
            except (TypeError, ValueError) as exc:
                raise InputError("character_card_id must be an integer when provided.") from exc  # noqa: TRY003

        origin_character_id = profile_data.get("origin_character_id")
        origin_character_name = profile_data.get("origin_character_name")
        origin_character_snapshot_at = profile_data.get("origin_character_snapshot_at")
        if character_card_id is not None:
            source_character = self.get_character_card_by_id(character_card_id)
            if source_character is None:
                raise InputError(  # noqa: TRY003
                    f"character_card_id '{character_card_id}' must reference an existing active character."
                )
            origin_character_id = source_character.get("id") or character_card_id
            source_character_name = str(source_character.get("name") or "").strip()
            origin_character_name = source_character_name or None
            origin_character_snapshot_at = origin_character_snapshot_at or now

        if origin_character_id is not None:
            try:
                origin_character_id = int(origin_character_id)
            except (TypeError, ValueError) as exc:
                raise InputError("origin_character_id must be an integer when provided.") from exc  # noqa: TRY003
        if origin_character_name is not None:
            origin_character_name = str(origin_character_name).strip() or None
        if origin_character_snapshot_at is not None:
            origin_character_snapshot_at = str(origin_character_snapshot_at)

        deleted_value = self._normalize_deleted_input(profile_data.get("deleted", False))
        version = self._parse_version_input(profile_data.get("version", 1))

        if self.backend_type == BackendType.POSTGRESQL:
            is_active_db = bool(is_active)
            use_persona_state_context_default_db = bool(use_persona_state_context_default)
            deleted_db = bool(deleted_value)
        else:
            is_active_db = int(is_active)
            use_persona_state_context_default_db = int(use_persona_state_context_default)
            deleted_db = int(deleted_value)

        query = (
            "INSERT INTO persona_profiles("
            "id, user_id, name, character_card_id, origin_character_id, origin_character_name, "
            "origin_character_snapshot_at, mode, system_prompt, voice_defaults_json, setup_json, "
            "is_active, use_persona_state_context_default, created_at, last_modified, deleted, version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            persona_id,
            user_id,
            name,
            character_card_id,
            origin_character_id,
            origin_character_name,
            origin_character_snapshot_at,
            mode,
            system_prompt,
            voice_defaults_json,
            setup_json,
            is_active_db,
            use_persona_state_context_default_db,
            profile_data.get("created_at") or now,
            profile_data.get("last_modified") or now,
            deleted_db,
            version,
        )

        try:
            self.execute_query(query, params, commit=True)
            return persona_id  # noqa: TRY300
        except sqlite3.IntegrityError as exc:
            msg = str(exc).lower()
            if "unique constraint failed" in msg and "persona_profiles.user_id, persona_profiles.name" in msg:
                raise ConflictError(  # noqa: TRY003
                    f"Persona profile name '{name}' already exists for user '{user_id}'.",
                    entity="persona_profiles",
                    entity_id=persona_id,
                ) from exc
            if "unique constraint failed" in msg and "persona_profiles.id" in msg:
                raise ConflictError(  # noqa: TRY003
                    f"Persona profile '{persona_id}' already exists.",
                    entity="persona_profiles",
                    entity_id=persona_id,
                ) from exc
            raise
        except BackendDatabaseError as exc:
            if self._is_unique_violation(exc):
                raise ConflictError(  # noqa: TRY003
                    f"Persona profile name '{name}' already exists for user '{user_id}'.",
                    entity="persona_profiles",
                    entity_id=persona_id,
                ) from exc
            raise CharactersRAGDBError(f"Failed creating persona profile: {exc}") from exc  # noqa: TRY003

    def get_persona_profile(
        self,
        persona_id: str,
        *,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM persona_profiles WHERE id = ? AND user_id = ?"
        params: list[Any] = [persona_id, user_id]
        if not include_deleted:
            query += " AND deleted = 0"
        cursor = self.execute_query(query, tuple(params))
        return self._persona_profile_row_to_dict(cursor.fetchone())

    def list_persona_profiles(
        self,
        *,
        user_id: str,
        include_deleted: bool = False,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if not include_deleted:
            clauses.append("deleted = 0")
        if active_only:
            clauses.append("is_active = 1")
        where_sql = " AND ".join(clauses)
        query = (
            "SELECT * FROM persona_profiles "  # nosec B608
            f"WHERE {where_sql} "
            "ORDER BY last_modified DESC, name ASC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, int(limit)), max(0, int(offset))])
        cursor = self.execute_query(query, tuple(params))
        return [self._persona_profile_row_to_dict(row) for row in cursor.fetchall() if row]

    def update_persona_profile(
        self,
        *,
        persona_id: str,
        user_id: str,
        update_data: dict[str, Any],
        expected_version: int | None = None,
    ) -> bool:
        if not update_data:
            raise InputError("No profile fields provided for update.")  # noqa: TRY003

        allowed_fields = {
            "name",
            "character_card_id",
            "mode",
            "system_prompt",
            "voice_defaults",
            "setup",
            "is_active",
            "use_persona_state_context_default",
            "deleted",
        }
        set_parts: list[str] = []
        params: list[Any] = []

        for key, value in update_data.items():
            if key not in allowed_fields:
                continue
            if key == "mode":
                params.append(self._normalize_persona_mode(value))
                set_parts.append("mode = ?")
            elif key in {"is_active", "use_persona_state_context_default", "deleted"}:
                cast_value = bool(self._as_bool(value)) if self.backend_type == BackendType.POSTGRESQL else int(
                    self._as_bool(value)
                )
                params.append(cast_value)
                set_parts.append(f"{key} = ?")
            elif key == "character_card_id":
                if value is None:
                    params.append(None)
                else:
                    try:
                        params.append(int(value))
                    except (TypeError, ValueError) as exc:
                        raise InputError("character_card_id must be an integer when provided.") from exc  # noqa: TRY003
                set_parts.append("character_card_id = ?")
            elif key == "voice_defaults":
                params.append(self._ensure_json_string(value if isinstance(value, dict) else {}))
                set_parts.append("voice_defaults_json = ?")
            elif key == "setup":
                params.append(self._ensure_json_string(value if isinstance(value, dict) else {}))
                set_parts.append("setup_json = ?")
            else:
                params.append(value)
                set_parts.append(f"{key} = ?")

        if not set_parts:
            raise InputError("No valid profile fields provided for update.")  # noqa: TRY003

        now = self._get_current_utc_timestamp_iso()
        set_parts.extend(["last_modified = ?", "version = version + 1"])
        params.append(now)

        where_sql = "id = ? AND user_id = ? AND deleted = 0"
        params.extend([persona_id, user_id])
        if expected_version is not None:
            where_sql += " AND version = ?"
            params.append(int(expected_version))

        query = f"UPDATE persona_profiles SET {', '.join(set_parts)} WHERE {where_sql}"  # nosec B608
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT version, deleted FROM persona_profiles WHERE id = ? AND user_id = ?",
                (persona_id, user_id),
            ).fetchone()
            if not existing:
                return False
            if self._as_bool(existing["deleted"]):
                return False
            if expected_version is not None and int(existing["version"]) != int(expected_version):
                raise ConflictError(  # noqa: TRY003
                    f"Persona profile version mismatch (db has {existing['version']}, expected {expected_version}).",
                    entity="persona_profiles",
                    entity_id=persona_id,
                )
            prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def soft_delete_persona_profile(
        self,
        *,
        persona_id: str,
        user_id: str,
        expected_version: int | None = None,
    ) -> bool:
        return self.update_persona_profile(
            persona_id=persona_id,
            user_id=user_id,
            update_data={"deleted": True, "is_active": False},
            expected_version=expected_version,
        )

    def restore_persona_profile(
        self,
        *,
        persona_id: str,
        user_id: str,
        expected_version: int,
    ) -> bool:
        now = self._get_current_utc_timestamp_iso()
        expected_version_value = self._parse_version_input(expected_version)
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        deleted_true = True if self.backend_type == BackendType.POSTGRESQL else 1
        is_active_true = True if self.backend_type == BackendType.POSTGRESQL else 1

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT version, deleted FROM persona_profiles WHERE id = ? AND user_id = ?",
                (persona_id, user_id),
            ).fetchone()
            if not row:
                return False
            if not self._as_bool(row["deleted"]):
                return True

            current_db_version = int(row["version"])
            if current_db_version != expected_version_value:
                raise ConflictError(  # noqa: TRY003
                    (
                        f"Restore for persona profile {persona_id} failed: "
                        f"version mismatch (db has {current_db_version}, expected {expected_version_value})."
                    ),
                    entity="persona_profiles",
                    entity_id=persona_id,
                )

            query = (
                "UPDATE persona_profiles "
                "SET deleted = ?, is_active = ?, last_modified = ?, version = version + 1 "
                "WHERE id = ? AND user_id = ? AND version = ? AND deleted = ?"
            )
            params = (
                deleted_false,
                is_active_true,
                now,
                persona_id,
                user_id,
                expected_version_value,
                deleted_true,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            cursor = conn.execute(prepared_query, prepared_params or ())
            if cursor.rowcount > 0:
                return True

            final_state = conn.execute(
                "SELECT version, deleted FROM persona_profiles WHERE id = ? AND user_id = ?",
                (persona_id, user_id),
            ).fetchone()
            return bool(final_state and not self._as_bool(final_state["deleted"]))

    def get_persona_buddy(
        self,
        *,
        persona_id: str,
        user_id: str,
        include_deleted_personas: bool = False,
    ) -> dict[str, Any] | None:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT pb.*
              FROM persona_buddies pb
              JOIN persona_profiles pp
                ON pp.id = pb.persona_id
               AND pp.user_id = pb.user_id
             WHERE pb.persona_id = ?
               AND pb.user_id = ?
               AND (? OR pp.deleted = ?)
             LIMIT 1
        """
        params = (
            persona_id,
            user_id,
            bool(include_deleted_personas),
            deleted_false,
        )
        cursor = self.execute_query(query, params)
        return self._persona_buddy_row_to_dict(cursor.fetchone())

    def list_persona_buddies(
        self,
        *,
        user_id: str,
        persona_ids: list[str],
        include_deleted_personas: bool = False,
    ) -> dict[str, dict[str, Any] | None]:
        normalized_persona_ids = list(
            dict.fromkeys(
                str(persona_id or "").strip()
                for persona_id in persona_ids
                if str(persona_id or "").strip()
            )
        )
        if not normalized_persona_ids:
            return {}

        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        placeholders = ", ".join("?" for _ in normalized_persona_ids)
        query = (
            "SELECT pb.* "
            "FROM persona_buddies pb "
            "JOIN persona_profiles pp "
            "  ON pp.id = pb.persona_id "
            " AND pp.user_id = pb.user_id "
            f"WHERE pb.user_id = ? AND pb.persona_id IN ({placeholders}) "  # nosec B608
            "AND (? OR pp.deleted = ?)"
        )
        params: list[Any] = [
            user_id,
            *normalized_persona_ids,
            bool(include_deleted_personas),
            deleted_false,
        ]
        cursor = self.execute_query(query, tuple(params))
        buddies: dict[str, dict[str, Any] | None] = {
            persona_id: None for persona_id in normalized_persona_ids
        }
        for row in cursor.fetchall():
            buddy = self._persona_buddy_row_to_dict(row)
            if buddy is None:
                continue
            persona_id = str(buddy.get("persona_id") or "").strip()
            if persona_id:
                buddies[persona_id] = buddy
        return buddies

    def upsert_persona_buddy(
        self,
        *,
        persona_id: str,
        user_id: str,
        derivation_version: int,
        source_fingerprint: str,
        derived_core: dict[str, Any] | None,
        overlay_preferences: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not source_fingerprint:
            raise InputError("source_fingerprint is required for persona buddy upsert.")  # noqa: TRY003
        try:
            derivation_version_value = int(derivation_version)
        except (TypeError, ValueError) as exc:
            raise InputError("derivation_version must be an integer >= 1.") from exc  # noqa: TRY003
        if derivation_version_value < 1:
            raise InputError("derivation_version must be an integer >= 1.")  # noqa: TRY003

        derived_core_json = self._ensure_json_string(derived_core if isinstance(derived_core, dict) else {}) or "{}"
        overlay_preferences_json = (
            self._ensure_json_string(overlay_preferences if isinstance(overlay_preferences, dict) else {}) or "{}"
        )
        now = self._get_current_utc_timestamp_iso()
        update_query = (
            "UPDATE persona_buddies "
            "SET user_id = ?, derivation_version = ?, source_fingerprint = ?, derived_core_json = ?, "
            "overlay_preferences_json = ?, last_modified = ?, version = version + 1 "
            "WHERE persona_id = ? AND ("
            "user_id <> ? OR derivation_version <> ? OR source_fingerprint <> ? OR "
            "derived_core_json <> ? OR overlay_preferences_json <> ?"
            ")"
        )
        update_params = (
            user_id,
            derivation_version_value,
            source_fingerprint,
            derived_core_json,
            overlay_preferences_json,
            now,
            persona_id,
            user_id,
            derivation_version_value,
            source_fingerprint,
            derived_core_json,
            overlay_preferences_json,
        )
        insert_query = (
            "INSERT INTO persona_buddies("
            "persona_id, user_id, derivation_version, source_fingerprint, derived_core_json, "
            "overlay_preferences_json, created_at, last_modified, version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        insert_params = (
            persona_id,
            user_id,
            derivation_version_value,
            source_fingerprint,
            derived_core_json,
            overlay_preferences_json,
            now,
            now,
            1,
        )

        def _load_persisted_item(connection: Any) -> dict[str, Any]:
            row = connection.execute(
                "SELECT * FROM persona_buddies WHERE persona_id = ? LIMIT 1",
                (persona_id,),
            ).fetchone()
            item = self._persona_buddy_row_to_dict(row)
            if not item:
                raise CharactersRAGDBError("Failed to load persona buddy after upsert.")  # noqa: TRY003
            return item

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            prepared_update, prepared_update_params = self._prepare_backend_statement(update_query, update_params)
            update_cursor = conn.execute(prepared_update, prepared_update_params or ())

            if update_cursor.rowcount == 0:
                existing_row = conn.execute(
                    "SELECT 1 FROM persona_buddies WHERE persona_id = ? LIMIT 1",
                    (persona_id,),
                ).fetchone()
                if existing_row:
                    return _load_persisted_item(conn)

                prepared_insert, prepared_insert_params = self._prepare_backend_statement(insert_query, insert_params)
                try:
                    conn.execute(prepared_insert, prepared_insert_params or ())
                except sqlite3.IntegrityError as exc:
                    msg = str(exc).lower()
                    if "unique constraint failed" not in msg:
                        raise
                    update_cursor = conn.execute(prepared_update, prepared_update_params or ())
                    if update_cursor.rowcount == 0:
                        return _load_persisted_item(conn)
                except BackendDatabaseError as exc:
                    if not self._is_unique_violation(exc):
                        raise
                    update_cursor = conn.execute(prepared_update, prepared_update_params or ())
                    if update_cursor.rowcount == 0:
                        return _load_persisted_item(conn)

            return _load_persisted_item(conn)

    def create_persona_visual_pack(
        self,
        *,
        persona_id: str,
        user_id: str,
        title: str,
        manifest: dict[str, Any] | None = None,
        renderer_type: str = "sprite_frames",
        status: str = "draft",
        parent_pack_id: str | None = None,
        parent_persona_id: str | None = None,
        revision_number: int = 1,
        provenance: str = "uploaded",
        pack_id: str | None = None,
    ) -> dict[str, Any]:
        persona_id = str(persona_id or "").strip()
        user_id = str(user_id or "").strip()
        title_value = str(title or "").strip()
        if not persona_id:
            raise InputError("persona_id is required for persona visual pack creation.")  # noqa: TRY003
        if not user_id:
            raise InputError("user_id is required for persona visual pack creation.")  # noqa: TRY003
        if not title_value:
            raise InputError("title is required for persona visual pack creation.")  # noqa: TRY003

        manifest_value = manifest if isinstance(manifest, dict) else {}
        manifest_version = manifest_value.get("manifest_version", 1)
        try:
            manifest_version_value = int(manifest_version)
        except (TypeError, ValueError) as exc:
            raise InputError("manifest_version must be an integer.") from exc  # noqa: TRY003
        try:
            revision_number_value = int(revision_number)
        except (TypeError, ValueError) as exc:
            raise InputError("revision_number must be an integer.") from exc  # noqa: TRY003
        if manifest_version_value < 1 or revision_number_value < 1:
            raise InputError("manifest_version and revision_number must be >= 1.")  # noqa: TRY003

        renderer_type_value = self._normalize_persona_visual_enum(
            renderer_type,
            allowed=self._ALLOWED_PERSONA_VISUAL_RENDERER_TYPES,
            field_name="renderer_type",
            default="sprite_frames",
        )
        status_value = self._normalize_persona_visual_enum(
            status,
            allowed=self._ALLOWED_PERSONA_VISUAL_PACK_STATUSES,
            field_name="status",
            default="draft",
        )
        provenance_value = self._normalize_persona_visual_enum(
            provenance,
            allowed=self._ALLOWED_PERSONA_VISUAL_PROVENANCE_TYPES,
            field_name="provenance",
            default="uploaded",
        )
        pack_id_value = str(pack_id or self._generate_uuid()).strip()
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        active_at = now if status_value == "active" else None

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            if parent_pack_id:
                self._require_persona_visual_pack_owner(
                    conn,
                    pack_id=str(parent_pack_id),
                    persona_id=str(parent_persona_id or persona_id),
                    user_id=user_id,
                )
            if status_value == "active":
                archive_query = (
                    "UPDATE persona_visual_packs "
                    "SET status = 'archived', active_at = NULL, last_modified = ?, version = version + 1 "
                    "WHERE user_id = ? AND persona_id = ? AND status = 'active' AND deleted = ?"
                )
                archive_params = (
                    now,
                    user_id,
                    persona_id,
                    bool_cast(False),
                )
                prepared_archive, prepared_archive_params = self._prepare_backend_statement(
                    archive_query,
                    archive_params,
                )
                conn.execute(prepared_archive, prepared_archive_params or ())

            insert_query = (
                "INSERT INTO persona_visual_packs("
                "id, persona_id, user_id, title, renderer_type, status, manifest_version, manifest_json, "
                "parent_pack_id, revision_number, provenance, active_at, created_at, last_modified, deleted, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            insert_params = (
                pack_id_value,
                persona_id,
                user_id,
                title_value,
                renderer_type_value,
                status_value,
                manifest_version_value,
                self._ensure_json_string(manifest_value) or "{}",
                str(parent_pack_id) if parent_pack_id else None,
                revision_number_value,
                provenance_value,
                active_at,
                now,
                now,
                bool_cast(False),
                1,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(insert_query, insert_params)
            conn.execute(prepared_query, prepared_params or ())

        pack = self.get_persona_visual_pack(
            pack_id=pack_id_value,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not pack:
            raise CharactersRAGDBError("Failed to load persona visual pack after creation.")  # noqa: TRY003
        return pack

    def get_persona_visual_pack(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
        include_deleted_personas: bool = False,
    ) -> dict[str, Any] | None:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT p.*
              FROM persona_visual_packs p
              JOIN persona_profiles pp
                ON pp.id = p.persona_id
               AND pp.user_id = p.user_id
             WHERE p.id = ?
               AND p.persona_id = ?
               AND p.user_id = ?
               AND (? OR p.deleted = ?)
               AND (? OR pp.deleted = ?)
             LIMIT 1
        """
        params = (
            pack_id,
            persona_id,
            user_id,
            bool(include_deleted),
            deleted_false,
            bool(include_deleted_personas),
            deleted_false,
        )
        cursor = self.execute_query(query, params)
        return self._persona_visual_pack_row_to_dict(cursor.fetchone())

    def list_persona_visual_packs(
        self,
        *,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
        include_deleted_personas: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT p.*
              FROM persona_visual_packs p
              JOIN persona_profiles pp
                ON pp.id = p.persona_id
               AND pp.user_id = p.user_id
             WHERE p.persona_id = ?
               AND p.user_id = ?
               AND (? OR p.deleted = ?)
               AND (? OR pp.deleted = ?)
             ORDER BY CASE WHEN p.status = 'active' THEN 0 ELSE 1 END,
                      p.last_modified DESC,
                      p.id ASC
             LIMIT ? OFFSET ?
        """
        params = (
            persona_id,
            user_id,
            bool(include_deleted),
            deleted_false,
            bool(include_deleted_personas),
            deleted_false,
            max(1, int(limit)),
            max(0, int(offset)),
        )
        cursor = self.execute_query(query, params)
        return [
            item
            for row in cursor.fetchall()
            if (item := self._persona_visual_pack_row_to_dict(row)) is not None
        ]

    def get_active_persona_visual_pack(
        self,
        *,
        persona_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT p.*
              FROM persona_visual_packs p
              JOIN persona_profiles pp
                ON pp.id = p.persona_id
               AND pp.user_id = p.user_id
             WHERE p.persona_id = ?
               AND p.user_id = ?
               AND p.status = 'active'
               AND p.deleted = ?
               AND pp.deleted = ?
             ORDER BY p.active_at DESC, p.last_modified DESC
             LIMIT 1
        """
        cursor = self.execute_query(
            query,
            (
                persona_id,
                user_id,
                deleted_false,
                deleted_false,
            ),
        )
        pack = self._persona_visual_pack_row_to_dict(cursor.fetchone())
        if not pack:
            return None
        assets = self.list_persona_visual_assets(
            pack_id=str(pack["id"]),
            persona_id=persona_id,
            user_id=user_id,
        )
        pack["assets"] = assets
        pack["assets_by_id"] = {str(asset["id"]): asset for asset in assets}
        return pack

    def activate_persona_visual_pack(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
    ) -> dict[str, Any]:
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            archive_query = (
                "UPDATE persona_visual_packs "
                "SET status = 'archived', active_at = NULL, last_modified = ?, version = version + 1 "
                "WHERE user_id = ? AND persona_id = ? AND status = 'active' AND deleted = ? AND id <> ?"
            )
            archive_params = (
                now,
                user_id,
                persona_id,
                bool_cast(False),
                pack_id,
            )
            prepared_archive, prepared_archive_params = self._prepare_backend_statement(
                archive_query,
                archive_params,
            )
            conn.execute(prepared_archive, prepared_archive_params or ())

            activate_query = (
                "UPDATE persona_visual_packs "
                "SET status = 'active', active_at = ?, last_modified = ?, version = version + 1 "
                "WHERE id = ? AND user_id = ? AND persona_id = ? AND deleted = ?"
            )
            activate_params = (
                now,
                now,
                pack_id,
                user_id,
                persona_id,
                bool_cast(False),
            )
            prepared_activate, prepared_activate_params = self._prepare_backend_statement(
                activate_query,
                activate_params,
            )
            cursor = conn.execute(prepared_activate, prepared_activate_params or ())
            if cursor.rowcount == 0:
                raise ConflictError(  # noqa: TRY003
                    "Persona visual pack could not be activated.",
                    entity="persona_visual_packs",
                    entity_id=pack_id,
                )

        active = self.get_active_persona_visual_pack(persona_id=persona_id, user_id=user_id)
        if not active or active["id"] != pack_id:
            raise CharactersRAGDBError("Failed to load active persona visual pack after activation.")  # noqa: TRY003
        return active

    def deactivate_persona_visual_pack(
        self,
        *,
        persona_id: str,
        user_id: str,
    ) -> bool:
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            query = (
                "UPDATE persona_visual_packs "
                "SET status = 'archived', active_at = NULL, last_modified = ?, version = version + 1 "
                "WHERE user_id = ? AND persona_id = ? AND status = 'active' AND deleted = ?"
            )
            params = (
                now,
                user_id,
                persona_id,
                bool_cast(False),
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def update_persona_visual_pack_manifest(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        manifest: dict[str, Any],
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        manifest_value = manifest if isinstance(manifest, dict) else {}
        now = self._get_current_utc_timestamp_iso()
        params: list[Any] = [
            self._ensure_json_string(manifest_value) or "{}",
            int(manifest_value.get("manifest_version", 1) or 1),
            now,
            pack_id,
            user_id,
            persona_id,
            False if self.backend_type == BackendType.POSTGRESQL else 0,
        ]
        where_sql = "id = ? AND user_id = ? AND persona_id = ? AND deleted = ?"
        if expected_version is not None:
            where_sql += " AND version = ?"
            params.append(int(expected_version))
        query = (
            "UPDATE persona_visual_packs "
            "SET manifest_json = ?, manifest_version = ?, last_modified = ?, version = version + 1 "
            f"WHERE {where_sql}"  # nosec B608
        )
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
            cursor = conn.execute(prepared_query, prepared_params or ())
            if cursor.rowcount == 0 and expected_version is not None:
                raise ConflictError(  # noqa: TRY003
                    "Persona visual pack version mismatch.",
                    entity="persona_visual_packs",
                    entity_id=pack_id,
                )
        return self.get_persona_visual_pack(pack_id=pack_id, persona_id=persona_id, user_id=user_id)

    def update_persona_visual_pack_status(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        status: str,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        status_value = self._normalize_persona_visual_enum(
            status,
            allowed=self._ALLOWED_PERSONA_VISUAL_PACK_STATUSES,
            field_name="status",
        )
        if status_value == "active":
            raise InputError("Use activate_persona_visual_pack for active status transitions.")  # noqa: TRY003
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        params: list[Any] = [
            status_value,
            None,
            now,
            pack_id,
            user_id,
            persona_id,
            bool_cast(False),
        ]
        where_sql = "id = ? AND user_id = ? AND persona_id = ? AND deleted = ?"
        if expected_version is not None:
            where_sql += " AND version = ?"
            params.append(int(expected_version))
        query = (
            "UPDATE persona_visual_packs "
            "SET status = ?, active_at = ?, last_modified = ?, version = version + 1 "
            f"WHERE {where_sql}"  # nosec B608
        )
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
            cursor = conn.execute(prepared_query, prepared_params or ())
            if cursor.rowcount == 0 and expected_version is not None:
                raise ConflictError(  # noqa: TRY003
                    "Persona visual pack version mismatch.",
                    entity="persona_visual_packs",
                    entity_id=pack_id,
                )
        return self.get_persona_visual_pack(pack_id=pack_id, persona_id=persona_id, user_id=user_id)

    def soft_delete_persona_visual_pack_with_assets(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
    ) -> bool:
        """Soft-delete a persona visual pack and all of its asset rows."""
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted_false = bool_cast(False)
        deleted_true = bool_cast(True)
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            asset_query = (
                "UPDATE persona_visual_assets "
                "SET deleted = ?, last_modified = ?, version = version + 1 "
                "WHERE pack_id = ? AND user_id = ? AND persona_id = ? AND deleted = ?"
            )
            asset_params = (
                deleted_true,
                now,
                pack_id,
                user_id,
                persona_id,
                deleted_false,
            )
            prepared_asset_query, prepared_asset_params = self._prepare_backend_statement(
                asset_query,
                asset_params,
            )
            conn.execute(prepared_asset_query, prepared_asset_params or ())

            pack_query = (
                "UPDATE persona_visual_packs "
                "SET deleted = ?, active_at = NULL, last_modified = ?, version = version + 1 "
                "WHERE id = ? AND user_id = ? AND persona_id = ? AND deleted = ?"
            )
            pack_params = (
                deleted_true,
                now,
                pack_id,
                user_id,
                persona_id,
                deleted_false,
            )
            prepared_pack_query, prepared_pack_params = self._prepare_backend_statement(
                pack_query,
                pack_params,
            )
            cursor = conn.execute(prepared_pack_query, prepared_pack_params or ())
            return cursor.rowcount > 0

    def _load_persona_visual_library_source(
        self,
        conn: Any,
        *,
        source_persona_id: str,
        source_pack_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        row = conn.execute(
            """
            SELECT p.id AS source_pack_id,
                   p.persona_id AS source_persona_id,
                   p.user_id AS user_id,
                   p.title AS source_pack_title,
                   p.version AS source_pack_version,
                   pp.name AS source_persona_name
              FROM persona_visual_packs p
              JOIN persona_profiles pp
                ON pp.id = p.persona_id
               AND pp.user_id = p.user_id
             WHERE p.id = ?
               AND p.persona_id = ?
               AND p.user_id = ?
               AND p.deleted = ?
               AND pp.deleted = ?
             LIMIT 1
            """,
            (source_pack_id, source_persona_id, user_id, deleted_false, deleted_false),
        ).fetchone()
        if not row:
            raise ConflictError(  # noqa: TRY003
                "Persona visual pack not found for library save.",
                entity="persona_visual_packs",
                entity_id=source_pack_id,
            )
        return dict(row)

    def upsert_persona_visual_library_item(
        self,
        *,
        user_id: str,
        source_persona_id: str,
        source_pack_id: str,
        title: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        item_id: str | None = None,
    ) -> dict[str, Any]:
        user_id = str(user_id or "").strip()
        source_persona_id = str(source_persona_id or "").strip()
        source_pack_id = str(source_pack_id or "").strip()
        if not user_id:
            raise InputError("user_id is required for persona visual library items.")  # noqa: TRY003
        if not source_persona_id:
            raise InputError("source_persona_id is required for persona visual library items.")  # noqa: TRY003
        if not source_pack_id:
            raise InputError("source_pack_id is required for persona visual library items.")  # noqa: TRY003

        item_id_value = str(item_id or self._generate_uuid()).strip()
        tags_value = self._normalize_persona_visual_library_tags(tags)
        tags_json = self._ensure_json_string(tags_value) or "[]"
        notes_value = self._normalize_nullable_text(notes)
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted_false = bool_cast(False)

        with self.transaction() as conn:
            source = self._load_persona_visual_library_source(
                conn,
                source_persona_id=source_persona_id,
                source_pack_id=source_pack_id,
                user_id=user_id,
            )
            title_value = str(title or source["source_pack_title"] or "").strip()
            if not title_value:
                raise InputError("title is required for persona visual library items.")  # noqa: TRY003

            existing_row = conn.execute(
                """
                SELECT id
                  FROM persona_visual_library_items
                 WHERE user_id = ?
                   AND source_persona_id = ?
                   AND source_pack_id = ?
                   AND deleted = ?
                 LIMIT 1
                """,
                (user_id, source_persona_id, source_pack_id, deleted_false),
            ).fetchone()
            if existing_row:
                item_id_value = str(dict(existing_row)["id"])
                update_query = (
                    "UPDATE persona_visual_library_items "
                    "SET title = ?, notes = ?, tags_json = ?, source_persona_name_snapshot = ?, "
                    "source_pack_title_snapshot = ?, source_pack_version = ?, "
                    "last_modified = ?, version = version + 1 "
                    "WHERE id = ? AND user_id = ? AND deleted = ?"
                )
                update_params = (
                    title_value,
                    notes_value,
                    tags_json,
                    source["source_persona_name"],
                    source["source_pack_title"],
                    int(source["source_pack_version"]),
                    now,
                    item_id_value,
                    user_id,
                    deleted_false,
                )
                prepared_update, prepared_update_params = self._prepare_backend_statement(update_query, update_params)
                conn.execute(prepared_update, prepared_update_params or ())
            else:
                insert_query = (
                    "INSERT INTO persona_visual_library_items("
                    "id, user_id, source_persona_id, source_pack_id, title, notes, tags_json, "
                    "source_persona_name_snapshot, source_pack_title_snapshot, source_pack_version, "
                    "created_at, last_modified, deleted, version"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                insert_params = (
                    item_id_value,
                    user_id,
                    source_persona_id,
                    source_pack_id,
                    title_value,
                    notes_value,
                    tags_json,
                    source["source_persona_name"],
                    source["source_pack_title"],
                    int(source["source_pack_version"]),
                    now,
                    now,
                    deleted_false,
                    1,
                )
                prepared_insert, prepared_insert_params = self._prepare_backend_statement(insert_query, insert_params)
                conn.execute(prepared_insert, prepared_insert_params or ())

        item = self.get_persona_visual_library_item(item_id=item_id_value, user_id=user_id)
        if not item:
            raise CharactersRAGDBError("Failed to load persona visual library item after upsert.")  # noqa: TRY003
        return item

    def list_persona_visual_library_items(
        self,
        *,
        user_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        user_id = str(user_id or "").strip()
        if not user_id:
            raise InputError("user_id is required for persona visual library item listing.")  # noqa: TRY003
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted_false = bool_cast(False)
        query = """
            SELECT l.*,
                   pp.name AS live_source_persona_name,
                   p.title AS live_source_pack_title,
                   p.version AS source_current_version,
                   CASE WHEN pp.id IS NOT NULL AND p.id IS NOT NULL THEN 1 ELSE 0 END AS source_available,
                   CASE
                     WHEN p.version IS NOT NULL
                      AND l.source_pack_version IS NOT NULL
                      AND p.version <> l.source_pack_version
                     THEN 1 ELSE 0
                   END AS source_changed
              FROM persona_visual_library_items l
              LEFT JOIN persona_profiles pp
                ON pp.id = l.source_persona_id
               AND pp.user_id = l.user_id
               AND pp.deleted = ?
              LEFT JOIN persona_visual_packs p
                ON p.id = l.source_pack_id
               AND p.persona_id = l.source_persona_id
               AND p.user_id = l.user_id
               AND p.deleted = ?
             WHERE l.user_id = ?
               AND (? OR l.deleted = ?)
             ORDER BY l.last_modified DESC, l.id ASC
             LIMIT ? OFFSET ?
        """
        params = (
            deleted_false,
            deleted_false,
            user_id,
            bool(include_deleted),
            deleted_false,
            max(1, int(limit)),
            max(0, int(offset)),
        )
        cursor = self.execute_query(query, params)
        return [
            item
            for row in cursor.fetchall()
            if (item := self._persona_visual_library_item_row_to_dict(row)) is not None
        ]

    def get_persona_visual_library_item(
        self,
        *,
        item_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        item_id = str(item_id or "").strip()
        user_id = str(user_id or "").strip()
        if not item_id or not user_id:
            return None
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted_false = bool_cast(False)
        query = """
            SELECT l.*,
                   pp.name AS live_source_persona_name,
                   p.title AS live_source_pack_title,
                   p.version AS source_current_version,
                   CASE WHEN pp.id IS NOT NULL AND p.id IS NOT NULL THEN 1 ELSE 0 END AS source_available,
                   CASE
                     WHEN p.version IS NOT NULL
                      AND l.source_pack_version IS NOT NULL
                      AND p.version <> l.source_pack_version
                     THEN 1 ELSE 0
                   END AS source_changed
              FROM persona_visual_library_items l
              LEFT JOIN persona_profiles pp
                ON pp.id = l.source_persona_id
               AND pp.user_id = l.user_id
               AND pp.deleted = ?
              LEFT JOIN persona_visual_packs p
                ON p.id = l.source_pack_id
               AND p.persona_id = l.source_persona_id
               AND p.user_id = l.user_id
               AND p.deleted = ?
             WHERE l.id = ?
               AND l.user_id = ?
               AND (? OR l.deleted = ?)
             LIMIT 1
        """
        cursor = self.execute_query(
            query,
            (
                deleted_false,
                deleted_false,
                item_id,
                user_id,
                bool(include_deleted),
                deleted_false,
            ),
        )
        return self._persona_visual_library_item_row_to_dict(cursor.fetchone())

    def update_persona_visual_library_item(
        self,
        *,
        item_id: str,
        user_id: str,
        title: str | object = _UNSET,
        notes: str | None | object = _UNSET,
        tags: list[str] | None | object = _UNSET,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        item_id = str(item_id or "").strip()
        user_id = str(user_id or "").strip()
        if not item_id or not user_id:
            return None

        updates: list[str] = []
        params: list[Any] = []
        if title is not _UNSET:
            title_value = str(title or "").strip()
            if not title_value:
                raise InputError("title cannot be empty for persona visual library items.")  # noqa: TRY003
            updates.append("title = ?")
            params.append(title_value)
        if notes is not _UNSET:
            updates.append("notes = ?")
            params.append(self._normalize_nullable_text(notes))
        if tags is not _UNSET:
            updates.append("tags_json = ?")
            params.append(self._ensure_json_string(self._normalize_persona_visual_library_tags(tags)) or "[]")
        if not updates:
            return self.get_persona_visual_library_item(item_id=item_id, user_id=user_id)

        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted_false = bool_cast(False)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM persona_visual_library_items WHERE id = ? AND user_id = ? AND deleted = ? LIMIT 1",
                (item_id, user_id, deleted_false),
            ).fetchone()
            if not row:
                return None

            updates.extend(["last_modified = ?", "version = version + 1"])
            params.extend([now, item_id, user_id, deleted_false])
            where_sql = "id = ? AND user_id = ? AND deleted = ?"
            if expected_version is not None:
                where_sql += " AND version = ?"
                params.append(int(expected_version))
            query = (
                "UPDATE persona_visual_library_items "
                f"SET {', '.join(updates)} "  # nosec B608
                f"WHERE {where_sql}"  # nosec B608
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
            cursor = conn.execute(prepared_query, prepared_params or ())
            if cursor.rowcount == 0 and expected_version is not None:
                raise ConflictError(  # noqa: TRY003
                    "Persona visual library item version mismatch.",
                    entity="persona_visual_library_items",
                    entity_id=item_id,
                )

        return self.get_persona_visual_library_item(item_id=item_id, user_id=user_id)

    def soft_delete_persona_visual_library_item(
        self,
        *,
        item_id: str,
        user_id: str,
    ) -> bool:
        item_id = str(item_id or "").strip()
        user_id = str(user_id or "").strip()
        if not item_id or not user_id:
            return False
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        deleted_false = bool_cast(False)
        deleted_true = bool_cast(True)
        query = (
            "UPDATE persona_visual_library_items "
            "SET deleted = ?, last_modified = ?, version = version + 1 "
            "WHERE id = ? AND user_id = ? AND deleted = ?"
        )
        params = (deleted_true, now, item_id, user_id, deleted_false)
        with self.transaction() as conn:
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def create_persona_visual_asset(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        asset_role: str,
        storage_key: str,
        original_filename: str | None,
        mime_type: str,
        byte_size: int,
        checksum_sha256: str,
        width: int | None = None,
        height: int | None = None,
        duration_ms: int | None = None,
        provenance: str = "uploaded",
        asset_id: str | None = None,
    ) -> dict[str, Any]:
        role_value = self._normalize_persona_visual_enum(
            asset_role,
            allowed=self._ALLOWED_PERSONA_VISUAL_ASSET_ROLES,
            field_name="asset_role",
            default="frame",
        )
        provenance_value = self._normalize_persona_visual_enum(
            provenance,
            allowed=self._ALLOWED_PERSONA_VISUAL_PROVENANCE_TYPES,
            field_name="provenance",
            default="uploaded",
        )
        storage_key_value = str(storage_key or "").strip()
        mime_type_value = str(mime_type or "").strip()
        checksum_value = str(checksum_sha256 or "").strip()
        if not storage_key_value:
            raise InputError("storage_key is required for persona visual assets.")  # noqa: TRY003
        if not mime_type_value:
            raise InputError("mime_type is required for persona visual assets.")  # noqa: TRY003
        if len(checksum_value) != 64:
            raise InputError("checksum_sha256 must be a 64-character hex digest.")  # noqa: TRY003
        try:
            byte_size_value = int(byte_size)
        except (TypeError, ValueError) as exc:
            raise InputError("byte_size must be an integer.") from exc  # noqa: TRY003
        if byte_size_value < 0:
            raise InputError("byte_size must be non-negative.")  # noqa: TRY003

        asset_id_value = str(asset_id or self._generate_uuid()).strip()
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        query = (
            "INSERT INTO persona_visual_assets("
            "id, pack_id, persona_id, user_id, asset_role, storage_key, original_filename, mime_type, "
            "byte_size, checksum_sha256, width, height, duration_ms, provenance, created_at, last_modified, deleted, version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            asset_id_value,
            pack_id,
            persona_id,
            user_id,
            role_value,
            storage_key_value,
            self._normalize_nullable_text(original_filename),
            mime_type_value,
            byte_size_value,
            checksum_value,
            int(width) if width is not None else None,
            int(height) if height is not None else None,
            int(duration_ms) if duration_ms is not None else None,
            provenance_value,
            now,
            now,
            bool_cast(False),
            1,
        )
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            conn.execute(prepared_query, prepared_params or ())

        asset = self.get_persona_visual_asset(
            asset_id=asset_id_value,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not asset:
            raise CharactersRAGDBError("Failed to load persona visual asset after creation.")  # noqa: TRY003
        return asset

    def get_persona_visual_asset(
        self,
        *,
        asset_id: str,
        pack_id: str,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT a.*
              FROM persona_visual_assets a
              JOIN persona_visual_packs p
                ON p.id = a.pack_id
               AND p.persona_id = a.persona_id
               AND p.user_id = a.user_id
             WHERE a.id = ?
               AND a.pack_id = ?
               AND a.persona_id = ?
               AND a.user_id = ?
               AND (? OR a.deleted = ?)
               AND p.deleted = ?
             LIMIT 1
        """
        cursor = self.execute_query(
            query,
            (
                asset_id,
                pack_id,
                persona_id,
                user_id,
                bool(include_deleted),
                deleted_false,
                deleted_false,
            ),
        )
        return self._persona_visual_asset_row_to_dict(cursor.fetchone())

    def list_persona_visual_assets(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT a.*
              FROM persona_visual_assets a
              JOIN persona_visual_packs p
                ON p.id = a.pack_id
               AND p.persona_id = a.persona_id
               AND p.user_id = a.user_id
             WHERE a.pack_id = ?
               AND a.persona_id = ?
               AND a.user_id = ?
               AND (? OR a.deleted = ?)
               AND p.deleted = ?
             ORDER BY a.created_at ASC, a.id ASC
             LIMIT ? OFFSET ?
        """
        cursor = self.execute_query(
            query,
            (
                pack_id,
                persona_id,
                user_id,
                bool(include_deleted),
                deleted_false,
                deleted_false,
                max(1, int(limit)),
                max(0, int(offset)),
            ),
        )
        return [
            item
            for row in cursor.fetchall()
            if (item := self._persona_visual_asset_row_to_dict(row)) is not None
        ]

    def create_persona_visual_candidate(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        job_id: str | None,
        proposed_manifest_patch: dict[str, Any] | None,
        generated_asset_ids: list[str] | None,
        prompt: str | None = None,
        status: str = "review",
        candidate_id: str | None = None,
    ) -> dict[str, Any]:
        status_value = self._normalize_persona_visual_enum(
            status,
            allowed=self._ALLOWED_PERSONA_VISUAL_CANDIDATE_STATUSES,
            field_name="candidate status",
            default="review",
        )
        manifest_patch_value = proposed_manifest_patch if isinstance(proposed_manifest_patch, dict) else {}
        generated_asset_ids_value = [
            str(asset_id).strip()
            for asset_id in (generated_asset_ids or [])
            if str(asset_id).strip()
        ]
        candidate_id_value = str(candidate_id or self._generate_uuid()).strip()
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        query = (
            "INSERT INTO persona_visual_candidates("
            "id, pack_id, persona_id, user_id, job_id, status, proposed_manifest_patch_json, "
            "generated_asset_ids_json, prompt, failure_reason, created_at, last_modified, deleted, version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            candidate_id_value,
            pack_id,
            persona_id,
            user_id,
            self._normalize_nullable_text(job_id),
            status_value,
            self._ensure_json_string(manifest_patch_value) or "{}",
            self._ensure_json_string(generated_asset_ids_value) or "[]",
            self._normalize_nullable_text(prompt),
            None,
            now,
            now,
            bool_cast(False),
            1,
        )
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            conn.execute(prepared_query, prepared_params or ())

        candidate = self.get_persona_visual_candidate(
            candidate_id=candidate_id_value,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not candidate:
            raise CharactersRAGDBError("Failed to load persona visual candidate after creation.")  # noqa: TRY003
        return candidate

    def get_persona_visual_candidate(
        self,
        *,
        candidate_id: str,
        pack_id: str,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        query = """
            SELECT c.*
              FROM persona_visual_candidates c
              JOIN persona_visual_packs p
                ON p.id = c.pack_id
               AND p.persona_id = c.persona_id
               AND p.user_id = c.user_id
             WHERE c.id = ?
               AND c.pack_id = ?
               AND c.persona_id = ?
               AND c.user_id = ?
               AND (? OR c.deleted = ?)
               AND p.deleted = ?
             LIMIT 1
        """
        cursor = self.execute_query(
            query,
            (
                candidate_id,
                pack_id,
                persona_id,
                user_id,
                bool(include_deleted),
                deleted_false,
                deleted_false,
            ),
        )
        return self._persona_visual_candidate_row_to_dict(cursor.fetchone())

    def list_persona_visual_candidates(
        self,
        *,
        pack_id: str,
        persona_id: str,
        user_id: str,
        status: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        deleted_false = False if self.backend_type == BackendType.POSTGRESQL else 0
        normalized_status = None
        if status is not None:
            normalized_status = self._normalize_persona_visual_enum(
                status,
                allowed=self._ALLOWED_PERSONA_VISUAL_CANDIDATE_STATUSES,
                field_name="candidate status",
            )
        query = """
            SELECT c.*
              FROM persona_visual_candidates c
              JOIN persona_visual_packs p
                ON p.id = c.pack_id
               AND p.persona_id = c.persona_id
               AND p.user_id = c.user_id
             WHERE c.pack_id = ?
               AND c.persona_id = ?
               AND c.user_id = ?
               AND (? IS NULL OR c.status = ?)
               AND (? OR c.deleted = ?)
               AND p.deleted = ?
             ORDER BY c.created_at DESC, c.id ASC
             LIMIT ? OFFSET ?
        """
        cursor = self.execute_query(
            query,
            (
                pack_id,
                persona_id,
                user_id,
                normalized_status,
                normalized_status,
                bool(include_deleted),
                deleted_false,
                deleted_false,
                max(1, int(limit)),
                max(0, int(offset)),
            ),
        )
        return [
            item
            for row in cursor.fetchall()
            if (item := self._persona_visual_candidate_row_to_dict(row)) is not None
        ]

    def update_persona_visual_candidate_status(
        self,
        *,
        candidate_id: str,
        pack_id: str,
        persona_id: str,
        user_id: str,
        status: str,
        failure_reason: str | None = None,
    ) -> dict[str, Any] | None:
        status_value = self._normalize_persona_visual_enum(
            status,
            allowed=self._ALLOWED_PERSONA_VISUAL_CANDIDATE_STATUSES,
            field_name="candidate status",
        )
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        query = (
            "UPDATE persona_visual_candidates "
            "SET status = ?, failure_reason = ?, last_modified = ?, version = version + 1 "
            "WHERE id = ? AND pack_id = ? AND persona_id = ? AND user_id = ? AND deleted = ?"
        )
        params = (
            status_value,
            self._normalize_nullable_text(failure_reason),
            now,
            candidate_id,
            pack_id,
            persona_id,
            user_id,
            bool_cast(False),
        )
        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            self._require_persona_visual_pack_owner(
                conn,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            conn.execute(prepared_query, prepared_params or ())

        return self.get_persona_visual_candidate(
            candidate_id=candidate_id,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )

    def list_persona_scope_rules(
        self,
        *,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        self.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        query = "SELECT * FROM persona_scope_rules WHERE persona_id = ? AND user_id = ?"
        params: list[Any] = [persona_id, user_id]
        if not include_deleted:
            query += " AND deleted = 0"
        query += " ORDER BY rule_type ASC, rule_value ASC, id ASC"
        cursor = self.execute_query(query, tuple(params))
        return [self._persona_scope_rule_row_to_dict(row) for row in cursor.fetchall() if row]

    def replace_persona_scope_rules(
        self,
        *,
        persona_id: str,
        user_id: str,
        rules: list[dict[str, Any]],
    ) -> int:
        if rules is None:
            rules = []
        now = self._get_current_utc_timestamp_iso()
        include_true = bool if self.backend_type == BackendType.POSTGRESQL else int

        normalized_rules: list[tuple[str, str, bool]] = []
        seen: set[tuple[str, str, bool]] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rule_type = self._normalize_persona_scope_rule_type(rule.get("rule_type"))
            rule_value = str(rule.get("rule_value") or "").strip()
            if not rule_value:
                raise InputError("persona scope rule_value is required.")  # noqa: TRY003
            include = self._as_bool(rule.get("include", True))
            key = (rule_type, rule_value, include)
            if key in seen:
                continue
            seen.add(key)
            normalized_rules.append(key)

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            prepared_update, update_params = self._prepare_backend_statement(
                (
                    "UPDATE persona_scope_rules "
                    "SET deleted = ?, last_modified = ?, version = version + 1 "
                    "WHERE persona_id = ? AND user_id = ? AND deleted = 0"
                ),
                (
                    include_true(True),
                    now,
                    persona_id,
                    user_id,
                ),
            )
            conn.execute(prepared_update, update_params or ())
            if not normalized_rules:
                return 0

            insert_query = (
                "INSERT INTO persona_scope_rules("
                "persona_id, user_id, rule_type, rule_value, include, created_at, last_modified, deleted, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            inserted = 0
            for rule_type, rule_value, include in normalized_rules:
                insert_params = (
                    persona_id,
                    user_id,
                    rule_type,
                    rule_value,
                    include_true(include),
                    now,
                    now,
                    include_true(False),
                    1,
                )
                prepared_insert, prepared_params = self._prepare_backend_statement(insert_query, insert_params)
                conn.execute(prepared_insert, prepared_params or ())
                inserted += 1
            return inserted

    def list_persona_policy_rules(
        self,
        *,
        persona_id: str,
        user_id: str,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        self.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        query = "SELECT * FROM persona_policy_rules WHERE persona_id = ? AND user_id = ?"
        params: list[Any] = [persona_id, user_id]
        if not include_deleted:
            query += " AND deleted = 0"
        query += " ORDER BY rule_kind ASC, rule_name ASC, id ASC"
        cursor = self.execute_query(query, tuple(params))
        return [self._persona_policy_rule_row_to_dict(row) for row in cursor.fetchall() if row]

    def replace_persona_policy_rules(
        self,
        *,
        persona_id: str,
        user_id: str,
        rules: list[dict[str, Any]],
    ) -> int:
        if rules is None:
            rules = []
        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int

        normalized_rules: list[tuple[str, str, bool, bool, int | None]] = []
        seen: set[tuple[str, str, bool, bool, int | None]] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rule_kind = self._normalize_persona_policy_rule_kind(rule.get("rule_kind"))
            rule_name = str(rule.get("rule_name") or "").strip()
            if not rule_name:
                raise InputError("persona policy rule_name is required.")  # noqa: TRY003
            allowed = self._as_bool(rule.get("allowed", True))
            require_confirmation = self._as_bool(rule.get("require_confirmation", False))
            max_calls = rule.get("max_calls_per_turn")
            if max_calls is not None:
                try:
                    max_calls = int(max_calls)
                except (TypeError, ValueError) as exc:
                    raise InputError("max_calls_per_turn must be an integer when provided.") from exc  # noqa: TRY003
                if max_calls < 1:
                    raise InputError("max_calls_per_turn must be >= 1 when provided.")  # noqa: TRY003
            key = (rule_kind, rule_name, allowed, require_confirmation, max_calls)
            if key in seen:
                continue
            seen.add(key)
            normalized_rules.append(key)

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            prepared_update, update_params = self._prepare_backend_statement(
                (
                    "UPDATE persona_policy_rules "
                    "SET deleted = ?, last_modified = ?, version = version + 1 "
                    "WHERE persona_id = ? AND user_id = ? AND deleted = 0"
                ),
                (
                    bool_cast(True),
                    now,
                    persona_id,
                    user_id,
                ),
            )
            conn.execute(prepared_update, update_params or ())
            if not normalized_rules:
                return 0

            insert_query = (
                "INSERT INTO persona_policy_rules("
                "persona_id, user_id, rule_kind, rule_name, allowed, require_confirmation, "
                "max_calls_per_turn, created_at, last_modified, deleted, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            inserted = 0
            for rule_kind, rule_name, allowed, require_confirmation, max_calls in normalized_rules:
                insert_params = (
                    persona_id,
                    user_id,
                    rule_kind,
                    rule_name,
                    bool_cast(allowed),
                    bool_cast(require_confirmation),
                    max_calls,
                    now,
                    now,
                    bool_cast(False),
                    1,
                )
                prepared_insert, prepared_params = self._prepare_backend_statement(insert_query, insert_params)
                conn.execute(prepared_insert, prepared_params or ())
                inserted += 1
            return inserted

    def create_persona_session(self, session_data: dict[str, Any]) -> str:
        persona_id = str(session_data.get("persona_id") or "").strip()
        user_id = str(session_data.get("user_id") or "").strip()
        if not persona_id:
            raise InputError("persona_id is required for persona session creation.")  # noqa: TRY003
        if not user_id:
            raise InputError("user_id is required for persona session creation.")  # noqa: TRY003

        session_id = str(session_data.get("id") or self._generate_uuid())
        scope_snapshot = session_data.get("scope_snapshot_json")
        if isinstance(scope_snapshot, str):
            scope_snapshot_json = scope_snapshot.strip() or "{}"
        else:
            scope_snapshot_json = self._ensure_json_string(scope_snapshot) or "{}"
        raw_preferences = session_data.get("preferences_json")
        if isinstance(raw_preferences, str):
            preferences_json = raw_preferences.strip() or "{}"
        else:
            preferences_json = self._ensure_json_string(raw_preferences) or "{}"

        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        activity_surface = self._normalize_persona_session_activity_surface(session_data.get("activity_surface"))

        with self.transaction() as conn:
            persona_row = self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            mode = self._normalize_persona_mode(session_data.get("mode") or persona_row.get("mode"))
            reuse_allowed_default = mode == "persistent_scoped"
            reuse_allowed = self._as_bool(session_data.get("reuse_allowed", reuse_allowed_default))
            status = self._normalize_persona_session_status(session_data.get("status") or "active")
            conversation_id = session_data.get("conversation_id")
            deleted_value = self._normalize_deleted_input(session_data.get("deleted", False))
            version = self._parse_version_input(session_data.get("version", 1))

            query = (
                "INSERT INTO persona_sessions("
                "id, persona_id, user_id, conversation_id, mode, reuse_allowed, status, "
                "scope_snapshot_json, preferences_json, activity_surface, created_at, last_modified, deleted, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            params = (
                session_id,
                persona_id,
                user_id,
                conversation_id,
                mode,
                bool_cast(reuse_allowed),
                status,
                scope_snapshot_json,
                preferences_json,
                activity_surface,
                session_data.get("created_at") or now,
                session_data.get("last_modified") or now,
                bool_cast(deleted_value),
                version,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            conn.execute(prepared_query, prepared_params or ())
        return session_id

    def get_persona_session(
        self,
        session_id: str,
        *,
        user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM persona_sessions WHERE id = ? AND user_id = ?"
        params: list[Any] = [session_id, user_id]
        if not include_deleted:
            query += " AND deleted = 0"
        cursor = self.execute_query(query, tuple(params))
        return self._persona_session_row_to_dict(cursor.fetchone())

    def list_persona_sessions(
        self,
        *,
        user_id: str,
        persona_id: str | None = None,
        activity_surface: str | None = None,
        status: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        if activity_surface is not None:
            clauses.append("activity_surface = ?")
            params.append(self._normalize_persona_session_activity_surface(activity_surface))
        if status is not None:
            normalized_status = self._normalize_persona_session_status(status)
            clauses.append("status = ?")
            params.append(normalized_status)
        if not include_deleted:
            clauses.append("deleted = 0")
        where_sql = " AND ".join(clauses)
        query = (
            "SELECT * FROM persona_sessions "  # nosec B608
            f"WHERE {where_sql} "
            "ORDER BY last_modified DESC, id ASC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, int(limit)), max(0, int(offset))])
        cursor = self.execute_query(query, tuple(params))
        return [self._persona_session_row_to_dict(row) for row in cursor.fetchall() if row]

    def update_persona_session(
        self,
        *,
        session_id: str,
        user_id: str,
        update_data: dict[str, Any],
        expected_version: int | None = None,
    ) -> bool:
        if not update_data:
            raise InputError("No session fields provided for update.")  # noqa: TRY003
        allowed_fields = {
            "conversation_id",
            "mode",
            "reuse_allowed",
            "status",
            "scope_snapshot_json",
            "preferences_json",
            "activity_surface",
            "deleted",
        }
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        set_parts: list[str] = []
        params: list[Any] = []

        for key, value in update_data.items():
            if key not in allowed_fields:
                continue
            if key == "mode":
                params.append(self._normalize_persona_mode(value))
                set_parts.append("mode = ?")
            elif key == "status":
                params.append(self._normalize_persona_session_status(value))
                set_parts.append("status = ?")
            elif key in {"reuse_allowed", "deleted"}:
                params.append(bool_cast(self._as_bool(value)))
                set_parts.append(f"{key} = ?")
            elif key == "scope_snapshot_json":
                if isinstance(value, str):
                    params.append(value.strip() or "{}")
                else:
                    params.append(self._ensure_json_string(value) or "{}")
                set_parts.append("scope_snapshot_json = ?")
            elif key == "preferences_json":
                if isinstance(value, str):
                    params.append(value.strip() or "{}")
                else:
                    params.append(self._ensure_json_string(value) or "{}")
                set_parts.append("preferences_json = ?")
            elif key == "activity_surface":
                params.append(self._normalize_persona_session_activity_surface(value))
                set_parts.append("activity_surface = ?")
            else:
                params.append(value)
                set_parts.append("conversation_id = ?")

        if not set_parts:
            raise InputError("No valid session fields provided for update.")  # noqa: TRY003

        now = self._get_current_utc_timestamp_iso()
        set_parts.extend(["last_modified = ?", "version = version + 1"])
        params.append(now)
        where_sql = "id = ? AND user_id = ? AND deleted = 0"
        params.extend([session_id, user_id])
        if expected_version is not None:
            where_sql += " AND version = ?"
            params.append(int(expected_version))

        query = f"UPDATE persona_sessions SET {', '.join(set_parts)} WHERE {where_sql}"  # nosec B608
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT version, deleted FROM persona_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if not existing:
                return False
            if self._as_bool(existing["deleted"]):
                return False
            if expected_version is not None and int(existing["version"]) != int(expected_version):
                raise ConflictError(  # noqa: TRY003
                    f"Persona session version mismatch (db has {existing['version']}, expected {expected_version}).",
                    entity="persona_sessions",
                    entity_id=session_id,
                )
            prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def add_persona_memory_entry(self, entry_data: dict[str, Any]) -> str:
        persona_id = str(entry_data.get("persona_id") or "").strip()
        user_id = str(entry_data.get("user_id") or "").strip()
        memory_type = str(entry_data.get("memory_type") or "").strip()
        content = str(entry_data.get("content") or "").strip()
        if not persona_id:
            raise InputError("persona_id is required for persona memory creation.")  # noqa: TRY003
        if not user_id:
            raise InputError("user_id is required for persona memory creation.")  # noqa: TRY003
        if not memory_type:
            raise InputError("memory_type is required for persona memory creation.")  # noqa: TRY003
        if not content:
            raise InputError("content is required for persona memory creation.")  # noqa: TRY003

        entry_id = str(entry_data.get("id") or self._generate_uuid())
        source_conversation_id = entry_data.get("source_conversation_id")
        scope_snapshot_id_raw = entry_data.get("scope_snapshot_id")
        session_id_raw = entry_data.get("session_id")
        scope_snapshot_id = str(scope_snapshot_id_raw).strip() if scope_snapshot_id_raw is not None else None
        if scope_snapshot_id == "":
            scope_snapshot_id = None
        session_id = str(session_id_raw).strip() if session_id_raw is not None else None
        if session_id == "":
            session_id = None
        try:
            salience = float(entry_data.get("salience", 0.0))
        except (TypeError, ValueError) as exc:
            raise InputError("salience must be a numeric value.") from exc  # noqa: TRY003

        now = self._get_current_utc_timestamp_iso()
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        archived = self._as_bool(entry_data.get("archived", False))
        deleted = self._as_bool(entry_data.get("deleted", False))
        version = int(entry_data.get("version", 1))
        if version < 1:
            raise InputError("version must be >= 1.")  # noqa: TRY003

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            query = (
                "INSERT INTO persona_memory_entries("
                "id, persona_id, user_id, memory_type, content, source_conversation_id, "
                "scope_snapshot_id, session_id, salience, archived, created_at, last_modified, deleted, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            params = (
                entry_id,
                persona_id,
                user_id,
                memory_type,
                content,
                source_conversation_id,
                scope_snapshot_id,
                session_id,
                salience,
                bool_cast(archived),
                entry_data.get("created_at") or now,
                entry_data.get("last_modified") or now,
                bool_cast(deleted),
                version,
            )
            prepared_query, prepared_params = self._prepare_backend_statement(query, params)
            conn.execute(prepared_query, prepared_params or ())
        return entry_id

    def list_persona_memory_entries(
        self,
        *,
        user_id: str,
        persona_id: str | None = None,
        memory_type: str | None = None,
        scope_snapshot_id: str | None = None,
        session_id: str | None = None,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(str(memory_type).strip())
        if scope_snapshot_id is not None:
            clauses.append("scope_snapshot_id = ?")
            params.append(str(scope_snapshot_id).strip())
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(str(session_id).strip())
        if not include_archived:
            clauses.append("archived = 0")
        if not include_deleted:
            clauses.append("deleted = 0")
        where_sql = " AND ".join(clauses)
        query = (
            "SELECT * FROM persona_memory_entries "  # nosec B608
            f"WHERE {where_sql} "
            "ORDER BY last_modified DESC, id ASC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, int(limit)), max(0, int(offset))])
        cursor = self.execute_query(query, tuple(params))
        return [self._persona_memory_row_to_dict(row) for row in cursor.fetchall() if row]

    def get_persona_memory_entry_by_id(
        self,
        *,
        entry_id: str,
        user_id: str,
        persona_id: str | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        clauses = ["id = ?", "user_id = ?"]
        params: list[Any] = [entry_id, user_id]
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        if not include_deleted:
            clauses.append("deleted = 0")
        query = f"SELECT * FROM persona_memory_entries WHERE {' AND '.join(clauses)}"  # nosec B608
        cursor = self.execute_query(query, tuple(params))
        return self._persona_memory_row_to_dict(cursor.fetchone())

    def count_persona_memory_entries(
        self,
        *,
        user_id: str,
        persona_id: str | None = None,
        memory_type: str | None = None,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> int:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(str(memory_type).strip())
        if not include_archived:
            clauses.append("archived = 0")
        if not include_deleted:
            clauses.append("deleted = 0")
        query = f"SELECT COUNT(*) FROM persona_memory_entries WHERE {' AND '.join(clauses)}"  # nosec B608
        cursor = self.execute_query(query, tuple(params))
        row = cursor.fetchone()
        return row[0] if row else 0

    def set_persona_memory_archived(
        self,
        *,
        entry_id: str,
        user_id: str,
        archived: bool = True,
        persona_id: str | None = None,
    ) -> bool:
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        now = self._get_current_utc_timestamp_iso()
        clauses = ["id = ?", "user_id = ?", "deleted = 0"]
        params: list[Any] = [entry_id, user_id]
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        where_sql = " AND ".join(clauses)
        query = (
            "UPDATE persona_memory_entries "
            "SET archived = ?, last_modified = ?, version = version + 1 "
            f"WHERE {where_sql}"  # nosec B608
        )
        update_params = [bool_cast(archived), now, *params]
        cursor = self.execute_query(query, tuple(update_params), commit=True)
        return bool(cursor.rowcount and cursor.rowcount > 0)

    def update_persona_memory_entry(
        self,
        *,
        entry_id: str,
        user_id: str,
        persona_id: str,
        update_data: dict[str, Any],
    ) -> bool:
        if not update_data:
            raise InputError("No persona memory fields provided for update.")  # noqa: TRY003

        allowed_fields = {
            "content",
            "memory_type",
            "salience",
            "source_conversation_id",
            "scope_snapshot_id",
            "session_id",
            "archived",
            "deleted",
        }
        set_parts: list[str] = []
        params: list[Any] = []
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int

        for key, value in update_data.items():
            if key not in allowed_fields:
                continue
            if key == "content":
                content = str(value or "").strip()
                if not content:
                    raise InputError("content cannot be empty.")  # noqa: TRY003
                params.append(content)
                set_parts.append("content = ?")
            elif key == "memory_type":
                mt = str(value or "").strip()
                if not mt:
                    raise InputError("memory_type cannot be empty.")  # noqa: TRY003
                params.append(mt)
                set_parts.append("memory_type = ?")
            elif key == "salience":
                try:
                    params.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise InputError("salience must be a numeric value.") from exc  # noqa: TRY003
                set_parts.append("salience = ?")
            elif key in {"archived", "deleted"}:
                params.append(bool_cast(self._as_bool(value)))
                set_parts.append(f"{key} = ?")
            else:
                normalized = None if value is None else str(value).strip() or None
                params.append(normalized)
                set_parts.append(f"{key} = ?")

        if not set_parts:
            raise InputError("No valid persona memory fields provided for update.")  # noqa: TRY003

        now = self._get_current_utc_timestamp_iso()
        set_parts.extend(["last_modified = ?", "version = version + 1"])
        params.append(now)

        with self.transaction() as conn:
            self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
            query = (
                "UPDATE persona_memory_entries "
                f"SET {', '.join(set_parts)} "
                "WHERE id = ? AND persona_id = ? AND user_id = ? AND deleted = 0"  # nosec B608
            )
            params.extend([entry_id, persona_id, user_id])
            prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
            cursor = conn.execute(prepared_query, prepared_params or ())
            return cursor.rowcount > 0

    def backfill_persona_memory_scope_namespace(
        self,
        *,
        user_id: str,
        persona_id: str,
        scope_snapshot_id: str,
        require_missing_session_id: bool = True,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> int:
        scope_value = str(scope_snapshot_id or "").strip()
        if not scope_value:
            raise InputError("scope_snapshot_id is required for namespace backfill.")  # noqa: TRY003

        now = self._get_current_utc_timestamp_iso()
        clauses = [
            "user_id = ?",
            "persona_id = ?",
            "(scope_snapshot_id IS NULL OR scope_snapshot_id = '')",
        ]
        params: list[Any] = [user_id, persona_id]
        if require_missing_session_id:
            clauses.append("(session_id IS NULL OR session_id = '')")
        if not include_archived:
            clauses.append("archived = 0")
        if not include_deleted:
            clauses.append("deleted = 0")
        where_sql = " AND ".join(clauses)
        query = (
            "UPDATE persona_memory_entries "
            "SET scope_snapshot_id = ?, last_modified = ?, version = version + 1 "
            f"WHERE {where_sql}"  # nosec B608
        )
        update_params = [scope_value, now, *params]
        cursor = self.execute_query(query, tuple(update_params), commit=True)
        return int(cursor.rowcount or 0)

    def soft_delete_persona_memory_entry(
        self,
        *,
        entry_id: str,
        user_id: str,
        persona_id: str | None = None,
    ) -> bool:
        bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
        now = self._get_current_utc_timestamp_iso()
        clauses = ["id = ?", "user_id = ?", "deleted = 0"]
        params: list[Any] = [entry_id, user_id]
        if persona_id is not None:
            clauses.append("persona_id = ?")
            params.append(persona_id)
        where_sql = " AND ".join(clauses)
        query = (
            "UPDATE persona_memory_entries "
            "SET deleted = ?, archived = ?, last_modified = ?, version = version + 1 "
            f"WHERE {where_sql}"  # nosec B608
        )
        update_params = [bool_cast(True), bool_cast(True), now, *params]
        cursor = self.execute_query(query, tuple(update_params), commit=True)
        return bool(cursor.rowcount and cursor.rowcount > 0)
