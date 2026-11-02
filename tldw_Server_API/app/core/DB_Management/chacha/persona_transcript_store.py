"""Bounded, owner-bound Persona transcripts; runtime plans are never persisted here."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.Persona.session_manager import TURN_METADATA_DIAGNOSTIC_CHARS

MAX_TURNS = 200
MAX_CONTENT_CHARS = 8192
MAX_METADATA_CHARS = 4096
TTL_SECONDS = 86400
MAX_SESSIONS = 200
# ASCII JSON expands an astral character to two six-byte surrogate escapes.
MAX_SNAPSHOT_BYTES = (
    MAX_TURNS * (MAX_CONTENT_CHARS * 12 + MAX_METADATA_CHARS + TURN_METADATA_DIAGNOSTIC_CHARS + 4 * 64 * 12 + 256) + 2
)


def persona_transcript_schema_sql() -> str:
    """The header survives payload retirement so an old runtime cannot republish it."""
    return """CREATE TABLE IF NOT EXISTS persona_session_transcripts (
        session_id TEXT PRIMARY KEY NOT NULL REFERENCES persona_sessions(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL,
        persona_id TEXT NOT NULL,
        revision BIGINT NOT NULL CHECK (revision >= 1),
        last_activity TEXT NOT NULL,
        turn_count INTEGER NOT NULL CHECK (turn_count BETWEEN 0 AND 200),
        turns_json TEXT NOT NULL
    )"""


def verify_persona_transcript_sqlite(conn: Any) -> None:
    """Reject missing or weakened current-schema storage without rebuilding user data."""
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='persona_session_transcripts'"
        ).fetchone()
        expected = re.sub(r"\s+", "", persona_transcript_schema_sql()).lower().replace("ifnotexists", "")
        actual = re.sub(r"\s+", "", str(row[0] if row else "")).lower()
        if actual != expected:
            raise SchemaError("Persona transcript SQLite schema is missing or unsafe.")
    except sqlite3.Error as exc:
        raise SchemaError("Persona transcript SQLite schema verification failed.") from exc


def persona_transcript_owner_predicate() -> str:
    return """user_id = current_setting('app.current_user_id', true)
        AND EXISTS (SELECT 1 FROM persona_sessions AS parent
            WHERE parent.id = persona_session_transcripts.session_id
              AND parent.user_id = persona_session_transcripts.user_id
              AND parent.persona_id = persona_session_transcripts.persona_id)"""


def persona_transcript_rls_sql() -> list[str]:
    predicate = persona_transcript_owner_predicate()
    return [
        "ALTER TABLE persona_session_transcripts ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE persona_session_transcripts FORCE ROW LEVEL SECURITY",
        "CREATE POLICY persona_transcript_owner ON persona_session_transcripts "
        f"USING ({predicate}) WITH CHECK ({predicate})",
    ]


def verify_persona_transcript_postgres(backend: Any, conn: Any, normalize: Any) -> None:
    """Check authority without request-time DDL or permissive-policy repair."""
    backend.execute("LOCK TABLE persona_session_transcripts IN ACCESS SHARE MODE", connection=conn)
    backend.execute(
        "SELECT session_id,user_id,persona_id,revision,last_activity,turn_count,turns_json "
        "FROM persona_session_transcripts LIMIT 0",
        connection=conn,
    )
    columns = backend.execute(
        "SELECT column_name,data_type,is_nullable FROM information_schema.columns "
        "WHERE table_schema=current_schema() AND table_name='persona_session_transcripts'",
        connection=conn,
    ).rows
    expected_columns = {
        name: (kind, "NO")
        for name, kind in (
            ("session_id", "text"),
            ("user_id", "text"),
            ("persona_id", "text"),
            ("revision", "bigint"),
            ("last_activity", "text"),
            ("turn_count", "integer"),
            ("turns_json", "text"),
        )
    }
    constraints = backend.execute(
        "SELECT convalidated,condeferrable,pg_get_constraintdef(oid) AS definition FROM pg_constraint "
        "WHERE conrelid='persona_session_transcripts'::regclass AND contype IN ('p','f','c')",
        connection=conn,
    ).rows
    expected_constraints = {
        normalize(value)
        for value in (
            "PRIMARY KEY (session_id)",
            "FOREIGN KEY (session_id) REFERENCES persona_sessions(id) ON DELETE CASCADE",
            "CHECK (revision >= 1)",
            "CHECK (turn_count >= 0 AND turn_count <= 200)",
        )
    }
    if (
        {row["column_name"]: (row["data_type"], row["is_nullable"]) for row in columns} != expected_columns
        or {normalize(row["definition"]) for row in constraints} != expected_constraints
        or len(constraints) != len(expected_constraints)
        or any(not row["convalidated"] or row["condeferrable"] for row in constraints)
    ):
        raise SchemaError("Persona transcript PostgreSQL schema constraints drifted.")
    tables = backend.execute(
        "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid='persona_session_transcripts'::regclass",
        connection=conn,
    ).rows
    policies = backend.execute(
        "SELECT policyname,permissive,roles::text,cmd,qual,with_check FROM pg_policies "
        "WHERE schemaname=current_schema() AND tablename='persona_session_transcripts'",
        connection=conn,
    ).rows
    expected = normalize(persona_transcript_owner_predicate()).replace("persona_session_transcripts.", "")
    if (
        len(tables) != 1
        or not tables[0]["relrowsecurity"]
        or not tables[0]["relforcerowsecurity"]
        or len(policies) != 1
    ):
        raise SchemaError("Persona transcript PostgreSQL ownership authority is missing.")
    policy = policies[0]
    if (
        policy["policyname"] != "persona_transcript_owner"
        or policy["permissive"] != "PERMISSIVE"
        or policy["roles"] != "{public}"
        or policy["cmd"] != "ALL"
        or normalize(policy["qual"]).replace("persona_session_transcripts.", "") != expected
        or normalize(policy["with_check"]).replace("persona_session_transcripts.", "") != expected
    ):
        raise SchemaError("Persona transcript PostgreSQL ownership policy drifted.")


class PersonaTranscriptStore:
    """Use the caller's ChaCha checkout and transaction ownership."""

    def __init__(self, db: Any) -> None:
        self.db = db

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _snapshot(row: Any, *, cutoff: str, include_turns: bool = True) -> dict[str, Any]:
        expired = str(row["last_activity"]) < cutoff and int(row["turn_count"]) > 0
        result = {
            "session_id": str(row["session_id"]),
            "persona_id": str(row["persona_id"]),
            "revision": int(row["revision"]),
            "retired": expired or int(row["turn_count"]) == 0,
            "turn_count": 0 if expired else int(row["turn_count"]),
        }
        if include_turns:
            result["turns"] = [] if expired else json.loads(row["turns_json"])
        return result

    def _query(self, conn: Any, sql: str, params: tuple[Any, ...]) -> Any:
        query, values = self.db._prepare_backend_statement(sql, params)
        return conn.execute(query, values or ())

    def get_persona_transcript(self, session_id: str, *, user_id: str, renew: bool = False) -> dict[str, Any] | None:
        row = self.db.execute_query(
            "SELECT t.* FROM persona_session_transcripts t JOIN persona_sessions s ON s.id=t.session_id "
            "WHERE t.session_id=? AND t.user_id=? AND s.user_id=? AND s.persona_id=t.persona_id AND s.deleted=0",
            (session_id, user_id, user_id),
        ).fetchone()
        if row is None:
            return None
        if not renew:
            cutoff = (self._now() - timedelta(seconds=TTL_SECONDS)).isoformat()
            return self._snapshot(row, cutoff=cutoff)
        with self.db.transaction() as conn:
            self._lock_active_session(conn, session_id=session_id, user_id=user_id, persona_id=str(row["persona_id"]))
            row = self._query(
                conn,
                "SELECT * FROM persona_session_transcripts WHERE session_id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                return None
            now = self._now()
            snapshot = self._snapshot(row, cutoff=(now - timedelta(seconds=TTL_SECONDS)).isoformat())
            if snapshot["retired"]:
                return snapshot
            self._query(
                conn,
                "UPDATE persona_session_transcripts SET last_activity=?,revision=revision+1 "
                "WHERE session_id=? AND user_id=?",
                (now.isoformat(), session_id, user_id),
            )
            return {**snapshot, "revision": int(snapshot["revision"]) + 1}

    def list_persona_transcript_counts(self, *, user_id: str, session_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(session_ids))
        if len(ids) > 200:
            raise InputError("Persona transcript count page exceeds 200 sessions.")
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.execute_query(
            "SELECT t.session_id,t.persona_id,t.revision,t.last_activity,t.turn_count "
            "FROM persona_session_transcripts t JOIN persona_sessions s ON s.id=t.session_id "
            "WHERE t.user_id=? AND s.user_id=? AND s.persona_id=t.persona_id AND s.deleted=0 "
            f"AND t.session_id IN ({placeholders})",  # nosec B608 -- Only bounded parameter placeholders are interpolated.
            (user_id, user_id, *ids),
        ).fetchall()
        cutoff = (self._now() - timedelta(seconds=TTL_SECONDS)).isoformat()
        return {str(row["session_id"]): self._snapshot(row, cutoff=cutoff, include_turns=False) for row in rows}

    def _lock_active_session(
        self, conn: Any, *, session_id: str, user_id: str, persona_id: str, allow_unpersisted: bool = False
    ) -> bool:
        if self.db.backend_type == BackendType.POSTGRESQL:
            digest = hashlib.sha256(f"persona_transcript_v1\0{user_id}".encode()).digest()
            keys = tuple(int.from_bytes(digest[i : i + 4], "big", signed=True) for i in (0, 4))
            self._query(conn, "SELECT pg_advisory_xact_lock(?, ?)", keys).fetchone()
        lock = " FOR UPDATE OF s" if self.db.backend_type == BackendType.POSTGRESQL else ""
        parent = self._query(
            conn,
            "SELECT s.id FROM persona_sessions s JOIN persona_profiles p ON p.id=s.persona_id "  # nosec B608 -- The only SQL suffix is the fixed backend row-lock clause; values stay bound.
            "WHERE s.id=? AND s.user_id=? AND s.persona_id=? AND s.deleted=0 "
            "AND s.status NOT IN ('closed','archived') "
            "AND p.user_id=? AND p.deleted=0 AND p.is_active=1" + lock,
            (session_id, user_id, persona_id, user_id),
        ).fetchone()
        if parent is None:
            if (
                allow_unpersisted
                and self._query(conn, "SELECT id FROM persona_sessions WHERE id=?", (session_id,)).fetchone() is None
            ):
                return False
            raise ConflictError(
                "Persona transcript session is unavailable.", entity="persona_sessions", entity_id=session_id
            )
        return True

    def validate_persona_transcript_session(
        self,
        *,
        session_id: str,
        user_id: str,
        persona_id: str,
        allow_unpersisted: bool = False,
    ) -> bool:
        """Admit an active owned parent or a truly absent temporary scaffold before effects."""
        with self.db.transaction() as conn:
            return self._lock_active_session(
                conn, session_id=session_id, user_id=user_id, persona_id=persona_id, allow_unpersisted=allow_unpersisted
            )

    def append_persona_transcript_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        persona_id: str,
        turn: dict[str, Any],
        parent_turn_id: str | None = None,
        allow_unpersisted: bool = False,
    ) -> dict[str, Any] | None:
        """Commit one prepared turn before runtime publication; serialize owner capacity."""
        if (
            set(turn) != {"turn_id", "timestamp", "role", "type", "content", "metadata"}
            or any(
                not isinstance(turn[key], str) or not 0 < len(turn[key]) <= 64
                for key in ("turn_id", "timestamp", "role", "type")
            )
            or not isinstance(turn["content"], str)
            or len(turn["content"]) > MAX_CONTENT_CHARS
            or not isinstance(turn["metadata"], dict)
            or len(json.dumps(turn["metadata"], ensure_ascii=True, sort_keys=True))
            > MAX_METADATA_CHARS + TURN_METADATA_DIAGNOSTIC_CHARS
        ):
            raise InputError("Invalid bounded Persona transcript turn.")
        invalidations = []
        with self.db.transaction() as conn:
            if not self._lock_active_session(
                conn, session_id=session_id, user_id=user_id, persona_id=persona_id, allow_unpersisted=allow_unpersisted
            ):
                return None
            # Waiting for another writer must not publish an older activity time.
            now = self._now()
            cutoff = (now - timedelta(seconds=TTL_SECONDS)).isoformat()
            row = self._query(
                conn,
                "SELECT * FROM persona_session_transcripts WHERE session_id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
            previous = json.loads(row["turns_json"]) if row is not None else []
            for saved in previous:
                if saved["turn_id"] == turn["turn_id"]:
                    if saved != turn:
                        raise ConflictError("Persona turn ID was reused with different content.")
                    return {**self._snapshot(row, cutoff=cutoff), "invalidations": []}
            snapshot = self._snapshot(row, cutoff=cutoff) if row is not None else {"revision": 0, "turns": []}
            turns = snapshot["turns"]
            if parent_turn_id is not None and not any(
                item["turn_id"] == parent_turn_id and item["role"] == "user" for item in turns
            ):
                raise ConflictError("The originating Persona turn is no longer retained.")
            turns = [*turns, turn][-MAX_TURNS:]
            payload = json.dumps(turns, ensure_ascii=True, sort_keys=True)
            if len(payload.encode("ascii")) > MAX_SNAPSHOT_BYTES:
                raise InputError("Persona transcript exceeds its bounded storage size.")
            revision = int(snapshot["revision"]) + 1
            self._query(
                conn,
                "INSERT INTO persona_session_transcripts(session_id,user_id,persona_id,revision,last_activity,turn_count,turns_json) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET revision=excluded.revision, "
                "last_activity=excluded.last_activity,turn_count=excluded.turn_count,turns_json=excluded.turns_json",
                (session_id, user_id, persona_id, revision, now.isoformat(), len(turns), payload),
            )
            expired = self._query(
                conn,
                "UPDATE persona_session_transcripts SET turns_json='[]',turn_count=0,revision=revision+1 "
                "WHERE user_id=? AND turn_count>0 AND last_activity<? RETURNING session_id,persona_id,revision",
                (user_id, cutoff),
            ).fetchall()
            invalidations.extend(dict(item) for item in expired)
            retained = self._query(
                conn,
                "SELECT session_id,persona_id,revision FROM persona_session_transcripts "
                "WHERE user_id=? AND turn_count>0 ORDER BY last_activity DESC,session_id DESC",
                (user_id,),
            ).fetchall()
            for old in retained[MAX_SESSIONS:]:
                self._query(
                    conn,
                    "UPDATE persona_session_transcripts SET turns_json='[]',turn_count=0,revision=revision+1 "
                    "WHERE session_id=? AND user_id=?",
                    (old["session_id"], user_id),
                )
                invalidations.append(
                    {
                        "session_id": old["session_id"],
                        "persona_id": old["persona_id"],
                        "revision": int(old["revision"]) + 1,
                    }
                )
        return {
            "session_id": session_id,
            "persona_id": persona_id,
            "revision": revision,
            "turn_count": len(turns),
            "turns": turns,
            "retired": False,
            "invalidations": invalidations,
        }
