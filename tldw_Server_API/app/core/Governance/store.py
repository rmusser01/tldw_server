"""Persistence primitives for governance policies, gaps, and validation traces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import aiosqlite

_VALID_ACTIONS = frozenset({"allow", "warn", "require_approval", "deny"})


@dataclass(frozen=True)
class GapRecord:
    """Normalized governance gap record."""

    id: int
    question: str
    question_fingerprint: str
    category: str
    status: str
    org_id: Optional[int]
    team_id: Optional[int]
    persona_id: Optional[str]
    workspace_id: Optional[str]
    resolution_mode: Optional[str]
    resolution_text: Optional[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "GapRecord":
        return cls(
            id=int(row["id"]),
            question=str(row["question"]),
            question_fingerprint=str(row["question_fingerprint"]),
            category=str(row["category"]),
            status=str(row["status"]),
            org_id=row["org_id"],
            team_id=row["team_id"],
            persona_id=row["persona_id"],
            workspace_id=row["workspace_id"],
            resolution_mode=row["resolution_mode"],
            resolution_text=row["resolution_text"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


class GovernanceStore:
    """SQLite-backed governance store focused on deterministic safety semantics."""

    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path

    def _connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.sqlite_path)

    async def ensure_schema(self) -> None:
        """Ensure governance storage tables exist."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS governance_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id INTEGER,
                    team_id INTEGER,
                    persona_id TEXT,
                    workspace_id TEXT,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL DEFAULT 'warn',
                    title TEXT NOT NULL DEFAULT '',
                    body_markdown TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    priority INTEGER NOT NULL DEFAULT 0,
                    effective_from TEXT,
                    expires_at TEXT,
                    created_by TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS governance_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id INTEGER,
                    team_id INTEGER,
                    persona_id TEXT,
                    workspace_id TEXT,
                    question TEXT NOT NULL,
                    question_fingerprint TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    resolution_mode TEXT,
                    resolution_text TEXT,
                    owner_user_id INTEGER,
                    review_due_at TEXT,
                    resolved_by TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS uq_governance_gaps_open_dedupe
                ON governance_gaps (
                    question_fingerprint,
                    category,
                    COALESCE(org_id, -1),
                    COALESCE(team_id, -1),
                    COALESCE(persona_id, ''),
                    COALESCE(workspace_id, '')
                )
                WHERE status = 'open';

                CREATE INDEX IF NOT EXISTS idx_governance_rules_active_category
                ON governance_rules (status, category, priority, updated_at);
                """
            )
            await self._ensure_schema_columns(db)
            await db.commit()

    async def _ensure_schema_columns(self, db: aiosqlite.Connection) -> None:
        """Apply lightweight additive migrations for existing governance DBs."""
        cursor = await db.execute("PRAGMA table_info(governance_rules)")
        columns = {str(row[1]) for row in await cursor.fetchall()}
        if "action" not in columns:
            await db.execute(
                "ALTER TABLE governance_rules ADD COLUMN action TEXT NOT NULL DEFAULT 'warn'"
            )

    async def table_exists(self, table_name: str) -> bool:
        """Return True when a SQLite table exists."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                (table_name,),
            )
            row = await cursor.fetchone()
            return bool(row)

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return " ".join(str(text or "").strip().split())

    @classmethod
    def _normalize_optional_scope_text(cls, text: str | None) -> str | None:
        normalized = cls._normalize_text(text)
        return normalized or None

    @staticmethod
    def _validate_optional_scope_id(field: str, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{field} must be non-negative")
        return normalized

    @classmethod
    def _coerce_metadata_scope_id(cls, field: str, metadata: Mapping[str, Any]) -> int | None:
        value = metadata.get(field)
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        try:
            normalized = int(str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        return cls._validate_optional_scope_id(field, normalized)

    @classmethod
    def _normalize_category(cls, category: str | None) -> str:
        return cls._normalize_text(category).lower() or "general"

    @classmethod
    def _question_fingerprint(cls, question: str) -> str:
        normalized = cls._normalize_text(question).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _scope_level(row: aiosqlite.Row) -> int:
        if row["workspace_id"] is not None:
            return 4
        if row["persona_id"] is not None:
            return 3
        if row["team_id"] is not None:
            return 2
        if row["org_id"] is not None:
            return 1
        return 0

    async def get_candidates(
        self,
        *,
        surface: str | None = None,
        summary: str | None = None,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return active governance rule candidates for a category and optional scope."""
        del surface, summary
        metadata_map = metadata or {}
        normalized_category = self._normalize_category(category)
        org_id = self._coerce_metadata_scope_id("org_id", metadata_map)
        team_id = self._coerce_metadata_scope_id("team_id", metadata_map)
        persona_id = self._normalize_optional_scope_text(
            None if metadata_map.get("persona_id") is None else str(metadata_map.get("persona_id"))
        )
        workspace_id = self._normalize_optional_scope_text(
            None if metadata_map.get("workspace_id") is None else str(metadata_map.get("workspace_id"))
        )

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id, org_id, team_id, persona_id, workspace_id,
                       category, action, title, priority, updated_at
                FROM governance_rules
                WHERE status = 'active'
                  AND LOWER(category) IN (?, 'general')
                  AND (effective_from IS NULL OR effective_from <= CURRENT_TIMESTAMP)
                  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                  AND (org_id IS NULL OR org_id = ?)
                  AND (team_id IS NULL OR team_id = ?)
                  AND (persona_id IS NULL OR persona_id = ?)
                  AND (workspace_id IS NULL OR workspace_id = ?)
                ORDER BY priority DESC, updated_at DESC, id ASC
                """,
                (normalized_category, org_id, team_id, persona_id, workspace_id),
            )
            rows = await cursor.fetchall()

        candidates: list[dict[str, Any]] = []
        for row in rows:
            action = self._normalize_text(row["action"]).lower()
            if action not in _VALID_ACTIONS:
                action = "deny"
            candidates.append(
                {
                    "action": action,
                    "scope_level": self._scope_level(row),
                    "priority": int(row["priority"]),
                    "updated_at": str(row["updated_at"]),
                    "source_id": f"governance_rules:{int(row['id'])}",
                    "reason": self._normalize_text(row["title"]) or None,
                }
            )
        return candidates

    async def upsert_open_gap(
        self,
        *,
        question: str,
        category: str,
        org_id: int | None = None,
        team_id: int | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        resolution_mode: str | None = None,
    ) -> GapRecord:
        """Create or return an existing open gap for the same normalized question/scope."""
        normalized_question = self._normalize_text(question)
        normalized_category = self._normalize_text(category).lower()
        normalized_org_id = self._validate_optional_scope_id("org_id", org_id)
        normalized_team_id = self._validate_optional_scope_id("team_id", team_id)
        normalized_persona_id = self._normalize_optional_scope_text(persona_id)
        normalized_workspace_id = self._normalize_optional_scope_text(workspace_id)
        if not normalized_question:
            raise ValueError("question is required")
        if not normalized_category:
            raise ValueError("category is required")

        fingerprint = self._question_fingerprint(normalized_question)
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            try:
                await db.execute(
                    """
                    INSERT INTO governance_gaps (
                        org_id, team_id, persona_id, workspace_id,
                        question, question_fingerprint, category, status, resolution_mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        normalized_org_id,
                        normalized_team_id,
                        normalized_persona_id,
                        normalized_workspace_id,
                        normalized_question,
                        fingerprint,
                        normalized_category,
                        resolution_mode,
                    ),
                )
                await db.commit()
            except aiosqlite.IntegrityError:
                # Existing open gap for same fingerprint/scope. Return that row.
                pass

            cursor = await db.execute(
                """
                SELECT id, org_id, team_id, persona_id, workspace_id,
                       question, question_fingerprint, category, status,
                       resolution_mode, resolution_text, created_at, updated_at
                FROM governance_gaps
                WHERE status = 'open'
                  AND question_fingerprint = ?
                  AND category = ?
                  AND COALESCE(org_id, -1) = COALESCE(?, -1)
                  AND COALESCE(team_id, -1) = COALESCE(?, -1)
                  AND COALESCE(persona_id, '') = COALESCE(?, '')
                  AND COALESCE(workspace_id, '') = COALESCE(?, '')
                ORDER BY id ASC
                LIMIT 1
                """,
                (
                    fingerprint,
                    normalized_category,
                    normalized_org_id,
                    normalized_team_id,
                    normalized_persona_id,
                    normalized_workspace_id,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("failed to load governance gap after upsert")
            return GapRecord.from_row(row)
