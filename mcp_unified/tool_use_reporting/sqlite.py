"""SQLAlchemy-backed SQLite store for MCP tool-use reporting."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    desc,
    insert,
    or_,
    select,
)
from sqlalchemy.engine import Engine, URL

from mcp_unified.tool_use_reporting.models import (
    ToolUseEvent,
    ToolUseEventExportFormat,
    ToolUseEventQuery,
)
from mcp_unified.tool_use_reporting.store import (
    cutoff_epoch_us,
    decode_event_cursor,
)

_T = TypeVar("_T")


class SQLiteToolUseEventStore:
    """SQLite-backed metadata-only tool-use event store."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = MetaData()
        self._table = Table(
            "tool_use_events",
            self._metadata,
            Column("event_id", String(64), primary_key=True),
            Column("created_at_epoch_us", Integer, nullable=False),
            Column("runtime_surface", String(32), nullable=False),
            Column("requested_tool_name", String(255), nullable=False),
            Column("effective_tool_name", String(255), nullable=False),
            Column("profile_id", String(255), nullable=True),
            Column("mode_id", String(255), nullable=True),
            Column("model_id", String(255), nullable=True),
            Column("tool_prompt_id", String(255), nullable=True),
            Column("status", String(64), nullable=False),
            Column("duration_ms", Float, nullable=True),
            Column("is_write", Boolean, nullable=True),
            Column("payload_json", Text, nullable=False),
        )
        Index(
            "ix_tool_use_events_time",
            self._table.c.created_at_epoch_us,
            self._table.c.event_id,
        )
        Index("ix_tool_use_events_profile", self._table.c.profile_id)
        Index("ix_tool_use_events_model", self._table.c.model_id)
        Index("ix_tool_use_events_requested_tool", self._table.c.requested_tool_name)
        Index("ix_tool_use_events_effective_tool", self._table.c.effective_tool_name)
        Index("ix_tool_use_events_prompt", self._table.c.tool_prompt_id)
        Index("ix_tool_use_events_status", self._table.c.status)
        Index("ix_tool_use_events_runtime_surface", self._table.c.runtime_surface)

        self._engine: Engine = create_engine(
            URL.create("sqlite", database=str(self._path)),
            connect_args={"check_same_thread": False},
            future=True,
        )
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    async def _run_db(self, fn: Callable[[], _T]) -> _T:
        """Run blocking SQLAlchemy work off the event loop."""

        return await asyncio.to_thread(fn)

    async def _ensure_schema(self) -> None:
        """Create the schema once."""

        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            await self._run_db(lambda: self._metadata.create_all(self._engine))
            self._schema_ready = True

    def _event_values(self, event: ToolUseEvent) -> dict[str, Any]:
        """Return SQL scalar columns plus full JSON payload for an event."""

        return {
            "event_id": event.event_id,
            "created_at_epoch_us": event.created_at_epoch_us,
            "runtime_surface": event.runtime_surface,
            "requested_tool_name": event.requested_tool_name,
            "effective_tool_name": event.effective_tool_name,
            "profile_id": event.profile_id,
            "mode_id": event.mode_id,
            "model_id": event.model_id,
            "tool_prompt_id": event.tool_prompt_id,
            "status": event.status,
            "duration_ms": event.duration_ms,
            "is_write": event.is_write,
            "payload_json": event.model_dump_json(),
        }

    def _query_clauses(self, query: ToolUseEventQuery) -> list[Any]:
        """Return SQLAlchemy WHERE clauses for a query."""

        table = self._table
        clauses: list[Any] = []
        cursor = decode_event_cursor(query.cursor)
        if cursor is not None:
            cursor_epoch_us, cursor_event_id = cursor
            clauses.append(
                or_(
                    table.c.created_at_epoch_us < cursor_epoch_us,
                    and_(
                        table.c.created_at_epoch_us == cursor_epoch_us,
                        table.c.event_id < cursor_event_id,
                    ),
                )
            )
        if query.created_at_epoch_us_gte is not None:
            clauses.append(table.c.created_at_epoch_us >= query.created_at_epoch_us_gte)
        if query.created_at_epoch_us_lt is not None:
            clauses.append(table.c.created_at_epoch_us < query.created_at_epoch_us_lt)
        for field_name in (
            "runtime_surface",
            "requested_tool_name",
            "effective_tool_name",
            "profile_id",
            "mode_id",
            "model_id",
            "tool_prompt_id",
            "status",
        ):
            expected = getattr(query, field_name)
            if expected is not None:
                clauses.append(getattr(table.c, field_name) == expected)
        return clauses

    async def append_event(self, event: ToolUseEvent) -> None:
        """Append one event to SQLite."""

        await self._ensure_schema()

        def _append() -> None:
            with self._engine.begin() as connection:
                connection.execute(insert(self._table).values(**self._event_values(event)))

        await self._run_db(_append)

    async def query_events(self, query: ToolUseEventQuery) -> list[ToolUseEvent]:
        """Return events matching a bounded query, newest first."""

        await self._ensure_schema()
        clauses = self._query_clauses(query)

        def _query() -> list[ToolUseEvent]:
            statement = (
                select(self._table.c.payload_json)
                .where(*clauses)
                .order_by(
                    desc(self._table.c.created_at_epoch_us),
                    desc(self._table.c.event_id),
                )
                .limit(query.limit)
            )
            with self._engine.connect() as connection:
                rows = connection.execute(statement).all()
            return [ToolUseEvent.model_validate_json(row[0]) for row in rows]

        return await self._run_db(_query)

    async def delete_events_older_than(self, cutoff: datetime | int) -> int:
        """Delete events older than the cutoff and return the number removed."""

        await self._ensure_schema()
        cutoff_us = cutoff_epoch_us(cutoff)

        def _delete_old() -> int:
            statement = delete(self._table).where(self._table.c.created_at_epoch_us < cutoff_us)
            with self._engine.begin() as connection:
                result = connection.execute(statement)
            return int(result.rowcount or 0)

        return await self._run_db(_delete_old)

    async def delete_events_over_limit(self, max_events: int) -> int:
        """Keep the newest max_events and delete older rows."""

        await self._ensure_schema()
        max_events = max(0, int(max_events))

        def _delete_over_limit() -> int:
            if max_events == 0:
                statement = delete(self._table)
            else:
                keep_ids = (
                    select(self._table.c.event_id)
                    .order_by(
                        desc(self._table.c.created_at_epoch_us),
                        desc(self._table.c.event_id),
                    )
                    .limit(max_events)
                    .subquery()
                )
                statement = delete(self._table).where(self._table.c.event_id.not_in(select(keep_ids.c.event_id)))
            with self._engine.begin() as connection:
                result = connection.execute(statement)
            return int(result.rowcount or 0)

        return await self._run_db(_delete_over_limit)

    async def export_events(
        self,
        query: ToolUseEventQuery,
        *,
        format: ToolUseEventExportFormat,
    ) -> str:
        """Export events matching the query as JSON or JSON Lines."""

        rows = await self.query_events(query)
        if format == "jsonl":
            return "\n".join(event.model_dump_json() for event in rows)
        return "[" + ",".join(event.model_dump_json() for event in rows) + "]"

    def close(self) -> None:
        """Dispose SQLAlchemy engine resources."""

        self._engine.dispose()

    async def aclose(self) -> None:
        """Dispose SQLAlchemy engine resources off the event loop."""

        await self._run_db(self.close)
