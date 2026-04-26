"""ACP Session persistence and admin query service.

Stores session metadata with SQLite-backed persistence via ACPSessionsDB,
enabling session listing, usage tracking, and admin visibility.

The public API (SessionRecord, SessionTokenUsage, async methods) is preserved
while all state is delegated to the SQLite backend.
"""
from __future__ import annotations

import asyncio
import copy
import fnmatch
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Usage.pricing_catalog import compute_token_cost


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionTokenUsage:
    """Accumulated token usage for a session."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class SessionRecord:
    """Persistent metadata for an ACP session."""
    session_id: str
    user_id: int
    agent_type: str = "custom"
    name: str = ""
    status: str = "active"  # active | closed | error
    cwd: str = ""
    created_at: str = ""
    last_activity_at: str | None = None
    message_count: int = 0
    usage: SessionTokenUsage = field(default_factory=SessionTokenUsage)
    tags: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None
    policy_snapshot_version: str | None = None
    policy_snapshot_fingerprint: str | None = None
    policy_snapshot_refreshed_at: str | None = None
    policy_summary: dict[str, Any] | None = None
    policy_provenance_summary: dict[str, Any] | None = None
    policy_refresh_error: str | None = None
    bootstrap_ready: bool = True
    needs_bootstrap: bool = False
    # Forking lineage
    forked_from: str | None = None
    ancestry_chain: list[str] = field(default_factory=list)
    # Model used for cost estimation
    model: str | None = None
    # Token budget fields
    token_budget: int | None = None
    auto_terminate_at_budget: bool = False
    budget_exhausted: bool = False

    def to_info_dict(self, *, has_websocket: bool = False) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "message_count": self.message_count,
            "usage": self.usage.to_dict(),
            "tags": list(self.tags),
            "has_websocket": has_websocket,
            "persona_id": self.persona_id,
            "workspace_id": self.workspace_id,
            "workspace_group_id": self.workspace_group_id,
            "scope_snapshot_id": self.scope_snapshot_id,
            "policy_snapshot_version": self.policy_snapshot_version,
            "policy_snapshot_fingerprint": self.policy_snapshot_fingerprint,
            "policy_snapshot_refreshed_at": self.policy_snapshot_refreshed_at,
            "policy_summary": self.policy_summary,
            "policy_provenance_summary": self.policy_provenance_summary,
            "policy_refresh_error": self.policy_refresh_error,
            "forked_from": self.forked_from,
            "ancestry_chain": list(self.ancestry_chain),
            "model": self.model,
            "estimated_cost_usd": compute_token_cost(
                model=self.model,
                prompt_tokens=self.usage.prompt_tokens,
                completion_tokens=self.usage.completion_tokens,
            ),
            "token_budget": self.token_budget,
            "auto_terminate_at_budget": self.auto_terminate_at_budget,
            "budget_exhausted": self.budget_exhausted,
            "budget_remaining": (
                max(0, self.token_budget - self.usage.total_tokens)
                if self.token_budget is not None
                else None
            ),
        }

    def to_detail_dict(
        self,
        *,
        has_websocket: bool = False,
        fork_lineage: list[str] | None = None,
    ) -> dict[str, Any]:
        d = self.to_info_dict(has_websocket=has_websocket)
        d["messages"] = list(self.messages)
        d["cwd"] = self.cwd
        d["fork_lineage"] = fork_lineage or []
        return d


# Re-use the canonical implementation from the DB layer
_normalize_text_content = ACPSessionsDB._normalize_text_content


def _normalize_prompt_messages(prompt: list[dict[str, Any]], timestamp: str) -> tuple[list[dict[str, Any]], bool]:
    entries: list[dict[str, Any]] = []
    bootstrap_ready = True
    for item in prompt:
        role = "user"
        if isinstance(item, dict):
            role = str(item.get("role") or "user")
            content = _normalize_text_content(item.get("content"))
            entry: dict[str, Any] = {"role": role, "content": content, "timestamp": timestamp}
            entry["raw_prompt"] = copy.deepcopy(item)
        else:
            content = None
            entry = {"role": role, "content": None, "timestamp": timestamp, "raw_prompt": copy.deepcopy(item)}
        if content is None:
            bootstrap_ready = False
        entries.append(entry)
    return entries, bootstrap_ready


def _normalize_result_message(result: dict[str, Any], timestamp: str) -> tuple[dict[str, Any], bool]:
    content = _normalize_text_content(result)
    entry: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "timestamp": timestamp,
        "raw_result": copy.deepcopy(result),
    }
    return entry, content is not None


def _messages_bootstrap_ready(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if not role or not isinstance(content, str) or not content.strip():
            return False
    return True


# ---------------------------------------------------------------------------
# Permission Policy store
# ---------------------------------------------------------------------------

@dataclass
class PermissionPolicyRule:
    tool_pattern: str
    tier: str  # auto | batch | individual


@dataclass
class PermissionPolicy:
    id: int
    name: str
    description: str = ""
    rules: list[PermissionPolicyRule] = field(default_factory=list)
    org_id: int | None = None
    team_id: int | None = None
    priority: int = 0
    created_at: str = ""
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Agent Config store
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    id: int
    type: str
    name: str
    description: str = ""
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    denied_tools: list[str] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_api_key: str | None = None
    org_id: int | None = None
    team_id: int | None = None
    enabled: bool = True
    created_at: str = ""
    updated_at: str | None = None
    default_token_budget: int | None = None
    default_auto_terminate_at_budget: bool = True
    max_token_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        import os
        is_configured = True
        if self.requires_api_key:
            is_configured = bool(os.getenv(self.requires_api_key, ""))
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "parameters": dict(self.parameters),
            "requires_api_key": self.requires_api_key,
            "org_id": self.org_id,
            "team_id": self.team_id,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_configured": is_configured,
            "default_token_budget": self.default_token_budget,
            "default_auto_terminate_at_budget": self.default_auto_terminate_at_budget,
            "max_token_budget": self.max_token_budget,
        }


# ---------------------------------------------------------------------------
# Singleton session store
# ---------------------------------------------------------------------------

class ACPSessionStore:
    """SQLite-backed session metadata store with query capabilities.

    Delegates all persistent state to an ``ACPSessionsDB`` instance while
    preserving the original async public API.
    """

    def __init__(self, db: ACPSessionsDB | None = None) -> None:
        if db is None:
            self._db = ACPSessionsDB()
        else:
            self._db = db
        self._lock = asyncio.Lock()
        # Agent configs — keyed by id (in-memory for now)
        self._agent_configs: dict[int, AgentConfig] = {}
        self._agent_config_seq = 0
        # Session TTL cleanup task
        self._cleanup_task: asyncio.Task | None = None
        # Quotas (loaded from config on first use)
        self._session_ttl_seconds: int = 86400
        self._max_concurrent_per_user: int = 5
        self._max_tokens_per_session: int = 1_000_000
        self._max_session_duration_seconds: int = 14400

    def get_db(self) -> ACPSessionsDB:
        """Return the underlying database instance."""
        return self._db

    # ------------------------------------------------------------------
    # Internal helpers — convert DB dicts to public dataclasses
    # ------------------------------------------------------------------

    def _db_messages_to_record_messages(self, db_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert DB message rows to the SessionRecord.messages format.

        DB stores a single ``raw_data`` column; the public API expects
        ``raw_prompt`` for user messages and ``raw_result`` for assistant
        messages.
        """
        result: list[dict[str, Any]] = []
        for msg in db_messages:
            content = msg.get("content")
            # The DB stores empty string for non-normalizable content;
            # the public API expects None in that case.
            if isinstance(content, str) and not content.strip():
                content = None
            entry: dict[str, Any] = {
                "role": msg["role"],
                "content": content,
                "timestamp": msg.get("timestamp", ""),
            }
            raw = msg.get("raw_data")
            if msg["role"] == "assistant":
                entry["raw_result"] = raw
            else:
                entry["raw_prompt"] = raw
            result.append(entry)
        return result

    def _dict_to_record(
        self,
        d: dict[str, Any],
        messages: list[dict[str, Any]] | None = None,
    ) -> SessionRecord:
        """Convert a DB session dict to a ``SessionRecord`` dataclass."""
        usage = SessionTokenUsage(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )
        return SessionRecord(
            session_id=d["session_id"],
            user_id=d["user_id"],
            agent_type=d.get("agent_type", "custom"),
            name=d.get("name", ""),
            status=d.get("status", "active"),
            cwd=d.get("cwd", ""),
            created_at=d.get("created_at", ""),
            last_activity_at=d.get("last_activity_at"),
            message_count=d.get("message_count", 0),
            usage=usage,
            tags=d.get("tags", []),
            messages=messages or [],
            mcp_servers=d.get("mcp_servers", []),
            persona_id=d.get("persona_id"),
            workspace_id=d.get("workspace_id"),
            workspace_group_id=d.get("workspace_group_id"),
            scope_snapshot_id=d.get("scope_snapshot_id"),
            policy_snapshot_version=d.get("policy_snapshot_version"),
            policy_snapshot_fingerprint=d.get("policy_snapshot_fingerprint"),
            policy_snapshot_refreshed_at=d.get("policy_snapshot_refreshed_at"),
            policy_summary=d.get("policy_summary"),
            policy_provenance_summary=d.get("policy_provenance_summary"),
            policy_refresh_error=d.get("policy_refresh_error"),
            bootstrap_ready=d.get("bootstrap_ready", True),
            needs_bootstrap=d.get("needs_bootstrap", False),
            forked_from=d.get("forked_from"),
            ancestry_chain=d.get("ancestry_chain_json") or [],
            model=d.get("model"),
            token_budget=d.get("token_budget"),
            auto_terminate_at_budget=d.get("auto_terminate_at_budget", False),
            budget_exhausted=d.get("budget_exhausted", False),
        )

    # ------------------------------------------------------------------
    # Quota configuration
    # ------------------------------------------------------------------

    def configure_quotas(
        self,
        session_ttl_seconds: int = 86400,
        max_concurrent_per_user: int = 5,
        max_tokens_per_session: int = 1_000_000,
        max_session_duration_seconds: int = 14400,
    ) -> None:
        """Set quota limits from config."""
        self._session_ttl_seconds = session_ttl_seconds
        self._max_concurrent_per_user = max_concurrent_per_user
        self._max_tokens_per_session = max_tokens_per_session
        self._max_session_duration_seconds = max_session_duration_seconds
        self._db.configure_quotas(
            max_concurrent_per_user=max_concurrent_per_user,
            max_tokens_per_session=max_tokens_per_session,
            session_ttl_seconds=session_ttl_seconds,
            max_session_duration_seconds=max_session_duration_seconds,
        )

    def start_cleanup_task(self) -> None:
        """Start background task to evict expired sessions."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.ensure_future(self._cleanup_loop())

    def stop_cleanup_task(self) -> None:
        """Cancel the cleanup background task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodically evict expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._evict_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("ACP session cleanup error: {}", exc)
                await asyncio.sleep(60)

    async def _evict_expired_sessions(self) -> int:
        """Evict sessions past TTL or max duration. Returns count evicted."""
        return self._db.evict_expired_sessions()

    async def check_session_quota(self, user_id: int) -> dict[str, Any] | None:
        """Check if user can create a new session. Returns None if ok, or error dict."""
        return self._db.check_session_quota(user_id)

    async def check_token_quota(self, session_id: str) -> dict[str, Any] | None:
        """Check if a session has exceeded its token quota. Returns None if ok."""
        return self._db.check_token_quota(session_id)

    async def get_quota_status(self, user_id: int, session_id: str | None = None) -> dict[str, Any]:
        """Get current quota usage for a user/session."""
        result = self._db.get_quota_status(user_id, session_id=session_id)
        # Add TTL fields that the DB layer doesn't include
        result["session_ttl_seconds"] = self._session_ttl_seconds
        result["max_session_duration_seconds"] = self._max_session_duration_seconds
        return result

    # -- Session CRUD -------------------------------------------------------

    async def register_session(
        self,
        session_id: str,
        user_id: int,
        agent_type: str = "custom",
        name: str = "",
        cwd: str = "",
        tags: list[str] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        policy_snapshot_version: str | None = None,
        policy_snapshot_fingerprint: str | None = None,
        policy_snapshot_refreshed_at: str | None = None,
        policy_summary: dict[str, Any] | None = None,
        policy_provenance_summary: dict[str, Any] | None = None,
        policy_refresh_error: str | None = None,
        forked_from: str | None = None,
        model: str | None = None,
        token_budget: int | None = None,
        auto_terminate_at_budget: bool = False,
    ) -> SessionRecord:
        d = self._db.register_session(
            session_id=session_id,
            user_id=user_id,
            agent_type=agent_type,
            name=name,
            cwd=cwd,
            tags=tags,
            mcp_servers=mcp_servers,
            persona_id=persona_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
            scope_snapshot_id=scope_snapshot_id,
            policy_snapshot_version=policy_snapshot_version,
            policy_snapshot_fingerprint=policy_snapshot_fingerprint,
            policy_snapshot_refreshed_at=policy_snapshot_refreshed_at,
            policy_summary=policy_summary,
            policy_provenance_summary=policy_provenance_summary,
            policy_refresh_error=policy_refresh_error,
            forked_from=forked_from,
            model=model,
            token_budget=token_budget,
            auto_terminate_at_budget=auto_terminate_at_budget,
        )
        logger.debug("Registered ACP session {} for user {}", session_id, user_id)
        return self._dict_to_record(d)

    async def update_session_budget(
        self,
        session_id: str,
        token_budget: int | None,
        auto_terminate_at_budget: bool,
    ) -> SessionRecord | None:
        """Update the token budget for a session. Returns updated record or None."""
        updated = self._db.update_session_budget(
            session_id, token_budget, auto_terminate_at_budget,
        )
        if not updated:
            return None
        return await self.get_session(session_id)

    async def check_and_enforce_budget(self, session_id: str) -> bool:
        """Check if session has exceeded its token budget.

        Returns True if the session was terminated due to budget exhaustion.
        """
        return self._db.check_budget_and_terminate(session_id)

    async def update_policy_snapshot_state(
        self,
        session_id: str,
        *,
        policy_snapshot_version: str | None,
        policy_snapshot_fingerprint: str | None,
        policy_snapshot_refreshed_at: str | None,
        policy_summary: dict[str, Any] | None,
        policy_provenance_summary: dict[str, Any] | None,
        policy_refresh_error: str | None,
    ) -> SessionRecord | None:
        self._db.update_policy_snapshot_state(
            session_id,
            policy_snapshot_version=policy_snapshot_version,
            policy_snapshot_fingerprint=policy_snapshot_fingerprint,
            policy_snapshot_refreshed_at=policy_snapshot_refreshed_at,
            policy_summary=policy_summary,
            policy_provenance_summary=policy_provenance_summary,
            policy_refresh_error=policy_refresh_error,
        )
        return await self.get_session(session_id)

    async def close_session(self, session_id: str) -> None:
        self._db.close_session(session_id)

    async def record_prompt(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> SessionTokenUsage | None:
        """Record a prompt+response exchange and accumulate token usage."""
        async with self._lock:
            # 1. Persist to DB
            usage_dict = self._db.record_prompt(session_id, prompt, result)
            if usage_dict is None:
                return None

            # 2. Handle bootstrap_ready tracking (DB doesn't track this)
            now = datetime.now(timezone.utc).isoformat()
            _, prompt_bootstrap_ready = _normalize_prompt_messages(prompt, now)
            _, result_bootstrap_ready = _normalize_result_message(result, now)

            if not (prompt_bootstrap_ready and result_bootstrap_ready):
                self._db.set_bootstrap_ready(session_id, False)

            return SessionTokenUsage(
                prompt_tokens=usage_dict["prompt_tokens"],
                completion_tokens=usage_dict["completion_tokens"],
                total_tokens=usage_dict["total_tokens"],
            )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        d = self._db.get_session(session_id)
        if d is None:
            return None
        db_messages = self._db.get_messages(session_id)
        messages = self._db_messages_to_record_messages(db_messages)
        return self._dict_to_record(d, messages)

    async def get_fork_lineage(self, session_id: str, *, max_depth: int = 50) -> list[str]:
        """Walk the forked_from chain and return ancestor session IDs (oldest first)."""
        return self._db.get_fork_lineage(session_id, max_depth=max_depth)

    async def build_bootstrap_prompt(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        async with self._lock:
            d = self._db.get_session(session_id)
            if d is None or not d.get("needs_bootstrap"):
                return copy.deepcopy(prompt), False

            db_messages = self._db.get_messages(session_id)
            messages = self._db_messages_to_record_messages(db_messages)

            if not _messages_bootstrap_ready(messages):
                self._db.set_bootstrap_ready(session_id, False)
                raise ValueError("fork_not_resumable")

            bootstrap_prompt = [
                {"role": str(msg["role"]), "content": str(msg["content"])}
                for msg in messages
            ]
            return bootstrap_prompt + copy.deepcopy(prompt), True

    async def clear_bootstrap(self, session_id: str) -> None:
        self._db.clear_needs_bootstrap(session_id)

    async def list_sessions(
        self,
        *,
        user_id: int | None = None,
        status: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[SessionRecord], int]:
        """List sessions with optional filters. Returns (records, total_count)."""
        rows, total = self._db.list_sessions(
            user_id=user_id, status=status, agent_type=agent_type,
            limit=limit, offset=offset,
        )
        records = [self._dict_to_record(d) for d in rows]
        return records, total

    async def get_agent_usage_stats(self, *, range_days: int = 7) -> list[dict[str, Any]]:
        """Return per-agent aggregated token usage for the last *range_days* days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=range_days)).isoformat()
        return await asyncio.to_thread(self._db.get_agent_usage_stats, since_iso=cutoff)

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        message_index: int,
        user_id: int,
        name: str | None = None,
    ) -> SessionRecord | None:
        """Fork a session, copying messages up to message_index."""
        async with self._lock:
            d = self._db.fork_session(
                source_session_id, new_session_id, message_index, user_id, name,
            )
            if d is None:
                return None

            db_messages = self._db.get_messages(new_session_id)
            messages = self._db_messages_to_record_messages(db_messages)

            # Check bootstrap readiness based on message content
            if not _messages_bootstrap_ready(messages):
                self._db.set_bootstrap_ready(new_session_id, False)
                d["bootstrap_ready"] = False

            return self._dict_to_record(d, messages)

    # -- Aggregation --------------------------------------------------------

    async def get_agent_metrics(self) -> list[dict[str, Any]]:
        """Aggregate session metrics per agent type.

        Delegates to the SQLite backend for an efficient GROUP BY query,
        then enriches each entry with ``total_estimated_cost_usd`` computed
        from per-session model and token counts via the pricing catalog.
        """
        metrics = self._db.aggregate_metrics_by_agent()

        # Compute per-session costs and aggregate by agent_type
        cost_by_agent: dict[str, float] = {}
        try:
            for row in self._db.get_session_cost_data():
                cost = compute_token_cost(
                    model=row.get("model"),
                    prompt_tokens=row.get("prompt_tokens", 0) or 0,
                    completion_tokens=row.get("completion_tokens", 0) or 0,
                )
                if cost is not None:
                    agent = row.get("agent_type", "custom")
                    cost_by_agent[agent] = cost_by_agent.get(agent, 0.0) + cost
        except Exception as exc:
            logger.warning("Failed to compute agent cost metrics: {}", exc)

        for m in metrics:
            agent = m["agent_type"]
            total_cost = cost_by_agent.get(agent)
            m["total_estimated_cost_usd"] = round(total_cost, 6) if total_cost else None

        return metrics

    # -- Run history queries ------------------------------------------------

    async def list_runs(
        self,
        *,
        user_id: int | None = None,
        status: str | None = None,
        agent_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query session records with filters. Returns paged results."""
        rows, total = self._db.list_runs(
            user_id=user_id,
            status=status,
            agent_type=agent_type,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            offset=offset,
        )
        records = [self._dict_to_record(d) for d in rows]
        items = [rec.to_info_dict() for rec in records]
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def aggregate_runs(
        self,
        *,
        user_id: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate token usage and costs across sessions."""
        agg = self._db.aggregate_runs(
            user_id=user_id,
            from_date=from_date,
            to_date=to_date,
        )

        # Compute estimated total cost from per-session data using Decimal
        # to avoid floating-point accumulation errors.
        from decimal import Decimal

        estimated_cost: float | None = None
        cost_rows = agg.pop("_cost_rows", [])
        try:
            total_cost = Decimal("0")
            has_any_cost = False
            for row in cost_rows:
                cost = compute_token_cost(
                    model=row.get("model"),
                    prompt_tokens=row.get("prompt_tokens", 0) or 0,
                    completion_tokens=row.get("completion_tokens", 0) or 0,
                )
                if cost is not None:
                    total_cost += Decimal(str(cost))
                    has_any_cost = True
            if has_any_cost:
                estimated_cost = float(total_cost.quantize(Decimal("0.000001")))
        except Exception as exc:
            logger.warning("Failed to compute aggregate cost: {}", exc)

        agg["estimated_cost_usd"] = estimated_cost
        return agg

    # -- Agent Config CRUD --------------------------------------------------

    async def create_agent_config(self, data: dict[str, Any]) -> AgentConfig:
        async with self._lock:
            self._agent_config_seq += 1
            now = datetime.now(timezone.utc).isoformat()
            config = AgentConfig(
                id=self._agent_config_seq,
                type=data["type"],
                name=data["name"],
                description=data.get("description", ""),
                system_prompt=data.get("system_prompt"),
                allowed_tools=data.get("allowed_tools"),
                denied_tools=data.get("denied_tools"),
                parameters=data.get("parameters", {}),
                requires_api_key=data.get("requires_api_key"),
                org_id=data.get("org_id"),
                team_id=data.get("team_id"),
                enabled=data.get("enabled", True),
                max_token_budget=data.get("max_token_budget"),
                created_at=now,
                default_token_budget=data.get("default_token_budget"),
                default_auto_terminate_at_budget=data.get("default_auto_terminate_at_budget", True),
            )
            self._agent_configs[config.id] = config
        return config

    async def update_agent_config(self, config_id: int, data: dict[str, Any]) -> AgentConfig | None:
        async with self._lock:
            config = self._agent_configs.get(config_id)
            if not config:
                return None
            for key in ("name", "description", "system_prompt", "allowed_tools",
                        "denied_tools", "parameters", "requires_api_key",
                        "org_id", "team_id", "enabled", "type",
                        "default_token_budget", "default_auto_terminate_at_budget",
                        "max_token_budget"):
                if key in data:
                    setattr(config, key, data[key])
            config.updated_at = datetime.now(timezone.utc).isoformat()
        return config

    async def delete_agent_config(self, config_id: int) -> bool:
        async with self._lock:
            return self._agent_configs.pop(config_id, None) is not None

    async def get_agent_config(self, config_id: int) -> AgentConfig | None:
        return self._agent_configs.get(config_id)

    async def list_agent_configs(
        self,
        *,
        org_id: int | None = None,
        team_id: int | None = None,
        enabled_only: bool = False,
    ) -> list[AgentConfig]:
        results = []
        for cfg in self._agent_configs.values():
            if enabled_only and not cfg.enabled:
                continue
            if org_id is not None and cfg.org_id is not None and cfg.org_id != org_id:
                continue
            if team_id is not None and cfg.team_id is not None and cfg.team_id != team_id:
                continue
            results.append(cfg)
        return results

    # -- Permission Policy CRUD --------------------------------------------

    async def create_permission_policy(self, data: dict[str, Any]) -> PermissionPolicy:
        import json as _json
        rules_raw = data.get("rules", [])
        rules_json = _json.dumps(rules_raw)
        policy_id = self._db.create_permission_policy(
            name=data["name"],
            rules_json=rules_json,
            priority=data.get("priority", 0),
            description=data.get("description", ""),
            org_id=data.get("org_id"),
            team_id=data.get("team_id"),
        )
        row = self._db.get_permission_policy(policy_id)
        return self._db_row_to_permission_policy(row)  # type: ignore[arg-type]

    async def update_permission_policy(self, policy_id: int, data: dict[str, Any]) -> PermissionPolicy | None:
        import json as _json
        kwargs: dict[str, Any] = {}
        for key in ("name", "description", "org_id", "team_id", "priority"):
            if key in data:
                kwargs[key] = data[key]
        if "rules" in data:
            kwargs["rules_json"] = _json.dumps(data["rules"])
        if not kwargs:
            row = self._db.get_permission_policy(policy_id)
            if row is None:
                return None
            return self._db_row_to_permission_policy(row)
        updated = self._db.update_permission_policy(policy_id, **kwargs)
        if not updated:
            return None
        row = self._db.get_permission_policy(policy_id)
        if row is None:
            return None
        return self._db_row_to_permission_policy(row)

    async def delete_permission_policy(self, policy_id: int) -> bool:
        return self._db.delete_permission_policy(policy_id)

    async def list_permission_policies(
        self,
        *,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> list[PermissionPolicy]:
        rows = self._db.list_permission_policies()
        results: list[PermissionPolicy] = []
        for row in rows:
            pol = self._db_row_to_permission_policy(row)
            if org_id is not None and pol.org_id is not None and pol.org_id != org_id:
                continue
            if team_id is not None and pol.team_id is not None and pol.team_id != team_id:
                continue
            results.append(pol)
        return results

    def resolve_permission_tier(self, tool_name: str) -> str | None:
        """Consult DB-backed policies to determine a permission tier for a tool.

        Returns None if no policy rule matches (caller should fall back to
        the default heuristic).
        """
        return self._db.resolve_permission_tier(tool_name)

    @staticmethod
    def _db_row_to_permission_policy(row: dict[str, Any]) -> PermissionPolicy:
        """Convert a DB dict row to a PermissionPolicy dataclass."""
        import json as _json
        raw = row.get("rules_json", "[]")
        try:
            rules_list = _json.loads(raw) if isinstance(raw, str) else raw
        except (_json.JSONDecodeError, TypeError):
            rules_list = []
        rules = [
            PermissionPolicyRule(tool_pattern=r.get("tool_pattern", ""), tier=r.get("tier", ""))
            for r in (rules_list or [])
        ]
        return PermissionPolicy(
            id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            rules=rules,
            org_id=row.get("org_id"),
            team_id=row.get("team_id"),
            priority=row.get("priority", 0),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at"),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: ACPSessionStore | None = None
_store_lock = asyncio.Lock()


async def get_acp_session_store() -> ACPSessionStore:
    global _store
    if _store is None:
        async with _store_lock:
            if _store is None:
                store = ACPSessionStore()
                # Initialize quotas from ACP config
                try:
                    from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config
                    cfg = load_acp_sandbox_config()
                    store.configure_quotas(
                        session_ttl_seconds=cfg.session_ttl_seconds,
                        max_concurrent_per_user=cfg.max_concurrent_sessions_per_user,
                        max_tokens_per_session=cfg.max_tokens_per_session,
                        max_session_duration_seconds=cfg.max_session_duration_seconds,
                    )
                except Exception as exc:
                    logger.warning("Failed to load ACP quota config, using defaults: {}", exc)

                # Wire the agent registry and health monitor with DB
                try:
                    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
                        get_agent_registry,
                        set_registry_db,
                    )
                    from tldw_Server_API.app.core.Agent_Client_Protocol.health_monitor import (
                        configure_health_monitor,
                    )

                    set_registry_db(store._db)
                    registry = get_agent_registry()
                    monitor = configure_health_monitor(registry=registry, db=store._db)
                    await monitor.start()
                except Exception as exc:
                    logger.warning("Failed to wire agent registry/health monitor: {}", exc)

                store.start_cleanup_task()
                _store = store
    return _store
