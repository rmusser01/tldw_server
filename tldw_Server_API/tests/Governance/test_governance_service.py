from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Governance.service import GovernanceService
from tldw_Server_API.app.core.Governance.store import GovernanceStore

pytestmark = pytest.mark.unit


@dataclass
class _FakeGap:
    id: int
    question: str
    category: str
    status: str = "open"


class _FakeStore:
    def __init__(self) -> None:
        self.last_upsert: dict[str, Any] | None = None

    async def upsert_open_gap(self, **kwargs: Any) -> _FakeGap:
        self.last_upsert = kwargs
        return _FakeGap(id=1, question=str(kwargs["question"]), category=str(kwargs["category"]))


class _FakePolicyLoader:
    def __init__(
        self,
        fallback_mode: str = "warn_only",
        *,
        should_fail: bool = True,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        self.fallback_mode = fallback_mode
        self.should_fail = should_fail
        self.candidates = candidates or []

    async def get_candidates(self, **_: Any) -> list[dict[str, Any]]:
        if self.should_fail:
            raise RuntimeError("backend unavailable")
        return list(self.candidates)


@pytest.mark.asyncio
async def test_validate_change_uses_shared_fallback_mode():
    svc = GovernanceService(
        store=_FakeStore(),
        policy_loader=_FakePolicyLoader("warn_only", should_fail=True),
    )

    out = await svc.validate_change(
        surface="mcp_tool",
        summary="Allow tool to update dependency versions",
        category="dependencies",
    )

    assert out.status in {"warn", "allow"}
    assert out.fallback_reason == "backend_unavailable"


@pytest.mark.asyncio
async def test_query_knowledge_returns_category_source():
    svc = GovernanceService(store=_FakeStore(), policy_loader=_FakePolicyLoader(should_fail=False))

    out = await svc.query_knowledge(query="auth rules", category="security")

    assert out.category_source in {"explicit", "metadata", "pattern", "default"}


@pytest.mark.asyncio
async def test_resolve_gap_uses_metadata_category_when_missing_explicit():
    store = _FakeStore()
    svc = GovernanceService(store=store, policy_loader=_FakePolicyLoader(should_fail=False))

    gap = await svc.resolve_gap(
        question="Should we require MFA for admins?",
        category=None,
        metadata={"category": "security"},
        org_id=7,
    )

    assert gap.status == "open"
    assert store.last_upsert is not None
    assert store.last_upsert["category"] == "security"


@pytest.mark.asyncio
async def test_validate_change_uses_store_backed_rules_by_default(tmp_path):
    db_path = tmp_path / "governance_rules.db"
    store = GovernanceStore(sqlite_path=str(db_path))
    await store.ensure_schema()
    async with store._connect() as db:
        await db.execute(
            """
            INSERT INTO governance_rules (category, title, action, priority)
            VALUES (?, ?, ?, ?)
            """,
            ("security", "Deny security-sensitive changes", "deny", 5),
        )
        await db.commit()

    svc = GovernanceService(store=store)

    out = await svc.validate_change(
        surface="mcp_tool",
        summary="Rotate an admin auth token",
        category="security",
    )

    assert out.status == "deny"
    assert out.matched_rules == ("governance_rules:1",)


@pytest.mark.asyncio
async def test_validate_change_denies_invalid_candidate_action():
    svc = GovernanceService(
        store=_FakeStore(),
        policy_loader=_FakePolicyLoader(
            should_fail=False,
            candidates=[{"action": "bogus", "scope_level": 1, "source_id": "bad-rule"}],
        ),
    )

    out = await svc.validate_change(
        surface="mcp_tool",
        summary="Change package policy",
        category="dependencies",
    )

    assert out.status == "deny"
    assert out.fallback_reason == "invalid_candidate_action"


@pytest.mark.asyncio
async def test_mapping_candidate_preserves_updated_at_for_tie_breaker():
    now = datetime.now(timezone.utc)
    svc = GovernanceService(
        store=_FakeStore(),
        policy_loader=_FakePolicyLoader(
            should_fail=False,
            candidates=[
                {
                    "action": "warn",
                    "scope_level": 2,
                    "priority": 1,
                    "updated_at": (now - timedelta(minutes=1)).isoformat(),
                    "source_id": "newer",
                },
                {
                    "action": "warn",
                    "scope_level": 2,
                    "priority": 1,
                    "updated_at": (now - timedelta(minutes=5)).isoformat(),
                    "source_id": "older",
                },
            ],
        ),
    )

    out = await svc.validate_change(
        surface="mcp_tool",
        summary="Change audit policy",
        category="compliance",
    )

    assert out.matched_rules == ("newer", "older")


@pytest.mark.asyncio
async def test_metadata_category_none_is_ignored():
    svc = GovernanceService(store=_FakeStore(), policy_loader=_FakePolicyLoader(should_fail=False))

    out = await svc.query_knowledge(query="general guidance", metadata={"category": None})

    assert out.category == "general"
    assert out.category_source == "default"
