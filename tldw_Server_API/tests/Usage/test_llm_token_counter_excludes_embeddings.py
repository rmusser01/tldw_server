"""The monthly LLM token counter leaves embeddings out and keeps rows with no operation (spec §6)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_tables

pytestmark = pytest.mark.unit


async def test_embeddings_are_excluded_and_null_operations_count(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """chat 100 + NULL-operation 10 are counted; embeddings 1000 and another user's 5 are not."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.sqlite'}")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    for user_id, operation, tokens in ((1, "chat", 100), (1, None, 10), (1, "embeddings", 1000), (2, "chat", 5)):
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, operation, total_tokens) VALUES (?, ?, ?)",
            user_id, operation, tokens,
        )

    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await AuthnzUsageRepo(pool).sum_user_llm_tokens_since(user_id=1, since=since) == 110.0
