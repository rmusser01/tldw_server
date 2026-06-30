from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
    AuthnzLLMProviderOverridesRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import AuthnzRateLimitsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)

pytestmark = pytest.mark.unit

_LEAK = "authnz backend exploded at /tmp/authnz-secret-token"


def _capture_logs() -> tuple[list[str], int]:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    return records, sink_id


def _assert_safe_log(rendered: str) -> None:
    assert "authnz backend exploded" not in rendered
    assert "/tmp/authnz-secret-token" not in rendered
    assert "exc_info" not in rendered


class _DictFallbackRow:
    def __init__(self) -> None:
        self._keys_calls = 0

    def keys(self):
        self._keys_calls += 1
        if self._keys_calls == 1:
            raise RuntimeError(_LEAK)
        return ["provider", "key_hint"]

    def __getitem__(self, key: str) -> Any:
        return {"provider": "openai", "key_hint": "1234"}[key]

    def __iter__(self):
        return iter((("provider", "openai"), ("key_hint", "1234")))


class _FlakyMappingRow:
    def __init__(self) -> None:
        self._keys_calls = 0

    def keys(self):
        self._keys_calls += 1
        if self._keys_calls == 1:
            raise RuntimeError(_LEAK)
        return ["provider", "is_enabled"]

    def __getitem__(self, key: str) -> Any:
        return {"provider": "openai", "is_enabled": True}[key]


def test_user_provider_secret_row_key_fallback_log_omits_raw_exception() -> None:
    records, sink_id = _capture_logs()
    try:
        row = AuthnzUserProviderSecretsRepo._row_to_dict(_DictFallbackRow())
    finally:
        logger.remove(sink_id)

    assert row == {"provider": "openai", "key_hint": "1234"}
    _assert_safe_log("\n".join(records))


def test_llm_provider_override_row_cast_fallback_log_omits_raw_exception() -> None:
    records, sink_id = _capture_logs()
    try:
        row = AuthnzLLMProviderOverridesRepo._row_to_dict(_FlakyMappingRow())
    finally:
        logger.remove(sink_id)

    assert row == {"provider": "openai", "is_enabled": True}
    _assert_safe_log("\n".join(records))


class _Tx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _FetchAllCursor:
    async def fetchall(self) -> list[Any]:
        return []


class _CommitFailConn:
    async def execute(self, query: str, *params: Any) -> _FetchAllCursor:  # noqa: ARG002
        return _FetchAllCursor()

    async def commit(self) -> None:
        raise RuntimeError(_LEAK)


class _RateLimitPool:
    pool = None

    def transaction(self) -> _Tx:
        return _Tx(_CommitFailConn())


@pytest.mark.asyncio
async def test_rate_limits_explicit_commit_fallback_log_omits_raw_exception() -> None:
    repo = AuthnzRateLimitsRepo(db_pool=_RateLimitPool())  # type: ignore[arg-type]

    records, sink_id = _capture_logs()
    try:
        await repo.ensure_schema()
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))
