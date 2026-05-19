import pytest

from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo


pytestmark = pytest.mark.unit


class _StrictSqlitePool:
    def __init__(self, sqlite_fs_path: str = "/tmp/strict.db"):
        self.pool = None
        self._sqlite_fs_path = sqlite_fs_path

    async def fetchone(self, query: str, *args):  # noqa: ANN001, ANN002
        lowered = query.lower()
        if "name='api_keys'" in lowered or "name = 'api_keys'" in lowered:
            return {"name": "api_keys"}
        if "name='api_key_audit_log'" in lowered or "name = 'api_key_audit_log'" in lowered:
            return {"name": "api_key_audit_log"}
        return None


@pytest.mark.asyncio
async def test_api_keys_repo_ensure_tables_calls_shared_validator_in_strict_mode(monkeypatch):
    pool = _StrictSqlitePool()
    repo = AuthnzApiKeysRepo(pool)
    seen_paths: list[str] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo.should_enforce_sqlite_schema_strictness",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo.validate_required_sqlite_api_key_schema",
        lambda path: seen_paths.append(path) or (_ for _ in ()).throw(RuntimeError("scope default")),
    )

    with pytest.raises(RuntimeError, match="scope default"):
        await repo.ensure_tables()
    assert seen_paths == ["/tmp/strict.db"]


@pytest.mark.asyncio
async def test_api_keys_repo_ensure_tables_skips_shared_validator_when_gate_is_off(monkeypatch):
    pool = _StrictSqlitePool()
    repo = AuthnzApiKeysRepo(pool)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo.should_enforce_sqlite_schema_strictness",
        lambda _path: False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo.validate_required_sqlite_api_key_schema",
        lambda _path: (_ for _ in ()).throw(AssertionError("validator should not run")),
    )

    await repo.ensure_tables()
