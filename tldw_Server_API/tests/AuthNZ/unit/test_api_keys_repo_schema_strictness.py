import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings


pytestmark = pytest.mark.unit


class _StrictSqlitePool:
    def __init__(self, sqlite_fs_path: str = "strict.db"):
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
    assert seen_paths == ["strict.db"]


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


@pytest.mark.asyncio
async def test_fetch_active_by_hash_candidates_tolerates_legacy_sqlite_virtual_columns(tmp_path):
    db_path = tmp_path / "legacy-api-keys.db"
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{db_path}",
            SINGLE_USER_API_KEY="test-api-key",
        )
    )
    await pool.initialize()
    await pool.execute("DROP TABLE IF EXISTS api_keys")
    await pool.execute(
        """
        CREATE TABLE api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT NOT NULL,
            key_id TEXT,
            key_prefix TEXT,
            name TEXT,
            description TEXT,
            scope TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_used_at TIMESTAMP,
            last_used_ip TEXT,
            usage_count INTEGER DEFAULT 0,
            rate_limit INTEGER,
            allowed_ips TEXT,
            metadata TEXT
        )
        """
    )
    await pool.execute(
        """
        INSERT INTO api_keys (
            user_id, key_hash, key_id, key_prefix, name, scope, status, usage_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (42, "legacy-hash", "key-id", "prefix", "Legacy", "read", "active", 3),
    )

    try:
        row = await AuthnzApiKeysRepo(pool).fetch_active_by_hash_candidates(["legacy-hash"])
    finally:
        await pool.close()

    assert row is not None
    assert row["id"] == 1
    assert row["user_id"] == 42
    assert row["is_virtual"] == 0
    assert row["parent_key_id"] is None
    assert row["llm_allowed_endpoints"] is None
