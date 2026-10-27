"""Verify guarded UsersDB initialization in a private native SQLite runtime."""

import os
from pathlib import Path

import pytest
import sqlglot

from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import (
    _run_runtime,
    _runtime_env,
)

pytestmark = pytest.mark.integration


def test_users_db_initializes_after_single_user_bootstrap(tmp_path: Path) -> None:
    """Repeated guarded initialization preserves the bootstrapped admin.

    Args:
        tmp_path (Path): Private runtime directory provided by pytest.
    Returns:
        None: Assertions verify native SQLite bootstrap and admin preservation.
    """
    env = _runtime_env(tmp_path, f"sqlite:///{tmp_path / 'auth.db'}", backend="sqlite")
    # Match the parser under test even when it is isolated from the root venv.
    env["PYTHONPATH"] = os.pathsep.join((str(Path(sqlglot.__file__).resolve().parent.parent), env["PYTHONPATH"]))
    result = _run_runtime(
        tmp_path,
        env,
        """
import asyncio
import json
import sqlglot
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

async def main():
    result = {'setup': await setup_database(), 'sqlglot': sqlglot.__version__}
    try:
        result['bootstrap'] = await bootstrap_single_user_profile()
        pool = await get_db_pool()
        result['backend'] = pool.backend_type
        users = UsersDB(pool)
        await users.initialize()
        await UsersDB(pool).initialize()
        result['repeated_initialize'] = True
        user = await users.get_user_by_id(get_settings().SINGLE_USER_FIXED_ID)
        result['admin_preserved'] = bool(user and user['role'] == 'admin'
            and user['is_active'] and user['is_verified'])
        result['repeated_bootstrap'] = await bootstrap_single_user_profile()
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
""",
    )
    assert result == {
        "setup": True,
        "sqlglot": sqlglot.__version__,
        "bootstrap": True,
        "backend": "sqlite",
        "repeated_initialize": True,
        "repeated_bootstrap": True,
        "admin_preserved": True,
    }
