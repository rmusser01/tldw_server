"""Disposable cold initializer/auth route control; never a listening server."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


async def main():
    from tldw_Server_API.app.core.testing import is_test_mode

    assert not is_test_mode()
    assert not any(key.startswith(("PYTEST_", "TLDW_TEST_")) for key in os.environ)
    from tldw_Server_API.app.core.AuthNZ import initialize

    await initialize.main(non_interactive=True)

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from tldw_Server_API.app.api.v1.endpoints import auth, users
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db

    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(users.router, prefix="/api/v1")
    assert not app.dependency_overrides
    statuses = {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://disposable.test") as client:
        if os.environ["AUTH_MODE"] == "single_user":
            headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
        else:
            user_db = await get_users_db()
            user = await user_db.create_user(
                "matrix_fixture",
                "matrix-fixture@example.invalid",
                PasswordService().hash_password(os.environ["MATRIX_ACCOUNT_PASSWORD"]),
                is_verified=True,
            )
            repo = await AuthnzUsersRepo.from_pool()
            await repo.assign_role_if_missing(user_id=user["id"], role_name="user")
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "matrix_fixture", "password": os.environ["MATRIX_ACCOUNT_PASSWORD"]},
            )
            statuses["login"] = response.status_code
            assert response.status_code == 200, f"login status {response.status_code}"
            headers = {"Authorization": "Bearer " + response.json()["access_token"]}
        response = await client.get("/api/v1/users/me/profile", headers=headers)
        statuses["profile"] = response.status_code
        assert response.status_code == 200, f"profile status {response.status_code}"

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation

    with chacha_operation(independent=True):
        db = await get_chacha_db_for_user_id(1)
        expected = urlsplit(os.environ["TLDW_CONTENT_PG_DSN"])
        identity = db.execute_query(
            "SELECT current_database() AS database, current_user AS role", read_only=True
        ).fetchone()
        assert identity["database"] == expected.path.lstrip("/")
        assert identity["role"] == expected.username
        note_id = db.add_note(
            title="Disposable role control", content="Restricted direct login can initialize content."
        )
        assert db.get_note_by_id(note_id)["title"] == "Disposable role control"
        db.close_connection()
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import shutdown_chacha_resources

    await shutdown_chacha_resources()
    await shutdown_all_audit_services()
    pool = await get_db_pool()
    await pool.close()
    Path(os.environ["MATRIX_COLD_RESULT"]).write_text(
        json.dumps(
            {
                "normal_initializer_returned": True,
                "test_mode": False,
                "dependency_overrides": 0,
                "mode": os.environ["AUTH_MODE"],
                "statuses": statuses,
                "content_roundtrip": True,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
