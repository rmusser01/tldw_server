"""
Tests for user management endpoints.
"""

import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.main import app


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)

    def info(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.infos.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)


class TestUserEndpoints:
    """Tests for user management endpoints."""

    @pytest.mark.asyncio
    async def test_get_user_profile(self, test_user, valid_access_token):
        """Test getting user profile."""
        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user

        async def mock_get_current_active_user():
            return test_user

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {valid_access_token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert data["role"] == test_user["role"]
        assert data["storage_quota_mb"] == test_user["storage_quota_mb"]
        assert data["storage_used_mb"] == test_user["storage_used_mb"]

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_update_user_profile(self, mock_db_pool, test_user, valid_access_token):
        """Test updating user profile."""
        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={**test_user, "email": "newemail@example.com"})
        mock_conn.commit = AsyncMock()

        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user, get_db_transaction

        async def mock_get_current_active_user():
            return test_user

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

        async def mock_get_db_transaction():
            yield mock_conn

        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={"email": "newemail@example.com"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newemail@example.com"

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_change_password(self, monkeypatch, mock_db_pool, password_service, test_user, valid_access_token):
        """Test changing user password."""
        # Setup user with known password
        test_user_copy = test_user.copy()
        test_user_copy["password_hash"] = password_service.hash_password("Old@Pass#2024")

        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_user_copy["password_hash"])
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()

        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep,
        )
        from tldw_Server_API.app.api.v1.endpoints import users as users_endpoint

        async def mock_get_current_active_user():
            return test_user_copy

        fake_repo = AsyncMock()
        fake_repo.get_user_by_id = AsyncMock(return_value=test_user_copy)

        monkeypatch.setattr(
            users_endpoint.AuthnzUsersRepo,
            "from_pool",
            AsyncMock(return_value=fake_repo),
        )
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

        async def mock_get_db_transaction():
            yield mock_conn

        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        app.dependency_overrides[get_password_service_dep] = lambda: password_service

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={"current_password": "Old@Pass#2024", "new_password": "New@Secure#2024!"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "Password changed successfully" in data["message"]

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(
        self, monkeypatch, mock_db_pool, password_service, test_user, valid_access_token
    ):
        """Test changing password with wrong current password."""
        test_user_copy = test_user.copy()
        test_user_copy["password_hash"] = password_service.hash_password("Old@Pass#2024")

        # Setup mock connection with proper transaction context
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=test_user_copy["password_hash"])

        # Mock the transaction context manager
        mock_db_pool.transaction.return_value.__aenter__.return_value = mock_conn
        mock_db_pool.transaction.return_value.__aexit__.return_value = None

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
            get_current_active_user,
            get_db_transaction,
            get_password_service_dep,
        )
        from tldw_Server_API.app.api.v1.endpoints import users as users_endpoint

        async def mock_get_current_active_user():
            return test_user_copy

        fake_repo = AsyncMock()
        fake_repo.get_user_by_id = AsyncMock(return_value=test_user_copy)

        monkeypatch.setattr(
            users_endpoint.AuthnzUsersRepo,
            "from_pool",
            AsyncMock(return_value=fake_repo),
        )
        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

        async def mock_get_db_transaction():
            yield mock_conn

        app.dependency_overrides[get_db_transaction] = mock_get_db_transaction
        app.dependency_overrides[get_password_service_dep] = lambda: password_service

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/users/change-password",
                headers={"Authorization": f"Bearer {valid_access_token}"},
                json={"current_password": "wrongpass", "new_password": "New@Secure#2024!"},
            )

        # Should return 401 for incorrect password authentication
        assert response.status_code == 401
        assert "Current password is incorrect" in response.json()["detail"]

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, mock_db_pool, session_manager, test_user, valid_access_token):
        """Test getting user sessions."""
        mock_sessions = [
            {
                "id": 1,  # Changed to integer to match database schema
                "user_id": test_user["id"],
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "ip_address": "127.0.0.1",
                "user_agent": "TestClient/1.0",
                "device_id": None,
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            },
            {
                "id": 2,  # Changed to integer to match database schema
                "user_id": test_user["id"],
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0",
                "device_id": None,
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            },
        ]

        session_manager.get_user_sessions = AsyncMock(return_value=mock_sessions)
        session_manager.get_active_sessions = AsyncMock(return_value=mock_sessions)

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user, get_session_manager_dep

        async def mock_get_current_active_user():
            return test_user

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_session_manager_dep] = lambda: session_manager

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/users/sessions", headers={"Authorization": f"Bearer {valid_access_token}"}
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[1]["id"] == 2

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_revoke_session(self, session_manager, test_user, valid_access_token):
        """Test revoking a user session."""
        # Mock get_user_sessions to return a session with id 123
        session_manager.get_user_sessions = AsyncMock(
            return_value=[
                {
                    "id": 123,
                    "user_id": test_user["id"],
                    "created_at": datetime.utcnow(),
                    "expires_at": datetime.utcnow() + timedelta(hours=1),
                }
            ]
        )
        session_manager.revoke_session = AsyncMock()

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user, get_session_manager_dep

        async def mock_get_current_active_user():
            return test_user

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_session_manager_dep] = lambda: session_manager

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete(
                "/api/v1/users/sessions/123",  # Use integer session ID
                headers={"Authorization": f"Bearer {valid_access_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "Session revoked successfully" in data["message"]

        # Verify the session was revoked
        session_manager.revoke_session.assert_called_once_with(
            123, revoked_by=test_user["id"], reason="User requested revocation"
        )

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_storage_quota(self, storage_service, test_user, valid_access_token):
        """Test getting storage quota information."""
        # Create a mock storage info dictionary matching the service's return format
        storage_info = {
            "user_id": test_user["id"],
            "total_mb": 100,
            "quota_mb": 1000,
            "available_mb": 900,
            "usage_percentage": 10.0,
            "user_data_mb": 100,
            "chromadb_mb": 0,
            "total_bytes": 104857600,
            "user_data_bytes": 104857600,
            "chromadb_bytes": 0,
            "calculated_at": "2024-01-01T00:00:00",
        }

        storage_service.calculate_user_storage = AsyncMock(return_value=storage_info)

        from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user, get_storage_service_dep

        async def mock_get_current_active_user():
            return test_user

        app.dependency_overrides[get_current_active_user] = mock_get_current_active_user
        app.dependency_overrides[get_storage_service_dep] = lambda: storage_service

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/users/storage", headers={"Authorization": f"Bearer {valid_access_token}"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["storage_quota_mb"] == 1000
        assert data["storage_used_mb"] == 100
        assert data["available_mb"] == 900
        assert data["usage_percentage"] == 10.0

        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_storage_quota_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import users

    class _FailingStorageService:
        async def calculate_user_storage(self, user_id: int, update_database: bool):  # noqa: ARG002
            raise RuntimeError("quota backend exploded at /private/quota.db")

    async def _fake_require_principal_active_verified(_principal):
        return {
            "id": 7,
            "username": "alice",
            "storage_quota_mb": 1000,
            "storage_used_mb": 250,
        }

    logger_stub = _LoggerStub()
    monkeypatch.setattr(users, "_require_principal_active_verified", _fake_require_principal_active_verified)
    monkeypatch.setattr(users, "logger", logger_stub)

    response = await users.get_storage_quota(
        principal=object(),
        storage_service=_FailingStorageService(),
    )

    assert response.user_id == 7
    assert response.storage_quota_mb == 1000
    assert response.storage_used_mb == 250
    assert response.available_mb == 750
    assert logger_stub.errors == ["Failed to get storage quota"]
    assert "quota backend exploded" not in str(logger_stub.errors)
    assert "/private/quota.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_recalculate_storage_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import users

    class _FailingStorageService:
        async def calculate_user_storage(self, user_id: int, update_database: bool):  # noqa: ARG002
            raise RuntimeError("recalculate backend exploded at /private/recalc.db")

    async def _fake_require_principal_active_verified(_principal):
        return {"id": 7, "username": "alice"}

    logger_stub = _LoggerStub()
    monkeypatch.setattr(users, "_require_principal_active_verified", _fake_require_principal_active_verified)
    monkeypatch.setattr(users, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await users.recalculate_storage(
            principal=object(),
            storage_service=_FailingStorageService(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to recalculate storage"
    assert logger_stub.errors == ["Failed to recalculate storage"]
    assert "recalculate backend exploded" not in str(logger_stub.errors)
    assert "/private/recalc.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_update_user_profile_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import users
    from tldw_Server_API.app.api.v1.schemas.auth_schemas import UpdateProfileRequest

    async def _failing_resolve_user_context(*_args, **_kwargs):
        raise RuntimeError("profile backend exploded at /private/profile.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(users, "_resolve_user_context", _failing_resolve_user_context)
    monkeypatch.setattr(users, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await users.update_user_profile(
            request=UpdateProfileRequest(email="new@example.com"),
            principal=object(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update profile"
    assert logger_stub.errors == ["Failed to update user profile"]
    assert "profile backend exploded" not in str(logger_stub.errors)
    assert "/private/profile.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_user_profile_audit_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import users

    async def _raise_audit_service(_user_id):
        raise RuntimeError("audit backend exploded at /private/audit.db")

    fake_audit_deps = SimpleNamespace(get_or_create_audit_service_for_user_id=_raise_audit_service)
    logger_stub = _LoggerStub()
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps",
        fake_audit_deps,
    )
    monkeypatch.setattr(users, "logger", logger_stub)

    await users._emit_user_profile_audit_event(
        SimpleNamespace(),
        user_id=7,
        update_keys=["email"],
        applied_count=0,
        skipped_count=1,
        dry_run=True,
    )

    assert logger_stub.debugs == ["User profile audit emission skipped"]
    assert "audit backend exploded" not in str(logger_stub.debugs)
    assert "/private/audit.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_change_password_failure_log_is_sanitized(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import users
    from tldw_Server_API.app.api.v1.schemas.auth_schemas import PasswordChangeRequest

    async def _failing_require_principal_active_verified(_principal):
        raise RuntimeError("password backend exploded at /private/password.db")

    current_password = "Current" + "@Pass" + "#2024"
    new_password = "Changed" + "@Pass" + "#2024"
    logger_stub = _LoggerStub()
    monkeypatch.setattr(users, "_require_principal_active_verified", _failing_require_principal_active_verified)
    monkeypatch.setattr(users, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await users.change_password(
            request=PasswordChangeRequest(
                current_password=current_password,
                new_password=new_password,
            ),
            principal=object(),
            password_service=object(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to change password"
    assert logger_stub.errors == ["Failed to change password"]
    assert "password backend exploded" not in str(logger_stub.errors)
    assert "/private/password.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_change_password_repo_lookup_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import users
    from tldw_Server_API.app.api.v1.schemas.auth_schemas import PasswordChangeRequest

    class _FailingUsersRepo:
        @classmethod
        async def from_pool(cls):
            raise RuntimeError("repo lookup exploded at /private/users.db")

    class _PasswordService:
        def verify_password(self, current_password: str, password_hash: str):
            assert current_password == "Current@Pass#2024"
            assert password_hash == "stored-hash"
            return True, False

        def validate_password_strength(self, new_password: str, username: str) -> None:
            assert new_password == "Changed@Pass#2024"
            assert username == "alice"

        def hash_password(self, new_password: str) -> str:
            assert new_password == "Changed@Pass#2024"
            return "new-hash"

    class _Db:
        def __init__(self) -> None:
            self.statements: list[tuple[str, tuple[object, ...]]] = []

        async def execute(self, statement: str, *args: object) -> None:
            self.statements.append((statement, args))

    async def _fake_require_principal_active_verified(_principal):
        return {"id": 7, "username": "alice"}

    async def _fake_fetch_password_hash_for_user(_db, user_id: int):
        assert user_id == 7
        return "stored-hash"

    logger_stub = _LoggerStub()
    db = _Db()
    monkeypatch.setattr(users, "_require_principal_active_verified", _fake_require_principal_active_verified)
    monkeypatch.setattr(users, "_fetch_password_hash_for_user", _fake_fetch_password_hash_for_user)
    monkeypatch.setattr(users, "AuthnzUsersRepo", _FailingUsersRepo)
    monkeypatch.setattr(users, "logger", logger_stub)

    response = await users.change_password(
        request=PasswordChangeRequest(
            current_password="Current@Pass#2024",
            new_password="Changed@Pass#2024",
        ),
        principal=SimpleNamespace(username="alice"),
        password_service=_PasswordService(),
        db=db,
    )

    assert response.message == "Password changed successfully"
    assert response.details == {"user_id": 7}
    assert len(db.statements) == 2
    assert logger_stub.debugs == ["User repo lookup skipped for password change"]
    assert "repo lookup exploded" not in str(logger_stub.debugs)
    assert "/private/users.db" not in str(logger_stub.debugs)
