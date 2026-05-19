from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.mfa_service import MFAService
from tldw_Server_API.app.core.AuthNZ.settings import Settings


@pytest.mark.asyncio
async def test_mfa_service_reports_initialized_sqlite_backend_as_supported(tmp_path: Path):
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'authnz_mfa_supported.sqlite'}",
        JWT_SECRET_KEY="test-secret-key-32-characters-minimum!",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )
    pool = DatabasePool(settings)
    await pool.initialize()

    try:
        service = MFAService(db_pool=pool, settings=settings)
        await service.initialize()

        assert service.supports_backend() is True
    finally:
        await pool.close()
