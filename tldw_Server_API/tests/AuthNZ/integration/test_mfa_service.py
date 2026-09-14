import asyncio
from pathlib import Path

import pyotp
import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.mfa_service import MFAService
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_mfa_end_to_end_flow(tmp_path: Path):
    db_path = tmp_path / "authnz_mfa.sqlite"
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="test-secret-key-32-characters-minimum!",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(settings)
    await pool.initialize()

    try:
        service = MFAService(db_pool=pool, settings=settings)
        secret = service.generate_secret()
        backup_codes = service.generate_backup_codes()

        column_rows = await pool.fetchall("PRAGMA table_info(users)")
        column_names = {
            row["name"] if isinstance(row, dict) else row[1] for row in column_rows
        }
        required_columns = {
            "two_factor_enabled": "INTEGER DEFAULT 0",
            "totp_secret": "TEXT",
            "backup_codes": "TEXT",
            "updated_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
        }
        for column, decl in required_columns.items():
            if column not in column_names:
                await pool.execute(f"ALTER TABLE users ADD COLUMN {column} {decl}")

        async with pool.transaction() as conn:
            await VersionedUserWriteGateway("sqlite").insert_user(
                conn,
                values={
                    "id": 1,
                    "username": "alice",
                    "email": "alice@example.com",
                    "password_hash": "hashed-password",
                    "is_active": 1,
                },
            )

        assert await service.enable_mfa(user_id=1, secret=secret, backup_codes=backup_codes)

        row = await pool.fetchone(
            "SELECT totp_secret, two_factor_enabled, backup_codes FROM users WHERE id = ?",
            1,
        )
        assert row is not None
        stored_secret = row["totp_secret"]
        assert stored_secret and stored_secret != secret
        assert bool(row["two_factor_enabled"])

        decrypted = await service.get_user_totp_secret(1)
        assert decrypted == secret

        totp = pyotp.TOTP(secret)
        assert service.verify_totp(secret, totp.now())
        assert not service.verify_totp(secret, "123456")

        status = await service.get_user_mfa_status(1)
        assert status == {
            "enabled": True,
            "has_secret": True,
            "has_backup_codes": True,
            "method": "totp",
        }

        assert await service.verify_backup_code(1, backup_codes[0])
        assert not await service.verify_backup_code(1, backup_codes[0])

        regenerated = await service.regenerate_backup_codes(1)
        assert regenerated and len(regenerated) == service.backup_codes_count

        assert await service.disable_mfa(1)
        status_after = await service.get_user_mfa_status(1)
        assert status_after == {
            "enabled": False,
            "has_secret": False,
            "has_backup_codes": False,
            "method": None,
        }
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_backup_code_consumption_is_atomic_under_concurrency(tmp_path: Path):
    """
    Verify that a single backup code cannot be successfully consumed twice
    when two verification calls run concurrently.
    """
    db_path = tmp_path / "authnz_mfa_atomic.sqlite"
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="test-secret-key-32-characters-minimum!",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(settings)
    await pool.initialize()

    try:
        service = MFAService(db_pool=pool, settings=settings)
        secret = service.generate_secret()
        backup_codes = service.generate_backup_codes()

        column_rows = await pool.fetchall("PRAGMA table_info(users)")
        column_names = {
            row["name"] if isinstance(row, dict) else row[1] for row in column_rows
        }
        required_columns = {
            "two_factor_enabled": "INTEGER DEFAULT 0",
            "totp_secret": "TEXT",
            "backup_codes": "TEXT",
            "updated_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
        }
        for column, decl in required_columns.items():
            if column not in column_names:
                await pool.execute(f"ALTER TABLE users ADD COLUMN {column} {decl}")

        async with pool.transaction() as conn:
            await VersionedUserWriteGateway("sqlite").insert_user(
                conn,
                values={
                    "id": 7,
                    "username": "bob",
                    "email": "bob@example.com",
                    "password_hash": "hashed-password",
                    "is_active": 1,
                },
            )

        assert await service.enable_mfa(user_id=7, secret=secret, backup_codes=backup_codes)

        code = backup_codes[0]

        async def _consume():
            return await service.verify_backup_code(7, code)

        # Fire two concurrent verification attempts for the same backup code.
        first, second = await asyncio.gather(_consume(), _consume())

        # Exactly one of the calls must succeed; the other must fail.
        assert sorted([first, second]) == [False, True]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_mfa_secret_and_backup_codes_survive_key_rotation(tmp_path: Path):
    db_path = tmp_path / "authnz_mfa_rotation.sqlite"
    primary_secret = "primary-secret-key-for-mfa-tests-0001"
    rotated_secret = "rotated-secret-key-for-mfa-tests-0002"

    primary_settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY=primary_secret,
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(primary_settings)
    await pool.initialize()

    try:
        service_primary = MFAService(db_pool=pool, settings=primary_settings)
        secret = service_primary.generate_secret()
        backup_codes = service_primary.generate_backup_codes()

        column_rows = await pool.fetchall("PRAGMA table_info(users)")
        column_names = {
            row["name"] if isinstance(row, dict) else row[1] for row in column_rows
        }
        required_columns = {
            "two_factor_enabled": "INTEGER DEFAULT 0",
            "totp_secret": "TEXT",
            "backup_codes": "TEXT",
            "updated_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
        }
        for column, decl in required_columns.items():
            if column not in column_names:
                await pool.execute(f"ALTER TABLE users ADD COLUMN {column} {decl}")

        async with pool.transaction() as conn:
            await VersionedUserWriteGateway("sqlite").insert_user(
                conn,
                values={
                    "id": 42,
                    "username": "rotating",
                    "email": "rotate@example.com",
                    "password_hash": "hashed-password",
                    "is_active": 1,
                },
            )

        assert await service_primary.enable_mfa(
            user_id=42, secret=secret, backup_codes=backup_codes
        )

        # Simulate key rotation: new JWT secret becomes primary, old secret retained as secondary
        rotated_settings = Settings(
            AUTH_MODE="multi_user",
            DATABASE_URL=f"sqlite:///{db_path}",
            JWT_SECRET_KEY=rotated_secret,
            JWT_SECONDARY_SECRET=primary_secret,
            ENABLE_REGISTRATION=True,
            REQUIRE_REGISTRATION_CODE=False,
        )
        rotated_service = MFAService(db_pool=pool, settings=rotated_settings)

        decrypted = await rotated_service.get_user_totp_secret(42)
        assert decrypted == secret

        totp = pyotp.TOTP(decrypted)
        assert rotated_service.verify_totp(decrypted, totp.now())

        assert await rotated_service.verify_backup_code(42, backup_codes[0])
        assert not await rotated_service.verify_backup_code(42, backup_codes[0])

        regenerated = await rotated_service.regenerate_backup_codes(42)
        assert regenerated and len(regenerated) == rotated_service.backup_codes_count
        assert await rotated_service.verify_backup_code(42, regenerated[0])
        assert not await rotated_service.verify_backup_code(42, regenerated[0])
    finally:
        await pool.close()
