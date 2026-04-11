#!/usr/bin/env python3
"""
CLI for resetting an admin or standard user's password in multi-user mode.

This script works with both SQLite and PostgreSQL backends by using the same
UsersDB abstraction as the rest of the AuthNZ subsystem.

Usage:
    python -m tldw_Server_API.app.core.AuthNZ.reset_admin_password \\
        --username admin --new-password 'N3wS3cure!'
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db


async def reset_user_password(
    username: str,
    new_password: str,
) -> bool:
    """Reset a user's password by username.

    Returns True on success, False on failure.
    Works with both SQLite and PostgreSQL backends.
    """
    settings = get_settings()

    if settings.AUTH_MODE != "multi_user":
        print("[reset-password] Not in multi_user mode; password reset is only for multi-user deployments.")
        return False

    # Validate password length against configured minimum
    _pw_svc = PasswordService()
    if len(new_password) < _pw_svc.min_length:
        print(f"[reset-password] Password must be at least {_pw_svc.min_length} characters.")
        return False

    try:
        users_db = await get_users_db()

        # Look up user by username
        user = await users_db.get_user_by_username(username)
        if not user:
            print(f"[reset-password] User '{username}' not found.")
            return False

        user_id = user["id"]
        user_role = user.get("role", "unknown")

        # Hash new password
        password_service = PasswordService()
        new_hash = password_service.hash_password(new_password)

        # Update password in database (works for both SQLite and PostgreSQL)
        await users_db.update_user(user_id, password_hash=new_hash)

        print(f"[reset-password] Password updated for user '{username}' (id={user_id}, role={user_role}).")
        return True

    except Exception as e:
        print("[reset-password] Failed to reset password. Check server logs for details.")
        logger.opt(exception=True).error("reset_user_password failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset a user's password in multi-user mode.",
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Username of the account to reset.",
    )
    parser.add_argument(
        "--new-password",
        required=True,
        help="New password (min 10 characters).",
    )

    args = parser.parse_args()

    success = asyncio.run(
        reset_user_password(
            username=args.username,
            new_password=args.new_password,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
