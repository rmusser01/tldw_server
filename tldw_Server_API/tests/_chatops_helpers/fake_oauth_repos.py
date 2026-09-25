"""Fake OAuth repos shared by the Discord and Slack lifecycle tests.

These two classes were 90 lines byte-identical -- not merely similar after
normalising the provider vocabulary, but character for character -- in
tests/Discord/test_discord_oauth_lifecycle.py and
tests/Slack/test_slack_oauth_lifecycle.py, at the same line numbers. They are
provider-agnostic already: the provider is a value passed in, never a literal.

See ADR-050.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

class FakeOAuthStateRepo:
    def __init__(self) -> None:
        self.states: dict[str, dict] = {}

    async def create_state(
        self,
        *,
        state: str,
        user_id: int,
        provider: str,
        auth_session_id: str,
        redirect_uri: str,
        pkce_verifier_encrypted: str,
        expires_at,
        return_path=None,
        created_at=None,
    ) -> dict:
        record = {
            "state": state,
            "user_id": int(user_id),
            "provider": provider,
            "auth_session_id": auth_session_id,
            "redirect_uri": redirect_uri,
            "pkce_verifier_encrypted": pkce_verifier_encrypted,
            "expires_at": expires_at,
            "return_path": return_path,
            "created_at": created_at or datetime.now(timezone.utc),
        }
        self.states[state] = record
        return record

    async def consume_state(
        self,
        *,
        state: str,
        provider: str,
        consumed_at=None,
    ) -> dict | None:
        record = self.states.pop(state, None)
        if not record:
            return None
        if record.get("provider") != provider:
            return None
        return record


class FakeUserSecretRepo:
    def __init__(self) -> None:
        self.row: dict | None = None

    async def fetch_secret_for_user(self, user_id: int, provider: str, *, include_revoked: bool = False) -> dict | None:
        if not self.row:
            return None
        return dict(self.row)

    async def upsert_secret(
        self,
        *,
        user_id: int,
        provider: str,
        encrypted_blob: str,
        key_hint: str | None,
        metadata: dict | None,
        updated_at,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict:
        self.row = {
            "user_id": int(user_id),
            "provider": provider,
            "encrypted_blob": encrypted_blob,
            "key_hint": key_hint,
            "metadata": metadata,
            "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at),
            "created_by": created_by,
            "updated_by": updated_by,
        }
        return dict(self.row)

    async def delete_secret(
        self,
        user_id: int,
        provider: str,
        *,
        revoked_by: int | None = None,
        revoked_at=None,
    ) -> bool:
        self.row = None
        return True



__all__ = ["FakeOAuthStateRepo", "FakeUserSecretRepo"]
