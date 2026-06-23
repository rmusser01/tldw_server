import os
import json
import pytest


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_account_tokens_encrypted_env(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.External_Sources import connectors_service as svc

    # Enable encryption
    # Valid base64-encoded 32-byte key (AES-256)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=")

    import aiosqlite
    db_path = tmp_path / "enc.db"
    async with aiosqlite.connect(str(db_path)) as db:
        # Create account with initial tokens
        acct = await svc.create_account(db, user_id=1, provider="drive", display_name="Drive", email="user@example.com", tokens={
            "access_token": "at1", "refresh_token": "rt1", "scope": "drive.readonly"
        })
        account_id = int(acct["id"]) if isinstance(acct, dict) else int(acct.id)

        # Stored access_token should be envelope JSON
        cur = await db.execute("SELECT access_token, refresh_token FROM external_accounts WHERE id = ?", (account_id,))
        row = await cur.fetchone()
        assert row is not None
        access_token_store, refresh_store = row[0], row[1]
        assert isinstance(access_token_store, str) and access_token_store.strip().startswith("{")
        env = json.loads(access_token_store)
        assert env.get("_enc") == "aesgcm:v1"
        # get_account_tokens returns decrypted tokens
        toks = await svc.get_account_tokens(db, 1, account_id)
        assert toks.get("access_token") == "at1"
        assert toks.get("refresh_token") == "rt1"

        # Update tokens (simulate refresh)
        ok = await svc.update_account_tokens(db, 1, account_id, {"access_token": "at2", "scope": "drive.readonly"})
        assert ok is True
        # New envelope persisted
        cur2 = await db.execute("SELECT access_token, refresh_token FROM external_accounts WHERE id = ?", (account_id,))
        row2 = await cur2.fetchone()
        env2 = json.loads(row2[0])
        assert env2.get("_enc") == "aesgcm:v1"
        # Decoded tokens reflect update and preserve refresh token
        toks2 = await svc.get_account_tokens(db, 1, account_id)
        assert toks2.get("access_token") == "at2"
        assert toks2.get("refresh_token") == "rt1"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_account_tokens_plaintext_without_env(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.External_Sources import connectors_service as svc
    # Ensure encryption disabled
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("CONNECTORS_REQUIRE_TOKEN_ENCRYPTION", raising=False)
    monkeypatch.delenv("tldw_production", raising=False)
    monkeypatch.delenv("WORKFLOWS_ARTIFACT_ENC_KEY", raising=False)

    import aiosqlite
    db_path = tmp_path / "plain.db"
    async with aiosqlite.connect(str(db_path)) as db:
        acct = await svc.create_account(db, user_id=1, provider="notion", display_name="Notion", email=None, tokens={
            "access_token": "n1", "refresh_token": "nr1"
        })
        account_id = int(acct["id"]) if isinstance(acct, dict) else int(acct.id)
        # Stored access_token is plaintext
        cur = await db.execute("SELECT access_token, refresh_token FROM external_accounts WHERE id = ?", (account_id,))
        row = await cur.fetchone()
        assert row[0] == "n1"
        # Update
        await svc.update_account_tokens(db, 1, account_id, {"access_token": "n2"})
        cur2 = await db.execute("SELECT access_token FROM external_accounts WHERE id = ?", (account_id,))
        row2 = await cur2.fetchone()
        assert row2[0] == "n2"
        # get_account_tokens reflects new value
        toks = await svc.get_account_tokens(db, 1, account_id)
        assert toks.get("access_token") == "n2"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_account_requires_encryption_in_multi_user(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.External_Sources import connectors_service as svc
    from tldw_Server_API.app.core.Security import crypto

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.delenv("CONNECTORS_REQUIRE_TOKEN_ENCRYPTION", raising=False)
    monkeypatch.delenv("tldw_production", raising=False)
    monkeypatch.delenv("WORKFLOWS_ARTIFACT_ENC_KEY", raising=False)
    monkeypatch.setattr(crypto, "encrypt_json_blob", lambda _tokens: None)

    import aiosqlite
    db_path = tmp_path / "required-encryption.db"
    async with aiosqlite.connect(str(db_path)) as db:
        with pytest.raises(RuntimeError, match="Connector secret encryption is required"):
            await svc.create_account(
                db,
                user_id=1,
                provider="drive",
                display_name="Drive",
                email="user@example.com",
                tokens={"access_token": "at1", "refresh_token": "rt1"},
            )

        cur = await db.execute("SELECT COUNT(*) FROM external_accounts")
        row = await cur.fetchone()
        assert row[0] == 0
