from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import LLMProviderOverrideRequest
from tldw_Server_API.app.services import admin_llm_providers_service as service

pytestmark = pytest.mark.unit


class _EventGatedOverrideRepo:
    """Model concurrent stale reads and force the metadata write to finish last."""

    def __init__(self) -> None:
        self.row: dict[str, Any] = {
            "provider": "openai",
            "is_enabled": True,
            "allowed_models": None,
            "config_json": None,
            "secret_blob": "old-secret",
            "api_key_hint": "old-hint",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        self._read_count = 0
        self._both_read = asyncio.Event()
        self._secret_written = asyncio.Event()
        self.seen_providers: list[str] = []

    async def fetch_override(self, provider: str) -> dict[str, Any]:
        self.seen_providers.append(provider)
        snapshot = deepcopy(self.row)
        self._read_count += 1
        if self._read_count == 2:
            self._both_read.set()
        await self._both_read.wait()
        return snapshot

    async def upsert_override(self, **values: Any) -> dict[str, Any]:
        """Reproduce the old full-row overwrite for the RED assertion."""
        is_secret_write = values["secret_blob"] != "old-secret"
        if is_secret_write:
            self.row.update(values)
            self._secret_written.set()
        else:
            await self._secret_written.wait()
            self.row.update(values)
        return deepcopy(self.row)

    async def patch_override(
        self,
        *,
        provider: str,
        fields: dict[str, Any],
        updated_at: datetime,
        compare_secret_blob: bool = False,
        expected_secret_blob: str | None = None,
    ) -> dict[str, Any] | None:
        """Apply only supplied columns, matching the repository patch contract."""
        self.seen_providers.append(provider)
        if "secret_blob" in fields:
            if compare_secret_blob and self.row["secret_blob"] != expected_secret_blob:
                return None
            self.row.update(fields)
            self._secret_written.set()
        else:
            await self._secret_written.wait()
            self.row.update(fields)
        self.row.update(provider=provider, updated_at=updated_at)
        return deepcopy(self.row)


class _EventGatedCredentialRepo:
    """Force a rotation/clear to commit after a credential-fields stale read."""

    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.row: dict[str, Any] = {
            "provider": "openai",
            "is_enabled": True,
            "allowed_models": None,
            "config_json": None,
            "secret_blob": json.dumps(
                {
                    "api_key": "old-key",
                    "credential_fields": {"org_id": "old-org"},
                },
                sort_keys=True,
            ),
            "api_key_hint": "old-hint",
            "created_at": now,
            "updated_at": now,
        }
        self._read_count = 0
        self._both_read = asyncio.Event()
        self._mutation_committed = asyncio.Event()
        self.seen_providers: list[str] = []

    async def fetch_override(self, provider: str) -> dict[str, Any]:
        self.seen_providers.append(provider)
        snapshot = deepcopy(self.row)
        self._read_count += 1
        if self._read_count <= 2:
            if self._read_count == 2:
                self._both_read.set()
            await self._both_read.wait()
        return snapshot

    async def patch_override(
        self,
        *,
        provider: str,
        fields: dict[str, Any],
        updated_at: datetime,
        compare_secret_blob: bool = False,
        expected_secret_blob: str | None = None,
    ) -> dict[str, Any] | None:
        self.seen_providers.append(provider)
        candidate = fields.get("secret_blob")
        decoded = json.loads(candidate) if isinstance(candidate, str) else {}
        credential_fields = decoded.get("credential_fields") or {}
        is_fields_only = credential_fields.get("org_id") == "new-org"

        if is_fields_only:
            await self._mutation_committed.wait()

        if compare_secret_blob and self.row["secret_blob"] != expected_secret_blob:
            return None

        self.row.update(fields)
        self.row.update(provider=provider, updated_at=updated_at)
        if not is_fields_only:
            self._mutation_committed.set()
        return deepcopy(self.row)


def _override_response() -> SimpleNamespace:
    return SimpleNamespace(
        provider="openai",
        is_enabled=True,
        allowed_models=None,
        config={},
        credential_fields={},
        api_key=None,
        api_key_hint=None,
        created_at=None,
        updated_at=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("secret_action", ["clear", "rotate"])
async def test_concurrent_secret_and_metadata_updates_do_not_restore_stale_secret(
    monkeypatch: pytest.MonkeyPatch,
    secret_action: str,
) -> None:
    repo = _EventGatedOverrideRepo()

    async def get_repo() -> _EventGatedOverrideRepo:
        return repo

    async def no_refresh(*, force: bool = False) -> None:
        assert force is True

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)
    monkeypatch.setattr(service, "_refresh_overrides_or_503", no_refresh)
    monkeypatch.setattr(service, "get_llm_provider_override", lambda _provider: _override_response())
    monkeypatch.setattr(service, "encrypt_byok_payload", lambda value: value)
    monkeypatch.setattr(service, "dumps_envelope", lambda value: str(value))

    secret_request = (
        LLMProviderOverrideRequest(clear_api_key=True)
        if secret_action == "clear"
        else LLMProviderOverrideRequest(api_key="new-key")
    )

    await asyncio.gather(
        service.upsert_override("OAI", secret_request),
        service.upsert_override("openai", LLMProviderOverrideRequest(is_enabled=False)),
    )

    assert repo.row["is_enabled"] is False
    if secret_action == "clear":
        assert repo.row["secret_blob"] is None
        assert repo.row["api_key_hint"] is None
    else:
        assert "new-key" in repo.row["secret_blob"]
        assert repo.row["api_key_hint"] != "old-hint"
    assert set(repo.seen_providers) == {"openai"}


@pytest.mark.asyncio
@pytest.mark.parametrize("secret_action", ["clear", "rotate"])
async def test_credential_fields_update_never_restores_key_changed_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    secret_action: str,
) -> None:
    repo = _EventGatedCredentialRepo()

    async def get_repo() -> _EventGatedCredentialRepo:
        return repo

    async def no_refresh(*, force: bool = False) -> None:
        assert force is True

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)
    monkeypatch.setattr(service, "_refresh_overrides_or_503", no_refresh)
    monkeypatch.setattr(service, "get_llm_provider_override", lambda _provider: _override_response())
    monkeypatch.setattr(service, "loads_envelope", json.loads)
    monkeypatch.setattr(service, "decrypt_byok_payload", lambda value: value)
    monkeypatch.setattr(service, "encrypt_byok_payload", lambda value: value)
    monkeypatch.setattr(
        service,
        "dumps_envelope",
        lambda value: json.dumps(value, sort_keys=True),
    )

    secret_request = (
        LLMProviderOverrideRequest(clear_api_key=True)
        if secret_action == "clear"
        else LLMProviderOverrideRequest(api_key="new-key")
    )
    results = await asyncio.gather(
        service.upsert_override(
            "OAI",
            LLMProviderOverrideRequest(credential_fields={"org_id": "new-org"}),
        ),
        service.upsert_override("openai", secret_request),
        return_exceptions=True,
    )

    stored = json.loads(repo.row["secret_blob"]) if repo.row["secret_blob"] else None
    if secret_action == "clear":
        assert isinstance(results[0], HTTPException)
        assert results[0].status_code == 400
        assert repo.row["secret_blob"] is None
        assert repo.row["api_key_hint"] is None
    else:
        assert not isinstance(results[0], BaseException)
        assert stored["api_key"] == "new-key"
        assert stored["credential_fields"] == {"org_id": "new-org"}
    assert repo._read_count == 3
    assert set(repo.seen_providers) == {"openai"}
