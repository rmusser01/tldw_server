"""The admin API sets, clears and returns a provider override's denied_models (TASK-15582)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.admin_schemas import LLMProviderOverrideRequest
from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.services import admin_llm_providers_service as service

pytestmark = pytest.mark.unit


class _RecordingRepo:
    """Records the columns each admin write patches."""

    def __init__(self) -> None:
        self.patches: list[dict[str, Any]] = []

    async def fetch_override(self, provider: str) -> None:
        return None

    async def patch_override(
        self, *, provider: str, fields: dict[str, Any], updated_at: datetime,
        compare_secret_blob: bool = False, expected_secret_blob: str | None = None,
    ) -> dict[str, Any]:
        self.patches.append(fields)
        return {"provider": provider, **fields}


@pytest.mark.asyncio
async def test_an_admin_sets_clears_and_reads_back_denied_models(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _RecordingRepo()

    async def get_repo() -> _RecordingRepo:
        return repo

    async def no_refresh(*, force: bool = False) -> None:
        return None

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)
    monkeypatch.setattr(service, "_refresh_overrides_or_503", no_refresh)
    monkeypatch.setattr(
        service,
        "get_llm_provider_override",
        lambda _provider: LLMProviderOverride(provider="openai", denied_models=["gpt-4o", "o1"]),
    )

    response = await service.upsert_override("openai", LLMProviderOverrideRequest(denied_models=[" gpt-4o ", "", "o1"]))
    await service.upsert_override("openai", LLMProviderOverrideRequest(denied_models=[]))

    assert (repo.patches, response.denied_models) == (
        [{"denied_models": '["gpt-4o", "o1"]'}, {"denied_models": None}],
        ["gpt-4o", "o1"],
    )
