from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import user_keys as user_keys_endpoint


@pytest.mark.asyncio
async def test_openai_oauth_token_exchange_sanitizes_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 400

        def json(self) -> dict[str, str]:
            return {"error_description": "client authentication failed"}

        async def aclose(self) -> None:
            return None

    async def _fake_http_afetch(**_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(user_keys_endpoint, "_http_afetch", _fake_http_afetch)

    with pytest.raises(HTTPException) as exc_info:
        await user_keys_endpoint._openai_oauth_token_exchange(
            token_url="https://oauth.example.com/token",
            form_data={"code": "bad-code"},
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "OpenAI OAuth token exchange failed"
