import pytest

from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.exceptions import TokenizerUnavailable


class _AllowingRateLimiter:
    async def check_user_rate_limit(self, user_id: int, scope: str):  # noqa: ARG002
        return True, {}


@pytest.mark.unit
def test_tokenizer_support_sanitizes_unavailable_error(monkeypatch):
    def _raise_unavailable(provider: str, model: str):  # noqa: ARG001
        raise TokenizerUnavailable("tokenizer config exploded at /private/tokenizer.json")

    monkeypatch.setattr(writing_endpoints, "_resolve_tokenizer_details", _raise_unavailable)

    support = writing_endpoints._tokenizer_support("openai", "gpt-test")

    assert support.available is False
    assert support.error == "Tokenizer unavailable for provider/model"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_requested_capabilities_sanitizes_tokenizer_error(monkeypatch):
    def _raise_unavailable(provider: str, model: str):  # noqa: ARG001
        raise TokenizerUnavailable("tokenizer config exploded at /private/tokenizer.json")

    monkeypatch.setattr(writing_endpoints, "_resolve_tokenizer_details", _raise_unavailable)

    response = await writing_endpoints.get_writing_capabilities(
        provider="openai",
        model="gpt-test",
        include_providers=False,
        rate_limiter=_AllowingRateLimiter(),
        current_user=User(id=1, username="tester", email="tester@example.com", is_active=True, is_admin=True),
        _=None,
    )

    assert response.requested is not None
    assert response.requested.tokenizer_available is False
    assert response.requested.tokenization_error == "Tokenizer unavailable for provider/model"
