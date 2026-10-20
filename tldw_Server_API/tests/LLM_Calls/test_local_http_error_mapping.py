
import httpx
import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatProviderError
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
    _chat_with_openai_compatible_local_server,
)


def _make_httpx_error(status_code: int, url: str = "http://local/v1/chat/completions") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, request=request, content=b'{"error":{"message":"failure"}}')
    return httpx.HTTPStatusError("failure", request=request, response=response)


@pytest.mark.unit
def test_local_openai_raises_bad_request_on_4xx():
    error = _make_httpx_error(400)

    def _fetch(**_kwargs):
        raise error

    with pytest.raises(ChatBadRequestError) as exc:
        _chat_with_openai_compatible_local_server(
            api_base_url="http://localhost:8000",
            model_name="local-model",
            input_data=[{"role": "user", "content": "ping"}],
            http_fetcher=_fetch,
        )
    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [401, 403, 404, 429])
def test_local_openai_rejection_keeps_the_upstream_status(status_code):
    error = _make_httpx_error(status_code)

    def _fetch(**_kwargs):
        raise error

    with pytest.raises(ChatBadRequestError) as exc:
        _chat_with_openai_compatible_local_server(
            api_base_url="http://localhost:8000",
            model_name="local-model",
            input_data=[{"role": "user", "content": "ping"}],
            http_fetcher=_fetch,
        )
    assert exc.value.status_code == 400
    assert exc.value.message == "The local model provider rejected the request."
    assert exc.value.upstream_status_code == status_code


@pytest.mark.unit
def test_local_openai_raises_provider_error_on_5xx():
    error = _make_httpx_error(502)

    def _fetch(**_kwargs):
        raise error

    with pytest.raises(ChatProviderError) as exc:
        _chat_with_openai_compatible_local_server(
            api_base_url="http://localhost:8000",
            model_name="local-model",
            input_data=[{"role": "user", "content": "ping"}],
            http_fetcher=_fetch,
        )
    assert exc.value.status_code == 502
