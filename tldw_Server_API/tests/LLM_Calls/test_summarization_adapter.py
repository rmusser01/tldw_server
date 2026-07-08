import inspect

import pytest

from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl
import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod


@pytest.mark.unit
@pytest.mark.parametrize("api_name", [None, "", "   ", "none", "None"])
def test_analyze_returns_clear_error_when_api_name_missing(api_name: str | None) -> None:
    """Missing analysis providers return a clear error instead of leaking dispatch internals."""
    result = sgl.analyze(
        api_name=api_name,
        input_data="hello",
        custom_prompt_arg="summarize",
    )

    assert result == "Error: Analysis API provider is required."
    assert "NoneType" not in result
    assert "Error calling API" not in result


class _FakeResp:
    def __init__(self, *, json_data=None, lines=None):
        self._json = json_data or {}
        self._lines = list(lines or [])
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._json

    def iter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, *, json_data=None, lines=None):
        self._json_data = json_data or {}
        self._lines = list(lines or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return _FakeResp(json_data=self._json_data)

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx(_FakeResp(lines=self._lines))


@pytest.mark.unit
def test_analyze_uses_adapter_non_stream(monkeypatch):
    fake_json = {"choices": [{"message": {"content": "summary"}}]}
    monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient(json_data=fake_json))

    result = sgl.analyze(
        api_name="openai",
        input_data="hello",
        custom_prompt_arg="summarize",
        api_key="x",
        system_message="system",
        temp=0.1,
        streaming=False,
        model_override="gpt-4o-mini",
    )

    assert result == "summary"


@pytest.mark.unit
def test_analyze_uses_adapter_stream(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        'data: [DONE]\n\n',
    ]
    monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient(lines=lines))

    result = sgl.analyze(
        api_name="openai",
        input_data="hello",
        custom_prompt_arg="summarize",
        api_key="x",
        system_message="system",
        temp=0.1,
        streaming=True,
        model_override="gpt-4o-mini",
    )

    assert inspect.isgenerator(result)
    assert "".join(list(result)) == "hello world"
