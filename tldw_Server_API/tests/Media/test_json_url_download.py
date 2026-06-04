import asyncio
import io
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils import (
    download_url_async,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _allow_fake_client_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils._validate_egress_or_raise",
        lambda *_args, **_kwargs: None,
    )


class _FakeResponse:
    def __init__(self, url: str, headers: dict[str, str], content: bytes):
        from types import SimpleNamespace
        self._url = SimpleNamespace(path=Path(url).name or "/")
        self.headers = headers
        self._content = content
        self.status_code = 200
        self.text = content.decode("utf-8", errors="ignore")

    @property
    def url(self):
        # Mimic httpx.URL-like with .path and maybe host
        class _U:
            def __init__(self, path):
                self.path = f"/{path}" if not path.startswith("/") else path
                self.host = "example.org"

        return _U(self._url.path)

    def raise_for_status(self):

        return None

    async def aiter_bytes(self, chunk_size=8192):  # pragma: no cover - simple stream
        yield self._content


class _FakeStreamContext:
    def __init__(self, resp: _FakeResponse):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncClient:
    def __init__(self, headers: dict[str, str], body: bytes):
        self._headers = headers
        self._body = body

    def stream(self, method: str, url: str, follow_redirects: bool = True, timeout: float = 60.0):
        return _FakeStreamContext(_FakeResponse(url=url, headers=self._headers, content=self._body))


@pytest.mark.asyncio
async def test_download_url_json_content_type(tmp_path):
    # Simulate URL without extension, rely on Content-Type: application/json
    client = _FakeAsyncClient(headers={"content-type": "application/json"}, body=b'{"k":1}')
    out_path = await download_url_async(
        client=client,
        url="https://example.org/data",  # no extension
        target_dir=tmp_path,
        allowed_extensions={".json"},
        check_extension=True,
        disallow_content_types={"application/msword", "application/octet-stream"},
    )
    # Should infer .json from content-type map
    assert out_path.suffix == ".json", out_path
    assert out_path.exists() and out_path.read_text() == '{"k":1}'


@pytest.mark.asyncio
async def test_download_url_json_content_disposition(tmp_path):
    # Simulate Content-Disposition: filename="file.json" when URL has no extension
    hdrs = {
        "content-type": "application/octet-stream",
        "content-disposition": 'attachment; filename="file.json"',
    }
    client = _FakeAsyncClient(headers=hdrs, body=b'{"v":2}')
    out_path = await download_url_async(
        client=client,
        url="https://cdn.example.org/download?id=abc",
        target_dir=tmp_path,
        allowed_extensions={".json"},
        check_extension=True,
        disallow_content_types={"application/msword"},
    )
    # Should respect Content-Disposition filename
    assert out_path.name.endswith("file.json"), out_path
    assert out_path.exists() and out_path.read_text() == '{"v":2}'


@pytest.mark.asyncio
async def test_download_url_rejects_dotdot_filename(tmp_path):
    # Reject path traversal attempts via Content-Disposition filename
    hdrs = {
        "content-type": "application/json",
        "content-disposition": 'attachment; filename=".."',
    }
    client = _FakeAsyncClient(headers=hdrs, body=b'{"v":3}')
    with pytest.raises(ValueError):
        await download_url_async(
            client=client,
            url="https://example.org/download",
            target_dir=tmp_path,
            allowed_extensions=set(),
            check_extension=False,
        )
