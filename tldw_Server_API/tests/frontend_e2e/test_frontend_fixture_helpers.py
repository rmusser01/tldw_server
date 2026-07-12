from tldw_Server_API.tests.frontend_e2e import conftest as frontend_conftest


class _FakeHeaders:
    @staticmethod
    def get_content_charset() -> str:
        return "utf-8"


class _FakeResponse:
    status = 200
    headers = _FakeHeaders()

    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0
        self.read_sizes: list[int] = []

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_fetch_frontend_reads_large_next_document_until_readiness_marker(monkeypatch) -> None:
    body = (
        b"<html><head>"
        + (b"x" * (64 * 1024 + 100))
        + b'</head><body><div id="__next"></div></body></html>'
    )
    response = _FakeResponse(body)
    monkeypatch.setattr(frontend_conftest.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    result = frontend_conftest._fetch_frontend("http://127.0.0.1:3000")

    assert frontend_conftest._is_frontend_response(result["body"])
    assert len(response.read_sizes) >= 2
    assert all(
        size == frontend_conftest._FRONTEND_RESPONSE_CHUNK_BYTES
        for size in response.read_sizes
    )


def test_fetch_frontend_caps_unmarked_response_body(monkeypatch) -> None:
    response = _FakeResponse(b"x" * (2 * 1024 * 1024))
    monkeypatch.setattr(
        frontend_conftest.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: response,
    )

    result = frontend_conftest._fetch_frontend("http://127.0.0.1:3000")

    assert len(result["body"].encode("utf-8")) == frontend_conftest._FRONTEND_RESPONSE_MAX_BYTES
    assert sum(response.read_sizes) == frontend_conftest._FRONTEND_RESPONSE_MAX_BYTES
