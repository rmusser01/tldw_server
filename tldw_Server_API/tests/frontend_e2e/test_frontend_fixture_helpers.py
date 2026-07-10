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

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]


def test_fetch_frontend_reads_large_next_document_until_readiness_marker(monkeypatch) -> None:
    body = b"<html><head>" + (b"x" * 5000) + b'</head><body><div id="__next"></div></body></html>'
    response = _FakeResponse(body)
    monkeypatch.setattr(frontend_conftest.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    result = frontend_conftest._fetch_frontend("http://127.0.0.1:3000")

    assert frontend_conftest._is_frontend_response(result["body"])
