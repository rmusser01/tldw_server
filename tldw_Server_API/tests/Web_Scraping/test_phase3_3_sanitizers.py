import pytest

from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws


pytestmark = pytest.mark.unit


_LEAKY_ERROR = "backend exploded at /tmp/secret-token with api_key=abc123"


def _assert_safe_text(value: object) -> None:
    text = str(value)
    assert "backend exploded" not in text
    assert "/tmp/secret-token" not in text
    assert "api_key" not in text.lower()


@pytest.mark.parametrize(
    ("helper_name", "provider_label"),
    [
        ("test_perform_websearch_google", "google"),
        ("test_perform_websearch_brave", "brave"),
        ("test_perform_websearch_ddg", "duckduckgo"),
        ("test_perform_websearch_kagi", "kagi"),
        ("test_perform_websearch_serper", "serper"),
        ("test_perform_websearch_tavily", "tavily"),
        ("test_perform_websearch_searx", "searx"),
        ("test_perform_websearch_yandex", "yandex"),
    ],
)
def test_provider_search_helpers_sanitize_stdout(monkeypatch, capsys, helper_name, provider_label):
    def fail_search(*_args, **_kwargs):
        raise RuntimeError(_LEAKY_ERROR)

    monkeypatch.setattr(ws, "perform_websearch", fail_search)

    getattr(ws, helper_name)()

    output = capsys.readouterr().out
    assert f"Error performing {provider_label} searches" in output
    _assert_safe_text(output)
