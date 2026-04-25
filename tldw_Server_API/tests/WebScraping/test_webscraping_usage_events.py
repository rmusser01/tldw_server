import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_ws_overrides(monkeypatch, client_with_single_user):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media.process_web_scraping as endpoint_mod

    async def stub_process_web_scraping_task(**kwargs):
        return {"status": "ok", "results": []}

    monkeypatch.setattr(
        endpoint_mod,
        "_resolve_process_web_scraping_task",
        lambda: stub_process_web_scraping_task,
    )

    yield client, usage_logger


def test_webscrape_process_usage_event(client_with_ws_overrides):
    client, usage_logger = client_with_ws_overrides
    payload = {
        "scrape_method": "Individual URLs",
        "url_input": "https://example.com\nhttps://example.org",
        "mode": "ephemeral",
        "max_pages": 5,
        "max_depth": 2,
    }
    r = client.post("/api/v1/media/process-web-scraping", json=payload)
    assert r.status_code == 200, r.text  # nosec B101
    assert any(e[0] == "webscrape.process" for e in usage_logger.events)  # nosec B101
