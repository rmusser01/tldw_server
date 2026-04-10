import pytest
from unittest.mock import MagicMock


from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import CookieManager


@pytest.mark.asyncio
async def test_cookie_manager_stores_and_returns_cookies(tmp_path):
    manager = CookieManager(storage_path=tmp_path / "cookies.json")
    manager.add_cookies(
        "example.com",
        [{"name": "foo", "value": "bar"}, {"name": "session", "value": "trace-id"}],
    )
    cookies = manager.get_cookies("https://example.com/some/path")
    assert cookies == [{"name": "foo", "value": "bar"}, {"name": "session", "value": "trace-id"}]  # nosec B101
    assert manager.get_cookies("https://other.example") is None  # nosec B101


def test_process_web_scraping_endpoint_receives_custom_headers(
    client_with_single_user,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import process_web_scraping as endpoint_mod
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.main import app

    fake_db = MagicMock(spec=MediaDatabase)
    app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    mocked_result = {"status": "success", "message": "ok", "count": 0, "results": []}
    received_kwargs = {}
    payload = {
        "scrape_method": "Individual URLs",
        "url_input": "https://example.com/article",
        "max_pages": 5,
        "max_depth": 1,
        "summarize_checkbox": False,
        "mode": "ephemeral",
        "user_agent": "IntegrationAgent/1.0",
        "custom_headers": {"X-Test": "true"},
    }

    async def fake_process_web_scraping_task(**kwargs):
        received_kwargs.update(kwargs)
        return mocked_result

    monkeypatch.setattr(
        endpoint_mod,
        "_resolve_process_web_scraping_task",
        lambda: fake_process_web_scraping_task,
    )

    response = None
    try:
        response = client.post("/api/v1/media/process-web-scraping", json=payload)
    finally:
        app.dependency_overrides.pop(get_media_db_for_user, None)

    assert response is not None  # nosec B101
    assert response.status_code == 200  # nosec B101
    assert response.json() == mocked_result  # nosec B101
    assert received_kwargs["custom_headers"] == payload["custom_headers"]  # nosec B101
    assert received_kwargs["user_agent"] == payload["user_agent"]  # nosec B101
    assert received_kwargs["mode"] == payload["mode"]  # nosec B101
