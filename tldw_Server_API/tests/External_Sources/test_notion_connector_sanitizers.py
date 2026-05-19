from unittest.mock import MagicMock

import pytest

import tldw_Server_API.app.core.External_Sources.notion as notion_mod
from tldw_Server_API.app.core.External_Sources.notion import NotionConnector


pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    async def aclose(self):
        return None


class _ExplodingProperties(dict):
    def __bool__(self):
        return True

    def values(self):
        raise RuntimeError("notion title exploded /private/notion.db")


def _assert_sanitized_debug(fake_logger: MagicMock, expected_message: str) -> None:
    fake_logger.debug.assert_called_once_with(expected_message)
    rendered = repr(fake_logger.debug.call_args)
    assert "notion title exploded" not in rendered
    assert "/private/notion.db" not in rendered


@pytest.mark.asyncio
async def test_list_sources_sanitizes_database_page_title_fallback_log(monkeypatch):
    fake_logger = MagicMock()

    async def _fake_afetch(**_kwargs):
        return _Response(
            {
                "results": [
                    {
                        "id": "page-1",
                        "properties": _ExplodingProperties(),
                        "last_edited_time": "2026-04-27T00:00:00Z",
                    }
                ],
                "next_cursor": None,
            }
        )

    monkeypatch.setattr(notion_mod, "logger", fake_logger)
    monkeypatch.setattr(notion_mod, "afetch", _fake_afetch)

    connector = NotionConnector(client_id="client", client_secret="secret", redirect_base="http://localhost")
    items, cursor = await connector.list_sources(
        {"access_token": "token"},
        parent_remote_id="database-1",
    )

    assert cursor is None
    assert items == [
        {
            "id": "page-1",
            "name": "page-1",
            "type": "page",
            "last_edited_time": "2026-04-27T00:00:00Z",
        }
    ]
    _assert_sanitized_debug(fake_logger, "Notion connector failed to extract page title")


@pytest.mark.asyncio
async def test_list_sources_sanitizes_search_result_title_fallback_log(monkeypatch):
    fake_logger = MagicMock()

    async def _fake_afetch(**_kwargs):
        return _Response(
            {
                "results": [
                    {
                        "object": "page",
                        "id": "page-2",
                        "properties": _ExplodingProperties(),
                        "last_edited_time": "2026-04-27T00:00:00Z",
                    }
                ],
                "next_cursor": None,
            }
        )

    monkeypatch.setattr(notion_mod, "logger", fake_logger)
    monkeypatch.setattr(notion_mod, "afetch", _fake_afetch)

    connector = NotionConnector(client_id="client", client_secret="secret", redirect_base="http://localhost")
    items, cursor = await connector.list_sources({"access_token": "token"})

    assert cursor is None
    assert items == [
        {
            "id": "page-2",
            "name": "page-2",
            "type": "page",
            "last_edited_time": "2026-04-27T00:00:00Z",
        }
    ]
    _assert_sanitized_debug(fake_logger, "Notion connector failed to extract search result title")
