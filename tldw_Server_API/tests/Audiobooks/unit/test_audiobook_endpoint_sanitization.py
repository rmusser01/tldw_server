from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints.audio import audiobooks as ab
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.warning_messages: list[str] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_messages.append(message.format(*args) if args else message)


class _SubtitleCollectionsDb:
    def __init__(
        self,
        *,
        fail_usage: bool = False,
        fail_artifact_link: bool = False,
    ) -> None:
        self.fail_usage = fail_usage
        self.fail_artifact_link = fail_artifact_link

    def get_output_artifact_by_title(
        self,
        title: str,
        *,
        format_: str,
        include_deleted: bool,
    ) -> None:
        return None

    def create_output_artifact(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(id=42, storage_path=kwargs["storage_path"])

    def update_audiobook_output_usage(self, size_bytes: int) -> None:
        if self.fail_usage:
            raise RuntimeError("subtitle usage ledger leaked /private/audiobooks/quota.db")

    def get_audiobook_project_by_project_id(self, project_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=7)

    def create_audiobook_artifact(self, **kwargs: Any) -> None:
        if self.fail_artifact_link:
            raise RuntimeError("subtitle artifact link leaked /private/audiobooks/artifacts.db")


def _subtitle_request(**kwargs: Any) -> ab.SubtitleExportRequest:
    payload: dict[str, Any] = {
        "format": "srt",
        "mode": "sentence",
        "variant": "wide",
        "persist": True,
        "alignment": {
            "engine": "kokoro",
            "sample_rate": 24000,
            "words": [
                {"word": "Hello", "start_ms": 0, "end_ms": 400},
                {"word": "world.", "start_ms": 450, "end_ms": 900},
            ],
        },
    }
    payload.update(kwargs)
    return ab.SubtitleExportRequest(**payload)


def _user() -> User:
    return User(id=1, username="tester", email="tester@example.com", is_active=True, is_admin=True)


@pytest.mark.asyncio
async def test_export_subtitles_sanitizes_usage_increment_failure_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(ab, "logger", logger)

    response = await ab.export_subtitles(
        request=_subtitle_request(),
        _current_user=_user(),
        collections_db=_SubtitleCollectionsDb(fail_usage=True),
    )

    assert response.status_code == 200
    assert logger.warning_messages == ["audiobook_quota: failed to increment subtitle usage"]
    assert "/private/audiobooks/quota.db" not in logger.warning_messages[0]


@pytest.mark.asyncio
async def test_export_subtitles_sanitizes_artifact_link_failure_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(ab, "logger", logger)

    response = await ab.export_subtitles(
        request=_subtitle_request(project_id="abk_test"),
        _current_user=_user(),
        collections_db=_SubtitleCollectionsDb(fail_artifact_link=True),
    )

    assert response.status_code == 200
    assert logger.warning_messages == ["audiobook subtitles: failed to link artifact"]
    assert "/private/audiobooks/artifacts.db" not in logger.warning_messages[0]
