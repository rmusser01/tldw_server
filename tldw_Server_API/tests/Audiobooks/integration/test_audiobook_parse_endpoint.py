import os
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.audio import audiobooks as audiobooks_module
from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.integration


SIMPLE_TEXT = """Chapter 1\nThis is the first chapter.\n\nChapter 2\nSecond chapter text.\n"""

CUSTOM_PATTERN_TEXT = """Part I\nAlpha section.\n\nPart II\nBeta section.\n"""

SRT_TEXT = """1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n2\n00:00:01,500 --> 00:00:02,000\nSecond line.\n"""

TEST_MEDIA_DIR = Path(__file__).resolve().parents[2] / "Media_Ingestion_Modification" / "test_media"
SAMPLE_EPUB_PATH = TEST_MEDIA_DIR / "sample.epub"
SAMPLE_PDF_PATH = TEST_MEDIA_DIR / "sample.pdf"


def _post_parse(client, payload):
    return client.post("/api/v1/audiobooks/parse", json=payload)


def _write_temp_upload(dest_dir: Path, source_path: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source_path.name
    shutil.copyfile(source_path, dest_path)
    return dest_path.name


def _write_temp_upload_content(dest_dir: Path, filename: str, content: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_text(content, encoding="utf-8")
    return dest_path.name


def _post_parse_upload(
    client,
    *,
    input_type: str,
    upload_id: str,
    detect_chapters: bool = True,
    custom_chapter_pattern: str | None = None,
):
    payload = {
        "source": {"input_type": input_type, "upload_id": upload_id},
        "detect_chapters": detect_chapters,
    }
    if custom_chapter_pattern is not None:
        payload["custom_chapter_pattern"] = custom_chapter_pattern
    return _post_parse(client, payload)


@pytest.fixture()
def user_temp_outputs_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    return DatabasePaths.get_user_temp_outputs_dir(1)


@pytest.fixture()
def client_user_only(monkeypatch, user_temp_outputs_dir):
    """Minimal app with audiobooks router for faster, deterministic tests."""
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "false")
    # Ensure audiobooks router is enabled even if stable-only gating is active
    existing_enable = (os.getenv("ROUTES_ENABLE") or "").strip()
    enable_parts = [p for p in existing_enable.replace(" ", ",").split(",") if p]
    if "audiobooks" not in [p.lower() for p in enable_parts]:
        enable_parts.append("audiobooks")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(enable_parts))

    fastapi_app = FastAPI()
    fastapi_app.include_router(audiobooks_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    with TestClient(fastapi_app) as client:
        yield client
    fastapi_app.dependency_overrides.clear()


def test_parse_raw_text_detects_chapters(client_user_only):
    payload = {
        "source": {"input_type": "txt", "raw_text": SIMPLE_TEXT},
        "detect_chapters": True,
    }
    resp = _post_parse(client_user_only, payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["normalized_text"].startswith("Chapter 1")
    assert len(data["chapters"]) == 2
    first = data["chapters"][0]
    assert first["chapter_id"] == "ch_001"
    assert first["title"] == "Chapter 1"
    assert first["start_offset"] < first["end_offset"]
    assert first["word_count"] > 0


def test_parse_custom_chapter_pattern(client_user_only):
    payload = {
        "source": {"input_type": "md", "raw_text": CUSTOM_PATTERN_TEXT},
        "detect_chapters": True,
        "custom_chapter_pattern": r"(?:Part)\s+[IVX]+",
    }
    resp = _post_parse(client_user_only, payload)
    assert resp.status_code == 200
    data = resp.json()
    titles = [c["title"] for c in data["chapters"]]
    assert titles == ["Part I", "Part II"]


def test_parse_chapter_detection_failure_sanitizes_warning_log(client_user_only, monkeypatch):
    logged_warnings = []

    class LoggerStub:
        def warning(self, message, *args, **kwargs):
            logged_warnings.append((message, args, kwargs))

    def _fail_detect_chapters(*args, **kwargs):
        raise RuntimeError("chapter detector exploded /private/chapter-cache")

    monkeypatch.setattr(audiobooks_module, "logger", LoggerStub())
    monkeypatch.setattr(audiobooks_module, "_detect_chapters", _fail_detect_chapters)

    payload = {
        "source": {"input_type": "txt", "raw_text": SIMPLE_TEXT},
        "detect_chapters": True,
    }
    resp = _post_parse(client_user_only, payload)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "chapter_detection_failed"
    assert logged_warnings == [("Chapter detection failed", (), {})]


def test_parse_srt_strips_timestamps(client_user_only):
    payload = {
        "source": {"input_type": "srt", "raw_text": SRT_TEXT},
        "detect_chapters": False,
    }
    resp = _post_parse(client_user_only, payload)
    assert resp.status_code == 200
    data = resp.json()
    text = data["normalized_text"]
    assert "Hello world." in text
    assert "Second line." in text
    assert "-->" not in text
    assert "00:00:00" not in text


def test_parse_tagged_text_overrides_chapters(client_user_only):
    raw = (
        "[[chapter:title=Intro]]\n"
        "Intro text.\n"
        "[[chapter:id=ch_custom]]\n"
        "[[chapter:title=Second]]\n"
        "Second text.\n"
    )
    payload = {
        "source": {"input_type": "txt", "raw_text": raw},
        "detect_chapters": True,
    }
    resp = _post_parse(client_user_only, payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "[[" not in data["normalized_text"]
    assert len(data["chapters"]) == 2
    assert data["chapters"][0]["title"] == "Intro"
    assert data["chapters"][1]["chapter_id"] == "ch_custom"
    assert "tag_markers" in data["metadata"]


def test_parse_media_id_uses_db_content(client_user_only, tmp_path):
    db_path = tmp_path / "Media_DB_v2.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test")
    media_id, _, _ = db.add_media_with_keywords(
        title="Sample Media",
        media_type="document",
        content=SIMPLE_TEXT,
        keywords=[],
    )
    assert media_id is not None

    client_user_only.app.dependency_overrides[get_media_db_for_user] = lambda: db
    try:
        payload = {
            "source": {"input_type": "txt", "media_id": media_id},
            "detect_chapters": True,
        }
        resp = _post_parse(client_user_only, payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"].get("title") == "Sample Media"
        assert len(data["chapters"]) == 2
    finally:
        client_user_only.app.dependency_overrides.pop(get_media_db_for_user, None)


def test_parse_epub_upload_id_includes_metadata_and_chapters(client_user_only, user_temp_outputs_dir):
    assert SAMPLE_EPUB_PATH.exists()
    upload_id = _write_temp_upload(user_temp_outputs_dir, SAMPLE_EPUB_PATH)
    resp = _post_parse_upload(
        client_user_only,
        input_type="epub",
        upload_id=upload_id,
        detect_chapters=True,
    )
    assert resp.status_code == 200
    data = resp.json()
    title = data["metadata"].get("title", "")
    author = data["metadata"].get("author", "")
    assert "Alice" in title
    assert "Carroll" in author
    assert len(data["chapters"]) >= 1
    assert data["chapters"][0]["title"] == "Chapter I"


def test_parse_pdf_upload_id_custom_pattern_chapters(client_user_only, user_temp_outputs_dir):
    assert SAMPLE_PDF_PATH.exists()
    upload_id = _write_temp_upload(user_temp_outputs_dir, SAMPLE_PDF_PATH)
    resp = _post_parse_upload(
        client_user_only,
        input_type="pdf",
        upload_id=upload_id,
        detect_chapters=True,
        custom_chapter_pattern=r"Page\s+\d+",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"].get("source_type") == "pdf"
    assert len(data["chapters"]) >= 1
    assert data["chapters"][0]["title"] == "Page 1"


def test_parse_vtt_upload_id_strips_timestamps(client_user_only, user_temp_outputs_dir):
    vtt_text = """WEBVTT

00:00:00.000 --> 00:00:01.000
Hello VTT.

00:00:01.500 --> 00:00:02.000
Second line.
"""
    upload_id = _write_temp_upload_content(user_temp_outputs_dir, "sample.vtt", vtt_text)
    resp = _post_parse_upload(
        client_user_only,
        input_type="vtt",
        upload_id=upload_id,
        detect_chapters=False,
    )
    assert resp.status_code == 200
    data = resp.json()
    text = data["normalized_text"]
    assert "Hello VTT." in text
    assert "Second line." in text
    assert "WEBVTT" not in text
    assert "-->" not in text


def test_parse_ass_upload_id_strips_dialogue_prefix(client_user_only, user_temp_outputs_dir):
    ass_text = """[Script Info]
Title: Example

[Events]
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello ASS
Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,Second line
"""
    upload_id = _write_temp_upload_content(user_temp_outputs_dir, "sample.ass", ass_text)
    resp = _post_parse_upload(
        client_user_only,
        input_type="ass",
        upload_id=upload_id,
        detect_chapters=False,
    )
    assert resp.status_code == 200
    data = resp.json()
    text = data["normalized_text"]
    assert "Hello ASS" in text
    assert "Second line" in text
    assert "Dialogue:" not in text
