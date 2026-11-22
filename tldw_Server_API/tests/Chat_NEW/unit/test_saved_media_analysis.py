"""Regression checks for preparing stored media for a chat completion."""
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.utils import media_analysis


def test_incident_sized_media_is_loaded_in_full(monkeypatch):
    text = "x" * 1_405_967
    db = Mock()
    lookup = Mock(return_value={"source_text": text, "source_invalid": False})
    monkeypatch.setattr(media_analysis, "get_media_source_projection", lookup)

    assert media_analysis.read_saved_media_text(db, 42, owner_user_id=7) == text
    combined = media_analysis.prepare_analysis_text("Focus on evidence", text)
    assert combined == "Focus on evidence\n\n" + text
    lookup.assert_called_once_with(
        db, 42, max_chars=2 * 1024 * 1024, owner_user_id="7", current_content_only=True, for_update=False,
    )


def test_missing_or_foreign_media_is_rejected(monkeypatch):
    db = Mock()
    monkeypatch.setattr(media_analysis, "get_media_source_projection", Mock(return_value=None))

    with pytest.raises(HTTPException) as error:
        media_analysis.read_saved_media_text(db, 42, owner_user_id=7)

    assert error.value.status_code == 404


def test_larger_media_is_rejected_by_utf8_size():
    with pytest.raises(HTTPException) as error:
        media_analysis.prepare_analysis_text("", "é" * 1_048_577)

    assert error.value.status_code == 413
    assert error.value.detail["code"] == "media_analysis_too_large"
