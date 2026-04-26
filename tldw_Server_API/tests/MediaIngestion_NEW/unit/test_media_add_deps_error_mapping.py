import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import media_add_deps


@pytest.mark.asyncio
async def test_get_add_media_form_sanitizes_unexpected_form_error(monkeypatch):
    def _raise_form_error(**_kwargs):
        raise RuntimeError("form backend exploded")

    monkeypatch.setattr(media_add_deps, "AddMediaForm", _raise_form_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_add_deps.get_add_media_form(media_type="video", transcription_model=None)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during form processing"
