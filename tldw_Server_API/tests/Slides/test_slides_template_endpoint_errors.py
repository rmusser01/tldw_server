import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import slides as slides_endpoints
from tldw_Server_API.app.core.Slides.slides_templates import SlidesTemplateInvalidError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_templates_sanitizes_invalid_template_error(monkeypatch):
    def _raise_invalid_templates():
        raise SlidesTemplateInvalidError("template manifest exploded")

    monkeypatch.setattr(slides_endpoints, "list_slide_templates", _raise_invalid_templates)

    with pytest.raises(HTTPException) as exc_info:
        await slides_endpoints.list_templates()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list slide templates"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_template_sanitizes_invalid_template_error(monkeypatch):
    def _raise_invalid_template(template_id):
        assert template_id == "broken-template"
        raise SlidesTemplateInvalidError("template manifest exploded")

    monkeypatch.setattr(slides_endpoints, "get_slide_template", _raise_invalid_template)

    with pytest.raises(HTTPException) as exc_info:
        await slides_endpoints.get_template("broken-template")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get slide template"
