import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chunking_templates
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


class _BrokenChunkingTemplatesDB:
    def list_chunking_templates(self, *args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("chunking backend exploded")

    def get_chunking_template(self, *args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("chunking backend exploded")

    def create_chunking_template(self, *args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("chunking backend exploded")


class _ExplodingTemplateRecord:
    def get(self, key, default=None):
        if key == "name":
            return "broken-template"
        if key == "is_builtin":
            return False
        return default

    def __getitem__(self, key):
        _ = key
        raise RuntimeError("chunking backend exploded")


class _DeleteExplodingTemplateRecord:
    def get(self, key, default=None):
        _ = (key, default)
        raise RuntimeError("chunking backend exploded")


class _UpdateResultExplodesDB:
    def list_chunking_templates(self, *args, **kwargs):
        _ = (args, kwargs)
        return [{"name": "broken-template", "is_builtin": False}]

    def get_chunking_template(self, *args, **kwargs):
        _ = (args, kwargs)
        return _ExplodingTemplateRecord()

    def update_chunking_template(self, *args, **kwargs):
        _ = (args, kwargs)
        return True


class _DeleteLookupExplodesDB:
    def list_chunking_templates(self, *args, **kwargs):
        _ = (args, kwargs)
        return []

    def get_chunking_template(self, *args, **kwargs):
        _ = (args, kwargs)
        return _DeleteExplodingTemplateRecord()

    def delete_chunking_template(self, *args, **kwargs):
        _ = (args, kwargs)
        return True


@pytest.fixture()
def current_user():
    return User(id=1, username="chunking-test-user", email=None, is_active=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_template_sanitizes_unexpected_backend_error(current_user):
    request = chunking_templates.ChunkingTemplateCreate(
        name="broken-template",
        template=chunking_templates.TemplateConfig(
            chunking={"method": "words", "config": {"max_size": 100}},
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.create_template(
            template_data=request,
            current_user=current_user,
            db=_BrokenChunkingTemplatesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {
        "success": False,
        "error": "Failed to create template",
        "error_code": "SERVER_ERROR",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_templates_sanitizes_backend_error(current_user):
    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.list_templates(
            current_user=current_user,
            db=_BrokenChunkingTemplatesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list chunking templates"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_template_sanitizes_backend_error(current_user):
    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.get_template(
            template_name="broken-template",
            current_user=current_user,
            db=_BrokenChunkingTemplatesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get chunking template"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_match_templates_sanitizes_backend_error(current_user):
    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.match_templates(
            media_type="text/plain",
            current_user=current_user,
            db=_BrokenChunkingTemplatesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to match chunking templates"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_learn_template_sanitizes_backend_error(current_user):
    request = chunking_templates.LearnTemplateRequest(
        name="learned-template",
        example_text="A sentence. Another sentence.",
        save=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.learn_template(
            req=request,
            current_user=current_user,
            db=_BrokenChunkingTemplatesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to learn chunking template"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_template_sanitizes_unexpected_backend_error(current_user):
    request = chunking_templates.ChunkingTemplateUpdate(description="updated")

    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.update_template(
            template_name="broken-template",
            template_update=request,
            current_user=current_user,
            db=_UpdateResultExplodesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {
        "success": False,
        "error": "Failed to update template",
        "error_code": "SERVER_ERROR",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_template_sanitizes_unexpected_backend_error(current_user):
    with pytest.raises(HTTPException) as exc_info:
        await chunking_templates.delete_template(
            template_name="broken-template",
            current_user=current_user,
            db=_DeleteLookupExplodesDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {
        "success": False,
        "error": "Failed to delete template",
        "error_code": "SERVER_ERROR",
    }
