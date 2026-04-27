from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chunking_templates as routes
from tldw_Server_API.app.api.v1.schemas.chunking_templates_schemas import (
    ChunkingTemplateCreate,
    ChunkingTemplateUpdate,
    TemplateConfig,
)


class _LoggerStub:
    def __init__(self):
        self.errors = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


class _ExplodingDB:
    def __getattribute__(self, name):
        if name in {
            "list_chunking_templates",
            "get_chunking_template",
            "create_chunking_template",
            "update_chunking_template",
            "delete_chunking_template",
        }:
            raise RuntimeError("chunking template backend exploded at /private/chunking-templates.db")
        return super().__getattribute__(name)


class _ApplyTemplateDB:
    def get_chunking_template(self, *, name):
        return {
            "name": name,
            "description": "demo",
            "tags": ["demo"],
            "template_json": '{"chunking": {"method": "words", "config": {}}}',
            "version": 1,
            "is_builtin": False,
        }

    def list_chunking_templates(self, **_kwargs):
        return []


def _template_config() -> TemplateConfig:
    return TemplateConfig(chunking={"method": "words", "config": {}})


async def _call_endpoint(route_name: str, db: _ExplodingDB):
    user = SimpleNamespace(id=1)
    if route_name == "list":
        return await routes.list_templates(current_user=user, db=db)
    if route_name == "get":
        return await routes.get_template("demo-template", current_user=user, db=db)
    if route_name == "create":
        return await routes.create_template(
            ChunkingTemplateCreate(
                name="demo-template",
                description="demo",
                tags=["demo"],
                template=_template_config(),
                user_id="1",
            ),
            current_user=user,
            db=db,
        )
    if route_name == "update":
        return await routes.update_template(
            "demo-template",
            ChunkingTemplateUpdate(description="updated"),
            current_user=user,
            db=db,
        )
    if route_name == "delete":
        return await routes.delete_template("demo-template", current_user=user, db=db)
    raise AssertionError(f"unknown route {route_name}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_name", "expected_detail", "expected_log"),
    [
        ("list", "Failed to list chunking templates", "Error listing templates"),
        ("get", "Failed to get chunking template", "Error getting template"),
        ("create", "Failed to create template", "Error creating template"),
        ("update", "Failed to update template", "Error updating template"),
        ("delete", "Failed to delete template", "Error deleting template"),
    ],
)
async def test_chunking_template_generic_failure_logs_are_sanitized(
    monkeypatch,
    route_name: str,
    expected_detail: str,
    expected_log: str,
):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await _call_endpoint(route_name, _ExplodingDB())

    assert exc_info.value.status_code == 500
    assert expected_detail in str(exc_info.value.detail)
    assert "/private/" not in str(exc_info.value.detail)
    assert logger_stub.errors == [expected_log]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]


@pytest.mark.asyncio
async def test_apply_template_failure_log_sanitizes_exception_text(monkeypatch):
    class ExplodingTemplateProcessor:
        def process_template(self, **_kwargs):
            raise RuntimeError("chunking apply exploded at /private/chunking-templates.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "logger", logger_stub)
    monkeypatch.setattr(routes, "TemplateProcessor", ExplodingTemplateProcessor)

    with pytest.raises(HTTPException) as exc_info:
        await routes.apply_template(
            routes.ApplyTemplateRequest(template_name="demo-template", text="hello world"),
            current_user=SimpleNamespace(id=1),
            db=_ApplyTemplateDB(),
        )

    assert exc_info.value.status_code == 400
    assert "chunking apply exploded" in str(exc_info.value.detail)
    assert logger_stub.errors == ["Error applying template"]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]


@pytest.mark.asyncio
async def test_validate_template_failure_log_sanitizes_exception_text(monkeypatch):
    def raise_model_validate(*_args, **_kwargs):
        raise RuntimeError("validation exploded at /private/chunking-templates.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "logger", logger_stub)
    monkeypatch.setattr(routes.TemplateConfig, "model_validate", raise_model_validate)

    response = await routes.validate_template({"chunking": {"method": "words", "config": {}}})

    assert response.valid is False
    assert response.errors is not None
    assert "validation exploded" in response.errors[0].message
    assert logger_stub.errors == ["Error validating template"]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]
