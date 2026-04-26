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
