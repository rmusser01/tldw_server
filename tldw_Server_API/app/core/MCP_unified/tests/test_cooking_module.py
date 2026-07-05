import pytest

from tldw_Server_API.app.core.MCP_unified import MCPServer
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.cooking_module import (
    CookingModule,
)

pytestmark = pytest.mark.unit


def _module() -> CookingModule:
    return CookingModule(ModuleConfig(name="cooking"))


def _minimal_recipe() -> dict:
    return {
        "title": "Toast",
        "servings": {"value": 2},
        "ingredients": [{"display": "2 slices bread"}],
        "steps": [{"display": "Toast the bread."}],
    }


@pytest.mark.asyncio
async def test_get_tools_exposes_recipe_card_render_contract() -> None:
    tools = await _module().get_tools()

    assert [tool["name"] for tool in tools] == ["cooking.recipe_card.render"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["title", "servings", "ingredients", "steps"]  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101
    assert tool["metadata"]["category"] == "cooking"  # nosec B101


@pytest.mark.asyncio
async def test_server_registers_cooking_module_from_default_yaml(monkeypatch) -> None:
    monkeypatch.delenv("MCP_MODULES_CONFIG", raising=False)
    monkeypatch.delenv("MCP_MODULES", raising=False)

    registrations = {}
    server = MCPServer()

    async def _register_module(module_id, cls, config):
        registrations[module_id] = cls

    monkeypatch.setattr(server.module_registry, "register_module", _register_module)

    await server._register_default_modules()

    assert registrations["cooking"].__name__ == "CookingModule"  # nosec B101


@pytest.mark.asyncio
async def test_full_recipe_returns_recipe_card_payload() -> None:
    result = await _module().execute_tool(
        "cooking.recipe_card.render",
        {
            "title": "Cajun Alfredo Sauce",
            "servings": {"value": 2, "label": "2 servings"},
            "ingredients": [
                {
                    "display": "3 tbsp butter",
                    "name": "butter",
                    "quantity": 3,
                    "unit": "tbsp",
                    "note": None,
                    "scalable": True,
                }
            ],
            "steps": [
                {"display": "Melt butter in a pan over medium heat.", "timer_seconds": None}
            ],
            "summary": "Rich Cajun-style Alfredo sauce for pasta.",
            "notes": ["Add pasta water slowly until the sauce loosens."],
        },
    )

    recipe = result["tldw_ui"]["recipe"]
    assert result["tldw_ui"]["kind"] == "recipe_card"  # nosec B101
    assert result["tldw_ui"]["version"] == 1  # nosec B101
    assert recipe["title"] == "Cajun Alfredo Sauce"  # nosec B101
    assert recipe["ingredients"][0]["display"] == "3 tbsp butter"  # nosec B101
    assert recipe["ingredients"][0]["quantity"] == 3  # nosec B101


@pytest.mark.asyncio
async def test_minimal_recipe_normalizes_optional_fields() -> None:
    result = await _module().execute_tool("cooking.recipe_card.render", _minimal_recipe())

    recipe = result["tldw_ui"]["recipe"]
    assert recipe["summary"] is None  # nosec B101
    assert recipe["notes"] == []  # nosec B101
    assert recipe["servings"]["label"] == "2 servings"  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"title": ""},
        {"title": "   "},
        {"title": "Toast -- comment"},
        {"servings": {"value": 0}},
        {"ingredients": [{"display": "salt"}] * 61},
        {"ingredients": [{"name": "salt"}]},
        {"steps": [{"display": "stir"}] * 41},
        {"steps": [{"display": "x" * 601}]},
        {"steps": [{"display": "Wait.", "timer_seconds": 86401}]},
    ],
)
async def test_invalid_arguments_return_structured_error(override: dict) -> None:
    arguments = _minimal_recipe()
    arguments.update(override)

    result = await _module().execute_tool("cooking.recipe_card.render", arguments)

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101
    assert result["error"]  # nosec B101


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error() -> None:
    result = await _module().execute_tool("cooking.nope", {})

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "unknown_tool"  # nosec B101
    assert result["error"] == "Unknown tool: cooking.nope"  # nosec B101
