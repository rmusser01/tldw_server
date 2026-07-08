"""Tests for the cooking recipe-card MCP module."""

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified import MCPServer
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.cooking_module import (
    CookingModule,
)

pytestmark = pytest.mark.unit


def _module() -> CookingModule:
    """Create a cooking module with minimal config."""
    return CookingModule(ModuleConfig(name="cooking"))


def _minimal_recipe() -> dict[str, Any]:
    """Return the smallest valid recipe arguments."""
    return {
        "title": "Toast",
        "servings": {"value": 2},
        "ingredients": [{"display": "2 slices bread"}],
        "steps": [{"display": "Toast the bread."}],
    }


@pytest.mark.asyncio
async def test_get_tools_exposes_recipe_card_render_contract() -> None:
    """The cooking tool advertises the bounded recipe-card schema."""
    tools = await _module().get_tools()

    assert [tool["name"] for tool in tools] == ["cooking.recipe_card.render"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["title", "servings", "ingredients", "steps"]  # nosec B101
    quantity_schema = tool["inputSchema"]["properties"]["ingredients"]["items"]["properties"]["quantity"]
    assert quantity_schema["exclusiveMinimum"] == 0  # nosec B101
    assert quantity_schema["maximum"] == 100000  # nosec B101
    assert tool["inputSchema"]["properties"]["notes"]["type"] == ["array", "null"]  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101
    assert tool["metadata"]["category"] == "cooking"  # nosec B101


@pytest.mark.asyncio
async def test_server_registers_cooking_module_from_default_yaml(monkeypatch) -> None:
    """The default MCP module config registers the cooking module."""
    monkeypatch.delenv("MCP_MODULES_CONFIG", raising=False)
    monkeypatch.delenv("MCP_MODULES", raising=False)

    registrations: dict[str, type[Any]] = {}
    server = MCPServer()

    async def _register_module(module_id: str, cls: type[Any], config: ModuleConfig) -> None:
        """Capture default module registrations."""
        del config
        registrations[module_id] = cls

    monkeypatch.setattr(server.module_registry, "register_module", _register_module)

    await server._register_default_modules()

    assert registrations["cooking"].__name__ == "CookingModule"  # nosec B101


@pytest.mark.asyncio
async def test_full_recipe_returns_recipe_card_payload() -> None:
    """A complete recipe returns a typed tldw_ui recipe-card payload."""
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
    """Missing optional fields normalize to stable empty/null values."""
    result = await _module().execute_tool("cooking.recipe_card.render", _minimal_recipe())

    recipe = result["tldw_ui"]["recipe"]
    assert recipe["summary"] is None  # nosec B101
    assert recipe["notes"] == []  # nosec B101
    assert recipe["servings"]["label"] == "2 servings"  # nosec B101


@pytest.mark.asyncio
async def test_null_notes_normalize_to_empty_list() -> None:
    """Explicit JSON null notes are accepted as no notes."""
    arguments = _minimal_recipe()
    arguments["notes"] = None

    result = await _module().execute_tool("cooking.recipe_card.render", arguments)

    assert result["tldw_ui"]["recipe"]["notes"] == []  # nosec B101


@pytest.mark.asyncio
async def test_recipe_text_allows_sql_comment_like_substrings() -> None:
    """Free-form recipe text may contain characters rejected by SQL sanitizers."""
    arguments = _minimal_recipe()
    arguments["title"] = "Toast -- broiler style"
    arguments["ingredients"] = [{"display": "1 tbsp sauce /* optional */"}]

    result = await _module().execute_tool("cooking.recipe_card.render", arguments)

    assert result["tldw_ui"]["recipe"]["title"] == "Toast -- broiler style"  # nosec B101
    assert result["tldw_ui"]["recipe"]["ingredients"][0]["display"] == "1 tbsp sauce /* optional */"  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"title": ""},
        {"title": "   "},
        {"servings": {"value": 0}},
        {"ingredients": [{"display": "salt"}] * 61},
        {"ingredients": [{"name": "salt"}]},
        {"ingredients": [{"display": "salt", "quantity": 0}]},
        {"ingredients": [{"display": "salt", "quantity": -1}]},
        {"ingredients": [{"display": "salt", "quantity": 100001}]},
        {"steps": [{"display": "stir"}] * 41},
        {"steps": [{"display": "x" * 601}]},
        {"steps": [{"display": "Wait.", "timer_seconds": 86401}]},
    ],
)
async def test_invalid_arguments_return_structured_error(override: dict[str, Any]) -> None:
    """Invalid recipe arguments return a structured tool error."""
    arguments = _minimal_recipe()
    arguments.update(override)

    result = await _module().execute_tool("cooking.recipe_card.render", arguments)

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101
    assert result["error"]  # nosec B101


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error() -> None:
    """Unknown cooking tool names return a structured tool error."""
    result = await _module().execute_tool("cooking.nope", {})

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "unknown_tool"  # nosec B101
    assert result["error"] == "Unknown tool: cooking.nope"  # nosec B101
