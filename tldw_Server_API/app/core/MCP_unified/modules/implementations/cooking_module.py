"""Cooking UI payload tools for Unified MCP."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..base import BaseModule, create_tool_definition


TOOL_NAME = "cooking.recipe_card.render"


class CookingModule(BaseModule):
    """Read-only cooking helpers for typed UI payloads."""

    async def on_initialize(self) -> None:
        """Initialize the stateless cooking module."""
        return None

    async def on_shutdown(self) -> None:
        """Shut down the stateless cooking module."""
        return None

    async def check_health(self) -> dict[str, bool]:
        """Report health for the stateless cooking module."""
        return {"initialized": True, "dependencies_ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        """Return the recipe card render tool contract."""
        return [
            create_tool_definition(
                name=TOOL_NAME,
                description="Render a typed recipe card UI payload.",
                parameters={
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 120},
                        "servings": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number", "minimum": 1, "maximum": 50},
                                "label": {"type": "string", "maxLength": 80},
                            },
                            "required": ["value"],
                        },
                        "ingredients": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 60,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "display": {"type": "string", "minLength": 1, "maxLength": 180},
                                    "name": {"type": "string", "maxLength": 120},
                                    "quantity": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": 100000,
                                    },
                                    "unit": {"type": "string", "maxLength": 32},
                                    "note": {"type": ["string", "null"], "maxLength": 160},
                                    "scalable": {"type": "boolean"},
                                },
                                "required": ["display"],
                            },
                        },
                        "steps": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 40,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "display": {"type": "string", "minLength": 1, "maxLength": 600},
                                    "timer_seconds": {
                                        "type": ["integer", "null"],
                                        "minimum": 1,
                                        "maximum": 86400,
                                    },
                                },
                                "required": ["display"],
                            },
                        },
                        "summary": {"type": ["string", "null"], "maxLength": 300},
                        "notes": {
                            "type": ["array", "null"],
                            "maxItems": 8,
                            "items": {"type": "string", "maxLength": 300},
                        },
                    },
                    "required": ["title", "servings", "ingredients", "steps"],
                },
                metadata={"readOnlyHint": True, "category": "cooking"},
            )
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        """Execute a cooking tool and return a typed UI payload or structured error."""
        del context
        if tool_name != TOOL_NAME:
            return _error("unknown_tool", f"Unknown tool: {tool_name}")

        try:
            recipe = _normalize_recipe(arguments)
        except ValueError as exc:
            return _error("invalid_arguments", str(exc))

        return {"tldw_ui": {"kind": "recipe_card", "version": 1, "recipe": recipe}}


def _error(reason_code: str, message: str) -> dict[str, Any]:
    """Build a structured tool error response."""
    return {"ok": False, "reason_code": reason_code, "error": message}


def _normalize_recipe(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize recipe arguments into the UI payload shape."""
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")

    title = _string(arguments.get("title"), "title", min_len=1, max_len=120)
    servings = _servings(arguments.get("servings"))
    ingredients = _items(arguments.get("ingredients"), "ingredients", 1, 60, _ingredient)
    steps = _items(arguments.get("steps"), "steps", 1, 40, _step)
    summary = arguments.get("summary")
    if summary is not None:
        summary = _string(summary, "summary", max_len=300)
    notes_value = arguments.get("notes")
    notes = _items(notes_value, "notes", 0, 8, _note) if notes_value is not None else []

    return {
        "title": title,
        "servings": servings,
        "ingredients": ingredients,
        "steps": steps,
        "summary": summary,
        "notes": notes,
    }


def _servings(value: Any) -> dict[str, Any]:
    """Normalize servings metadata and default the human label."""
    if not isinstance(value, dict):
        raise ValueError("servings must be an object")
    serving_value = _number(value.get("value"), "servings.value", min_value=1, max_value=50)
    label = value.get("label")
    if label is None:
        label = f"{serving_value:g} serving" if serving_value == 1 else f"{serving_value:g} servings"
    else:
        label = _string(label, "servings.label", max_len=80)
    return {"value": serving_value, "label": label}


def _ingredient(value: Any) -> dict[str, Any]:
    """Normalize one ingredient row."""
    if not isinstance(value, dict):
        raise ValueError("ingredient must be an object")
    quantity = value.get("quantity")
    if quantity is not None:
        quantity = _number(quantity, "ingredient.quantity", max_value=100000)
        if quantity <= 0:
            raise ValueError("ingredient.quantity must be greater than 0")
    scalable = value.get("scalable", False)
    if not isinstance(scalable, bool):
        raise ValueError("ingredient.scalable must be a boolean")
    return {
        "display": _string(value.get("display"), "ingredient.display", min_len=1, max_len=180),
        "name": _optional_string(value.get("name"), "ingredient.name", 120),
        "quantity": quantity,
        "unit": _optional_string(value.get("unit"), "ingredient.unit", 32),
        "note": _optional_string(value.get("note"), "ingredient.note", 160),
        "scalable": scalable,
    }


def _step(value: Any) -> dict[str, Any]:
    """Normalize one cooking step."""
    if not isinstance(value, dict):
        raise ValueError("step must be an object")
    timer_seconds = value.get("timer_seconds")
    if timer_seconds is not None:
        if not isinstance(timer_seconds, int) or isinstance(timer_seconds, bool):
            raise ValueError("step.timer_seconds must be an integer")
        if not 1 <= timer_seconds <= 86400:
            raise ValueError("step.timer_seconds must be between 1 and 86400")
    return {
        "display": _string(value.get("display"), "step.display", min_len=1, max_len=600),
        "timer_seconds": timer_seconds,
    }


def _note(value: Any) -> str:
    """Normalize one recipe note."""
    return _string(value, "note", max_len=300)


def _items(
    value: Any,
    field: str,
    min_len: int,
    max_len: int,
    normalize: Callable[[Any], Any],
) -> list[Any]:
    """Normalize a bounded list field."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if not min_len <= len(value) <= max_len:
        raise ValueError(f"{field} must contain between {min_len} and {max_len} items")
    return [normalize(item) for item in value]


def _optional_string(value: Any, field: str, max_len: int) -> str | None:
    """Normalize an optional string field."""
    if value is None:
        return None
    return _string(value, field, max_len=max_len)


def _string(value: Any, field: str, *, min_len: int = 0, max_len: int) -> str:
    """Validate and trim a bounded string."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not min_len <= len(text) <= max_len:
        raise ValueError(f"{field} must be between {min_len} and {max_len} characters")
    return text


def _number(
    value: Any,
    field: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> int | float:
    """Validate a finite numeric field with optional inclusive bounds."""
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if min_value is not None and value < min_value:
        raise ValueError(f"{field} must be at least {min_value:g}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field} must be at most {max_value:g}")
    return value
