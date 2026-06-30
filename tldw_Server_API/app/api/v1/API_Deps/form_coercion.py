from __future__ import annotations

import json
from typing import Any

from fastapi import status

from tldw_Server_API.app.core.exceptions import APIValidationError

try:
    HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT
except AttributeError:  # Starlette < 0.27
    HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_ENTITY


def coerce_form_bool(value: Any) -> bool:
    if hasattr(value, "default"):
        value = value.default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def coerce_form_string(value: Any) -> str | None:
    if hasattr(value, "default"):
        value = value.default
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def coerce_hierarchical_template(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "default"):
        value = value.default
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise APIValidationError(
                detail=[
                    {
                        "loc": ["body", "hierarchical_template"],
                        "msg": "hierarchical_template must be a JSON object",
                        "type": "value_error.jsondecode",
                    }
                ],
                status_code=HTTP_422_UNPROCESSABLE,
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise APIValidationError(
        detail=[
            {
                "loc": ["body", "hierarchical_template"],
                "msg": "hierarchical_template must be a JSON object",
                "type": "type_error.dict",
            }
        ],
        status_code=HTTP_422_UNPROCESSABLE,
    )


def chunking_contract_kwargs(
    *,
    chunking_mode: Any,
    auto_chunking_goal: Any,
    auto_chunking_use_llm: Any,
    auto_apply_template: Any,
    chunking_template_name: Any,
    hierarchical_chunking: Any,
    hierarchical_template: Any,
) -> dict[str, Any]:
    return {
        "chunking_mode": coerce_form_string(chunking_mode),
        "auto_chunking_goal": coerce_form_string(auto_chunking_goal) or "balanced",
        "auto_chunking_use_llm": coerce_form_bool(auto_chunking_use_llm),
        "auto_apply_template": coerce_form_bool(auto_apply_template),
        "chunking_template_name": coerce_form_string(chunking_template_name),
        "hierarchical_chunking": coerce_form_bool(hierarchical_chunking),
        "hierarchical_template": coerce_hierarchical_template(hierarchical_template),
    }
