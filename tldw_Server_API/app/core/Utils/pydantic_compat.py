"""
Pydantic compatibility helpers shared across the codebase.

Provides utilities to work with both Pydantic v1 and v2 models without relying
on deprecated APIs such as ``BaseModel.dict``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from fastapi.encoders import jsonable_encoder
from pydantic import Field as _PydField


def model_dump_compat(
    obj: Any,
    *,
    exclude_none: bool = False,
    exclude: Iterable[Any] | None = None,
    exclude_unset: bool = False,
) -> dict[str, Any]:
    """
    Convert a Pydantic model or dict-like object into a plain dictionary.

    Prefers ``model_dump`` when available (Pydantic v2), falls back to
    ``model_dump_json`` or ``jsonable_encoder`` to maintain compatibility with
    Pydantic v1 models as well as dataclass-like inputs.
    """
    if obj is None:
        return {}

    exclude_set: set[Any] | None = set(exclude) if exclude is not None else None

    dump_method = getattr(obj, "model_dump", None)
    if callable(dump_method):
        try:
            kwargs: dict[str, Any] = {"exclude_none": exclude_none}
            if exclude_set is not None:
                kwargs["exclude"] = exclude_set
            if exclude_unset:
                kwargs["exclude_unset"] = True
            return dump_method(**kwargs)
        except TypeError:
            return dump_method()

    dump_json_method = getattr(obj, "model_dump_json", None)
    if callable(dump_json_method):
        try:
            kwargs: dict[str, Any] = {}
            if exclude_set is not None:
                kwargs["exclude"] = list(exclude_set)
            if exclude_unset:
                kwargs["exclude_unset"] = True
            payload = json.loads(dump_json_method(**kwargs))
            if isinstance(payload, dict):
                if exclude_set:
                    for key in exclude_set:
                        payload.pop(key, None)
                if exclude_none:
                    return {k: v for k, v in payload.items() if v is not None}
                return payload
        except Exception as json_dump_error:
            # Continue to encoder-based handling if JSON dump fails
            _ = json_dump_error

    if isinstance(obj, dict):
        data = dict(obj)
        if exclude_set:
            for key in exclude_set:
                data.pop(key, None)
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data

    encoded = jsonable_encoder(obj)
    if isinstance(encoded, dict):
        if exclude_set:
            for key in exclude_set:
                encoded.pop(key, None)
        if exclude_none:
            return {k: v for k, v in encoded.items() if v is not None}
        return encoded

    if hasattr(obj, "__dict__"):
        data = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        if exclude_set:
            for key in exclude_set:
                data.pop(key, None)
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data

    raise TypeError(f"Unsupported object for model_dump_compat: {type(obj)}")


try:  # Detect Pydantic v2 APIs
    from pydantic import model_validator as _mv  # type: ignore
    _IS_PYDANTIC_V2 = True
    del _mv
except Exception:  # pragma: no cover - conservative fallback
    _IS_PYDANTIC_V2 = False


def Field(*args: Any, **kwargs: Any):  # type: ignore
    """Compatibility Field that supports example -> json_schema_extra on v2.

    - On Pydantic v2: if an ``example=...`` kwarg is provided, it is moved into
      ``json_schema_extra={"example": ...}`` while preserving any existing
      extras.
    - On Pydantic v1: passes through unchanged.
    """
    if _IS_PYDANTIC_V2 and "example" in kwargs:
        ex = kwargs.pop("example")
        extra = kwargs.get("json_schema_extra")
        # Ensure a dict for extras; preserve existing extras when provided
        if not isinstance(extra, dict) or extra is None:
            extra = {} if extra is None else {"value": extra}
        extra.setdefault("example", ex)
        kwargs["json_schema_extra"] = extra
    return _PydField(*args, **kwargs)
