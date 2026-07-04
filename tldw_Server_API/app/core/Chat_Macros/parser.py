"""YAML definition loading and slash-argument parsing for chat macros."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Any

import yaml
from pydantic import ValidationError

from .exceptions import MacroValidationError
from .models import MacroArgSpec, MacroDefinition


def load_macro_definition(raw: str) -> MacroDefinition:
    """Load and validate a macro definition from YAML text."""
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise MacroValidationError(f"invalid macro YAML: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise MacroValidationError("macro definition must be a YAML mapping")

    try:
        return MacroDefinition.model_validate(dict(loaded))
    except ValidationError as exc:
        raise MacroValidationError(str(exc)) from exc


def parse_macro_args(
    raw: str | None,
    arg_specs: Mapping[str, MacroArgSpec],
    *,
    max_questions: int = 8,
) -> dict[str, Any]:
    """Parse shell-style slash args and return canonical argument names."""
    values = {name: _default_value(spec) for name, spec in arg_specs.items()}
    aliases = _arg_aliases(arg_specs)

    try:
        tokens = shlex.split(raw or "")
    except ValueError as exc:
        raise MacroValidationError(f"invalid macro arguments: {exc}") from exc

    index = 0
    seen_names: set[str] = set()
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise MacroValidationError(f"unexpected macro argument: {token}")

        option, has_inline_value, inline_value = token[2:].partition("=")
        name = aliases.get(option)
        if name is None:
            raise MacroValidationError(f"unknown macro argument: {option}")

        spec = arg_specs[name]
        if not spec.repeated and name in seen_names:
            raise MacroValidationError(f"duplicate macro argument: {option}")

        if spec.type == "boolean" and not has_inline_value:
            value: Any = True
        else:
            if has_inline_value:
                raw_value = inline_value
            else:
                index += 1
                if index >= len(tokens) or tokens[index].startswith("--"):
                    raise MacroValidationError(f"macro argument requires a value: {option}")
                raw_value = tokens[index]
            value = _coerce_value(raw_value, spec)

        if spec.repeated:
            values.setdefault(name, [])
            values[name].append(value)
            if name == "question" and len(values[name]) > max_questions:
                raise MacroValidationError(f"too many question arguments; max is {max_questions}")
        else:
            values[name] = value
            seen_names.add(name)
        index += 1

    return values


def _arg_aliases(arg_specs: Mapping[str, MacroArgSpec]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, spec in arg_specs.items():
        aliases[name] = name
        aliases[name.replace("_", "-")] = name
        for alias in spec.aliases:
            aliases[alias] = name
    return aliases


def _default_value(spec: MacroArgSpec) -> Any:
    if spec.repeated:
        if spec.default is None:
            return []
        return list(spec.default)
    return spec.default


def _coerce_value(raw_value: str, spec: MacroArgSpec) -> Any:
    if spec.type == "string":
        return raw_value
    if spec.type == "boolean":
        return _coerce_bool(raw_value)
    if spec.type == "integer":
        try:
            return int(raw_value)
        except ValueError as exc:
            raise MacroValidationError(f"invalid integer macro argument: {raw_value}") from exc
    if spec.type == "number":
        try:
            return float(raw_value)
        except ValueError as exc:
            raise MacroValidationError(f"invalid numeric macro argument: {raw_value}") from exc
    raise MacroValidationError(f"unsupported macro argument type: {spec.type}")


def _coerce_bool(raw_value: str) -> bool:
    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise MacroValidationError(f"invalid boolean macro argument: {raw_value}")
