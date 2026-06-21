"""JSON-serializable model contracts for LSP code intelligence results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

JsonDict = dict[str, object]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")


def _to_dict(value: object) -> JsonValue:
    if hasattr(value, "to_dict"):
        return _to_json_value(value.to_dict())
    return _to_json_value(value)


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError("JSON value must not be NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("JSON object keys must be strings")
        for key in sorted(keys):
            normalized[key] = _to_json_value(value[key])
        return normalized
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _to_optional_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _to_string_sequence(name: str, value: Sequence[object]) -> list[str]:
    values = list(value)
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain only strings")
    return sorted(values)


@dataclass(frozen=True, slots=True)
class LspPosition:
    """Zero-based UTF-16 LSP document position."""

    line: int
    character: int

    def __post_init__(self) -> None:
        _validate_non_negative_int("line", self.line)
        _validate_non_negative_int("character", self.character)

    def to_dict(self) -> JsonDict:
        return {"line": _to_dict(self.line), "character": _to_dict(self.character)}


@dataclass(frozen=True, slots=True)
class LspRange:
    start: LspPosition
    end: LspPosition

    def to_dict(self) -> JsonDict:
        return {"start": _to_dict(self.start), "end": _to_dict(self.end)}


@dataclass(frozen=True, slots=True)
class LspLocation:
    path: str
    range: LspRange

    def to_dict(self) -> JsonDict:
        return {"path": _to_dict(self.path), "range": _to_dict(self.range)}


@dataclass(frozen=True, slots=True)
class LspDiagnostic:
    path: str
    range: LspRange
    message: str
    severity: str | None = None
    code: str | int | None = None
    source: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "path": _to_dict(self.path),
            "range": _to_dict(self.range),
            "message": _to_dict(self.message),
            "severity": _to_dict(self.severity),
            "code": _to_dict(self.code),
            "source": _to_dict(self.source),
        }


@dataclass(frozen=True, slots=True)
class LspSymbol:
    name: str
    kind: str
    location: LspLocation
    container_name: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "name": _to_dict(self.name),
            "kind": _to_dict(self.kind),
            "location": _to_dict(self.location),
            "container_name": _to_dict(self.container_name),
        }


@dataclass(frozen=True, slots=True)
class LspHover:
    contents: str
    range: LspRange | None = None
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "contents": _to_dict(self.contents),
            "range": _to_dict(self.range) if self.range else None,
            "truncated": _to_dict(self.truncated),
        }


@dataclass(frozen=True, slots=True)
class LspSignatureHelp:
    signatures: Sequence[str]
    active_signature: int | None = None
    active_parameter: int | None = None

    def to_dict(self) -> JsonDict:
        return {
            "signatures": _to_dict(list(self.signatures)),
            "active_signature": _to_dict(self.active_signature),
            "active_parameter": _to_dict(self.active_parameter),
        }


@dataclass(frozen=True, slots=True)
class LspBackendStatus:
    name: str
    healthy: bool
    capabilities: Sequence[str] = field(default_factory=tuple)
    version: str | None = None
    detail: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "name": _to_optional_string("name", self.name),
            "healthy": _to_dict(self.healthy),
            "capabilities": _to_string_sequence("capabilities", self.capabilities),
            "version": _to_optional_string("version", self.version),
            "detail": _to_optional_string("detail", self.detail),
        }


@dataclass(frozen=True, slots=True)
class LspTextEdit:
    range: LspRange
    new_text: str

    def to_dict(self) -> JsonDict:
        return {"range": _to_dict(self.range), "new_text": _to_dict(self.new_text)}


@dataclass(frozen=True, slots=True)
class LspPreview:
    path: str
    text_edits: Sequence[LspTextEdit] = field(default_factory=tuple)
    preview: str | None = None
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "path": _to_dict(self.path),
            "text_edits": [_to_dict(edit) for edit in self.text_edits],
            "preview": _to_dict(self.preview),
            "truncated": _to_dict(self.truncated),
        }


@dataclass(frozen=True, slots=True)
class LspCodeAction:
    title: str
    kind: str | None = None
    diagnostics: Sequence[LspDiagnostic] = field(default_factory=tuple)
    edits: Sequence[LspTextEdit] = field(default_factory=tuple)
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "title": _to_dict(self.title),
            "kind": _to_dict(self.kind),
            "diagnostics": [_to_dict(diagnostic) for diagnostic in self.diagnostics],
            "edits": [_to_dict(edit) for edit in self.edits],
            "data": _to_dict(self.data),
        }


@dataclass(frozen=True, slots=True)
class LspDiagnosticsResult:
    diagnostics: Sequence[LspDiagnostic]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "diagnostics": [_to_dict(diagnostic) for diagnostic in self.diagnostics],
            "truncated": _to_dict(self.truncated),
        }


@dataclass(frozen=True, slots=True)
class LspSymbolsResult:
    symbols: Sequence[LspSymbol]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {"symbols": [_to_dict(symbol) for symbol in self.symbols], "truncated": _to_dict(self.truncated)}


@dataclass(frozen=True, slots=True)
class LspLocationsResult:
    locations: Sequence[LspLocation]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "locations": [_to_dict(location) for location in self.locations],
            "truncated": _to_dict(self.truncated),
        }


@dataclass(frozen=True, slots=True)
class LspCodeActionsResult:
    actions: Sequence[LspCodeAction]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {"actions": [_to_dict(action) for action in self.actions], "truncated": _to_dict(self.truncated)}
