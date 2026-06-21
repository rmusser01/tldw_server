"""JSON-serializable model contracts for LSP code intelligence results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


JsonDict = dict[str, object]


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")


def _to_dict(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    if isinstance(value, Mapping):
        return {str(key): _to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LspPosition:
    """Zero-based UTF-16 LSP document position."""

    line: int
    character: int

    def __post_init__(self) -> None:
        _validate_non_negative_int("line", self.line)
        _validate_non_negative_int("character", self.character)

    def to_dict(self) -> JsonDict:
        return {"line": self.line, "character": self.character}


@dataclass(frozen=True, slots=True)
class LspRange:
    start: LspPosition
    end: LspPosition

    def to_dict(self) -> JsonDict:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}


@dataclass(frozen=True, slots=True)
class LspLocation:
    path: str
    range: LspRange

    def to_dict(self) -> JsonDict:
        return {"path": self.path, "range": self.range.to_dict()}


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
            "path": self.path,
            "range": self.range.to_dict(),
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class LspSymbol:
    name: str
    kind: str
    location: LspLocation
    container_name: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "kind": self.kind,
            "location": self.location.to_dict(),
            "container_name": self.container_name,
        }


@dataclass(frozen=True, slots=True)
class LspHover:
    contents: str
    range: LspRange | None = None
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "contents": self.contents,
            "range": self.range.to_dict() if self.range else None,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class LspSignatureHelp:
    signatures: Sequence[str]
    active_signature: int | None = None
    active_parameter: int | None = None

    def to_dict(self) -> JsonDict:
        return {
            "signatures": list(self.signatures),
            "active_signature": self.active_signature,
            "active_parameter": self.active_parameter,
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
            "name": self.name,
            "healthy": self.healthy,
            "capabilities": sorted(self.capabilities),
            "version": self.version,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class LspTextEdit:
    range: LspRange
    new_text: str

    def to_dict(self) -> JsonDict:
        return {"range": self.range.to_dict(), "new_text": self.new_text}


@dataclass(frozen=True, slots=True)
class LspPreview:
    path: str
    text_edits: Sequence[LspTextEdit] = field(default_factory=tuple)
    preview: str | None = None
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "path": self.path,
            "text_edits": [_to_dict(edit) for edit in self.text_edits],
            "preview": self.preview,
            "truncated": self.truncated,
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
            "title": self.title,
            "kind": self.kind,
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
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class LspSymbolsResult:
    symbols: Sequence[LspSymbol]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {"symbols": [_to_dict(symbol) for symbol in self.symbols], "truncated": self.truncated}


@dataclass(frozen=True, slots=True)
class LspLocationsResult:
    locations: Sequence[LspLocation]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "locations": [_to_dict(location) for location in self.locations],
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class LspCodeActionsResult:
    actions: Sequence[LspCodeAction]
    truncated: bool = False

    def to_dict(self) -> JsonDict:
        return {"actions": [_to_dict(action) for action in self.actions], "truncated": self.truncated}
