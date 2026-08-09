"""Revision-aware descriptor, result, and safe-error projection."""

from __future__ import annotations

import json
import re
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .protocol_errors import (
    GatewayApplicationError,
    GatewayInvalidApplicationResult,
    GatewayResourceNotFound,
    GatewayResultTooLarge,
    GatewayToolExecutionError,
)
from .protocol_limits import GatewayLimits
from .protocol_profiles import GatewayProtocolProfile
from .protocol_validation import _JSONStructureError, _validate_json_structure
from .runtime import GatewayJSONValue

GatewayDescriptorKind: TypeAlias = Literal["tool", "resource", "resource_template", "prompt"]
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_SCHEME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*\Z")
_META_KEY_PATTERN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]*)*/)?"
    r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9_.-]*[A-Za-z0-9])?\Z"
)
_SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
_TOOL_ERROR_KEY = "io.github.rmusser01.mcp-unified/error"
_URI_TEMPLATE_VARIABLE = re.compile(
    r"(?:[A-Za-z0-9_]|%[0-9A-Fa-f]{2})+"
    r"(?:\.(?:[A-Za-z0-9_]|%[0-9A-Fa-f]{2})+)*\Z"
)
_ANNOTATION_LAST_MODIFIED_VERSIONS = frozenset({"2026-07-28", "2025-11-25", "2025-06-18"})


def _invalid() -> GatewayInvalidApplicationResult:
    return GatewayInvalidApplicationResult()


def _deterministic_json(value: object, *, max_depth: int) -> str:
    """Serialize finite JSON with stable Unicode and key ordering."""

    try:
        _validate_json_structure(value, max_depth=max_depth)
    except _JSONStructureError as exc:
        raise _invalid() from exc
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _invalid() from exc


def _json_clone(value: object, *, max_depth: int) -> GatewayJSONValue:
    """Return a detached finite JSON value with string-key validation."""

    return json.loads(_deterministic_json(value, max_depth=max_depth))


def _required_string(value: object, *, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise _invalid()
    return value


def _optional_string(value: object, *, maximum: int | None = 4_096) -> str:
    if not isinstance(value, str) or (maximum is not None and len(value) > maximum):
        raise _invalid()
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or _NAME_PATTERN.fullmatch(value) is None:
        raise _invalid()
    return value


def _normalize_uri(value: object, *, template: bool = False) -> str:
    """Validate and conservatively normalize an absolute URI identity."""

    uri = _required_string(value, maximum=2_048)
    if any(character.isspace() or ord(character) < 32 for character in uri):
        raise _invalid()
    if template:
        _validate_uri_template(uri)
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise _invalid() from exc
    if _SCHEME_PATTERN.fullmatch(parsed.scheme) is None:
        raise _invalid()
    scheme = parsed.scheme.lower()

    netloc = parsed.netloc
    authority_has_expression = template and "{" in netloc
    if authority_has_expression:
        if "@" in netloc:
            raise _invalid()
        host_literal = netloc.split(":", 1)[0]
        if scheme in {"http", "https"} and not host_literal:
            raise _invalid()
        netloc = _lower_uri_template_literals(netloc)
    else:
        try:
            port = parsed.port
        except ValueError as exc:
            raise _invalid() from exc
        if scheme in {"http", "https"} and not parsed.hostname:
            raise _invalid()
        if parsed.username is not None or parsed.password is not None:
            raise _invalid()
        if parsed.hostname is not None:
            host = parsed.hostname.lower()
            if ":" in host:
                host = f"[{host}]"
            default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
            netloc = host if port is None or default_port else f"{host}:{port}"
    normalized = urlunsplit(SplitResult(scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    if template and ("{" not in normalized or "}" not in normalized):
        # A concrete URI is still a valid zero-variable RFC 6570 template.
        return normalized
    return normalized


def _lower_uri_template_literals(value: str) -> str:
    """Lowercase authority literals while preserving expansion expressions."""

    result: list[str] = []
    in_expression = False
    for character in value:
        if character == "{":
            in_expression = True
        result.append(character if in_expression else character.lower())
        if character == "}":
            in_expression = False
    return "".join(result)


def _validate_uri_template(value: str) -> None:
    """Validate the expression grammar needed for RFC 6570 URI templates."""

    position = 0
    while position < len(value):
        if value[position] == "}":
            raise _invalid()
        if value[position] != "{":
            position += 1
            continue
        end = value.find("}", position + 1)
        if end < 0:
            raise _invalid()
        expression = value[position + 1 : end]
        if not expression or "{" in expression:
            raise _invalid()
        if expression[0] in "+#./;?&":
            expression = expression[1:]
        if not expression:
            raise _invalid()
        for variable in expression.split(","):
            if not variable:
                raise _invalid()
            if variable.endswith("*"):
                variable = variable[:-1]
            elif ":" in variable:
                variable, prefix = variable.rsplit(":", 1)
                if not prefix.isdigit() or not 1 <= len(prefix) <= 4 or prefix[0] == "0":
                    raise _invalid()
            if _URI_TEMPLATE_VARIABLE.fullmatch(variable) is None:
                raise _invalid()
        position = end + 1


def _is_reserved_meta_key(key: str) -> bool:
    if key in {_SERVER_INFO_KEY, _TOOL_ERROR_KEY}:
        return True
    if "/" not in key:
        return False
    prefix = key.split("/", 1)[0].split(".")
    return len(prefix) >= 2 and prefix[1] in {"mcp", "modelcontextprotocol"}


def _metadata(
    runtime_meta: object | None,
    profile: GatewayProtocolProfile,
    *,
    reserved_meta: Mapping[str, GatewayJSONValue] | None = None,
    gateway_owned: Mapping[str, GatewayJSONValue] | None = None,
    max_depth: int,
) -> dict[str, GatewayJSONValue]:
    """Validate vendor metadata and overwrite every gateway-owned key."""

    if runtime_meta is None:
        source: Mapping[object, object] = {}
    elif isinstance(runtime_meta, Mapping):
        source = runtime_meta
    else:
        raise _invalid()

    cloned_source = _json_clone(dict(source), max_depth=max_depth)
    if not isinstance(cloned_source, dict):
        raise _invalid()
    projected: dict[str, GatewayJSONValue] = {}
    for key, value in cloned_source.items():
        if not isinstance(key, str) or _META_KEY_PATTERN.fullmatch(key) is None:
            raise _invalid()
        if not _is_reserved_meta_key(key):
            projected[key] = value

    if profile.era == "modern" and reserved_meta:
        cloned_reserved = _json_clone(dict(reserved_meta), max_depth=max_depth)
        if not isinstance(cloned_reserved, dict):
            raise _invalid()
        for key, value in cloned_reserved.items():
            if not isinstance(key, str) or _META_KEY_PATTERN.fullmatch(key) is None:
                raise _invalid()
            projected[key] = value
    if gateway_owned:
        cloned_gateway = _json_clone(dict(gateway_owned), max_depth=max_depth)
        if not isinstance(cloned_gateway, dict):
            raise _invalid()
        projected.update(cloned_gateway)
    return projected


def _copy_optional_text(
    source: Mapping[str, object],
    target: dict[str, GatewayJSONValue],
    field: str,
) -> None:
    if field in source:
        target[field] = _optional_string(source[field])


def _annotations(
    value: object,
    profile: GatewayProtocolProfile,
    *,
    tool: bool = False,
    max_depth: int,
) -> dict[str, GatewayJSONValue]:
    cloned = _json_clone(value, max_depth=max_depth)
    if not isinstance(cloned, dict):
        raise _invalid()
    if tool and profile.version == "2024-11-05":
        return {}
    if "audience" in cloned:
        audience = cloned["audience"]
        if not isinstance(audience, list) or not all(
            isinstance(role, str) and role in {"user", "assistant"} for role in audience
        ):
            raise _invalid()
    if "priority" in cloned:
        priority = cloned["priority"]
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 0 <= priority <= 1:
            raise _invalid()
    if "lastModified" in cloned:
        if not isinstance(cloned["lastModified"], str):
            raise _invalid()
        if profile.version not in _ANNOTATION_LAST_MODIFIED_VERSIONS:
            del cloned["lastModified"]
    if tool:
        if "title" in cloned:
            cloned["title"] = _optional_string(cloned["title"])
        for field in (
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        ):
            if field in cloned and not isinstance(cloned[field], bool):
                raise _invalid()
    return cloned


def _base64_data(value: object) -> str:
    """Return a string only when it contains standard padded base64 data."""

    data = _optional_string(value, maximum=16_777_216)
    try:
        b64decode(data, validate=True)
    except (Base64Error, ValueError) as exc:
        raise _invalid() from exc
    return data


def _icons(value: object) -> list[dict[str, GatewayJSONValue]]:
    if not isinstance(value, list):
        raise _invalid()
    projected: list[dict[str, GatewayJSONValue]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise _invalid()
        icon: dict[str, GatewayJSONValue] = {"src": _normalize_uri(raw.get("src"))}
        if "mimeType" in raw:
            icon["mimeType"] = _required_string(raw["mimeType"], maximum=255)
        if "sizes" in raw:
            sizes = raw["sizes"]
            if not isinstance(sizes, list) or not all(isinstance(size, str) and 0 < len(size) <= 32 for size in sizes):
                raise _invalid()
            icon["sizes"] = list(sizes)
        if "theme" in raw:
            if raw["theme"] not in {"light", "dark"}:
                raise _invalid()
            icon["theme"] = raw["theme"]
        projected.append(icon)
    return projected


def _prompt_arguments(value: object) -> list[dict[str, GatewayJSONValue]]:
    if not isinstance(value, list):
        raise _invalid()
    projected: list[dict[str, GatewayJSONValue]] = []
    identities: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise _invalid()
        name = _required_string(raw.get("name"), maximum=128)
        if name in identities:
            raise _invalid()
        identities.add(name)
        argument: dict[str, GatewayJSONValue] = {"name": name}
        if "description" in raw:
            argument["description"] = _optional_string(raw["description"])
        if "required" in raw:
            if not isinstance(raw["required"], bool):
                raise _invalid()
            argument["required"] = raw["required"]
        projected.append(argument)
    return projected


def project_descriptor(
    kind: GatewayDescriptorKind,
    descriptor: Mapping[str, object],
    profile: GatewayProtocolProfile,
    *,
    reserved_meta: Mapping[str, GatewayJSONValue] | None = None,
    limits: GatewayLimits = GatewayLimits(),
) -> dict[str, GatewayJSONValue]:
    """Normalize and project one runtime descriptor for a protocol profile."""

    if kind not in {"tool", "resource", "resource_template", "prompt"}:
        raise ValueError("unsupported descriptor kind")
    if not isinstance(descriptor, Mapping):
        raise _invalid()

    descriptor_name = (
        _name(descriptor.get("name"))
        if kind in {"tool", "prompt"}
        else _required_string(descriptor.get("name"), maximum=512)
    )
    projected: dict[str, GatewayJSONValue] = {"name": descriptor_name}
    _copy_optional_text(descriptor, projected, "description")
    if profile.supports_titles and "title" in descriptor:
        projected["title"] = _required_string(descriptor["title"], maximum=512)
    if profile.supports_icons and "icons" in descriptor:
        projected["icons"] = _icons(descriptor["icons"])
    if "annotations" in descriptor:
        annotations = _annotations(
            descriptor["annotations"],
            profile,
            tool=kind == "tool",
            max_depth=limits.max_json_depth,
        )
        if annotations:
            projected["annotations"] = annotations

    if kind == "tool":
        input_schema = _json_clone(
            descriptor.get("inputSchema"),
            max_depth=limits.max_json_depth,
        )
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            raise _invalid()
        projected["inputSchema"] = input_schema
        if "outputSchema" in descriptor and profile.structured_content_mode != "none":
            output_schema = _json_clone(
                descriptor["outputSchema"],
                max_depth=limits.max_json_depth,
            )
            if not isinstance(output_schema, dict):
                raise _invalid()
            if profile.structured_content_mode == "any" or output_schema.get("type") == "object":
                projected["outputSchema"] = output_schema
    elif kind == "resource":
        projected["uri"] = _normalize_uri(descriptor.get("uri"))
        if "mimeType" in descriptor:
            projected["mimeType"] = _required_string(descriptor["mimeType"], maximum=255)
        if "size" in descriptor:
            size = descriptor["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise _invalid()
            projected["size"] = size
    elif kind == "resource_template":
        projected["uriTemplate"] = _normalize_uri(descriptor.get("uriTemplate"), template=True)
        if "mimeType" in descriptor:
            projected["mimeType"] = _required_string(descriptor["mimeType"], maximum=255)
    elif "arguments" in descriptor:
        projected["arguments"] = _prompt_arguments(descriptor["arguments"])

    meta = (
        _metadata(
            descriptor.get("_meta"),
            profile,
            reserved_meta=reserved_meta,
            max_depth=limits.max_json_depth,
        )
        if profile.supports_titles
        else {}
    )
    if meta:
        projected["_meta"] = meta
    return projected


def _resource_contents(
    value: object,
    profile: GatewayProtocolProfile,
    *,
    max_depth: int,
) -> dict[str, GatewayJSONValue]:
    if not isinstance(value, Mapping):
        raise _invalid()
    has_text = "text" in value
    has_blob = "blob" in value
    if has_text == has_blob:
        raise _invalid()
    projected: dict[str, GatewayJSONValue] = {
        "uri": _normalize_uri(value.get("uri")),
    }
    if "mimeType" in value:
        projected["mimeType"] = _required_string(value["mimeType"], maximum=255)
    field = "text" if has_text else "blob"
    projected[field] = (
        _optional_string(value[field], maximum=16_777_216) if field == "text" else _base64_data(value[field])
    )
    if profile.supports_titles and "_meta" in value:
        meta = _metadata(value["_meta"], profile, max_depth=max_depth)
        if meta:
            projected["_meta"] = meta
    return projected


def _content_block(
    value: object,
    profile: GatewayProtocolProfile,
    *,
    max_depth: int,
) -> dict[str, GatewayJSONValue]:
    if not isinstance(value, Mapping):
        raise _invalid()
    block_type = value.get("type")
    projected: dict[str, GatewayJSONValue]
    if block_type == "text":
        projected = {
            "type": "text",
            "text": _optional_string(value.get("text"), maximum=None),
        }
    elif block_type in {"image", "audio"}:
        if block_type == "audio" and not (profile.supports_resource_links or profile.accepts_batches):
            raise _invalid()
        projected = {
            "type": block_type,
            "data": _base64_data(value.get("data")),
            "mimeType": _required_string(value.get("mimeType"), maximum=255),
        }
    elif block_type == "resource":
        projected = {
            "type": "resource",
            "resource": _resource_contents(
                value.get("resource"),
                profile,
                max_depth=max_depth,
            ),
        }
    elif block_type == "resource_link" and profile.supports_resource_links:
        projected = {
            "type": "resource_link",
            "uri": _normalize_uri(value.get("uri")),
            "name": _required_string(value.get("name"), maximum=512),
        }
        _copy_optional_text(value, projected, "description")
        if profile.supports_titles and "title" in value:
            projected["title"] = _required_string(value["title"], maximum=512)
        if profile.supports_icons and "icons" in value:
            projected["icons"] = _icons(value["icons"])
        if "mimeType" in value:
            projected["mimeType"] = _required_string(value["mimeType"], maximum=255)
        if "size" in value:
            size = value["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise _invalid()
            projected["size"] = size
    else:
        raise _invalid()
    if "annotations" in value:
        projected["annotations"] = _annotations(
            value["annotations"],
            profile,
            max_depth=max_depth,
        )
    if profile.supports_titles and "_meta" in value:
        meta = _metadata(value["_meta"], profile, max_depth=max_depth)
        if meta:
            projected["_meta"] = meta
    return projected


def _content_blocks(
    value: object,
    profile: GatewayProtocolProfile,
    *,
    max_depth: int,
) -> list[dict[str, GatewayJSONValue]]:
    if not isinstance(value, list):
        raise _invalid()
    return [_content_block(block, profile, max_depth=max_depth) for block in value]


def _apply_complete_result(
    projected: dict[str, GatewayJSONValue],
    profile: GatewayProtocolProfile,
) -> None:
    if profile.requires_result_type:
        projected["resultType"] = "complete"


def project_tool_result(
    result: GatewayJSONValue | GatewayToolExecutionError,
    profile: GatewayProtocolProfile,
    *,
    content: Sequence[Mapping[str, object]] | None = None,
    metadata: Mapping[str, GatewayJSONValue] | None = None,
    reserved_meta: Mapping[str, GatewayJSONValue] | None = None,
    limits: GatewayLimits = GatewayLimits(),
) -> dict[str, GatewayJSONValue]:
    """Project one raw tool value or safe typed tool failure."""

    if isinstance(result, GatewayToolExecutionError):
        projected: dict[str, GatewayJSONValue] = {
            "content": [{"type": "text", "text": result.public_message}],
            "isError": True,
        }
        gateway_owned = {
            _TOOL_ERROR_KEY: {
                "reasonCode": result.reason_code,
                "kind": result.kind,
            }
        }
    else:
        structured = _json_clone(result, max_depth=limits.max_json_depth)
        if content is None:
            blocks = [
                {
                    "type": "text",
                    "text": _deterministic_json(
                        structured,
                        max_depth=limits.max_json_depth,
                    ),
                }
            ]
        else:
            blocks = _content_blocks(
                list(content),
                profile,
                max_depth=limits.max_json_depth,
            )
        projected = {"content": blocks}
        if profile.structured_content_mode == "any" or (
            profile.structured_content_mode == "object" and isinstance(structured, dict)
        ):
            projected["structuredContent"] = structured
        gateway_owned = None

    _apply_complete_result(projected, profile)
    meta = _metadata(
        metadata,
        profile,
        reserved_meta=reserved_meta,
        gateway_owned=gateway_owned,
        max_depth=limits.max_json_depth,
    )
    if meta:
        projected["_meta"] = meta
    return projected


def project_resource_result(
    result: Mapping[str, object],
    profile: GatewayProtocolProfile,
    *,
    reserved_meta: Mapping[str, GatewayJSONValue] | None = None,
    limits: GatewayLimits = GatewayLimits(),
) -> dict[str, GatewayJSONValue]:
    """Validate and project a resource read result."""

    if not isinstance(result, Mapping) or not isinstance(result.get("contents"), list):
        raise _invalid()
    projected: dict[str, GatewayJSONValue] = {
        "contents": [
            _resource_contents(
                item,
                profile,
                max_depth=limits.max_json_depth,
            )
            for item in result["contents"]
        ]
    }
    if profile.cache_hints:
        projected["ttlMs"] = 0
        projected["cacheScope"] = "private"
    _apply_complete_result(projected, profile)
    meta = _metadata(
        result.get("_meta"),
        profile,
        reserved_meta=reserved_meta,
        max_depth=limits.max_json_depth,
    )
    if meta:
        projected["_meta"] = meta
    return projected


def project_prompt_result(
    result: Mapping[str, object],
    profile: GatewayProtocolProfile,
    *,
    reserved_meta: Mapping[str, GatewayJSONValue] | None = None,
    limits: GatewayLimits = GatewayLimits(),
) -> dict[str, GatewayJSONValue]:
    """Validate and project a prompt result and its message roles/content."""

    if not isinstance(result, Mapping) or not isinstance(result.get("messages"), list):
        raise _invalid()
    messages: list[dict[str, GatewayJSONValue]] = []
    for raw in result["messages"]:
        if not isinstance(raw, Mapping) or raw.get("role") not in {"user", "assistant"}:
            raise _invalid()
        messages.append(
            {
                "role": raw["role"],
                "content": _content_block(
                    raw.get("content"),
                    profile,
                    max_depth=limits.max_json_depth,
                ),
            }
        )
    projected: dict[str, GatewayJSONValue] = {"messages": messages}
    _copy_optional_text(result, projected, "description")
    _apply_complete_result(projected, profile)
    meta = _metadata(
        result.get("_meta"),
        profile,
        reserved_meta=reserved_meta,
        max_depth=limits.max_json_depth,
    )
    if meta:
        projected["_meta"] = meta
    return projected


def project_application_error(
    error: GatewayApplicationError,
    profile: GatewayProtocolProfile,
    *,
    limits: GatewayLimits = GatewayLimits(),
) -> dict[str, GatewayJSONValue]:
    """Project only allowlisted fields from a safe application exception."""

    del limits
    if isinstance(error, GatewayInvalidApplicationResult):
        return {"code": -32603, "message": "Internal error"}
    data: dict[str, GatewayJSONValue] = {
        "reasonCode": error.reason_code,
        "kind": error.kind,
    }
    if isinstance(error, GatewayResourceNotFound):
        code = profile.missing_resource_code
    elif isinstance(error, GatewayResultTooLarge):
        code = -33001
        data["limitBytes"] = error.limit_bytes
    else:
        code = -33002
    return {"code": code, "message": error.public_message, "data": data}


__all__ = [
    "GatewayDescriptorKind",
    "project_application_error",
    "project_descriptor",
    "project_prompt_result",
    "project_resource_result",
    "project_tool_result",
]
