"""Literal profile-projection and authenticated-pagination contract tests."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema.validators import validator_for
from mcp_unified.gateway.protocol_errors import (
    GatewayInvalidApplicationResult,
    GatewayResourceNotFound,
    GatewayResultTooLarge,
    GatewayToolExecutionError,
)
from mcp_unified.gateway.protocol_limits import GatewayLimits
from mcp_unified.gateway.protocol_profiles import PROTOCOL_PROFILES

pytestmark = pytest.mark.unit

_SERVER_META = {
    "io.modelcontextprotocol/serverInfo": {
        "name": "mcp-unified",
        "version": "1.0",
    }
}
_FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "mcp_protocol"


def _projection_api() -> Any:
    from mcp_unified.gateway import protocol_projection

    return protocol_projection


def _pagination_api() -> Any:
    from mcp_unified.gateway import protocol_pagination

    return protocol_pagination


@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
def test_tool_descriptor_projection_uses_every_literal_profile_flag(version: str) -> None:
    """A newer descriptor field must not leak into an older revision."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES[version]
    descriptor = {
        "name": "lookup.weather",
        "title": "Weather lookup",
        "description": "Fetch current weather",
        "icons": [{"src": "https://example.com/icon.png", "mimeType": "image/png"}],
        "annotations": {"readOnlyHint": True},
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
        "outputSchema": {"type": "array", "items": {"type": "string"}},
        "unknownFutureField": "strip-me",
        "_meta": {
            "com.example/vendor": {"visible": True},
            "io.modelcontextprotocol/serverInfo": {"name": "forged", "version": "0"},
        },
    }

    projected = api.project_descriptor(
        "tool",
        descriptor,
        profile,
        reserved_meta=_SERVER_META,
    )

    assert projected["name"] == "lookup.weather"
    assert projected["description"] == "Fetch current weather"
    assert projected["inputSchema"] == descriptor["inputSchema"]
    assert "unknownFutureField" not in projected
    assert ("title" in projected) is profile.supports_titles
    assert ("icons" in projected) is profile.supports_icons
    assert ("outputSchema" in projected) is (profile.structured_content_mode == "any")
    if profile.supports_titles:
        assert projected["_meta"]["com.example/vendor"] == {"visible": True}
    else:
        assert "_meta" not in projected
    if profile.era == "modern":
        assert projected["_meta"]["io.modelcontextprotocol/serverInfo"] == {
            "name": "mcp-unified",
            "version": "1.0",
        }
    elif "_meta" in projected:
        assert "io.modelcontextprotocol/serverInfo" not in projected["_meta"]


def test_resource_template_and_prompt_descriptors_validate_identity_and_projection() -> None:
    """Wrong descriptor kind routing must fail literal name and URI identities."""

    api = _projection_api()
    modern = PROTOCOL_PROFILES["2026-07-28"]

    resource = api.project_descriptor(
        "resource",
        {
            "name": "guide",
            "uri": "HTTPS://EXAMPLE.COM:443/docs/guide",
            "title": "Guide",
            "mimeType": "text/markdown",
            "size": 42,
        },
        modern,
    )
    template = api.project_descriptor(
        "resource_template",
        {
            "name": "user-file",
            "uriTemplate": "file:///users/{user}/files/{name}",
            "description": "A user file",
        },
        modern,
    )
    prompt = api.project_descriptor(
        "prompt",
        {
            "name": "summarize",
            "title": "Summarize",
            "arguments": [{"name": "style", "description": "Output style", "required": True}],
        },
        modern,
    )

    assert resource == {
        "name": "guide",
        "uri": "https://example.com/docs/guide",
        "title": "Guide",
        "mimeType": "text/markdown",
        "size": 42,
    }
    assert template == {
        "name": "user-file",
        "uriTemplate": "file:///users/{user}/files/{name}",
        "description": "A user file",
    }
    assert prompt == {
        "name": "summarize",
        "title": "Summarize",
        "arguments": [{"name": "style", "description": "Output style", "required": True}],
    }


def test_human_resource_names_prompt_arguments_and_empty_content_are_valid() -> None:
    """Display names and empty protocol strings must not inherit tool-name rules."""

    api = _projection_api()
    modern = PROTOCOL_PROFILES["2026-07-28"]
    assert (
        api.project_descriptor(
            "resource",
            {"name": "User Guide", "uri": "file:///guide.txt"},
            modern,
        )["name"]
        == "User Guide"
    )
    assert api.project_descriptor(
        "prompt",
        {"name": "summarize", "arguments": [{"name": "output style"}]},
        modern,
    )["arguments"] == [{"name": "output style"}]
    assert api.project_tool_result(
        "",
        modern,
        content=[{"type": "text", "text": ""}],
    )["content"] == [{"type": "text", "text": ""}]
    assert api.project_resource_result(
        {"contents": [{"uri": "file:///empty.txt", "text": ""}]},
        modern,
    )["contents"] == [{"uri": "file:///empty.txt", "text": ""}]


@pytest.mark.parametrize(
    ("kind", "descriptor"),
    [
        ("tool", {"name": "has spaces", "inputSchema": {"type": "object"}}),
        ("tool", {"name": "valid", "inputSchema": {"type": "array"}}),
        ("resource", {"name": "valid", "uri": "not a uri"}),
        ("resource_template", {"name": "valid", "uriTemplate": "relative/{id}"}),
        ("prompt", {"name": "valid", "arguments": [{"name": ""}]}),
    ],
)
def test_invalid_descriptor_identities_fail_closed(
    kind: str,
    descriptor: dict[str, Any],
) -> None:
    """Invalid names, URIs, or input roots must not be published."""

    api = _projection_api()
    with pytest.raises(GatewayInvalidApplicationResult):
        api.project_descriptor(
            kind,
            descriptor,
            PROTOCOL_PROFILES["2026-07-28"],
        )


@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
def test_arbitrary_root_tool_result_uses_exact_profile_projection(version: str) -> None:
    """Legacy fallback must preserve data as deterministic text without a wrapper."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES[version]
    projected = api.project_tool_result(
        [3, {"z": 1, "a": 2}],
        profile,
        metadata={"com.example/trace": "safe"},
        reserved_meta=_SERVER_META,
    )

    assert projected["content"] == [{"type": "text", "text": '[3,{"a":2,"z":1}]'}]
    assert ("structuredContent" in projected) is (profile.structured_content_mode == "any")
    if profile.structured_content_mode == "any":
        assert projected["structuredContent"] == [3, {"z": 1, "a": 2}]
    assert (projected.get("resultType") == "complete") is profile.requires_result_type
    assert projected["_meta"]["com.example/trace"] == "safe"
    if profile.era == "modern":
        assert projected["_meta"]["io.modelcontextprotocol/serverInfo"] == {
            "name": "mcp-unified",
            "version": "1.0",
        }
    else:
        assert "io.modelcontextprotocol/serverInfo" not in projected["_meta"]


@pytest.mark.parametrize("version", ["2025-11-25", "2025-06-18"])
def test_object_structured_content_is_retained_only_by_object_legacy_profiles(
    version: str,
) -> None:
    """Object-capable legacy profiles must not lose valid structured content."""

    api = _projection_api()
    projected = api.project_tool_result(
        {"z": 1, "a": 2},
        PROTOCOL_PROFILES[version],
    )
    assert projected == {
        "content": [{"type": "text", "text": '{"a":2,"z":1}'}],
        "structuredContent": {"z": 1, "a": 2},
    }


def test_current_tool_result_accepts_literal_content_block_kinds() -> None:
    """Dropping a current content-block variant must fail at the projection boundary."""

    api = _projection_api()
    content = [
        {
            "type": "text",
            "text": "hello",
            "_meta": {
                "com.example/block": "safe",
                "io.modelcontextprotocol/serverInfo": {
                    "name": "forged",
                    "version": "0",
                },
            },
        },
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {"type": "audio", "data": "aGVsbG8=", "mimeType": "audio/wav"},
        {
            "type": "resource",
            "resource": {"uri": "file:///tmp/a.txt", "text": "embedded"},
        },
        {
            "type": "resource_link",
            "uri": "file:///tmp/b.txt",
            "name": "linked",
            "title": "Linked file",
            "mimeType": "text/plain",
            "icons": [{"src": "https://example.com/file.png"}],
        },
    ]
    projected = api.project_tool_result(
        {"ok": True},
        PROTOCOL_PROFILES["2026-07-28"],
        content=content,
    )
    expected = [dict(block) for block in content]
    expected[0]["_meta"] = {"com.example/block": "safe"}
    assert projected["content"] == expected


def test_audio_content_obeys_the_literal_legacy_profile_matrix() -> None:
    """The oldest revision must not receive audio added in 2025-03-26."""

    api = _projection_api()
    content = [{"type": "audio", "data": "", "mimeType": "audio/wav"}]
    assert (
        api.project_tool_result(
            None,
            PROTOCOL_PROFILES["2025-03-26"],
            content=content,
        )["content"]
        == content
    )
    with pytest.raises(GatewayInvalidApplicationResult):
        api.project_tool_result(
            None,
            PROTOCOL_PROFILES["2024-11-05"],
            content=content,
        )


def test_typed_tool_error_has_safe_authoritative_metadata() -> None:
    """A runtime must not forge or displace the stable typed error classification."""

    api = _projection_api()
    error = GatewayToolExecutionError(
        "Tool is not implemented",
        reason_code="not_implemented",
    )
    projected = api.project_tool_result(
        error,
        PROTOCOL_PROFILES["2026-07-28"],
        metadata={
            "io.github.rmusser01.mcp-unified/error": {
                "reasonCode": "forged",
                "kind": "application",
            }
        },
        reserved_meta=_SERVER_META,
    )

    assert projected == {
        "content": [{"type": "text", "text": "Tool is not implemented"}],
        "isError": True,
        "resultType": "complete",
        "_meta": {
            "io.github.rmusser01.mcp-unified/error": {
                "reasonCode": "not_implemented",
                "kind": "tool",
            },
            **_SERVER_META,
        },
    }


@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
def test_resource_and_prompt_results_use_revision_correct_fields(version: str) -> None:
    """Modern cache/result fields must appear exactly where legacy profiles omit them."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES[version]
    resource = api.project_resource_result(
        {
            "contents": [
                {
                    "uri": "file:///tmp/guide.txt",
                    "mimeType": "text/plain",
                    "text": "guide",
                }
            ],
            "ttlMs": 999,
            "cacheScope": "public",
            "resultType": "forged",
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": "forged",
                    "version": "0",
                }
            },
        },
        profile,
        reserved_meta=_SERVER_META,
    )
    prompt = api.project_prompt_result(
        {
            "description": "A prompt",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "Hello"}},
                {
                    "role": "assistant",
                    "content": {"type": "text", "text": "Hi"},
                },
            ],
        },
        profile,
        reserved_meta=_SERVER_META,
    )

    assert resource["contents"] == [
        {
            "uri": "file:///tmp/guide.txt",
            "mimeType": "text/plain",
            "text": "guide",
        }
    ]
    assert prompt["messages"] == [
        {"role": "user", "content": {"type": "text", "text": "Hello"}},
        {"role": "assistant", "content": {"type": "text", "text": "Hi"}},
    ]
    if profile.cache_hints:
        assert resource["ttlMs"] == 0
        assert resource["cacheScope"] == "private"
    else:
        assert "ttlMs" not in resource
        assert "cacheScope" not in resource
    assert (resource.get("resultType") == "complete") is profile.requires_result_type
    assert (prompt.get("resultType") == "complete") is profile.requires_result_type


@pytest.mark.parametrize(
    "result",
    [
        {"contents": [{"uri": "not uri", "text": "bad"}]},
        {"contents": [{"uri": "file:///a", "text": "x", "blob": "eA=="}]},
        {"messages": [{"role": "system", "content": {"type": "text", "text": "x"}}]},
        {"messages": [{"role": "user", "content": {"type": "unknown"}}]},
    ],
)
def test_invalid_resource_prompt_roles_and_content_fail_closed(
    result: dict[str, Any],
) -> None:
    """Invalid URI, role, or content shapes must become a generic application failure."""

    api = _projection_api()
    function = api.project_resource_result if "contents" in result else api.project_prompt_result
    with pytest.raises(GatewayInvalidApplicationResult):
        function(result, PROTOCOL_PROFILES["2026-07-28"])


@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
def test_safe_application_error_projection_uses_version_correct_resource_code(
    version: str,
) -> None:
    """Changing a profile's missing-resource code must change its projected error."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES[version]
    projected = api.project_application_error(GatewayResourceNotFound(), profile)
    assert projected == {
        "code": profile.missing_resource_code,
        "message": "Resource not found",
        "data": {"reasonCode": "resource_not_found", "kind": "resource"},
    }


def test_result_too_large_and_invalid_result_have_allowlisted_error_projection() -> None:
    """Error projection must not expose actual results or internal exception text."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    assert api.project_application_error(
        GatewayResultTooLarge(limit_bytes=123),
        profile,
    ) == {
        "code": -33001,
        "message": "Application result exceeds the configured limit",
        "data": {
            "reasonCode": "result_too_large",
            "kind": "application",
            "limitBytes": 123,
        },
    }
    assert api.project_application_error(
        GatewayInvalidApplicationResult(),
        profile,
    ) == {"code": -32603, "message": "Internal error"}


def test_projection_and_pagination_reject_non_string_json_object_keys() -> None:
    """Runtime mappings must not be normalized by silently stringifying bad keys."""

    projection = _projection_api()
    pagination = _pagination_api()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    with pytest.raises(GatewayInvalidApplicationResult):
        projection.project_tool_result({1: "value"}, profile)  # type: ignore[dict-item]
    with pytest.raises(GatewayInvalidApplicationResult):
        projection.project_tool_result(("not", "a", "json", "array"), profile)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite JSON"):
        pagination.GatewayCatalogPaginator().page(
            method="tools/list",
            profile=profile,
            items=[{"name": "tool", 1: "value"}],  # type: ignore[dict-item]
            cursor=None,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        pagination.GatewayCatalogPaginator().page(
            method="tools/list",
            profile=profile,
            items=[{"name": "tool", "value": ("not", "json")}],  # type: ignore[dict-item]
            cursor=None,
        )


def test_empty_catalog_returns_an_empty_page_without_cursor() -> None:
    """Empty runtime catalogs are valid capabilities, not unavailable servers."""

    api = _pagination_api()
    paginator = api.GatewayCatalogPaginator()
    page = paginator.page(
        method="tools/list",
        profile=PROTOCOL_PROFILES["2026-07-28"],
        items=[],
        cursor=None,
    )
    assert page == api.GatewayCatalogPage(items=[], next_cursor=None)


def test_catalog_rejects_duplicate_identity_before_pagination() -> None:
    """Duplicate identities must never be hidden on separate sorted pages."""

    api = _pagination_api()
    paginator = api.GatewayCatalogPaginator()
    with pytest.raises(ValueError, match="duplicate catalog identity"):
        paginator.page(
            method="tools/list",
            profile=PROTOCOL_PROFILES["2026-07-28"],
            items=[{"name": "same"}, {"name": "same"}],
            cursor=None,
        )


def test_catalog_enforces_max_items_before_sorting_or_hashing() -> None:
    """Aggregate limits must be enforced before expensive catalog processing."""

    api = _pagination_api()
    limits = replace(GatewayLimits(), max_catalog_items=100)
    paginator = api.GatewayCatalogPaginator(limits=limits)
    items = [{"name": f"tool-{index:03d}"} for index in range(101)]
    with pytest.raises(ValueError, match="catalog item limit"):
        paginator.page(
            method="tools/list",
            profile=PROTOCOL_PROFILES["2026-07-28"],
            items=items,
            cursor=None,
        )


def test_first_page_sorts_stably_uses_fifty_items_and_continues() -> None:
    """Runtime order must not affect the fixed initial page or continuation order."""

    api = _pagination_api()
    paginator = api.GatewayCatalogPaginator()
    items = [{"name": f"tool-{index:03d}"} for index in reversed(range(51))]
    first = paginator.page(
        method="tools/list",
        profile=PROTOCOL_PROFILES["2026-07-28"],
        items=items,
        cursor=None,
    )
    assert [item["name"] for item in first.items] == [f"tool-{index:03d}" for index in range(50)]
    assert first.next_cursor is not None

    second = paginator.page(
        method="tools/list",
        profile=PROTOCOL_PROFILES["2026-07-28"],
        items=list(reversed(items)),
        cursor=first.next_cursor,
    )
    assert second.items == [{"name": "tool-050"}]
    assert second.next_cursor is None


def test_cursor_rejects_cross_method_tamper_profile_and_catalog_changes() -> None:
    """Every cursor binding must be authenticated before a continuation is served."""

    api = _pagination_api()
    paginator = api.GatewayCatalogPaginator()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    items = [{"name": f"tool-{index:03d}"} for index in range(51)]
    cursor = paginator.page(
        method="tools/list",
        profile=profile,
        items=items,
        cursor=None,
    ).next_cursor
    assert cursor is not None

    bad_inputs = [
        ("prompts/list", profile, items, cursor),
        ("tools/list", PROTOCOL_PROFILES["2025-11-25"], items, cursor),
        ("tools/list", profile, items, cursor[:-1] + ("A" if cursor[-1] != "A" else "B")),
        ("tools/list", profile, [*items[:-1], {"name": "changed"}], cursor),
    ]
    for method, cursor_profile, catalog, value in bad_inputs:
        with pytest.raises(ValueError, match="invalid cursor"):
            paginator.page(
                method=method,
                profile=cursor_profile,
                items=catalog,
                cursor=value,
            )


def test_cursor_rejects_non_ascii_and_oversize_text_as_stable_invalid_cursor() -> None:
    """Malformed text must not escape as an encoding exception or bypass the size cap."""

    api = _pagination_api()
    paginator = api.GatewayCatalogPaginator()
    for cursor in ["é.invalid", "x" * 2_049, "\ud800.invalid"]:
        with pytest.raises(ValueError, match="invalid cursor"):
            paginator.page(
                method="tools/list",
                profile=PROTOCOL_PROFILES["2026-07-28"],
                items=[{"name": "tool"}],
                cursor=cursor,
            )


@pytest.mark.parametrize(
    ("method", "item"),
    [
        ("tools/list", {"description": "missing name"}),
        ("prompts/list", {"name": 1}),
        ("resources/list", {"uri": "not a uri"}),
        ("resources/templates/list", {"uriTemplate": "relative/{id}"}),
    ],
)
def test_catalog_requires_concrete_method_identity(
    method: str,
    item: dict[str, Any],
) -> None:
    """Missing or malformed method-specific identities must fail before hashing."""

    api = _pagination_api()
    with pytest.raises(ValueError, match="catalog identity"):
        api.GatewayCatalogPaginator().page(
            method=method,
            profile=PROTOCOL_PROFILES["2026-07-28"],
            items=[item],
            cursor=None,
        )


@pytest.mark.timeout(2)
def test_projection_and_pagination_reject_cycles_without_hanging() -> None:
    """Runtime cycles must fail before deterministic encoding begins."""

    projection = _projection_api()
    pagination = _pagination_api()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    with pytest.raises(GatewayInvalidApplicationResult):
        projection.project_tool_result(cycle, profile)
    with pytest.raises(ValueError, match="finite JSON"):
        pagination.GatewayCatalogPaginator().page(
            method="tools/list",
            profile=profile,
            items=[{"name": "cycle", "value": cycle}],
            cursor=None,
        )


def test_projection_and_pagination_translate_encoder_depth_to_bounded_errors() -> None:
    """Values deeper than policy and Python's encoder must not leak RecursionError."""

    projection = _projection_api()
    pagination = _pagination_api()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    nested: object = None
    for _ in range(1_100):
        nested = [nested]
    with pytest.raises(GatewayInvalidApplicationResult):
        projection.project_tool_result(nested, profile)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite JSON"):
        pagination.GatewayCatalogPaginator().page(
            method="tools/list",
            profile=profile,
            items=[{"name": "deep", "value": nested}],  # type: ignore[dict-item]
            cursor=None,
        )


@pytest.mark.parametrize(
    ("kind", "annotations"),
    [
        ("resource", {"audience": ["system"]}),
        ("resource", {"audience": "user"}),
        ("resource", {"priority": True}),
        ("resource", {"priority": -0.1}),
        ("resource", {"priority": 1.1}),
        ("resource", {"lastModified": 123}),
        ("tool", {"title": 123}),
        ("tool", {"readOnlyHint": "yes"}),
        ("tool", {"destructiveHint": 1}),
        ("tool", {"idempotentHint": None}),
        ("tool", {"openWorldHint": []}),
    ],
)
def test_known_annotation_fields_are_typed_and_range_checked(
    kind: str,
    annotations: dict[str, Any],
) -> None:
    """Known annotation hints must not pass through as arbitrary JSON."""

    api = _projection_api()
    descriptor: dict[str, Any] = {
        "name": "safe",
        "annotations": annotations,
    }
    if kind == "tool":
        descriptor["inputSchema"] = {"type": "object"}
    else:
        descriptor["uri"] = "file:///safe"
    with pytest.raises(GatewayInvalidApplicationResult):
        api.project_descriptor(
            kind,
            descriptor,
            PROTOCOL_PROFILES["2026-07-28"],
        )


@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
def test_annotation_projection_obeys_revision_specific_fields(version: str) -> None:
    """Tool hints and resource timestamps appear only in revisions that define them."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES[version]
    resource = api.project_descriptor(
        "resource",
        {
            "name": "safe",
            "uri": "file:///safe",
            "annotations": {
                "audience": ["user", "assistant"],
                "priority": 0.5,
                "lastModified": "2026-08-08T00:00:00Z",
            },
        },
        profile,
    )
    assert resource["annotations"]["audience"] == ["user", "assistant"]
    assert resource["annotations"]["priority"] == 0.5
    assert ("lastModified" in resource["annotations"]) is (version in {"2026-07-28", "2025-11-25", "2025-06-18"})

    tool = api.project_descriptor(
        "tool",
        {
            "name": "safe",
            "inputSchema": {"type": "object"},
            "annotations": {
                "title": "Safe tool",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        profile,
    )
    assert ("annotations" in tool) is (version != "2024-11-05")


@pytest.mark.parametrize(
    "uri_template",
    [
        "file:///users/{",
        "file:///users/}",
        "file:///users/{bad var}",
        "file:///users/{}",
        "file:///users/{id:}",
        "file:///users/{{id}}",
    ],
)
def test_resource_template_requires_valid_rfc6570_syntax(uri_template: str) -> None:
    """Malformed URI-template expressions must fail at the projection boundary."""

    with pytest.raises(GatewayInvalidApplicationResult):
        _projection_api().project_descriptor(
            "resource_template",
            {"name": "safe", "uriTemplate": uri_template},
            PROTOCOL_PROFILES["2026-07-28"],
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "image", "data": "***", "mimeType": "image/png"},
        {"type": "audio", "data": "not base64", "mimeType": "audio/wav"},
        {
            "type": "resource",
            "resource": {"uri": "file:///bad", "blob": "%%%"},
        },
    ],
)
def test_binary_content_requires_canonical_base64(payload: dict[str, Any]) -> None:
    """Image, audio, and resource blobs must contain valid base64 data."""

    with pytest.raises(GatewayInvalidApplicationResult):
        _projection_api().project_tool_result(
            None,
            PROTOCOL_PROFILES["2026-07-28"],
            content=[payload],
        )


def test_text_content_over_4096_is_preserved_and_matches_fallback() -> None:
    """Per-block text has no undocumented 4 KiB limit below aggregate result bounds."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    value = "x" * 5_000
    fallback = api.project_tool_result(value, profile)
    explicit = api.project_tool_result(
        value,
        profile,
        content=[{"type": "text", "text": json.dumps(value)}],
    )
    assert explicit["content"] == fallback["content"]
    prompt = api.project_prompt_result(
        {"messages": [{"role": "user", "content": {"type": "text", "text": value}}]},
        profile,
    )
    assert prompt["messages"][0]["content"]["text"] == value


@pytest.mark.parametrize("identity", ["has spaces", "has/slash", "has!punctuation"])
@pytest.mark.parametrize("method", ["tools/list", "prompts/list"])
def test_catalog_reuses_canonical_tool_and_prompt_name_validation(
    method: str,
    identity: str,
) -> None:
    """Catalog identities must use the same grammar as projected descriptors."""

    with pytest.raises(ValueError, match="catalog identity"):
        _pagination_api().GatewayCatalogPaginator().page(
            method=method,
            profile=PROTOCOL_PROFILES["2026-07-28"],
            items=[{"name": identity}],
            cursor=None,
        )


def test_catalog_stores_the_same_normalized_uri_used_for_sorting() -> None:
    """Published identity and fingerprint identity must be canonical and identical."""

    page = (
        _pagination_api()
        .GatewayCatalogPaginator()
        .page(
            method="resources/list",
            profile=PROTOCOL_PROFILES["2026-07-28"],
            items=[{"uri": "HTTPS://EXAMPLE.COM:443/docs"}],
            cursor=None,
        )
    )
    assert page.items == [{"uri": "https://example.com/docs"}]


@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
def test_all_projected_descriptors_and_results_match_pinned_official_schemas(
    version: str,
) -> None:
    """Every public projection shape must validate against its revision snapshot."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES[version]
    schema = json.loads((_FIXTURE_ROOT / version / "schema.json").read_text("utf-8"))
    validator = validator_for(schema)(schema)
    definitions_key = "$defs" if "$defs" in schema else "definitions"

    def validate(definition: str, value: object) -> None:
        validator.evolve(schema={"$ref": f"#/{definitions_key}/{definition}"}).validate(value)

    descriptors = {
        "Tool": api.project_descriptor(
            "tool",
            {"name": "safe", "inputSchema": {"type": "object"}},
            profile,
        ),
        "Resource": api.project_descriptor(
            "resource",
            {"name": "Safe resource", "uri": "file:///safe"},
            profile,
        ),
        "ResourceTemplate": api.project_descriptor(
            "resource_template",
            {"name": "Safe template", "uriTemplate": "file:///safe/{id}"},
            profile,
        ),
        "Prompt": api.project_descriptor("prompt", {"name": "safe"}, profile),
    }
    results = {
        "CallToolResult": api.project_tool_result({"ok": True}, profile),
        "ReadResourceResult": api.project_resource_result(
            {"contents": [{"uri": "file:///safe", "text": "safe"}]},
            profile,
        ),
        "GetPromptResult": api.project_prompt_result(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": "safe"},
                    }
                ]
            },
            profile,
        ),
    }
    for definition, value in {**descriptors, **results}.items():
        validate(definition, value)


def test_public_projection_entrypoints_expose_keyword_only_limits() -> None:
    """Callers must be able to apply connection-specific JSON depth policy."""

    api = _projection_api()
    for name in (
        "project_descriptor",
        "project_tool_result",
        "project_resource_result",
        "project_prompt_result",
        "project_application_error",
    ):
        parameter = inspect.signature(getattr(api, name)).parameters["limits"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == GatewayLimits()


def test_projection_honors_smaller_configured_depth_through_every_clone_path() -> None:
    """Descriptor, metadata, and all result projectors share the configured depth."""

    api = _projection_api()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    limits = replace(GatewayLimits(), max_json_depth=2)
    deep = {"one": {"two": {"three": 3}}}
    deep_schema = {"type": "object", "nested": deep}
    calls = [
        lambda: api.project_descriptor(
            "tool",
            {"name": "safe", "inputSchema": deep_schema},
            profile,
            limits=limits,
        ),
        lambda: api.project_tool_result(deep, profile, limits=limits),
        lambda: api.project_tool_result(
            None,
            profile,
            content=[
                {
                    "type": "text",
                    "text": "safe",
                    "annotations": deep,
                }
            ],
            limits=limits,
        ),
        lambda: api.project_resource_result(
            {"contents": [{"uri": "file:///safe", "text": "safe"}], "_meta": deep},
            profile,
            limits=limits,
        ),
        lambda: api.project_prompt_result(
            {
                "messages": [{"role": "user", "content": {"type": "text", "text": "safe"}}],
                "_meta": deep,
            },
            profile,
            limits=limits,
        ),
    ]
    for call in calls:
        with pytest.raises(GatewayInvalidApplicationResult):
            call()


def test_projection_accepts_values_within_a_larger_configured_depth() -> None:
    """The default depth must not override an explicitly larger valid policy."""

    api = _projection_api()
    nested: Any = None
    for _ in range(80):
        nested = [nested]
    projected = api.project_tool_result(
        nested,
        PROTOCOL_PROFILES["2026-07-28"],
        limits=replace(GatewayLimits(), max_json_depth=128),
    )
    assert projected["structuredContent"] == nested


@pytest.mark.parametrize(
    ("uri_template", "expected"),
    [
        (
            "https://example.com:{port}/resource",
            "https://example.com:{port}/resource",
        ),
        (
            "https://{host}:{port}/resource",
            "https://{host}:{port}/resource",
        ),
        (
            "HTTPS://EXAMPLE.COM:{port}/resource",
            "https://example.com:{port}/resource",
        ),
        (
            "https://example.com/resource{?q,lang}",
            "https://example.com/resource{?q,lang}",
        ),
        ("https://example.com/{+path}", "https://example.com/{+path}"),
        ("https://example.com/{id:3}", "https://example.com/{id:3}"),
        ("https://example.com/{list*}", "https://example.com/{list*}"),
    ],
)
def test_resource_template_accepts_authority_and_operator_expressions(
    uri_template: str,
    expected: str,
) -> None:
    """Valid RFC 6570 expressions must survive conservative URI normalization."""

    projected = _projection_api().project_descriptor(
        "resource_template",
        {"name": "safe", "uriTemplate": uri_template},
        PROTOCOL_PROFILES["2026-07-28"],
    )
    assert projected["uriTemplate"] == expected


@pytest.mark.parametrize(
    "uri_template",
    [
        "https://user:pass@example.com/{id}",
        "https://example.com:not-a-port/resource",
        "https://:{port}/resource",
        "https://example.com/{id:0001}",
    ],
)
def test_resource_template_rejects_credentials_bad_literals_and_bad_modifiers(
    uri_template: str,
) -> None:
    """Template-aware parsing must retain concrete URI security checks."""

    with pytest.raises(GatewayInvalidApplicationResult):
        _projection_api().project_descriptor(
            "resource_template",
            {"name": "safe", "uriTemplate": uri_template},
            PROTOCOL_PROFILES["2026-07-28"],
        )


@pytest.mark.parametrize(
    "uri_template",
    [
        "https://{host}:not-a-port/resource",
        "https://{host}:99999/resource",
        "https://{host}:bad{port}/resource",
        "https://{host}:-1/resource",
        "https://{host}:/resource",
        "https://{host}:1:2/resource",
        "https://example.com:bad{port}/resource",
    ],
)
def test_resource_template_rejects_invalid_literal_or_mixed_ports(
    uri_template: str,
) -> None:
    """Only a valid concrete port or one complete expression may follow the host."""

    with pytest.raises(GatewayInvalidApplicationResult):
        _projection_api().project_descriptor(
            "resource_template",
            {"name": "safe", "uriTemplate": uri_template},
            PROTOCOL_PROFILES["2026-07-28"],
        )


@pytest.mark.parametrize(
    ("uri_template", "expected"),
    [
        (
            "https://[2001:db8::1]:{port}/resource",
            "https://[2001:db8::1]:{port}/resource",
        ),
        (
            "HTTPS://[2001:DB8::1]:443/resource",
            "https://[2001:db8::1]/resource",
        ),
        (
            "https://{host}:{port}/resource{?query}",
            "https://{host}:{port}/resource{?query}",
        ),
    ],
)
def test_resource_template_preserves_valid_port_ipv6_and_operator_forms(
    uri_template: str,
    expected: str,
) -> None:
    """Port validation must preserve complete expansions and concrete IPv6 URIs."""

    projected = _projection_api().project_descriptor(
        "resource_template",
        {"name": "safe", "uriTemplate": uri_template},
        PROTOCOL_PROFILES["2026-07-28"],
    )
    assert projected["uriTemplate"] == expected
