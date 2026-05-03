from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ENVELOPE_FIELDS = {"success", "data", "error", "error_code", "metadata"}


def _openapi_spec(monkeypatch) -> dict[str, Any]:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "phase4-openapi-local-key-1234567890")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")

    from tldw_Server_API.app.main import app

    app.openapi_schema = None
    return app.openapi()


def _schema_for(
    spec: Mapping[str, Any],
    path: str,
    method: str,
    status_code: str,
) -> Mapping[str, Any]:
    operation = spec["paths"][path][method]
    content = operation["responses"][status_code].get("content", {})
    return content.get("application/json", {}).get("schema", {})


def _schema_for_optional_path(
    spec: Mapping[str, Any],
    path: str,
    method: str,
    status_code: str,
) -> Mapping[str, Any] | None:
    if path not in spec["paths"]:
        return None
    return _schema_for(spec, path, method, status_code)


def _schema_refers_to_response_envelope(schema: Mapping[str, Any]) -> bool:
    ref = schema.get("$ref")
    if isinstance(ref, str) and "ResponseEnvelope" in ref:
        return True

    properties = schema.get("properties")
    if isinstance(properties, Mapping) and ENVELOPE_FIELDS <= set(properties):
        return True

    for key in ("allOf", "anyOf", "oneOf"):
        candidates = schema.get(key)
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, Mapping) and _schema_refers_to_response_envelope(candidate):
                    return True

    return False


def test_shared_response_envelope_is_not_published_until_route_opt_in(monkeypatch) -> None:
    """The helper schema exists in code, but v1 OpenAPI should not expose it by default."""
    spec = _openapi_spec(monkeypatch)
    components = spec["components"]["schemas"]

    envelope_components = [
        component_name
        for component_name, schema in components.items()
        if "ResponseEnvelope" in component_name
        or (isinstance(schema, Mapping) and _schema_refers_to_response_envelope(schema))
    ]

    assert envelope_components == []


def test_provider_compatible_routes_keep_non_envelope_openapi_shapes(monkeypatch) -> None:
    """OpenAI-compatible routes must not silently switch to the shared envelope contract."""
    spec = _openapi_spec(monkeypatch)

    schemas = {
        "chat completions": _schema_for(spec, "/api/v1/chat/completions", "post", "200"),
        "embeddings": _schema_for(spec, "/api/v1/embeddings", "post", "200"),
    }
    audio_speech_schema = _schema_for_optional_path(spec, "/api/v1/audio/speech", "post", "200")
    if audio_speech_schema is not None:
        schemas["audio speech"] = audio_speech_schema

    assert schemas["embeddings"].get("$ref", "").endswith("/CreateEmbeddingResponse")
    for route_name, schema in schemas.items():
        assert not _schema_refers_to_response_envelope(schema), route_name


def test_no_content_operations_do_not_declare_json_response_bodies(monkeypatch) -> None:
    """204 routes should not advertise JSON payloads or future response envelopes."""
    spec = _openapi_spec(monkeypatch)
    offenders: list[str] = []

    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue

            response = operation.get("responses", {}).get("204")
            if not isinstance(response, Mapping):
                continue
            if "content" in response:
                offenders.append(f"{method.upper()} {path}")

    assert offenders == []


def test_openapi_auth_security_scheme_names_remain_stable(monkeypatch) -> None:
    """Frontend and docs tooling depend on stable auth scheme names."""
    spec = _openapi_spec(monkeypatch)
    security_schemes = spec["components"]["securitySchemes"]

    assert {
        "type": security_schemes["ApiKeyAuth"].get("type"),
        "in": security_schemes["ApiKeyAuth"].get("in"),
        "name": security_schemes["ApiKeyAuth"].get("name"),
    } == {"type": "apiKey", "in": "header", "name": "X-API-KEY"}
    assert {
        "type": security_schemes["BearerAuth"].get("type"),
        "scheme": security_schemes["BearerAuth"].get("scheme"),
        "bearerFormat": security_schemes["BearerAuth"].get("bearerFormat"),
    } == {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
