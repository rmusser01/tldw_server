from __future__ import annotations

from collections.abc import Iterator
import re
from typing import Any

from fastapi import FastAPI
import pytest


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
PATH_PARAMETER_PATTERN = re.compile(r"{([^}/]+)}")
CANONICAL_PAGINATION_COMPONENT_FIELDS = {
    "OffsetPaginationMeta": {"mode", "limit", "offset", "total", "has_more", "next_offset"},
    "CursorPaginationMeta": {"mode", "limit", "cursor", "next_cursor", "has_more"},
    "PagePaginationMeta": {"mode", "page", "per_page", "total", "total_pages", "has_more"},
}
SKILLS_EXPORT_PATH = "/api/v1/skills/{skill_name}/export"
SKILLS_ITEM_PATH = "/api/v1/skills/{skill_name}"
STREAMING_RESPONSE_EXEMPTIONS = {
    ("get", "/api/v1/admin/events/stream"): "text/event-stream",
    ("get", "/api/v1/media/ingest/jobs/events/stream"): "text/event-stream",
    ("post", "/api/v1/media/mediawiki/ingest-dump"): "application/x-ndjson",
    ("post", "/api/v1/media/mediawiki/process-dump"): "application/x-ndjson",
    ("get", "/api/v1/research/runs/{session_id}/events/stream"): "text/event-stream",
    ("post", "/api/v1/rag/search/stream"): "application/x-ndjson",
}
AUDIO_SPEECH_CONTENT_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/L16",
    "audio/mpeg",
    "audio/opus",
    "audio/wav",
}
AUDIO_TRANSCRIPTION_CONTENT_TYPES = {
    "application/json",
    "application/x-subrip",
    "text/plain",
    "text/vtt",
}
AUDIO_TOKENIZER_DECODE_CONTENT_TYPES = {
    "application/octet-stream",
    "audio/wav",
}
AUDIO_TTS_HISTORY_PATH = "/api/v1/audio/history/{history_id}"
CONFIG_QUICKSTART_PATH = "/api/v1/config/quickstart"
ROOT_PATH = "/"
FEDERATION_LOGIN_PATH = "/api/v1/auth/federation/{provider_slug}/login"
OPENAI_OAUTH_CALLBACK_PATH = "/api/v1/users/keys/openai/oauth/callback"
OUTPUT_DOWNLOAD_CONTENT_TYPES = {
    "application/octet-stream",
    "audio/mpeg",
    "text/html; charset=utf-8",
    "text/markdown; charset=utf-8",
}
OUTPUT_DOWNLOAD_PATH = "/api/v1/outputs/{output_id}/download"
FILE_ARTIFACT_EXPORT_CONTENT_TYPES = {
    "application/json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "image/jpeg",
    "image/png",
    "image/webp",
    "text/calendar",
    "text/csv",
    "text/html",
    "text/markdown",
}
FILE_ARTIFACT_EXPORT_PATH = "/api/v1/files/{file_id}/export"
CHAT_COMPLETIONS_PATH = "/api/v1/chat/completions"
MESSAGES_COMPLETION_PATHS = ("/api/v1/messages", "/v1/messages")
PROMETHEUS_TEXT_RESPONSE_PATHS = ("/api/v1/metrics/text", "/api/v1/mcp/metrics/prometheus")
MCP_REQUEST_PATH = "/api/v1/mcp/request"
CLAIMS_ANALYTICS_EXPORT_DOWNLOAD_PATH = "/api/v1/claims/analytics/export/{export_id}"
PRIVILEGE_SNAPSHOT_CSV_EXPORT_PATH = "/api/v1/privileges/snapshots/{snapshot_id}/export.csv"
EVALUATIONS_ABTEST_EVENTS_PATH = "/api/v1/evaluations/embeddings/abtest/{test_id}/events"
EVALUATIONS_ABTEST_EXPORT_PATH = "/api/v1/evaluations/embeddings/abtest/{test_id}/export"
EVALUATIONS_METRICS_PATH = "/api/v1/evaluations/metrics"
STORAGE_FILE_DOWNLOAD_PATH = "/api/v1/storage/files/{file_id}/download"
JOBS_EVENTS_STREAM_PATH = "/api/v1/jobs/events/stream"
WATCHLIST_SOURCES_EXPORT_PATH = "/api/v1/watchlists/sources/export"
WATCHLIST_RUNS_CSV_EXPORT_PATH = "/api/v1/watchlists/runs/export.csv"
WATCHLIST_RUN_TALLIES_CSV_EXPORT_PATH = "/api/v1/watchlists/runs/{run_id}/tallies.csv"
WATCHLIST_OUTPUT_DOWNLOAD_PATH = "/api/v1/watchlists/outputs/{output_id}/download"
WATCHLIST_OUTPUT_DOWNLOAD_CONTENT_TYPES = {
    "audio/mpeg",
    "text/html",
    "text/markdown",
}
CHARACTER_EXPORT_PATH = "/api/v1/characters/{character_id}/export"
ADMIN_USAGE_CSV_EXPORT_PATHS = (
    "/api/v1/admin/usage/daily/export.csv",
    "/api/v1/admin/usage/top/export.csv",
    "/api/v1/admin/llm-usage/export.csv",
)
ADMIN_JSON_CSV_EXPORT_PATHS = (
    "/api/v1/admin/users/export",
    "/api/v1/admin/audit-log/export",
)
FLASHCARD_ASSET_CONTENT_PATH = "/api/v1/flashcards/assets/{asset_uuid}/content"
FLASHCARD_EXPORT_PATH = "/api/v1/flashcards/export"
FLASHCARD_EXPORT_CONTENT_TYPES = {
    "application/apkg",
    "application/json; charset=utf-8",
    "text/csv; charset=utf-8",
    "text/tab-separated-values; charset=utf-8",
}
READING_TTS_PATH = "/api/v1/reading/items/{item_id}/tts"
READING_EXPORT_PATH = "/api/v1/reading/export"
READING_EXPORT_CONTENT_TYPES = {
    "application/x-ndjson",
    "application/zip",
}
AUDIT_EXPORT_PATH = "/api/v1/audit/export"
AUDIT_EXPORT_CONTENT_TYPES = {
    "application/json",
    "application/x-ndjson",
    "text/csv",
}
MEETING_EVENTS_PATH = "/api/v1/meetings/sessions/{session_id}/events"
NOTIFICATIONS_STREAM_PATH = "/api/v1/notifications/stream"
VN_ASSET_CONTENT_PATH = "/api/v1/vn/vn-assets/packs/{pack_id}/items/{item_id}/content"
VN_POLICY_EVALUATE_PATH = "/api/v1/vn/vn-policy/evaluate"
VN_POLICY_PROFILES_PATH = "/api/v1/vn/vn-policy/profiles"
VN_POLICY_GENERATION_PROFILES_PATH = "/api/v1/vn/vn-policy/generation-profiles"
VN_SCRIPTS_PATH = "/api/v1/vn/vn-scripts/scripts"
VN_SCRIPT_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}"
VN_SCRIPT_DRAFT_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/draft"
VN_SCRIPT_DRAFT_VALIDATE_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/draft/validate"
VN_SCRIPT_DRAFT_DIAGNOSTICS_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/draft/diagnostics"
VN_SCRIPT_DRAFT_PLAYTEST_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/draft/playtest"
VN_SCRIPT_PUBLISH_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/publish"
VN_SCRIPT_VERSIONS_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/versions"
VN_SCRIPT_VERSION_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}"
VN_SCRIPT_MANIFEST_SNAPSHOT_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/manifest-snapshot"
VN_SCRIPT_VERSION_POLICY_EVALUATE_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/policy/evaluate"
VN_SCRIPT_VERSION_PLAYTEST_PATH = "/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/playtest"
SLIDES_EXPORT_PATH = "/api/v1/slides/presentations/{presentation_id}/export"
SLIDES_EXPORT_CONTENT_TYPES = {
    "application/json",
    "application/pdf",
    "application/zip",
    "text/markdown",
}
WEBSUB_VERIFY_CALLBACK_PATH = "/api/v1/websub/callback/{user_id}/{callback_token}"
WEBSUB_PUSH_CALLBACK_PATH = "/api/v1/websub/callback/{user_id}/{callback_token}"
NOTES_CSV_EXPORT_PATHS = (
    "/api/v1/notes/export.csv",
)
NOTES_ATTACHMENT_DOWNLOAD_PATH = "/api/v1/notes/{note_id}/attachments/{file_name}"
CONNECTOR_PROVIDER_WEBHOOK_PATH = "/api/v1/connectors/providers/{provider}/webhook"
WORKFLOW_ARTIFACT_DOWNLOAD_PATH = "/api/v1/workflows/artifacts/{artifact_id}/download"
WORKFLOW_RUN_ARTIFACTS_DOWNLOAD_PATH = "/api/v1/workflows/runs/{run_id}/artifacts/download"
DATA_TABLE_EXPORT_PATH = "/api/v1/data-tables/{table_uuid}/export"
DATA_TABLE_EXPORT_CONTENT_TYPES = {
    "application/json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
}
CHAT_DOCUMENT_GENERATE_PATH = "/api/v1/chat/documents/generate"
ACP_SESSION_EVENTS_STREAM_PATH = "/api/v1/acp/sessions/{session_id}/events/stream"
MCP_HUB_EVENTS_STREAM_PATH = "/api/v1/mcp/hub/events/stream"
EMBEDDINGS_ORCHESTRATOR_EVENTS_PATH = "/api/v1/embeddings/orchestrator/events"
CHARACTER_CHAT_COMPLETION_V2_PATH = "/api/v1/chats/{chat_id}/complete-v2"
PAPER_SEARCH_RAW_JSON_XML_HTML_PATHS = (
    "/api/v1/paper-search/medrxiv/raw/details",
    "/api/v1/paper-search/biorxiv/raw/details",
)
PAPER_SEARCH_RAW_JSON_CSV_PATHS = (
    "/api/v1/paper-search/medrxiv/raw/pubs",
    "/api/v1/paper-search/medrxiv/raw/pub",
    "/api/v1/paper-search/biorxiv/raw/pubs",
    "/api/v1/paper-search/biorxiv/raw/pub",
    "/api/v1/paper-search/biorxiv/raw/reports/summary",
    "/api/v1/paper-search/biorxiv/raw/reports/usage",
)
PAPER_SEARCH_RAW_JSON_XML_PATHS = (
    "/api/v1/paper-search/biorxiv/raw/funder",
)
PAPER_SEARCH_RAW_XML_PATHS = (
    "/api/v1/paper-search/chemrxiv/oai",
    "/api/v1/paper-search/zenodo/oai",
    "/api/v1/paper-search/figshare/oai",
)
PAPER_SEARCH_RAW_JSON_PATHS = (
    "/api/v1/paper-search/iacr/conf/raw",
    "/api/v1/paper-search/osf/raw",
    "/api/v1/paper-search/osf/raw/by-id",
)
PAPER_SEARCH_PMC_OA_PDF_PATH = "/api/v1/paper-search/pmc-oa/fetch-pdf"
PAPER_SEARCH_HAL_RAW_PATH = "/api/v1/paper-search/hal/raw"
MEDIA_FILE_PATH = "/api/v1/media/{media_id}/file"
NOT_MODIFIED_RESPONSE_OPERATIONS = (
    ("get", "/api/v1/users/profile/catalog"),
    ("get", "/api/v1/media/"),
    ("get", "/api/v1/media/trash"),
    ("get", "/api/v1/media/metadata-search"),
    ("get", "/api/v1/media/by-identifier"),
    ("post", "/api/v1/media/search"),
    ("get", "/api/v1/media/{media_id}"),
    ("get", MEDIA_FILE_PATH),
)


@pytest.fixture()
def openapi_spec(client_user_only) -> dict[str, Any]:
    response = client_user_only.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def _iter_internal_refs(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            yield ".".join(path + ("$ref",)), ref
        for key, value in node.items():
            yield from _iter_internal_refs(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_internal_refs(value, path + (str(index),))


def _resolve_json_pointer(document: dict[str, Any], pointer: str) -> Any:
    current: Any = document
    for raw_part in pointer.removeprefix("#/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _iter_http_operations(openapi_spec: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for route, operations in openapi_spec.get("paths", {}).items():
        if not isinstance(operations, dict):
            continue
        for method, operation in operations.items():
            if method.lower() in HTTP_METHODS and isinstance(operation, dict):
                yield route, method, operation


def _contains_ref_fragment(node: Any, fragment: str) -> bool:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and fragment in ref:
            return True
        return any(_contains_ref_fragment(value, fragment) for value in node.values())
    if isinstance(node, list):
        return any(_contains_ref_fragment(item, fragment) for item in node)
    return False


def _assert_streaming_response_content(operation: dict[str, Any], media_type: str) -> None:
    content = operation["responses"]["200"].get("content", {})

    assert media_type in content
    assert "application/json" not in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


def _assert_file_response_content(operation: dict[str, Any], media_types: set[str]) -> None:
    content = operation["responses"]["200"].get("content", {})

    assert media_types.issubset(content)
    assert "application/json" not in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


def _assert_no_body_response(operation: dict[str, Any], status_code: str = "200") -> None:
    response = operation["responses"][status_code]

    assert "content" not in response
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


def _assert_redirect_response(operation: dict[str, Any], status_code: str) -> None:
    response = operation["responses"][status_code]
    headers = {key.lower(): value for key, value in response.get("headers", {}).items()}

    assert "content" not in response
    assert "location" in headers
    assert headers["location"]["schema"]["type"] == "string"
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


def _assert_raw_response_content(operation: dict[str, Any], media_types: set[str]) -> None:
    content = operation["responses"]["200"].get("content", {})

    assert media_types.issubset(content)
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_openapi_internal_refs_resolve(openapi_spec: dict[str, Any]) -> None:
    """Every internal OpenAPI JSON pointer must resolve inside the generated spec."""
    missing: list[str] = []
    for path, ref in _iter_internal_refs(openapi_spec):
        try:
            _resolve_json_pointer(openapi_spec, ref)
        except KeyError:
            missing.append(f"{path}: {ref}")

    assert not missing, "Unresolved OpenAPI refs:\n" + "\n".join(missing[:50])


@pytest.mark.integration
def test_openapi_operation_ids_are_present_and_unique(openapi_spec: dict[str, Any]) -> None:
    """Each HTTP operation must expose one globally unique operationId for client generation."""
    seen: dict[str, str] = {}
    missing: list[str] = []
    duplicates: list[str] = []

    for route, method, operation in _iter_http_operations(openapi_spec):
        location = f"{method.upper()} {route}"
        operation_id = operation.get("operationId")
        if not isinstance(operation_id, str) or not operation_id.strip():
            missing.append(location)
            continue

        existing_location = seen.setdefault(operation_id, location)
        if existing_location != location:
            duplicates.append(f"{operation_id}: {existing_location} and {location}")

    assert not missing, "OpenAPI operations missing operationId:\n" + "\n".join(missing[:50])
    assert not duplicates, "Duplicate OpenAPI operationIds:\n" + "\n".join(duplicates[:50])


@pytest.mark.integration
def test_openapi_declares_all_operation_tags(openapi_spec: dict[str, Any]) -> None:
    """Every operation tag must be declared at the top level so docs and generators can group them."""
    declared_tags = {
        tag["name"]
        for tag in openapi_spec.get("tags", [])
        if isinstance(tag, dict) and isinstance(tag.get("name"), str)
    }
    used_tags: set[str] = set()
    for _, _, operation in _iter_http_operations(openapi_spec):
        used_tags.update(tag for tag in operation.get("tags", []) if isinstance(tag, str))

    missing_tags = sorted(used_tags - declared_tags)
    assert not missing_tags, "OpenAPI operation tags missing top-level declarations:\n" + "\n".join(missing_tags[:80])


@pytest.mark.integration
def test_custom_openapi_reuses_cached_schema_for_tag_declarations(monkeypatch) -> None:
    """Operation tag declaration normalization should run only on the first schema build."""
    from tldw_Server_API.app import main as main_module

    helper_calls = 0
    original_schema = main_module.app.openapi_schema
    original_helper = main_module._ensure_openapi_operation_tags_declared

    def counting_helper(openapi_schema: dict[str, Any]) -> None:
        nonlocal helper_calls
        helper_calls += 1
        original_helper(openapi_schema)

    monkeypatch.setattr(main_module, "_ensure_openapi_operation_tags_declared", counting_helper)
    main_module.app.openapi_schema = None
    try:
        first_schema = main_module.app.openapi()
        second_schema = main_module.app.openapi()
    finally:
        main_module.app.openapi_schema = original_schema

    assert second_schema is first_schema
    assert helper_calls == 1


@pytest.mark.integration
def test_openapi_operation_tag_declaration_ignores_malformed_tag_values() -> None:
    """Malformed scalar tag values must not be expanded into one-character top-level tags."""
    from tldw_Server_API.app import main as main_module

    openapi_schema: dict[str, Any] = {
        "tags": [],
        "paths": {
            "/valid": {"get": {"tags": ["media"]}},
            "/malformed": {"get": {"tags": "health"}},
        },
    }

    main_module._ensure_openapi_operation_tags_declared(openapi_schema)

    declared_tags = {
        tag["name"]
        for tag in openapi_schema["tags"]
        if isinstance(tag, dict) and isinstance(tag.get("name"), str)
    }
    assert "media" in declared_tags
    assert "health" not in declared_tags
    assert not set("health") & declared_tags


@pytest.mark.integration
def test_openapi_path_parameters_match_route_templates(openapi_spec: dict[str, Any]) -> None:
    """Path placeholders and OpenAPI path parameters must stay in sync for generated clients."""
    missing_parameters: list[str] = []
    extra_parameters: list[str] = []
    optional_parameters: list[str] = []

    for route, method, operation in _iter_http_operations(openapi_spec):
        route_parameters = set(PATH_PARAMETER_PATTERN.findall(route))
        operation_parameters = [
            parameter
            for parameter in operation.get("parameters", [])
            if isinstance(parameter, dict) and parameter.get("in") == "path"
        ]
        declared_parameters = {
            parameter["name"]
            for parameter in operation_parameters
            if isinstance(parameter.get("name"), str)
        }
        location = f"{method.upper()} {route}"

        for name in sorted(route_parameters - declared_parameters):
            missing_parameters.append(f"{location}: missing {name}")
        for name in sorted(declared_parameters - route_parameters):
            extra_parameters.append(f"{location}: extra {name}")
        for parameter in operation_parameters:
            if parameter.get("required") is not True:
                optional_parameters.append(f"{location}: {parameter.get('name')}")

    assert not missing_parameters, "OpenAPI path parameters missing route placeholders:\n" + "\n".join(
        missing_parameters[:50]
    )
    assert not extra_parameters, "OpenAPI path parameters not present in route template:\n" + "\n".join(
        extra_parameters[:50]
    )
    assert not optional_parameters, "OpenAPI path parameters must be required:\n" + "\n".join(
        optional_parameters[:50]
    )


@pytest.mark.integration
def test_openapi_declares_canonical_pagination_components(openapi_spec: dict[str, Any]) -> None:
    """Canonical pagination helper component names must stay stable for clients."""
    schemas = openapi_spec["components"]["schemas"]
    missing_components: list[str] = []
    missing_fields: list[str] = []

    for component_name, expected_fields in CANONICAL_PAGINATION_COMPONENT_FIELDS.items():
        schema = schemas.get(component_name)
        if not isinstance(schema, dict):
            missing_components.append(component_name)
            continue

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            missing_fields.append(f"{component_name}: missing properties")
            continue

        for field_name in sorted(expected_fields - set(properties)):
            missing_fields.append(f"{component_name}: missing {field_name}")

    assert not missing_components, "Missing canonical pagination schemas:\n" + "\n".join(missing_components)
    assert not missing_fields, "Canonical pagination schema fields drifted:\n" + "\n".join(missing_fields)


@pytest.mark.integration
def test_openapi_preserves_skills_file_and_no_content_exemptions(openapi_spec: dict[str, Any]) -> None:
    """Skills file downloads and 204 deletes must not be modeled as JSON envelopes."""
    paths = openapi_spec["paths"]

    export_operation = paths[SKILLS_EXPORT_PATH]["get"]
    export_response = export_operation["responses"]["200"]
    export_content = export_response.get("content", {})
    assert "application/zip" in export_content
    assert "application/json" not in export_content
    assert not _contains_ref_fragment(export_operation, "ResponseEnvelope")

    delete_response = paths[SKILLS_ITEM_PATH]["delete"]["responses"]["204"]
    assert not delete_response.get("content")
    assert not _contains_ref_fragment(delete_response, "ResponseEnvelope")


@pytest.mark.integration
def test_openapi_preserves_streaming_response_exemptions(openapi_spec: dict[str, Any]) -> None:
    """Streaming SSE and NDJSON routes must not be modeled as JSON envelopes."""
    paths = openapi_spec["paths"]

    for (method, path), media_type in STREAMING_RESPONSE_EXEMPTIONS.items():
        operation = paths[path][method]

        _assert_streaming_response_content(operation, media_type)


@pytest.mark.integration
def test_audio_job_progress_stream_openapi_documents_sse_response() -> None:
    """Audio job progress SSE docs must stay correct even when minimal app skips the router."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import router as audio_jobs_router

    app = FastAPI()
    app.include_router(audio_jobs_router, prefix="/api/v1/audio")
    operation = app.openapi()["paths"]["/api/v1/audio/jobs/{job_id}/progress/stream"]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_audio_speech_openapi_documents_audio_response() -> None:
    """OpenAI-compatible speech responses must be documented as audio, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_tts import router as audio_tts_router

    app = FastAPI()
    app.include_router(audio_tts_router, prefix="/api/v1/audio")
    operation = app.openapi()["paths"]["/api/v1/audio/speech"]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert AUDIO_SPEECH_CONTENT_TYPES.issubset(content)
    assert "application/json" not in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_audio_transcription_openapi_documents_text_and_json_responses() -> None:
    """OpenAI-compatible transcription and translation responses must document text and JSON formats."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions import router as audio_transcriptions_router

    app = FastAPI()
    app.include_router(audio_transcriptions_router, prefix="/api/v1/audio")
    paths = app.openapi()["paths"]

    for path in ("/api/v1/audio/transcriptions", "/api/v1/audio/translations"):
        operation = paths[path]["post"]
        content = operation["responses"]["200"].get("content", {})

        assert AUDIO_TRANSCRIPTION_CONTENT_TYPES.issubset(content)
        assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_audio_voice_preview_openapi_documents_audio_response() -> None:
    """Custom voice preview responses must be documented as audio, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_voices import router as audio_voices_router

    app = FastAPI()
    app.include_router(audio_voices_router, prefix="/api/v1/audio")
    operation = app.openapi()["paths"]["/api/v1/audio/voices/{voice_id}/preview"]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert "audio/mpeg" in content
    assert "application/json" not in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_audio_tokenizer_decode_openapi_documents_audio_response() -> None:
    """Audio tokenizer decode responses must be documented as audio bytes, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_tokenizer import router as audio_tokenizer_router

    app = FastAPI()
    app.include_router(audio_tokenizer_router, prefix="/api/v1/audio")
    operation = app.openapi()["paths"]["/api/v1/audio/tokenizer/decode"]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert AUDIO_TOKENIZER_DECODE_CONTENT_TYPES.issubset(content)
    assert "application/json" not in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_audio_tts_history_delete_openapi_documents_no_content_response() -> None:
    """TTS history deletes must document the 204 no-content response they return."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_history import router as audio_history_router

    app = FastAPI()
    app.include_router(audio_history_router, prefix="/api/v1/audio")
    operation = app.openapi()["paths"][AUDIO_TTS_HISTORY_PATH]["delete"]

    _assert_no_body_response(operation, "204")


@pytest.mark.integration
def test_config_quickstart_openapi_documents_redirect_and_html_fallback(openapi_spec: dict[str, Any]) -> None:
    """Quickstart must document its redirect response and built-in HTML fallback."""
    operation = openapi_spec["paths"][CONFIG_QUICKSTART_PATH]["get"]
    fallback_content = operation["responses"]["200"].get("content", {})

    assert "307" in operation["responses"]
    assert "text/html" in fallback_content
    assert "application/json" not in fallback_content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_redirect_routes_openapi_document_location_responses(openapi_spec: dict[str, Any]) -> None:
    """Routes that return redirects must document the redirect status and Location header."""
    cases = (
        (ROOT_PATH, "get", "307"),
        (FEDERATION_LOGIN_PATH, "get", "307"),
        (OPENAI_OAUTH_CALLBACK_PATH, "get", "303"),
    )

    for path, method, status_code in cases:
        operation = openapi_spec["paths"][path][method]

        _assert_redirect_response(operation, status_code)


@pytest.mark.integration
def test_zip_download_openapi_documents_file_responses() -> None:
    """Zip download routes must be documented as files, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.admin.admin_bundle_ops import router as admin_bundle_router
    from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router

    app = FastAPI()
    app.include_router(admin_bundle_router, prefix="/api/v1/admin")
    app.include_router(chatbooks_router, prefix="/api/v1")
    paths = app.openapi()["paths"]

    _assert_file_response_content(
        paths["/api/v1/admin/backups/bundles/{bundle_id}/download"]["get"],
        {"application/zip"},
    )
    _assert_file_response_content(
        paths["/api/v1/chatbooks/download/{job_id}"]["get"],
        {"application/zip"},
    )


@pytest.mark.integration
def test_output_download_openapi_documents_file_responses() -> None:
    """Output artifact downloads must advertise file media types instead of JSON."""
    from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

    app = FastAPI()
    app.include_router(outputs_router, prefix="/api/v1")
    paths = app.openapi()["paths"]

    _assert_file_response_content(
        paths["/api/v1/outputs/{output_id}/download"]["get"],
        OUTPUT_DOWNLOAD_CONTENT_TYPES,
    )
    _assert_file_response_content(
        paths["/api/v1/outputs/download/by-name"]["get"],
        OUTPUT_DOWNLOAD_CONTENT_TYPES,
    )


@pytest.mark.integration
def test_output_download_head_openapi_documents_no_body_response() -> None:
    """Output download HEAD checks must be documented as header-only responses."""
    from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

    app = FastAPI()
    app.include_router(outputs_router, prefix="/api/v1")
    operation = app.openapi()["paths"][OUTPUT_DOWNLOAD_PATH]["head"]

    _assert_no_body_response(operation)


@pytest.mark.integration
def test_file_artifact_export_openapi_documents_file_responses(openapi_spec: dict[str, Any]) -> None:
    """File artifact exports must advertise raw file media types for generated clients."""
    operation = openapi_spec["paths"][FILE_ARTIFACT_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert FILE_ARTIFACT_EXPORT_CONTENT_TYPES.issubset(content)
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_openai_chat_completion_openapi_documents_json_and_sse(openapi_spec: dict[str, Any]) -> None:
    """OpenAI-compatible chat completions must document JSON and SSE provider-shaped responses."""
    operation = openapi_spec["paths"][CHAT_COMPLETIONS_PATH]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "text/event-stream" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_anthropic_messages_openapi_documents_json_and_sse(openapi_spec: dict[str, Any]) -> None:
    """Anthropic-compatible messages endpoints must document JSON and SSE provider-shaped responses."""
    for path in MESSAGES_COMPLETION_PATHS:
        operation = openapi_spec["paths"][path]["post"]
        content = operation["responses"]["200"].get("content", {})

        assert "application/json" in content
        assert "text/event-stream" in content
        assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_prometheus_text_openapi_documents_plaintext_response(openapi_spec: dict[str, Any]) -> None:
    """Prometheus scrape endpoints must be documented as text, not JSON envelopes."""
    for path in PROMETHEUS_TEXT_RESPONSE_PATHS:
        operation = openapi_spec["paths"][path]["get"]
        content = operation["responses"]["200"].get("content", {})

        assert "text/plain; version=0.0.4" in content or "text/plain; version=0.0.4; charset=utf-8" in content
        assert "application/json" not in content
        assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_mcp_request_openapi_documents_json_and_no_content_response(openapi_spec: dict[str, Any]) -> None:
    """MCP HTTP requests can return JSON-RPC results or a 204 no-content acknowledgement."""
    operation = openapi_spec["paths"][MCP_REQUEST_PATH]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    _assert_no_body_response(operation, "204")


@pytest.mark.integration
def test_conditional_cache_routes_openapi_document_304_no_body_response(openapi_spec: dict[str, Any]) -> None:
    """Conditional cache routes with ETags must document 304 no-body responses."""
    for method, path in NOT_MODIFIED_RESPONSE_OPERATIONS:
        operation = openapi_spec["paths"][path][method]
        _assert_no_body_response(operation, "304")


@pytest.mark.integration
def test_claims_analytics_export_openapi_documents_csv_response(openapi_spec: dict[str, Any]) -> None:
    """Claims analytics export downloads must document CSV as an alternate response format."""
    operation = openapi_spec["paths"][CLAIMS_ANALYTICS_EXPORT_DOWNLOAD_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "text/csv" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_privilege_snapshot_csv_export_openapi_documents_csv_response() -> None:
    """Privilege snapshot CSV exports must be documented as CSV streams, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router

    app = FastAPI()
    app.include_router(privileges_router, prefix="/api/v1")
    operation = app.openapi()["paths"][PRIVILEGE_SNAPSHOT_CSV_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert "text/csv" in content
    assert "application/json" not in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_evaluations_abtest_events_openapi_documents_sse_response() -> None:
    """Evaluation A/B test event streams must be documented as SSE, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evaluations_router

    app = FastAPI()
    app.include_router(evaluations_router, prefix="/api/v1")
    operation = app.openapi()["paths"][EVALUATIONS_ABTEST_EVENTS_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_evaluations_abtest_export_openapi_documents_json_and_csv_response() -> None:
    """Evaluation A/B test exports must document both JSON and CSV formats."""
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evaluations_router

    app = FastAPI()
    app.include_router(evaluations_router, prefix="/api/v1")
    operation = app.openapi()["paths"][EVALUATIONS_ABTEST_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "text/csv" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_evaluations_metrics_openapi_documents_json_and_prometheus_response() -> None:
    """Evaluation metrics must document both JSON and Prometheus text formats."""
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evaluations_router

    app = FastAPI()
    app.include_router(evaluations_router, prefix="/api/v1")
    operation = app.openapi()["paths"][EVALUATIONS_METRICS_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "text/plain; version=0.0.4; charset=utf-8" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_storage_file_download_openapi_documents_file_response() -> None:
    """Generated file downloads must be documented as files, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.storage import router as storage_router

    app = FastAPI()
    app.include_router(storage_router, prefix="/api/v1")
    operation = app.openapi()["paths"][STORAGE_FILE_DOWNLOAD_PATH]["get"]

    _assert_file_response_content(operation, {"application/octet-stream"})


@pytest.mark.integration
def test_jobs_events_stream_openapi_documents_sse_response() -> None:
    """Jobs event streams must be documented as SSE, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router

    app = FastAPI()
    app.include_router(jobs_admin_router, prefix="/api/v1")
    operation = app.openapi()["paths"][JOBS_EVENTS_STREAM_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_watchlist_sources_export_openapi_documents_xml_response() -> None:
    """Watchlist source OPML exports must be documented as XML, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

    app = FastAPI()
    app.include_router(watchlists_router, prefix="/api/v1")
    operation = app.openapi()["paths"][WATCHLIST_SOURCES_EXPORT_PATH]["get"]

    _assert_file_response_content(operation, {"application/xml"})


@pytest.mark.integration
def test_watchlist_runs_csv_exports_openapi_document_csv_responses() -> None:
    """Watchlist run CSV exports must document CSV media types for generated clients."""
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

    app = FastAPI()
    app.include_router(watchlists_router, prefix="/api/v1")
    paths = app.openapi()["paths"]

    for path in (WATCHLIST_RUNS_CSV_EXPORT_PATH, WATCHLIST_RUN_TALLIES_CSV_EXPORT_PATH):
        _assert_file_response_content(paths[path]["get"], {"text/csv; charset=utf-8"})


@pytest.mark.integration
def test_watchlist_output_download_openapi_documents_rendered_file_responses() -> None:
    """Watchlist rendered output downloads must advertise their raw media types."""
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

    app = FastAPI()
    app.include_router(watchlists_router, prefix="/api/v1")
    operation = app.openapi()["paths"][WATCHLIST_OUTPUT_DOWNLOAD_PATH]["get"]

    _assert_file_response_content(operation, WATCHLIST_OUTPUT_DOWNLOAD_CONTENT_TYPES)


@pytest.mark.integration
def test_character_export_openapi_documents_json_and_png_responses() -> None:
    """Character exports must document JSON card data and PNG character-card downloads."""
    from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as characters_router

    app = FastAPI()
    app.include_router(characters_router, prefix="/api/v1/characters")
    operation = app.openapi()["paths"][CHARACTER_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "image/png" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_admin_usage_csv_exports_openapi_document_csv_responses() -> None:
    """Admin usage CSV exports must document CSV media types for generated clients."""
    from tldw_Server_API.app.api.v1.endpoints.admin.admin_usage import router as admin_usage_router

    app = FastAPI()
    app.include_router(admin_usage_router, prefix="/api/v1/admin")
    paths = app.openapi()["paths"]

    for path in ADMIN_USAGE_CSV_EXPORT_PATHS:
        _assert_file_response_content(paths[path]["get"], {"text/csv"})


@pytest.mark.integration
def test_admin_json_csv_exports_openapi_document_raw_response_formats() -> None:
    """Admin exports that can return JSON or CSV must document both raw formats."""
    from tldw_Server_API.app.api.v1.endpoints.admin.admin_system import router as admin_system_router
    from tldw_Server_API.app.api.v1.endpoints.admin.admin_user import router as admin_user_router

    app = FastAPI()
    app.include_router(admin_user_router, prefix="/api/v1/admin")
    app.include_router(admin_system_router, prefix="/api/v1/admin")
    paths = app.openapi()["paths"]

    for path in ADMIN_JSON_CSV_EXPORT_PATHS:
        content = paths[path]["get"]["responses"]["200"].get("content", {})

        assert "application/json" in content
        assert "text/csv" in content
        assert not _contains_ref_fragment(paths[path]["get"], "ResponseEnvelope")


@pytest.mark.integration
def test_flashcard_asset_content_openapi_documents_binary_response() -> None:
    """Flashcard asset content must be documented as a raw binary response."""
    from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

    app = FastAPI()
    app.include_router(flashcards_router, prefix="/api/v1")
    operation = app.openapi()["paths"][FLASHCARD_ASSET_CONTENT_PATH]["get"]

    _assert_file_response_content(operation, {"application/octet-stream"})


@pytest.mark.integration
def test_flashcard_export_openapi_documents_raw_response_formats() -> None:
    """Flashcard export must document all supported raw download formats."""
    from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

    app = FastAPI()
    app.include_router(flashcards_router, prefix="/api/v1")
    operation = app.openapi()["paths"][FLASHCARD_EXPORT_PATH]["get"]

    _assert_file_response_content(operation, FLASHCARD_EXPORT_CONTENT_TYPES)


@pytest.mark.integration
def test_reading_tts_openapi_documents_audio_responses() -> None:
    """Reading item TTS must document supported audio response formats."""
    from tldw_Server_API.app.api.v1.endpoints.reading import router as reading_router

    app = FastAPI()
    app.include_router(reading_router, prefix="/api/v1")
    operation = app.openapi()["paths"][READING_TTS_PATH]["post"]

    _assert_file_response_content(operation, AUDIO_SPEECH_CONTENT_TYPES)


@pytest.mark.integration
def test_reading_export_openapi_documents_raw_response_formats() -> None:
    """Reading exports must document NDJSON and ZIP download formats."""
    from tldw_Server_API.app.api.v1.endpoints.reading import router as reading_router

    app = FastAPI()
    app.include_router(reading_router, prefix="/api/v1")
    operation = app.openapi()["paths"][READING_EXPORT_PATH]["get"]

    _assert_file_response_content(operation, READING_EXPORT_CONTENT_TYPES)


@pytest.mark.integration
def test_audit_export_openapi_documents_raw_response_formats() -> None:
    """Audit export must document raw JSON, NDJSON, and CSV response formats."""
    from tldw_Server_API.app.api.v1.endpoints.audit import router as audit_router

    app = FastAPI()
    app.include_router(audit_router, prefix="/api/v1")
    operation = app.openapi()["paths"][AUDIT_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert AUDIT_EXPORT_CONTENT_TYPES.issubset(content)
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_meeting_events_openapi_documents_sse_response() -> None:
    """Meeting session events must document SSE stream responses."""
    from tldw_Server_API.app.api.v1.endpoints.meetings import router as meetings_router

    app = FastAPI()
    app.include_router(meetings_router, prefix="/api/v1")
    operation = app.openapi()["paths"][MEETING_EVENTS_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_notifications_stream_openapi_documents_sse_response() -> None:
    """Notifications stream must document SSE responses."""
    from tldw_Server_API.app.api.v1.endpoints.notifications import router as notifications_router

    app = FastAPI()
    app.include_router(notifications_router, prefix="/api/v1")
    operation = app.openapi()["paths"][NOTIFICATIONS_STREAM_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_vn_asset_content_openapi_documents_file_response() -> None:
    """VN asset item content must document a raw file response."""
    from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router

    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")
    operation = app.openapi()["paths"][VN_ASSET_CONTENT_PATH]["get"]

    _assert_file_response_content(operation, {"application/octet-stream", "image/jpeg", "image/png", "image/webp"})


@pytest.mark.integration
def test_vn_policy_openapi_documents_canonical_paths() -> None:
    """VN policy profile endpoints must be exposed under the canonical VN namespace."""
    from tldw_Server_API.app.api.v1.endpoints.vn_policy import router as vn_policy_router

    app = FastAPI()
    app.include_router(vn_policy_router, prefix="/api/v1/vn")
    paths = app.openapi()["paths"]

    assert VN_POLICY_EVALUATE_PATH in paths
    assert VN_POLICY_PROFILES_PATH in paths
    assert VN_POLICY_GENERATION_PROFILES_PATH in paths
    assert "post" in paths[VN_POLICY_EVALUATE_PATH]
    assert {"get", "post"}.issubset(paths[VN_POLICY_PROFILES_PATH])
    assert {"get", "post"}.issubset(paths[VN_POLICY_GENERATION_PROFILES_PATH])


@pytest.mark.integration
def test_vn_scripts_openapi_documents_canonical_paths() -> None:
    """VN script authoring endpoints must be exposed under the canonical VN namespace."""
    from tldw_Server_API.app.api.v1.endpoints.vn_scripts import router as vn_scripts_router

    app = FastAPI()
    app.include_router(vn_scripts_router, prefix="/api/v1/vn")
    paths = app.openapi()["paths"]

    assert VN_SCRIPTS_PATH in paths
    assert VN_SCRIPT_PATH in paths
    assert VN_SCRIPT_DRAFT_PATH in paths
    assert VN_SCRIPT_DRAFT_VALIDATE_PATH in paths
    assert VN_SCRIPT_DRAFT_DIAGNOSTICS_PATH in paths
    assert VN_SCRIPT_DRAFT_PLAYTEST_PATH in paths
    assert VN_SCRIPT_PUBLISH_PATH in paths
    assert VN_SCRIPT_VERSIONS_PATH in paths
    assert VN_SCRIPT_VERSION_PATH in paths
    assert VN_SCRIPT_MANIFEST_SNAPSHOT_PATH in paths
    assert VN_SCRIPT_VERSION_POLICY_EVALUATE_PATH in paths
    assert VN_SCRIPT_VERSION_PLAYTEST_PATH in paths
    assert {"get", "post"}.issubset(paths[VN_SCRIPTS_PATH])
    assert {"get", "patch", "delete"}.issubset(paths[VN_SCRIPT_PATH])
    assert {"get", "put"}.issubset(paths[VN_SCRIPT_DRAFT_PATH])
    assert "post" in paths[VN_SCRIPT_DRAFT_VALIDATE_PATH]
    assert "get" in paths[VN_SCRIPT_DRAFT_DIAGNOSTICS_PATH]
    assert "post" in paths[VN_SCRIPT_DRAFT_PLAYTEST_PATH]
    assert "post" in paths[VN_SCRIPT_PUBLISH_PATH]
    assert "get" in paths[VN_SCRIPT_VERSIONS_PATH]
    assert "get" in paths[VN_SCRIPT_VERSION_PATH]
    assert "get" in paths[VN_SCRIPT_MANIFEST_SNAPSHOT_PATH]
    assert "post" in paths[VN_SCRIPT_VERSION_POLICY_EVALUATE_PATH]
    assert "post" in paths[VN_SCRIPT_VERSION_PLAYTEST_PATH]


@pytest.mark.integration
def test_slides_export_openapi_documents_raw_response_formats() -> None:
    """Slides exports must document supported raw download formats."""
    from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router

    app = FastAPI()
    app.include_router(slides_router, prefix="/api/v1")
    operation = app.openapi()["paths"][SLIDES_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert SLIDES_EXPORT_CONTENT_TYPES.issubset(content)
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_websub_verify_callback_openapi_documents_plaintext_response() -> None:
    """WebSub verification callbacks must document their plaintext challenge response."""
    from tldw_Server_API.app.api.v1.endpoints.collections_websub import callback_router

    app = FastAPI()
    app.include_router(callback_router, prefix="/api/v1")
    operation = app.openapi()["paths"][WEBSUB_VERIFY_CALLBACK_PATH]["get"]

    _assert_file_response_content(operation, {"text/plain"})


@pytest.mark.integration
def test_websub_push_callback_openapi_documents_no_body_response() -> None:
    """WebSub push callbacks acknowledge hubs with an empty response body."""
    from tldw_Server_API.app.api.v1.endpoints.collections_websub import callback_router

    app = FastAPI()
    app.include_router(callback_router, prefix="/api/v1")
    operation = app.openapi()["paths"][WEBSUB_PUSH_CALLBACK_PATH]["post"]

    _assert_no_body_response(operation)


@pytest.mark.integration
def test_notes_csv_exports_openapi_document_csv_response() -> None:
    """Notes CSV exports must document CSV streams instead of JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router

    app = FastAPI()
    app.include_router(notes_router, prefix="/api/v1/notes")
    paths = app.openapi()["paths"]

    for method in ("get", "post"):
        operation = paths[NOTES_CSV_EXPORT_PATHS[0]][method]
        _assert_file_response_content(operation, {"text/csv; charset=utf-8"})


@pytest.mark.integration
def test_notes_attachment_download_openapi_documents_file_response() -> None:
    """Note attachment downloads must document raw file content for generated clients."""
    from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router

    app = FastAPI()
    app.include_router(notes_router, prefix="/api/v1/notes")
    operation = app.openapi()["paths"][NOTES_ATTACHMENT_DOWNLOAD_PATH]["get"]

    _assert_file_response_content(operation, {"application/octet-stream"})


@pytest.mark.integration
def test_connector_provider_webhook_openapi_documents_validation_token_response() -> None:
    """Connector provider webhooks must document JSON callbacks and plaintext validation tokens."""
    from tldw_Server_API.app.api.v1.endpoints.connectors import router as connectors_router

    app = FastAPI()
    app.include_router(connectors_router, prefix="/api/v1")
    paths = app.openapi()["paths"]

    for method in ("get", "post"):
        operation = paths[CONNECTOR_PROVIDER_WEBHOOK_PATH][method]
        content = operation["responses"]["200"].get("content", {})

        assert "application/json" in content
        assert "text/plain" in content
        assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_workflow_artifact_downloads_openapi_document_file_responses() -> None:
    """Workflow artifact downloads must document raw file and ZIP response formats."""
    from tldw_Server_API.app.api.v1.endpoints.workflows import router as workflows_router

    app = FastAPI()
    app.include_router(workflows_router)
    paths = app.openapi()["paths"]

    _assert_file_response_content(
        paths[WORKFLOW_ARTIFACT_DOWNLOAD_PATH]["get"],
        {"application/octet-stream"},
    )
    _assert_file_response_content(
        paths[WORKFLOW_RUN_ARTIFACTS_DOWNLOAD_PATH]["get"],
        {"application/zip"},
    )


@pytest.mark.integration
def test_data_table_export_openapi_documents_json_and_download_responses() -> None:
    """Data table exports must document JSON metadata and direct download media types."""
    from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router

    app = FastAPI()
    app.include_router(data_tables_router, prefix="/api/v1")
    operation = app.openapi()["paths"][DATA_TABLE_EXPORT_PATH]["get"]
    content = operation["responses"]["200"].get("content", {})

    assert DATA_TABLE_EXPORT_CONTENT_TYPES.issubset(content)
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_chat_document_generate_openapi_documents_json_and_sse_response() -> None:
    """Chat document generation must document JSON results and optional SSE streams."""
    from tldw_Server_API.app.api.v1.endpoints.chat_documents import router as chat_documents_router

    app = FastAPI()
    app.include_router(chat_documents_router, prefix="/api/v1/chat")
    operation = app.openapi()["paths"][CHAT_DOCUMENT_GENERATE_PATH]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "text/event-stream" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_acp_session_events_stream_openapi_documents_sse_response() -> None:
    """ACP session event streams must be documented as SSE, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import router as acp_router

    app = FastAPI()
    app.include_router(acp_router, prefix="/api/v1")
    operation = app.openapi()["paths"][ACP_SESSION_EVENTS_STREAM_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_mcp_hub_events_stream_openapi_documents_sse_response() -> None:
    """MCP Hub governance event streams must be documented as SSE, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.mcp_hub_management import router as mcp_hub_router

    app = FastAPI()
    app.include_router(mcp_hub_router, prefix="/api/v1")
    operation = app.openapi()["paths"][MCP_HUB_EVENTS_STREAM_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_embeddings_orchestrator_events_openapi_documents_sse_response() -> None:
    """Embeddings orchestrator event streams must be documented as SSE, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router

    app = FastAPI()
    app.include_router(embeddings_router, prefix="/api/v1")
    operation = app.openapi()["paths"][EMBEDDINGS_ORCHESTRATOR_EVENTS_PATH]["get"]

    _assert_streaming_response_content(operation, "text/event-stream")


@pytest.mark.integration
def test_character_chat_completion_v2_openapi_documents_json_and_sse_response() -> None:
    """Character chat completion v2 must document JSON results and optional SSE streams."""
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import router as character_chats_router

    app = FastAPI()
    app.include_router(character_chats_router, prefix="/api/v1/chats")
    operation = app.openapi()["paths"][CHARACTER_CHAT_COMPLETION_V2_PATH]["post"]
    content = operation["responses"]["200"].get("content", {})

    assert "application/json" in content
    assert "text/event-stream" in content
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_paper_search_raw_passthrough_openapi_documents_raw_content_types() -> None:
    """Paper-search raw passthrough routes must document upstream content types, not JSON envelopes."""
    from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router

    app = FastAPI()
    app.include_router(paper_search_router, prefix="/api/v1/paper-search")
    paths = app.openapi()["paths"]

    for path in PAPER_SEARCH_RAW_JSON_XML_HTML_PATHS:
        _assert_raw_response_content(paths[path]["get"], {"application/json", "application/xml", "text/html"})

    for path in PAPER_SEARCH_RAW_JSON_CSV_PATHS:
        _assert_raw_response_content(paths[path]["get"], {"application/json", "text/csv"})

    for path in PAPER_SEARCH_RAW_JSON_XML_PATHS:
        _assert_raw_response_content(paths[path]["get"], {"application/json", "application/xml"})

    for path in PAPER_SEARCH_RAW_XML_PATHS:
        _assert_raw_response_content(paths[path]["get"], {"application/xml"})

    for path in PAPER_SEARCH_RAW_JSON_PATHS:
        _assert_raw_response_content(paths[path]["get"], {"application/json"})

    _assert_raw_response_content(paths[PAPER_SEARCH_PMC_OA_PDF_PATH]["get"], {"application/pdf"})
    _assert_raw_response_content(
        paths[PAPER_SEARCH_HAL_RAW_PATH]["get"],
        {
            "application/atom+xml",
            "application/json",
            "application/octet-stream",
            "application/rss+xml",
            "application/xml",
            "text/csv",
            "text/plain",
        },
    )


@pytest.mark.integration
def test_media_file_openapi_documents_full_and_partial_file_responses() -> None:
    """Media file downloads must document both full and range-streamed file response content."""
    from tldw_Server_API.app.api.v1.endpoints.media.file import router as media_file_router

    app = FastAPI()
    app.include_router(media_file_router, prefix="/api/v1/media")
    operation = app.openapi()["paths"][MEDIA_FILE_PATH]["get"]
    full_content = operation["responses"]["200"].get("content", {})
    partial_content = operation["responses"]["206"].get("content", {})

    assert {"application/pdf", "application/octet-stream"}.issubset(full_content)
    assert {"application/pdf", "application/octet-stream"}.issubset(partial_content)
    assert not _contains_ref_fragment(operation, "ResponseEnvelope")


@pytest.mark.integration
def test_openapi_documents_supported_auth_schemes(openapi_spec: dict[str, Any]) -> None:
    """The generated OpenAPI document must advertise both supported authentication schemes."""
    security_schemes = openapi_spec["components"]["securitySchemes"]

    assert security_schemes["ApiKeyAuth"]["type"] == "apiKey"
    assert security_schemes["ApiKeyAuth"]["in"] == "header"
    assert security_schemes["ApiKeyAuth"]["name"] == "X-API-KEY"
    assert security_schemes["BearerAuth"]["type"] == "http"
    assert security_schemes["BearerAuth"]["scheme"] == "bearer"
    assert {"ApiKeyAuth": []} in openapi_spec["security"]
    assert {"BearerAuth": []} in openapi_spec["security"]


@pytest.mark.integration
@pytest.mark.parametrize("path", ["/health", "/ready", "/health/ready"])
def test_control_plane_head_routes_are_registered(client_user_only, path: str) -> None:
    """The HEAD health probes must remain routable after splitting OpenAPI operations."""
    response = client_user_only.head(path)

    assert response.status_code in {200, 503}
    assert response.text == ""


@pytest.mark.integration
@pytest.mark.parametrize("path", ["/health", "/ready", "/health/ready"])
def test_control_plane_routes_keep_health_openapi_tag(openapi_spec: dict[str, Any], path: str) -> None:
    """Control-plane probe routes should remain grouped under the health tag."""
    for method in ("get", "head"):
        operation = openapi_spec["paths"][path][method]
        assert "health" in operation.get("tags", [])
