"""OpenAPI contract tests for the canonical webhook control and delivery surface."""

import json

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints.admin import admin_webhooks
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_webhooks import (
    webhooks_router as evaluation_webhooks_router,
)


def _openapi() -> dict[str, object]:
    app = FastAPI()
    app.include_router(admin_webhooks.status_router, prefix="/api/v1/admin")
    app.include_router(admin_webhooks.canonical_router, prefix="/api/v1/admin")
    return app.openapi()


@pytest.mark.unit
def test_openapi_exposes_reviewed_control_plane_and_delivery_operations() -> None:
    paths = _openapi()["paths"]

    assert "/api/v1/admin/webhooks/catalog" in paths
    assert "/api/v1/admin/webhooks/status" in paths
    assert "/api/v1/admin/webhooks" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/rotate-secret" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/test" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/deliveries" in paths
    assert (
        "/api/v1/admin/webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver"
        in paths
    )
    assert set(paths["/api/v1/admin/webhooks/{webhook_id}/test"]) == {"post"}
    assert set(paths["/api/v1/admin/webhooks/{webhook_id}/deliveries"]) == {"get"}
    assert set(
        paths[
            "/api/v1/admin/webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver"
        ]
    ) == {"post"}


@pytest.mark.unit
def test_status_openapi_has_closed_sanitized_delivery_capability() -> None:
    spec = _openapi()
    schemas = spec["components"]["schemas"]
    status = schemas["AdminWebhookStatusResponse"]

    assert status["properties"]["delivery"] == {
        "$ref": "#/components/schemas/DeliveryCapabilityStatusResponse"
    }
    for name in (
        "AdminWebhookStatusResponse",
        "DeliveryCapabilityStatusResponse",
        "DeliveryComponentStatusResponse",
        "DeliveryBacklogCountsResponse",
    ):
        assert schemas[name]["additionalProperties"] is False
    encoded = json.dumps(
        {
            name: schemas[name]
            for name in schemas
            if name.startswith("DeliveryCapability")
            or name.startswith("DeliveryComponent")
            or name.startswith("DeliveryBacklog")
        },
        sort_keys=True,
    ).lower()
    for forbidden in ("instance_id", "url", "hostname", "secret", "payload"):
        assert forbidden not in encoded


@pytest.mark.unit
def test_delivery_openapi_uses_only_fixed_request_response_and_header_contracts() -> None:
    spec = _openapi()
    paths = spec["paths"]
    schemas = spec["components"]["schemas"]
    test_operation = paths["/api/v1/admin/webhooks/{webhook_id}/test"]["post"]
    redelivery_operation = paths[
        "/api/v1/admin/webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver"
    ]["post"]
    history_operation = paths["/api/v1/admin/webhooks/{webhook_id}/deliveries"]["get"]

    assert test_operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/WebhookTestRequest"
    }
    assert redelivery_operation["requestBody"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/WebhookRedeliveryRequest"}
    assert test_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/WebhookTestResponse"}
    assert test_operation["responses"]["202"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/WebhookTestResponse"}
    assert redelivery_operation["responses"]["202"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/WebhookRedeliveryResponse"}
    assert history_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/WebhookDeliveryListResponse"}
    for name in (
        "WebhookTestRequest",
        "WebhookRedeliveryRequest",
        "WebhookTestResponse",
        "WebhookRedeliveryResponse",
        "WebhookDeliveryResponse",
        "WebhookDeliveryAttemptResponse",
        "WebhookDeliveryHistoryItemResponse",
        "WebhookDeliveryListResponse",
    ):
        assert schemas[name]["additionalProperties"] is False
    encoded = json.dumps(
        {
            "paths": {
                path: paths[path]
                for path in (
                    "/api/v1/admin/webhooks/{webhook_id}/test",
                    "/api/v1/admin/webhooks/{webhook_id}/deliveries",
                    "/api/v1/admin/webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver",
                )
            },
            "schemas": {
                name: schemas[name]
                for name in schemas
                if name.startswith("WebhookDelivery")
                or name.startswith("WebhookTest")
                or name.startswith("WebhookRedelivery")
            },
        },
        sort_keys=True,
    ).lower()
    for forbidden in (
        "ciphertext",
        "key_id",
        "target_url",
        "request_headers",
        "response_body",
        "response_headers",
        "jobs_job_id",
        "lease_id",
        "test_attempt_token",
        "idempotency_key",
    ):
        assert forbidden not in encoded


@pytest.mark.unit
def test_delivery_openapi_declares_exact_mutation_and_success_headers() -> None:
    paths = _openapi()["paths"]
    test_operation = paths["/api/v1/admin/webhooks/{webhook_id}/test"]["post"]
    redelivery_operation = paths[
        "/api/v1/admin/webhooks/{webhook_id}/deliveries/{delivery_id}/redeliver"
    ]["post"]
    history_operation = paths[
        "/api/v1/admin/webhooks/{webhook_id}/deliveries"
    ]["get"]

    for operation in (test_operation, redelivery_operation):
        assert [
            parameter
            for parameter in operation["parameters"]
            if parameter["in"] == "query"
        ] == []
        request_headers = {
            parameter["name"]: parameter
            for parameter in operation["parameters"]
            if parameter["in"] == "header"
        }
        assert set(request_headers) == {"Idempotency-Key", "If-Match"}
        assert request_headers["If-Match"]["required"] is True
        assert request_headers["If-Match"]["schema"]["type"] == "string"
        assert "anyOf" not in request_headers["If-Match"]["schema"]
        idempotency_schema = request_headers["Idempotency-Key"]["schema"]
        assert request_headers["Idempotency-Key"]["required"] is True
        assert idempotency_schema["type"] == "string"
        assert idempotency_schema["minLength"] == 16
        assert idempotency_schema["maxLength"] == 255
        assert idempotency_schema["pattern"] == r"^[A-Za-z0-9._:-]{16,255}$"

    request_id_schema = {
        "type": "string",
        "minLength": 1,
        "maxLength": 128,
        "pattern": r"^[A-Za-z0-9._:-]{1,128}$",
    }
    cache_schema = {"type": "string", "enum": ["no-store"]}
    retry_schema = {"type": "integer", "minimum": 0, "maximum": 86_400}

    expected_headers = (
        (history_operation, "200", False),
        (test_operation, "200", False),
        (test_operation, "202", True),
        (redelivery_operation, "202", False),
    )
    for operation, status, has_retry_after in expected_headers:
        response_headers = operation["responses"][status]["headers"]
        expected_names = {"X-Request-ID", "Cache-Control"}
        if has_retry_after:
            expected_names.add("Retry-After")
        assert set(response_headers) == expected_names
        assert response_headers["X-Request-ID"]["schema"] == request_id_schema
        assert response_headers["Cache-Control"]["schema"] == cache_schema
        if has_retry_after:
            assert response_headers["Retry-After"]["schema"] == retry_schema


@pytest.mark.unit
def test_every_canonical_operation_uses_bounded_validation_error_schema() -> None:
    spec = _openapi()
    paths = spec["paths"]

    for path_item in paths.values():
        for operation in path_item.values():
            responses = operation["responses"]
            validation = responses["422"]["content"]["application/json"]["schema"]
            assert validation == {"$ref": "#/components/schemas/WebhookErrorResponse"}

    schemas = spec["components"]["schemas"]
    assert "HTTPValidationError" not in schemas


@pytest.mark.unit
def test_canonical_operations_declare_stable_error_envelopes() -> None:
    paths = _openapi()["paths"]
    expected = {"401", "403", "404", "409", "412", "422", "428", "429", "500", "503"}

    for path_item in paths.values():
        for operation in path_item.values():
            responses = operation["responses"]
            assert expected.issubset(responses)
            for status in expected:
                schema = responses[status]["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/WebhookErrorResponse"}


@pytest.mark.unit
def test_schema_examples_use_reserved_hosts_and_an_obvious_fake_secret() -> None:
    encoded = json.dumps(_openapi(), sort_keys=True)

    assert "receiver.example" in encoded
    assert "whsec_" + ("0" * 64) in encoded
    assert "localhost" not in encoded
    assert "example.com" not in encoded


@pytest.mark.unit
def test_canonical_models_do_not_rename_evaluation_webhook_schemas() -> None:
    app = FastAPI()
    app.include_router(evaluation_webhooks_router, prefix="/api/v1/evaluations")
    app.include_router(admin_webhooks.status_router, prefix="/api/v1/admin")
    app.include_router(admin_webhooks.canonical_router, prefix="/api/v1/admin")

    spec = app.openapi()
    evaluation_path = spec["paths"]["/api/v1/evaluations/webhooks"]

    assert evaluation_path["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["items"] == {
        "$ref": "#/components/schemas/WebhookStatusResponse"
    }
    assert evaluation_path["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {
        "$ref": "#/components/schemas/WebhookRegistrationResponse"
    }
