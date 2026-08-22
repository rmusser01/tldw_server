"""OpenAPI contract tests for the PR 1 canonical webhook surface."""

import json

from fastapi import FastAPI

from tldw_Server_API.app.api.v1.endpoints.admin import admin_webhooks


def _openapi() -> dict[str, object]:
    app = FastAPI()
    app.include_router(admin_webhooks.status_router, prefix="/api/v1/admin")
    app.include_router(admin_webhooks.canonical_router, prefix="/api/v1/admin")
    return app.openapi()


def test_pr1_openapi_has_only_control_plane_operations() -> None:
    paths = _openapi()["paths"]

    assert "/api/v1/admin/webhooks/catalog" in paths
    assert "/api/v1/admin/webhooks/status" in paths
    assert "/api/v1/admin/webhooks" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/rotate-secret" in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/test" not in paths
    assert "/api/v1/admin/webhooks/{webhook_id}/deliveries" not in paths


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


def test_schema_examples_use_reserved_hosts_and_an_obvious_fake_secret() -> None:
    encoded = json.dumps(_openapi(), sort_keys=True)

    assert "receiver.example" in encoded
    assert "whsec_" + ("0" * 64) in encoded
    assert "localhost" not in encoded
    assert "example.com" not in encoded
