from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path("Docs/Deployment/Production_Reference_Deployment.md")
PUBLISHED_MIRRORS = (
    Path("Docs/Deployment/First_Time_Production_Setup.md"),
    Path("Docs/Deployment/Long_Term_Admin_Guide.md"),
    Path("Docs/Deployment/Reverse_Proxy_Examples.md"),
    RUNBOOK,
)
LINKING_DOCS = (
    Path("Docs/Deployment/First_Time_Production_Setup.md"),
    Path("Docs/Deployment/Long_Term_Admin_Guide.md"),
    Path("Docs/Deployment/Reverse_Proxy_Examples.md"),
    Path("Dockerfiles/README.md"),
)


def _shell_blocks(text: str) -> str:
    return "\n".join(re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL))


def _rotation_block(text: str) -> str:
    return next(
        block
        for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
        if "TLDW_OLD_METRICS_API_KEY_ID" in block and "metrics-credential-init" in block
    )


def _reference_link_target(path: Path) -> Path:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\[[^]]+\]\(([^)]*Production_Reference_Deployment\.md)\)", text)
    assert match is not None, path
    return (path.parent / match.group(1)).resolve()


def _declared_operator_api_paths() -> set[str]:
    from tldw_Server_API.app.api.v1.endpoints import health, metrics

    return {f"/api/v1{route.path}" for router in (health.router, metrics.router) for route in router.routes}


def test_reference_runbook_covers_the_fail_closed_operator_workflow() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    commands = _shell_blocks(text)

    for target in ("production-preflight", "production-deploy", "production-rollback"):
        assert re.search(rf"make {target}\s+\\\n\s+PRODUCTION_ENV_FILE=", commands), target
    assert "PRODUCTION_MANIFEST=" in commands
    assert 'chmod 600 "$PRODUCTION_ENV_FILE"' in commands
    assert 'chmod 600 "$TLDW_METRICS_API_KEY_FILE"' in commands
    assert "Dockerfiles/Monitoring/docker-compose.production.yml" in commands
    for path in (
        "/health",
        "/internal/ready",
        "/ready",
        "/health/ready",
        "/api/v1/healthz",
        "/api/v1/readyz",
        "/setup",
        "/api/v1/metrics/text",
    ):
        assert path in text
    assert '{"status":"ok"}' in text
    assert "404" in text
    assert "system.logs" in text
    assert "Authorization: Bearer" in text
    assert "exec -T -e TLDW_OPERATOR_TOKEN app" in text
    assert "http://127.0.0.1:8000" not in text
    assert "TLDW_APP_IMAGE" in text
    assert "TLDW_ROLLBACK_IMAGE" in text
    assert "pg_restore --list" in text
    assert "redis-check-rdb" in text
    assert "archive inspection" in text.lower()
    assert "disposable restore drill" in text.lower()


def test_executable_operator_api_checks_use_declared_routes() -> None:
    commands = _shell_blocks(RUNBOOK.read_text(encoding="utf-8"))
    match = re.search(r"for path in ([^;]+); do", commands)
    assert match is not None
    executable_paths = set(match.group(1).split())

    assert executable_paths == {"/api/v1/health", "/api/v1/metrics/text"}
    assert executable_paths <= _declared_operator_api_paths()


def test_monitoring_runbook_requires_operator_inputs_and_explains_networking() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    commands = _shell_blocks(text)

    assert "ALERTMANAGER_CONFIG=/" in commands
    assert 'test "${ALERTMANAGER_CONFIG#/}" != "$ALERTMANAGER_CONFIG"' in commands
    assert "PROMETHEUS_UID" in commands
    assert "PROMETHEUS_GID" in commands
    assert '--env-file "$PRODUCTION_ENV_FILE"' in commands
    assert "receiver" in text.lower()
    assert "outbound webhook" in text.lower()
    assert "network is not marked `internal`" in text


def test_monitoring_rotation_uses_read_scope_without_api_key_superuser_bypass() -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_api_keys
    from tldw_Server_API.app.api.v1.schemas.api_key_schemas import (
        APIKeyCreateRequest,
        APIKeyCreateResponse,
        APIKeyMetadata,
        APIKeyRevokeResponse,
    )
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import has_scope

    rotation = _rotation_block(RUNBOOK.read_text(encoding="utf-8"))
    route_contract = {
        (route.path, method): route.response_model for route in admin_api_keys.router.routes for method in route.methods
    }

    assert route_contract[("/users/{user_id}/api-keys", "POST")] is APIKeyCreateResponse
    assert route_contract[("/users/{user_id}/api-keys", "GET")] == list[APIKeyMetadata]
    assert route_contract[("/users/{user_id}/api-keys/{key_id}", "DELETE")] is APIKeyRevokeResponse
    assert {"id", "key"} <= set(APIKeyCreateResponse.model_fields)
    assert {"user_id", "key_id"} <= set(APIKeyRevokeResponse.model_fields)

    request = APIKeyCreateRequest(name="prometheus-rotation", scope="read")
    assert request.scope == "read"
    assert has_scope({"read"}, "read")
    assert not has_scope({"read"}, "write")
    assert has_scope({"service"}, "write")
    assert '"scope": "read"' in rotation
    assert '"scope": "service"' not in rotation


def test_monitoring_rotation_stages_restarts_verifies_then_revokes_old_key() -> None:
    rotation = _rotation_block(RUNBOOK.read_text(encoding="utf-8"))
    stage_function = rotation[rotation.index("stage_and_restart()") : rotation.index("wait_for_tldw_target()")]
    assert stage_function.index("run --rm --no-deps metrics-credential-init") < stage_function.index(
        "restart prometheus"
    )

    transaction = rotation[rotation.index("TLDW_NEW_METRICS_KEY_RESPONSE=") :]
    ordered_transaction = (
        'method="POST"',
        'mv -f -- "$TLDW_NEW_METRICS_API_KEY_FILE" "$TLDW_METRICS_API_KEY_FILE"',
        "stage_and_restart",
        "wait_for_tldw_target",
        'revoke_metrics_key "$TLDW_OLD_METRICS_API_KEY_ID"',
    )

    positions = [transaction.index(item) for item in ordered_transaction]
    assert positions == sorted(positions)
    assert "/api/v1/targets?state=active" in rotation
    assert 'chmod 600 "$TLDW_NEW_METRICS_API_KEY_FILE"' in transaction
    assert "TLDW_ADMIN_TOKEN" in rotation
    assert "TLDW_OPERATOR_TOKEN" not in rotation
    assert "reload prometheus" not in rotation


def test_monitoring_rotation_installs_cleanup_before_key_activation_and_disarms_after_success() -> None:
    rotation = _rotation_block(RUNBOOK.read_text(encoding="utf-8"))
    cleanup = rotation[
        rotation.index("cleanup_metrics_rotation()") : rotation.index("trap cleanup_metrics_rotation EXIT")
    ]

    assert rotation.index("trap cleanup_metrics_rotation EXIT") < rotation.index("TLDW_NEW_METRICS_KEY_RESPONSE=")
    for required_cleanup in (
        "lookup_metrics_key_id_by_name",
        'install -m 600 "$TLDW_OLD_METRICS_API_KEY_FILE"',
        "stage_and_restart && wait_for_tldw_target",
        'revoke_metrics_key "$cleanup_key_id"',
        "manual recovery required",
    ):
        assert required_cleanup in cleanup

    success = rotation[rotation.index("TLDW_NEW_METRICS_KEY_RESPONSE=") :]
    ordered_success = (
        "stage_and_restart",
        "wait_for_tldw_target",
        'revoke_metrics_key "$TLDW_OLD_METRICS_API_KEY_ID"',
        "TLDW_ROTATION_COMMITTED=1",
        "trap - EXIT",
    )
    positions = [success.index(item) for item in ordered_success]
    assert positions == sorted(positions)


def test_monitoring_rotation_failure_restages_old_key_before_revoking_new_key() -> None:
    rotation = _rotation_block(RUNBOOK.read_text(encoding="utf-8"))
    rollback = rotation[
        rotation.index("cleanup_metrics_rotation()") : rotation.index("trap cleanup_metrics_rotation EXIT")
    ]
    ordered_rollback = (
        'install -m 600 "$TLDW_OLD_METRICS_API_KEY_FILE" "$TLDW_ROLLBACK_METRICS_API_KEY_FILE"',
        'mv -f -- "$TLDW_ROLLBACK_METRICS_API_KEY_FILE" "$TLDW_METRICS_API_KEY_FILE"',
        "stage_and_restart && wait_for_tldw_target",
        'revoke_metrics_key "$cleanup_key_id"',
    )

    positions = [rollback.index(item) for item in ordered_rollback]
    assert positions == sorted(positions)


def test_monitoring_retirement_unmounts_ephemeral_credentials_without_volume_deletion() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    retirement = next(
        block
        for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
        if "docker-compose.production.yml" in block and re.search(r"\sdown(?:\s|$)", block)
    )

    assert re.search(r"\sdown(?:\s|$)", retirement)
    assert " -v" not in retirement


def test_reference_runbook_assigns_each_deferred_boundary_to_the_right_task() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    followups = {
        task: line.lower()
        for task, line in re.findall(
            r"^- `(TASK-(?:13013\.[79]|13144))` ([^\n]+)$",
            text,
            flags=re.MULTILINE,
        )
    }

    assert set(followups) == {"TASK-13013.7", "TASK-13013.9", "TASK-13144"}
    assert "provenance" in followups["TASK-13013.7"]
    for boundary in ("capacity", "restore-time", "soak"):
        assert boundary in followups["TASK-13013.9"]
    assert "client" in followups["TASK-13144"]
    assert "identity" in followups["TASK-13144"]


def test_existing_deployment_guides_link_to_the_reference_and_label_legacy() -> None:
    for path in LINKING_DOCS:
        text = path.read_text(encoding="utf-8")
        assert _reference_link_target(path) == RUNBOOK.resolve()
        assert "non-production" in text.lower(), path


def test_reverse_proxy_guide_uses_the_checked_in_sample_paths() -> None:
    for path in (
        Path("Docs/Deployment/Reverse_Proxy_Examples.md"),
        Path("Dockerfiles/README.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "Helper_Scripts/Samples/Caddy/Caddyfile.compose" in text, path
        assert "Helper_Scripts/Samples/Nginx/nginx.conf" in text, path
        assert "`Samples/Nginx/nginx.conf`" not in text, path


def test_published_deployment_guides_match_their_sources() -> None:
    for source in PUBLISHED_MIRRORS:
        published = Path("Docs/Published") / source.relative_to("Docs")
        assert published.read_bytes() == source.read_bytes(), source
