from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app

EXPECTED_SPLIT_ADMIN_OPERATIONS: set[tuple[str, str]] = {
    ("POST", "/api/v1/admin/kanban/fts/{action}"),
    ("GET", "/api/v1/admin/roles"),
    ("POST", "/api/v1/admin/roles"),
    ("DELETE", "/api/v1/admin/roles/{role_id}"),
    ("GET", "/api/v1/admin/roles/{role_id}/permissions"),
    ("GET", "/api/v1/admin/permissions/tools"),
    ("POST", "/api/v1/admin/permissions/tools"),
    ("DELETE", "/api/v1/admin/permissions/tools/{perm_name}"),
    ("POST", "/api/v1/admin/roles/{role_id}/permissions/tools"),
    ("DELETE", "/api/v1/admin/roles/{role_id}/permissions/tools/{tool_name}"),
    ("GET", "/api/v1/admin/roles/{role_id}/permissions/tools"),
    ("POST", "/api/v1/admin/roles/{role_id}/permissions/tools/batch"),
    ("POST", "/api/v1/admin/roles/{role_id}/permissions/tools/batch/revoke"),
    ("POST", "/api/v1/admin/roles/{role_id}/permissions/tools/prefix/grant"),
    ("POST", "/api/v1/admin/roles/{role_id}/permissions/tools/prefix/revoke"),
    ("GET", "/api/v1/admin/roles/matrix"),
    ("GET", "/api/v1/admin/roles/matrix-boolean"),
    ("GET", "/api/v1/admin/permissions/categories"),
    ("GET", "/api/v1/admin/permissions"),
    ("POST", "/api/v1/admin/permissions"),
    ("POST", "/api/v1/admin/roles/{role_id}/permissions/{permission_id}"),
    ("DELETE", "/api/v1/admin/roles/{role_id}/permissions/{permission_id}"),
    ("GET", "/api/v1/admin/users/{user_id}/roles"),
    ("POST", "/api/v1/admin/users/{user_id}/roles/{role_id}"),
    ("DELETE", "/api/v1/admin/users/{user_id}/roles/{role_id}"),
    ("GET", "/api/v1/admin/users/{user_id}/overrides"),
    ("POST", "/api/v1/admin/users/{user_id}/overrides"),
    ("DELETE", "/api/v1/admin/users/{user_id}/overrides/{permission_id}"),
    ("GET", "/api/v1/admin/users/{user_id}/effective-permissions"),
    ("GET", "/api/v1/admin/roles/{role_id}/permissions/effective"),
    ("GET", "/api/v1/admin/rate-limits"),
    ("POST", "/api/v1/admin/roles/{role_id}/rate-limits"),
    ("DELETE", "/api/v1/admin/roles/{role_id}/rate-limits"),
    ("POST", "/api/v1/admin/users/{user_id}/rate-limits"),
    ("GET", "/api/v1/admin/backups"),
    ("POST", "/api/v1/admin/backups"),
    ("POST", "/api/v1/admin/backups/{backup_id}/restore"),
    ("GET", "/api/v1/admin/backup-schedules"),
    ("POST", "/api/v1/admin/backup-schedules"),
    ("PATCH", "/api/v1/admin/backup-schedules/{schedule_id}"),
    ("POST", "/api/v1/admin/backup-schedules/{schedule_id}/pause"),
    ("POST", "/api/v1/admin/backup-schedules/{schedule_id}/resume"),
    ("DELETE", "/api/v1/admin/backup-schedules/{schedule_id}"),
    ("GET", "/api/v1/admin/retention-policies"),
    ("POST", "/api/v1/admin/retention-policies/{policy_key}/preview"),
    ("PUT", "/api/v1/admin/retention-policies/{policy_key}"),
    ("GET", "/api/v1/admin/maintenance"),
    ("PUT", "/api/v1/admin/maintenance"),
    ("GET", "/api/v1/admin/feature-flags"),
    ("PUT", "/api/v1/admin/feature-flags/{flag_key}"),
    ("DELETE", "/api/v1/admin/feature-flags/{flag_key}"),
    ("GET", "/api/v1/admin/incidents"),
    ("POST", "/api/v1/admin/incidents"),
    ("PATCH", "/api/v1/admin/incidents/{incident_id}"),
    ("POST", "/api/v1/admin/incidents/{incident_id}/events"),
    ("DELETE", "/api/v1/admin/incidents/{incident_id}"),
    ("POST", "/api/v1/admin/llm-usage/pricing/reload"),
    ("POST", "/api/v1/admin/chat/model-aliases/reload"),
    ("GET", "/api/v1/admin/monitoring/alert-rules"),
    ("POST", "/api/v1/admin/monitoring/alert-rules"),
    ("DELETE", "/api/v1/admin/monitoring/alert-rules/{rule_id}"),
    ("POST", "/api/v1/admin/monitoring/alerts/{alert_identity}/assign"),
    ("POST", "/api/v1/admin/monitoring/alerts/{alert_identity}/snooze"),
    ("POST", "/api/v1/admin/monitoring/alerts/{alert_identity}/escalate"),
    ("GET", "/api/v1/admin/monitoring/alerts/history"),
    ("GET", "/api/v1/admin/errors/breakdown"),
    ("GET", "/api/v1/admin/rate-limits/summary"),
}


def _load_openapi_spec(monkeypatch, *, tmp_path: Path):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("ENABLE_OPENAPI", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key-openapi")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users_test_admin_openapi.db'}")
    monkeypatch.delenv("tldw_production", raising=False)
    with TestClient(app) as client:
        res = client.get("/openapi.json")
        assert res.status_code == 200
        spec = res.json()
    assert "paths" in spec
    return spec


def test_admin_split_openapi_contains_expected_operations(monkeypatch, tmp_path) -> None:
    spec = _load_openapi_spec(monkeypatch, tmp_path=tmp_path)
    paths = spec["paths"]
    actual_operations: set[tuple[str, str]] = set()
    for path, operations in paths.items():
        if not path.startswith("/api/v1/admin/"):
            continue
        for method in operations:
            upper = method.upper()
            if upper in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                actual_operations.add((upper, path))

    missing = sorted(EXPECTED_SPLIT_ADMIN_OPERATIONS - actual_operations)
    assert not missing, f"Missing admin split OpenAPI operations: {missing}"


def test_admin_split_openapi_schema_contracts(monkeypatch, tmp_path) -> None:
    spec = _load_openapi_spec(monkeypatch, tmp_path=tmp_path)
    paths = spec["paths"]

    roles_get = paths["/api/v1/admin/roles"]["get"]
    roles_schema = roles_get["responses"]["200"]["content"]["application/json"]["schema"]
    assert roles_schema["type"] == "array"
    assert roles_schema["items"]["$ref"].endswith("/RoleResponse")

    backups_get = paths["/api/v1/admin/backups"]["get"]
    backups_schema = backups_get["responses"]["200"]["content"]["application/json"]["schema"]
    assert backups_schema["$ref"].endswith("/BackupListResponse")

    backup_schedules_get = paths["/api/v1/admin/backup-schedules"]["get"]
    backup_schedules_schema = backup_schedules_get["responses"]["200"]["content"]["application/json"]["schema"]
    assert backup_schedules_schema["$ref"].endswith("/BackupScheduleListResponse")

    backup_schedules_post = paths["/api/v1/admin/backup-schedules"]["post"]
    backup_schedules_post_schema = backup_schedules_post["responses"]["200"]["content"]["application/json"]["schema"]
    assert backup_schedules_post_schema["$ref"].endswith("/BackupScheduleMutationResponse")

    maintenance_put = paths["/api/v1/admin/maintenance"]["put"]
    maintenance_schema = maintenance_put["responses"]["200"]["content"]["application/json"]["schema"]
    assert maintenance_schema["$ref"].endswith("/MaintenanceState")

    rate_limits_get = paths["/api/v1/admin/rate-limits"]["get"]
    rate_limits_get_schema = rate_limits_get["responses"]["200"]["content"]["application/json"]["schema"]
    assert rate_limits_get_schema["type"] == "array"
    assert rate_limits_get_schema["items"]["$ref"].endswith("/RateLimitResponse")

    rate_limit_post = paths["/api/v1/admin/roles/{role_id}/rate-limits"]["post"]
    rate_limit_schema = rate_limit_post["responses"]["200"]["content"]["application/json"]["schema"]
    assert rate_limit_schema["$ref"].endswith("/RateLimitResponse")

    monitoring_rules_get = paths["/api/v1/admin/monitoring/alert-rules"]["get"]
    monitoring_rules_schema = monitoring_rules_get["responses"]["200"]["content"]["application/json"]["schema"]
    assert monitoring_rules_schema["$ref"].endswith("/AdminAlertRuleListResponse")

    monitoring_assign_post = paths["/api/v1/admin/monitoring/alerts/{alert_identity}/assign"]["post"]
    monitoring_assign_schema = monitoring_assign_post["responses"]["200"]["content"]["application/json"]["schema"]
    assert monitoring_assign_schema["$ref"].endswith("/AdminAlertStateMutationResponse")

    monitoring_history_get = paths["/api/v1/admin/monitoring/alerts/history"]["get"]
    monitoring_history_schema = monitoring_history_get["responses"]["200"]["content"]["application/json"]["schema"]
    assert monitoring_history_schema["$ref"].endswith("/AdminAlertHistoryListResponse")


def test_admin_root_module_has_no_route_handlers() -> None:
    root_path = Path(__file__).resolve().parents[2] / "app" / "api" / "v1" / "endpoints" / "admin" / "__init__.py"
    content = root_path.read_text(encoding="utf-8")
    assert "@router.get(" not in content
    assert "@router.post(" not in content
    assert "@router.put(" not in content
    assert "@router.patch(" not in content
    assert "@router.delete(" not in content
