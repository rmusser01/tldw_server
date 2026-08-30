from __future__ import annotations

from pathlib import Path

RUNBOOK = Path("Docs/Deployment/Production_Reference_Deployment.md")
PUBLISHED_RUNBOOK = Path(
    "Docs/Published/Deployment/Production_Reference_Deployment.md"
)
LINKING_DOCS = (
    Path("Docs/Deployment/First_Time_Production_Setup.md"),
    Path("Docs/Deployment/Long_Term_Admin_Guide.md"),
    Path("Docs/Deployment/Reverse_Proxy_Examples.md"),
    Path("Dockerfiles/README.md"),
)
RUNBOOK_MARKERS = (
    "make production-preflight",
    "make production-deploy",
    "make production-rollback",
    "chmod 600",
    "pg_restore --list",
    "redis-check-rdb",
    "system.logs",
    "TLDW_METRICS_API_KEY_FILE",
    "Dockerfiles/Monitoring/docker-compose.production.yml",
    "tldw-production_edge",
    "host loopback",
    "disposable restore drill",
    "TASK-13013.7",
    "TASK-13013.9",
    "TASK-13144",
)
REQUIRED_CONCEPTS = (
    "only Caddy publishes",
    '{"status":"ok"}',
    "/internal/ready",
    "/api/v1/metrics/text",
    "raw env",
    "immutable",
    "PostgreSQL",
    "Redis",
    "app-data",
    "secret/config file",
    "upgrade",
    "restore-backed rollback",
    "archive inspection",
    "environment-only",
)


def test_reference_runbook_covers_the_fail_closed_operator_workflow() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    for marker in (*RUNBOOK_MARKERS, *REQUIRED_CONCEPTS):
        assert marker.lower() in text.lower(), marker
    for path in (
        "/ready",
        "/health/ready",
        "/api/v1/healthz",
        "/api/v1/readyz",
        "/setup",
    ):
        assert path in text
    assert "404" in text
    assert "Authorization: Bearer" in text
    assert "exec -T -e TLDW_OPERATOR_TOKEN app" in text
    assert "http://127.0.0.1:8000" not in text
    assert "TLDW_APP_IMAGE" in text
    assert "TLDW_ROLLBACK_IMAGE" in text


def test_existing_deployment_guides_link_to_the_reference_and_label_legacy() -> None:
    for path in LINKING_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "Production_Reference_Deployment.md" in text, path
        assert "non-production" in text.lower(), path


def test_reverse_proxy_guide_uses_the_checked_in_sample_paths() -> None:
    text = Path("Docs/Deployment/Reverse_Proxy_Examples.md").read_text(
        encoding="utf-8"
    )

    assert "Helper_Scripts/Samples/Caddy/Caddyfile.compose" in text
    assert "Helper_Scripts/Samples/Nginx/nginx.conf" in text


def test_published_runbook_matches_the_source() -> None:
    assert PUBLISHED_RUNBOOK.read_bytes() == RUNBOOK.read_bytes()
