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


def _reference_link_target(path: Path) -> Path:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\[[^]]+\]\(([^)]*Production_Reference_Deployment\.md)\)", text)
    assert match is not None, path
    return (path.parent / match.group(1)).resolve()


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
