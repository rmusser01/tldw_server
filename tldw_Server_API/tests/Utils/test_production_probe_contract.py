from __future__ import annotations

from pathlib import Path

import yaml

DOCKER_PROBE_FILES = (
    Path("Dockerfiles/Dockerfile.prod"),
    Path("Dockerfiles/docker-compose.yml"),
    Path("Dockerfiles/docker-compose.single-user.yml"),
    Path("Dockerfiles/docker-compose.multi-user-postgres.yml"),
    Path("Dockerfiles/docker-compose.host-storage.yml"),
    Path("Dockerfiles/docker-compose.production.yml"),
)
KUBERNETES_APP = Path("Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml")
PROMETHEUS_CONFIG = Path("Dockerfiles/Monitoring/prometheus.yml")
MONITORING_COMPOSE = Path("Dockerfiles/Monitoring/docker-compose.monitoring.yml")
PRODUCTION_MONITORING_COMPOSE = Path(
    "Dockerfiles/Monitoring/docker-compose.production.yml"
)


def _yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_every_docker_application_probe_uses_loopback_internal_readiness() -> None:
    for path in DOCKER_PROBE_FILES:
        text = path.read_text(encoding="utf-8")
        assert "http://localhost:8000/ready" not in text, path
        assert "http://localhost:8000/internal/ready" in text, path


def test_kubernetes_readiness_is_container_local_and_liveness_stays_public() -> None:
    documents = tuple(yaml.safe_load_all(KUBERNETES_APP.read_text(encoding="utf-8")))
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    readiness = container["readinessProbe"]
    command = " ".join(readiness["exec"]["command"])
    assert "httpGet" not in readiness
    assert "http://localhost:8000/internal/ready" in command
    assert container["livenessProbe"]["httpGet"] == {
        "path": "/health",
        "port": 8000,
    }


def test_prometheus_uses_scoped_bearer_credential_file() -> None:
    config = _yaml(PROMETHEUS_CONFIG)
    job = next(
        item for item in config["scrape_configs"] if item["job_name"] == "tldw_server"
    )

    assert job["metrics_path"] == "/api/v1/metrics/text"
    assert job["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/run/secrets/tldw_metrics_api_key",
    }


def test_monitoring_compose_mounts_operator_credential_read_only() -> None:
    compose = _yaml(MONITORING_COMPOSE)
    volumes = compose["services"]["prometheus"]["volumes"]
    credential_mount = (
        "${TLDW_METRICS_API_KEY_FILE:?Set a mode-0600 API-key file whose principal "
        "has system.logs}:/run/secrets/tldw_metrics_api_key:ro"
    )

    assert credential_mount in volumes
    tracked_text = PROMETHEUS_CONFIG.read_text(
        encoding="utf-8"
    ) + MONITORING_COMPOSE.read_text(encoding="utf-8")
    assert "Bearer " not in tracked_text
    assert "tldw_sk_" not in tracked_text


def test_production_monitoring_joins_reference_edge_and_binds_uis_to_loopback() -> None:
    compose = _yaml(PRODUCTION_MONITORING_COMPOSE)

    assert compose["networks"]["default"] == {
        "external": True,
        "name": "tldw-production_edge",
    }
    for service_name in ("prometheus", "alertmanager", "grafana"):
        for published_port in compose["services"][service_name]["ports"]:
            assert str(published_port).startswith("127.0.0.1:")

    prometheus = compose["services"]["prometheus"]
    credential_mount = (
        "${TLDW_METRICS_API_KEY_FILE:?Set a mode-0600 API-key file whose principal "
        "has system.logs}:/run/secrets/tldw_metrics_api_key:ro"
    )
    assert credential_mount in prometheus["volumes"]
