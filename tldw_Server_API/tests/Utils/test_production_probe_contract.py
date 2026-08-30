from __future__ import annotations

from pathlib import Path

import yaml

DOCKERFILE = Path("Dockerfiles/Dockerfile.prod")
DOCKERFILES_ROOT = Path("Dockerfiles")
KUBERNETES_APP = Path("Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml")
PROMETHEUS_CONFIG = Path("Dockerfiles/Monitoring/prometheus.yml")
MONITORING_COMPOSE_FILES = (
    Path("Dockerfiles/Monitoring/docker-compose.monitoring.yml"),
    Path("Dockerfiles/Monitoring/docker-compose.production.yml"),
)


def _yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_every_docker_application_probe_uses_loopback_internal_readiness() -> None:
    checked_paths: set[Path] = set()
    for path in sorted(DOCKERFILES_ROOT.rglob("*.yml")):
        compose = _yaml(path)
        app = compose.get("services", {}).get("app", {})
        if not isinstance(app, dict) or "healthcheck" not in app:
            continue
        checked_paths.add(path)
        command = " ".join(str(part) for part in app["healthcheck"]["test"])
        assert "http://localhost:8000/internal/ready" in command, path
        assert "http://localhost:8000/ready" not in command, path

    assert checked_paths == {
        Path("Dockerfiles/docker-compose.yml"),
        Path("Dockerfiles/docker-compose.host-storage.yml"),
        Path("Dockerfiles/docker-compose.multi-user-postgres.yml"),
        Path("Dockerfiles/docker-compose.production.yml"),
        Path("Dockerfiles/docker-compose.single-user.yml"),
    }

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "http://localhost:8000/internal/ready" in dockerfile
    assert "http://localhost:8000/ready" not in dockerfile


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
    job = next(item for item in config["scrape_configs"] if item["job_name"] == "tldw_server")

    assert job["metrics_path"] == "/api/v1/metrics/text"
    assert job["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/run/secrets/tldw_metrics_api_key",
    }


def test_monitoring_composes_mount_operator_credential_and_bind_loopback() -> None:
    credential_mount = (
        "${TLDW_METRICS_API_KEY_FILE:?Set a mode-0600 API-key file whose principal "
        "has system.logs}:/run/secrets/tldw_metrics_api_key:ro"
    )

    for path in MONITORING_COMPOSE_FILES:
        compose = _yaml(path)
        assert credential_mount in compose["services"]["prometheus"]["volumes"], path
        for service_name in ("prometheus", "alertmanager", "grafana"):
            for published_port in compose["services"][service_name]["ports"]:
                assert str(published_port).startswith("127.0.0.1:"), (
                    path,
                    service_name,
                    published_port,
                )

    tracked_text = PROMETHEUS_CONFIG.read_text(encoding="utf-8") + "".join(
        path.read_text(encoding="utf-8") for path in MONITORING_COMPOSE_FILES
    )
    assert "Bearer " not in tracked_text
    assert "tldw_sk_" not in tracked_text


def test_production_monitoring_limits_edge_access_to_prometheus() -> None:
    compose = _yaml(MONITORING_COMPOSE_FILES[1])

    assert compose["networks"]["edge"] == {
        "external": True,
        "name": "tldw-production_edge",
    }
    assert compose["networks"]["monitoring"] == {}
    assert set(compose["services"]["prometheus"]["networks"]) == {
        "edge",
        "monitoring",
    }
    assert compose["services"]["alertmanager"]["networks"] == ["monitoring"]
    assert compose["services"]["grafana"]["networks"] == ["monitoring"]
