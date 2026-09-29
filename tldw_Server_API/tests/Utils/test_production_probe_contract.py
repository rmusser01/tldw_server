from __future__ import annotations

import json
import os
import shutil

# This contract test runs a locally resolved docker binary with fixed argv and no shell.
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml

DOCKERFILE = Path("Dockerfiles/Dockerfile.prod")
DOCKERFILES_ROOT = Path("Dockerfiles")
KUBERNETES_APP = Path("Helper_Scripts/Samples/Kubernetes/tldw-app-deployment.yaml")
PROMETHEUS_CONFIG = Path("Dockerfiles/Monitoring/prometheus.yml")
MONITORING_COMPOSE_FILES = (
    Path("Dockerfiles/Monitoring/docker-compose.monitoring.yml"),
    Path("Dockerfiles/Monitoring/docker-compose.production.yml"),
)
CREDENTIAL_VOLUME_NAME = "metrics_credential_tmpfs_v2"
LEGACY_CREDENTIAL_VOLUME_NAME = "metrics_credential"
EXPECTED_CREDENTIAL_VOLUME = {
    "driver": "local",
    "driver_opts": {
        "type": "tmpfs",
        "device": "tmpfs",
        "o": (
            "uid=0,gid=${PROMETHEUS_GID:?Set the numeric GID used by the pinned " "Prometheus image},mode=0710,size=64k"
        ),
    },
}


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


def test_monitoring_composes_stage_operator_credential_for_non_root_prometheus() -> None:
    for path in MONITORING_COMPOSE_FILES:
        compose = _yaml(path)
        services = compose["services"]
        init = services["metrics-credential-init"]
        prometheus = services["prometheus"]

        assert init["network_mode"] == "none", path
        assert "networks" not in init, path
        assert init["read_only"] is True, path
        assert init["restart"] == "no", path
        assert init["user"] == "0:0", path
        assert init["security_opt"] == ["no-new-privileges:true"], path
        assert init["cap_drop"] == ["ALL"], path
        assert set(init["cap_add"]) == {
            "CHOWN",
            "DAC_OVERRIDE",
            "SETGID",
            "SETUID",
        }, path
        assert ":?" in init["image"] and ":-" not in init["image"], path
        assert {key: init["environment"][key] for key in ("PROMETHEUS_UID", "PROMETHEUS_GID")} == {
            "PROMETHEUS_UID": "${PROMETHEUS_UID:?Set the numeric UID used by the pinned Prometheus image}",
            "PROMETHEUS_GID": "${PROMETHEUS_GID:?Set the numeric GID used by the pinned Prometheus image}",
        }, path

        init_volumes = init["volumes"]
        assert any(
            volume.startswith("${TLDW_METRICS_API_KEY_FILE:?")
            and volume.endswith(":/run/source/tldw_metrics_api_key:ro")
            for volume in init_volumes
        ), path
        assert f"{CREDENTIAL_VOLUME_NAME}:/run/staged" in init_volumes, path

        script = "\n".join(init["command"])
        for contract in (
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "source_info = os.fstat(source_fd)",
            "stat.S_ISREG(source_info.st_mode)",
            "stat.S_IMODE(source_info.st_mode) != 0o600",
            "MAX_CREDENTIAL_BYTES = 16 * 1024",
            "source_info.st_size > MAX_CREDENTIAL_BYTES",
            "MAX_CREDENTIAL_BYTES + 1 - copied",
            "copied != source_info.st_size",
            "os.fchmod(target_fd, 0o400)",
            "os.fchown(target_fd, uid, gid)",
            "os.replace(temporary, target)",
            "os.setgroups([])",
            "os.setgid(gid)",
            "os.setuid(uid)",
            "os.read(check_fd, 1)",
        ):
            assert contract in script, (path, contract)

        source_open = script[script.index("source_fd = os.open") : script.index("source_info = os.fstat")]
        assert source_open.index("os.O_NONBLOCK") < script.index("source_info = os.fstat"), path

        assert prometheus["user"] == (
            "${PROMETHEUS_UID:?Set the numeric UID used by the pinned Prometheus image}:"
            "${PROMETHEUS_GID:?Set the numeric GID used by the pinned Prometheus image}"
        ), path
        assert prometheus["depends_on"]["metrics-credential-init"] == {
            "condition": "service_completed_successfully"
        }, path
        assert f"{CREDENTIAL_VOLUME_NAME}:/run/secrets:ro" in prometheus["volumes"], path
        assert not any("TLDW_METRICS_API_KEY_FILE" in volume for volume in prometheus["volumes"]), path
        assert compose["volumes"][CREDENTIAL_VOLUME_NAME] == EXPECTED_CREDENTIAL_VOLUME, path
        assert LEGACY_CREDENTIAL_VOLUME_NAME not in compose["volumes"], path

        for service_name in ("prometheus", "alertmanager", "grafana"):
            for published_port in services[service_name]["ports"]:
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


@pytest.mark.unit
def test_monitoring_credential_storage_is_bounded_tmpfs_not_persistent_volume_data() -> None:
    """A new logical volume identity must prevent reuse of legacy disk-backed secret data."""
    for path in MONITORING_COMPOSE_FILES:
        compose = _yaml(path)
        credential_volume = compose["volumes"][CREDENTIAL_VOLUME_NAME]

        assert credential_volume == EXPECTED_CREDENTIAL_VOLUME, path
        assert LEGACY_CREDENTIAL_VOLUME_NAME not in compose["volumes"], path
        assert credential_volume["driver_opts"]["type"] == "tmpfs", path
        assert "size=64k" in credential_volume["driver_opts"]["o"], path
        assert "mode=0710" in credential_volume["driver_opts"]["o"], path


def test_production_alertmanager_requires_an_operator_owned_absolute_config() -> None:
    compose = _yaml(MONITORING_COMPOSE_FILES[1])
    config_mount = compose["services"]["alertmanager"]["volumes"][0]

    assert config_mount.startswith("${ALERTMANAGER_CONFIG:?")
    assert config_mount.endswith(":/etc/alertmanager/alertmanager.yml:ro")
    assert ":-" not in config_mount
    assert "alertmanager_webhook_only.yml" not in MONITORING_COMPOSE_FILES[1].read_text(encoding="utf-8")


@pytest.mark.integration
def test_production_monitoring_render_preserves_private_credential_boundary(tmp_path: Path) -> None:
    docker = shutil.which("docker")
    assert docker is not None, "docker compose is required for this integration contract"
    # The executable is locally resolved and every argv element is fixed.
    version = subprocess.run(  # nosec B603
        [docker, "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert version.returncode == 0, version.stderr

    metrics_key = tmp_path / "tldw-metrics-key"
    alertmanager_config = tmp_path / "tldw-alertmanager.yml"
    env = {
        **os.environ,
        "TLDW_APP_IMAGE": ("ghcr.io/example/tldw:0.1.0@sha256:" + "a" * 64),
        "PROMETHEUS_IMAGE": "prom/prometheus:v3.13.2@sha256:" + "b" * 64,
        "ALERTMANAGER_IMAGE": "prom/alertmanager:v0.34.0@sha256:" + "c" * 64,
        "GRAFANA_IMAGE": "grafana/grafana:13.2.1@sha256:" + "d" * 64,
        "PROMETHEUS_UID": "65534",
        "PROMETHEUS_GID": "65534",
        "TLDW_METRICS_API_KEY_FILE": str(metrics_key),
        "ALERTMANAGER_CONFIG": str(alertmanager_config),
        "GRAFANA_ADMIN_USER": "tldw-operator",
        "GRAFANA_ADMIN_PASSWORD": tmp_path.name,
    }
    # The executable is locally resolved and every argv element is fixed.
    rendered = subprocess.run(  # nosec B603
        [
            docker,
            "compose",
            "-f",
            str(MONITORING_COMPOSE_FILES[1]),
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert rendered.returncode == 0, rendered.stderr
    compose = json.loads(rendered.stdout)
    init = compose["services"]["metrics-credential-init"]
    prometheus = compose["services"]["prometheus"]
    alertmanager = compose["services"]["alertmanager"]
    assert init["network_mode"] == "none"
    init_source = next(volume for volume in init["volumes"] if volume["target"] == "/run/source/tldw_metrics_api_key")
    assert init_source["type"] == "bind"
    assert init_source["source"] == str(metrics_key)
    assert init_source["read_only"] is True
    assert prometheus["user"] == "65534:65534"
    staged_secret = next(volume for volume in prometheus["volumes"] if volume["target"] == "/run/secrets")
    assert staged_secret["type"] == "volume"
    assert staged_secret["source"] == CREDENTIAL_VOLUME_NAME
    assert staged_secret["read_only"] is True
    assert prometheus["depends_on"]["metrics-credential-init"]["condition"] == ("service_completed_successfully")
    assert alertmanager["volumes"][0]["source"] == str(alertmanager_config)
    assert compose["volumes"][CREDENTIAL_VOLUME_NAME]["driver"] == "local"
    assert compose["volumes"][CREDENTIAL_VOLUME_NAME]["driver_opts"] == {
        "device": "tmpfs",
        "o": "uid=0,gid=65534,mode=0710,size=64k",
        "type": "tmpfs",
    }


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
