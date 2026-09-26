from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPOSE = Path("Dockerfiles/docker-compose.production.yml")
MONITORING_COMPOSE = Path("Dockerfiles/Monitoring/docker-compose.production.yml")
CADDYFILE = Path("Dockerfiles/Production/Caddyfile")
ENV_EXAMPLE = Path("Dockerfiles/production.env.example")

DENIED_PATHS = (
    "/internal/ready",
    "/ready",
    "/health/ready",
    "/api/v1/healthz",
    "/api/v1/readyz",
    "/setup",
    "/setup/*",
    "/api/v1/setup",
    "/api/v1/setup/*",
)

EXPECTED_ENV_NAMES = {
    "TLDW_PUBLIC_DOMAIN",
    "TLDW_ACME_EMAIL",
    "ALLOWED_ORIGINS",
    "JWT_SECRET_KEY",
    "SESSION_ENCRYPTION_KEY",
    "POSTGRES_USER",
    "POSTGRES_DB",
    "POSTGRES_PASSWORD",
    "DATABASE_URL",
    "REDIS_PASSWORD",
    "REDIS_URL",
    "ADMIN_USERNAME",
    "ADMIN_PASSWORD",
    "ADMIN_EMAIL",
    "TLDW_EXISTING_INSTALLATION",
    "TLDW_SETUP_COMPLETED",
    "TLDW_EDGE_SUBNET",
    "TLDW_BACKEND_SUBNET",
    "TLDW_APP_IMAGE",
    "TLDW_ROLLBACK_IMAGE",
    "CADDY_IMAGE",
    "POSTGRES_IMAGE",
    "REDIS_IMAGE",
    "PROMETHEUS_IMAGE",
    "ALERTMANAGER_IMAGE",
    "GRAFANA_IMAGE",
    "TLDW_BACKUP_DIR",
}
EXPECTED_SERVICES = {"preflight", "caddy", "app", "postgres", "redis"}
EXPECTED_NETWORKS = {"edge", "backend"}
EXPECTED_VOLUMES = {
    "app-data",
    "postgres_data",
    "redis_data",
    "caddy_data",
    "caddy_config",
}


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_only_caddy_publishes_ports() -> None:
    services = _compose()["services"]

    assert services["caddy"]["ports"] == ["80:80", "443:443"]
    for name in ("preflight", "app", "postgres", "redis"):
        assert "ports" not in services[name]


def test_compose_project_and_resource_sets_are_exact() -> None:
    compose = _compose()

    assert compose["name"] == "tldw-production"
    assert set(compose["services"]) == EXPECTED_SERVICES
    assert set(compose["networks"]) == EXPECTED_NETWORKS
    assert set(compose["volumes"]) == EXPECTED_VOLUMES


def test_network_membership_and_backend_isolation_are_exact() -> None:
    compose = _compose()

    assert compose["networks"]["backend"]["internal"] is True
    assert set(compose["services"]["caddy"]["networks"]) == {"edge"}
    assert set(compose["services"]["app"]["networks"]) == {"edge", "backend"}
    assert set(compose["services"]["postgres"]["networks"]) == {"backend"}
    assert set(compose["services"]["redis"]["networks"]) == {"backend"}


def test_app_enforces_production_mode_and_bounded_proxy_trust() -> None:
    environment = _compose()["services"]["app"]["environment"]
    edge_subnet = "${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}"

    assert environment["AUTH_MODE"] == "multi_user"
    assert environment["tldw_production"] == "true"
    assert environment["TLDW_SETUP_ALLOW_REMOTE"] == "0"
    assert environment["AUTH_TRUST_X_FORWARDED_FOR"] == "true"
    assert environment["RG_CLIENT_IP_HEADER"] == "X-Forwarded-For"
    assert environment["MCP_TRUST_X_FORWARDED"] == "true"
    for name in (
        "FORWARDED_ALLOW_IPS",
        "AUTH_TRUSTED_PROXY_IPS",
        "TLDW_TRUSTED_PROXIES",
        "RG_TRUSTED_PROXIES",
        "MCP_TRUSTED_PROXY_IPS",
    ):
        assert environment[name] == edge_subnet


def test_preflight_is_offline_and_blocks_app_startup_on_failure() -> None:
    services = _compose()["services"]

    assert services["preflight"]["network_mode"] == "none"
    assert services["preflight"]["restart"] == "no"
    assert services["app"]["depends_on"]["preflight"] == {"condition": "service_completed_successfully"}


def test_root_preflight_is_confined_to_read_only_inputs() -> None:
    preflight = _compose()["services"]["preflight"]

    assert preflight["user"] == "0:0"
    assert preflight["read_only"] is True
    assert preflight["cap_drop"] == ["ALL"]
    assert preflight["security_opt"] == ["no-new-privileges:true"]
    assert all(str(volume).endswith(":ro") for volume in preflight["volumes"])


def test_preflight_consumes_injected_environment_without_raw_env_mount() -> None:
    preflight = _compose()["services"]["preflight"]

    assert preflight["env_file"] == [
        {
            "path": "${TLDW_ENV_FILE:?Set TLDW_ENV_FILE to the validated absolute raw env path}",
            "required": True,
            "format": "raw",
        }
    ]
    assert preflight["command"] == [
        "--from-environment",
        "--compose-file",
        "/run/tldw/docker-compose.production.yml",
        "--proxy-file",
        "/run/tldw/Caddyfile",
        "--runtime-backup-dir",
        "/backups",
    ]
    assert all("/run/tldw/production.env" not in str(volume) for volume in preflight["volumes"])


def test_persistent_service_mounts_are_exact() -> None:
    services = _compose()["services"]

    assert set(services["app"]["volumes"]) == {"app-data:/app/Databases"}
    assert set(services["postgres"]["volumes"]) == {"postgres_data:/var/lib/postgresql"}
    assert set(services["redis"]["volumes"]) == {"redis_data:/data"}
    assert set(services["caddy"]["volumes"]) == {
        "./Production/Caddyfile:/etc/caddy/Caddyfile:ro",
        "caddy_data:/data",
        "caddy_config:/config",
    }


def test_caddy_receives_only_required_inputs_and_read_only_config() -> None:
    caddy = _compose()["services"]["caddy"]

    assert caddy["environment"] == {
        "TLDW_PUBLIC_DOMAIN": "${TLDW_PUBLIC_DOMAIN:?Set TLDW_PUBLIC_DOMAIN}",
        "TLDW_ACME_EMAIL": "${TLDW_ACME_EMAIL:?Set TLDW_ACME_EMAIL}",
    }
    assert "./Production/Caddyfile:/etc/caddy/Caddyfile:ro" in caddy["volumes"]


def test_stateful_services_require_external_credentials() -> None:
    services = _compose()["services"]
    redis_command = " ".join(services["redis"]["command"])
    redis_healthcheck = " ".join(services["redis"]["healthcheck"]["test"])

    assert "requirepass %s" in redis_command
    assert "$$REDIS_PASSWORD" in redis_command
    assert "REDISCLI_AUTH" in redis_healthcheck
    assert "$$REDIS_PASSWORD" in redis_healthcheck
    assert "POSTGRES_PASSWORD" not in services["postgres"].get("environment", {})


def test_images_are_required_external_inputs() -> None:
    services = _compose()["services"]

    assert services["preflight"]["image"] == (
        "${TLDW_APP_IMAGE:?Set TLDW_APP_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["app"]["image"] == (
        "${TLDW_APP_IMAGE:?Set TLDW_APP_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["caddy"]["image"] == (
        "${CADDY_IMAGE:?Set CADDY_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["postgres"]["image"] == (
        "${POSTGRES_IMAGE:?Set POSTGRES_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["redis"]["image"] == (
        "${REDIS_IMAGE:?Set REDIS_IMAGE to version tag plus @sha256 digest}"
    )


def test_monitoring_images_require_tag_plus_digest_inputs() -> None:
    """The companion stack uses the same explicit immutable-image contract."""
    services = yaml.safe_load(MONITORING_COMPOSE.read_text(encoding="utf-8"))["services"]

    assert services["metrics-credential-init"]["image"] == (
        "${TLDW_APP_IMAGE:?Set TLDW_APP_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["prometheus"]["image"] == (
        "${PROMETHEUS_IMAGE:?Set PROMETHEUS_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["alertmanager"]["image"] == (
        "${ALERTMANAGER_IMAGE:?Set ALERTMANAGER_IMAGE to version tag plus @sha256 digest}"
    )
    assert services["grafana"]["image"] == (
        "${GRAFANA_IMAGE:?Set GRAFANA_IMAGE to version tag plus @sha256 digest}"
    )


def test_topology_has_no_builds_container_names_or_docker_socket() -> None:
    compose = _compose()

    for service in compose["services"].values():
        assert "build" not in service
        assert "container_name" not in service
        assert all("/var/run/docker.sock" not in str(volume) for volume in service.get("volumes", []))


def test_app_healthcheck_uses_loopback_internal_readiness() -> None:
    test_command = " ".join(_compose()["services"]["app"]["healthcheck"]["test"])

    assert "http://localhost:8000/internal/ready" in test_command


def test_names_only_env_example_has_exact_empty_assignments() -> None:
    assignments = [
        line for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")
    ]

    assert assignments
    assert all(line.endswith("=") for line in assignments)
    assert {line[:-1] for line in assignments} == EXPECTED_ENV_NAMES


def test_caddy_denies_private_legacy_and_setup_routes_before_proxy() -> None:
    text = CADDYFILE.read_text(encoding="utf-8")
    matcher = next(line for line in text.splitlines() if "@private_control path" in line)

    assert text.index("respond @private_control 404") < text.index("reverse_proxy app:8000")
    for path in DENIED_PATHS:
        assert path in matcher
    for public_path in ("/health", "/metrics", "/api/v1/health"):
        assert public_path not in matcher.split()


def test_caddy_overwrites_client_identity_headers() -> None:
    text = CADDYFILE.read_text(encoding="utf-8")

    assert "header_up X-Forwarded-For {remote_host}" in text
    assert "header_up X-Real-IP {remote_host}" in text
    assert "header_up X-Forwarded-Proto https" in text
