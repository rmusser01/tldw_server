"""Offline validation for the production reference deployment."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSE_FILE = REPO_ROOT / "Dockerfiles/docker-compose.production.yml"
DEFAULT_PROXY_FILE = REPO_ROOT / "Dockerfiles/Production/Caddyfile"

REQUIRED_ENV_NAMES = (
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
    "TLDW_BACKUP_DIR",
)

SECRET_NAMES = (
    "JWT_SECRET_KEY",
    "SESSION_ENCRYPTION_KEY",
    "POSTGRES_PASSWORD",
    "REDIS_PASSWORD",
    "ADMIN_PASSWORD",
)

TRUST_ENVIRONMENT = (
    "FORWARDED_ALLOW_IPS",
    "AUTH_TRUSTED_PROXY_IPS",
    "TLDW_TRUSTED_PROXIES",
    "RG_TRUSTED_PROXIES",
    "MCP_TRUSTED_PROXY_IPS",
)

DENIED_PROXY_PATHS = (
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

APP_IMAGE_PATTERNS = (
    re.compile(r"^.+@sha256:[0-9a-f]{64}$"),
    re.compile(r"^.+:sha-[0-9a-f]{7,64}$"),
)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_THIRD_PARTY_VERSION = re.compile(r"(?:^|[^0-9])\d+\.\d+(?:\.\d+)?(?:[^0-9]|$)")
_DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_SAMPLE_MARKERS = ("example.com", "example.invalid", "localhost", "change-me")
_PLACEHOLDER_SECRETS = {
    "password",
    "postgres",
    "redis",
    "secret",
    "changeme",
    "change-me",
    "admin",
    "default",
}
_STATIC_IMAGE_INPUTS = {
    "preflight": "${TLDW_APP_IMAGE:?Set immutable TLDW_APP_IMAGE}",
    "app": "${TLDW_APP_IMAGE:?Set immutable TLDW_APP_IMAGE}",
    "caddy": "${CADDY_IMAGE:?Set exact CADDY_IMAGE version or digest}",
    "postgres": "${POSTGRES_IMAGE:?Set exact POSTGRES_IMAGE version or digest}",
    "redis": "${REDIS_IMAGE:?Set exact REDIS_IMAGE version or digest}",
}
_EXPECTED_NETWORKS = {
    "caddy": {"edge"},
    "app": {"edge", "backend"},
    "postgres": {"backend"},
    "redis": {"backend"},
}
_EXPECTED_VOLUMES = {
    "app-data",
    "postgres_data",
    "redis_data",
    "caddy_data",
    "caddy_config",
}
_EXPECTED_SERVICE_VOLUMES = {
    "preflight": {
        "./docker-compose.production.yml:/run/tldw/docker-compose.production.yml:ro",
        "./Production/Caddyfile:/run/tldw/Caddyfile:ro",
        "${TLDW_BACKUP_DIR:?Set absolute TLDW_BACKUP_DIR}:/backups:ro",
    },
    "caddy": {
        "./Production/Caddyfile:/etc/caddy/Caddyfile:ro",
        "caddy_data:/data",
        "caddy_config:/config",
    },
    "app": {"app-data:/app/Databases"},
    "postgres": {"postgres_data:/var/lib/postgresql"},
    "redis": {"redis_data:/data"},
}
_EXPECTED_CADDY_ENVIRONMENT = {
    "TLDW_PUBLIC_DOMAIN": "${TLDW_PUBLIC_DOMAIN:?Set TLDW_PUBLIC_DOMAIN}",
    "TLDW_ACME_EMAIL": "${TLDW_ACME_EMAIL:?Set TLDW_ACME_EMAIL}",
}
_EXPECTED_RAW_ENV_FILE = [
    {
        "path": "${TLDW_ENV_FILE:?Set TLDW_ENV_FILE to the validated absolute raw env path}",
        "required": True,
        "format": "raw",
    }
]


@dataclass(frozen=True, order=True)
class PreflightIssue:
    """One sanitized production-preflight failure."""

    code: str
    field: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    """Deterministic collection of production-preflight failures."""

    issues: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool:
        """Return whether every preflight invariant passed."""

        return not self.issues


def _issue(code: str, field: str, message: str) -> PreflightIssue:
    """Build an issue whose message never includes a candidate value."""

    return PreflightIssue(code=code, field=field, message=message)


def _sorted_issues(issues: list[PreflightIssue]) -> tuple[PreflightIssue, ...]:
    """Return stable de-duplicated issue output."""

    return tuple(sorted(set(issues)))


def load_raw_env(path: Path) -> dict[str, str]:
    """Parse a literal dotenv file without interpolation or shell evaluation."""

    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise ValueError(f"line {line_number}: export syntax is not supported")
        if "=" not in line:
            raise ValueError(f"line {line_number}: expected NAME=value")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"line {line_number}: invalid variable name")
        if name in values:
            raise ValueError(f"line {line_number}: duplicate variable name")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"line {line_number}: unmatched quote")
            value = value[1:-1]
        elif value[-1:] in {"'", '"'}:
            raise ValueError(f"line {line_number}: unmatched quote")
        values[name] = value
    return values


def _is_true(value: str) -> bool:
    """Return whether a production boolean is explicitly true."""

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_false(value: str) -> bool:
    """Return whether a production boolean is explicitly false."""

    return value.strip().lower() in {"0", "false", "no", "off"}


def _is_placeholder_secret(value: str) -> bool:
    """Return whether a secret is a recognized deployable placeholder."""

    normalized = value.strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    return (
        normalized in _PLACEHOLDER_SECRETS
        or compact in {"changeme", "testpassword", "password123", "defaultsecret"}
        or any(marker in normalized for marker in ("example", "placeholder"))
    )


def _validate_secrets(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate secret strength, placeholders, and accidental reuse."""

    issues: list[PreflightIssue] = []
    existing = _is_true(values.get("TLDW_EXISTING_INSTALLATION", ""))
    active_names = SECRET_NAMES if not existing else SECRET_NAMES[:-1]
    seen: dict[str, str] = {}
    for name in active_names:
        value = values.get(name, "")
        if not value:
            continue
        if _is_placeholder_secret(value):
            issues.append(_issue("placeholder_secret", name, "must not use a known placeholder"))
        elif len(value) < 32:
            issues.append(_issue("weak_secret", name, "must contain at least 32 characters"))
        if value in seen:
            issues.append(_issue("shared_secret", name, "must be independent from other secrets"))
        else:
            seen[value] = name
    return issues


def _validate_database_url(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate the PostgreSQL URL against separately supplied credentials."""

    raw = values.get("DATABASE_URL", "")
    if not raw:
        return []
    try:
        parsed = urlsplit(raw)
        username = unquote(parsed.username or "")
        password = unquote(parsed.password or "")
        database = unquote(parsed.path.lstrip("/"))
    except (TypeError, ValueError):
        return [_issue("invalid_url", "DATABASE_URL", "must be a valid PostgreSQL URL")]
    expected = (
        parsed.scheme in {"postgres", "postgresql"}
        and parsed.hostname == "postgres"
        and username == values.get("POSTGRES_USER")
        and password == values.get("POSTGRES_PASSWORD")
        and database == values.get("POSTGRES_DB")
    )
    if expected:
        return []
    return [
        _issue(
            "credential_mismatch",
            "DATABASE_URL",
            "must target postgres and match the separate database credentials",
        )
    ]


def _validate_redis_url(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate the Redis URL against the separate password."""

    raw = values.get("REDIS_URL", "")
    if not raw:
        return []
    try:
        parsed = urlsplit(raw)
        password = unquote(parsed.password or "")
    except (TypeError, ValueError):
        return [_issue("invalid_url", "REDIS_URL", "must be a valid Redis URL")]
    if parsed.scheme in {"redis", "rediss"} and parsed.hostname == "redis" and password == values.get("REDIS_PASSWORD"):
        return []
    return [
        _issue(
            "credential_mismatch",
            "REDIS_URL",
            "must target redis and match the separate Redis password",
        )
    ]


def _parse_origins(raw: str) -> list[str]:
    """Parse a JSON-list or comma-separated origin setting."""

    if raw.lstrip().startswith("["):
        loaded = json.loads(raw)
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("origins must be strings")
        return [item.strip() for item in loaded]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_domain_and_origins(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate TLS identity and HTTPS CORS origin alignment."""

    issues: list[PreflightIssue] = []
    domain = values.get("TLDW_PUBLIC_DOMAIN", "").strip().lower()
    contact = values.get("TLDW_ACME_EMAIL", "").strip().lower()
    if domain and (
        any(marker in domain for marker in _SAMPLE_MARKERS) or ":" in domain or "/" in domain or "." not in domain
    ):
        issues.append(_issue("sample_value", "TLDW_PUBLIC_DOMAIN", "must be a real DNS name"))
    if contact and ("@" not in contact or any(marker in contact for marker in _SAMPLE_MARKERS)):
        issues.append(_issue("sample_value", "TLDW_ACME_EMAIL", "must be a real contact address"))
    raw_origins = values.get("ALLOWED_ORIGINS", "")
    if not raw_origins:
        return issues
    try:
        origins = _parse_origins(raw_origins)
    except (ValueError, json.JSONDecodeError):
        return issues + [_issue("unsafe_origin", "ALLOWED_ORIGINS", "must be an explicit HTTPS origin list")]
    if not origins or "*" in origins:
        issues.append(_issue("unsafe_origin", "ALLOWED_ORIGINS", "must not contain a wildcard"))
        return issues
    for origin in origins:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            issues.append(_issue("unsafe_origin", "ALLOWED_ORIGINS", "must contain HTTPS origins only"))
            continue
        if domain and parsed.hostname.lower() != domain:
            issues.append(
                _issue(
                    "origin_mismatch",
                    "ALLOWED_ORIGINS",
                    "must match the configured public domain",
                )
            )
    return issues


def _network(value: str, field: str) -> tuple[ipaddress.IPv4Network | None, PreflightIssue | None]:
    """Parse one bounded private IPv4 network."""

    try:
        network = ipaddress.ip_network(value, strict=True)
    except ValueError:
        return None, _issue("unsafe_network", field, "must be a strict private IPv4 CIDR")
    if network.version != 4 or not network.is_private or network.prefixlen == 0:
        return None, _issue("unsafe_network", field, "must be a bounded private IPv4 CIDR")
    return network, None  # type: ignore[return-value]


def _validate_networks(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate distinct private edge and backend CIDRs."""

    issues: list[PreflightIssue] = []
    edge, edge_issue = _network(values.get("TLDW_EDGE_SUBNET", ""), "TLDW_EDGE_SUBNET")
    backend, backend_issue = _network(values.get("TLDW_BACKEND_SUBNET", ""), "TLDW_BACKEND_SUBNET")
    for issue in (edge_issue, backend_issue):
        if issue is not None:
            issues.append(issue)
    if edge is not None and backend is not None and edge.overlaps(backend):
        issues.append(
            _issue(
                "overlapping_network",
                "TLDW_BACKEND_SUBNET",
                "must not overlap the edge network",
            )
        )
    return issues


def _is_immutable_app_image(value: str) -> bool:
    """Return whether an application image is digest or commit pinned."""

    return any(pattern.fullmatch(value) for pattern in APP_IMAGE_PATTERNS)


def _is_exact_third_party_image(value: str) -> bool:
    """Return whether a third-party image uses a digest or full numeric tag."""

    if _DIGEST_IMAGE.fullmatch(value):
        return True
    last_segment = value.rsplit("/", 1)[-1]
    if ":" not in last_segment:
        return False
    tag = last_segment.rsplit(":", 1)[-1].lower()
    return tag != "latest" and bool(_THIRD_PARTY_VERSION.search(tag))


def _validate_images(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate current, rollback, and dependency image immutability."""

    issues: list[PreflightIssue] = []
    for field in ("TLDW_APP_IMAGE", "TLDW_ROLLBACK_IMAGE"):
        value = values.get(field, "")
        if value and not _is_immutable_app_image(value):
            issues.append(_issue("mutable_image", field, "must use a digest or commit-pinned tag"))
    if values.get("TLDW_APP_IMAGE") and values.get("TLDW_APP_IMAGE") == values.get("TLDW_ROLLBACK_IMAGE"):
        issues.append(
            _issue(
                "identical_images",
                "TLDW_ROLLBACK_IMAGE",
                "must differ from the target application image",
            )
        )
    for field in ("CADDY_IMAGE", "POSTGRES_IMAGE", "REDIS_IMAGE"):
        value = values.get(field, "")
        if value and not _is_exact_third_party_image(value):
            issues.append(
                _issue(
                    "inexact_third_party_image",
                    field,
                    "must use a digest or full numeric version tag",
                )
            )
    return issues


def _validate_setup(values: Mapping[str, str]) -> list[PreflightIssue]:
    """Validate completed setup and mutually exclusive bootstrap modes."""

    issues: list[PreflightIssue] = []
    if not _is_true(values.get("TLDW_SETUP_COMPLETED", "")):
        issues.append(_issue("setup_incomplete", "TLDW_SETUP_COMPLETED", "must be explicitly true"))
    raw_existing = values.get("TLDW_EXISTING_INSTALLATION", "")
    if not (_is_true(raw_existing) or _is_false(raw_existing)):
        issues.append(
            _issue(
                "invalid_boolean",
                "TLDW_EXISTING_INSTALLATION",
                "must be explicitly true or false",
            )
        )
        return issues
    bootstrap = ("ADMIN_USERNAME", "ADMIN_PASSWORD", "ADMIN_EMAIL")
    if _is_true(raw_existing):
        for field in bootstrap:
            if values.get(field, ""):
                issues.append(
                    _issue(
                        "unexpected_bootstrap_secret",
                        field,
                        "must be empty for an existing installation",
                    )
                )
    else:
        for field in bootstrap:
            if not values.get(field, ""):
                issues.append(_issue("missing_required", field, "must be provided"))
    return issues


def _validate_backup(
    values: Mapping[str, str],
    runtime_backup_dir: Path | None,
    *,
    require_writable: bool,
) -> list[PreflightIssue]:
    """Validate the operator backup destination without modifying it."""

    raw = values.get("TLDW_BACKUP_DIR", "")
    if not raw:
        return []
    configured = Path(raw)
    if not configured.is_absolute():
        return [_issue("unsafe_backup_path", "TLDW_BACKUP_DIR", "must be an absolute path")]
    banned = (Path("/app/Databases"), Path("/var/lib/postgresql"), Path("/data"))
    try:
        resolved = configured.resolve(strict=False)
    except OSError:
        resolved = configured
    if any(resolved == path or path in resolved.parents for path in banned):
        return [_issue("live_data_path", "TLDW_BACKUP_DIR", "must be separate from live data")]
    visible = runtime_backup_dir if runtime_backup_dir is not None else configured
    if not visible.exists():
        return [_issue("backup_unavailable", "TLDW_BACKUP_DIR", "must already exist")]
    if not visible.is_dir():
        return [_issue("backup_unavailable", "TLDW_BACKUP_DIR", "must be a directory")]
    if not require_writable:
        return []
    try:
        writable_bits = visible.stat().st_mode & 0o222
    except OSError:
        writable_bits = 0
    if not writable_bits or not os.access(visible, os.W_OK):
        return [_issue("backup_unwritable", "TLDW_BACKUP_DIR", "must be writable")]
    return []


def validate_environment(
    values: Mapping[str, str],
    *,
    env_path: Path | None,
    runtime_backup_dir: Path | None = None,
    require_backup_writable: bool = True,
) -> tuple[PreflightIssue, ...]:
    """Validate all semantic operator-environment invariants."""

    issues: list[PreflightIssue] = []
    existing = _is_true(values.get("TLDW_EXISTING_INSTALLATION", ""))
    for name in REQUIRED_ENV_NAMES:
        if not values.get(name, "") and not (
            existing and name in SECRET_NAMES[-1:] + ("ADMIN_USERNAME", "ADMIN_EMAIL")
        ):
            issues.append(_issue("missing_required", name, "must be provided"))
    if env_path is not None and env_path.exists():
        try:
            mode = env_path.stat().st_mode & 0o777
        except OSError:
            mode = 0o777
        if not env_path.is_file() or mode & 0o077:
            issues.append(
                _issue(
                    "env_permissions",
                    "TLDW_ENV_FILE",
                    "must be a regular owner-only file",
                )
            )
    issues.extend(_validate_secrets(values))
    issues.extend(_validate_database_url(values))
    issues.extend(_validate_redis_url(values))
    issues.extend(_validate_domain_and_origins(values))
    issues.extend(_validate_networks(values))
    issues.extend(_validate_images(values))
    issues.extend(_validate_setup(values))
    issues.extend(
        _validate_backup(
            values,
            runtime_backup_dir,
            require_writable=require_backup_writable,
        )
    )
    return _sorted_issues(issues)


def _network_names(service: Mapping[str, Any]) -> set[str]:
    """Return service network names from short or long Compose syntax."""

    networks = service.get("networks", [])
    if isinstance(networks, Mapping):
        return {str(name) for name in networks}
    if isinstance(networks, list):
        return {str(name) for name in networks}
    return set()


def _command_text(service: Mapping[str, Any]) -> str:
    """Return a service command or healthcheck test as inspection-only text."""

    command = service.get("command", service.get("test", ""))
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command)


def _ports(service: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Normalize short and rendered Compose port declarations."""

    result: set[tuple[int, int]] = set()
    for port in service.get("ports", []) or []:
        if isinstance(port, str):
            parts = port.split(":")
            if len(parts) >= 2:
                try:
                    result.add((int(parts[-2]), int(parts[-1].split("/")[0])))
                except ValueError:
                    result.add((-1, -1))
        elif isinstance(port, Mapping):
            try:
                result.add((int(port.get("published")), int(port.get("target"))))
            except (TypeError, ValueError):
                result.add((-1, -1))
    return result


def validate_compose(document: Mapping[str, Any]) -> tuple[PreflightIssue, ...]:
    """Validate the unrendered production Compose topology."""

    issues: list[PreflightIssue] = []
    if document.get("name") != "tldw-production":
        issues.append(
            _issue(
                "topology_project",
                "name",
                "must use the standalone production project name",
            )
        )
    services = document.get("services", {})
    networks = document.get("networks", {})
    if not isinstance(services, Mapping) or set(services) != {
        "preflight",
        "caddy",
        "app",
        "postgres",
        "redis",
    }:
        return (_issue("topology_services", "services", "must match the reference services"),)
    if not isinstance(networks, Mapping) or set(networks) != {"edge", "backend"}:
        issues.append(_issue("topology_network", "networks", "must contain only edge and backend"))
    elif networks.get("backend", {}).get("internal") is not True:
        issues.append(_issue("topology_network", "backend", "must be an internal network"))
    for name, service_value in services.items():
        service = service_value if isinstance(service_value, Mapping) else {}
        declared_ports = _ports(service)
        if name == "caddy":
            if declared_ports != {(80, 80), (443, 443)}:
                issues.append(_issue("topology_ports", name, "must publish only host ports 80 and 443"))
        elif declared_ports:
            issues.append(_issue("topology_ports", name, "must not publish a host port"))
        if "build" in service or "container_name" in service:
            issues.append(_issue("topology_service", name, "must use the minimal immutable service shape"))
        if any("/var/run/docker.sock" in str(item) for item in service.get("volumes", []) or []):
            issues.append(_issue("topology_socket", name, "must not mount the Docker socket"))
    for name, expected in _EXPECTED_NETWORKS.items():
        if _network_names(services[name]) != expected:
            issues.append(_issue("topology_network", name, "has unexpected network membership"))
    if services["preflight"].get("network_mode") != "none":
        issues.append(_issue("topology_preflight", "preflight", "must run without networking"))
    dependency = services["app"].get("depends_on", {}).get("preflight")
    if dependency != {"condition": "service_completed_successfully"}:
        issues.append(_issue("topology_preflight", "app", "must fail closed on preflight"))
    expected_preflight_command = [
        "--from-environment",
        "--compose-file",
        "/run/tldw/docker-compose.production.yml",
        "--proxy-file",
        "/run/tldw/Caddyfile",
        "--runtime-backup-dir",
        "/backups",
    ]
    if services["preflight"].get("command") != expected_preflight_command:
        issues.append(
            _issue(
                "topology_preflight",
                "preflight.command",
                "must validate the Compose-injected environment",
            )
        )
    if services["preflight"].get("env_file") != _EXPECTED_RAW_ENV_FILE:
        issues.append(
            _issue(
                "topology_preflight",
                "preflight.env_file",
                "must consume the required raw environment through Compose",
            )
        )
    for name, expected in _STATIC_IMAGE_INPUTS.items():
        if services[name].get("image") != expected:
            issues.append(_issue("topology_image", name, "must use the required image input"))
    app_environment = services["app"].get("environment", {})
    edge_input = "${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}"
    if app_environment.get("AUTH_MODE") != "multi_user" or app_environment.get("tldw_production") != "true":
        issues.append(_issue("topology_mode", "app", "must explicitly use multi-user production mode"))
    for field in TRUST_ENVIRONMENT:
        if app_environment.get(field) != edge_input:
            issues.append(_issue("topology_trust", field, "must derive from the private edge CIDR"))
    if services["caddy"].get("environment") != _EXPECTED_CADDY_ENVIRONMENT:
        issues.append(
            _issue(
                "topology_proxy",
                "caddy.environment",
                "must receive only the required domain and ACME contact",
            )
        )
    redis_command = _command_text(services["redis"])
    redis_health = _command_text(services["redis"].get("healthcheck", {}))
    if "requirepass %s" not in redis_command or "$$REDIS_PASSWORD" not in redis_command:
        issues.append(_issue("topology_redis_auth", "redis.command", "must require the external password"))
    if "REDISCLI_AUTH" not in redis_health or "$$REDIS_PASSWORD" not in redis_health:
        issues.append(_issue("topology_redis_auth", "redis.healthcheck", "must authenticate"))
    if set(document.get("volumes", {})) != _EXPECTED_VOLUMES:
        issues.append(_issue("topology_volumes", "volumes", "must match the persistent volume boundary"))
    for name, expected in _EXPECTED_SERVICE_VOLUMES.items():
        actual = {str(item) for item in services[name].get("volumes", []) or []}
        if actual != expected:
            issues.append(
                _issue(
                    "topology_mounts",
                    name,
                    "must match the required persistent and read-only mounts",
                )
            )
    return _sorted_issues(issues)


def validate_rendered_compose(document: Mapping[str, Any], values: Mapping[str, str]) -> tuple[PreflightIssue, ...]:
    """Revalidate security invariants after Compose interpolation."""

    issues: list[PreflightIssue] = []
    services = document.get("services", {})
    networks = document.get("networks", {})
    if not isinstance(services, Mapping):
        return (_issue("rendered_services", "services", "must be a mapping"),)
    for name in ("preflight", "caddy", "app", "postgres", "redis"):
        if name not in services or not isinstance(services[name], Mapping):
            issues.append(_issue("rendered_services", name, "is missing from the rendered model"))
    if issues:
        return _sorted_issues(issues)
    for name, service in services.items():
        declared_ports = _ports(service)
        if name == "caddy":
            if declared_ports != {(80, 80), (443, 443)}:
                issues.append(_issue("rendered_ports", name, "must publish only ports 80 and 443"))
        elif declared_ports:
            issues.append(_issue("rendered_ports", name, "must not publish ports"))
    if not isinstance(networks, Mapping) or networks.get("backend", {}).get("internal") is not True:
        issues.append(_issue("rendered_network", "backend", "must remain internal"))
    for name, expected in _EXPECTED_NETWORKS.items():
        if _network_names(services[name]) != expected:
            issues.append(_issue("rendered_network", name, "has unexpected network membership"))
    expected_images = {
        "preflight": values.get("TLDW_APP_IMAGE"),
        "app": values.get("TLDW_APP_IMAGE"),
        "caddy": values.get("CADDY_IMAGE"),
        "postgres": values.get("POSTGRES_IMAGE"),
        "redis": values.get("REDIS_IMAGE"),
    }
    for name, expected in expected_images.items():
        if not expected or services[name].get("image") != expected:
            issues.append(_issue("rendered_image", name, "does not match the validated image input"))
    app_environment = services["app"].get("environment", {})
    edge = values.get("TLDW_EDGE_SUBNET")
    if not isinstance(app_environment, Mapping):
        issues.append(_issue("rendered_trust", "app", "environment must be a mapping"))
    else:
        for field in TRUST_ENVIRONMENT:
            if (
                not edge
                or app_environment.get(field) != edge
                or app_environment.get(field)
                in {
                    "*",
                    "0.0.0.0/0",
                }
            ):
                issues.append(_issue("rendered_trust", field, "must equal the bounded edge CIDR"))
    sensitive_values = [
        values.get(name, "") for name in ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "DATABASE_URL", "REDIS_URL")
    ]
    for name, service in services.items():
        command = _command_text(service)
        if any(value and value in command for value in sensitive_values):
            issues.append(_issue("rendered_secret", name, "must not contain resolved credential text"))
    return _sorted_issues(issues)


def validate_proxy(text: str) -> tuple[PreflightIssue, ...]:
    """Validate the production Caddy public/private route boundary."""

    issues: list[PreflightIssue] = []
    matcher = next((line for line in text.splitlines() if "@private_control path" in line), "")
    matcher_paths = set(matcher.split()[2:]) if matcher else set()
    for path in DENIED_PROXY_PATHS:
        if path not in matcher_paths:
            issues.append(_issue("proxy_path", path, "must be denied by the public proxy"))
    deny = "respond @private_control 404"
    proxy = "reverse_proxy app:8000"
    if deny not in text or proxy not in text or text.index(deny) > text.index(proxy):
        issues.append(_issue("proxy_order", "Caddyfile", "must deny private routes before proxying"))
    upstreams = re.findall(r"(?m)^\s*reverse_proxy\s+(\S+)", text)
    if upstreams != ["app:8000"]:
        issues.append(_issue("proxy_upstream", "Caddyfile", "must proxy only to the application"))
    if not re.search(r"(?m)^\s*tls\s+\{\$TLDW_ACME_EMAIL\}\s*$", text):
        issues.append(_issue("proxy_tls", "Caddyfile", "must terminate TLS"))
    for directive in (
        "header_up X-Forwarded-For {remote_host}",
        "header_up X-Real-IP {remote_host}",
        "header_up X-Forwarded-Proto https",
    ):
        if directive not in text:
            issues.append(_issue("proxy_headers", "Caddyfile", "must overwrite client identity headers"))
    return _sorted_issues(issues)


def _validate_reference_files(
    compose_file: Path,
    proxy_file: Path,
) -> list[PreflightIssue]:
    """Validate the static Compose and proxy assets."""

    issues: list[PreflightIssue] = []
    try:
        compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
        if not isinstance(compose, Mapping):
            raise ValueError("Compose root must be a mapping")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        issues.append(_issue("compose_parse", "compose_file", "could not parse the production Compose file"))
    else:
        issues.extend(validate_compose(compose))
    try:
        proxy_text = proxy_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        issues.append(_issue("proxy_parse", "proxy_file", "could not read the production proxy file"))
    else:
        issues.extend(validate_proxy(proxy_text))
    return issues


def run_preflight(
    env_file: Path,
    compose_file: Path,
    proxy_file: Path,
    *,
    runtime_backup_dir: Path | None = None,
) -> PreflightReport:
    """Run host preflight, including raw-file and permission validation."""

    issues: list[PreflightIssue] = []
    try:
        values = load_raw_env(env_file)
    except (OSError, UnicodeError, ValueError):
        issues.append(_issue("env_parse", "TLDW_ENV_FILE", "could not parse the raw environment file"))
    else:
        issues.extend(
            validate_environment(
                values,
                env_path=env_file,
                runtime_backup_dir=runtime_backup_dir,
            )
        )
    issues.extend(_validate_reference_files(compose_file, proxy_file))
    return PreflightReport(_sorted_issues(issues))


def run_preflight_from_environment(
    values: Mapping[str, str],
    compose_file: Path,
    proxy_file: Path,
    *,
    runtime_backup_dir: Path | None = None,
) -> PreflightReport:
    """Run confined-container preflight using Compose-injected values."""

    issues = list(
        validate_environment(
            values,
            env_path=None,
            runtime_backup_dir=runtime_backup_dir,
            require_backup_writable=False,
        )
    )
    issues.extend(_validate_reference_files(compose_file, proxy_file))
    return PreflightReport(_sorted_issues(issues))


def _parser() -> argparse.ArgumentParser:
    """Build the side-effect-free production-preflight argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--env-file", type=Path)
    source.add_argument("--from-environment", action="store_true")
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--proxy-file", type=Path, default=DEFAULT_PROXY_FILE)
    parser.add_argument("--runtime-backup-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate CLI inputs and emit only sanitized deterministic diagnostics."""

    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if args.from_environment:
        report = run_preflight_from_environment(
            os.environ,
            args.compose_file,
            args.proxy_file,
            runtime_backup_dir=args.runtime_backup_dir,
        )
    else:
        report = run_preflight(
            args.env_file,
            args.compose_file,
            args.proxy_file,
            runtime_backup_dir=args.runtime_backup_dir,
        )
    if report.ok:
        print("Production preflight passed.")
        return 0
    for issue in report.issues:
        print(
            f"ERROR [{issue.code}] {issue.field}: {issue.message}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
