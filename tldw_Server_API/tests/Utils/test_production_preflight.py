from __future__ import annotations

import copy
import re
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml
from Helper_Scripts.Deployment.production_preflight import (
    PreflightIssue,
    load_raw_env,
    main,
    run_preflight,
    validate_compose,
    validate_environment,
    validate_proxy,
    validate_rendered_compose,
)

COMPOSE_PATH = Path("Dockerfiles/docker-compose.production.yml")
PROXY_PATH = Path("Dockerfiles/Production/Caddyfile")
_INTERPOLATION = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]+\}")


def _valid_env(tmp_path: Path) -> dict[str, str]:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(exist_ok=True)
    postgres_password = "pg-" + "P" * 40
    redis_password = "redis-" + "R" * 40
    return {
        "TLDW_PUBLIC_DOMAIN": "tldw.acme.internal",
        "TLDW_ACME_EMAIL": "ops@acme.internal",
        "ALLOWED_ORIGINS": "https://tldw.acme.internal",
        "JWT_SECRET_KEY": "jwt-" + "J" * 48,
        "SESSION_ENCRYPTION_KEY": "session-" + "S" * 48,
        "POSTGRES_USER": "tldw_app",
        "POSTGRES_DB": "tldw",
        "POSTGRES_PASSWORD": postgres_password,
        "DATABASE_URL": ("postgresql://tldw_app:" f"{quote(postgres_password, safe='')}@postgres:5432/tldw"),
        "REDIS_PASSWORD": redis_password,
        "REDIS_URL": f"redis://:{quote(redis_password, safe='')}@redis:6379/0",
        "ADMIN_USERNAME": "initial-admin",
        "ADMIN_PASSWORD": "admin-" + "A" * 40,
        "ADMIN_EMAIL": "admin@acme.internal",
        "TLDW_EXISTING_INSTALLATION": "false",
        "TLDW_SETUP_COMPLETED": "true",
        "TLDW_EDGE_SUBNET": "172.30.0.0/24",
        "TLDW_BACKEND_SUBNET": "172.31.0.0/24",
        "TLDW_APP_IMAGE": "registry.acme.internal/tldw@sha256:" + "a" * 64,
        "TLDW_ROLLBACK_IMAGE": "registry.acme.internal/tldw@sha256:" + "b" * 64,
        "CADDY_IMAGE": "caddy:2.10.2-alpine",
        "POSTGRES_IMAGE": "postgres:18.0-bookworm",
        "REDIS_IMAGE": "redis:7.4.1-alpine",
        "TLDW_BACKUP_DIR": str(backup_dir),
    }


def _real_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _render(value: object, values: dict[str, str]) -> object:
    if isinstance(value, str):
        return _INTERPOLATION.sub(lambda match: values[match.group(1)], value)
    if isinstance(value, list):
        return [_render(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, values) for key, item in value.items()}
    return value


def _rendered_compose(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    values = _valid_env(tmp_path)
    values["TLDW_ENV_FILE"] = str(tmp_path / "production.env")
    return _render(_real_compose(), values), values  # type: ignore[return-value]


def _rendered_compose_json(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    """Model the long-form shapes emitted by `docker compose config --format json`."""

    compose, values = _rendered_compose(tmp_path)
    for service in compose["services"].values():
        if "env_file" in service:
            service.pop("env_file")
            service["environment"] = {**values, **service.get("environment", {})}
        if isinstance(service.get("networks"), list):
            service["networks"] = dict.fromkeys(service["networks"])
        if isinstance(service.get("depends_on"), dict):
            service["depends_on"] = {
                name: {**dependency, "required": True} for name, dependency in service["depends_on"].items()
            }
        if isinstance(service.get("ports"), list):
            rendered_ports = []
            for declaration in service["ports"]:
                published, target = declaration.split(":")
                rendered_ports.append(
                    {
                        "mode": "ingress",
                        "target": int(target),
                        "published": published,
                        "protocol": "tcp",
                    }
                )
            service["ports"] = rendered_ports
        if isinstance(service.get("volumes"), list):
            rendered_volumes = []
            for declaration in service["volumes"]:
                source, target, *options = declaration.split(":")
                is_bind = source.startswith((".", "/"))
                if is_bind and source.startswith("."):
                    source = str((COMPOSE_PATH.parent / source).resolve())
                mount = {
                    "type": "bind" if is_bind else "volume",
                    "source": source,
                    "target": target,
                    "bind" if is_bind else "volume": {},
                }
                if "ro" in options:
                    mount["read_only"] = True
                rendered_volumes.append(mount)
            service["volumes"] = rendered_volumes
    for name, network in compose["networks"].items():
        network["name"] = f"tldw-production_{name}"
    compose["volumes"] = {name: {"name": f"tldw-production_{name}"} for name in compose["volumes"]}
    return compose, values


def _write_env(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "\n".join(f"{name}={value}" for name, value in values.items()) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _codes(issues: tuple[PreflightIssue, ...]) -> set[str]:
    return {issue.code for issue in issues}


def test_load_raw_env_preserves_literal_values_without_expansion(tmp_path: Path) -> None:
    path = tmp_path / "production.env"
    path.write_bytes(
        b"# comment\r\nPLAIN=value\r\nHASH='value#part'\r\n"
        b"INLINE_HASH=value#part\r\nVARIABLE=$OTHER\r\nESCAPE=one\\ntwo\r\n"
        b'DOUBLE="quoted value"\r\nDOLLAR=$(whoami)\r\nBACKTICK=`id`\r\n'
    )

    assert load_raw_env(path) == {
        "PLAIN": "value",
        "HASH": "value#part",
        "INLINE_HASH": "value#part",
        "VARIABLE": "$OTHER",
        "ESCAPE": r"one\ntwo",
        "DOUBLE": "quoted value",
        "DOLLAR": "$(whoami)",
        "BACKTICK": "`id`",
    }


@pytest.mark.parametrize(
    "text",
    (
        "export KEY=value\n",
        "1INVALID=value\n",
        "MISSING_EQUALS\n",
        "DUPLICATE=one\nDUPLICATE=two\n",
        "UNMATCHED='value\n",
    ),
)
def test_load_raw_env_rejects_ambiguous_or_duplicate_input(tmp_path: Path, text: str) -> None:
    path = tmp_path / "production.env"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_raw_env(path)


def test_report_aggregates_without_secret_values(tmp_path: Path) -> None:
    secret = "super-secret-value-that-must-not-leak"
    values = _valid_env(tmp_path)
    values["JWT_SECRET_KEY"] = "short"
    values["POSTGRES_PASSWORD"] = secret
    values["DATABASE_URL"] = "postgresql://tldw_app:different@postgres:5432/tldw"

    issues = validate_environment(values, env_path=tmp_path / "production.env")
    rendered = "\n".join(f"{item.code}:{item.field}:{item.message}" for item in issues)

    assert "weak_secret:JWT_SECRET_KEY" in rendered
    assert "credential_mismatch:DATABASE_URL" in rendered
    assert secret not in rendered
    assert "postgresql://" not in rendered


def test_environment_issues_are_deterministic_and_deduplicated(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    values["JWT_SECRET_KEY"] = "short"
    values["SESSION_ENCRYPTION_KEY"] = "short"
    values["ALLOWED_ORIGINS"] = "*"

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert issues == tuple(sorted(set(issues)))
    assert {issue.code for issue in issues} >= {"shared_secret", "unsafe_origin", "weak_secret"}


def test_environment_requires_every_secret_to_be_independent(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    values["SESSION_ENCRYPTION_KEY"] = values["JWT_SECRET_KEY"]

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert "shared_secret" in _codes(issues)


@pytest.mark.parametrize(
    "database_url",
    (
        "postgresql://other:pg-PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP@postgres:5432/tldw",
        "postgresql://tldw_app:different@postgres:5432/tldw",
        "postgresql://tldw_app:pg-PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP@postgres:5432/other",
    ),
)
def test_environment_rejects_each_database_url_credential_mismatch(
    tmp_path: Path,
    database_url: str,
) -> None:
    values = _valid_env(tmp_path)
    values["DATABASE_URL"] = database_url

    assert "credential_mismatch" in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


def test_environment_rejects_redis_host_mismatch_after_url_decoding(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    password = quote(values["REDIS_PASSWORD"], safe="")
    values["REDIS_URL"] = f"redis://:{password}@other-redis:6379/0"

    assert "credential_mismatch" in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


@pytest.mark.parametrize(
    ("port", "expected_code"),
    (("5433", "credential_mismatch"), ("not-a-port", "invalid_url"), ("99999", "invalid_url"), ("", "invalid_url")),
)
def test_environment_rejects_invalid_or_unexpected_database_ports(
    tmp_path: Path,
    port: str,
    expected_code: str,
) -> None:
    values = _valid_env(tmp_path)
    password = quote(values["POSTGRES_PASSWORD"], safe="")
    values["DATABASE_URL"] = f"postgresql://tldw_app:{password}@postgres:{port}/tldw"

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert expected_code in _codes(issues)
    assert values["POSTGRES_PASSWORD"] not in "\n".join(issue.message for issue in issues)


def test_environment_accepts_database_default_port(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    password = quote(values["POSTGRES_PASSWORD"], safe="")
    values["DATABASE_URL"] = f"postgresql://tldw_app:{password}@postgres/tldw"

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert not {"invalid_url", "credential_mismatch"} & _codes(issues)


@pytest.mark.parametrize(
    ("username", "port", "expected_code"),
    (
        ("", "6380", "credential_mismatch"),
        ("", "not-a-port", "invalid_url"),
        ("", "99999", "invalid_url"),
        ("operator", "6379", "credential_mismatch"),
    ),
)
def test_environment_rejects_invalid_redis_port_or_username(
    tmp_path: Path,
    username: str,
    port: str,
    expected_code: str,
) -> None:
    values = _valid_env(tmp_path)
    password = quote(values["REDIS_PASSWORD"], safe="")
    authority = f"{username}:{password}" if username else f":{password}"
    values["REDIS_URL"] = f"redis://{authority}@redis:{port}/0"

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert expected_code in _codes(issues)
    assert values["REDIS_PASSWORD"] not in "\n".join(issue.message for issue in issues)


@pytest.mark.parametrize("username", ("", "default"))
@pytest.mark.parametrize("include_port", (False, True))
def test_environment_accepts_requirepass_redis_uri_forms(
    tmp_path: Path,
    username: str,
    include_port: bool,
) -> None:
    values = _valid_env(tmp_path)
    password = quote(values["REDIS_PASSWORD"], safe="")
    authority = f"{username}:{password}" if username else f":{password}"
    port = ":6379" if include_port else ""
    values["REDIS_URL"] = f"redis://{authority}@redis{port}/0"

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert not {"invalid_url", "credential_mismatch"} & _codes(issues)


def test_environment_rejects_origin_with_a_path(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    values["ALLOWED_ORIGINS"] = "https://tldw.acme.internal/application"

    assert "unsafe_origin" in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("JWT_SECRET_KEY", "short", "weak_secret"),
        ("POSTGRES_PASSWORD", "change-me", "placeholder_secret"),
        ("REDIS_PASSWORD", "redis", "placeholder_secret"),
        ("ALLOWED_ORIGINS", "*", "unsafe_origin"),
        ("TLDW_EDGE_SUBNET", "0.0.0.0/0", "unsafe_network"),
        ("TLDW_BACKEND_SUBNET", "172.30.0.0/24", "overlapping_network"),
        ("TLDW_SETUP_COMPLETED", "false", "setup_incomplete"),
        ("TLDW_APP_IMAGE", "registry/tldw:latest", "mutable_image"),
        ("TLDW_ROLLBACK_IMAGE", "registry/tldw:prod", "mutable_image"),
        ("CADDY_IMAGE", "caddy:2", "inexact_third_party_image"),
    ),
)
def test_environment_rejects_each_unsafe_value(tmp_path: Path, field: str, value: str, expected_code: str) -> None:
    values = _valid_env(tmp_path)
    values[field] = value

    assert expected_code in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


def test_environment_requires_independent_target_and_rollback_images(
    tmp_path: Path,
) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_ROLLBACK_IMAGE"] = values["TLDW_APP_IMAGE"]

    assert "identical_images" in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


def test_environment_requires_bootstrap_only_for_new_installations(
    tmp_path: Path,
) -> None:
    new_values = _valid_env(tmp_path)
    new_values["ADMIN_PASSWORD"] = ""
    existing_values = _valid_env(tmp_path)
    existing_values["TLDW_EXISTING_INSTALLATION"] = "true"

    assert "missing_required" in _codes(validate_environment(new_values, env_path=tmp_path / "production.env"))
    assert "unexpected_bootstrap_secret" in _codes(
        validate_environment(existing_values, env_path=tmp_path / "production.env")
    )


@pytest.mark.parametrize("field", ("ADMIN_USERNAME", "ADMIN_PASSWORD", "ADMIN_EMAIL"))
def test_new_installation_requires_each_bootstrap_field(tmp_path: Path, field: str) -> None:
    values = _valid_env(tmp_path)
    values[field] = ""

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert any(issue.code == "missing_required" and issue.field == field for issue in issues)


def test_environment_requires_origin_domain_and_contact_alignment(
    tmp_path: Path,
) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_PUBLIC_DOMAIN"] = "example.com"
    values["TLDW_ACME_EMAIL"] = "ops@example.com"
    values["ALLOWED_ORIGINS"] = "https://other.acme.internal"

    codes = _codes(validate_environment(values, env_path=tmp_path / "production.env"))

    assert {"sample_value", "origin_mismatch"} <= codes


@pytest.mark.parametrize(
    "domain",
    (
        "*.acme.internal",
        "bad_name.acme.internal",
        ".acme.internal",
        "acme..internal",
        "-bad.acme.internal",
        "bad-.acme.internal",
        "tldw",
        "tldwé.acme.internal",
        f"{'a' * 64}.acme.internal",
        ".".join(["a" * 63] * 4),
    ),
)
def test_environment_rejects_non_dns_public_identity(tmp_path: Path, domain: str) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_PUBLIC_DOMAIN"] = domain

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert any(issue.code == "sample_value" and issue.field == "TLDW_PUBLIC_DOMAIN" for issue in issues)


@pytest.mark.parametrize(
    "contact",
    (
        "ops",
        "@acme.internal",
        "ops@",
        "ops @acme.internal",
        "ops@bad_name.internal",
        "ops@-bad.internal",
        "ops@localhost",
        "ops@tldwé.acme.internal",
        "ops@@acme.internal",
        ".ops@acme.internal",
        "ops..tls@acme.internal",
        f"{'o' * 65}@acme.internal",
    ),
)
def test_environment_rejects_invalid_acme_contact(tmp_path: Path, contact: str) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_ACME_EMAIL"] = contact

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert any(issue.code == "sample_value" and issue.field == "TLDW_ACME_EMAIL" for issue in issues)


def test_environment_accepts_bounded_dns_and_acme_email_boundaries(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    domain = f"{'a' * 63}.{'b' * 63}.{'c' * 63}.{'d' * 61}"
    values["TLDW_PUBLIC_DOMAIN"] = domain
    values["TLDW_ACME_EMAIL"] = "ops+tls@acme.internal"
    values["ALLOWED_ORIGINS"] = f"https://{domain}"

    issues = validate_environment(values, env_path=tmp_path / "production.env")

    assert not any(issue.code == "sample_value" for issue in issues)


@pytest.mark.parametrize(
    ("backup_value", "expected_code"),
    (("relative/backups", "unsafe_backup_path"), ("/app/Databases", "live_data_path")),
)
def test_environment_rejects_unsafe_backup_paths(tmp_path: Path, backup_value: str, expected_code: str) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_BACKUP_DIR"] = backup_value

    assert expected_code in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


def test_environment_rejects_backup_parent_of_live_data(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_BACKUP_DIR"] = "/app"

    assert "live_data_path" in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


def test_environment_rejects_missing_and_non_directory_backup_targets(tmp_path: Path) -> None:
    missing_values = _valid_env(tmp_path)
    missing_values["TLDW_BACKUP_DIR"] = str(tmp_path / "missing")
    file_target = tmp_path / "backup-file"
    file_target.write_text("not a directory", encoding="utf-8")
    file_values = _valid_env(tmp_path)
    file_values["TLDW_BACKUP_DIR"] = str(file_target)

    assert "backup_unavailable" in _codes(validate_environment(missing_values, env_path=tmp_path / "production.env"))
    assert "backup_unavailable" in _codes(validate_environment(file_values, env_path=tmp_path / "production.env"))


def test_static_compose_and_proxy_match_the_reference_contract() -> None:
    assert validate_compose(_real_compose()) == ()
    assert validate_proxy(PROXY_PATH.read_text(encoding="utf-8")) == ()


def test_static_compose_aggregates_independent_project_and_service_failures() -> None:
    compose = copy.deepcopy(_real_compose())
    compose["name"] = "other-project"
    compose["services"]["debug"] = {}

    assert _codes(validate_compose(compose)) == {"topology_project", "topology_services"}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("project_name", "topology_project"),
        ("app_port", "topology_ports"),
        ("postgres_port", "topology_ports"),
        ("redis_port", "topology_ports"),
        ("caddy_extra_port", "topology_ports"),
        ("backend_public", "topology_network"),
        ("caddy_backend", "topology_network"),
        ("postgres_edge", "topology_network"),
        ("wrong_edge_ipam", "topology_network"),
        ("wrong_backend_ipam", "topology_network"),
        ("missing_preflight", "topology_preflight"),
        ("unconfined_preflight", "topology_preflight"),
        ("wrong_preflight_entrypoint", "topology_preflight"),
        ("missing_preflight_env_file", "topology_preflight"),
        ("missing_app_env_file", "topology_env"),
        ("unexpected_service", "topology_services"),
        ("malformed_service", "topology_service"),
        ("malformed_backend", "topology_network"),
        ("missing_caddyfile", "topology_mounts"),
        ("missing_caddy_environment", "topology_proxy"),
        ("docker_socket", "topology_socket"),
        ("wildcard_trust", "topology_trust"),
        ("unsafe_setup_remote", "topology_trust"),
        ("preserved_forwarded_for", "topology_trust"),
        ("wrong_client_ip_header", "topology_trust"),
        ("disabled_mcp_forwarding", "topology_trust"),
        ("missing_redis_auth", "topology_redis_auth"),
        ("missing_postgres_auth", "topology_postgres_auth"),
        ("wrong_app_healthcheck", "topology_health"),
        ("production_fallback", "topology_mode"),
        ("image_default", "topology_image"),
    ),
)
def test_static_compose_mutations_fail_closed(mutation: str, expected_code: str) -> None:
    compose = copy.deepcopy(_real_compose())
    services = compose["services"]
    if mutation == "project_name":
        compose["name"] = "other-project"
    elif mutation == "app_port":
        services["app"]["ports"] = ["8000:8000"]
    elif mutation == "postgres_port":
        services["postgres"]["ports"] = ["5432:5432"]
    elif mutation == "redis_port":
        services["redis"]["ports"] = ["6379:6379"]
    elif mutation == "caddy_extra_port":
        services["caddy"]["ports"].append("8443:443")
    elif mutation == "backend_public":
        compose["networks"]["backend"]["internal"] = False
    elif mutation == "caddy_backend":
        services["caddy"]["networks"].append("backend")
    elif mutation == "postgres_edge":
        services["postgres"]["networks"].append("edge")
    elif mutation == "wrong_edge_ipam":
        compose["networks"]["edge"]["ipam"]["config"][0][
            "subnet"
        ] = "${TLDW_BACKEND_SUBNET:?Set private TLDW_BACKEND_SUBNET}"
    elif mutation == "wrong_backend_ipam":
        compose["networks"]["backend"]["ipam"]["config"][0][
            "subnet"
        ] = "${TLDW_EDGE_SUBNET:?Set private TLDW_EDGE_SUBNET}"
    elif mutation == "missing_preflight":
        del services["app"]["depends_on"]["preflight"]
    elif mutation == "unconfined_preflight":
        services["preflight"]["read_only"] = False
    elif mutation == "wrong_preflight_entrypoint":
        services["preflight"]["entrypoint"] = ["sh"]
    elif mutation == "missing_preflight_env_file":
        del services["preflight"]["env_file"]
    elif mutation == "missing_app_env_file":
        del services["app"]["env_file"]
    elif mutation == "unexpected_service":
        services["debug"] = {"ports": ["9000:9000"]}
    elif mutation == "malformed_service":
        services["redis"] = "not-a-service"
    elif mutation == "malformed_backend":
        compose["networks"]["backend"] = "not-a-network"
    elif mutation == "missing_caddyfile":
        services["caddy"]["volumes"].remove("./Production/Caddyfile:/etc/caddy/Caddyfile:ro")
    elif mutation == "missing_caddy_environment":
        del services["caddy"]["environment"]["TLDW_PUBLIC_DOMAIN"]
    elif mutation == "docker_socket":
        services["preflight"]["volumes"].append("/var/run/docker.sock:/var/run/docker.sock")
    elif mutation == "wildcard_trust":
        services["app"]["environment"]["RG_TRUSTED_PROXIES"] = "0.0.0.0/0"
    elif mutation == "unsafe_setup_remote":
        services["app"]["environment"]["TLDW_SETUP_ALLOW_REMOTE"] = "1"
    elif mutation == "preserved_forwarded_for":
        services["app"]["environment"]["AUTH_TRUST_X_FORWARDED_FOR"] = "false"
    elif mutation == "wrong_client_ip_header":
        services["app"]["environment"]["RG_CLIENT_IP_HEADER"] = "X-Real-IP"
    elif mutation == "disabled_mcp_forwarding":
        services["app"]["environment"]["MCP_TRUST_X_FORWARDED"] = "false"
    elif mutation == "missing_redis_auth":
        services["redis"]["command"] = ["redis-server"]
    elif mutation == "missing_postgres_auth":
        services["postgres"]["healthcheck"]["test"] = ["CMD", "true"]
    elif mutation == "wrong_app_healthcheck":
        services["app"]["healthcheck"]["test"][-1] = "print('healthy')"
    elif mutation == "production_fallback":
        services["app"]["environment"]["tldw_production"] = "${tldw_production:-false}"
    elif mutation == "image_default":
        services["app"]["image"] = "${TLDW_APP_IMAGE:-registry/tldw:latest}"

    assert expected_code in _codes(validate_compose(compose))


@pytest.mark.parametrize(
    ("field", "expected_code"),
    (
        ("ports", "topology_ports"),
        ("depends_on", "topology_preflight"),
        ("environment", "topology_mode"),
        ("healthcheck", "topology_health"),
        ("volumes", "topology_volumes"),
    ),
)
def test_static_compose_malformed_nested_shapes_fail_closed(field: str, expected_code: str) -> None:
    compose = copy.deepcopy(_real_compose())
    if field == "ports":
        compose["services"]["caddy"]["ports"] = 443
    elif field == "depends_on":
        compose["services"]["app"]["depends_on"] = "preflight"
    elif field == "environment":
        compose["services"]["app"]["environment"] = "AUTH_MODE=multi_user"
    elif field == "healthcheck":
        compose["services"]["app"]["healthcheck"] = "healthy"
    elif field == "volumes":
        compose["volumes"] = None

    assert expected_code in _codes(validate_compose(compose))


@pytest.mark.parametrize(
    "ports",
    (
        [8000],
        ["8000"],
        [{"target": 8000}],
        [True],
        ["not-a-port"],
    ),
)
def test_static_compose_treats_every_ports_entry_as_publication(ports: list[object]) -> None:
    compose = copy.deepcopy(_real_compose())
    compose["services"]["app"]["ports"] = ports

    assert "topology_ports" in _codes(validate_compose(compose))


@pytest.mark.parametrize(
    "ports",
    (
        ["nonsense:80:80", "443:443"],
        [
            {"target": 80, "published": 80, "protocol": "tcp", "mode": "ingress", "mystery": True},
            {"target": 443, "published": 443, "protocol": "tcp", "mode": "ingress"},
        ],
    ),
)
def test_static_compose_rejects_unrecognized_caddy_port_syntax(ports: list[object]) -> None:
    compose = copy.deepcopy(_real_compose())
    compose["services"]["caddy"]["ports"] = ports

    assert "topology_ports" in _codes(validate_compose(compose))


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("missing_path", "proxy_path"),
        ("late_deny", "proxy_order"),
        ("missing_tls", "proxy_tls"),
        ("stateful_upstream", "proxy_upstream"),
        ("forwarded_incoming", "proxy_headers"),
    ),
)
def test_proxy_mutations_fail_closed(mutation: str, expected_code: str) -> None:
    text = PROXY_PATH.read_text(encoding="utf-8")
    if mutation == "missing_path":
        text = text.replace(" /ready", "")
    elif mutation == "late_deny":
        deny = "  respond @private_control 404\n"
        text = text.replace(deny, "").replace("  tls ", deny + "\n  tls ")
    elif mutation == "missing_tls":
        text = text.replace("  tls {$TLDW_ACME_EMAIL}\n", "")
    elif mutation == "stateful_upstream":
        text = text.replace("reverse_proxy app:8000", "reverse_proxy postgres:5432")
    elif mutation == "forwarded_incoming":
        text = text.replace(
            "header_up X-Forwarded-For {remote_host}",
            "header_up X-Forwarded-For {http.request.header.X-Forwarded-For}",
        )

    assert expected_code in _codes(validate_proxy(text))


@pytest.mark.parametrize(
    "path",
    (
        "/internal/ready",
        "/ready",
        "/health/ready",
        "/api/v1/healthz",
        "/api/v1/readyz",
        "/setup",
        "/setup/*",
        "/api/v1/setup",
        "/api/v1/setup/*",
    ),
)
def test_proxy_requires_every_private_path(path: str) -> None:
    text = PROXY_PATH.read_text(encoding="utf-8")
    matcher = next(line for line in text.splitlines() if "@private_control path" in line)
    text = text.replace(matcher, matcher.replace(f" {path}", "", 1))

    assert "proxy_path" in _codes(validate_proxy(text))


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("commented_matcher", "proxy_path"),
        ("commented_deny", "proxy_order"),
        ("wrong_response", "proxy_response"),
        ("commented_header", "proxy_headers"),
        ("public_health_denied", "proxy_public_path"),
        ("missing_domain", "proxy_domain"),
    ),
)
def test_proxy_directives_must_be_active_and_fail_closed(mutation: str, expected_code: str) -> None:
    text = PROXY_PATH.read_text(encoding="utf-8")
    if mutation == "commented_matcher":
        text = text.replace("  @private_control path", "  # @private_control path")
    elif mutation == "commented_deny":
        text = text.replace("  respond @private_control 404", "  # respond @private_control 404")
    elif mutation == "wrong_response":
        text = text.replace("respond @private_control 404", "respond @private_control 403")
    elif mutation == "commented_header":
        text = text.replace(
            "    header_up X-Real-IP {remote_host}",
            "    # header_up X-Real-IP {remote_host}",
        )
    elif mutation == "public_health_denied":
        text = text.replace(" /internal/ready", " /health /internal/ready")
    elif mutation == "missing_domain":
        text = text.replace("{$TLDW_PUBLIC_DOMAIN} {", "https://fixed.invalid {")

    assert expected_code in _codes(validate_proxy(text))


@pytest.mark.parametrize(
    "extra",
    ("/*", "/api/*", "/other", "/internal/ready", "#", "# comment"),
)
def test_proxy_private_matcher_must_equal_the_intended_path_set(extra: str) -> None:
    text = PROXY_PATH.read_text(encoding="utf-8")
    matcher = next(line for line in text.splitlines() if "@private_control path" in line)
    text = text.replace(matcher, f"{matcher} {extra}")

    assert "proxy_path" in _codes(validate_proxy(text))


@pytest.mark.parametrize(
    ("directive", "expected_code"),
    (
        ("matcher", "proxy_order"),
        ("header", "proxy_headers"),
        ("tls", "proxy_tls"),
    ),
)
def test_proxy_security_directives_must_be_inside_their_active_blocks(
    directive: str,
    expected_code: str,
) -> None:
    text = PROXY_PATH.read_text(encoding="utf-8")
    if directive == "matcher":
        line = next(line for line in text.splitlines() if "@private_control path" in line)
    elif directive == "header":
        line = "    header_up X-Real-IP {remote_host}"
    else:
        line = "  tls {$TLDW_ACME_EMAIL}"
    text = text.replace(f"{line}\n", "", 1)
    text = f"{line}\n{text}"

    assert expected_code in _codes(validate_proxy(text))


def test_proxy_rejects_duplicate_private_matcher_directives() -> None:
    text = PROXY_PATH.read_text(encoding="utf-8")
    matcher = next(line for line in text.splitlines() if "@private_control path" in line)
    text = text.replace(matcher, f"{matcher}\n{matcher}")

    assert "proxy_path" in _codes(validate_proxy(text))


def test_rendered_compose_matches_concrete_environment(tmp_path: Path) -> None:
    compose, values = _rendered_compose(tmp_path)

    assert validate_rendered_compose(compose, values) == ()


def test_rendered_compose_accepts_actual_json_shapes_and_injected_secrets(tmp_path: Path) -> None:
    compose, values = _rendered_compose_json(tmp_path)

    assert values["POSTGRES_PASSWORD"] in compose["services"]["app"]["environment"].values()
    assert validate_rendered_compose(compose, values) == ()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("secret_command", "rendered_secret"),
        ("published_app", "rendered_ports"),
        ("wrong_image", "rendered_image"),
        ("public_backend", "rendered_network"),
        ("wildcard_trust", "rendered_trust"),
        ("unexpected_app_network", "rendered_network"),
        ("unexpected_service", "rendered_services"),
        ("unexpected_network", "rendered_network"),
        ("wrong_edge_subnet", "rendered_network"),
        ("wrong_backend_subnet", "rendered_network"),
        ("unconfined_preflight", "rendered_preflight"),
        ("wrong_caddy_environment", "rendered_proxy"),
        ("malformed_backend", "rendered_network"),
    ),
)
def test_rendered_compose_mutations_fail_closed(tmp_path: Path, mutation: str, expected_code: str) -> None:
    compose, values = _rendered_compose(tmp_path)
    if mutation == "secret_command":
        compose["services"]["redis"]["command"] = [values["REDIS_PASSWORD"]]
    elif mutation == "published_app":
        compose["services"]["app"]["ports"] = ["8000:8000"]
    elif mutation == "wrong_image":
        compose["services"]["app"]["image"] = values["TLDW_ROLLBACK_IMAGE"]
    elif mutation == "public_backend":
        compose["networks"]["backend"]["internal"] = False
    elif mutation == "wildcard_trust":
        compose["services"]["app"]["environment"]["RG_TRUSTED_PROXIES"] = "0.0.0.0/0"
    elif mutation == "unexpected_app_network":
        compose["services"]["app"]["networks"].append("debug")
    elif mutation == "unexpected_service":
        compose["services"]["debug"] = {"image": "debug:1.2.3"}
    elif mutation == "unexpected_network":
        compose["networks"]["debug"] = {"internal": True}
    elif mutation == "wrong_edge_subnet":
        compose["networks"]["edge"]["ipam"]["config"][0]["subnet"] = values["TLDW_BACKEND_SUBNET"]
    elif mutation == "wrong_backend_subnet":
        compose["networks"]["backend"]["ipam"]["config"][0]["subnet"] = values["TLDW_EDGE_SUBNET"]
    elif mutation == "unconfined_preflight":
        compose["services"]["preflight"]["network_mode"] = "default"
    elif mutation == "wrong_caddy_environment":
        compose["services"]["caddy"]["environment"]["TLDW_PUBLIC_DOMAIN"] = "other.invalid"
    elif mutation == "malformed_backend":
        compose["networks"]["backend"] = "not-a-network"

    assert expected_code in _codes(validate_rendered_compose(compose, values))


@pytest.mark.parametrize(
    "ports",
    (
        [8000],
        ["8000"],
        [{"target": 8000}],
        [True],
        ["not-a-port"],
    ),
)
def test_rendered_compose_treats_every_ports_entry_as_publication(
    tmp_path: Path,
    ports: list[object],
) -> None:
    compose, values = _rendered_compose(tmp_path)
    compose["services"]["app"]["ports"] = ports

    assert "rendered_ports" in _codes(validate_rendered_compose(compose, values))


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("preflight_command", "rendered_preflight"),
        ("preflight_entrypoint", "rendered_preflight"),
        ("preflight_restart", "rendered_preflight"),
        ("preflight_dependency", "rendered_preflight"),
        ("preflight_capabilities", "rendered_preflight"),
        ("missing_mount", "rendered_mounts"),
        ("writable_mount", "rendered_mounts"),
        ("extra_mount", "rendered_mounts"),
        ("docker_socket", "rendered_socket"),
        ("app_health", "rendered_health"),
        ("app_mode", "rendered_mode"),
        ("setup_remote", "rendered_trust"),
        ("forwarded_for", "rendered_trust"),
        ("redis_command", "rendered_redis_auth"),
        ("redis_health", "rendered_redis_auth"),
        ("postgres_health", "rendered_postgres_auth"),
        ("declared_volumes", "rendered_volumes"),
    ),
)
def test_rendered_compose_rechecks_final_wiring(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    compose, values = _rendered_compose_json(tmp_path)
    services = compose["services"]
    if mutation == "preflight_command":
        services["preflight"]["command"] = ["--from-environment"]
    elif mutation == "preflight_entrypoint":
        services["preflight"]["entrypoint"] = ["sh"]
    elif mutation == "preflight_restart":
        services["preflight"]["restart"] = "unless-stopped"
    elif mutation == "preflight_dependency":
        services["app"]["depends_on"]["preflight"]["condition"] = "service_started"
    elif mutation == "preflight_capabilities":
        services["preflight"]["cap_drop"] = []
    elif mutation == "missing_mount":
        services["preflight"]["volumes"].pop()
    elif mutation == "writable_mount":
        services["preflight"]["volumes"][-1]["read_only"] = False
    elif mutation == "extra_mount":
        services["app"]["volumes"].append({"type": "volume", "source": "extra", "target": "/extra", "volume": {}})
    elif mutation == "docker_socket":
        services["preflight"]["volumes"].append(
            {
                "type": "bind",
                "source": "/var/run/docker.sock",
                "target": "/var/run/docker.sock",
                "bind": {},
            }
        )
    elif mutation == "app_health":
        services["app"]["healthcheck"]["test"] = ["CMD", "true"]
    elif mutation == "app_mode":
        services["app"]["environment"]["AUTH_MODE"] = "single_user"
    elif mutation == "setup_remote":
        services["app"]["environment"]["TLDW_SETUP_ALLOW_REMOTE"] = "1"
    elif mutation == "forwarded_for":
        services["app"]["environment"]["AUTH_TRUST_X_FORWARDED_FOR"] = "false"
    elif mutation == "redis_command":
        services["redis"]["command"] = ["redis-server"]
    elif mutation == "redis_health":
        services["redis"]["healthcheck"]["test"] = ["CMD", "redis-cli", "ping"]
    elif mutation == "postgres_health":
        services["postgres"]["healthcheck"]["test"] = ["CMD", "true"]
    elif mutation == "declared_volumes":
        compose["volumes"]["extra"] = {"name": "tldw-production_extra"}

    assert expected_code in _codes(validate_rendered_compose(compose, values))


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("optional_preflight", "rendered_preflight"),
        ("malformed_extra_service", "rendered_services"),
        ("invalid_mount_source", "rendered_mounts"),
        ("wrong_volume_name", "rendered_volumes"),
        ("wrong_network_name", "rendered_network"),
        ("external_network", "rendered_network"),
    ),
)
def test_rendered_compose_fail_closed_edge_shapes(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    compose, values = _rendered_compose_json(tmp_path)
    if mutation == "optional_preflight":
        compose["services"]["app"]["depends_on"]["preflight"]["required"] = False
    elif mutation == "malformed_extra_service":
        compose["services"]["debug"] = "not-a-service"
    elif mutation == "invalid_mount_source":
        compose["services"]["preflight"]["volumes"][0]["source"] = "\0"
    elif mutation == "wrong_volume_name":
        compose["volumes"]["app-data"]["name"] = "shared_app_data"
    elif mutation == "wrong_network_name":
        compose["networks"]["edge"]["name"] = "shared_edge"
    elif mutation == "external_network":
        compose["networks"]["edge"]["external"] = True

    issues = validate_rendered_compose(compose, values)

    assert expected_code in _codes(issues)


def test_run_preflight_accepts_a_complete_offline_fixture(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert report.ok
    assert report.issues == ()


def test_cli_accepts_compose_injected_environment_without_raw_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = _valid_env(tmp_path)
    (tmp_path / "backups").chmod(0o500)
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    exit_code = main(
        [
            "--from-environment",
            "--compose-file",
            str(COMPOSE_PATH),
            "--proxy-file",
            str(PROXY_PATH),
            "--runtime-backup-dir",
            str(tmp_path / "backups"),
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "Production preflight passed.\n"


def test_host_preflight_remains_authoritative_for_env_permissions(
    tmp_path: Path,
) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)
    env_file.chmod(0o644)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert "env_permissions" in _codes(report.issues)


@pytest.mark.parametrize("mode", (0o400, 0o640, 0o700))
def test_host_preflight_requires_exact_env_mode_0600(tmp_path: Path, mode: int) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)
    env_file.chmod(mode)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert "env_permissions" in _codes(report.issues)


def test_host_preflight_requires_env_owner_to_match_effective_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)
    monkeypatch.setattr(
        "Helper_Scripts.Deployment.production_preflight.os.geteuid",
        lambda: env_file.stat().st_uid + 1,
    )

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert "env_permissions" in _codes(report.issues)


def test_host_preflight_rejects_symlinked_raw_env_file(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    target = tmp_path / "target.env"
    _write_env(target, values)
    env_file = tmp_path / "production.env"
    env_file.symlink_to(target)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert "env_permissions" in _codes(report.issues)


def test_container_environment_mode_skips_raw_file_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _valid_env(tmp_path)
    monkeypatch.setattr(
        "Helper_Scripts.Deployment.production_preflight.os.geteuid",
        lambda: -1,
    )

    issues = validate_environment(values, env_path=None)

    assert "env_permissions" not in _codes(issues)


def test_host_preflight_remains_authoritative_for_backup_writability(
    tmp_path: Path,
) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)
    (tmp_path / "backups").chmod(0o500)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert "backup_unwritable" in _codes(report.issues)


def test_host_preflight_does_not_substitute_runtime_backup_directory(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    configured_backup = tmp_path / "backups"
    configured_backup.chmod(0o500)
    substitute_backup = tmp_path / "substitute-backups"
    substitute_backup.mkdir()
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)

    report = run_preflight(
        env_file,
        COMPOSE_PATH,
        PROXY_PATH,
        runtime_backup_dir=substitute_backup,
    )

    assert "backup_unwritable" in _codes(report.issues)


def test_raw_environment_validation_cannot_bypass_host_backup_authority(tmp_path: Path) -> None:
    values = _valid_env(tmp_path)
    (tmp_path / "backups").chmod(0o500)
    substitute_backup = tmp_path / "substitute-backups"
    substitute_backup.mkdir()
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)

    issues = validate_environment(
        values,
        env_path=env_file,
        runtime_backup_dir=substitute_backup,
        require_backup_writable=False,
    )

    assert "backup_unwritable" in _codes(issues)


def test_run_preflight_converts_parse_errors_to_sorted_issues(tmp_path: Path) -> None:
    env_file = tmp_path / "production.env"
    env_file.write_text("export SECRET=value\n", encoding="utf-8")
    env_file.chmod(0o600)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert not report.ok
    assert [issue.code for issue in report.issues] == sorted(issue.code for issue in report.issues)
    assert report.issues[0].code == "env_parse"
    assert "value" not in report.issues[0].message


def test_run_preflight_converts_unreadable_inputs_to_sanitized_issues(tmp_path: Path) -> None:
    report = run_preflight(
        tmp_path / "missing.env",
        tmp_path / "missing-compose.yml",
        tmp_path / "missing-Caddyfile",
    )

    assert [issue.code for issue in report.issues] == [
        "compose_parse",
        "env_parse",
        "proxy_parse",
    ]
    rendered = "\n".join(issue.message for issue in report.issues)
    assert str(tmp_path) not in rendered


def test_cli_returns_two_for_missing_or_conflicting_source_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    assert main(["--env-file", str(tmp_path / "production.env"), "--from-environment"]) == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_rejects_runtime_backup_directory_with_raw_env_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)

    exit_code = main(
        [
            "--env-file",
            str(env_file),
            "--runtime-backup-dir",
            str(tmp_path / "backups"),
        ]
    )

    assert exit_code == 2
    assert "--runtime-backup-dir requires --from-environment" in capsys.readouterr().err


def test_cli_prints_sorted_sanitized_errors_only_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "must-never-appear-" + "S" * 40
    values = _valid_env(tmp_path)
    values["JWT_SECRET_KEY"] = "short"
    values["POSTGRES_PASSWORD"] = secret
    values["DATABASE_URL"] = "postgresql://tldw_app:different@postgres:5432/tldw"
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)

    exit_code = main(
        [
            "--env-file",
            str(env_file),
            "--compose-file",
            str(COMPOSE_PATH),
            "--proxy-file",
            str(PROXY_PATH),
        ]
    )
    captured = capsys.readouterr()
    lines = captured.err.splitlines()

    assert exit_code == 1
    assert captured.out == ""
    assert lines == sorted(lines)
    assert all(line.startswith("ERROR [") for line in lines)
    assert secret not in captured.err
    assert "postgresql://" not in captured.err
