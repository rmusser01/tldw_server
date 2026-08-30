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
        b'DOUBLE="quoted value"\r\nDOLLAR=$(whoami)\r\nBACKTICK=`id`\r\n'
    )

    assert load_raw_env(path) == {
        "PLAIN": "value",
        "HASH": "value#part",
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
    ("backup_value", "expected_code"),
    (("relative/backups", "unsafe_backup_path"), ("/app/Databases", "live_data_path")),
)
def test_environment_rejects_unsafe_backup_paths(tmp_path: Path, backup_value: str, expected_code: str) -> None:
    values = _valid_env(tmp_path)
    values["TLDW_BACKUP_DIR"] = backup_value

    assert expected_code in _codes(validate_environment(values, env_path=tmp_path / "production.env"))


def test_static_compose_and_proxy_match_the_reference_contract() -> None:
    assert validate_compose(_real_compose()) == ()
    assert validate_proxy(PROXY_PATH.read_text(encoding="utf-8")) == ()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("project_name", "topology_project"),
        ("app_port", "topology_ports"),
        ("backend_public", "topology_network"),
        ("caddy_backend", "topology_network"),
        ("missing_preflight", "topology_preflight"),
        ("missing_preflight_env_file", "topology_preflight"),
        ("unexpected_service", "topology_services"),
        ("missing_caddyfile", "topology_mounts"),
        ("missing_caddy_environment", "topology_proxy"),
        ("docker_socket", "topology_socket"),
        ("wildcard_trust", "topology_trust"),
        ("missing_redis_auth", "topology_redis_auth"),
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
    elif mutation == "backend_public":
        compose["networks"]["backend"]["internal"] = False
    elif mutation == "caddy_backend":
        services["caddy"]["networks"].append("backend")
    elif mutation == "missing_preflight":
        del services["app"]["depends_on"]["preflight"]
    elif mutation == "missing_preflight_env_file":
        del services["preflight"]["env_file"]
    elif mutation == "unexpected_service":
        services["debug"] = {"ports": ["9000:9000"]}
    elif mutation == "missing_caddyfile":
        services["caddy"]["volumes"].remove("./Production/Caddyfile:/etc/caddy/Caddyfile:ro")
    elif mutation == "missing_caddy_environment":
        del services["caddy"]["environment"]["TLDW_PUBLIC_DOMAIN"]
    elif mutation == "docker_socket":
        services["preflight"]["volumes"].append("/var/run/docker.sock:/var/run/docker.sock")
    elif mutation == "wildcard_trust":
        services["app"]["environment"]["RG_TRUSTED_PROXIES"] = "0.0.0.0/0"
    elif mutation == "missing_redis_auth":
        services["redis"]["command"] = ["redis-server"]
    elif mutation == "image_default":
        services["app"]["image"] = "${TLDW_APP_IMAGE:-registry/tldw:latest}"

    assert expected_code in _codes(validate_compose(compose))


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


def test_rendered_compose_matches_concrete_environment(tmp_path: Path) -> None:
    compose, values = _rendered_compose(tmp_path)

    assert validate_rendered_compose(compose, values) == ()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("secret_command", "rendered_secret"),
        ("published_app", "rendered_ports"),
        ("wrong_image", "rendered_image"),
        ("public_backend", "rendered_network"),
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

    assert expected_code in _codes(validate_rendered_compose(compose, values))


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


def test_host_preflight_remains_authoritative_for_backup_writability(
    tmp_path: Path,
) -> None:
    values = _valid_env(tmp_path)
    env_file = tmp_path / "production.env"
    _write_env(env_file, values)
    (tmp_path / "backups").chmod(0o500)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert "backup_unwritable" in _codes(report.issues)


def test_run_preflight_converts_parse_errors_to_sorted_issues(tmp_path: Path) -> None:
    env_file = tmp_path / "production.env"
    env_file.write_text("export SECRET=value\n", encoding="utf-8")
    env_file.chmod(0o600)

    report = run_preflight(env_file, COMPOSE_PATH, PROXY_PATH)

    assert not report.ok
    assert [issue.code for issue in report.issues] == sorted(issue.code for issue in report.issues)
    assert report.issues[0].code == "env_parse"
    assert "value" not in report.issues[0].message
