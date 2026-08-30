from __future__ import annotations

import io
import json
import os
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import Helper_Scripts.Deployment.production_deploy as production_deploy
import pytest
from Helper_Scripts.Deployment.production_artifacts import (
    ArtifactRecord,
    DeploymentManifest,
    load_verified_manifest,
    sha256_file,
    verify_tar_archive,
    write_manifest,
)
from Helper_Scripts.Deployment.production_deploy import (
    CommandResult,
    DeploymentConfig,
    DeploymentError,
    deploy,
    rollback,
)
from Helper_Scripts.Deployment.production_preflight import PreflightIssue, PreflightReport


def _write_tar(path: Path, *, member_name: str = "data/state.db") -> None:
    payload = b"application-data"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _record(path: Path, kind: str) -> ArtifactRecord:
    return ArtifactRecord(
        kind=kind,
        path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _manifest(
    tmp_path: Path, *, compose_file: Path | None = None
) -> tuple[DeploymentManifest, Path]:
    postgres = tmp_path / "postgres.dump"
    postgres.write_bytes(b"custom-dump-fixture")
    redis = tmp_path / "redis.rdb"
    redis.write_bytes(b"REDIS0011fixture")
    app_data = tmp_path / "app-data.tar"
    _write_tar(app_data)
    manifest = DeploymentManifest(
        created_at="2026-08-30T00:00:00Z",
        target_image="registry/tldw:sha-1234567",
        rollback_image="registry/tldw:sha-7654321",
        compose_file_sha256=(sha256_file(compose_file) if compose_file else "a" * 64),
        artifacts=(
            _record(postgres, "postgresql"),
            _record(redis, "redis"),
            _record(app_data, "app_data"),
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    return manifest, path


def test_manifest_contains_checksums_but_no_secrets(tmp_path: Path) -> None:
    artifact = tmp_path / "postgres.dump"
    artifact.write_bytes(b"custom-dump-fixture")
    record = _record(artifact, "postgresql")
    manifest = DeploymentManifest(
        created_at="2026-08-30T00:00:00Z",
        target_image="registry/tldw:sha-1234567",
        rollback_image="registry/tldw:sha-7654321",
        compose_file_sha256="a" * 64,
        artifacts=(record,),
    )
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    text = path.read_text(encoding="utf-8")
    assert "password" not in text.lower()
    assert "database_url" not in text.lower()
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_verified_manifest(path) == manifest


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda body: body["artifacts"][0].update(path="../postgres.dump"), "path"),
        (lambda body: body["artifacts"][0].update(sha256="0" * 64), "checksum"),
        (lambda body: body["artifacts"][0].update(size_bytes=0), "size"),
        (lambda body: body["artifacts"][0].update(kind="unknown"), "kind"),
        (lambda body: body["artifacts"].append(dict(body["artifacts"][0])), "duplicate"),
    ),
)
def test_manifest_rejects_unsafe_or_unverifiable_artifacts(
    tmp_path: Path, mutation, expected: str
) -> None:
    _, path = _manifest(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    mutation(body)
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        load_verified_manifest(path)


def test_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_verified_manifest(path)


def test_manifest_and_artifacts_must_not_be_symbolic_links(tmp_path: Path) -> None:
    manifest, path = _manifest(tmp_path)
    artifact = tmp_path / manifest.artifacts[0].path
    real_artifact = tmp_path / "real-postgres.dump"
    artifact.rename(real_artifact)
    artifact.symlink_to(real_artifact)

    with pytest.raises(ValueError, match="symbolic link"):
        load_verified_manifest(path)

    real_manifest = tmp_path / "real-manifest.json"
    path.rename(real_manifest)
    path.symlink_to(real_manifest)
    with pytest.raises(ValueError, match="symbolic link"):
        load_verified_manifest(path)


@pytest.mark.parametrize("member_name", ("/etc/passwd", "../escape", "data/../../escape"))
def test_tar_verification_rejects_unsafe_member_paths(
    tmp_path: Path, member_name: str
) -> None:
    path = tmp_path / "unsafe.tar"
    _write_tar(path, member_name=member_name)

    with pytest.raises(ValueError, match="unsafe"):
        verify_tar_archive(path)


def test_tar_verification_rejects_links_and_archives_without_regular_data(
    tmp_path: Path,
) -> None:
    link_path = tmp_path / "link.tar"
    with tarfile.open(link_path, "w") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(ValueError, match="link"):
        verify_tar_archive(link_path)

    empty_path = tmp_path / "empty.tar"
    with tarfile.open(empty_path, "w") as archive:
        archive.addfile(tarfile.TarInfo("data/"))
    with pytest.raises(ValueError, match="regular"):
        verify_tar_archive(empty_path)


def test_tar_verification_rejects_unreadable_archive(tmp_path: Path) -> None:
    path = tmp_path / "broken.tar"
    path.write_bytes(b"not a tar archive")

    with pytest.raises(ValueError, match="archive"):
        verify_tar_archive(path)


def test_default_streaming_runner_writes_stdout_directly_to_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "postgres.dump"
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed.update(kwargs)
        kwargs["stdout"].write(b"custom-")
        kwargs["stdout"].write(b"postgres-dump")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(production_deploy.subprocess, "run", fake_run)

    result = production_deploy.default_streaming_command_runner(
        ("docker", "compose", "exec", "postgres", "pg_dump"),
        {"PATH": "/usr/bin"},
        destination,
    )

    assert destination.read_bytes() == b"custom-postgres-dump"
    assert destination.stat().st_mode & 0o777 == 0o600
    assert result == CommandResult(returncode=0, stdout=b"", stderr=b"")
    assert observed["shell"] is False
    assert observed["check"] is False
    assert "capture_output" not in observed


def test_default_streaming_runner_removes_partial_failed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "postgres.dump"

    def fake_run(argv, **kwargs):
        kwargs["stdout"].write(b"partial-secret-dump")
        return SimpleNamespace(returncode=19, stderr=b"raw-secret")

    monkeypatch.setattr(production_deploy.subprocess, "run", fake_run)

    result = production_deploy.default_streaming_command_runner(
        ("docker", "compose", "exec", "postgres", "pg_dump"),
        None,
        destination,
    )

    assert result.returncode == 19
    assert not destination.exists()


def test_default_streaming_runner_enforces_mode_under_restrictive_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "postgres.dump"

    def fake_run(argv, **kwargs):
        kwargs["stdout"].write(b"custom-postgres-dump")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(production_deploy.subprocess, "run", fake_run)
    previous_umask = os.umask(0o777)
    try:
        production_deploy.default_streaming_command_runner(
            ("docker", "compose", "exec", "postgres", "pg_dump"),
            None,
            destination,
        )
    finally:
        os.umask(previous_umask)

    assert destination.stat().st_mode & 0o777 == 0o600


class RecordingRunner:
    def __init__(self, *, fail_when: str | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], Mapping[str, str] | None, bytes | None]] = []
        self.fail_when = fail_when

    def __call__(
        self,
        argv: Sequence[str],
        env: Mapping[str, str] | None,
        input_bytes: bytes | None,
    ) -> CommandResult:
        args = tuple(argv)
        self.calls.append((args, env, input_bytes))
        joined = " ".join(args)
        if self.fail_when and self.fail_when in joined:
            return CommandResult(
                returncode=19,
                stdout=b"postgresql://operator:raw-secret@postgres/tldw",
                stderr=b"raw-secret",
            )
        if args[-3:] == ("config", "--format", "json"):
            return CommandResult(0, b"{}", b"")
        if "pg_dump" in args:
            return CommandResult(0, b"custom-postgres-dump", b"")
        if "cp" in args and "redis:/data/dump.rdb" in args:
            Path(args[-1]).write_bytes(b"REDIS0011fixture")
        if "production_app-data:/data:ro" in joined:
            mount = args[args.index("-v", args.index("-v") + 1) + 1]
            backup_dir = Path(mount.rsplit(":/backup", 1)[0])
            _write_tar(backup_dir / "app-data.tar")
        return CommandResult(0, b"", b"")


class RecordingStreamRunner:
    def __init__(self) -> None:
        self.calls: list[
            tuple[tuple[str, ...], Mapping[str, str] | None, Path]
        ] = []

    def __call__(
        self,
        argv: Sequence[str],
        env: Mapping[str, str] | None,
        destination: Path,
    ) -> CommandResult:
        self.calls.append((tuple(argv), env, destination))
        destination.write_bytes(b"custom-postgres-dump")
        return CommandResult(0, b"", b"")


def _config(tmp_path: Path) -> DeploymentConfig:
    env_file = tmp_path / "production.env"
    env_file.write_text("names-only-fixture=true\n", encoding="utf-8")
    env_file.chmod(0o600)
    compose_file = tmp_path / "compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    proxy_file = tmp_path / "Caddyfile"
    proxy_file.write_text("fixture\n", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    return DeploymentConfig(
        env_file=env_file,
        compose_file=compose_file,
        proxy_file=proxy_file,
        backup_dir=backup_dir,
        values={
            "TLDW_APP_IMAGE": "registry/tldw:sha-1234567",
            "TLDW_ROLLBACK_IMAGE": "registry/tldw:sha-7654321",
            "POSTGRES_IMAGE": "postgres:18.0-bookworm",
            "REDIS_IMAGE": "redis:7.4.1-alpine",
            "POSTGRES_USER": "tldw_app",
            "POSTGRES_DB": "tldw",
        },
    )


@pytest.fixture
def passing_deployment_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.Deployment.production_deploy.run_preflight",
        lambda *args, **kwargs: PreflightReport(()),
    )
    monkeypatch.setattr(
        "Helper_Scripts.Deployment.production_deploy.validate_rendered_compose",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        "Helper_Scripts.Deployment.production_deploy.run_preflight_from_environment",
        lambda *args, **kwargs: PreflightReport(()),
        raising=False,
    )


def _commands(runner: RecordingRunner) -> tuple[str, ...]:
    return tuple(" ".join(call[0]) for call in runner.calls)


def test_deploy_runs_every_gate_before_final_start(
    tmp_path: Path, passing_deployment_checks: None
) -> None:
    config = _config(tmp_path)
    runner = RecordingRunner()

    manifest = deploy(config, runner=runner)

    commands = _commands(runner)
    markers = (
        "config --format json",
        "docker pull registry/tldw:sha-1234567",
        "docker pull registry/tldw:sha-7654321",
        "--network none --entrypoint python registry/tldw:sha-1234567",
        "--network none --entrypoint python registry/tldw:sha-7654321",
        "up -d --wait postgres redis",
        "stop app caddy",
        "pg_dump --format=custom",
        "--entrypoint pg_restore",
        "redis-cli SAVE",
        "redis:/data/dump.rdb",
        "--entrypoint redis-check-rdb",
        "tldw-production_app-data:/data:ro",
        "up -d --remove-orphans",
    )
    positions = [next(i for i, command in enumerate(commands) if marker in command) for marker in markers]
    assert positions == sorted(positions)
    data_start = next(command for command in commands if "up -d" in command and "postgres redis" in command)
    assert "--wait" in data_start
    assert {item.kind for item in manifest.artifacts} == {
        "postgresql",
        "redis",
        "app_data",
    }
    manifest_path = next(config.backup_dir.rglob("manifest.json"))
    assert load_verified_manifest(manifest_path) == manifest


def test_deploy_uses_injected_streaming_runner_for_postgres_dump(
    tmp_path: Path, passing_deployment_checks: None
) -> None:
    runner = RecordingRunner()
    stream_runner = RecordingStreamRunner()

    deploy(_config(tmp_path), runner=runner, stream_runner=stream_runner)

    assert len(stream_runner.calls) == 1
    assert "pg_dump --format=custom" in " ".join(stream_runner.calls[0][0])
    assert not any("pg_dump" in command for command in _commands(runner))
    assert stream_runner.calls[0][2].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("outcome", ("empty", "exception"))
def test_deploy_removes_unusable_injected_stream_output(
    tmp_path: Path,
    passing_deployment_checks: None,
    outcome: str,
) -> None:
    config = _config(tmp_path)

    def unusable_stream_runner(argv, env, destination):
        destination.write_bytes(b"partial" if outcome == "exception" else b"")
        if outcome == "exception":
            raise RuntimeError("raw-secret")
        return CommandResult(0, b"", b"")

    with pytest.raises(DeploymentError) as exc_info:
        deploy(
            config,
            runner=RecordingRunner(),
            stream_runner=unusable_stream_runner,
        )

    assert "raw-secret" not in str(exc_info.value)
    assert not tuple(config.backup_dir.rglob("postgres.dump"))
    assert not tuple(config.backup_dir.rglob("manifest.json"))


@pytest.mark.parametrize(
    "failed_gate",
    (
        "config --format json",
        "docker pull registry/tldw:sha-1234567",
        "--entrypoint python registry/tldw:sha-1234567",
        "up -d --wait postgres redis",
        "pg_dump --format=custom",
        "--entrypoint redis-check-rdb",
    ),
)
def test_deploy_stops_after_each_failed_gate_without_leaking_output(
    tmp_path: Path, passing_deployment_checks: None, failed_gate: str
) -> None:
    runner = RecordingRunner(fail_when=failed_gate)

    with pytest.raises(DeploymentError) as exc_info:
        deploy(_config(tmp_path), runner=runner)

    assert "raw-secret" not in str(exc_info.value)
    assert "postgresql://" not in str(exc_info.value)
    assert not any("up -d --remove-orphans" in item for item in _commands(runner))


def test_rollback_verifies_and_restores_all_artifacts_before_prior_image_start(
    tmp_path: Path, passing_deployment_checks: None
) -> None:
    config = _config(tmp_path)
    manifest, manifest_path = _manifest(
        config.backup_dir, compose_file=config.compose_file
    )
    runner = RecordingRunner()

    rollback(config, manifest_path, runner=runner)

    commands = _commands(runner)
    markers = (
        "config --format json",
        "stop app caddy",
        "pg_restore --clean --if-exists --no-owner",
        "stop redis",
        "tldw-production_redis_data:/data",
        "tldw-production_app-data:/data",
        "up -d --remove-orphans",
    )
    positions = [next(i for i, command in enumerate(commands) if marker in command) for marker in markers]
    assert positions == sorted(positions)
    final_env = runner.calls[-1][1]
    assert final_env is not None
    assert final_env["TLDW_APP_IMAGE"] == "registry/tldw:sha-7654321"
    assert manifest_path.read_text(encoding="utf-8") == (
        config.backup_dir / "manifest.json"
    ).read_text(encoding="utf-8")
    assert tuple(config.backup_dir.glob("rollback-*.json"))
    restore_calls = tuple(call[0] for call in runner.calls)
    assert manifest.artifacts[1].sha256 in next(
        " ".join(call) for call in restore_calls if "production_redis_data:/data" in " ".join(call)
    )
    assert manifest.artifacts[2].sha256 in next(
        " ".join(call) for call in restore_calls if "production_app-data:/data" in " ".join(call)
    )


def test_rollback_preflights_swapped_prior_image_values_before_restore(
    tmp_path: Path,
    passing_deployment_checks: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _, manifest_path = _manifest(
        config.backup_dir, compose_file=config.compose_file
    )
    preflight_values: list[Mapping[str, str]] = []

    def record_preflight(values, compose_file, proxy_file, **kwargs):
        preflight_values.append(dict(values))
        return PreflightReport(())

    monkeypatch.setattr(
        production_deploy,
        "run_preflight_from_environment",
        record_preflight,
        raising=False,
    )
    runner = RecordingRunner()

    rollback(config, manifest_path, runner=runner)

    assert len(preflight_values) == 1
    assert preflight_values[0]["TLDW_APP_IMAGE"] == "registry/tldw:sha-7654321"
    assert preflight_values[0]["TLDW_ROLLBACK_IMAGE"] == "registry/tldw:sha-1234567"
    assert any("pg_restore --clean" in command for command in _commands(runner))


def test_rollback_stops_before_commands_when_swapped_preflight_fails(
    tmp_path: Path,
    passing_deployment_checks: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _, manifest_path = _manifest(
        config.backup_dir, compose_file=config.compose_file
    )
    monkeypatch.setattr(
        production_deploy,
        "run_preflight_from_environment",
        lambda *args, **kwargs: PreflightReport(
            (
                PreflightIssue(
                    "mutable_image",
                    "TLDW_APP_IMAGE",
                    "raw-secret must not appear",
                ),
            )
        ),
        raising=False,
    )
    runner = RecordingRunner()

    with pytest.raises(DeploymentError) as exc_info:
        rollback(config, manifest_path, runner=runner)

    assert "raw-secret" not in str(exc_info.value)
    assert runner.calls == []


@pytest.mark.parametrize("failed_restore", ("pg_restore --clean", "production_redis_data:/data", "production_app-data:/data"))
def test_rollback_never_starts_prior_image_after_restore_failure(
    tmp_path: Path, passing_deployment_checks: None, failed_restore: str
) -> None:
    config = _config(tmp_path)
    _, manifest_path = _manifest(
        config.backup_dir, compose_file=config.compose_file
    )
    runner = RecordingRunner(fail_when=failed_restore)

    with pytest.raises(DeploymentError):
        rollback(config, manifest_path, runner=runner)

    assert not any("up -d --remove-orphans" in item for item in _commands(runner))


@pytest.mark.parametrize("breakage", ("missing", "checksum", "image"))
def test_rollback_rejects_unverified_or_mismatched_state_before_start(
    tmp_path: Path, passing_deployment_checks: None, breakage: str
) -> None:
    config = _config(tmp_path)
    manifest, manifest_path = _manifest(
        config.backup_dir, compose_file=config.compose_file
    )
    if breakage == "missing":
        (config.backup_dir / manifest.artifacts[0].path).unlink()
    elif breakage == "checksum":
        (config.backup_dir / manifest.artifacts[0].path).write_bytes(b"tampered")
    else:
        config = DeploymentConfig(
            env_file=config.env_file,
            compose_file=config.compose_file,
            proxy_file=config.proxy_file,
            backup_dir=config.backup_dir,
            values={**config.values, "TLDW_ROLLBACK_IMAGE": "registry/tldw:sha-deadbee"},
        )
    runner = RecordingRunner()

    with pytest.raises((DeploymentError, ValueError)):
        rollback(config, manifest_path, runner=runner)

    assert not any("up -d --remove-orphans" in item for item in _commands(runner))
