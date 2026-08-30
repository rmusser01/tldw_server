"""Fail-closed production deployment and restore-backed rollback orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess  # nosec B404
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from Helper_Scripts.Deployment.production_artifacts import (
    ArtifactRecord,
    DeploymentManifest,
    load_verified_manifest,
    sha256_file,
    verify_tar_archive,
    write_manifest,
)
from Helper_Scripts.Deployment.production_preflight import (
    DEFAULT_COMPOSE_FILE,
    DEFAULT_PROXY_FILE,
    load_raw_env,
    run_preflight,
    run_preflight_from_environment,
    validate_rendered_compose,
)

_IMPORT_SMOKE = "import tldw_Server_API.app.main"
_ARCHIVE_SCRIPT = """\
import os, sys, tarfile
root = sys.argv[1]
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for name in sorted(os.listdir(root)):
        archive.add(os.path.join(root, name), arcname=name, recursive=True)
"""
_RESTORE_APP_SCRIPT = """\
import hashlib, os, shutil, sys, tarfile, tempfile
from pathlib import Path, PurePosixPath

source = Path(sys.argv[1])
expected = sys.argv[2]
data = Path(sys.argv[3])
stage = Path(tempfile.mkdtemp(prefix=".tldw-app-restore-", dir=str(data)))
staged_archive = stage / "archive.tar"
new = stage / "new"
old = stage / "old"
new.mkdir()
old.mkdir()
moved_old = []
installed = []
replacement_started = False
rollback_complete = False

def remove_path(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)

def apply_metadata(path, member):
    os.chmod(path, member.mode & 0o777)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.chown(path, member.uid, member.gid, follow_symlinks=False)

try:
    digest = hashlib.sha256()
    with open(source, "rb") as input_stream, open(staged_archive, "xb") as output_stream:
        while True:
            chunk = input_stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output_stream.write(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if digest.hexdigest() != expected:
        raise RuntimeError("app-data checksum mismatch")

    seen = set()
    directory_metadata = []
    with tarfile.open(staged_archive, "r:*") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if (
                not member.name
                or member_path.is_absolute()
                or ".." in member_path.parts
                or not (member.isdir() or member.isfile())
                or member.name in seen
            ):
                raise RuntimeError("app-data archive contains an unsafe member")
            seen.add(member.name)

    with tarfile.open(staged_archive, "r:*") as archive:
        for member in archive:
            target = new.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directory_metadata.append((target, member))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError("app-data archive member is unreadable")
            with extracted, open(target, "xb") as output_stream:
                shutil.copyfileobj(extracted, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            apply_metadata(target, member)
    for target, member in sorted(directory_metadata, key=lambda item: len(item[0].parts), reverse=True):
        apply_metadata(target, member)

    replacement_started = True
    for entry in sorted(data.iterdir(), key=lambda path: path.name):
        if entry == stage:
            continue
        destination = old / entry.name
        os.replace(entry, destination)
        moved_old.append(destination)
    for entry in sorted(new.iterdir(), key=lambda path: path.name):
        destination = data / entry.name
        os.replace(entry, destination)
        installed.append(destination)
except BaseException:
    if replacement_started:
        rollback_complete = True
        for entry in reversed(installed):
            try:
                remove_path(entry)
            except OSError:
                rollback_complete = False
        for entry in reversed(moved_old):
            try:
                os.replace(entry, data / entry.name)
            except OSError:
                rollback_complete = False
    raise
else:
    rollback_complete = True
finally:
    if not replacement_started or rollback_complete:
        shutil.rmtree(stage, ignore_errors=True)
"""
_RESTORE_REDIS_SCRIPT = """\
import hashlib, os, sys, tempfile
from pathlib import Path

source = Path(sys.argv[1])
expected = sys.argv[2]
data = Path(sys.argv[3])
descriptor, staged_name = tempfile.mkstemp(prefix=".tldw-redis-restore-", dir=str(data))
staged = Path(staged_name)
success = False
try:
    digest = hashlib.sha256()
    with open(source, "rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
        descriptor = -1
        while True:
            chunk = input_stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output_stream.write(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if digest.hexdigest() != expected:
        raise RuntimeError("Redis checksum mismatch")
    owner = data.stat().st_uid, data.stat().st_gid
    os.chown(staged, *owner)
    os.chmod(staged, 0o600)
    os.replace(staged, data / "dump.rdb")
    success = True
finally:
    if descriptor >= 0:
        os.close(descriptor)
    if not success:
        staged.unlink(missing_ok=True)
"""
_INHERITED_ENV_NAMES = (
    "PATH",
    "HOME",
    "TMPDIR",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "XDG_RUNTIME_DIR",
)


@dataclass(frozen=True)
class CommandResult:
    """Captured result from one fixed-argument command."""

    returncode: int
    stdout: bytes
    stderr: bytes


CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None, bytes | None], CommandResult]
StreamingCommandRunner = Callable[[Sequence[str], Mapping[str, str] | None, Path], CommandResult]


@dataclass(frozen=True)
class DeploymentConfig:
    """Validated file locations and raw values for one production operation."""

    env_file: Path
    compose_file: Path
    proxy_file: Path
    backup_dir: Path
    values: Mapping[str, str]


class DeploymentError(RuntimeError):
    """Sanitized deployment gate failure suitable for operator output."""


def default_command_runner(
    argv: Sequence[str],
    env: Mapping[str, str] | None,
    input_bytes: bytes | None,
) -> CommandResult:
    """Run one explicit argv without a shell and capture binary output."""

    # The only caller paths assemble reviewed argv tuples and explicitly disable a shell.
    completed = subprocess.run(  # nosec B603
        list(argv),
        shell=False,
        check=False,
        env=dict(env) if env is not None else None,
        input=input_bytes,
        capture_output=True,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def default_streaming_command_runner(
    argv: Sequence[str],
    env: Mapping[str, str] | None,
    destination: Path,
) -> CommandResult:
    """Run one explicit argv while streaming stdout to a private new file."""

    descriptor = -1
    created = False
    success = False
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            completed = subprocess.run(  # nosec B603
                list(argv),
                shell=False,
                check=False,
                env=dict(env) if env is not None else None,
                stdout=stream,
                stderr=subprocess.PIPE,
            )
            stream.flush()
            os.fsync(stream.fileno())
        result = CommandResult(
            returncode=completed.returncode,
            stdout=b"",
            stderr=completed.stderr,
        )
        success = result.returncode == 0
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created and not success:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass


def _command_env(config: DeploymentConfig, **overrides: str) -> dict[str, str]:
    """Build the bounded process environment used by Docker Compose."""

    env = {name: value for name in _INHERITED_ENV_NAMES if (value := os.environ.get(name)) is not None}
    env.update(config.values)
    env["TLDW_ENV_FILE"] = str(config.env_file)
    env.update(overrides)
    return env


def _compose_prefix(config: DeploymentConfig) -> tuple[str, ...]:
    """Return the canonical fixed Compose prefix."""

    return (
        "docker",
        "compose",
        "--env-file",
        str(config.env_file),
        "-f",
        str(config.compose_file),
    )


def _run_gate(
    runner: CommandRunner,
    label: str,
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    input_bytes: bytes | None = None,
) -> CommandResult:
    """Run a command and expose only a gate label and status on failure."""

    try:
        result = runner(tuple(argv), env, input_bytes)
    except (OSError, RuntimeError) as exc:
        raise DeploymentError(f"{label} gate could not execute") from exc
    if result.returncode != 0:
        raise DeploymentError(f"{label} gate failed with exit status {result.returncode}")
    return result


def _run_streaming_gate(
    runner: StreamingCommandRunner,
    label: str,
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    destination: Path,
) -> CommandResult:
    """Run a command whose stdout must stream to one private artifact."""

    success = False
    try:
        try:
            result = runner(tuple(argv), env, destination)
        except Exception as exc:
            raise DeploymentError(f"{label} gate could not execute") from exc
        if result.returncode != 0:
            raise DeploymentError(f"{label} gate failed with exit status {result.returncode}")
        try:
            _require_nonempty_file(destination, label)
        except (DeploymentError, OSError) as exc:
            raise DeploymentError(f"{label} gate produced an unusable artifact") from exc
        success = True
        return result
    finally:
        if not success:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass


def _preflight(config: DeploymentConfig) -> None:
    """Run the complete static gate without exposing candidate values."""

    report = run_preflight(config.env_file, config.compose_file, config.proxy_file)
    if report.issues:
        codes = ", ".join(sorted({issue.code for issue in report.issues}))
        raise DeploymentError(f"preflight gate failed ({codes})")


def _render_and_validate(
    config: DeploymentConfig,
    runner: CommandRunner,
    env: Mapping[str, str],
    values: Mapping[str, str],
) -> None:
    """Render Compose in memory and revalidate the resolved topology."""

    result = _run_gate(
        runner,
        "compose render",
        (*_compose_prefix(config), "config", "--format", "json"),
        env=env,
    )
    try:
        document = json.loads(result.stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentError("compose render gate returned invalid JSON") from exc
    if not isinstance(document, Mapping):
        raise DeploymentError("compose render gate returned an invalid model")
    issues = validate_rendered_compose(document, values)
    if issues:
        codes = ", ".join(sorted({issue.code for issue in issues}))
        raise DeploymentError(f"rendered compose gate failed ({codes})")


def _write_private_bytes(path: Path, data: bytes, label: str) -> None:
    """Persist a nonempty command artifact with owner-only permissions."""

    if not data:
        raise DeploymentError(f"{label} gate produced an empty artifact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _require_nonempty_file(path: Path, label: str) -> None:
    """Fail if a container did not create a usable artifact."""

    try:
        valid = path.is_file() and path.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        raise DeploymentError(f"{label} gate did not create a nonempty artifact")
    path.chmod(0o600)


def _created_at() -> str:
    """Return an unambiguous UTC manifest timestamp."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _snapshot_directory(backup_dir: Path, created_at: str) -> Path:
    """Return a deterministic safe directory for one backup set."""

    token = created_at.replace(":", "").replace("-", "").replace(".", "-")
    return backup_dir / f"deployment-{token}"


def _record(kind: str, path: Path) -> ArtifactRecord:
    """Create a verified artifact record."""

    _require_nonempty_file(path, kind)
    return ArtifactRecord(
        kind=kind,
        path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _image_smoke(image: str, runner: CommandRunner, env: Mapping[str, str], label: str) -> None:
    """Verify an application image can import the server without network access."""

    _run_gate(
        runner,
        label,
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            image,
            "-c",
            _IMPORT_SMOKE,
        ),
        env=env,
    )


def _existing_installation(value: str) -> bool:
    """Return the validated installation mode without accepting ambiguity."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DeploymentError("installation state gate is invalid")


def _verify_running_application(
    config: DeploymentConfig,
    runner: CommandRunner,
    env: Mapping[str, str],
) -> None:
    """Bind an upgrade's rollback image to the currently running app."""

    compose = _compose_prefix(config)
    result = _run_gate(
        runner,
        "running application discovery",
        (*compose, "ps", "-q", "--status", "running", "app"),
        env=env,
    )
    try:
        container_ids = tuple(line.strip() for line in result.stdout.decode("utf-8").splitlines() if line.strip())
    except UnicodeError as exc:
        raise DeploymentError("running application discovery gate returned invalid output") from exc
    existing = _existing_installation(config.values.get("TLDW_EXISTING_INSTALLATION", ""))
    if not existing:
        if container_ids:
            raise DeploymentError("initial installation gate found a running application")
        return
    if len(container_ids) != 1:
        raise DeploymentError("running application gate requires exactly one container")
    container_id = container_ids[0]
    if not (12 <= len(container_id) <= 64) or any(character not in "0123456789abcdef" for character in container_id):
        raise DeploymentError("running application gate returned an invalid container identifier")
    image_result = _run_gate(
        runner,
        "running application image",
        ("docker", "inspect", "--format", "{{.Config.Image}}", container_id),
        env=env,
    )
    try:
        image_lines = tuple(line.strip() for line in image_result.stdout.decode("utf-8").splitlines() if line.strip())
    except UnicodeError as exc:
        raise DeploymentError("running application image gate returned invalid output") from exc
    if len(image_lines) != 1 or image_lines[0] != config.values["TLDW_ROLLBACK_IMAGE"]:
        raise DeploymentError("running application rollback image gate failed")


def deploy(
    config: DeploymentConfig,
    *,
    runner: CommandRunner = default_command_runner,
    stream_runner: StreamingCommandRunner | None = None,
) -> DeploymentManifest:
    """Back up current state and start the target only after every gate passes."""

    _preflight(config)
    env = _command_env(config)
    _render_and_validate(config, runner, env, config.values)
    target_image = config.values["TLDW_APP_IMAGE"]
    rollback_image = config.values["TLDW_ROLLBACK_IMAGE"]
    postgres_image = config.values["POSTGRES_IMAGE"]
    redis_image = config.values["REDIS_IMAGE"]
    for label, image in (
        ("target image pull", target_image),
        ("rollback image pull", rollback_image),
    ):
        _run_gate(runner, label, ("docker", "pull", image), env=env)
    _image_smoke(target_image, runner, env, "target image smoke")
    _image_smoke(rollback_image, runner, env, "rollback image smoke")
    _verify_running_application(config, runner, env)

    compose = _compose_prefix(config)
    _run_gate(
        runner,
        "data services start",
        (*compose, "up", "-d", "--wait", "postgres", "redis"),
        env=env,
    )
    _run_gate(
        runner,
        "application quiesce",
        (*compose, "stop", "app", "caddy"),
        env=env,
    )

    created_at = _created_at()
    snapshot = _snapshot_directory(config.backup_dir, created_at)
    snapshot.mkdir(mode=0o700, parents=False, exist_ok=False)
    postgres_path = snapshot / "postgres.dump"
    redis_path = snapshot / "redis.rdb"
    app_path = snapshot / "app-data.tar"

    pg_dump_argv = (
        *compose,
        "exec",
        "-T",
        "postgres",
        "pg_dump",
        "--format=custom",
        "--no-owner",
        "--username",
        config.values["POSTGRES_USER"],
        "--dbname",
        config.values["POSTGRES_DB"],
    )
    active_stream_runner = stream_runner
    if active_stream_runner is None and runner is default_command_runner:
        active_stream_runner = default_streaming_command_runner
    if active_stream_runner is not None:
        _run_streaming_gate(
            active_stream_runner,
            "PostgreSQL backup",
            pg_dump_argv,
            env=env,
            destination=postgres_path,
        )
    else:
        pg_dump = _run_gate(
            runner,
            "PostgreSQL backup",
            pg_dump_argv,
            env=env,
        )
        _write_private_bytes(postgres_path, pg_dump.stdout, "PostgreSQL backup")
    _run_gate(
        runner,
        "PostgreSQL archive verification",
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "pg_restore",
            "-v",
            f"{snapshot}:/backup:ro",
            postgres_image,
            "--list",
            "/backup/postgres.dump",
        ),
        env=env,
    )

    _run_gate(
        runner,
        "Redis save",
        (
            *compose,
            "exec",
            "-T",
            "redis",
            "/bin/sh",
            "-ec",
            'REDISCLI_AUTH="$REDIS_PASSWORD" exec redis-cli SAVE',
        ),
        env=env,
    )
    _run_gate(
        runner,
        "Redis copy",
        (*compose, "cp", "redis:/data/dump.rdb", str(redis_path)),
        env=env,
    )
    _require_nonempty_file(redis_path, "Redis copy")
    _run_gate(
        runner,
        "Redis archive verification",
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "redis-check-rdb",
            "-v",
            f"{snapshot}:/backup:ro",
            redis_image,
            "/backup/redis.rdb",
        ),
        env=env,
    )

    app_archive_argv = (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "0:0",
        "--entrypoint",
        "python",
        "-v",
        "tldw-production_app-data:/data:ro",
        target_image,
        "-c",
        _ARCHIVE_SCRIPT,
        "/data",
    )
    if active_stream_runner is not None:
        _run_streaming_gate(
            active_stream_runner,
            "application data archive",
            app_archive_argv,
            env=env,
            destination=app_path,
        )
    else:
        app_archive = _run_gate(
            runner,
            "application data archive",
            app_archive_argv,
            env=env,
        )
        _write_private_bytes(app_path, app_archive.stdout, "application data archive")
    try:
        verify_tar_archive(app_path)
    except ValueError as exc:
        try:
            app_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            raise DeploymentError("application data cleanup gate failed") from cleanup_exc
        raise DeploymentError("application data verification gate failed") from exc

    manifest = DeploymentManifest(
        created_at=created_at,
        target_image=target_image,
        rollback_image=rollback_image,
        compose_file_sha256=sha256_file(config.compose_file),
        artifacts=(
            _record("postgresql", postgres_path),
            _record("redis", redis_path),
            _record("app_data", app_path),
        ),
    )
    write_manifest(snapshot / "manifest.json", manifest)
    _run_gate(
        runner,
        "target profile start",
        (*compose, "up", "-d", "--remove-orphans", "--wait"),
        env=env,
    )
    return manifest


def _artifacts_by_kind(manifest: DeploymentManifest, manifest_path: Path) -> dict[str, tuple[ArtifactRecord, Path]]:
    """Require a complete deploy backup set for rollback."""

    records = {record.kind: (record, manifest_path.parent / record.path) for record in manifest.artifacts}
    required = {"postgresql", "redis", "app_data"}
    if set(records) != required:
        raise DeploymentError("rollback manifest must contain all three artifact kinds")
    return records


def _read_verified_bytes(record: ArtifactRecord, path: Path) -> bytes:
    """Read an artifact once and recheck its digest against host-side races."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DeploymentError("rollback artifact became unreadable") from exc
    if hashlib.sha256(data).hexdigest() != record.sha256:
        raise DeploymentError("rollback artifact checksum changed during restore")
    return data


def rollback(
    config: DeploymentConfig,
    manifest_path: Path,
    *,
    runner: CommandRunner = default_command_runner,
) -> None:
    """Restore a verified backup set before starting its matching prior image."""

    _preflight(config)
    try:
        manifest = load_verified_manifest(manifest_path)
    except ValueError as exc:
        raise DeploymentError("rollback manifest verification gate failed") from exc
    if sha256_file(config.compose_file) != manifest.compose_file_sha256:
        raise DeploymentError("rollback compose checksum does not match manifest")
    if (
        config.values.get("TLDW_APP_IMAGE") != manifest.target_image
        or config.values.get("TLDW_ROLLBACK_IMAGE") != manifest.rollback_image
    ):
        raise DeploymentError("rollback image inputs do not match manifest")
    artifacts = _artifacts_by_kind(manifest, manifest_path)
    try:
        verify_tar_archive(artifacts["app_data"][1])
    except ValueError as exc:
        raise DeploymentError("rollback app-data verification gate failed") from exc

    rollback_values = {
        **config.values,
        "TLDW_APP_IMAGE": manifest.rollback_image,
        "TLDW_ROLLBACK_IMAGE": manifest.target_image,
    }
    rollback_report = run_preflight_from_environment(
        rollback_values,
        config.compose_file,
        config.proxy_file,
    )
    if rollback_report.issues:
        codes = ", ".join(sorted({issue.code for issue in rollback_report.issues}))
        raise DeploymentError(f"rollback preflight gate failed ({codes})")
    env = _command_env(
        config,
        TLDW_APP_IMAGE=manifest.rollback_image,
        TLDW_ROLLBACK_IMAGE=manifest.target_image,
    )
    _render_and_validate(config, runner, env, rollback_values)
    compose = _compose_prefix(config)
    _run_gate(
        runner,
        "rollback application quiesce",
        (*compose, "stop", "app", "caddy"),
        env=env,
    )
    _run_gate(
        runner,
        "rollback PostgreSQL start",
        (*compose, "up", "-d", "--wait", "postgres"),
        env=env,
    )
    _run_gate(
        runner,
        "PostgreSQL restore",
        (
            *compose,
            "exec",
            "-T",
            "postgres",
            "pg_restore",
            "--clean",
            "--if-exists",
            "--no-owner",
            "--username",
            config.values["POSTGRES_USER"],
            "--dbname",
            config.values["POSTGRES_DB"],
        ),
        env=env,
        input_bytes=_read_verified_bytes(*artifacts["postgresql"]),
    )
    _run_gate(
        runner,
        "Redis stop",
        (*compose, "stop", "redis"),
        env=env,
    )
    _run_gate(
        runner,
        "Redis restore",
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "--entrypoint",
            "python",
            "-v",
            "tldw-production_redis_data:/data",
            "-v",
            f"{manifest_path.parent}:/backup:ro",
            manifest.rollback_image,
            "-c",
            _RESTORE_REDIS_SCRIPT,
            f"/backup/{artifacts['redis'][0].path}",
            artifacts["redis"][0].sha256,
            "/data",
        ),
        env=env,
    )
    _run_gate(
        runner,
        "application data restore",
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "--entrypoint",
            "python",
            "-v",
            "tldw-production_app-data:/data",
            "-v",
            f"{manifest_path.parent}:/backup:ro",
            manifest.rollback_image,
            "-c",
            _RESTORE_APP_SCRIPT,
            f"/backup/{artifacts['app_data'][0].path}",
            artifacts["app_data"][0].sha256,
            "/data",
        ),
        env=env,
    )
    _run_gate(
        runner,
        "rollback profile start",
        (*compose, "up", "-d", "--remove-orphans", "--wait"),
        env=env,
    )
    completion = DeploymentManifest(
        created_at=_created_at(),
        target_image=manifest.rollback_image,
        rollback_image=manifest.target_image,
        compose_file_sha256=manifest.compose_file_sha256,
        artifacts=manifest.artifacts,
    )
    token = completion.created_at.replace(":", "").replace("-", "").replace(".", "-")
    write_manifest(manifest_path.parent / f"rollback-{token}.json", completion)


def _config_from_args(args: argparse.Namespace) -> DeploymentConfig:
    """Build CLI configuration from the literal operator env file."""

    try:
        values = load_raw_env(args.env_file)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DeploymentError("environment file could not be parsed") from exc
    backup_value = values.get("TLDW_BACKUP_DIR", "")
    if not backup_value:
        raise DeploymentError("TLDW_BACKUP_DIR is required")
    return DeploymentConfig(
        env_file=args.env_file,
        compose_file=args.compose_file,
        proxy_file=args.proxy_file,
        backup_dir=Path(backup_value),
        values=values,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the production deployment CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("deploy", "rollback"):
        command = subparsers.add_parser(name)
        command.add_argument("--env-file", type=Path, required=True)
        command.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE_FILE)
        command.add_argument("--proxy-file", type=Path, default=DEFAULT_PROXY_FILE)
        if name == "rollback":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--restore-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a production deploy or explicit artifact-backed rollback."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.operation == "deploy":
            manifest = deploy(config)
            manifest_path = _snapshot_directory(config.backup_dir, manifest.created_at) / "manifest.json"
            print(f"Production deployment completed. Manifest: {manifest_path}")
        else:
            if not args.restore_artifacts:
                raise DeploymentError("rollback requires --restore-artifacts")
            rollback(config, args.manifest)
            print("Production rollback completed with verified artifacts.")
    except (DeploymentError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("ERROR: deployment filesystem gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
