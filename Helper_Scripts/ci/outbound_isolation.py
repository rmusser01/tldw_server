"""Prepare and run outbound regression fixtures in disconnected Docker containers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import select
import shutil
import socket
import stat
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

PROJECT_DIR = "/opt/outbound-project"
CONTAINER_REPORT = "/outbound-report/result.json"
CONTAINER_PREFLIGHT_FAILURE = {"preflight": "failed"}
PREFLIGHT_ENV = "TLDW_OUTBOUND_PREFLIGHT"
_USER = "10001:10001"
_DOCKERFILE = "Dockerfiles/Dockerfile.outbound-check"
_SHIPPED_CONFIG = Path("tldw_Server_API/Config_Files/config.txt")
_TARGETS = ("backend", "browser")
_BACKEND_INPUTS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "LICENSES",
    "tldw_Server_API",
    "Helper_Scripts",
    "Docs/Published",
    "apps/mcp-unified/src",
    # Inventory validation checks the declaration source paths.
    "apps/extension/wxt.config.ts",
    "apps/extension/entrypoints/background.ts",
    "apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx",
    "packages/tldw_profile_core",
    _DOCKERFILE,
)
_UNTRACKED_INPUTS = (
    _DOCKERFILE,
    "Helper_Scripts/ci/outbound_*",
    "tldw_Server_API/tests/CI/test_outbound_*",
    "apps/tldw-frontend/playwright.outbound.config.ts",
    "apps/tldw-frontend/e2e/outbound-privacy.spec.ts",
    "apps/tldw-frontend/e2e/utils/outbound-observer.ts",
    "apps/extension/playwright.outbound.config.ts",
    "apps/extension/tests/e2e/outbound-privacy.spec.ts",
)
_EXCLUDED_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".env",
    ".key",
    ".lockfile",
    ".pem",
    ".pfx",
    ".p12",
    ".pyc",
    ".sock",
    ".sqlite",
    ".sqlite3",
}
_EXCLUDED_DIRS = {".git", ".cache", ".next", ".venv", ".wxt", "__pycache__", "node_modules"}
_REPORT_LIMIT = 256 * 1024
_ARCHIVE_LIMIT = _REPORT_LIMIT + 32 * 1024
_BUILD_LIMIT = 512 * 1024 * 1024
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")


class IsolationError(RuntimeError):
    """A container or preparation result failed an isolation requirement."""

    def __init__(self, message: str, *, code: str = "isolation_failed") -> None:
        super().__init__(message)
        self.code = code


def _project_files(root: Path, target: str) -> Iterator[Path]:
    """Yield only versioned inputs and named new harness files."""
    if target not in _TARGETS:
        raise IsolationError("unsupported outbound image target")
    pathspecs = (*_BACKEND_INPUTS, "apps") if target == "browser" else _BACKEND_INPUTS
    # This fixed command inventories the checkout without updating it.
    output = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "-z", "--", *pathspecs],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    names = {Path(os.fsdecode(path)) for path in output.split(b"\0") if path}
    for pattern in _UNTRACKED_INPUTS:
        names.update(path.relative_to(root) for path in root.glob(pattern))
    total = 0
    for name in sorted(names):
        path = root / name
        lower = name.name.lower()
        if (
            any(part in _EXCLUDED_DIRS for part in name.parts)
            or lower.startswith((".env", "session_encryption.key"))
            or path.suffix.lower() in _EXCLUDED_SUFFIXES
            or not path.is_file()
            or path.resolve() != path
        ):
            continue
        total += path.stat().st_size
        if total > _BUILD_LIMIT:
            raise IsolationError("outbound build context exceeds its size budget")
        yield name


def _shipped_config(root: Path) -> bytes | None:
    """Require the production defaults before copying anything into a build context."""
    config = root / _SHIPPED_CONFIG
    if not config.exists() and not config.is_symlink():
        return None
    message = "Outbound preparation needs committed defaults in tldw_Server_API/Config_Files/config.txt"
    if not config.is_file() or config.resolve() != config:
        raise IsolationError(message)
    committed = subprocess.run(  # nosec B603 B607
        ["git", "show", f"HEAD:{_SHIPPED_CONFIG.as_posix()}"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    with config.open("rb") as handle:
        local = handle.read(1024 * 1024 + 1)
    if committed.returncode != 0 or len(local) > 1024 * 1024 or local != committed.stdout:
        raise IsolationError(message)
    return committed.stdout


def prepare_project(root: Path, temp: Path, *, target: str = "backend") -> Path:
    """Copy current checkout inputs into a separate Docker build context."""
    root = root.resolve()
    defaults = _shipped_config(root)
    project = temp / "project"
    project.mkdir(parents=True)
    for name in _project_files(root, target):
        source = root / name
        dest = project / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name == _SHIPPED_CONFIG and defaults is not None:
            dest.write_bytes(defaults)
        else:
            shutil.copyfile(source, dest)
        dest.chmod(0o755 if source.stat().st_mode & stat.S_IXUSR else 0o644)
    return project


def _source_fingerprint(root: Path, files: Iterator[Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        path = root / name
        digest.update(os.fsencode(name.as_posix()) + b"\0")
        digest.update(b"x" if path.stat().st_mode & stat.S_IXUSR else b"-")
        with path.open("rb") as source:
            digest.update(hashlib.file_digest(source, "sha256").digest())
    return digest.hexdigest()


def _docker(args: Sequence[str], *, timeout: float, capture: bool = True) -> subprocess.CompletedProcess:
    """Keep isolated app output private while allowing dependency-build diagnostics."""
    return subprocess.run(  # nosec B603 B607
        ["docker", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.DEVNULL if capture else None,
        timeout=timeout,
        check=False,
    )


def _write_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(payload).encode("utf-8"))
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_image(root: Path, target: str, report_dir: Path) -> str:
    """Build one fixture image and write its source fingerprint on success."""
    if target not in _TARGETS:
        raise IsolationError("unsupported outbound image target")
    report_dir.mkdir(parents=True, exist_ok=True)
    record = report_dir / f"prepared-{target}.json"
    record.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="tldw-outbound-build-") as folder:
        temp = Path(folder)
        project = prepare_project(root, temp, target=target)
        files = (path.relative_to(project) for path in project.rglob("*") if path.is_file())
        fingerprint = _source_fingerprint(project, files)
        image_file = temp / "image-id"
        built = _docker(
            [
                "build",
                "--target",
                target,
                "--iidfile",
                str(image_file),
                "--file",
                str(project / _DOCKERFILE),
                str(project),
            ],
            timeout=1800,
            capture=False,
        )
        if built.returncode != 0 or not image_file.is_file():
            raise IsolationError(f"outbound {target} image build failed")
        image_id = image_file.read_text(encoding="ascii").strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise IsolationError("Docker returned an invalid outbound image ID")
    _write_json(record, {"image_id": image_id, "source_fingerprint": fingerprint})
    return image_id


def load_prepared_image(root: Path, target: str, report_dir: Path) -> str:
    """Reject an image prepared for any other source snapshot."""
    if target not in _TARGETS:
        raise IsolationError("unsupported outbound image target")
    record = report_dir / f"prepared-{target}.json"
    try:
        with record.open("rb") as handle:
            item = json.loads(handle.read(4096))
        image_id = item["image_id"]
        fingerprint = item["source_fingerprint"]
        if not _IMAGE_ID.fullmatch(image_id):
            raise ValueError("bad image ID")
    except FileNotFoundError as exc:
        raise IsolationError(
            f"missing outbound {target} preparation record", code="prepared_image_missing"
        ) from exc
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise IsolationError(
            f"invalid outbound {target} preparation record", code="prepared_image_invalid"
        ) from exc
    root = root.resolve()
    if fingerprint != _source_fingerprint(root, _project_files(root, target)):
        raise IsolationError(f"outbound {target} preparation record is stale", code="prepared_image_stale")
    return image_id


def _verify_container(data: bytes, image_id: str) -> None:
    try:
        (container,) = json.loads(data)
        config = container["Config"]
        host = container["HostConfig"]
        network = container["NetworkSettings"]
        isolated = (
            container["Image"] == image_id
            and config["User"] == _USER
            and not any(config.get(key) for key in ("OpenStdin", "Tty", "Volumes"))
            and host["NetworkMode"] == "none"
            and set(network["Networks"]) == {"none"}
            and not any((network.get("Ports") or {}).values())
            and host["CapDrop"] == ["ALL"]
            and bool({"no-new-privileges", "no-new-privileges:true"} & set(host["SecurityOpt"]))
            and not any(
                host.get(key)
                for key in (
                    "Privileged",
                    "AutoRemove",
                    "CapAdd",
                    "Binds",
                    "Mounts",
                    "VolumesFrom",
                    "PublishAllPorts",
                    "PortBindings",
                    "Devices",
                    "DeviceRequests",
                )
            )
            and host.get("IpcMode", "private") in ("private", "none")
            and host.get("PidMode", "") == ""
            and container["Mounts"] == []
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        isolated = False
    if not isolated:
        raise IsolationError("Docker container isolation verification failed")


@contextlib.contextmanager
def _host_socket() -> Iterator[tuple[str, socket.socket]]:
    """Create a host-only socket and calibrate it with a local control connection."""
    # Keep within the AF_UNIX path limit even when TMPDIR is deeply nested.
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="tldw-egress-") as directory:  # nosec B108
        path = str(Path(directory) / "probe.sock")
        with socket.socket(socket.AF_UNIX) as listener, socket.socket(socket.AF_UNIX) as client:
            listener.bind(path)
            listener.listen(1)
            client.settimeout(2)
            listener.settimeout(2)
            client.connect(path)
            peer, _ = listener.accept()
            peer.close()
            listener.setblocking(False)
            yield path, listener


def _active_interfaces(net_dir: Path = Path("/sys/class/net")) -> set[str]:
    # The kernel can create inactive tunnel devices in a disconnected namespace.
    return {
        name for _, name in socket.if_nameindex() if int((net_dir / name / "flags").read_text(encoding="ascii"), 16) & 1
    }


def _require_private_runtime() -> None:
    if sys.platform != "linux" or (os.getuid(), os.getgid()) != (10001, 10001):
        raise IsolationError("unexpected outbound container user")
    if _active_interfaces() != {"lo"}:
        raise IsolationError("outbound container has a non-loopback network interface")


def assert_isolated_runtime() -> None:
    """Prevent accidental direct execution of internal fixture commands."""
    if os.environ.get(PREFLIGHT_ENV) != "1":
        raise IsolationError("outbound container preflight is required")
    _require_private_runtime()


def _preflight(path: str) -> None:
    """Require unprivileged execution, no host socket and working local networking."""
    _require_private_runtime()
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        try:
            client.connect(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise IsolationError("host socket probe returned an unexpected result") from exc
        else:
            raise IsolationError("host socket is reachable from the outbound container")
    with socket.socket(socket.AF_INET) as listener, socket.socket(socket.AF_INET) as client:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2)
        client.settimeout(2)
        client.connect(listener.getsockname())
        peer, _ = listener.accept()
        with peer:
            peer.settimeout(2)
            peer.sendall(b"1")
            if client.recv(1) != b"1":
                raise IsolationError("outbound container loopback preflight failed")


def _report_bytes(data: bytes) -> bytes:
    """Extract only the bounded JSON file, without writing any tar member to disk."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            (entry,) = archive.getmembers()
            if entry.name != "result.json" or not entry.isfile() or not 0 < entry.size <= _REPORT_LIMIT:
                raise ValueError("unexpected archive entry")
            file = archive.extractfile(entry)
            if file is None:
                raise ValueError("report file is missing")
            body = file.read(_REPORT_LIMIT + 1)
        if len(body) != entry.size or not isinstance(json.loads(body), dict):
            raise ValueError("unexpected report payload")
    except (tarfile.TarError, ValueError, OSError) as exc:
        raise IsolationError("invalid outbound report archive or JSON", code="invalid_scenario_evidence") from exc
    return body


def _copy_report(container: str, target: Path, *, timeout: float) -> None:
    # Docker cp outputs an archive even for a single file. Bound it before decoding.
    args = ["docker", "cp", f"{container}:{CONTAINER_REPORT}", "-"]
    end = time.monotonic() + timeout
    with subprocess.Popen(  # nosec B603
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as child:
        if child.stdout is None:
            raise IsolationError("outbound report copy has no output pipe")
        data = bytearray()
        try:
            while len(data) <= _ARCHIVE_LIMIT:
                readable, _, _ = select.select([child.stdout], [], [], max(0, end - time.monotonic()))
                if not readable:
                    raise IsolationError("outbound report copy timed out", code="container_timeout")
                chunk = child.stdout.read1(min(65536, _ARCHIVE_LIMIT + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > _ARCHIVE_LIMIT:
                raise IsolationError("outbound report archive exceeds its size budget")
            if child.wait(timeout=max(0, end - time.monotonic())) != 0:
                raise IsolationError("missing outbound container report", code="container_report_missing")
        finally:
            if child.poll() is None:
                child.kill()
    target.write_bytes(_report_bytes(bytes(data)))


def run_isolated(image_id: str, argv: Sequence[str], report_dir: Path, *, timeout: float) -> int:
    """Run a Python fixture under verified Docker isolation and collect its final report."""
    if not _IMAGE_ID.fullmatch(image_id) or not argv or timeout <= 0:
        raise IsolationError("invalid outbound image ID, command or timeout")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "result.json"
    report.unlink(missing_ok=True)
    name = "tldw-outbound-" + uuid.uuid4().hex
    container = ""
    with _host_socket() as (probe_path, listener):
        try:
            created = _docker(
                [
                    "create",
                    "--name",
                    name,
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--user",
                    _USER,
                    "--init",
                    "--log-driver",
                    "none",
                    "--workdir",
                    PROJECT_DIR,
                    "--entrypoint",
                    "python",
                    image_id,
                    "-m",
                    "Helper_Scripts.ci.outbound_isolation",
                    "_inside",
                    probe_path,
                    "--",
                    *argv,
                ],
                timeout=30,
            )
            container = created.stdout.decode("ascii").strip()
            if created.returncode != 0 or not _CONTAINER_ID.fullmatch(container):
                raise IsolationError("could not create outbound container")
            inspected = _docker(["inspect", "--type", "container", container], timeout=15)
            _verify_container(inspected.stdout, image_id)
            if inspected.returncode != 0 or _docker(["start", container], timeout=30).returncode != 0:
                raise IsolationError("could not start outbound container")
            ended = _docker(["wait", container], timeout=timeout)
            if ended.returncode != 0 or not ended.stdout.strip().isdigit():
                raise IsolationError("could not determine outbound container exit status")
            try:
                incoming, _ = listener.accept()
            except BlockingIOError:
                pass
            else:
                incoming.close()
                raise IsolationError("outbound container reached the host socket")
            _copy_report(container, report, timeout=15)
            return int(ended.stdout)
        except subprocess.TimeoutExpired as exc:
            raise IsolationError("outbound container operation timed out", code="container_timeout") from exc
        finally:
            failed = sys.exc_info()[0] is not None
            try:
                removed = _docker(["rm", "-f", name], timeout=15)
                if removed.returncode != 0 and _CONTAINER_ID.fullmatch(container):
                    raise IsolationError("could not remove outbound container")
            except (IsolationError, OSError, subprocess.TimeoutExpired):
                if not failed:
                    raise


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare fixture images or perform the in-container preflight and exec."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 4 and args[0] == "_inside" and args[2] == "--":
        try:
            _preflight(args[1])
        except (IsolationError, OSError, ValueError):
            _write_json(Path(CONTAINER_REPORT), CONTAINER_PREFLIGHT_FAILURE)
            return 1
        os.environ[PREFLIGHT_ENV] = "1"
        os.execv(sys.executable, [sys.executable, *args[3:]])  # nosec B606
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("target", choices=(*_TARGETS, "all"))
    parser.add_argument("--report-dir", required=True, type=Path)
    parsed = parser.parse_args(args)
    root = Path(__file__).resolve().parents[2]
    for target in _TARGETS if parsed.target == "all" else (parsed.target,):
        prepare_image(root, target, parsed.report_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IsolationError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc) if isinstance(exc, IsolationError) else type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None
