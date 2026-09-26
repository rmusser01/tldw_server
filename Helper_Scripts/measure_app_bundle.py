"""Measure CI candidate transfers and filesystems without image-size estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat

# Only fixed Docker operations with validated refs/IDs are used, never a shell.
import subprocess  # nosec B404
import time
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from urllib.parse import urlsplit

IMAGE = re.compile(
    r"(localhost|127\.0\.0\.1):([1-9][0-9]{0,4})/(tldw/(backend|webui|gateway|control))@sha256:([0-9a-f]{64})"
)


def download(url: str, digest: str, size: int | None, *, capture: bool = False) -> tuple[int, bytes]:
    """Count and hash real local-registry response bytes, with strict bounds."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or not parsed.port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("invalid job-local registry download")
    if size is not None and (type(size) is not int or not 0 < size <= 32 * 1024**3):
        raise ValueError("invalid download size")
    limit = size if size is not None else 2 * 1024**2
    started = time.monotonic()
    count, checksum, chunks = 0, hashlib.sha256(), []
    headers = {
        "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
        "Accept-Encoding": "identity",
    }
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    response = None
    try:
        connection.request("GET", parsed.path, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("registry response is not successful; redirects are refused")
        while chunk := response.read1(512 * 1024):
            count += len(chunk)
            if count > limit or time.monotonic() - started > 600 or (capture and count > 2 * 1024**2):
                raise ValueError("download exceeds measurement bound")
            checksum.update(chunk)
            if capture:
                chunks.append(chunk)
    finally:
        if response is not None:
            response.close()
        connection.close()
    if (size is not None and count != size) or checksum.hexdigest() != digest:
        raise ValueError("download length or digest differs")
    return count, b"".join(chunks)


def _docker(args: list[str]) -> str:
    """Execute bounded Docker operations; callers validate refs, volumes and IDs."""
    binary = shutil.which("docker")
    if binary is None:
        raise RuntimeError("Docker is required for measurements")
    # Callers supply fixed operations and regex-validated image/volume/container IDs.
    result = subprocess.run([binary, *args], check=True, capture_output=True, text=True, timeout=180)  # nosec B603
    return result.stdout.strip()


def docker_usage(image: str, volumes: dict[str, str] | None = None) -> dict[str, int]:
    """Run filesystem du in one owned, isolated, read-only measurement container."""
    match = IMAGE.fullmatch(image)
    if match is None or int(match[2]) > 65535:
        raise ValueError("invalid pinned measurement image")
    mounts: list[str] = []
    paths = ["/"]
    if volumes:
        if set(volumes) != {"/data", "/config"}:
            raise ValueError("invalid measurement mounts")
        for path, name in volumes.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+_backend_(data|config)", name):
                raise ValueError("invalid owned volume")
            project = name.rsplit("_backend_", 1)[0]
            label = _docker(["volume", "inspect", "--format", '{{index .Labels "com.docker.compose.project"}}', name])
            if label != project:
                raise ValueError("volume ownership differs")
            mounts.extend(["--mount", f"type=volume,src={name},dst={path},readonly"])
        paths = list(volumes)
    owned = _docker(
        [
            "create",
            "--label",
            "tldw.task=TASK-13343-measurement",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "DAC_OVERRIDE",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "0",
            *mounts,
            "--entrypoint",
            "du",
            image,
            "-s",
            "-x",
            "-B1",
            *paths,
        ]
    )
    if not re.fullmatch(r"[0-9a-f]{64}", owned):
        raise ValueError("invalid measurement container identity")
    try:
        usage = {}
        for line in _docker(["start", "-a", owned]).splitlines():
            entry = re.fullmatch(r"([0-9]+)\s+(/|/data|/config)", line)
            if entry is None or entry[2] in usage:
                raise ValueError("invalid filesystem measurement")
            usage[entry[2]] = int(entry[1])
        if set(usage) != set(paths):
            raise ValueError("filesystem measurement is incomplete")
        return usage
    finally:
        try:
            _docker(["rm", "-f", "-v", owned])
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"measurement cleanup failed; owned container {owned}") from exc


def measure_images(images: Path, platform: str, commit: str) -> dict[str, object]:
    """Measure each pinned image and deduplicate downloaded content by digest."""
    if platform not in {"linux/amd64", "linux/arm64"} or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid measurement source/platform")
    blobs: dict[str, int] = {}
    roles: dict[str, object] = {}
    manifest_total = 0
    for line in images.read_text().splitlines():
        role, image, _docker_metadata_size = line.split("\t")
        match = IMAGE.fullmatch(image)
        if match is None or match[4] != role or role in roles or int(match[2]) > 65535:
            raise ValueError("invalid image measurement inventory")
        base = f"http://{match[1]}:{match[2]}/v2/{match[3]}"
        manifest_bytes, raw = download(f"{base}/manifests/sha256:{match[5]}", match[5], None, capture=True)
        manifest = json.loads(raw)
        config = manifest["config"]
        config_hash = config["digest"].removeprefix("sha256:")
        count, config_raw = download(f"{base}/blobs/{config['digest']}", config_hash, config["size"], capture=True)
        metadata = json.loads(config_raw)
        if (
            metadata["architecture"] != platform.split("/")[1]
            or metadata["os"] != "linux"
            or metadata["config"]["Labels"]["org.opencontainers.image.revision"] != commit
        ):
            raise ValueError("downloaded image source/platform differs")
        blobs[config_hash] = count
        payload = count + manifest_bytes
        for layer in manifest["layers"]:
            digest = layer["digest"].removeprefix("sha256:")
            if digest not in blobs:
                blobs[digest], _body = download(f"{base}/blobs/{layer['digest']}", digest, layer["size"])
            if blobs[digest] != layer["size"]:
                raise ValueError("shared blob length differs")
            payload += blobs[digest]
        manifest_total += manifest_bytes
        roles[role] = {
            "image": image,
            "compressed_download_payload_bytes": payload,
            "rootfs_allocated_bytes": docker_usage(image)["/"],
        }
    if set(roles) != {"backend", "webui", "gateway", "control"}:
        raise ValueError("all four measured roles are required")
    return {
        "schema_version": 1,
        "source_commit": commit,
        "platform": platform,
        "method": "SHA256-verified registry response bytes; filesystem du -sx -B1 in read-only images",
        "download_unique_image_payload_bytes": manifest_total + sum(blobs.values()),
        "roles": roles,
        "storage_note": "Per-image merged-filesystem allocation; shared layers may reduce total engine storage. Build cache and Docker VM overhead are excluded.",
    }


def measure_state(image: str, project: str, state: Path) -> dict[str, object]:
    """Measure fresh owned persistent volumes and private helper-state allocation."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", project) or not state.is_dir():
        raise ValueError("invalid initialized measurement state")
    usage = docker_usage(image, {"/data": f"{project}_backend_data", "/config": f"{project}_backend_config"})
    allocated = 0
    logical = 0
    for path in [state, *state.rglob("*")]:
        metadata = path.lstat()
        allocated += metadata.st_blocks * 512
        if stat.S_ISREG(metadata.st_mode):
            logical += metadata.st_size
    return {
        "backend_data_allocated_bytes": usage["/data"],
        "backend_config_allocated_bytes": usage["/config"],
        "helper_state_allocated_bytes": allocated,
        "helper_state_logical_file_bytes": logical,
        "method": "read-only du of owned named volumes; lstat allocation of private helper state",
    }


def main() -> None:
    """Write a public measurement record, suppressing private diagnostic output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("images", "state"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--platform", choices=("linux/amd64", "linux/arm64"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control-image")
    parser.add_argument("--project-id")
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source commit must be a full 40-character Git commit SHA")
    if args.mode == "images" and args.images is None:
        parser.error("images mode requires --images")
    if args.mode == "state" and any(value is None for value in (args.control_image, args.project_id, args.state_dir)):
        parser.error("state mode requires --control-image, --project-id and --state-dir")
    try:
        if args.mode == "images":
            result = measure_images(args.images, args.platform, args.source_commit)
        else:
            result = measure_state(args.control_image, args.project_id, args.state_dir)
            result.update({"schema_version": 1, "source_commit": args.source_commit, "platform": args.platform})
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except RuntimeError as exc:
        parser.exit(1, str(exc) + "\n")
    except (OSError, ValueError, KeyError, TypeError, HTTPException, subprocess.SubprocessError):
        parser.exit(1, "Candidate measurement failed; no success record was emitted.\n")


if __name__ == "__main__":
    main()
