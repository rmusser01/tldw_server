"""Verify a signed paired Docker bundle and initialize one private instance."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tldw_Server_API.app.core.Release.manifest import (
    ManifestError,
    ReleaseManifest,
    verify_artifact,
    verify_manifest,
)
from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
    format_api_key,
    generate_api_key_id,
    generate_api_key_secret,
    parse_api_key,
)


CONTROL_VERSION = (0, 1, 0)
REQUIRED_IMAGE_ROLES = ("backend", "webui", "gateway")
_SAFE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
_IMAGE_REF = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
_CONFIG_KEYS = (
    "TLDW_PROJECT_ID",
    "TLDW_PUBLIC_PORT",
    "SINGLE_USER_API_KEY",
    "TLDW_GATEWAY_HOP_SECRET",
    "SINGLE_USER_SESSION_COOKIE_NAME",
    "CSRF_COOKIE_NAME",
    "TLDW_BACKEND_IMAGE",
    "TLDW_WEBUI_IMAGE",
    "TLDW_GATEWAY_IMAGE",
)


class BundleControlError(ValueError):
    """The signed bundle or persistent instance cannot be safely used."""


@dataclass(frozen=True)
class VerifiedBundle:
    """One signed, platform-selected release with checked local files."""

    manifest: ReleaseManifest
    manifest_sha256: str
    platform: str
    images: Mapping[str, str]


@dataclass(frozen=True)
class InstanceConfig:
    """Persistent identity and credentials for a single Compose instance."""

    project_id: str
    public_port: int
    api_key: str
    gateway_hop_secret: str
    session_cookie_name: str
    csrf_cookie_name: str
    images: Mapping[str, str]


def _version_tuple(value: str) -> tuple[int, int, int]:
    """Extract the numeric compatibility portion of a validated version."""
    return tuple(int(part) for part in value.split("-", 1)[0].split("."))  # type: ignore[return-value]


def verify_bundle(
    manifest_path: Path,
    signature_path: Path,
    trusted_keys: Mapping[str, bytes],
    *,
    platform: str,
    bundle_root: Path,
) -> VerifiedBundle:
    """Authenticate release metadata, selected images, and every local file."""
    try:
        if manifest_path.is_symlink() or signature_path.is_symlink() or bundle_root.is_symlink():
            raise BundleControlError("bundle input is unsafe")
        raw = manifest_path.read_bytes()
        signature = signature_path.read_bytes()
        manifest = verify_manifest(raw, signature, trusted_keys, platform=platform)
        compatibility = manifest.compatibility
        if _version_tuple(compatibility["min_launcher"]) > CONTROL_VERSION:
            raise BundleControlError("bundle requires a newer control version")
        if _version_tuple(compatibility["python_version"])[:2] != (3, 12):
            raise BundleControlError("bundle requires an unsupported Python runtime")
        if _version_tuple(compatibility["node_version"])[0] != 24:
            raise BundleControlError("bundle requires an unsupported Node runtime")
        selected = [artifact for artifact in manifest.artifacts if artifact.platform == platform]
        images: dict[str, str] = {}
        for artifact in selected:
            if artifact.kind == "file":
                if artifact.path is None:
                    raise BundleControlError("file artifact lacks a path")
                current = bundle_root
                for part in Path(artifact.path).parts[:-1]:
                    current = current / part
                    if current.is_symlink():
                        raise BundleControlError("artifact path traverses a symlink")
                verify_artifact(bundle_root / artifact.path, artifact)
            elif artifact.role in REQUIRED_IMAGE_ROLES:
                if artifact.role in images:
                    raise BundleControlError(f"duplicate {artifact.role} image for platform")
                if not _IMAGE_REF.fullmatch(artifact.location):
                    raise BundleControlError("image location contains unsafe characters")
                images[artifact.role] = artifact.location
        if set(images) != set(REQUIRED_IMAGE_ROLES):
            raise BundleControlError("bundle lacks a required backend, WebUI, or gateway image")
        return VerifiedBundle(manifest, hashlib.sha256(raw).hexdigest(), platform, images)
    except (OSError, ManifestError) as exc:
        raise BundleControlError(str(exc)) from exc


def _new_config(release: VerifiedBundle, public_port: int) -> InstanceConfig:
    """Generate per-instance names and secrets after bundle verification."""
    if not 1 <= public_port <= 65535:
        raise BundleControlError("public port is outside 1..65535")
    instance = secrets.token_hex(8)
    return InstanceConfig(
        project_id=f"tldw_{instance}",
        public_port=public_port,
        api_key=format_api_key(generate_api_key_id(), generate_api_key_secret()),
        gateway_hop_secret=secrets.token_urlsafe(48),
        session_cookie_name=f"tldw_session_{instance}",
        csrf_cookie_name=f"tldw_csrf_{instance}",
        images=dict(release.images),
    )


def _config_text(config: InstanceConfig) -> str:
    values = {
        "TLDW_PROJECT_ID": config.project_id,
        "TLDW_PUBLIC_PORT": str(config.public_port),
        "SINGLE_USER_API_KEY": config.api_key,
        "TLDW_GATEWAY_HOP_SECRET": config.gateway_hop_secret,
        "SINGLE_USER_SESSION_COOKIE_NAME": config.session_cookie_name,
        "CSRF_COOKIE_NAME": config.csrf_cookie_name,
        "TLDW_BACKEND_IMAGE": config.images["backend"],
        "TLDW_WEBUI_IMAGE": config.images["webui"],
        "TLDW_GATEWAY_IMAGE": config.images["gateway"],
    }
    return "".join(f"{key}={values[key]}\n" for key in _CONFIG_KEYS)


def _private_file(path: Path) -> bytes:
    """Read a regular, owned, mode-0600 file without following a final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise BundleControlError("existing instance file has unsafe ownership or permissions")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                content = stream.read(64 * 1024 + 1)
                if len(content) > 64 * 1024:
                    raise BundleControlError("existing instance file is oversized")
                return content
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise BundleControlError("existing instance file is missing or unsafe") from exc


def _existing(state: Path, release: VerifiedBundle, public_port: int | None) -> InstanceConfig:
    """Reuse only a complete instance bound to the exact signed release."""
    info = state.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise BundleControlError("existing instance directory is unsafe")
    try:
        identity = json.loads(_private_file(state / "identity.json"))
        expected = {
            "schema_version": 1,
            "manifest_sha256": release.manifest_sha256,
            "platform": release.platform,
            "version": release.manifest.version,
            "source_commit": release.manifest.source_commit,
        }
        if identity != expected:
            raise BundleControlError("existing instance belongs to a different release")
        lines = _private_file(state / "config.env").decode("utf-8").splitlines()
        entries = [line.split("=", 1) for line in lines]
        if len(entries) != len(_CONFIG_KEYS) or [key for key, _ in entries] != list(_CONFIG_KEYS):
            raise BundleControlError("existing instance config is incomplete")
        values = dict(entries)
        config = InstanceConfig(
            project_id=values["TLDW_PROJECT_ID"],
            public_port=int(values["TLDW_PUBLIC_PORT"]),
            api_key=values["SINGLE_USER_API_KEY"],
            gateway_hop_secret=values["TLDW_GATEWAY_HOP_SECRET"],
            session_cookie_name=values["SINGLE_USER_SESSION_COOKIE_NAME"],
            csrf_cookie_name=values["CSRF_COOKIE_NAME"],
            images={role: values[f"TLDW_{role.upper()}_IMAGE"] for role in REQUIRED_IMAGE_ROLES},
        )
        if (
            not _SAFE_NAME.fullmatch(config.project_id)
            or not _SAFE_NAME.fullmatch(config.session_cookie_name)
            or not _SAFE_NAME.fullmatch(config.csrf_cookie_name)
            or parse_api_key(config.api_key) is None
            or len(config.gateway_hop_secret) < 32
            or not 1 <= config.public_port <= 65535
            or config.images != release.images
            or (public_port is not None and config.public_port != public_port)
            or _config_text(config).encode() != _private_file(state / "config.env")
        ):
            raise BundleControlError("existing instance config conflicts with signed release")
        return config
    except (UnicodeError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, BundleControlError):
            raise
        raise BundleControlError("existing instance metadata is invalid") from exc


def initialize_bundle(
    state: Path,
    release: VerifiedBundle,
    *,
    public_port: int | None = None,
) -> InstanceConfig:
    """Create state atomically once, or validate and reuse its exact credentials."""
    state = Path(state)
    if state.is_symlink():
        raise BundleControlError("existing instance directory is unsafe")
    state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = state.parent / f".{state.name}.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        lock_info = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.getuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            raise BundleControlError("instance lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if state.is_symlink():
            raise BundleControlError("existing instance directory is unsafe")
        if state.exists():
            return _existing(state, release, public_port)
        if public_port is None:
            public_port = 8080
        config = _new_config(release, public_port)
        temporary = Path(tempfile.mkdtemp(prefix=f".{state.name}-", dir=state.parent))
        try:
            os.chmod(temporary, 0o700)
            identity = {
                "schema_version": 1,
                "manifest_sha256": release.manifest_sha256,
                "platform": release.platform,
                "version": release.manifest.version,
                "source_commit": release.manifest.source_commit,
            }
            for name, data in (
                ("config.env", _config_text(config).encode()),
                ("identity.json", (json.dumps(identity, sort_keys=True) + "\n").encode()),
            ):
                fd = os.open(temporary / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(temporary, state)
            return config
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
    finally:
        os.close(lock_fd)


def _trusted_keys(directory: Path) -> dict[str, bytes]:
    """Load only raw Ed25519 public keys embedded in the pinned control image."""
    keys: dict[str, bytes] = {}
    if directory.is_symlink() or not directory.is_dir():
        raise BundleControlError("trusted key directory is unavailable")
    for path in directory.glob("*.pub"):
        if not _SAFE_NAME.fullmatch(path.stem) or path.is_symlink():
            raise BundleControlError("trusted key path is unsafe")
        key = path.read_bytes()
        if len(key) != 32:
            raise BundleControlError("trusted public key has the wrong length")
        keys[path.stem] = key
    if not keys:
        raise BundleControlError("control image contains no trusted public keys")
    return keys


def main(argv: list[str] | None = None) -> int:
    """Run `verify` or `init` without printing credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "init"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux/amd64", "linux/arm64"), required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path("/bundle"))
    parser.add_argument("--trusted-keys", type=Path, default=Path("/opt/tldw/trusted-keys"))
    parser.add_argument("--expected-signer")
    parser.add_argument("--public-port", type=int)
    args = parser.parse_args(argv)
    try:
        verified = verify_bundle(
            args.manifest,
            args.signature,
            _trusted_keys(args.trusted_keys),
            platform=args.platform,
            bundle_root=args.bundle_root,
        )
        if args.expected_signer is not None and verified.manifest.signer_id != args.expected_signer:
            raise BundleControlError("manifest signer differs from pinned helper key identity")
        if args.command == "init":
            initialize_bundle(args.state, verified, public_port=args.public_port)
        elif args.state.exists() or args.state.is_symlink():
            _existing(args.state, verified, args.public_port)
    except BundleControlError as exc:
        print(f"bundle control: {exc}", file=sys.stderr)
        return 1
    print(f"{args.command} complete for {verified.manifest.version} ({args.platform})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
