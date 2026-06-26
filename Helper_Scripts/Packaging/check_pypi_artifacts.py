"""Validate that root PyPI artifacts stay backend/API-only."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from loguru import logger

BLOCKED_COMPONENT_NAMES = {
    ".next",
    "node_modules",
}

BLOCKED_COMPONENT_SEQUENCES = (
    ("apps", "tldw-frontend"),
)

BLOCKED_FILE_NAMES = {
    "bun.lock",
    "package-lock.json",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}

REQUIRED_PACKAGE_ROOTS = (
    "tldw_Server_API",
    "mcp_unified",
)


def _normalize_name(name: str) -> str:
    """Return a safe, POSIX-style archive path for policy matching."""
    normalized = str(PurePosixPath(name.replace("\\", "/")))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _strip_sdist_prefix(paths: list[str]) -> list[str]:
    """Remove the top-level sdist directory when every path shares one."""
    if not paths:
        return paths

    first_parts = [PurePosixPath(path).parts for path in paths]
    first_component = first_parts[0][0] if first_parts[0] else ""
    if not first_component:
        return paths

    if all(parts and parts[0] == first_component for parts in first_parts):
        return [
            str(PurePosixPath(*parts[1:]))
            for parts in first_parts
            if len(parts) > 1
        ]
    return paths


def _wheel_paths(path: Path) -> list[str]:
    """Return normalized file member paths from a built wheel."""
    with zipfile.ZipFile(path) as archive:
        return [
            _normalize_name(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
        ]


def _sdist_paths(path: Path) -> list[str]:
    """Return normalized file member paths from a source distribution."""
    with tarfile.open(path, mode="r:gz") as archive:
        paths = [
            _normalize_name(member.name)
            for member in archive.getmembers()
            if member.isfile()
        ]
    return _strip_sdist_prefix(paths)


def _archive_paths(path: Path) -> list[str]:
    """Return normalized file paths for a supported distribution artifact."""
    if path.suffix == ".whl":
        return _wheel_paths(path)
    if path.name.endswith(".tar.gz"):
        return _sdist_paths(path)
    raise ValueError(f"Unsupported distribution artifact: {path}")


def _contains_component_sequence(
    parts: tuple[str, ...], sequence: tuple[str, ...]
) -> bool:
    """Return whether a path contains an exact adjacent component sequence."""
    sequence_len = len(sequence)
    if sequence_len == 0 or len(parts) < sequence_len:
        return False
    return any(
        parts[index : index + sequence_len] == sequence
        for index in range(0, len(parts) - sequence_len + 1)
    )


def _blocked_paths(paths: list[str]) -> list[str]:
    """Return archive paths that violate frontend/Node artifact policy."""
    blocked: list[str] = []
    for path in paths:
        parts = PurePosixPath(path).parts
        if any(part in BLOCKED_COMPONENT_NAMES for part in parts):
            blocked.append(path)
            continue
        if any(
            _contains_component_sequence(parts, sequence)
            for sequence in BLOCKED_COMPONENT_SEQUENCES
        ):
            blocked.append(path)
            continue
        if parts and parts[-1] in BLOCKED_FILE_NAMES:
            blocked.append(path)
    return blocked


def _missing_required_roots(paths: list[str]) -> list[str]:
    """Return required package roots absent from archive path components."""
    return [
        root
        for root in REQUIRED_PACKAGE_ROOTS
        if not any(
            root in PurePosixPath(path).parts
            for path in paths
        )
    ]


def _validate_artifact(path: Path) -> list[str]:
    """Return validation errors for one built distribution artifact."""
    errors: list[str] = []
    archive_paths = _archive_paths(path)

    blocked = _blocked_paths(archive_paths)
    if blocked:
        examples = ", ".join(blocked[:5])
        errors.append(f"{path.name}: blocked frontend/Node paths found: {examples}")

    missing_roots = _missing_required_roots(archive_paths)
    if missing_roots:
        missing = ", ".join(missing_roots)
        errors.append(f"{path.name}: missing expected backend package roots: {missing}")

    return errors


def _distribution_paths(dist_dir: Path) -> list[Path]:
    """Return supported distribution artifacts in deterministic order."""
    return sorted(
        path
        for path in dist_dir.iterdir()
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    )


def main(argv: list[str] | None = None) -> int:
    """Validate built PyPI artifacts and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Validate tldw-server PyPI artifacts stay backend/API-only."
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Directory containing built wheel and sdist artifacts.",
    )
    args = parser.parse_args(argv)

    if not args.dist_dir.is_dir():
        logger.error("dist directory not found: {}", args.dist_dir)
        return 1

    artifacts = _distribution_paths(args.dist_dir)
    if not artifacts:
        logger.error("no wheel or sdist artifacts found in {}", args.dist_dir)
        return 1

    errors: list[str] = []
    for artifact in artifacts:
        errors.extend(_validate_artifact(artifact))

    if errors:
        for error in errors:
            logger.error(error)
        return 1

    artifact_names = ", ".join(path.name for path in artifacts)
    logger.info("backend/API-only artifact check passed: {}", artifact_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
