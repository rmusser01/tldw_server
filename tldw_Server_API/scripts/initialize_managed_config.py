"""Seed missing configuration assets in a paired app's persistent config volume."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from loguru import logger


def initialize_managed_config(destination: Path, *, source: Path | None = None) -> None:
    """Copy packaged defaults without replacing existing settings or runtime secrets.

    Publish each complete file with a same-directory hard link, so interruption
    cannot leave a partial default and a concurrent writer cannot be overwritten.
    Existing symlinks and file/directory conflicts fail startup instead of directing
    configuration writes outside the persistent volume.
    """
    source = source or Path(__file__).resolve().parents[1] / "Config_Files"
    if not (source / "config.txt").is_file():
        raise ValueError("Packaged configuration is missing config.txt")
    if destination.is_symlink():
        raise ValueError("Persistent configuration root must not be a symlink")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)

    for default in sorted(source.rglob("*")):
        relative = default.relative_to(source)
        if (
            default.is_symlink()
            or "__pycache__" in relative.parts
            or default.name.lower() == ".env"
            or default.suffix.lower() in {".key", ".lock", ".bak", ".py", ".pyc"}
        ):
            continue
        target = destination / relative
        if target.is_symlink():
            raise ValueError(f"Persistent configuration asset is a symlink: {relative}")
        if default.is_dir():
            target.mkdir(mode=0o700, exist_ok=True)
            continue
        if not default.is_file():
            continue
        if target.exists():
            if not target.is_file():
                raise ValueError(f"Persistent configuration asset must be a file: {relative}")
            continue
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".config-default-") as staged:
            with default.open("rb") as packaged:
                shutil.copyfileobj(packaged, staged)
            staged.flush()
            os.fsync(staged.fileno())
            try:
                os.link(staged.name, target)
            except FileExistsError:
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"Persistent configuration asset must be a file: {relative}") from None


def main() -> int:
    """Initialize the explicit managed config directory, failing closed on errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        initialize_managed_config(args.destination)
    except ValueError as exc:
        logger.error("Persistent configuration initialization failed: {}", exc)
        return 1
    except OSError as exc:
        asset = Path(exc.filename).name if exc.filename else "config volume"
        logger.error(
            "Persistent configuration initialization failed: filesystem operation on {} (errno {})", asset, exc.errno
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
