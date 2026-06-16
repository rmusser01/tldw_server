#!/usr/bin/env python3
"""Prepare a disposable image-store-backed bundle for VZ host smoke runs."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tldw_Server_API.app.core.Sandbox.image_store import (  # noqa: E402
    CloneItem,
    ImageStoreError,
    SandboxImageStore,
)


DEFAULT_TEMPLATE_NAME = "host-smoke-source"
TARGET_SUBDIR = "bundle"
METADATA_FILES = ("manifest.json", "build-info.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a disposable VZ Linux smoke bundle from a canonical source bundle.",
    )
    parser.add_argument("--source-bundle", required=True, help="Canonical source bundle directory.")
    parser.add_argument("--store-root", required=True, help="Private image-store root for this smoke run.")
    parser.add_argument("--run-id", required=True, help="Image-store run id for the disposable bundle.")
    parser.add_argument(
        "--template-name",
        default=DEFAULT_TEMPLATE_NAME,
        help=f"Image-store template name to use for the source bundle. Defaults to {DEFAULT_TEMPLATE_NAME}.",
    )
    parser.add_argument(
        "--print-path-only",
        action="store_true",
        help="Validate inputs and print the resolved disposable bundle path without writing image-store state.",
    )
    return parser.parse_args(argv)


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"json invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"json expected object: {path}")
    return payload


def bundle_artifact_names(source_bundle: Path) -> list[str]:
    bundle_manifest = read_optional_json(source_bundle / "manifest.json")
    kernel = str(bundle_manifest.get("kernel", "kernel"))
    rootfs = str(bundle_manifest.get("rootfs", "rootfs.img"))
    initrd = bundle_manifest.get("initrd")
    names = [kernel, rootfs]
    if initrd is not None:
        names.append(str(initrd))
    elif (source_bundle / "initrd").is_file():
        names.append("initrd")
    return names


def validate_bundle(source_bundle: Path) -> None:
    if not source_bundle.is_dir():
        raise RuntimeError(f"bundle directory does not exist: {source_bundle}")
    resolved_source = source_bundle.resolve()
    for artifact_name in bundle_artifact_names(source_bundle):
        artifact_fragment = Path(artifact_name)
        if artifact_fragment.is_absolute() or artifact_fragment.name != artifact_name:
            raise RuntimeError(f"bundle artifact path invalid: {artifact_name}")
        artifact_path = (resolved_source / artifact_fragment).resolve()
        if not artifact_path.is_relative_to(resolved_source):
            raise RuntimeError(f"bundle artifact path escapes bundle: {artifact_name}")
        if not artifact_path.is_file():
            raise RuntimeError(f"bundle missing {artifact_name}: {artifact_path}")


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except PermissionError:
        raise RuntimeError(f"directory permissions could not be set: {path}") from None


def clonefile(source: Path, target: Path) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        clonefile_func = ctypes.CDLL(None, use_errno=True).clonefile
    except (AttributeError, OSError):
        return False
    clonefile_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clonefile_func.restype = ctypes.c_int
    result = clonefile_func(os.fsencode(source), os.fsencode(target), 0)
    if result == 0:
        return True
    return False


def copy_file(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir():
            raise RuntimeError(f"clone target is a directory: {target}")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if clonefile(source, target):
        shutil.copystat(source, target)
        return
    shutil.copy2(source, target)


def materialize_clone_items(clone_items: list[CloneItem]) -> None:
    for item in clone_items:
        if item.mode != "clone":
            raise RuntimeError(f"unsupported clone item mode: {item.mode}")
        source = Path(item.source_path)
        target = Path(item.target_path)
        if not source.is_file():
            raise RuntimeError(f"clone source missing: {source}")
        copy_file(source, target)


def copy_bundle_metadata(source_bundle: Path, run_bundle: Path) -> None:
    for metadata_name in METADATA_FILES:
        source = source_bundle / metadata_name
        if source.is_file():
            copy_file(source, run_bundle / metadata_name)


def prepare_bundle(args: argparse.Namespace) -> Path:
    source_bundle = Path(args.source_bundle).expanduser().resolve()
    store_root = Path(args.store_root).expanduser().resolve()
    validate_bundle(source_bundle)
    ensure_private_directory(store_root)

    store = SandboxImageStore(root_path=store_root)
    template_id = store.register_bundle(
        runtime="vz_linux",
        template_name=args.template_name,
        bundle_path=source_bundle,
        labels={"purpose": "host-e2e-smoke"},
        allow_existing=True,
    )
    manifest = store.prepare_run_clone(
        template_id=template_id,
        run_id=args.run_id,
        target_subdir=TARGET_SUBDIR,
    )
    run_bundle = store_root / "runs" / manifest.run_id / TARGET_SUBDIR
    ensure_private_directory(run_bundle.parent)
    ensure_private_directory(run_bundle)
    materialize_clone_items(manifest.clone_items)
    copy_bundle_metadata(source_bundle, run_bundle)
    return run_bundle


def resolve_run_bundle_path(args: argparse.Namespace) -> Path:
    source_bundle = Path(args.source_bundle).expanduser().resolve()
    store_root = Path(args.store_root).expanduser().resolve()
    validate_bundle(source_bundle)
    store = SandboxImageStore(root_path=store_root, create_root=False)
    normalized_run_id = store._normalize_manifest_segment(args.run_id, "run_id")
    return store_root / "runs" / normalized_run_id / TARGET_SUBDIR


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_bundle = resolve_run_bundle_path(args) if args.print_path_only else prepare_bundle(args)
    except (ImageStoreError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(run_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
