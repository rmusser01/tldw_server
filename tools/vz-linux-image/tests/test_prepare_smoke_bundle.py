from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = IMAGE_DIR / "scripts" / "prepare-smoke-bundle.py"


def _create_bundle(root: Path) -> Path:
    bundle = root / "source-bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "initrd").write_bytes(b"initrd")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": "1",
                "boot_mode": "bundle",
                "kernel": "kernel",
                "rootfs": "rootfs.img",
                "initrd": "initrd",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "build-info.json").write_text(
        json.dumps({"suite": "bookworm", "architecture": "arm64"}),
        encoding="utf-8",
    )
    return bundle


def _run_prepare(
    *,
    source_bundle: Path,
    store_root: Path,
    run_id: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--source-bundle",
            str(source_bundle),
            "--store-root",
            str(store_root),
            "--run-id",
            run_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_prepare_smoke_bundle_materializes_image_store_run_bundle(tmp_path: Path) -> None:
    source_bundle = _create_bundle(tmp_path)
    store_root = tmp_path / "image-store"

    result = _run_prepare(
        source_bundle=source_bundle,
        store_root=store_root,
        run_id="smoke-run",
    )

    assert result.returncode == 0, result.stderr
    run_bundle = Path(result.stdout.strip())
    assert run_bundle == store_root / "runs" / "smoke-run" / "bundle"
    assert (run_bundle / "kernel").read_bytes() == b"kernel"
    assert (run_bundle / "rootfs.img").read_bytes() == b"rootfs"
    assert (run_bundle / "initrd").read_bytes() == b"initrd"
    assert json.loads((run_bundle / "manifest.json").read_text(encoding="utf-8"))["rootfs"] == "rootfs.img"
    assert json.loads((run_bundle / "build-info.json").read_text(encoding="utf-8"))["suite"] == "bookworm"
    assert (store_root / "templates" / "vz_linux" / "host-smoke-source" / "manifest.json").is_file()
    assert (store_root / "runs" / "smoke-run" / "manifest.json").is_file()
    assert (store_root / "runs" / "smoke-run" / "bundle" / "manifest.json").is_file()


def test_prepare_smoke_bundle_does_not_mutate_source_rootfs(tmp_path: Path) -> None:
    source_bundle = _create_bundle(tmp_path)
    source_rootfs = source_bundle / "rootfs.img"
    source_before = source_rootfs.read_bytes()
    source_mtime_before = source_rootfs.stat().st_mtime_ns
    store_root = tmp_path / "image-store"

    result = _run_prepare(
        source_bundle=source_bundle,
        store_root=store_root,
        run_id="smoke-run",
    )

    assert result.returncode == 0, result.stderr
    run_rootfs = Path(result.stdout.strip()) / "rootfs.img"
    run_rootfs.write_bytes(b"mutated run rootfs")
    assert source_rootfs.read_bytes() == source_before
    assert source_rootfs.stat().st_mtime_ns == source_mtime_before


def test_prepare_smoke_bundle_rejects_missing_source_artifact(tmp_path: Path) -> None:
    source_bundle = _create_bundle(tmp_path)
    (source_bundle / "rootfs.img").unlink()

    result = _run_prepare(
        source_bundle=source_bundle,
        store_root=tmp_path / "image-store",
        run_id="smoke-run",
    )

    assert result.returncode != 0
    assert "bundle missing rootfs.img" in result.stderr
