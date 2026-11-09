from __future__ import annotations

from pathlib import Path
import shutil
import json
import subprocess  # nosec B404

import pytest


SCRIPT_SOURCE = Path("apps/tldw-frontend/scripts/copy-pdf-worker.mjs")
NODE_BIN = shutil.which("node")


def _write_worker(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _write_viewer_packages(node_modules: Path) -> None:
    """Resolve the worker through the same shared-viewer package chain."""
    for name in ("@tldw/ui", "react-pdf", "pdfjs-dist"):
        package = node_modules / name / "package.json"
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_text(json.dumps({"name": name, "version": "1.0.0"}), encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


@pytest.mark.skipif(NODE_BIN is None, reason="node is required to execute copy-pdf-worker.mjs")
def test_copy_pdf_worker_script_supports_local_workspace_node_modules(tmp_path: Path):
    project_root = tmp_path / "apps" / "tldw-frontend"
    script_path = project_root / "scripts" / "copy-pdf-worker.mjs"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(SCRIPT_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    expected = "local-worker"
    local_worker = project_root / "node_modules" / "pdfjs-dist" / "build" / "pdf.worker.min.mjs"
    _write_worker(local_worker, expected)
    _write_viewer_packages(project_root / "node_modules")

    # Invokes local Node script with test-controlled args.
    result = subprocess.run(
        [NODE_BIN, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )  # nosec B603

    _require(result.returncode == 0, result.stderr)
    copied = project_root / "public" / "pdf.worker.min.mjs"
    _require(copied.exists(), f"Expected copied worker at: {copied}")
    _require(copied.read_text(encoding="utf-8") == expected, "Copied worker content mismatch")


@pytest.mark.skipif(NODE_BIN is None, reason="node is required to execute copy-pdf-worker.mjs")
def test_copy_pdf_worker_script_supports_hoisted_workspace_node_modules(tmp_path: Path):
    project_root = tmp_path / "apps" / "tldw-frontend"
    script_path = project_root / "scripts" / "copy-pdf-worker.mjs"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(SCRIPT_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    expected = "hoisted-worker"
    hoisted_worker = tmp_path / "apps" / "node_modules" / "pdfjs-dist" / "build" / "pdf.worker.min.mjs"
    _write_worker(hoisted_worker, expected)
    _write_viewer_packages(tmp_path / "apps" / "node_modules")

    # Invokes local Node script with test-controlled args.
    result = subprocess.run(
        [NODE_BIN, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )  # nosec B603

    _require(result.returncode == 0, result.stderr)
    copied = project_root / "public" / "pdf.worker.min.mjs"
    _require(copied.exists(), f"Expected copied worker at: {copied}")
    _require(copied.read_text(encoding="utf-8") == expected, "Copied worker content mismatch")
