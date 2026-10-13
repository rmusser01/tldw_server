from __future__ import annotations

import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest


SCRIPT_SOURCE = Path("apps/tldw-frontend/scripts/copy-pdf-worker.mjs")
MONACO_SCRIPT_SOURCE = Path("apps/scripts/prepare-monaco-assets.mjs")
MONACO_WORKER_LABELS = ("editor", "json", "css", "html", "ts")
NODE_BIN = shutil.which("node")


def _write_worker(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _prepare_workspace(project_root: Path, *, hoisted: bool) -> Path:
    """Run the actual preparation helper against a minimal installed package layout."""
    script_path = project_root / "scripts" / "copy-pdf-worker.mjs"
    _write_worker(script_path, SCRIPT_SOURCE.read_text(encoding="utf-8"))
    _write_worker(
        project_root.parent / "scripts" / "prepare-monaco-assets.mjs",
        MONACO_SCRIPT_SOURCE.read_text(encoding="utf-8"),
    )
    node_modules = (project_root.parent if hoisted else project_root) / "node_modules"
    react = node_modules / "@monaco-editor" / "react"
    _write_worker(react / "package.json", '{"name":"@monaco-editor/react","main":"index.js"}')
    _write_worker(react / "index.js", "module.exports = {};\n")
    monaco = node_modules / "monaco-editor"
    _write_worker(monaco / "package.json", '{"name":"monaco-editor"}')
    _write_worker(monaco / "min" / "vs" / "loader.js", "monaco-loader-fixture\n")
    _write_worker(monaco / "LICENSE", "monaco-license-fixture\n")
    _write_worker(monaco / "ThirdPartyNotices.txt", "monaco-notices-fixture\n")
    for label in MONACO_WORKER_LABELS:
        _write_worker(
            monaco / "min" / "vs" / "assets" / f"{label}.worker-fixture.js",
            f"monaco-{label}-worker-fixture\n",
        )
    return script_path


def _assert_prepared_monaco_assets(project_root: Path) -> None:
    for target in (
        project_root / "public" / "monaco",
        project_root.parent / "packages" / "ui" / "src" / "public" / "monaco",
    ):
        assert (target / "vs" / "loader.js").read_text(encoding="utf-8") == "monaco-loader-fixture\n"
        assert (target / "LICENSE").read_text(encoding="utf-8") == "monaco-license-fixture\n"
        assert (target / "ThirdPartyNotices.txt").read_text(encoding="utf-8") == "monaco-notices-fixture\n"
        for label in MONACO_WORKER_LABELS:
            assert (target / "vs" / "assets" / f"{label}.worker-fixture.js").read_text(
                encoding="utf-8"
            ) == f"monaco-{label}-worker-fixture\n"
            assert (target / f"{label}-worker.js").read_text(encoding="utf-8") == (
                f"importScripts('./vs/assets/{label}.worker-fixture.js');\n"
                "globalThis.postMessage({type:'vscode-worker-ready'});\n"
            )


@pytest.mark.skipif(NODE_BIN is None, reason="node is required to execute copy-pdf-worker.mjs")
def test_copy_pdf_worker_script_supports_local_workspace_node_modules(tmp_path: Path):
    project_root = tmp_path / "apps" / "tldw-frontend"
    script_path = _prepare_workspace(project_root, hoisted=False)

    expected = "local-worker"
    local_worker = project_root / "node_modules" / "pdfjs-dist" / "build" / "pdf.worker.min.mjs"
    _write_worker(local_worker, expected)

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
    _assert_prepared_monaco_assets(project_root)


@pytest.mark.skipif(NODE_BIN is None, reason="node is required to execute copy-pdf-worker.mjs")
def test_copy_pdf_worker_script_supports_hoisted_workspace_node_modules(tmp_path: Path):
    project_root = tmp_path / "apps" / "tldw-frontend"
    script_path = _prepare_workspace(project_root, hoisted=True)

    expected = "hoisted-worker"
    hoisted_worker = tmp_path / "apps" / "node_modules" / "pdfjs-dist" / "build" / "pdf.worker.min.mjs"
    _write_worker(hoisted_worker, expected)

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
    _assert_prepared_monaco_assets(project_root)
