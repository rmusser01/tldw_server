"""Bounded owned PDF children; cancellation drains before capacity is released."""

from __future__ import annotations

import base64
import json
import os
import subprocess  # nosec B404 # Only the fixed same-package PDF worker is launched.
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool, await_bounded_sync_call
from tldw_Server_API.app.core.Document_Workspace.models import (
    MAX_PDF_BYTES,
    MAX_WORK_BYTES,
    PDF_ERRORS,
    DocumentRenderRequest,
    PdfAnnotation,
)

_PDF_POOL = BoundedDaemonPool(2)
_PDF_REQUESTS = BoundedDaemonPool(2)


def pdf_operation():
    """Admit before reading source bytes and hold until owned work drains."""
    return _PDF_REQUESTS.lease(exhaustion_message="Document PDF processing is busy. Try again shortly.")


def _run_pdf(payload: dict[str, Any]) -> bytes:
    """Use one fixed child, its matching package root and a ten-second deadline."""
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_WORK_BYTES:
        raise ValueError(PDF_ERRORS["too_large"])
    package_root = str(Path(__file__).resolve().parents[4])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = package_root
    try:
        result = subprocess.run(  # nosec B603 # Fixed interpreter/module; no shell, paths or commands from input.
            [sys.executable, "-m", "tldw_Server_API.app.core.Document_Workspace.pdf_worker"],
            input=raw,
            capture_output=True,
            check=False,
            timeout=10,
            env=environment,
            cwd=package_root,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Document PDF processing timed out.") from exc
    if result.returncode != 0:
        code = result.stderr.decode("ascii", errors="replace").strip()
        raise ValueError(PDF_ERRORS.get(code, PDF_ERRORS["invalid_document"]))
    if not result.stdout.startswith(b"%PDF-") or not result.stdout.rstrip().endswith(b"%%EOF"):
        raise ValueError(PDF_ERRORS["invalid_document"])
    if len(result.stdout) > MAX_PDF_BYTES:
        raise ValueError(PDF_ERRORS["too_large"])
    return result.stdout


async def _process(payload: Callable[[], dict[str, Any]]) -> bytes:
    return await await_bounded_sync_call(
        lambda: _run_pdf(payload()),
        pool=_PDF_POOL,
        exhaustion_message="Document PDF processing is busy. Try again shortly.",
    )


async def render_document_pdf(draft: DocumentRenderRequest) -> bytes:
    """Render a bounded reviewed draft without persisting it."""
    return await _process(lambda: {"operation": "render", "draft": draft.model_dump()})


async def annotate_document_pdf(source: bytes, annotations: list[PdfAnnotation]) -> bytes:
    """Export canonical saved annotations into a separate PDF."""
    if len(source) > MAX_PDF_BYTES:
        raise ValueError(PDF_ERRORS["too_large"])
    return await _process(
        lambda: {
            "operation": "annotate",
            "pdf": base64.b64encode(source).decode("ascii"),
            "annotations": [item.model_dump() for item in annotations],
        }
    )
