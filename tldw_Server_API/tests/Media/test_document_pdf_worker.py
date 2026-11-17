"""Actual isolated PDF rendering: no mocked document engine."""

from __future__ import annotations

import base64
import json
import subprocess  # nosec B404 # Tests execute only the fixed owned PDF worker module.
import sys

import pytest

WORKER = "tldw_Server_API.app.core.Document_Workspace.pdf_worker"
DRAFT = {
    "title": "Rowan Observatory",
    "pages": [
        {"heading": "The telescope", "text": "The telescope is named Selene. <script> is plain text."},
        {"heading": "Visiting", "text": "Tours start on Friday at six in the evening."},
    ],
}


def run_worker(payload):
    return subprocess.run(
        [sys.executable, "-m", WORKER], input=json.dumps(payload).encode(), capture_output=True, timeout=15, check=False
    )  # nosec B603 # Fixed interpreter/module; payload travels through stdin only.


def read_pdf(data):
    try:
        import pymupdf as fitz
    except ModuleNotFoundError as exc:
        if exc.name != "pymupdf":
            raise
        import fitz
    return fitz.open(stream=data, filetype="pdf")


def test_render_preserves_two_plain_text_pages():
    result = run_worker({"operation": "render", "draft": DRAFT})
    assert result.returncode == 0, result.stderr.decode(errors="replace")[:300]  # nosec B101 # Real child must produce a successful PDF.
    with read_pdf(result.stdout) as pdf:
        text = [page.get_text() for page in pdf]
        assert len(text) == 2 and all(
            page["heading"] in value and page["text"] in value for page, value in zip(DRAFT["pages"], text)
        )  # nosec B101 # Every original page and literal escaped text remains readable.


def test_annotated_export_preserves_pages_and_exact_selected_text_and_note():
    source = run_worker({"operation": "render", "draft": DRAFT})
    assert source.returncode == 0  # nosec B101 # Build a genuine source before export.
    result = run_worker(
        {
            "operation": "annotate",
            "pdf": base64.b64encode(source.stdout).decode(),
            "annotations": [
                {
                    "page": 2,
                    "text": "Friday at six",
                    "note": "Ask about the Selene tour.",
                    "color": "blue",
                    "kind": "highlight",
                }
            ],
        }
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[:300]  # nosec B101 # Export returns a complete document, not a sidecar.
    with read_pdf(source.stdout) as before, read_pdf(result.stdout) as after:
        comments = list(after[1].annots())
        assert (
            [page.get_text() for page in after] == [page.get_text() for page in before]
            and len(comments) == 1
            and "Friday at six" in comments[0].info["content"]
            and "Ask about the Selene tour." in comments[0].info["content"]
        )  # nosec B101 # Canonical page content and exact annotation text/note survive.


@pytest.mark.parametrize("text", ["W" * 300, "W " * 3000])
def test_render_never_silently_clips_an_oversized_page(text):
    result = run_worker(
        {"operation": "render", "draft": {"title": "Large page", "pages": [{"heading": "Large", "text": text}]}}
    )
    assert (result.returncode, result.stdout, result.stderr.strip()) == (2, b"", b"page_overflow")  # nosec B101 # Refuse unreadable overflow without publishing a partial PDF.


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "render", "draft": {"title": "Rowan", "pages": []}},
        {"operation": "render", "draft": {"title": "Rowan", "pages": [{"heading": "Rowan", "text": "x" * 6001}]}},
        {"operation": "render", "draft": {"title": "Rowan", "pages": DRAFT["pages"], "path": "/etc/passwd"}},
        {"operation": "annotate", "pdf": "not base64", "annotations": []},
        {"operation": "annotate", "pdf": base64.b64encode(b"<html>not a PDF</html>").decode(), "annotations": []},
        {"operation": "other", "path": "/etc/passwd"},
    ],
)
def test_invalid_inputs_return_no_pdf_or_input_details(payload):
    result = run_worker(payload)
    assert (result.returncode, result.stdout, result.stderr.strip()) == (2, b"", b"invalid_document")  # nosec B101 # Invalid input cannot become a successful or sensitive output.


def test_canonical_annotation_must_reference_an_actual_page():
    source = run_worker({"operation": "render", "draft": DRAFT})
    result = run_worker(
        {
            "operation": "annotate",
            "pdf": base64.b64encode(source.stdout).decode(),
            "annotations": [{"page": 3, "text": "Wrong page", "note": "Retain", "color": "blue", "kind": "highlight"}],
        }
    )
    assert (result.returncode, result.stdout, result.stderr.strip()) == (2, b"", b"invalid_annotation")  # nosec B101 # No annotation may be silently dropped onto a guessed page.


def test_combined_export_refuses_a_pdf_the_engine_would_silently_repair():
    import base64
    import re

    source = run_worker({"operation": "render", "draft": DRAFT}).stdout
    damaged = re.sub(rb"startxref\s+[0-9]+", b"startxref\n0", source)
    with read_pdf(damaged) as opened:
        assert opened.is_repaired  # nosec B101 # The control is genuinely corrupt, although the engine can salvage it.
    result = run_worker({"operation": "annotate", "pdf": base64.b64encode(damaged).decode("ascii"), "annotations": []})
    assert result.returncode == 2 and result.stdout == b"" and result.stderr.strip() == b"invalid_document"  # nosec B101 # Export must never silently substitute a repaired source.
