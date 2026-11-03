"""One isolated PDF operation; stdin JSON in, complete PDF bytes out."""

from __future__ import annotations

import base64
import html
import io
import json
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tldw_Server_API.app.core.Document_Workspace.models import (
    MAX_ANNOTATION_BYTES,
    MAX_PDF_BYTES,
    MAX_WORK_BYTES,
    PDF_ERRORS,
    DocumentRenderRequest,
    PdfAnnotation,
)


class RenderWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["render"]
    draft: DocumentRenderRequest


class AnnotateWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["annotate"]
    pdf: str = Field(max_length=((MAX_PDF_BYTES + 2) // 3) * 4)
    annotations: list[PdfAnnotation] = Field(max_length=500)


class PdfFailure(ValueError):
    """A fixed, non-sensitive failure code for the parent process."""


def _engine():
    """Support the declared PyMuPDF floor and its modern module name."""
    try:
        import pymupdf as fitz
    except ModuleNotFoundError as exc:
        if exc.name != "pymupdf":
            raise
        import fitz
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    return fitz


def render(draft: DocumentRenderRequest) -> bytes:
    """Render each escaped plain-text draft page without truncation."""
    fitz = _engine()
    output = io.BytesIO()
    writer = fitz.DocumentWriter(output)
    try:
        for page in draft.pages:
            heading = html.escape(page.heading)
            text = html.escape(page.text).replace("\n", "<br>")
            story = fitz.Story(
                html=f"<h1>{heading}</h1><p>{text}</p>",
                user_css="body {font-family: sans-serif; font-size: 12pt; line-height: 1.5;} h1 {font-size: 18pt;}",
            )
            device = writer.begin_page(fitz.Rect(0, 0, 595, 842))
            try:
                more, _ = story.place(fitz.Rect(48, 48, 547, 794))
                if more:
                    raise PdfFailure("page_overflow")
                story.draw(device)
            finally:
                writer.end_page()
    finally:
        writer.close()
    with fitz.open(stream=output.getvalue(), filetype="pdf") as document:
        for page, requested in zip(document, draft.pages):
            expected = " ".join((requested.heading + " " + requested.text).split())
            actual = " ".join(page.get_text().split())
            spans = [
                span
                for block in page.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for span in line["spans"]
            ]
            if actual != expected or any(span["bbox"][0] < 47 or span["bbox"][2] > 548 for span in spans):
                raise PdfFailure("page_overflow")
        document.set_metadata({"title": draft.title, "creator": "tldw Document Workspace"})
        return document.tobytes(garbage=4, deflate=True)


def annotate(work: AnnotateWork) -> bytes:
    """Retain original pages and add canonical selected text/notes as page comments."""
    source = base64.b64decode(work.pdf, validate=True)
    if len(source) > MAX_PDF_BYTES:
        raise PdfFailure("too_large")
    annotations = [item.model_dump() for item in work.annotations]
    if len(json.dumps(annotations, ensure_ascii=False).encode()) > MAX_ANNOTATION_BYTES:
        raise PdfFailure("invalid_annotation")
    fitz = _engine()
    colors = {"yellow": (1, 0.85, 0.2), "green": (0.2, 0.7, 0.3), "blue": (0.2, 0.5, 0.9), "pink": (0.9, 0.3, 0.6)}
    with fitz.open(stream=source, filetype="pdf") as document:
        if not document.is_pdf or document.is_repaired or not document.page_count:
            raise PdfFailure("invalid_document")
        if document.needs_pass:
            raise PdfFailure("locked_pdf")
        counts: dict[int, int] = {}
        for item in work.annotations:
            if item.page > document.page_count:
                raise PdfFailure("invalid_annotation")
            page = document[item.page - 1]
            count = counts.get(item.page, 0)
            counts[item.page] = count + 1
            columns = max(1, int(page.rect.width // 18) - 2)
            point = fitz.Point(
                min(page.rect.width / 2, 18) + 16 * (count % columns),
                max(0, min(page.rect.height - 1, 18 + 16 * (count // columns))),
            )
            comment = page.add_text_annot(point, f"Selected text:\n{item.text}\n\nNote:\n{item.note}", icon="Note")
            comment.set_info(title="tldw", subject=f"{item.kind} ({item.color})")
            comment.set_colors(stroke=colors[item.color])
            comment.update()
        return document.tobytes(garbage=4, deflate=True)


def main() -> int:
    """Never publish partial bytes or untrusted exception details."""
    try:
        raw = sys.stdin.buffer.read(MAX_WORK_BYTES + 1)
        if len(raw) > MAX_WORK_BYTES:
            raise PdfFailure("too_large")
        work = TypeAdapter(Annotated[RenderWork | AnnotateWork, Field(discriminator="operation")]).validate_python(
            json.loads(raw)
        )
        result = render(work.draft) if isinstance(work, RenderWork) else annotate(work)
        if len(result) > MAX_PDF_BYTES:
            raise PdfFailure("too_large")
        sys.stdout.buffer.write(result)
        return 0
    except PdfFailure as exc:
        code = str(exc) if str(exc) in PDF_ERRORS else "invalid_document"
    except Exception:  # noqa: BLE001 - the child protocol returns only a fixed error, never private parser details.
        code = "invalid_document"
    sys.stderr.write(code + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
