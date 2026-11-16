"""Extract a PDF outline for both reader navigation and background outputs."""

from typing import BinaryIO

from loguru import logger

from tldw_Server_API.app.core.Media.document_outline_schemas import OutlineEntry

# Maximum file size for outline extraction (500 MB)
MAX_OUTLINE_FILE_SIZE = 500 * 1024 * 1024


def _check_pymupdf_available() -> bool:
    """Check if PyMuPDF is available for import."""
    try:
        import pymupdf  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_pdf_outline(pdf_data: bytes | BinaryIO) -> tuple[list[OutlineEntry], int]:
    """
    Extract table of contents from PDF using PyMuPDF.

    Args:
        pdf_data: Raw PDF file content as bytes, or a file-like object (BinaryIO).
                  If a BinaryIO is passed, its contents will be read into memory.

    Returns:
        Tuple of (list of outline entries, total page count)

    Raises:
        RuntimeError: If PyMuPDF is not installed
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed") from exc

    # Handle both bytes and file-like objects
    pdf_bytes = pdf_data.read() if hasattr(pdf_data, "read") else pdf_data

    entries: list[OutlineEntry] = []

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = doc.page_count

        # get_toc returns list of [level, title, page_number, ...]
        # page_number is 1-indexed
        toc = doc.get_toc()

        for item in toc:
            if len(item) >= 3:
                level, title, page = item[0], item[1], item[2]
                # Validate and clamp values
                if isinstance(level, int) and isinstance(title, str) and isinstance(page, int):
                    # Clamp level to 1-6 range
                    level = max(1, min(6, level))
                    # Ensure page is within valid range (some PDFs have invalid page refs)
                    page = max(1, min(total_pages, page)) if total_pages > 0 else max(1, page)
                    # Filter out entries with empty titles
                    title_stripped = title.strip()
                    if title_stripped:
                        entries.append(OutlineEntry(level=level, title=title_stripped, page=page))

        doc.close()
        logger.debug(
            "Extracted {} outline entries from PDF with {} pages",
            len(entries),
            total_pages,
        )
        return entries, total_pages

    except Exception:  # noqa: BLE001 - preserve optional source fallback on parser/backend failures
        logger.error("Error extracting PDF outline")
        return [], 0
