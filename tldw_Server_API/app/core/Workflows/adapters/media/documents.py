"""Document processing adapters.

This module includes adapters for document operations:
- pdf_extract: Extract content from PDF files
- ocr: Optical character recognition on images
- document_table_extract: Extract tables from documents
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import (
    extract_openai_content,
    resolve_artifacts_dir,
    resolve_workflow_file_path,
    resolve_workflow_file_uri,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.media._config import (
    DocumentTableExtractConfig,
    OCRConfig,
    PDFExtractConfig,
)

_DOCUMENT_PATH_RESOLUTION_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    AttributeError,
    KeyError,
)

_DOCUMENT_FILE_IO_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
)

_DOCUMENT_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
    ConnectionError,
    TimeoutError,
)


@registry.register(
    "pdf_extract",
    category="media",
    description="Extract content from PDF",
    parallelizable=True,
    tags=["media", "document"],
    config_model=PDFExtractConfig,
)
async def run_pdf_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract text and metadata from a PDF file.

    Config:
      - pdf_uri: str (templated, file:// path - required)
      - parser: Literal["pymupdf4llm", "pymupdf", "docling"] (default: "pymupdf4llm")
      - title: str (optional, templated - title override)
      - author: str (optional, templated - author override)
      - keywords: List[str] (optional)
      - perform_chunking: bool (default: True)
      - chunk_method: str (default: "sentences")
      - max_chunk_size: int (default: 500)
      - chunk_overlap: int (default: 100)
      - enable_ocr: bool (default: False)
      - ocr_backend: str (optional)
      - ocr_lang: str (default: "eng")
      - ocr_mode: Literal["fallback", "always"] (default: "fallback")
      - enable_vlm: bool (default: False)
      - vlm_backend: str (optional)
      - vlm_detect_tables_only: bool (default: True)
    Output:
      - {status, content, text, metadata, chunks, keywords, page_count, warnings}
    """
    # Check cancellation
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    # Get and template pdf_uri
    pdf_uri_t = config.get("pdf_uri")
    if not pdf_uri_t:
        return {"error": "missing_pdf_uri", "status": "Error", "content": "", "text": ""}

    pdf_uri = _tmpl(str(pdf_uri_t), context) or str(pdf_uri_t)
    pdf_uri = pdf_uri.strip()

    if not pdf_uri:
        return {"error": "missing_pdf_uri", "status": "Error", "content": "", "text": ""}

    # Get other config options with templating where needed
    parser = str(config.get("parser") or "pymupdf4llm").strip()
    if parser not in ("pymupdf4llm", "pymupdf", "docling"):
        parser = "pymupdf4llm"

    title_t = config.get("title")
    title_override = _tmpl(str(title_t), context) if title_t else None

    author_t = config.get("author")
    author_override = _tmpl(str(author_t), context) if author_t else None

    keywords = config.get("keywords")
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    elif not isinstance(keywords, list):
        keywords = None

    perform_chunking = config.get("perform_chunking")
    perform_chunking = True if perform_chunking is None else bool(perform_chunking)

    chunk_method = str(config.get("chunk_method") or "sentences").strip()
    max_chunk_size = int(config.get("max_chunk_size") or 500)
    chunk_overlap = int(config.get("chunk_overlap") or 100)

    # OCR options
    enable_ocr = bool(config.get("enable_ocr"))
    ocr_backend = config.get("ocr_backend")
    ocr_lang = str(config.get("ocr_lang") or "eng")
    ocr_mode = str(config.get("ocr_mode") or "fallback")
    if ocr_mode not in ("fallback", "always"):
        ocr_mode = "fallback"

    # VLM options
    enable_vlm = bool(config.get("enable_vlm"))
    vlm_backend = config.get("vlm_backend")
    vlm_detect_tables_only = config.get("vlm_detect_tables_only")
    vlm_detect_tables_only = True if vlm_detect_tables_only is None else bool(vlm_detect_tables_only)

    # Test mode simulation
    if is_test_mode():
        simulated_content = f"[TEST_MODE PDF] Simulated text extraction from {pdf_uri}"
        simulated_chunks = [
            {"text": f"Chunk 1 from {pdf_uri}", "index": 0},
            {"text": f"Chunk 2 from {pdf_uri}", "index": 1},
        ] if perform_chunking else []

        return {
            "status": "Success",
            "content": simulated_content,
            "text": simulated_content,  # Alias for chaining
            "metadata": {
                "title": title_override or "Simulated Document",
                "author": author_override or "Unknown",
                "page_count": 5,
                "parser_used": parser,
            },
            "chunks": simulated_chunks,
            "keywords": keywords or [],
            "page_count": 5,
            "warnings": [],
            "simulated": True,
        }

    # Resolve file URI to local path
    try:
        if pdf_uri.startswith("file://"):
            local_path = resolve_workflow_file_uri(pdf_uri, context, config)
        else:
            local_path = resolve_workflow_file_path(pdf_uri, context, config)
    except AdapterError:
        return {"error": "invalid_pdf_path", "status": "Error", "content": "", "text": ""}
    except _DOCUMENT_PATH_RESOLUTION_EXCEPTIONS:
        logger.debug("PDF extract adapter: failed to resolve path")
        return {"error": "invalid_pdf_path", "status": "Error", "content": "", "text": ""}

    if not local_path.exists():
        return {"error": "pdf_not_found", "status": "Error", "content": "", "text": ""}

    # Read PDF bytes
    try:
        pdf_bytes = local_path.read_bytes()
    except _DOCUMENT_FILE_IO_EXCEPTIONS:
        logger.exception("PDF extract adapter: failed to read PDF")
        return {"error": "pdf_read_error", "status": "Error", "content": "", "text": ""}

    # Process PDF
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf

        # Build chunk options if chunking is enabled
        chunk_options = None
        if perform_chunking:
            chunk_options = {
                "method": chunk_method,
                "max_size": max_chunk_size,
                "overlap": chunk_overlap,
            }

        # Call process_pdf (sync function, wrap with asyncio.to_thread)
        result = await asyncio.to_thread(
            process_pdf,
            file_input=pdf_bytes,
            filename=str(local_path.name),
            parser=parser,
            title_override=title_override,
            author_override=author_override,
            keywords=keywords,
            perform_chunking=perform_chunking,
            chunk_options=chunk_options,
            perform_analysis=False,  # Don't do LLM analysis in workflow step
            enable_ocr=enable_ocr,
            ocr_backend=ocr_backend,
            ocr_lang=ocr_lang,
            ocr_mode=ocr_mode,
            enable_vlm=enable_vlm,
            vlm_backend=vlm_backend,
            vlm_detect_tables_only=vlm_detect_tables_only,
        )

        if result is None:
            return {"error": "pdf_processing_failed", "status": "Error", "content": "", "text": ""}

        # Extract page count from metadata
        page_count = 0
        if isinstance(result.get("metadata"), dict):
            page_count = result["metadata"].get("page_count", 0) or result["metadata"].get("raw", {}).get("page_count", 0)

        content = result.get("content") or ""

        # Optional artifact persistence
        try:
            if bool(config.get("save_artifact")) and callable(context.get("add_artifact")):
                step_run_id = str(context.get("step_run_id") or "")
                art_dir = resolve_artifacts_dir(step_run_id or f"pdf_{int(time.time()*1000)}")
                art_dir.mkdir(parents=True, exist_ok=True)
                fpath = art_dir / "pdf_content.txt"
                fpath.write_text(content, encoding="utf-8")
                context["add_artifact"](
                    type="pdf_text",
                    uri=f"file://{fpath}",
                    size_bytes=len(content.encode("utf-8")),
                    mime_type="text/plain",
                    metadata={"parser": parser, "page_count": page_count},
                )
        except _DOCUMENT_NONCRITICAL_EXCEPTIONS:
            logger.debug("PDF extract adapter: failed to persist artifact")

        return {
            "status": result.get("status") or "Success",
            "content": content,
            "text": content,  # Alias for chaining
            "metadata": result.get("metadata") or {},
            "chunks": result.get("chunks") or [],
            "keywords": result.get("keywords") or [],
            "page_count": page_count,
            "warnings": result.get("warnings") or [],
        }

    except _DOCUMENT_NONCRITICAL_EXCEPTIONS:
        logger.exception("PDF extract adapter error")
        return {"error": "pdf_extract_error", "status": "Error", "content": "", "text": ""}


@registry.register(
    "ocr",
    category="media",
    description="Optical character recognition",
    parallelizable=True,
    tags=["media", "ocr"],
    config_model=OCRConfig,
)
async def run_ocr_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Run OCR on an image to extract text.

    Config:
      - image_uri: str (templated, file:// path or artifact URI - required)
      - backend: str (optional: "auto", "tesseract", "deepseek", "nemotron_parse", etc.)
      - language: str (default: "eng")
      - output_format: Literal["text", "markdown", "html", "json"] (default: "text")
      - prompt_preset: str (optional, backend-specific)
    Output:
      - {text, format, blocks, tables, meta, warnings}
    """
    # Check cancellation
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    # Get and template image_uri
    image_uri_t = config.get("image_uri")
    if not image_uri_t:
        return {"error": "missing_image_uri", "text": ""}

    image_uri = _tmpl(str(image_uri_t), context) or str(image_uri_t)
    image_uri = image_uri.strip()

    if not image_uri:
        return {"error": "missing_image_uri", "text": ""}

    # Get other config options
    backend_name = config.get("backend") or None
    language = str(config.get("language") or "eng").strip()
    output_format = str(config.get("output_format") or "text").strip().lower()
    if output_format not in ("text", "markdown", "html", "json"):
        output_format = "text"
    prompt_preset = config.get("prompt_preset")

    # Test mode simulation
    if is_test_mode():
        return {
            "text": f"[TEST_MODE OCR] Simulated text extraction from {image_uri}",
            "format": output_format,
            "blocks": [{"text": "Simulated block 1", "bbox": [0, 0, 100, 50], "block_type": "paragraph"}],
            "tables": [],
            "meta": {"backend": backend_name or "tesseract", "language": language},
            "warnings": [],
            "simulated": True,
        }

    # Resolve file URI to local path
    try:
        if image_uri.startswith("file://"):
            local_path = resolve_workflow_file_uri(image_uri, context, config)
        else:
            local_path = resolve_workflow_file_path(image_uri, context, config)
    except AdapterError:
        return {"error": "invalid_image_path", "text": ""}
    except _DOCUMENT_PATH_RESOLUTION_EXCEPTIONS:
        logger.debug("OCR adapter: failed to resolve image path")
        return {"error": "invalid_image_path", "text": ""}

    if not local_path.exists():
        return {"error": "image_not_found", "text": ""}

    # Read image bytes
    try:
        image_bytes = local_path.read_bytes()
    except _DOCUMENT_FILE_IO_EXCEPTIONS:
        logger.exception("OCR adapter: failed to read image")
        return {"error": "image_read_error", "text": ""}

    # Get OCR backend
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

        backend = get_backend(backend_name)
        if backend is None:
            return {"error": "ocr_backend_unavailable", "text": ""}

        # Use structured OCR if non-text format or prompt preset specified
        if output_format != "text" or prompt_preset:
            result = backend.ocr_image_structured(
                image_bytes,
                lang=language,
                output_format=output_format,
                prompt_preset=prompt_preset,
            )
            output = result.as_dict()
        else:
            text = backend.ocr_image(image_bytes, lang=language)
            output = {
                "text": text or "",
                "format": "text",
                "blocks": [],
                "tables": [],
                "meta": {"backend": getattr(backend, "name", "unknown"), "language": language},
                "warnings": [],
            }

        # Optional artifact persistence
        try:
            if bool(config.get("save_artifact")) and callable(context.get("add_artifact")):
                step_run_id = str(context.get("step_run_id") or "")
                art_dir = resolve_artifacts_dir(step_run_id or f"ocr_{int(time.time()*1000)}")
                art_dir.mkdir(parents=True, exist_ok=True)
                fpath = art_dir / "ocr_result.txt"
                fpath.write_text(output.get("text") or "", encoding="utf-8")
                context["add_artifact"](
                    type="ocr_text",
                    uri=f"file://{fpath}",
                    size_bytes=len((output.get("text") or "").encode("utf-8")),
                    mime_type="text/plain",
                    metadata={"backend": output.get("meta", {}).get("backend"), "format": output_format},
                )
        except _DOCUMENT_NONCRITICAL_EXCEPTIONS:
            logger.debug("OCR adapter: failed to persist artifact")

        return output

    except _DOCUMENT_NONCRITICAL_EXCEPTIONS:
        logger.exception("OCR adapter error")
        return {"error": "ocr_error", "text": ""}


@registry.register(
    "document_table_extract",
    category="media",
    description="Extract tables from documents",
    parallelizable=False,
    tags=["media", "document"],
    config_model=DocumentTableExtractConfig,
)
async def run_document_table_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract tables from documents as structured JSON/CSV.

    Config:
      - file_path: str - Path to document (PDF, image, etc.)
      - file_uri: str - Alternative: file:// URI
      - output_format: str - "json" or "csv" (default: "json")
      - table_index: int - Specific table index to extract (default: all)
      - provider: str - Extraction provider: "docling", "llm" (default: "docling")
    Output:
      - tables: list[dict] - Extracted tables
      - count: int
      - format: str
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    file_path = config.get("file_path")
    file_uri = config.get("file_uri")

    if file_uri:
        try:
            file_path = resolve_workflow_file_uri(file_uri, context, config)
        except AdapterError:
            return {"error": "invalid_file_uri", "tables": [], "count": 0}
        except _DOCUMENT_PATH_RESOLUTION_EXCEPTIONS:
            return {"error": "invalid_file_uri", "tables": [], "count": 0}
    elif file_path:
        if isinstance(file_path, str):
            file_path = _tmpl(file_path, context) or file_path
        try:
            file_path = resolve_workflow_file_path(file_path, context, config)
        except AdapterError:
            return {"error": "file_access_denied", "tables": [], "count": 0}
        except _DOCUMENT_PATH_RESOLUTION_EXCEPTIONS:
            return {"error": "file_access_denied", "tables": [], "count": 0}
    else:
        return {"error": "missing_file_path", "tables": [], "count": 0}

    output_format = str(config.get("output_format", "json")).lower()
    table_index = config.get("table_index")
    provider = str(config.get("provider", "docling")).lower()

    tables = []

    try:
        if provider == "docling":
            # Use docling for table extraction
            try:
                from docling.document_converter import DocumentConverter

                converter = DocumentConverter()
                result = converter.convert(str(file_path))

                for i, table in enumerate(result.document.tables):
                    if table_index is not None and i != table_index:
                        continue

                    table_data = {
                        "index": i,
                        "rows": [],
                        "headers": [],
                    }

                    # Extract table data
                    if hasattr(table, "export_to_dataframe"):
                        df = table.export_to_dataframe()
                        table_data["headers"] = list(df.columns)
                        table_data["rows"] = df.values.tolist()
                    elif hasattr(table, "data"):
                        table_data["rows"] = table.data

                    tables.append(table_data)

            except ImportError:
                logger.warning("Docling not available, falling back to LLM extraction")
                provider = "llm"

        if provider == "llm" or not tables:
            # Fallback to LLM-based extraction
            # Read file content
            content = ""
            if str(file_path).lower().endswith(".pdf"):
                try:
                    import pymupdf
                    doc = pymupdf.open(str(file_path))
                    for page in doc:
                        content += page.get_text()
                    doc.close()
                except _DOCUMENT_NONCRITICAL_EXCEPTIONS:
                    logger.debug("PDF read error")
            else:
                with contextlib.suppress(_DOCUMENT_FILE_IO_EXCEPTIONS):
                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")

            if content:
                from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

                system_prompt = """Extract all tables from the document content.
Return a JSON array of tables, each with "headers" (array of column names) and "rows" (array of row arrays).
Example: [{"headers": ["Name", "Value"], "rows": [["A", "1"], ["B", "2"]]}]"""

                messages = [{"role": "user", "content": f"Extract tables from:\n\n{content[:10000]}"}]
                response = await perform_chat_api_call_async(
                    messages=messages,
                    system_message=system_prompt,
                    max_tokens=4000,
                    temperature=0.3,
                )

                text = extract_openai_content(response) or "[]"
                try:
                    json_match = re.search(r'\[[\s\S]*\]', text)
                    if json_match:
                        tables = json.loads(json_match.group())
                except (ValueError, TypeError):
                    pass

        # Convert to CSV if requested
        if output_format == "csv":
            for table in tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                csv_lines = []
                if headers:
                    csv_lines.append(",".join(str(h) for h in headers))
                for row in rows:
                    csv_lines.append(",".join(str(c) for c in row))
                table["csv"] = "\n".join(csv_lines)

        return {
            "tables": tables,
            "count": len(tables),
            "format": output_format,
        }

    except _DOCUMENT_NONCRITICAL_EXCEPTIONS:
        logger.exception("Document table extract error")
        return {"error": "table_extract_error", "tables": [], "count": 0}
