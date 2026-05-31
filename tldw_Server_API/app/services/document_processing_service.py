# /Server_API/app/services/document_processing_service.py
#
# Legacy placeholder service kept for non-production scaffolding only.
# Production deployments should use dedicated media/document ingestion paths.

import json
import os
import tempfile
import time
import zipfile
from typing import Any, Optional

import pypandoc
from docx2txt import docx2txt
from fastapi import HTTPException
from pypandoc import convert_file

from tldw_Server_API.app.core.Chunking import improved_chunking_process
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_repository
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.services._placeholder_guard import ensure_placeholder_service_enabled

_DOC_PROCESSING_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _ensure_placeholder_enabled():
    ensure_placeholder_service_enabled("Document")


def _file_security_strict() -> bool:
    import os as _os
    return is_truthy(_os.getenv("FILE_SECURITY_STRICT", "true"))


def build_plaintext_chunks(
    text: str,
    method: Optional[str] = None,
    max_size: int = 500,
    overlap: int = 50,
    language: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Chunk plaintext into compact metadata records for DB persistence."""
    method = method or "sentences"
    ck = Chunker()
    flat = ck.chunk_text_hierarchical_flat(
        text,
        method=method,
        max_size=max_size,
        overlap=overlap,
        language=language or "en",
    )
    kind_map = {
        "paragraph": "text",
        "list_unordered": "list",
        "list_ordered": "list",
        "code_fence": "code",
        "table_md": "table",
        "header_line": "heading",
        "header_atx": "heading",
    }
    chunks_out: list[dict[str, Any]] = []
    for item in flat:
        md = item.get("metadata", {}) or {}
        start = md.get("start_offset")
        end = md.get("end_offset")
        p_kind = md.get("paragraph_kind")
        ctype = kind_map.get(str(p_kind or "").lower(), "text")
        meta_small = {}
        if md.get("ancestry_titles"):
            meta_small["ancestry_titles"] = md.get("ancestry_titles")
        if md.get("section_path"):
            meta_small["section_path"] = md.get("section_path")
        chunks_out.append(
            {
                "text": item.get("text", ""),
                "start_char": start,
                "end_char": end,
                "chunk_type": ctype,
                "metadata": meta_small,
            }
        )
    return chunks_out


def _store_document_in_db(
    *,
    document_url: str,
    title: str,
    text_content: str,
    summary_text: str,
    custom_keywords: list[str],
    custom_prompt_input: Optional[str],
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    chunk_language: Optional[str],
    overwrite_existing: bool,
) -> int:
    """Persist document content through the repository-backed media DB API."""
    effective_user_id = 1
    db_path = get_user_media_db_path(effective_user_id)
    with managed_media_database(
        client_id="document_processing_service",
        db_path=db_path,
        initialize=False,
    ) as db:
        safe_metadata = json.dumps(
            {
                "title": title,
                "source": "document",
                "url": document_url,
            },
            ensure_ascii=False,
        )
        chunks = build_plaintext_chunks(
            text_content,
            method=chunk_method or "sentences",
            max_size=max_chunk_size,
            overlap=chunk_overlap,
            language=chunk_language or "en",
        )
        db_id, _, _ = get_media_repository(db).add_media_with_keywords(
            url=document_url,
            title=title,
            media_type="document",
            content=text_content,
            keywords=custom_keywords,
            prompt=custom_prompt_input,
            analysis_content=summary_text,
            safe_metadata=safe_metadata,
            transcription_model="document-import",
            author=None,
            ingestion_date=None,
            overwrite=overwrite_existing,
            chunks=chunks,
        )
        return db_id


async def process_documents(
    doc_urls: Optional[list[str]],
    doc_files: Optional[list[str]],
    api_name: Optional[str],
    api_key: Optional[str],
    custom_prompt_input: Optional[str],
    system_prompt_input: Optional[str],
    use_cookies: bool,
    cookies: Optional[str],
    keep_original: bool,
    custom_keywords: list[str],
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    store_in_db: bool = False,
    overwrite_existing: bool = False,
    custom_title: Optional[str] = None
) -> dict[str, Any]:
    """
    Process a set of documents (URLs or local files).
    1) Download/Read the files
    2) Convert each to raw text
    3) Optionally chunk & summarize
    4) Return a structured dict describing results
    """

    _ensure_placeholder_enabled()
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    progress_log: list[str] = []
    results: list[dict[str, Any]] = []

    # Track temporary files for cleanup if needed
    temp_files: list[str] = []

    def update_progress(message: str):
        logging.info(message)
        progress_log.append(message)

    def cleanup_temp_files():
        """Remove any downloaded/temporary files if keep_original=False."""
        for fp in temp_files:
            if not fp:
                continue
            try:
                if os.path.exists(fp):
                    os.remove(fp)
                    update_progress(f"Removed temp file: {fp}")
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS:
                logging.debug("Failed to remove temp file")
                update_progress("Failed to remove temp file")

    def download_document_file(url: str, use_cookies: bool, cookies: Optional[str]) -> str:
        """
        Downloads the document from a remote URL.
        Returns a local file path if successful, or raises an exception.
        """
        try:
            headers = {}
            if use_cookies and cookies:
                # You can parse cookies string if needed
                headers['Cookie'] = cookies

            # Enforce egress/SSRF policy
            try:
                from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
                pol = evaluate_url_policy(url)
                if not getattr(pol, 'allowed', False):
                    msg = f"Egress blocked: {getattr(pol, 'reason', 'denied')}"
                    if _file_security_strict():
                        raise RuntimeError(msg)
                    else:
                        logging.warning(f"[file_security] non-strict: {msg}")
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as _e:
                if _file_security_strict():
                    raise RuntimeError(f"Egress policy failure: {_e}") from _e
                else:
                    logging.warning(f"[file_security] non-strict: Egress check error: {_e}")

            # Validate with HEAD before download (size/MIME)
            from loguru import logger as _logger

            from tldw_Server_API.app.core.http_client import (
                RetryPolicy,
            )
            from tldw_Server_API.app.core.http_client import (
                create_client as http_create_client,
            )
            from tldw_Server_API.app.core.http_client import (
                download as http_download,
            )
            from tldw_Server_API.app.core.http_client import (
                fetch as http_fetch,
            )

            head_headers: dict = {}
            _last_err: Optional[Exception] = None
            # Prefer a tiny GET with Range over HEAD for preflight, to handle servers that
            # mishandle HEAD or HTTP/2. Keep body discard by using stream() and not reading.
            try:
                sc0 = http_create_client()
                try:
                    req_headers = dict(headers or {})
                    req_headers.setdefault("Range", "bytes=0-0")
                    _fallback_to = float(os.getenv("DOC_PREFLIGHT_RANGE_TIMEOUT", "5"))
                    with sc0.stream("GET", url, headers=req_headers, timeout=_fallback_to) as resp:
                        # Validate status: accept 2xx (including 206 Partial Content)
                        status = getattr(resp, "status_code", getattr(resp, "status", None))
                        if status is None or not (200 <= int(status) < 300):
                            _last_err = RuntimeError(f"Preflight GET Range returned status {status} for {url}")
                            # Force fallback path
                            raise _last_err
                        head_headers = dict(resp.headers or {})
                finally:
                    try:
                        sc0.close()
                    except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as cleanup_err:
                        _logger.debug(f"Ignored client close error (preflight GET): {cleanup_err}")
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as e_gr:
                _last_err = e_gr
                _logger.debug("Preflight GET Range failed, trying single HEAD with retries=1 (no http2): {}", e_gr)
                # As a best-effort fallback, try a single HEAD with http2 disabled and limited retries
                try:
                    sc = http_create_client(http2=False)
                    try:
                        head_resp2 = http_fetch(
                            method="HEAD",
                            url=url,
                            headers=headers,
                            timeout=60,
                            client=sc,
                            retry=RetryPolicy(attempts=1),
                        )
                        try:
                            # Validate HEAD status
                            status2 = getattr(head_resp2, "status_code", getattr(head_resp2, "status", None))
                            if status2 is None or not (200 <= int(status2) < 300):
                                raise RuntimeError(f"Preflight HEAD returned status {status2} for {url}")
                            head_headers = dict(head_resp2.headers or {})
                        finally:
                            try:
                                head_resp2.close()
                            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as cleanup_err:
                                _logger.debug(f"Ignored response close error (preflight HEAD): {cleanup_err}")
                    finally:
                        try:
                            sc.close()
                        except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as cleanup_err:
                            _logger.debug(f"Ignored client close error (preflight HEAD): {cleanup_err}")
                except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as e2:
                    _last_err = e2
                    _logger.warning("All preflight attempts failed for URL {}: {}", url, e2)
                    raise _last_err from e2
            try:
                max_bytes = int(os.getenv("DOC_DOWNLOAD_MAX_BYTES", "52428800"))  # 50 MB default
                cl = head_headers.get('content-length')
                if cl and cl.isdigit() and int(cl) > max_bytes:
                    if _file_security_strict():
                        raise RuntimeError("Document too large")
                    else:
                        logging.warning("[file_security] non-strict: Document exceeds size; continuing")
                allowed_mimes = [s.strip() for s in (os.getenv("DOC_DOWNLOAD_ALLOWED_MIME", "text/plain,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/markdown,text/html").split(",")) if s.strip()]
                ct = head_headers.get('content-type', '')
                if allowed_mimes and ct:
                    mt = ct.split(';', 1)[0].strip().lower()
                    if not any(mt == a or (a.endswith('/*') and mt.startswith(a[:-1])) for a in allowed_mimes):
                        if _file_security_strict():
                            raise RuntimeError(f"MIME not allowed: {mt}")
                        else:
                            logging.warning(f"[file_security] non-strict: MIME {mt} not allowed; continuing")
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as _guard:
                if _file_security_strict():
                    raise RuntimeError(str(_guard)) from _guard
                else:
                    logging.warning(f"[file_security] non-strict: guard error {str(_guard)}; continuing")

            # Create a temp file name with the same extension if possible and stream to file
            basename = os.path.basename(url).split("?")[0]
            ext = os.path.splitext(basename)[1] or ".bin"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                http_download(url=url, dest=tmp_path, headers=headers, retry=RetryPolicy())
                temp_files.append(tmp_path)
                return tmp_path
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as e:
                try:
                    os.unlink(tmp_path)
                except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    logging.debug(f"Ignored tmp unlink error: {cleanup_err}")
                raise RuntimeError(f"Download from '{url}' failed: {e}") from e
        except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as e:
            raise RuntimeError(f"Download from '{url}' failed: {str(e)}") from e

    def convert_to_text(file_path: str) -> str:
        """
        Given a local file path, attempts to read or convert it to plain text.
        Example logic for .txt, .docx, .rtf, .md, etc.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in [".txt", ".md"]:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()

        elif ext == ".docx":
            return docx2txt.process(file_path)

        elif ext == ".rtf":
            # pypandoc can handle RTF -> plain
            return pypandoc.convert_file(file_path, "plain", format="rtf")

        elif ext == ".pdf":
            # Minimal placeholder for PDF - prefer the dedicated PDF pipeline elsewhere
            try:
                return pypandoc.convert_file(file_path, 'plain', format='pdf')
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS:
                return "[PDF format not handled here - use a separate PDF pipeline?]"

        else:
            return "[Unsupported file extension or not recognized]"

    # Helper for chunking + summarization
    def summarize_text_if_needed(full_text: str) -> str:
        """
        Runs chunking + summarization if `api_name` is set. Otherwise returns "No summary" or an empty string.
        """
        if not api_name or api_name.lower() == "none":
            return ""  # no summarization
        try:
            # Load defaults if none provided
            nonlocal custom_prompt_input, system_prompt_input
            if not custom_prompt_input:
                custom_prompt_input = load_prompt("document", "document_summary_user") or custom_prompt_input
            if not system_prompt_input:
                system_prompt_input = load_prompt("document", "document_summary_system") or system_prompt_input
            # Prepare chunk options
            chunk_opts = {
                'method': chunk_method,
                'max_size': max_chunk_size,
                'overlap': chunk_overlap,
                'adaptive': use_adaptive_chunking,
                'multi_level': use_multi_level_chunking,
                'language': chunk_language
            }
            # Perform chunking
            chunked_texts = improved_chunking_process(full_text, chunk_opts)
            if not chunked_texts:
                # Fallback if chunking returned empty
                summary = analyze(api_name, full_text, custom_prompt_input, api_key, system_prompt_input)
                return summary or "No summary"
            else:
                # Summarize each chunk
                chunk_summaries = []
                for chunk_block in chunked_texts:
                    s = analyze(api_name, chunk_block["text"], custom_prompt_input, api_key, system_prompt_input)
                    if s:
                        chunk_summaries.append(s)
                # Combine them in a single pass
                combined_summary = "\n\n".join(chunk_summaries)
                return combined_summary
        except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS:
            logging.debug("Summarization failed")
            update_progress("Summarization failed")
            return "Summary generation failed"

    # Process doc URLs
    if doc_urls:
        for i, url in enumerate(doc_urls, start=1):
            item_result = {
                "input": url,
                "filename": None,
                "success": False,
                "text_content": None,
                "summary": None,
                "error": None,
                "db_id": None,
            }
            try:
                update_progress(f"Downloading document from URL {i}/{len(doc_urls)}: {url}")
                local_path = download_document_file(url, use_cookies, cookies)

                text_content = convert_to_text(local_path)
                item_result["filename"] = os.path.basename(local_path)
                item_result["text_content"] = text_content

                # Summarize
                summary_text = summarize_text_if_needed(text_content)
                item_result["summary"] = summary_text

                # (Optionally) Store in DB
                if store_in_db:
                    item_result["db_id"] = _store_document_in_db(
                        document_url=url,
                        title=custom_title or os.path.basename(local_path),
                        text_content=text_content,
                        summary_text=summary_text,
                        custom_keywords=custom_keywords,
                        custom_prompt_input=custom_prompt_input,
                        chunk_method=chunk_method,
                        max_chunk_size=max_chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_language=chunk_language,
                        overwrite_existing=overwrite_existing,
                    )

                processed_count += 1
                item_result["success"] = True
                update_progress(f"Processed URL {i} successfully.")
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as exc:
                failed_count += 1
                item_result["error"] = str(exc)
                logging.debug(f"Failed to process URL {i}")
                update_progress(f"Failed to process URL {i}")

            results.append(item_result)

    # Process local doc files
    if doc_files:
        for i, file_path in enumerate(doc_files, start=1):
            item_result = {
                "input": file_path,
                "filename": os.path.basename(file_path),
                "success": False,
                "text_content": None,
                "summary": None,
                "error": None,
                "db_id": None,
            }
            try:
                # Possibly check size if you want
                # if os.path.getsize(file_path) > MAX_FILE_SIZE:
                #     raise ValueError("File too large...")

                text_content = convert_to_text(file_path)
                item_result["text_content"] = text_content

                summary_text = summarize_text_if_needed(text_content)
                item_result["summary"] = summary_text

                if store_in_db:
                    item_result["db_id"] = _store_document_in_db(
                        document_url=file_path,
                        title=custom_title or os.path.basename(file_path),
                        text_content=text_content,
                        summary_text=summary_text,
                        custom_keywords=custom_keywords,
                        custom_prompt_input=custom_prompt_input,
                        chunk_method=chunk_method,
                        max_chunk_size=max_chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_language=chunk_language,
                        overwrite_existing=overwrite_existing,
                    )

                processed_count += 1
                item_result["success"] = True
                update_progress(f"Processed file {i}/{len(doc_files)}: {file_path}")
            except _DOC_PROCESSING_NONCRITICAL_EXCEPTIONS as exc:
                failed_count += 1
                item_result["error"] = str(exc)
                logging.debug(f"Failed to process file {i} ({file_path})")
                update_progress(f"Failed to process file {i} ({file_path})")

            results.append(item_result)

    # Cleanup any temp files if not keeping originals
    if not keep_original:
        cleanup_temp_files()

    total_time = time.time() - start_time
    update_progress(f"Document processing complete. Success: {processed_count}, Failed: {failed_count}, Time: {total_time:.1f}s")

    return {
        "status": "success" if failed_count == 0 else "partial",
        "message": f"Processed: {processed_count}, Failed: {failed_count}",
        "progress": progress_log,
        "results": results
    }


def _extract_zip_and_combine(zip_path: str) -> str:
    """
    Example helper: extracts a .zip that might contain .docx/.txt/etc.
    Then reads each file’s contents and concatenates them into one big string.
    Adjust logic as you see fit.
    """
    combined_text = []
    def _safe_extractall(zf: zipfile.ZipFile, dst: str) -> None:
        base = os.path.abspath(dst)
        for member in zf.infolist():
            # Prevent Zip Slip (path traversal)
            name = member.filename
            if name.endswith('/'):
                # Directory entry
                out_path = os.path.abspath(os.path.join(base, name))
                if not out_path.startswith(base + os.sep) and out_path != base:
                    if _file_security_strict():
                        raise RuntimeError("Unsafe zip entry path")
                    else:
                        logging.warning(f"[file_security] non-strict: skipping unsafe dir entry: {name}")
                        continue
                os.makedirs(out_path, exist_ok=True)
                continue
            out_path = os.path.abspath(os.path.join(base, name))
            if not out_path.startswith(base + os.sep):
                if _file_security_strict():
                    raise RuntimeError("Unsafe zip entry path")
                else:
                    logging.warning(f"[file_security] non-strict: skipping unsafe file entry: {name}")
                    continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(member, 'r') as src, open(out_path, 'wb') as dst_f:
                dst_f.write(src.read())

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            _safe_extractall(zip_ref, temp_dir)

        for root, _, files in os.walk(temp_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                extracted_path = os.path.join(root, f)
                # Read each file
                if ext in [".txt", ".md"]:
                    with open(extracted_path, encoding="utf-8", errors="replace") as f_obj:
                        combined_text.append(f_obj.read())
                elif ext == ".docx":
                    combined_text.append(docx2txt.process(extracted_path))
                elif ext == ".rtf":
                    combined_text.append(convert_file(extracted_path, "plain"))
                # etc. or skip unknown
    return "\n\n".join(combined_text)
