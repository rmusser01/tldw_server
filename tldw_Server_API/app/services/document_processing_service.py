# /Server_API/app/services/document_processing_service.py

# FIXME - This file is incomplete and needs to be completed. The code below is a placeholder and needs to be replaced.

import os
import tempfile
import time
import zipfile

import pypandoc
import requests
from fastapi import HTTPException
from typing import Optional, List, Dict, Any
from docx2txt import docx2txt
from pypandoc import convert_file

from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import perform_summarization
from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process
from tldw_Server_API.app.core.Utils.Utils import logger
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_to_database
from tldw_Server_API.app.core.Utils.Utils import logging

async def process_documents(
    doc_urls: Optional[List[str]],
    doc_files: Optional[List[str]],
    api_name: Optional[str],
    api_key: Optional[str],
    custom_prompt_input: Optional[str],
    system_prompt_input: Optional[str],
    use_cookies: bool,
    cookies: Optional[str],
    keep_original: bool,
    custom_keywords: List[str],
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    store_in_db: bool = False,
    overwrite_existing: bool = False,
    custom_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a set of documents (URLs or local files).
    1) Download/Read the files
    2) Convert each to raw text
    3) Optionally chunk & summarize
    4) Return a structured dict describing results
    """

    start_time = time.time()
    processed_count = 0
    failed_count = 0

    progress_log: List[str] = []
    results: List[Dict[str, Any]] = []

    # Track temporary files for cleanup if needed
    temp_files: List[str] = []

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
            except Exception as e:
                update_progress(f"Failed to remove {fp}: {str(e)}")

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

            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()

            # Create a temp file name with the same extension if possible
            basename = os.path.basename(url).split("?")[0]  # strip query
            ext = os.path.splitext(basename)[1] or ".bin"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(r.content)
                temp_files.append(tmp.name)
                return tmp.name
        except Exception as e:
            raise RuntimeError(f"Download from '{url}' failed: {str(e)}")

    def convert_to_text(file_path: str) -> str:
        """
        Given a local file path, attempts to read or convert it to plain text.
        Example logic for .txt, .docx, .rtf, .md, etc.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        elif ext == ".docx":
            return docx2txt.process(file_path)

        elif ext == ".rtf":
            # pypandoc can handle RTF -> plain
            return pypandoc.convert_file(file_path, "plain", format="rtf")

        elif ext == ".pdf":
            # If you want partial PDF support here (rather than a separate pdf codepath):
            # return pypandoc.convert_file(file_path, 'plain', format='pdf')
            # or use your PDF parsing approach
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
                summary = perform_summarization(api_name, full_text, custom_prompt_input, api_key, system_prompt=system_prompt_input)
                return summary or "No summary"
            else:
                # Summarize each chunk
                chunk_summaries = []
                for chunk_block in chunked_texts:
                    s = perform_summarization(api_name, chunk_block["text"], custom_prompt_input, api_key, system_prompt=system_prompt_input)
                    if s:
                        chunk_summaries.append(s)
                # Combine them in a single pass
                combined_summary = "\n\n".join(chunk_summaries)
                return combined_summary
        except Exception as e:
            update_progress(f"Summarization failed: {str(e)}")
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
                    # Use your own DB logic
                    db_id = add_media_to_database(
                        url=url,
                        title=custom_title or os.path.basename(local_path),
                        media_type="document",
                        content=text_content,
                        summary=summary_text,
                        keywords=custom_keywords,
                        prompt=custom_prompt_input,
                        system_prompt=system_prompt_input,
                        overwrite=overwrite_existing
                    )
                    item_result["db_id"] = db_id

                processed_count += 1
                item_result["success"] = True
                update_progress(f"Processed URL {i} successfully.")
            except Exception as exc:
                failed_count += 1
                item_result["error"] = str(exc)
                update_progress(f"Failed to process URL {i}: {str(exc)}")

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
                    db_id = add_media_to_database(
                        url=file_path,
                        title=custom_title or os.path.basename(file_path),
                        media_type="document",
                        content=text_content,
                        summary=summary_text,
                        keywords=custom_keywords,
                        prompt=custom_prompt_input,
                        system_prompt=system_prompt_input,
                        overwrite=overwrite_existing
                    )
                    item_result["db_id"] = db_id

                processed_count += 1
                item_result["success"] = True
                update_progress(f"Processed file {i}/{len(doc_files)}: {file_path}")
            except Exception as exc:
                failed_count += 1
                item_result["error"] = str(exc)
                update_progress(f"Failed to process file {i} ({file_path}): {str(exc)}")

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
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, _, files in os.walk(temp_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                extracted_path = os.path.join(root, f)
                # Read each file
                if ext in [".txt", ".md"]:
                    with open(extracted_path, "r", encoding="utf-8", errors="replace") as f_obj:
                        combined_text.append(f_obj.read())
                elif ext == ".docx":
                    combined_text.append(docx2txt.process(extracted_path))
                elif ext == ".rtf":
                    combined_text.append(convert_file(extracted_path, "plain"))
                # etc. or skip unknown
    return "\n\n".join(combined_text)
