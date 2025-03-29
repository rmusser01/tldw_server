# /Server_API/app/services/pdf_processing_service.py


# FIXME - This is a placeholder for the actual PDF processing logic

# Also need to add support for individual steps, so that we can call them from the UI and not just the entire pipeline

import os
import tempfile
from typing import Optional, List

from fastapi import HTTPException
from tldw_Server_API.app.core.logging import logger
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_to_database
from tldw_Server_API.app.core.Utils.Utils import logging

# Assume you have these from your existing library:
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Ingestion_Lib import (
    extract_metadata_from_pdf,
    pymupdf4llm_parse_pdf,
    extract_text_and_format_from_pdf,
    process_and_ingest_pdf,  # or we can call each step directly
)

# If you want optional summarization:
# from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization


async def process_pdf_task(
    file_bytes: bytes,
    filename: str,
    parser: str = "pymupdf4llm",
    custom_prompt: Optional[str] = None,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_summarize: bool = False,
    keywords: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Processes a PDF file (in memory), extracts text + metadata, optionally summarizes,
    and returns a dictionary with final data. This is the same pattern used
    for your existing ingestion flows.
    """
    try:
        logger.info(f"Processing PDF {filename} with parser={parser}")

        # 1) Save PDF bytes to a temporary file
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_bytes)

        # 2) Extract text based on chosen parser
        if parser == "pymupdf":
            # E.g. “extract_text_and_format_from_pdf”
            text_content = extract_text_and_format_from_pdf(tmp_path)
        elif parser == "docling":
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            parsed_pdf = converter.convert(tmp_path)
            text_content = parsed_pdf.document.export_to_markdown()
        else:
            # default to pymupdf4llm
            text_content = pymupdf4llm_parse_pdf(tmp_path)

        # 3) Extract metadata
        metadata = extract_metadata_from_pdf(tmp_path)
        # Title, author from metadata if not provided by user
        derived_title = metadata.get("title") or os.path.splitext(filename)[0]
        derived_author = metadata.get("author") or "Unknown"

        # 4) Summarize if requested
        summary_text = "No summary"
        if auto_summarize and api_name and api_name.lower() != "none":
            combined_prompt = (system_prompt or "") + "\n\n" + (custom_prompt or "")
            # summary_text = perform_summarization(
            #     api_name=api_name,
            #     input_data=text_content,
            #     custom_prompt=combined_prompt,
            #     api_key=api_key
            # )
            summary_text = f"[Auto-summarized with {api_name}]"

        return {
            "filename": filename,
            "title": derived_title,
            "author": derived_author,
            "keywords": keywords or [],
            "text_content": text_content,
            "summary": summary_text,
            "metadata": metadata,
            "parser_used": parser,
        }

    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
