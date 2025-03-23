# /Server_API/app/services/document_processing_service.py

# FIXME - This file is incomplete and needs to be completed. The code below is a placeholder and needs to be replaced.

import os
import tempfile
import zipfile
from fastapi import HTTPException
from typing import Optional, List
from docx2txt import docx2txt
from pypandoc import convert_file

from Server_API.app.core.logging import logger
from Server_API.app.services.ephemeral_store import ephemeral_storage
from Server_API.app.core.DB_Management.DB_Manager import add_media_to_database
from Server_API.app.core.Utils.Utils import logging
# If you have a summarization library:
# from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization

async def process_document_task(
    file_bytes: bytes,
    filename: str,
    custom_prompt: Optional[str],
    api_name: Optional[str],
    api_key: Optional[str],
    keywords: List[str],
    system_prompt: Optional[str] = None,
    auto_summarize: bool = False,
) -> dict:
    """
    Takes in raw file bytes + filename, converts to plain text if necessary,
    optionally runs summarization, and returns a dictionary with all final data.
    """
    try:
        # 1) Save the uploaded bytes to a tmp path
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_bytes)

        # 2) Convert or read the file to text
        text_content = ""
        extension = suffix.lower()

        if extension in [".txt", ".md"]:
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                text_content = f.read()

        elif extension == ".docx":
            text_content = docx2txt.process(tmp_path)

        elif extension == ".rtf":
            # Convert to .md or .txt using pypandoc
            # e.g., pypandoc:
            out = convert_file(tmp_path, "plain")
            text_content = out

        elif extension == ".zip":
            # If you want to handle a zip of multiple docs, do so here
            # Then combine them or handle them each separately. Example:
            text_content = _extract_zip_and_combine(tmp_path)
        else:
            text_content = "[Unsupported extension or not recognized]"

        # 3) Summarize if requested
        summary_text = "No summary available"
        if auto_summarize and api_name and api_name.lower() != "none":
            # summary_text = perform_summarization(
            #     api_name=api_name,
            #     input_data=text_content,
            #     custom_prompt=(system_prompt or "") + "\n\n" + (custom_prompt or ""),
            #     api_key=api_key
            # )
            summary_text = f"[Auto-summarized with {api_name}]"

        return {
            "filename": filename,
            "text_content": text_content,
            "summary": summary_text,
            "keywords": keywords,
            "prompts": {
                "system_prompt": system_prompt,
                "custom_prompt": custom_prompt
            }
        }
    except Exception as e:
        logger.error(f"Error processing document file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
