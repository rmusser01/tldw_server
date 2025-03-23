# /Server_API/app/services/xml_processing_service.py


# FIXME - This is a placeholder for the actual XML processing logic

# Parse the file (extract text, chunking, etc.).
# Optionally summarize.
# Return all final data in a dictionary.

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional, List

from fastapi import HTTPException
from Server_API.app.core.logging import logger
from Server_API.app.services.ephemeral_store import ephemeral_storage
from Server_API.app.core.DB_Management.DB_Manager import add_media_to_database
from Server_API.app.core.Utils.Utils import logging

from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from App_Function_Libraries.Chunk_Lib import chunk_xml

async def process_xml_task(
    file_bytes: bytes,
    filename: str,
    title: Optional[str],
    author: Optional[str],
    keywords: List[str],
    system_prompt: Optional[str],
    custom_prompt: Optional[str],
    auto_summarize: bool,
    api_name: Optional[str],
    api_key: Optional[str]
) -> dict:
    """
    Reads & chunks an XML file, optionally runs summarization,
    and returns a dict with final data.
    """

    try:
        logger.info(f"Processing XML file: {filename}")

        # 1) Save the incoming bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_bytes)

        # 2) Parse the XML with built-in logic
        try:
            tree = ET.parse(tmp_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise HTTPException(status_code=400, detail=f"Invalid XML: {str(e)}")

        # 3) Chunk the XML. For instance:
        chunk_options = {
            'method': 'xml',
            'max_size': 1000,
            'overlap': 200,
            'language': 'english'
        }
        # Convert root to string
        xml_string = ET.tostring(root, encoding='unicode')
        chunks = chunk_xml(xml_string, chunk_options)

        # 4) Summarization
        summary_text = "No summary provided"
        if auto_summarize and api_name and api_name.lower() != "none" and api_key:
            # Combine all chunk text
            full_text = '\n'.join(ch['text'] for ch in chunks)
            combined_prompt = (system_prompt or "") + "\n\n" + (custom_prompt or "")
            # summary_text = perform_summarization(api_name, full_text, combined_prompt, api_key)
            summary_text = f"[Auto-summarized with {api_name}]"

        # 5) Build final result dictionary
        #    segments can store each chunk with text + metadata
        segments = []
        for ch in chunks:
            segments.append({
                "Text": ch["text"],
                "metadata": ch.get("metadata", {})
            })

        final_title = title or "Untitled XML Document"
        final_author = author or "Unknown"

        info_dict = {
            "title": final_title,
            "uploader": final_author,
            "file_type": "xml",
            "root_element": root.tag  # example piece of metadata
        }

        return {
            "filename": filename,
            "info_dict": info_dict,
            "segments": segments,
            "summary": summary_text,
            "keywords": keywords,
            "custom_prompt": custom_prompt,
            "system_prompt": system_prompt
        }

    except Exception as e:
        logger.error(f"Error processing XML file: {filename} -> {e}")
        raise HTTPException(status_code=500, detail=str(e))
