# /Server_API/app/services/xml_processing_service.py


# Legacy placeholder service kept for non-production scaffolding only.

# Parse the file (extract text, chunking, etc.).
# Optionally summarize.
# Return all final data in a dictionary.

import os
import tempfile
from typing import Optional

from defusedxml import ElementTree as DET
from defusedxml.common import DefusedXmlException
from fastapi import HTTPException

from tldw_Server_API.app.core.Chunking import improved_chunking_process
from tldw_Server_API.app.core.Utils.Utils import logger
from tldw_Server_API.app.services._placeholder_guard import ensure_placeholder_service_enabled


async def process_xml_task(
    file_bytes: bytes,
    filename: str,
    title: Optional[str],
    author: Optional[str],
    keywords: list[str],
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
    ensure_placeholder_service_enabled("XML")

    tmp_path: Optional[str] = None
    try:
        logger.info(f"Processing XML file: {filename}")

        # 1) Save the incoming bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_bytes)

        # 2) Parse the XML with built-in logic
        try:
            tree = DET.parse(tmp_path)
            root = tree.getroot()
        except (DET.ParseError, DefusedXmlException) as e:
            logger.warning("Invalid XML input")
            raise HTTPException(status_code=400, detail="Invalid XML") from e

        # 3) Chunk the XML. For instance:
        chunk_options = {
            'method': 'xml',
            'max_size': 1000,
            'overlap': 200,
            'language': 'english'
        }
        # Convert root to string and chunk using xml method
        xml_string = DET.tostring(root, encoding='unicode')
        chunk_options['method'] = 'xml'
        chunks = improved_chunking_process(xml_string, chunk_options)

        # 4) Summarization
        summary_text = "No summary provided"
        if auto_summarize and api_name and api_name.lower() != "none" and api_key:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing XML file: {filename} -> {e}")
        raise HTTPException(status_code=500, detail="Failed to process XML file") from e
    finally:
        if tmp_path:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError as cleanup_error:
                logger.debug(f"Failed to clean up XML temp file '{tmp_path}': {cleanup_error}")
