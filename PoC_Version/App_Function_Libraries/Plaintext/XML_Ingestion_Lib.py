# XML_Ingestion.py
# Description: This file contains functions for reading and writing XML files.
# Imports
import xml.etree.ElementTree as ET
#
# External Imports
#
# Local Imports
from PoC_Version.App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from PoC_Version.App_Function_Libraries.Chunk_Lib import chunk_xml
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_to_database
from PoC_Version.App_Function_Libraries.Utils.Utils import logging
#
#######################################################################################################################
#
# Functions:

def xml_to_text(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Extract text content recursively
        text_content = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_content.append(elem.text.strip())
        return '\n'.join(text_content)
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file: {str(e)}")
        return None


def import_xml_handler(import_file, title, author, keywords, system_prompt,
                       custom_prompt, auto_summarize, api_name, api_key):
    if not import_file:
        return "Please upload an XML file"

    try:
        # Parse XML and extract text with structure
        tree = ET.parse(import_file.name)
        root = tree.getroot()

        # Create chunk options
        chunk_options = {
            'method': 'xml',
            'max_size': 1000,  # Adjust as needed
            'overlap': 200,  # Adjust as needed
            'language': 'english'  # Add language detection if needed
        }

        # Use the chunk_xml function to get structured chunks
        chunks = chunk_xml(ET.tostring(root, encoding='unicode'), chunk_options)

        # Convert chunks to segments format expected by add_media_to_database
        segments = []
        for chunk in chunks:
            segment = {
                'Text': chunk['text'],
                'metadata': chunk['metadata']  # Preserve XML structure metadata
            }
            segments.append(segment)

        # Create info_dict
        info_dict = {
            'title': title or 'Untitled XML Document',
            'uploader': author or 'Unknown',
            'file_type': 'xml',
            'structure': root.tag  # Save root element type
        }

        # Process keywords
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()] if keywords else []

        # Handle summarization
        if auto_summarize and api_name and api_key:
            # Combine all chunks for summarization
            full_text = '\n'.join(chunk['text'] for chunk in chunks)
            summary = perform_summarization(api_name, full_text, custom_prompt, api_key)
        else:
            summary = "No summary provided"

        # Add to database
        result = add_media_to_database(
            url=import_file.name,  # Using filename as URL
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=custom_prompt,
            whisper_model="XML Import",
            media_type="xml_document",
            overwrite=False
        )

        return f"XML file '{import_file.name}' import complete. Database result: {result}"

    except ET.ParseError as e:
        logging.error(f"XML parsing error: {str(e)}")
        return f"Error parsing XML file: {str(e)}"
    except Exception as e:
        logging.error(f"Error processing XML file: {str(e)}")
        return f"Error processing XML file: {str(e)}"

#
# End of XML_Ingestion_Lib.py
#######################################################################################################################
