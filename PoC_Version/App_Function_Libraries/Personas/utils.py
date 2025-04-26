# utils.py
import base64
import json
import re
from typing import Any, Dict, List, Optional
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from PIL import Image, PngImagePlugin


def decode_base64(data: str) -> bytes:
    """Decodes a Base64 encoded string."""
    try:
        return base64.b64decode(data)
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid Base64 data: {e}")


def extract_text_chunks_from_png(png_bytes: bytes) -> Dict[str, str]:
    """Extracts tEXt chunks from a PNG/APNG file."""
    try:
        with Image.open(BytesIO(png_bytes)) as img:
            info = img.info
            return info
    except Exception as e:
        raise ValueError(f"Failed to extract text chunks: {e}")


def extract_json_from_charx(charx_bytes: bytes) -> Dict[str, Any]:
    """Extracts and parses card.json from a CHARX file."""
    try:
        with ZipFile(BytesIO(charx_bytes)) as zip_file:
            if 'card.json' not in zip_file.namelist():
                raise ValueError("CHARX file does not contain card.json")
            with zip_file.open('card.json') as json_file:
                return json.load(json_file)
    except BadZipFile:
        raise ValueError("Invalid CHARX file: Not a valid zip archive")
    except Exception as e:
        raise ValueError(f"Failed to extract JSON from CHARX: {e}")


def parse_json_file(json_bytes: bytes) -> Dict[str, Any]:
    """Parses a JSON byte stream."""
    try:
        return json.loads(json_bytes.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")


def validate_iso_639_1(code: str) -> bool:
    """Validates if the code is a valid ISO 639-1 language code."""
    # For brevity, a small subset of ISO 639-1 codes
    valid_codes = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
        # Add more as needed
    }
    return code in valid_codes


def parse_uri(uri: str) -> Dict[str, Any]:
    """Parses the URI field and categorizes its type."""
    if uri.startswith('http://') or uri.startswith('https://'):
        return {'scheme': 'http', 'value': uri}
    elif uri.startswith('embeded://'):
        return {'scheme': 'embeded', 'value': uri.replace('embeded://', '')}
    elif uri.startswith('ccdefault:'):
        return {'scheme': 'ccdefault', 'value': None}
    elif uri.startswith('data:'):
        return {'scheme': 'data', 'value': uri}
    else:
        return {'scheme': 'unknown', 'value': uri}