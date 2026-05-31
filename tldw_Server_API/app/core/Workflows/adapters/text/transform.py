"""Text transformation adapters.

This module includes adapters for text transformation operations:
- json_transform: Transform JSON data
- json_validate: Validate JSON data
- xml_transform: Transform XML data
- template_render: Render Jinja templates
- regex_extract: Extract with regex patterns
- text_clean: Clean and normalize text
"""

from __future__ import annotations

import contextlib
import html
import re
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.TTS.utils import clean_text_for_tts
from tldw_Server_API.app.core.Workflows.adapters._common import resolve_workflow_file_path
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.text._config import (
    JSONTransformConfig,
    JSONValidateConfig,
    RegexExtractConfig,
    TemplateRenderConfig,
    TextCleanConfig,
    XMLTransformConfig,
)


@registry.register(
    "json_transform",
    category="text",
    description="Transform JSON data",
    parallelizable=True,
    tags=["text", "json"],
    config_model=JSONTransformConfig,
)
async def run_json_transform_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Transform JSON data using jq-like expressions or mappings.

    Config:
      - json: dict | list (templated) - JSON data to transform
      - expression: str (optional) - jq-like expression
      - mapping: dict (optional) - Field mapping rules
    Output:
      - {"result": Any, "text": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    data = config.get("data")
    if data is None:
        prev = context.get("prev") or context.get("last") or {}
        data = prev if isinstance(prev, (dict, list)) else {}

    expression = config.get("expression") or config.get("query") or "."

    try:
        import jmespath
        result = jmespath.search(expression, data)
        return {"result": result, "expression": expression}
    except ImportError:
        # Fallback: simple path extraction
        if expression == ".":
            return {"result": data, "expression": expression}
        parts = re.findall(r"[^\.\[\]]+|\[\d+\]", expression.strip("."))
        result = data
        for part in parts:
            if part.startswith("[") and part.endswith("]"):
                idx_str = part[1:-1]
                if isinstance(result, list) and idx_str.isdigit():
                    idx = int(idx_str)
                    result = result[idx] if 0 <= idx < len(result) else None
                else:
                    result = None
                    break
                continue
            if isinstance(result, dict):
                result = result.get(part)
            elif isinstance(result, list) and part.isdigit():
                idx = int(part)
                result = result[idx] if 0 <= idx < len(result) else None
            else:
                result = None
                break
        return {"result": result, "expression": expression}
    except Exception:
        logger.exception("JSON transform error")
        return {"error": "json_transform_error", "result": None}


@registry.register(
    "json_validate",
    category="text",
    description="Validate JSON data",
    parallelizable=True,
    tags=["text", "json"],
    config_model=JSONValidateConfig,
)
async def run_json_validate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Validate JSON data against a schema.

    Config:
      - json: Any (templated) - JSON data to validate
      - schema: dict (optional) - JSON Schema to validate against
      - strict: bool = False - Strict validation mode
    Output:
      - {"valid": bool, "errors": [str]}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    data = config.get("data")
    if data is None:
        prev = context.get("prev") or context.get("last") or {}
        data = prev if isinstance(prev, (dict, list)) else {}

    schema = config.get("schema")
    if not schema:
        return {"error": "missing_schema", "valid": False, "errors": []}

    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return {"valid": True, "errors": []}
    except ImportError:
        return {"error": "jsonschema_not_installed", "valid": False, "errors": ["Install jsonschema package"]}
    except jsonschema.ValidationError as e:
        return {"valid": False, "errors": [str(e.message)], "path": list(e.path)}
    except Exception:
        logger.exception("JSON validate error")
        return {"error": "json_validate_error", "valid": False, "errors": ["json_validate_error"]}


@registry.register(
    "xml_transform",
    category="text",
    description="Transform XML data",
    parallelizable=True,
    tags=["text", "xml"],
    config_model=XMLTransformConfig,
)
async def run_xml_transform_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Transform XML data using XPath or XSLT.

    Config:
      - xml: str (templated) - XML content to transform
      - xpath: str (optional) - XPath expression
      - xslt: str (optional) - XSLT stylesheet
      - output_format: Literal["xml", "json", "text"] = "xml"
    Output:
      - {"result": Any, "text": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    xml_data = config.get("xml") or config.get("data") or ""
    if isinstance(xml_data, str):
        xml_data = _tmpl(xml_data, context) or xml_data

    if not xml_data:
        prev = context.get("prev") or context.get("last") or {}
        xml_data = prev.get("xml") or prev.get("text") or "" if isinstance(prev, dict) else ""

    xpath = config.get("xpath") or config.get("query")
    if not xpath:
        return {"error": "missing_xpath", "results": []}

    try:
        from lxml import etree
        root = etree.fromstring(xml_data.encode() if isinstance(xml_data, str) else xml_data)
        results = root.xpath(xpath)
        output = []
        for r in results:
            if hasattr(r, 'text'):
                output.append({"tag": r.tag, "text": r.text, "attrib": dict(r.attrib)})
            else:
                output.append(str(r))
        return {"results": output, "count": len(output), "xpath": xpath}
    except ImportError:
        return {"error": "lxml_not_installed", "results": []}
    except Exception:
        logger.exception("XML transform error")
        return {"error": "xml_transform_error", "results": []}


@registry.register(
    "template_render",
    category="text",
    description="Render Jinja templates",
    parallelizable=True,
    tags=["text", "template"],
    config_model=TemplateRenderConfig,
)
async def run_template_render_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Render Jinja2 templates with context data.

    Config:
      - template: str (templated) - Jinja2 template string
      - variables: dict (optional) - Template variables
      - strict: bool = False - Fail on undefined variables
    Output:
      - {"rendered": str, "text": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    template = config.get("template") or ""
    template_file = config.get("template_file")

    if template_file:
        try:
            file_path = resolve_workflow_file_path(template_file, context, config)
            template = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            return {"error": "template_file_error", "text": ""}

    if not template:
        return {"error": "missing_template", "text": ""}

    variables = config.get("variables") or {}
    # Merge context inputs
    render_context = {**context.get("inputs", {}), **variables}
    render_context["prev"] = context.get("prev") or context.get("last") or {}

    try:
        rendered = _tmpl(template, render_context) or template
        return {"text": rendered}
    except Exception:
        logger.exception("Template render error")
        return {"error": "template_render_error", "text": ""}


@registry.register(
    "regex_extract",
    category="text",
    description="Extract with regex patterns",
    parallelizable=True,
    tags=["text", "extraction"],
    config_model=RegexExtractConfig,
)
async def run_regex_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract data using regular expressions.

    Config:
      - text: str (templated) - Text to search
      - pattern: str - Regular expression pattern
      - group: int = 0 - Capture group to return
      - all_matches: bool = False - Return all matches
      - flags: str (optional) - Regex flags (i, m, s)
    Output:
      - {"matches": [str], "count": int, "text": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text
    text = str(text).strip()

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        text = prev.get("text") or prev.get("content") or "" if isinstance(prev, dict) else ""

    pattern = config.get("pattern")
    if not pattern:
        return {"error": "missing_pattern", "matches": [], "count": 0}

    flags = 0
    if config.get("ignore_case"):
        flags |= re.IGNORECASE
    if config.get("multiline"):
        flags |= re.MULTILINE
    if config.get("dotall"):
        flags |= re.DOTALL

    try:
        regex = re.compile(pattern, flags)
        matches = []
        for match in regex.finditer(text):
            m = {"full": match.group(0), "start": match.start(), "end": match.end()}
            if match.groupdict():
                m["groups"] = match.groupdict()
            elif match.groups():
                m["groups"] = list(match.groups())
            matches.append(m)

        return {"matches": matches, "count": len(matches), "pattern": pattern}
    except re.error:
        return {"error": "invalid_pattern", "matches": [], "count": 0}
    except Exception:
        logger.exception("Regex extract error")
        return {"error": "regex_extract_error", "matches": [], "count": 0}


@registry.register(
    "text_clean",
    category="text",
    description="Clean and normalize text",
    parallelizable=True,
    tags=["text", "cleaning"],
    config_model=TextCleanConfig,
)
async def run_text_clean_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Clean and normalize text content.

    Config:
      - text: str (templated) - Text to clean
      - lowercase: bool = False - Convert to lowercase
      - strip_html: bool = True - Remove HTML tags
      - normalize_whitespace: bool = True - Normalize whitespace
      - remove_punctuation: bool = False - Remove punctuation
      - remove_numbers: bool = False - Remove numbers
      - remove_urls: bool = False - Remove URLs
    Output:
      - {"text": str, "original_length": int, "cleaned_length": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    text = config.get("text") or ""
    if isinstance(text, str):
        text = _tmpl(text, context) or text
    text = str(text)

    if not text:
        prev = context.get("prev") or context.get("last") or {}
        text = prev.get("text") or prev.get("content") or "" if isinstance(prev, dict) else ""

    operations = config.get("operations", ["strip_html", "normalize_whitespace", "fix_encoding"])

    original_len = len(text)

    if "strip_html" in operations:
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)

    if "fix_encoding" in operations:
        with contextlib.suppress(Exception):
            text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # strip_markdown MUST run before normalize_whitespace so that
    # re.MULTILINE patterns (^-anchored headers, list markers, blockquotes)
    # can match at actual line boundaries.
    if "strip_markdown" in operations:
        # Headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'[*_](.+?)[*_]', r'\1', text)
        # Links -> text only
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Images
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        # Code blocks and inline code
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # List markers
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # Blockquotes and horizontal rules
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)

    if "strip_reasoning_blocks" in operations:
        text = re.sub(
            r"<(?:think|thinking|reasoning)>[\s\S]*?</(?:think|thinking|reasoning)>\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"</?(?:think|thinking|reasoning)>", "", text, flags=re.IGNORECASE)

    if "tts_normalize" in operations:
        text = clean_text_for_tts(text)

    if "normalize_whitespace" in operations:
        text = re.sub(r'\s+', ' ', text)

    if "strip" in operations or "normalize_whitespace" in operations:
        text = text.strip()

    if "lowercase" in operations:
        text = text.lower()

    if "remove_urls" in operations:
        text = re.sub(r'https?://\S+', '', text)

    if "remove_emails" in operations:
        text = re.sub(r'\S+@\S+\.\S+', '', text)

    if "normalize_unicode" in operations:
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        text = text.replace('\u2026', '...')

    return {"text": text, "original_length": original_len, "cleaned_length": len(text), "operations": operations}
