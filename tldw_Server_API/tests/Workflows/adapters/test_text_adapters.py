"""Comprehensive tests for text workflow adapters.

This module tests all 16 text adapters:
- Conversion: html_to_markdown, markdown_to_html, csv_to_json, json_to_csv
- Transform: json_transform, json_validate, xml_transform, template_render, regex_extract, text_clean
- NLP: keyword_extract, sentiment_analyze, language_detect, topic_model, entity_extract, token_count
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# =============================================================================
# Conversion Adapters Tests
# =============================================================================


class TestHTMLToMarkdownAdapter:
    """Tests for run_html_to_markdown_adapter."""

    @pytest.mark.asyncio
    async def test_html_to_markdown_basic(self):
        """Test basic HTML to Markdown conversion."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {"html": "<h1>Title</h1><p>This is a paragraph.</p>"}
        context = {}

        result = await run_html_to_markdown_adapter(config, context)

        assert "markdown" in result
        assert "Title" in result["markdown"]
        # Check that HTML tags were converted
        assert "<h1>" not in result["markdown"]

    @pytest.mark.asyncio
    async def test_html_to_markdown_with_links(self):
        """Test HTML to Markdown with hyperlinks."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {"html": '<p>Visit <a href="https://example.com">Example</a></p>'}
        context = {}

        result = await run_html_to_markdown_adapter(config, context)

        assert "markdown" in result
        # Link should be preserved in some form
        assert "Example" in result["markdown"]

    @pytest.mark.asyncio
    async def test_html_to_markdown_strips_script_tags(self):
        """Test that script tags are stripped."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {"html": "<p>Text</p><script>alert('evil');</script>"}
        context = {}

        result = await run_html_to_markdown_adapter(config, context)

        assert "markdown" in result
        assert "Text" in result["markdown"]
        assert "alert" not in result["markdown"]
        assert "<script>" not in result["markdown"]

    @pytest.mark.asyncio
    async def test_html_to_markdown_strips_style_tags(self):
        """Test that style tags are stripped."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {"html": "<style>.cls{color:red}</style><p>Content</p>"}
        context = {}

        result = await run_html_to_markdown_adapter(config, context)

        assert "markdown" in result
        assert "Content" in result["markdown"]
        assert "color:red" not in result["markdown"]

    @pytest.mark.asyncio
    async def test_html_to_markdown_with_text_key(self):
        """Test HTML provided via 'text' key."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {"text": "<strong>Bold text</strong>"}
        context = {}

        result = await run_html_to_markdown_adapter(config, context)

        assert "markdown" in result
        assert "Bold text" in result["markdown"]

    @pytest.mark.asyncio
    async def test_html_to_markdown_from_previous_context(self):
        """Test HTML from previous step context."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {}
        context = {"prev": {"html": "<em>Emphasized</em>"}}

        result = await run_html_to_markdown_adapter(config, context)

        assert "markdown" in result
        assert "Emphasized" in result["markdown"]

    @pytest.mark.asyncio
    async def test_html_to_markdown_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_html_to_markdown_adapter

        config = {"html": "<p>Test</p>"}
        context = {"is_cancelled": lambda: True}

        result = await run_html_to_markdown_adapter(config, context)

        assert result.get("__status__") == "cancelled"


class TestMarkdownToHTMLAdapter:
    """Tests for run_markdown_to_html_adapter."""

    @pytest.mark.asyncio
    async def test_markdown_to_html_basic(self):
        """Test basic Markdown to HTML conversion."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {"markdown": "# Title\n\nThis is a paragraph."}
        context = {}

        result = await run_markdown_to_html_adapter(config, context)

        assert "html" in result
        assert "Title" in result["html"]
        # Should contain HTML tags
        assert "<" in result["html"]

    @pytest.mark.asyncio
    async def test_markdown_to_html_with_links(self):
        """Test Markdown to HTML with links."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {"markdown": "Visit [Example](https://example.com)"}
        context = {}

        result = await run_markdown_to_html_adapter(config, context)

        assert "html" in result
        # Link should be converted
        assert "Example" in result["html"]

    @pytest.mark.asyncio
    async def test_markdown_to_html_with_code_block(self):
        """Test Markdown to HTML with code block."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {"markdown": "```python\nprint('hello')\n```"}
        context = {}

        result = await run_markdown_to_html_adapter(config, context)

        assert "html" in result
        assert "print" in result["html"]

    @pytest.mark.asyncio
    async def test_markdown_to_html_with_text_key(self):
        """Test Markdown provided via 'text' key."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {"text": "**Bold** and *italic*"}
        context = {}

        result = await run_markdown_to_html_adapter(config, context)

        assert "html" in result
        assert "Bold" in result["html"]
        assert "italic" in result["html"]

    @pytest.mark.asyncio
    async def test_markdown_to_html_with_custom_extensions(self):
        """Test Markdown with custom extensions."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {
            "markdown": "| Header |\n|--------|\n| Cell   |",
            "extensions": ["tables"],
        }
        context = {}

        result = await run_markdown_to_html_adapter(config, context)

        assert "html" in result
        assert "Header" in result["html"]

    @pytest.mark.asyncio
    async def test_markdown_to_html_from_previous_context(self):
        """Test Markdown from previous step context."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {}
        context = {"prev": {"markdown": "## Subtitle"}}

        result = await run_markdown_to_html_adapter(config, context)

        assert "html" in result
        assert "Subtitle" in result["html"]

    @pytest.mark.asyncio
    async def test_markdown_to_html_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_markdown_to_html_adapter

        config = {"markdown": "# Test"}
        context = {"is_cancelled": lambda: True}

        result = await run_markdown_to_html_adapter(config, context)

        assert result.get("__status__") == "cancelled"


class TestCSVToJSONAdapter:
    """Tests for run_csv_to_json_adapter."""

    @pytest.mark.asyncio
    async def test_csv_to_json_basic(self):
        """Test basic CSV to JSON conversion."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        config = {"csv_data": "name,age\nAlice,30\nBob,25"}
        context = {}

        result = await run_csv_to_json_adapter(config, context)

        assert "records" in result
        assert len(result["records"]) == 2
        assert result["records"][0]["name"] == "Alice"
        assert result["records"][0]["age"] == "30"
        assert result["records"][1]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_csv_to_json_custom_delimiter(self):
        """Test CSV with custom delimiter."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        config = {"csv_data": "name;age\nAlice;30", "delimiter": ";"}
        context = {}

        result = await run_csv_to_json_adapter(config, context)

        assert "records" in result
        assert len(result["records"]) == 1
        assert result["records"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_csv_to_json_no_header(self):
        """Test CSV without header row."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        config = {"csv_data": "Alice,30\nBob,25", "has_header": False}
        context = {}

        result = await run_csv_to_json_adapter(config, context)

        assert "records" in result
        assert len(result["records"]) == 2
        # Should use auto-generated column names
        assert "col_0" in result["records"][0]
        assert "col_1" in result["records"][0]

    @pytest.mark.asyncio
    async def test_csv_to_json_empty_data(self):
        """Test CSV with empty data."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        config = {"csv_data": ""}
        context = {}

        result = await run_csv_to_json_adapter(config, context)

        assert "error" in result or result.get("count", 0) == 0

    @pytest.mark.asyncio
    async def test_csv_to_json_with_data_key(self):
        """Test CSV provided via 'data' key."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        config = {"data": "id,value\n1,test"}
        context = {}

        result = await run_csv_to_json_adapter(config, context)

        assert "records" in result
        assert len(result["records"]) == 1

    @pytest.mark.asyncio
    async def test_csv_to_json_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        config = {"csv_data": "a,b\n1,2"}
        context = {"is_cancelled": lambda: True}

        result = await run_csv_to_json_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_csv_to_json_sanitizes_backend_errors(self, monkeypatch):
        """Test CSV conversion hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_csv_to_json_adapter

        def raise_reader(*args, **kwargs):
            raise RuntimeError("csv backend exploded at /private/csv-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.text.conversion.csv.reader",
            raise_reader,
        )

        result = await run_csv_to_json_adapter({"csv_data": "a,b\n1,2"}, {})

        assert result == {"error": "csv_to_json_error", "records": [], "count": 0}


class TestJSONToCSVAdapter:
    """Tests for run_json_to_csv_adapter."""

    @pytest.mark.asyncio
    async def test_json_to_csv_basic(self):
        """Test basic JSON to CSV conversion."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        config = {"records": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        context = {}

        result = await run_json_to_csv_adapter(config, context)

        assert "csv" in result
        assert "name" in result["csv"]
        assert "Alice" in result["csv"]
        assert "Bob" in result["csv"]
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_json_to_csv_custom_delimiter(self):
        """Test JSON to CSV with custom delimiter."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        config = {"records": [{"a": "1", "b": "2"}], "delimiter": ";"}
        context = {}

        result = await run_json_to_csv_adapter(config, context)

        assert "csv" in result
        assert ";" in result["csv"]

    @pytest.mark.asyncio
    async def test_json_to_csv_no_header(self):
        """Test JSON to CSV without header."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        config = {"records": [{"name": "Test"}], "include_header": False}
        context = {}

        result = await run_json_to_csv_adapter(config, context)

        assert "csv" in result
        lines = result["csv"].strip().split("\n")
        # Should not have header row
        assert "Test" in lines[0]

    @pytest.mark.asyncio
    async def test_json_to_csv_empty_records(self):
        """Test JSON to CSV with empty records."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        config = {"records": []}
        context = {}

        result = await run_json_to_csv_adapter(config, context)

        assert "error" in result or result.get("count", 0) == 0

    @pytest.mark.asyncio
    async def test_json_to_csv_from_previous_context(self):
        """Test JSON from previous step context."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        config = {}
        context = {"prev": {"records": [{"x": "1"}]}}

        result = await run_json_to_csv_adapter(config, context)

        assert "csv" in result
        assert "1" in result["csv"]

    @pytest.mark.asyncio
    async def test_json_to_csv_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        config = {"records": [{"a": "b"}]}
        context = {"is_cancelled": lambda: True}

        result = await run_json_to_csv_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_json_to_csv_sanitizes_backend_errors(self, monkeypatch):
        """Test JSON conversion hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_to_csv_adapter

        class ExplodingDictWriter:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("csv writer exploded at /private/json-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.text.conversion.csv.DictWriter",
            ExplodingDictWriter,
        )

        result = await run_json_to_csv_adapter({"records": [{"a": "b"}]}, {})

        assert result == {"error": "json_to_csv_error", "csv": "", "count": 0}


# =============================================================================
# Transform Adapters Tests
# =============================================================================


class TestJSONTransformAdapter:
    """Tests for run_json_transform_adapter."""

    @pytest.mark.asyncio
    async def test_json_transform_identity(self):
        """Test JSON transform with identity expression."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_transform_adapter

        config = {"data": {"name": "test", "value": 123}, "expression": "."}
        context = {}

        result = await run_json_transform_adapter(config, context)

        assert "result" in result
        assert result["result"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_json_transform_field_access(self):
        """Test JSON transform with field access."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_transform_adapter

        config = {"data": {"user": {"name": "Alice", "email": "alice@test.com"}}, "expression": "user.name"}
        context = {}

        result = await run_json_transform_adapter(config, context)

        assert "result" in result
        assert result["result"] == "Alice"

    @pytest.mark.asyncio
    async def test_json_transform_array_access(self):
        """Test JSON transform with array access."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_transform_adapter

        config = {"data": {"items": [1, 2, 3]}, "expression": "items[0]"}
        context = {}

        result = await run_json_transform_adapter(config, context)

        assert "result" in result
        assert result["result"] == 1

    @pytest.mark.asyncio
    async def test_json_transform_from_previous_context(self):
        """Test JSON from previous step context."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_transform_adapter

        config = {"expression": "value"}
        context = {"prev": {"value": 42}}

        result = await run_json_transform_adapter(config, context)

        assert "result" in result
        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_json_transform_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_transform_adapter

        config = {"data": {}, "expression": "."}
        context = {"is_cancelled": lambda: True}

        result = await run_json_transform_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_json_transform_sanitizes_backend_errors(self, monkeypatch):
        """Test JSON transform hides backend exception details."""
        import sys
        import types

        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_transform_adapter

        def raise_search(*args, **kwargs):
            raise RuntimeError("jmespath backend exploded at /private/json-transform")

        monkeypatch.setitem(
            sys.modules,
            "jmespath",
            types.SimpleNamespace(search=raise_search),
        )

        result = await run_json_transform_adapter({"data": {"a": 1}, "expression": "a"}, {})

        assert result == {"error": "json_transform_error", "result": None}


class TestJSONValidateAdapter:
    """Tests for run_json_validate_adapter."""

    @pytest.mark.asyncio
    async def test_json_validate_valid(self):
        """Test JSON validation with valid data."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_validate_adapter

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        config = {"data": {"name": "Alice", "age": 30}, "schema": schema}
        context = {}

        result = await run_json_validate_adapter(config, context)

        # May return valid=True or error depending on jsonschema availability
        if "error" not in result or result["error"] != "jsonschema_not_installed":
            assert result.get("valid") is True
            assert result.get("errors") == []

    @pytest.mark.asyncio
    async def test_json_validate_invalid(self):
        """Test JSON validation with invalid data."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_validate_adapter

        schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
        config = {"data": {"age": "not_an_integer"}, "schema": schema}
        context = {}

        result = await run_json_validate_adapter(config, context)

        # May return valid=False or error depending on jsonschema availability
        if "error" not in result or result["error"] != "jsonschema_not_installed":
            assert result.get("valid") is False
            assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    async def test_json_validate_missing_schema(self):
        """Test JSON validation without schema."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_validate_adapter

        config = {"data": {"name": "test"}}
        context = {}

        result = await run_json_validate_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_schema"

    @pytest.mark.asyncio
    async def test_json_validate_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_validate_adapter

        config = {"data": {}, "schema": {"type": "object"}}
        context = {"is_cancelled": lambda: True}

        result = await run_json_validate_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_json_validate_sanitizes_backend_errors(self, monkeypatch):
        """Test JSON validate hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_json_validate_adapter

        jsonschema = pytest.importorskip("jsonschema")
        monkeypatch.setattr(
            jsonschema,
            "validate",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("jsonschema backend exploded at /private/json-schema")
            ),
        )

        result = await run_json_validate_adapter(
            {"data": {"a": 1}, "schema": {"type": "object"}},
            {},
        )

        assert result == {"error": "json_validate_error", "valid": False, "errors": ["json_validate_error"]}


class TestXMLTransformAdapter:
    """Tests for run_xml_transform_adapter."""

    @pytest.mark.asyncio
    async def test_xml_transform_xpath(self):
        """Test XML transform with XPath."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_xml_transform_adapter

        xml_data = "<root><item>Value1</item><item>Value2</item></root>"
        config = {"xml": xml_data, "xpath": "//item"}
        context = {}

        result = await run_xml_transform_adapter(config, context)

        # May return error if lxml not installed
        if "error" not in result or result["error"] != "lxml_not_installed":
            assert "results" in result
            assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_xml_transform_with_attributes(self):
        """Test XML transform extracting attributes."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_xml_transform_adapter

        xml_data = '<root><item id="1">First</item><item id="2">Second</item></root>'
        config = {"xml": xml_data, "xpath": "//item[@id='1']"}
        context = {}

        result = await run_xml_transform_adapter(config, context)

        if "error" not in result or result["error"] != "lxml_not_installed":
            assert "results" in result
            assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_xml_transform_missing_xpath(self):
        """Test XML transform without XPath."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_xml_transform_adapter

        config = {"xml": "<root/>"}
        context = {}

        result = await run_xml_transform_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_xpath"

    @pytest.mark.asyncio
    async def test_xml_transform_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_xml_transform_adapter

        config = {"xml": "<root/>", "xpath": "//item"}
        context = {"is_cancelled": lambda: True}

        result = await run_xml_transform_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_xml_transform_sanitizes_backend_errors(self, monkeypatch):
        """Test XML transform hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_xml_transform_adapter

        etree = pytest.importorskip("lxml.etree")
        monkeypatch.setattr(
            etree,
            "fromstring",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("xml backend exploded at /private/xml-transform")
            ),
        )

        result = await run_xml_transform_adapter({"xml": "<root/>", "xpath": "//item"}, {})

        assert result == {"error": "xml_transform_error", "results": []}


class TestTemplateRenderAdapter:
    """Tests for run_template_render_adapter."""

    @pytest.mark.asyncio
    async def test_template_render_basic(self):
        """Test basic template rendering."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        config = {"template": "Hello, {{ name }}!", "variables": {"name": "World"}}
        context = {}

        result = await run_template_render_adapter(config, context)

        assert "text" in result
        assert "Hello, World!" in result["text"]

    @pytest.mark.asyncio
    async def test_template_render_with_loop(self):
        """Test template rendering with loop."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        config = {
            "template": "{% for item in items %}{{ item }},{% endfor %}",
            "variables": {"items": ["a", "b", "c"]},
        }
        context = {}

        result = await run_template_render_adapter(config, context)

        assert "text" in result
        assert "a" in result["text"]
        assert "b" in result["text"]
        assert "c" in result["text"]

    @pytest.mark.asyncio
    async def test_template_render_with_conditional(self):
        """Test template rendering with conditional."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        config = {
            "template": "{% if show %}Visible{% else %}Hidden{% endif %}",
            "variables": {"show": True},
        }
        context = {}

        result = await run_template_render_adapter(config, context)

        assert "text" in result
        assert "Visible" in result["text"]

    @pytest.mark.asyncio
    async def test_template_render_uses_context_inputs(self):
        """Test template uses context inputs."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        config = {"template": "Value: {{ value }}"}
        context = {"inputs": {"value": "from_context"}}

        result = await run_template_render_adapter(config, context)

        assert "text" in result
        assert "from_context" in result["text"]

    @pytest.mark.asyncio
    async def test_template_render_missing_template(self):
        """Test template rendering without template."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        config = {}
        context = {}

        result = await run_template_render_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_template"

    @pytest.mark.asyncio
    async def test_template_render_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        config = {"template": "test"}
        context = {"is_cancelled": lambda: True}

        result = await run_template_render_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_template_render_sanitizes_template_file_errors(self, monkeypatch):
        """Test template file errors hide path resolution details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.text.transform.resolve_workflow_file_path",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("template file denied at /private/template-path")
            ),
        )

        result = await run_template_render_adapter({"template_file": "private-template.j2"}, {})

        assert result == {"error": "template_file_error", "text": ""}

    @pytest.mark.asyncio
    async def test_template_render_sanitizes_backend_errors(self, monkeypatch):
        """Test template rendering hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_template_render_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.prompt_template_manager.apply_template_to_string",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("template backend exploded at /private/template-cache")
            ),
        )

        result = await run_template_render_adapter({"template": "Hello {{ name }}"}, {})

        assert result == {"error": "template_render_error", "text": ""}


class TestRegexExtractAdapter:
    """Tests for run_regex_extract_adapter."""

    @pytest.mark.asyncio
    async def test_regex_extract_email(self):
        """Test regex extraction of email."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "Contact us at info@example.com for more details.", "pattern": r"[\w.]+@[\w.]+"}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert "matches" in result
        assert result["count"] >= 1
        assert any("info@example.com" in m.get("full", "") for m in result["matches"])

    @pytest.mark.asyncio
    async def test_regex_extract_multiple_matches(self):
        """Test regex extraction with multiple matches."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "Numbers: 123, 456, 789", "pattern": r"\d+"}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert "matches" in result
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_regex_extract_with_groups(self):
        """Test regex extraction with capture groups."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "Name: John, Age: 30", "pattern": r"Name: (\w+), Age: (\d+)"}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert "matches" in result
        assert result["count"] == 1
        assert "groups" in result["matches"][0]

    @pytest.mark.asyncio
    async def test_regex_extract_case_insensitive(self):
        """Test case insensitive regex extraction."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "Hello WORLD", "pattern": r"hello", "ignore_case": True}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert "matches" in result
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_regex_extract_no_matches(self):
        """Test regex extraction with no matches."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "No numbers here", "pattern": r"\d+"}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert "matches" in result
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_regex_extract_missing_pattern(self):
        """Test regex extraction without pattern."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "Some text"}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_pattern"

    @pytest.mark.asyncio
    async def test_regex_extract_invalid_pattern(self):
        """Test regex extraction with invalid pattern."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "Some text", "pattern": r"[invalid"}
        context = {}

        result = await run_regex_extract_adapter(config, context)

        assert result == {"error": "invalid_pattern", "matches": [], "count": 0}

    @pytest.mark.asyncio
    async def test_regex_extract_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        config = {"text": "test", "pattern": r"\w+"}
        context = {"is_cancelled": lambda: True}

        result = await run_regex_extract_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_regex_extract_sanitizes_backend_errors(self, monkeypatch):
        """Test regex extraction hides backend exception details."""
        import re

        from tldw_Server_API.app.core.Workflows.adapters.text import run_regex_extract_adapter

        original_compile = re.compile

        def raise_for_adapter_pattern(pattern, flags=0):
            if pattern == r"\w+":
                raise RuntimeError("regex backend exploded at /private/regex-cache")
            return original_compile(pattern, flags)

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.text.transform.re.compile",
            raise_for_adapter_pattern,
        )

        result = await run_regex_extract_adapter({"text": "abc", "pattern": r"\w+"}, {})

        assert result == {"error": "regex_extract_error", "matches": [], "count": 0}


class TestTextCleanAdapter:
    """Tests for run_text_clean_adapter."""

    @pytest.mark.asyncio
    async def test_text_clean_strip_html(self):
        """Test text cleaning strips HTML."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "<p>Hello <b>World</b></p>", "operations": ["strip_html"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "<p>" not in result["text"]
        assert "<b>" not in result["text"]
        assert "Hello" in result["text"]
        assert "World" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_normalize_whitespace(self):
        """Test text cleaning normalizes whitespace."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "Multiple   spaces    here", "operations": ["normalize_whitespace"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "   " not in result["text"]
        assert "Multiple spaces here" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_lowercase(self):
        """Test text cleaning converts to lowercase."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "UPPERCASE TEXT", "operations": ["lowercase"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert result["text"] == "uppercase text"

    @pytest.mark.asyncio
    async def test_text_clean_remove_urls(self):
        """Test text cleaning removes URLs."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "Visit https://example.com for more", "operations": ["remove_urls"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "https://example.com" not in result["text"]
        assert "Visit" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_remove_emails(self):
        """Test text cleaning removes emails."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "Contact test@example.com today", "operations": ["remove_emails"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "test@example.com" not in result["text"]
        assert "Contact" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_multiple_operations(self):
        """Test text cleaning with multiple operations."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {
            "text": "<p>HELLO   WORLD</p>",
            "operations": ["strip_html", "normalize_whitespace", "lowercase"],
        }
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "<p>" not in result["text"]
        assert "hello world" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_reports_lengths(self):
        """Test text cleaning reports original and cleaned lengths."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "<div>Test</div>", "operations": ["strip_html"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "original_length" in result
        assert "cleaned_length" in result
        assert result["original_length"] > result["cleaned_length"]

    @pytest.mark.asyncio
    async def test_text_clean_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "test", "operations": ["strip_html"]}
        context = {"is_cancelled": lambda: True}

        result = await run_text_clean_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_headers(self):
        """Test strip_markdown removes markdown headers."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "# Title\n## Subtitle\n### H3\nPlain text", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert result["text"].startswith("Title")
        assert "# " not in result["text"]
        assert "## " not in result["text"]
        assert "### " not in result["text"]
        assert "Plain text" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_bold_italic(self):
        """Test strip_markdown removes bold and italic markers."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "This is **bold** and *italic* text", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "**" not in result["text"]
        assert "*" not in result["text"]
        assert "bold" in result["text"]
        assert "italic" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_links(self):
        """Test strip_markdown converts links to text only."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "Visit [Example](https://example.com) for more", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "Example" in result["text"]
        assert "https://example.com" not in result["text"]
        assert "[" not in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_code_blocks(self):
        """Test strip_markdown removes code blocks and inline code."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {
            "text": "Use `print()` or:\n```python\nprint('hi')\n```\nDone",
            "operations": ["strip_markdown"],
        }
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "```" not in result["text"]
        assert "`" not in result["text"]
        assert "print()" in result["text"]
        assert "Done" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_list_markers(self):
        """Test strip_markdown removes list markers."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "- Item one\n* Item two\n1. Item three", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "Item one" in result["text"]
        assert "Item two" in result["text"]
        assert "Item three" in result["text"]
        # Markers removed
        assert not any(line.startswith("- ") for line in result["text"].split("\n") if line.strip())

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_blockquotes(self):
        """Test strip_markdown removes blockquote markers."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "> This is a quote\n> Another line", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "This is a quote" in result["text"]
        assert "> " not in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_images(self):
        """Test strip_markdown removes image tags."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "See ![alt text](image.png) here", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "![" not in result["text"]
        assert "image.png" not in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_horizontal_rules(self):
        """Test strip_markdown removes horizontal rules."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "Above\n---\nBelow", "operations": ["strip_markdown"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "Above" in result["text"]
        assert "Below" in result["text"]
        assert "---" not in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_normalize_unicode_smart_quotes(self):
        """Test normalize_unicode converts smart quotes to ASCII."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "\u201cHello\u201d \u2018world\u2019", "operations": ["normalize_unicode"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert '"Hello"' in result["text"]
        assert "'world'" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_normalize_unicode_dashes(self):
        """Test normalize_unicode converts em/en dashes to hyphens."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "en\u2013dash and em\u2014dash", "operations": ["normalize_unicode"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "en-dash" in result["text"]
        assert "em-dash" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_normalize_unicode_ellipsis(self):
        """Test normalize_unicode converts ellipsis to three dots."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "Wait for it\u2026", "operations": ["normalize_unicode"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "Wait for it..." in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_combined_markdown_and_unicode(self):
        """Test strip_markdown and normalize_unicode chain correctly."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {
            "text": "## \u201cBreaking\u201d **News**\n- Story with em\u2014dash\u2026",
            "operations": ["strip_markdown", "normalize_unicode"],
        }
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "## " not in result["text"]
        assert "**" not in result["text"]
        assert '"Breaking"' in result["text"]
        assert "News" in result["text"]
        assert "em-dash..." in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_strip_markdown_empty_text(self):
        """Test strip_markdown handles empty text gracefully."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {"text": "", "operations": ["strip_markdown", "normalize_unicode"]}
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_text_clean_strip_reasoning_blocks(self):
        """Test strip_reasoning_blocks removes hidden reasoning segments."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {
            "text": "Visible <think>hidden</think> text <reasoning>internal</reasoning> done",
            "operations": ["strip_reasoning_blocks", "normalize_whitespace"],
        }
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "<think>" not in result["text"].lower()
        assert "<reasoning>" not in result["text"].lower()
        assert "hidden" not in result["text"].lower()
        assert "internal" not in result["text"].lower()
        assert "Visible" in result["text"]
        assert "done" in result["text"]

    @pytest.mark.asyncio
    async def test_text_clean_tts_normalize(self):
        """Test tts_normalize applies speech-friendly symbol cleanup."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_text_clean_adapter

        config = {
            "text": "Line 1\nLine 2 + A&B \u2014 done",
            "operations": ["tts_normalize"],
        }
        context = {}

        result = await run_text_clean_adapter(config, context)

        assert "text" in result
        assert "\n" not in result["text"]
        assert "plus" in result["text"]
        assert " and " in result["text"]
        assert "\u2014" not in result["text"]


# =============================================================================
# NLP Adapters Tests
# =============================================================================


class TestKeywordExtractAdapter:
    """Tests for run_keyword_extract_adapter."""

    @pytest.mark.asyncio
    async def test_keyword_extract_llm_method(self):
        """Test keyword extraction with LLM method."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_keyword_extract_adapter

        mock_response = {"choices": [{"message": {"content": "machine learning\nartificial intelligence\ndata science"}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "text": "Machine learning is a subset of artificial intelligence that focuses on data science.",
                "method": "llm",
                "max_keywords": 5,
            }
            context = {}

            result = await run_keyword_extract_adapter(config, context)

            assert "keywords" in result
            assert len(result["keywords"]) > 0

    @pytest.mark.asyncio
    async def test_keyword_extract_missing_text(self):
        """Test keyword extraction with missing text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_keyword_extract_adapter

        config = {"method": "llm"}
        context = {}

        result = await run_keyword_extract_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"

    @pytest.mark.asyncio
    async def test_keyword_extract_with_max_keywords(self):
        """Test keyword extraction with max_keywords limit."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_keyword_extract_adapter

        mock_response = {"choices": [{"message": {"content": "keyword1\nkeyword2\nkeyword3\nkeyword4\nkeyword5"}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "Sample text for keyword extraction.", "method": "llm", "max_keywords": 3}
            context = {}

            result = await run_keyword_extract_adapter(config, context)

            assert "keywords" in result
            assert len(result["keywords"]) <= 3

    @pytest.mark.asyncio
    async def test_keyword_extract_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_keyword_extract_adapter

        config = {"text": "test", "method": "llm"}
        context = {"is_cancelled": lambda: True}

        result = await run_keyword_extract_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_keyword_extract_sanitizes_backend_errors(self):
        """Test keyword extraction hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_keyword_extract_adapter

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("keyword backend exploded at /private/keyword-cache"),
        ):
            result = await run_keyword_extract_adapter({"text": "Sample text", "method": "llm"}, {})

        assert result == {"keywords": [], "scored_keywords": [], "error": "keyword_extract_error"}


class TestSentimentAnalyzeAdapter:
    """Tests for run_sentiment_analyze_adapter."""

    @pytest.mark.asyncio
    async def test_sentiment_analyze_positive(self):
        """Test sentiment analysis with positive text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        mock_response = {"choices": [{"message": {"content": '{"sentiment": "positive", "score": 0.8, "confidence": 0.9}'}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "I love this product! It's amazing and wonderful."}
            context = {}

            result = await run_sentiment_analyze_adapter(config, context)

            assert "sentiment" in result
            assert result["sentiment"] == "positive"
            assert result["score"] > 0

    @pytest.mark.asyncio
    async def test_sentiment_analyze_negative(self):
        """Test sentiment analysis with negative text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        mock_response = {"choices": [{"message": {"content": '{"sentiment": "negative", "score": -0.7, "confidence": 0.85}'}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "This is terrible and disappointing."}
            context = {}

            result = await run_sentiment_analyze_adapter(config, context)

            assert "sentiment" in result
            assert result["sentiment"] == "negative"
            assert result["score"] < 0

    @pytest.mark.asyncio
    async def test_sentiment_analyze_neutral(self):
        """Test sentiment analysis with neutral text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        mock_response = {"choices": [{"message": {"content": '{"sentiment": "neutral", "score": 0.0, "confidence": 0.7}'}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "The meeting is scheduled for tomorrow."}
            context = {}

            result = await run_sentiment_analyze_adapter(config, context)

            assert "sentiment" in result
            assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_sentiment_analyze_missing_text(self):
        """Test sentiment analysis with missing text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        config = {}
        context = {}

        result = await run_sentiment_analyze_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"
        assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_sentiment_analyze_fallback_parsing(self):
        """Test sentiment analysis with non-JSON response."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        mock_response = {"choices": [{"message": {"content": "The sentiment is positive overall."}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "Great work!"}
            context = {}

            result = await run_sentiment_analyze_adapter(config, context)

            assert "sentiment" in result
            # Should fallback to text parsing
            assert result["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_sentiment_analyze_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        config = {"text": "test"}
        context = {"is_cancelled": lambda: True}

        result = await run_sentiment_analyze_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_sentiment_analyze_sanitizes_backend_errors(self):
        """Test sentiment analysis hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_sentiment_analyze_adapter

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("sentiment backend exploded at /private/sentiment-cache"),
        ):
            result = await run_sentiment_analyze_adapter({"text": "Sample text"}, {})

        assert result == {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "error": "sentiment_analyze_error",
        }


class TestLanguageDetectAdapter:
    """Tests for run_language_detect_adapter."""

    @pytest.mark.asyncio
    async def test_language_detect_english(self):
        """Test language detection for English text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        config = {"text": "This is an English sentence for language detection."}
        context = {}

        result = await run_language_detect_adapter(config, context)

        # May return error if langdetect not installed
        if "error" not in result or result["error"] != "langdetect_not_installed":
            assert "language" in result
            assert result["language"] == "en"
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_language_detect_spanish(self):
        """Test language detection for Spanish text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        config = {"text": "Este es un texto en espanol para detectar el idioma."}
        context = {}

        result = await run_language_detect_adapter(config, context)

        if "error" not in result or result["error"] != "langdetect_not_installed":
            assert "language" in result
            assert result["language"] == "es"

    @pytest.mark.asyncio
    async def test_language_detect_french(self):
        """Test language detection for French text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        config = {"text": "Ceci est un texte en francais pour la detection de la langue."}
        context = {}

        result = await run_language_detect_adapter(config, context)

        if "error" not in result or result["error"] != "langdetect_not_installed":
            assert "language" in result
            assert result["language"] == "fr"

    @pytest.mark.asyncio
    async def test_language_detect_missing_text(self):
        """Test language detection with missing text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        config = {}
        context = {}

        result = await run_language_detect_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"
        assert result["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_language_detect_returns_language_name(self):
        """Test language detection returns language name."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        config = {"text": "This is English text."}
        context = {}

        result = await run_language_detect_adapter(config, context)

        if "error" not in result or result["error"] != "langdetect_not_installed":
            assert "language_name" in result
            assert result["language_name"] == "English"

    @pytest.mark.asyncio
    async def test_language_detect_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        config = {"text": "test"}
        context = {"is_cancelled": lambda: True}

        result = await run_language_detect_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_language_detect_sanitizes_backend_errors(self, monkeypatch):
        """Test language detection hides backend exception details."""
        import sys
        import types

        from tldw_Server_API.app.core.Workflows.adapters.text import run_language_detect_adapter

        def raise_detect(*args, **kwargs):
            raise RuntimeError("language backend exploded at /private/language-cache")

        monkeypatch.setitem(
            sys.modules,
            "langdetect",
            types.SimpleNamespace(detect=raise_detect, detect_langs=lambda *args, **kwargs: []),
        )

        result = await run_language_detect_adapter({"text": "This is sample text."}, {})

        assert result == {
            "language": "unknown",
            "language_name": "Unknown",
            "confidence": 0.0,
            "error": "language_detect_error",
        }


class TestTopicModelAdapter:
    """Tests for run_topic_model_adapter."""

    @pytest.mark.asyncio
    async def test_topic_model_basic(self):
        """Test basic topic modeling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_topic_model_adapter

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '[{"label": "Technology", "keywords": ["AI", "machine learning"]}, {"label": "Science", "keywords": ["research", "data"]}]'
                    }
                }
            ]
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {
                "text": "Artificial intelligence and machine learning are revolutionizing research and data science.",
                "num_topics": 3,
            }
            context = {}

            result = await run_topic_model_adapter(config, context)

            assert "topics" in result
            assert len(result["topics"]) > 0

    @pytest.mark.asyncio
    async def test_topic_model_respects_num_topics(self):
        """Test topic modeling respects num_topics limit."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_topic_model_adapter

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '[{"label": "Topic1", "keywords": []}, {"label": "Topic2", "keywords": []}, {"label": "Topic3", "keywords": []}]'
                    }
                }
            ]
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "Sample text for topic modeling.", "num_topics": 2}
            context = {}

            result = await run_topic_model_adapter(config, context)

            assert "topics" in result
            assert len(result["topics"]) <= 2

    @pytest.mark.asyncio
    async def test_topic_model_missing_text(self):
        """Test topic modeling with missing text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_topic_model_adapter

        config = {"num_topics": 3}
        context = {}

        result = await run_topic_model_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"

    @pytest.mark.asyncio
    async def test_topic_model_fallback_parsing(self):
        """Test topic modeling with non-JSON response."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_topic_model_adapter

        mock_response = {"choices": [{"message": {"content": "Topic 1: Technology\nTopic 2: Science"}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "Sample text", "num_topics": 2}
            context = {}

            result = await run_topic_model_adapter(config, context)

            assert "topics" in result
            # Should fallback to text parsing
            assert len(result["topics"]) > 0

    @pytest.mark.asyncio
    async def test_topic_model_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_topic_model_adapter

        config = {"text": "test", "num_topics": 3}
        context = {"is_cancelled": lambda: True}

        result = await run_topic_model_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_topic_model_sanitizes_backend_errors(self):
        """Test topic modeling hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_topic_model_adapter

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("topic backend exploded at /private/topic-cache"),
        ):
            result = await run_topic_model_adapter({"text": "Sample text"}, {})

        assert result == {"topics": [], "error": "topic_model_error"}


class TestEntityExtractAdapter:
    """Tests for run_entity_extract_adapter."""

    @pytest.mark.asyncio
    async def test_entity_extract_basic(self):
        """Test basic entity extraction."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"person": [{"name": "John Smith", "type": "person"}], "place": [{"name": "New York", "type": "place"}]}'
                    }
                }
            ]
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "John Smith visited New York last week."}
            context = {}

            result = await run_entity_extract_adapter(config, context)

            assert "entities" in result
            assert "total_count" in result
            assert result["total_count"] >= 2

    @pytest.mark.asyncio
    async def test_entity_extract_with_entity_types(self):
        """Test entity extraction with specific entity types."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        mock_response = {"choices": [{"message": {"content": '{"person": [{"name": "Jane Doe", "type": "person"}]}'}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "Jane Doe is a software engineer.", "entity_types": ["person"]}
            context = {}

            result = await run_entity_extract_adapter(config, context)

            assert "entities" in result
            assert "person" in result["entities"]

    @pytest.mark.asyncio
    async def test_entity_extract_missing_text(self):
        """Test entity extraction with missing text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        config = {}
        context = {}

        result = await run_entity_extract_adapter(config, context)

        assert "error" in result
        assert result["error"] == "missing_text"

    @pytest.mark.asyncio
    async def test_entity_extract_from_previous_context(self):
        """Test entity extraction from previous context."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        mock_response = {"choices": [{"message": {"content": '{"organization": [{"name": "Acme Corp", "type": "organization"}]}'}}]}

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {}
            context = {"prev": {"text": "Acme Corp announced new products."}}

            result = await run_entity_extract_adapter(config, context)

            assert "entities" in result

    @pytest.mark.asyncio
    async def test_entity_extract_with_include_context(self):
        """Test entity extraction with context included."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        mock_response = {
            "choices": [
                {"message": {"content": '{"person": [{"name": "Alice", "type": "person", "context": "Alice is the CEO"}]}'}}
            ]
        }

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            config = {"text": "Alice is the CEO of the company.", "include_context": True}
            context = {}

            result = await run_entity_extract_adapter(config, context)

            assert "entities" in result

    @pytest.mark.asyncio
    async def test_entity_extract_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        config = {"text": "test"}
        context = {"is_cancelled": lambda: True}

        result = await run_entity_extract_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_entity_extract_sanitizes_backend_errors(self):
        """Test entity extraction hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_entity_extract_adapter

        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("entity backend exploded at /private/entity-cache"),
        ):
            result = await run_entity_extract_adapter({"text": "Sample text"}, {})

        assert result == {"error": "entity_extract_error", "entities": {}, "total_count": 0}


class TestTokenCountAdapter:
    """Tests for run_token_count_adapter."""

    @pytest.mark.asyncio
    async def test_token_count_basic(self):
        """Test basic token counting."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {"text": "This is a test sentence for token counting."}
        context = {}

        result = await run_token_count_adapter(config, context)

        assert "token_count" in result
        assert "char_count" in result
        assert "word_count" in result
        assert result["token_count"] > 0
        assert result["char_count"] > 0
        assert result["word_count"] > 0

    @pytest.mark.asyncio
    async def test_token_count_empty_text(self):
        """Test token counting with empty text."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {"text": ""}
        context = {}

        result = await run_token_count_adapter(config, context)

        assert "token_count" in result
        assert result["token_count"] == 0
        assert result["char_count"] == 0

    @pytest.mark.asyncio
    async def test_token_count_with_model(self):
        """Test token counting with specific model."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {"text": "Sample text", "model": "gpt-4"}
        context = {}

        result = await run_token_count_adapter(config, context)

        assert "token_count" in result
        assert "model" in result or "estimated" in result

    @pytest.mark.asyncio
    async def test_token_count_from_previous_context(self):
        """Test token counting from previous context."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {}
        context = {"prev": {"text": "Text from previous step"}}

        result = await run_token_count_adapter(config, context)

        assert "token_count" in result
        assert result["token_count"] > 0

    @pytest.mark.asyncio
    async def test_token_count_word_count_accuracy(self):
        """Test word count accuracy."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {"text": "one two three four five"}
        context = {}

        result = await run_token_count_adapter(config, context)

        assert result["word_count"] == 5

    @pytest.mark.asyncio
    async def test_token_count_char_count_accuracy(self):
        """Test character count accuracy."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {"text": "hello"}
        context = {}

        result = await run_token_count_adapter(config, context)

        assert result["char_count"] == 5

    @pytest.mark.asyncio
    async def test_token_count_cancelled(self):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.text import run_token_count_adapter

        config = {"text": "test"}
        context = {"is_cancelled": lambda: True}

        result = await run_token_count_adapter(config, context)

        assert result.get("__status__") == "cancelled"


# =============================================================================
# Integration Tests
# =============================================================================


class TestTextAdapterImports:
    """Tests for text adapter imports."""

    def test_all_text_adapters_importable(self):
        """Test that all text adapters can be imported."""
        from tldw_Server_API.app.core.Workflows.adapters.text import (
            run_html_to_markdown_adapter,
            run_markdown_to_html_adapter,
            run_csv_to_json_adapter,
            run_json_to_csv_adapter,
            run_json_transform_adapter,
            run_json_validate_adapter,
            run_xml_transform_adapter,
            run_template_render_adapter,
            run_regex_extract_adapter,
            run_text_clean_adapter,
            run_keyword_extract_adapter,
            run_sentiment_analyze_adapter,
            run_language_detect_adapter,
            run_topic_model_adapter,
            run_entity_extract_adapter,
            run_token_count_adapter,
        )

        # All should be callable
        assert callable(run_html_to_markdown_adapter)
        assert callable(run_markdown_to_html_adapter)
        assert callable(run_csv_to_json_adapter)
        assert callable(run_json_to_csv_adapter)
        assert callable(run_json_transform_adapter)
        assert callable(run_json_validate_adapter)
        assert callable(run_xml_transform_adapter)
        assert callable(run_template_render_adapter)
        assert callable(run_regex_extract_adapter)
        assert callable(run_text_clean_adapter)
        assert callable(run_keyword_extract_adapter)
        assert callable(run_sentiment_analyze_adapter)
        assert callable(run_language_detect_adapter)
        assert callable(run_topic_model_adapter)
        assert callable(run_entity_extract_adapter)
        assert callable(run_token_count_adapter)

    def test_text_adapters_are_async(self):
        """Test that all text adapters are async functions."""
        import asyncio

        from tldw_Server_API.app.core.Workflows.adapters.text import (
            run_html_to_markdown_adapter,
            run_markdown_to_html_adapter,
            run_csv_to_json_adapter,
            run_json_to_csv_adapter,
            run_json_transform_adapter,
            run_json_validate_adapter,
            run_xml_transform_adapter,
            run_template_render_adapter,
            run_regex_extract_adapter,
            run_text_clean_adapter,
            run_keyword_extract_adapter,
            run_sentiment_analyze_adapter,
            run_language_detect_adapter,
            run_topic_model_adapter,
            run_entity_extract_adapter,
            run_token_count_adapter,
        )

        adapters = [
            run_html_to_markdown_adapter,
            run_markdown_to_html_adapter,
            run_csv_to_json_adapter,
            run_json_to_csv_adapter,
            run_json_transform_adapter,
            run_json_validate_adapter,
            run_xml_transform_adapter,
            run_template_render_adapter,
            run_regex_extract_adapter,
            run_text_clean_adapter,
            run_keyword_extract_adapter,
            run_sentiment_analyze_adapter,
            run_language_detect_adapter,
            run_topic_model_adapter,
            run_entity_extract_adapter,
            run_token_count_adapter,
        ]

        for adapter in adapters:
            assert asyncio.iscoroutinefunction(adapter), f"{adapter.__name__} is not async"


class TestTextAdapterRegistry:
    """Tests for text adapter registry integration."""

    def test_text_adapters_registered(self):
        """Test that text adapters are registered in the registry."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        expected_adapters = [
            "html_to_markdown",
            "markdown_to_html",
            "csv_to_json",
            "json_to_csv",
            "json_transform",
            "json_validate",
            "xml_transform",
            "template_render",
            "regex_extract",
            "text_clean",
            "keyword_extract",
            "sentiment_analyze",
            "language_detect",
            "topic_model",
            "entity_extract",
            "token_count",
        ]

        registered = registry.list_adapters()
        for adapter_name in expected_adapters:
            assert adapter_name in registered, f"{adapter_name} not found in registry"

    def test_text_adapters_have_text_category(self):
        """Test that text adapters are in the text category."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        catalog = registry.get_catalog()
        assert "text" in catalog
        assert len(catalog["text"]) >= 16  # At least 16 text adapters
