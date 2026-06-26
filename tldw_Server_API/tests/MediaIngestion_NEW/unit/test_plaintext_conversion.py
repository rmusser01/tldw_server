from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext import Plaintext_Files as docs


@pytest.mark.unit
def test_convert_document_to_text_requires_defusedxml(monkeypatch, tmp_path):
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text("<root><value>1</value></root>", encoding="utf-8")

    monkeypatch.setattr(docs, "_DEFUSED_AVAILABLE", False)
    monkeypatch.setattr(docs, "DET", None)

    with pytest.raises(ValueError, match="defusedxml"):
        docs.convert_document_to_text(xml_path)


@pytest.mark.unit
def test_convert_document_to_text_rejects_outside_base_dir(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="outside allowed base directory"):
        docs.convert_document_to_text(outside, base_dir=base_dir)


@pytest.mark.unit
def test_convert_document_to_text_reads_markdown_extension(tmp_path):
    markdown_path = tmp_path / "sample.markdown"
    markdown_path.write_text("# Heading\n\nBody", encoding="utf-8")

    content, source_format, metadata = docs.convert_document_to_text(markdown_path)

    assert "# Heading" in content
    assert "Body" in content
    assert source_format == "markdown"
    assert metadata["extracted_title"] is None


@pytest.mark.unit
def test_convert_document_to_text_converts_xhtml_as_html(tmp_path):
    xhtml_path = tmp_path / "sample.xhtml"
    xhtml_path.write_text(
        "<html><head><title>XHTML Title</title></head><body><h1>Hello</h1><script>bad()</script></body></html>",
        encoding="utf-8",
    )

    content, source_format, metadata = docs.convert_document_to_text(xhtml_path)

    assert "Hello" in content
    assert "bad()" not in content
    assert source_format == "html"
    assert metadata["html_title"] == "XHTML Title"
