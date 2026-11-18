import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import (
    normalize_pdf_text_for_storage,
)

pytestmark = pytest.mark.unit


def test_native_pdf_page_map_matches_normalized_text_and_blank_pages(tmp_path):
    import hashlib

    import pymupdf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf

    path = tmp_path / "pages.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "First page has a short paragraph.")
        doc.new_page()
        doc.new_page().insert_text((72, 72), "Final page has a different paragraph.")
        doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1}])
        doc.save(path)
    result = process_pdf(path, path.name, perform_chunking=False)
    page_map = result["metadata"].get("page_map", [])
    assert len(page_map) == 3
    assert [entry["page_number"] for entry in page_map] == [1, 2, 3]
    assert page_map[1]["start_char"] == page_map[1]["end_char"]
    assert page_map[2]["printed_page_label"] == "iii"
    assert "Final page" in result["content"][page_map[2]["start_char"] : page_map[2]["end_char"]]
    assert page_map[-1]["end_char"] == len(result["content"])
    assert result["metadata"]["page_map_content_hash"] == hashlib.sha256(result["content"].encode()).hexdigest()


@pytest.mark.parametrize("page_keys", [("page",), ("page_number",), ("page", "page_number")])
def test_pdf_page_map_uses_normalized_unicode_offsets_without_inventing_labels(tmp_path, monkeypatch, page_keys):
    import pymupdf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf

    path = tmp_path / "unicode.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(path)
    monkeypatch.setattr(
        pdf.pymupdf4llm,
        "to_markdown",
        lambda *args, **kwargs: [
            {"metadata": dict.fromkeys(page_keys, 1), "text": "Café 😀\ncontinues here."},
            {"metadata": dict.fromkeys(page_keys, 2), "text": "東京\n文章。"},
        ],
    )
    result = pdf.process_pdf(path, path.name, perform_chunking=False)
    page_map = result["metadata"]["page_map"]
    assert result["content"][: page_map[0]["end_char"]] == "Café 😀 continues here."
    assert result["content"][page_map[1]["start_char"] :] == "東京文章。"
    assert all("printed_page_label" not in entry for entry in page_map)


@pytest.mark.parametrize(
    "metadata",
    [
        {}, None, {"page_number": None}, {"page_number": True}, {"page_number": 0},
        {"page_number": "1"}, {"page_number": 2}, {"page": 0},
        {"page": 1, "page_number": 2}, {"page": 1, "page_number": None},
        {"page": True, "page_number": 1},
    ],
)
def test_invalid_parser_page_identity_preserves_text_without_provenance(tmp_path, monkeypatch, metadata):
    import pymupdf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf

    path = tmp_path / "invalid-page.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1}])
        doc.save(path)
    monkeypatch.setattr(
        pdf.pymupdf4llm, "to_markdown",
        lambda *args, **kwargs: [{"metadata": metadata, "text": "Café 😀\ncontinues here."}],
    )

    result = pdf.process_pdf(path, path.name, perform_chunking=False)

    assert result["content"] == "Café 😀 continues here."
    assert result["metadata"]["page_count"] == 1
    assert "page_map" not in result["metadata"]
    assert "page_map_content_hash" not in result["metadata"]


def test_ocr_replacement_does_not_keep_native_page_map(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pymupdf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf

    path = tmp_path / "ocr.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Native text before replacement.")
        doc.save(path)
    monkeypatch.setattr(pdf, "_get_ocr_backend", lambda *args: SimpleNamespace(name="test"))
    monkeypatch.setattr(pdf, "_ocr_pdf_pages", lambda **kwargs: ("Replacement OCR content", 1, 1, None))
    result = pdf.process_pdf(path, path.name, perform_chunking=False, enable_ocr=True, ocr_mode="always")
    assert result["content"] == "Replacement OCR content"
    assert "page_map" not in result["metadata"]


def test_reflows_soft_wrapped_paragraph_lines():
    src = (
        "We are not just interested in models that perform well on a\n"
        "single physical task, but rather models that robustly generalize.\n\n"
        "Therefore, we test generalization."
    )
    out = normalize_pdf_text_for_storage(src)
    assert "perform well on a single physical task" in out
    assert "\n\nTherefore, we test generalization." in out


def test_preserves_structural_blocks():
    src = (
        "## Page 1\n\n"
        "# Heading\n"
        "- list item one\n"
        "- list item two\n\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n\n"
        "Paragraph line one\n"
        "line two\n\n"
        "---\n"
    )
    out = normalize_pdf_text_for_storage(src)
    assert "# Heading" in out
    assert "- list item one" in out
    assert "| a | b |" in out
    assert "Paragraph line one line two" in out
    assert "## Page 1" in out
    assert "\n---\n" in f"\n{out}\n"


def test_repairs_hyphenated_soft_wraps():
    src = "generaliza-\ntion improves.\n\nnon-\nLinear stays separated."
    out = normalize_pdf_text_for_storage(src)
    assert "generalization improves." in out
    assert "non- Linear stays separated." in out


def test_idempotent_normalization():
    src = "Line one\nline two\n\n# Keep heading\n"
    first = normalize_pdf_text_for_storage(src)
    second = normalize_pdf_text_for_storage(first)
    assert first == second


def test_preserves_fenced_code_blocks():
    src = "```python\n" "foo = 1\n" "bar = 2\n" "```\n\n" "Paragraph line one\n" "line two"
    out = normalize_pdf_text_for_storage(src)
    assert "```python\nfoo = 1\nbar = 2\n```" in out
    assert "Paragraph line one line two" in out


def test_preserves_blockquote_lines():
    src = "> quoted line one\n" "> quoted line two\n\n" "Paragraph starts\n" "and continues."
    out = normalize_pdf_text_for_storage(src)
    assert "> quoted line one" in out
    assert "> quoted line two" in out
    assert "Paragraph starts and continues." in out


def test_reflows_with_crlf_input():
    src = "Alpha line one\r\nline two\r\n\r\nBeta line one\r\nline two"
    out = normalize_pdf_text_for_storage(src)
    assert out == "Alpha line one line two\n\nBeta line one line two"


def test_preserves_indented_code_style_blocks():
    src = "    def foo():\n" "        return 1\n\n" "Paragraph starts\n" "and continues"
    out = normalize_pdf_text_for_storage(src)
    assert "    def foo():" in out
    assert "        return 1" in out
    assert "Paragraph starts and continues" in out


def test_preserves_ordered_list_blocks():
    src = "1. first item\n" "2. second item\n\n" "Paragraph starts\n" "and continues."
    out = normalize_pdf_text_for_storage(src)
    assert "1. first item" in out
    assert "2. second item" in out
    assert "Paragraph starts and continues." in out


def test_single_pipe_text_still_reflows():
    src = "A | B relation starts here\nand continues there."
    out = normalize_pdf_text_for_storage(src)
    assert out == "A | B relation starts here and continues there."


def test_preserves_page_markers_and_hrules():
    src = (
        "## Page 2\n\n"
        "Paragraph line one\n"
        "line two\n\n"
        "---\n\n"
        "## Page 3\n\n"
        "Next paragraph\n"
        "continues here"
    )
    out = normalize_pdf_text_for_storage(src)
    assert "## Page 2" in out
    assert "## Page 3" in out
    assert "\n---\n" in f"\n{out}\n"
    assert "Paragraph line one line two" in out
    assert "Next paragraph continues here" in out


def test_table_separator_block_is_preserved():
    src = "| col a | col b |\n" "| :--- | ---: |\n" "| 1 | 2 |\n\n" "Paragraph starts\n" "and continues."
    out = normalize_pdf_text_for_storage(src)
    assert "| col a | col b |" in out
    assert "| :--- | ---: |" in out
    assert "| 1 | 2 |" in out
    assert "Paragraph starts and continues." in out


def test_mixed_list_and_paragraph_boundaries_are_preserved():
    src = "- item one\n" "- item two\n\n" "Paragraph starts here\n" "and continues here.\n\n" "- item three\n"
    out = normalize_pdf_text_for_storage(src)
    assert "- item one" in out
    assert "- item two" in out
    assert "- item three" in out
    assert "Paragraph starts here and continues here." in out


def test_list_continuation_line_stays_attached_to_list_context():
    src = "1. first item\n" "  continuation detail\n" "2. second item\n\n" "Paragraph starts\n" "and ends."
    out = normalize_pdf_text_for_storage(src)
    assert "1. first item" in out
    assert "  continuation detail" in out
    assert "2. second item" in out
    assert "Paragraph starts and ends." in out


def test_cjk_lines_join_without_inserting_ascii_space():
    src = "我们正在研究模型的泛化能力\n并测试其在新任务上的表现。"
    out = normalize_pdf_text_for_storage(src)
    assert out == "我们正在研究模型的泛化能力并测试其在新任务上的表现。"


def test_unicode_quotes_are_preserved_during_reflow():
    src = "“This quoted sentence starts\nand continues with unicode punctuation.”"
    out = normalize_pdf_text_for_storage(src)
    assert out == "“This quoted sentence starts and continues with unicode punctuation.”"


def test_math_like_wrapped_lines_reflow_cleanly():
    src = "f(x) = x^2 + 2x + 1\nfor x in R.\n\ng(x)=sin(x)\nfor x in [0, pi]."
    out = normalize_pdf_text_for_storage(src)
    assert "f(x) = x^2 + 2x + 1 for x in R." in out
    assert "g(x)=sin(x) for x in [0, pi]." in out


def test_unindented_bullet_continuation_stays_in_list_context():
    src = "- First bullet starts here\n" "continues on wrapped line\n" "- Second bullet remains separate"
    out = normalize_pdf_text_for_storage(src)
    assert out == ("- First bullet starts here continues on wrapped line\n" "- Second bullet remains separate")


def test_mixed_script_wrap_keeps_expected_spacing():
    src = "Model 使用中文\nand English mixed output."
    out = normalize_pdf_text_for_storage(src)
    assert out == "Model 使用中文 and English mixed output."


def test_table_caption_boundaries_are_preserved():
    src = (
        "Table 1: Benchmark results\n"
        "across evaluated models\n"
        "| model | acc |\n"
        "|---|---|\n"
        "| A | 90 |\n"
        "Caption after table starts\n"
        "and continues here."
    )
    out = normalize_pdf_text_for_storage(src)
    assert "Table 1: Benchmark results across evaluated models" in out
    assert "| model | acc |" in out
    assert "|---|---|" in out
    assert "| A | 90 |" in out
    assert "Caption after table starts and continues here." in out
