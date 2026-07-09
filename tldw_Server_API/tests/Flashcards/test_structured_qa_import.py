from tldw_Server_API.app.core.Flashcards.structured_qa_import import (
    parse_structured_qa_preview,
)


def test_parse_structured_qa_preview_builds_multiline_pairs():
    result = parse_structured_qa_preview(
        """Q: What is ATP?
A: Primary cellular energy currency.
Still part of the answer.

Question: What is glycolysis?
Answer: Cytosolic glucose breakdown.
"""
    )

    assert [draft.front for draft in result.drafts] == [
        "What is ATP?",
        "What is glycolysis?",
    ]
    assert result.drafts[0].back == (
        "Primary cellular energy currency.\nStill part of the answer."
    )
    assert result.errors == []


def test_parse_structured_qa_preview_reports_incomplete_blocks():
    result = parse_structured_qa_preview(
        """Q: Complete pair
A: Complete answer

Q: Missing answer
"""
    )

    assert len(result.drafts) == 1
    assert result.errors[0].line == 4
    assert "Missing answer" in result.errors[0].error


def test_parse_structured_qa_preview_respects_line_caps():
    result = parse_structured_qa_preview(
        "Q: One\nA: First\nQ: Two\nA: Second\n",
        max_lines=2,
    )

    assert [draft.front for draft in result.drafts] == ["One"]
    assert any("Maximum preview line limit" in error.error for error in result.errors)


def test_parse_structured_qa_preview_preserves_multiline_question_spacing():
    result = parse_structured_qa_preview(
        """Question: Explain the steps:
  1. Gather input.

  2. Return output.
Answer: Carefully.
"""
    )

    assert len(result.drafts) == 1
    assert result.drafts[0].front == (
        "Explain the steps:\n"
        "  1. Gather input.\n"
        "\n"
        "  2. Return output."
    )


def test_parse_structured_qa_preview_accepts_short_labels_with_alt_separators():
    result = parse_structured_qa_preview("  q. One?\n  a- Two.\n")

    assert len(result.drafts) == 1
    assert result.drafts[0].front == "One?"
    assert result.drafts[0].back == "Two."
    assert result.errors == []
