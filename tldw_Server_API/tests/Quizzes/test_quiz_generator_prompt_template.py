import os

import pytest

os.environ.setdefault("TEST_MODE", "1")
pytestmark = pytest.mark.unit

from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services.quiz_generator import QUIZ_GENERATION_PROMPT


def test_quiz_generation_prompt_template_formats_with_literal_citation_object():
    rendered_prompt = QUIZ_GENERATION_PROMPT.format(
        num_questions=3,
        content="Sample content",
        difficulty="mixed",
        question_types="multiple_choice, true_false",
        focus_instruction="- Focus on these topics: testing",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert '"label": "Optional citation label"' in rendered_prompt
    assert '"source_type": "media" | "note" | "flashcard_deck" | "flashcard_card"' in rendered_prompt
    assert "Allowed sources for source_citations.source_type/source_id: note:note-1" in rendered_prompt
    assert "{num_questions}" not in rendered_prompt
    assert "{content}" not in rendered_prompt


def test_quiz_generation_prompt_includes_all_planned_question_shapes():
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=5,
        content="Sample content",
        difficulty="mixed",
        question_types=None,
        question_plan=[
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            {"question_type": "multi_select", "count": 1, "option_count": 5},
            {"question_type": "matching", "count": 1, "pair_count": 2},
            {"question_type": "true_false", "count": 1},
            {"question_type": "fill_blank", "count": 1},
        ],
        focus_instruction="",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert "multiple_choice: 1 question(s), exactly 5 options" in rendered_prompt
    assert "multi_select: 1 question(s), exactly 5 options" in rendered_prompt
    assert "matching: 1 question(s), exactly 2 pairs" in rendered_prompt
    assert "true_false: 1 question(s)" in rendered_prompt
    assert "fill_blank: 1 question(s)" in rendered_prompt
    assert '"question_type": "multi_select"' in rendered_prompt
    assert '"question_type": "matching"' in rendered_prompt
    assert "options must be array of 4 strings" not in rendered_prompt


def test_quiz_generation_prompt_preserves_source_content_when_removing_legacy_hints():
    content = 'Evidence excerpt: keep literal "options": ["A", "B", "C", "D"] from the source.'

    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content=content,
        difficulty="mixed",
        question_types=None,
        question_plan=[{"question_type": "multiple_choice", "count": 1, "option_count": 5}],
        focus_instruction="",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert content in rendered_prompt
    assert "Planned question requirements" in rendered_prompt
