import uuid
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services.quiz_generator import (
    _build_test_mode_questions,
    _normalize_questions,
    generate_quiz_from_sources,
)

pytestmark = pytest.mark.integration


def _valid_emq_questions(*, group_id: str = "emq-1") -> list[dict]:
    options = ["First option", "Second option", "Third option"]
    group_prompt = "Choose the single best option for each stem."
    return [
        {
            "question_type": "multiple_choice",
            "question_text": f"EMQ stem {index + 1}",
            "group_id": group_id,
            "group_prompt": group_prompt,
            "options": list(options),
            "correct_answer": index,
            "explanation": f"Explanation for stem {index + 1}.",
            "source_citations": [
                {
                    "source_type": "note",
                    "source_id": "note-emq",
                    "quote": f"Evidence for stem {index + 1}.",
                }
            ],
        }
        for index in range(2)
    ]


ASSERTION_REASONING_OPTIONS = (
    "Both the assertion and reason are true, and the reason correctly explains the assertion.",
    "Both the assertion and reason are true, but the reason does not explain the assertion.",
    "The assertion is true, but the reason is false.",
    "The assertion is false, but the reason is true.",
    "Both the assertion and reason are false.",
)


def _valid_assertion_reasoning_question(**overrides: object) -> dict:
    question = {
        "question_type": "multiple_choice",
        "assertion": "The review board meets every Friday.",
        "reason": "The selected note states that Friday review boards are required.",
        "options": ["LLM-supplied option"],
        "correct_answer": 0,
        "explanation": "The citation directly supports both statements and their relationship.",
        "source_citations": [
            {
                "source_type": "note",
                "source_id": "note-ar",
                "quote": "Review boards meet every Friday.",
            }
        ],
    }
    question.update(overrides)
    return question


def _normalize_assertion_reasoning_questions(**overrides: object) -> list[dict]:
    return _normalize_questions(
        [_valid_assertion_reasoning_question(**overrides)],
        default_source_type="note",
        default_source_id="note-ar",
        generation_profile="assertion_reasoning",
    )


def _normalize_assertion_reasoning_question(**overrides: object) -> dict:
    questions = _normalize_assertion_reasoning_questions(**overrides)
    assert len(questions) == 1
    return questions[0]


@pytest.fixture(scope="function")
def quizzes_db(tmp_path):
    db_path = tmp_path / "quiz-generator-test-mode.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.fixture(scope="function")
def media_db(tmp_path):
    db_path = tmp_path / "quiz-generator-media.db"
    db = MediaDatabase(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_returns_deterministic_payload_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Workspace Alpha",
        content="Alpha program requires citations, review boards, and Friday freshness checks.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=2,
        question_types=["multiple_choice", "true_false"],
        workspace_tag="workspace:test",
    )

    assert result["quiz"]["workspace_tag"] == "workspace:test"
    assert len(result["questions"]) == 2
    assert result["questions"][0]["source_citations"][0]["source_type"] == "note"
    assert result["questions"][0]["source_citations"][0]["source_id"] == note_id


def test_build_test_mode_questions_prefers_evidence_source_identity() -> None:
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-b",
                "text": "Beta evidence should keep its own citation identity.",
            }
        ],
        normalized_sources=[
            {"source_type": "note", "source_id": "note-a"},
            {"source_type": "note", "source_id": "note-b"},
        ],
        num_questions=1,
        question_types=["multiple_choice"],
    )

    citation = questions[0]["source_citations"][0]
    assert citation["source_type"] == "note"
    assert citation["source_id"] == "note-b"
    assert citation["quote"] == "Beta evidence should keep its own citation identity."


def test_build_test_mode_questions_honors_planned_question_shapes() -> None:
    plan = [
        {"question_type": "multiple_choice", "count": 1, "option_count": 5},
        {"question_type": "multi_select", "count": 1, "option_count": 5},
        {"question_type": "matching", "count": 1, "pair_count": 3},
        {"question_type": "true_false", "count": 1},
        {"question_type": "fill_blank", "count": 1},
    ]
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-plan",
                "text": "Alpha evidence supports planned deterministic questions.",
            }
        ],
        normalized_sources=[{"source_type": "note", "source_id": "note-plan"}],
        num_questions=5,
        question_types=None,
        question_plan=plan,
    )

    assert [question["question_type"] for question in questions] == [
        "multiple_choice",
        "multi_select",
        "matching",
        "true_false",
        "fill_blank",
    ]
    assert len(questions[0]["options"]) == 5
    assert len(questions[1]["options"]) == 5
    assert len(questions[2]["options"]) == 3
    assert len(questions[2]["correct_answer"]) == 3
    assert all(question["source_citations"][0]["source_id"] == "note-plan" for question in questions)


def test_build_test_mode_questions_emits_five_options_for_best_of_five_profile() -> None:
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-bof",
                "text": "BOF evidence should produce one best answer.",
            }
        ],
        normalized_sources=[{"source_type": "note", "source_id": "note-bof"}],
        num_questions=1,
        question_types=None,
        generation_profile="best_of_five",
    )

    question = questions[0]
    assert question["question_type"] == "multiple_choice"
    assert len(question["options"]) == 5
    assert question["correct_answer"] == 0
    assert question["tags"] == ["best_of_five"]


def test_build_test_mode_questions_emits_complete_emq_group_when_one_stem_requested() -> None:
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-emq",
                "text": "EMQ evidence for deterministic generation.",
            }
        ],
        normalized_sources=[{"source_type": "note", "source_id": "note-emq"}],
        num_questions=1,
        question_types=None,
        generation_profile="emq",
    )

    assert len(questions) >= 2
    assert {question["question_type"] for question in questions} == {"multiple_choice"}
    assert len({question["group_id"] for question in questions}) == 1
    assert len({question["group_prompt"] for question in questions}) == 1
    assert questions[0]["group_id"]
    assert questions[0]["group_prompt"]
    assert 2 <= len(questions[0]["options"]) <= 10
    assert all(question["options"] == questions[0]["options"] for question in questions)
    assert all(question["explanation"] for question in questions)
    assert all(question["source_citations"] for question in questions)
    assert len({id(question["source_citations"]) for question in questions}) == len(questions)


@pytest.mark.parametrize(
    ("raw_answer", "expected_index"),
    [
        (1, 1),
        ("1", 1),
        ("B", 1),
        ("Second option", 1),
    ],
)
def test_normalize_questions_accepts_strict_emq_answer_forms(
    raw_answer: object,
    expected_index: int,
) -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[0]["correct_answer"] = raw_answer

    questions = _normalize_questions(
        raw_questions,
        default_source_type="note",
        default_source_id="note-emq",
        generation_profile="emq",
    )

    assert questions[0]["correct_answer"] == expected_index
    assert questions[0]["group_id"] == "emq-1"
    assert questions[0]["group_prompt"] == "Choose the single best option for each stem."


@pytest.mark.parametrize(
    ("options", "raw_answer"),
    [
        (["1", "x"], "1"),
        (["B", "x"], "B"),
        (["x", "x"], "x"),
    ],
)
def test_normalize_questions_rejects_ambiguous_emq_string_answers(
    options: list[str],
    raw_answer: str,
) -> None:
    raw_questions = _valid_emq_questions()
    for question in raw_questions:
        question["options"] = list(options)
    raw_questions[0]["correct_answer"] = raw_answer
    raw_questions[1]["correct_answer"] = 1

    with pytest.raises(ValueError, match="ambiguous"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("group_id", None),
        ("group_id", "   "),
        ("group_id", "g" * 129),
        ("group_prompt", None),
        ("group_prompt", "   "),
        ("group_prompt", "p" * 2001),
    ],
)
def test_normalize_questions_rejects_invalid_emq_group_metadata(
    field: str,
    invalid_value: object,
) -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[0][field] = invalid_value

    with pytest.raises(ValueError, match=field):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


def test_normalize_questions_rejects_non_multiple_choice_emq_stem() -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[0].update(
        question_type="true_false",
        options=None,
        correct_answer="true",
    )

    with pytest.raises(ValueError, match="multiple_choice"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


@pytest.mark.parametrize("explanation", [None, "   "])
def test_normalize_questions_rejects_emq_stem_without_explanation(explanation: object) -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[0]["explanation"] = explanation

    with pytest.raises(ValueError, match="explanation"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        ["Only option"],
        [f"Option {index}" for index in range(11)],
    ],
)
def test_normalize_questions_rejects_invalid_emq_option_count(options: object) -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[0]["options"] = options

    with pytest.raises(ValueError, match="options"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


def test_normalize_questions_rejects_different_emq_option_banks() -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[1]["options"] = ["First option", "Different option", "Third option"]

    with pytest.raises(ValueError, match="option bank"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


def test_normalize_questions_rejects_different_emq_group_prompts() -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[1]["group_prompt"] = "A different group prompt."

    with pytest.raises(ValueError, match="group_prompt"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


def test_normalize_questions_rejects_emq_group_with_one_stem() -> None:
    with pytest.raises(ValueError, match="at least two stems"):
        _normalize_questions(
            _valid_emq_questions()[:1],
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


@pytest.mark.parametrize(
    "raw_answer",
    [None, "", 3, "3", "D", 1.0, "1.0", "not an option", "SECOND OPTION"],
)
def test_normalize_questions_rejects_invalid_emq_answers(raw_answer: object) -> None:
    raw_questions = _valid_emq_questions()
    raw_questions[0]["correct_answer"] = raw_answer

    with pytest.raises(ValueError, match="correct_answer"):
        _normalize_questions(
            raw_questions,
            default_source_type="note",
            default_source_id="note-emq",
            generation_profile="emq",
        )


def test_normalize_questions_preserves_legacy_mcq_answer_fallback() -> None:
    questions = _normalize_questions(
        [
            {
                "question_type": "multiple_choice",
                "question_text": "Legacy MCQ",
                "options": ["First", "Second"],
                "correct_answer": "not an option",
            }
        ],
        default_source_type="note",
        default_source_id="note-legacy",
    )

    assert questions[0]["correct_answer"] == 0


@pytest.mark.parametrize("generation_profile", ["standard_recall", "mixed_assessment", "best_of_five"])
def test_normalize_questions_clears_group_metadata_for_non_emq_profiles(
    generation_profile: str,
) -> None:
    options = ["A", "B", "C", "D", "E"] if generation_profile == "best_of_five" else ["A", "B"]
    questions = _normalize_questions(
        [
            {
                "question_type": "multiple_choice",
                "question_text": "Ordinary multiple choice question",
                "group_id": "llm-supplied-group",
                "group_prompt": "LLM-supplied group prompt",
                "options": options,
                "correct_answer": 0,
            }
        ],
        default_source_type="note",
        default_source_id="note-ordinary",
        generation_profile=generation_profile,
    )

    assert questions[0]["group_id"] is None
    assert questions[0]["group_prompt"] is None


def test_validate_emq_groups_does_not_mutate_answers_before_all_invariants_pass() -> None:
    questions = _valid_emq_questions()
    questions[0]["correct_answer"] = "B"
    questions[1]["options"] = ["First option", "Different option", "Third option"]

    with pytest.raises(ValueError, match="option bank"):
        quiz_generator._validate_emq_groups(questions)

    assert questions[0]["correct_answer"] == "B"


def test_emq_limiter_keeps_whole_groups_and_non_emq_uses_plain_slicing() -> None:
    limiter = getattr(quiz_generator, "_limit_questions_by_profile", None)
    assert limiter is not None
    questions = _valid_emq_questions(group_id="emq-a") + _valid_emq_questions(group_id="emq-b")

    assert limiter(questions, num_questions=1, generation_profile="emq") == questions[:2]
    assert limiter(questions, num_questions=3, generation_profile="emq") == questions
    assert limiter(questions, num_questions=3, generation_profile="standard_recall") == questions[:3]


def test_normalize_questions_marks_best_of_five_with_existing_tags() -> None:
    questions = _normalize_questions(
        [
            {
                "question_type": "multiple_choice",
                "question_text": "Which is the best next step?",
                "options": ["A", "B", "C", "D", "E"],
                "correct_answer": 2,
                "explanation": "C is best supported by the citation.",
                "tags": ["cardiology", "best_of_five"],
                "source_citations": [
                    {
                        "source_type": "note",
                        "source_id": "note-bof",
                        "quote": "Best answer evidence.",
                    }
                ],
            }
        ],
        default_source_type="note",
        default_source_id="note-bof",
        generation_profile="best_of_five",
    )

    assert len(questions) == 1
    assert questions[0]["tags"] == ["cardiology", "best_of_five"]


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_persists_best_of_five_tags_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Best of Five Note",
        content="The patient needs the best supported answer from five plausible options.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=1,
        question_types=None,
        generation_profile="best_of_five",
    )

    question = result["questions"][0]
    assert question["tags"] == ["best_of_five"]
    assert len(question["options"]) == 5


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_persists_complete_emq_group_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="EMQ Note",
        content="Each EMQ stem needs a shared bank and its own source-backed rationale.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=1,
        question_types=None,
        generation_profile="emq",
    )

    questions = result["questions"]
    assert len(questions) >= 2
    assert len({question["group_id"] for question in questions}) == 1
    assert len({question["group_prompt"] for question in questions}) == 1
    assert questions[0]["group_id"]
    assert questions[0]["group_prompt"]
    assert all(question["options"] == questions[0]["options"] for question in questions)
    assert len({question["question_text"] for question in questions}) == len(questions)
    assert len({question["explanation"] for question in questions}) == len(questions)
    assert all(question["source_citations"] for question in questions)


@pytest.mark.asyncio
async def test_generate_quiz_requests_at_least_two_emq_stems_from_production_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_llm(**kwargs):
        captured.update(kwargs)
        return {"questions": _valid_emq_questions()}

    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [
            {"source_type": "note", "source_id": "note-emq", "text": "EMQ evidence."}
        ],
    )
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_llm)
    monkeypatch.setattr(quiz_generator, "extract_response_content", lambda raw: raw)
    monkeypatch.setattr(
        quiz_generator,
        "_persist_generated_quiz",
        lambda **kwargs: {"quiz": {"id": 1}, "questions": kwargs["questions"]},
    )

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-emq"}],
        num_questions=1,
        generation_profile="emq",
    )

    assert len(result["questions"]) == 2
    assert "generate 2 quiz questions" in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_quiz_does_not_persist_emq_invalidated_after_limiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_llm(**_kwargs):
        return {"questions": _valid_emq_questions()}

    persistence = Mock()
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [
            {"source_type": "note", "source_id": "note-emq", "text": "EMQ evidence."}
        ],
    )
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_llm)
    monkeypatch.setattr(quiz_generator, "extract_response_content", lambda raw: raw)
    monkeypatch.setattr(
        quiz_generator,
        "_limit_questions_by_profile",
        lambda questions, **_kwargs: list(questions)[:1],
    )
    monkeypatch.setattr(quiz_generator, "_persist_generated_quiz", persistence)

    with pytest.raises(ValueError, match="at least two stems"):
        await generate_quiz_from_sources(
            db=Mock(),
            media_db=Mock(),
            sources=[{"source_type": "note", "source_id": "note-emq"}],
            num_questions=1,
            generation_profile="emq",
        )

    persistence.assert_not_called()


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_persists_exact_planned_type_counts_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Planned Quiz Note",
        content="CPUs execute instructions. RAM stores working data. GPUs accelerate graphics.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=5,
        question_plan=[
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            {"question_type": "multi_select", "count": 1, "option_count": 5},
            {"question_type": "matching", "count": 1, "pair_count": 3},
            {"question_type": "true_false", "count": 1},
            {"question_type": "fill_blank", "count": 1},
        ],
    )

    questions = result["questions"]
    assert [question["question_type"] for question in questions] == [
        "multiple_choice",
        "multi_select",
        "matching",
        "true_false",
        "fill_blank",
    ]
    assert len(questions[0]["options"]) == 5
    assert len(questions[1]["options"]) == 5
    assert len(questions[2]["options"]) == 3
    assert len(questions[2]["correct_answer"]) == 3


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_normalizes_patched_llm_with_question_plan(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_MODE", "1")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [
            {
                "source_type": "note",
                "source_id": "note-plan",
                "text": "CPU means processor. RAM means memory.",
            }
        ],
    )
    monkeypatch.setattr(
        quiz_generator,
        "_resolve_generated_quiz_metadata",
        lambda **_kwargs: ("Planned Quiz", "Generated from a plan."),
    )
    monkeypatch.setattr(
        quiz_generator,
        "_persist_generated_quiz",
        lambda **kwargs: {"quiz": {"title": kwargs["quiz_title"]}, "questions": kwargs["questions"]},
    )

    async def fake_generate_llm(**kwargs):
        captured.update(kwargs)
        return {
            "questions": [
                {
                    "question_type": "multiple_choice",
                    "question_text": "What does CPU mean?",
                    "options": ["Processor", "Memory", "Storage", "Display", "Network"],
                    "correct_answer": "A",
                    "source_citations": [{"source_type": "note", "source_id": "note-plan", "quote": "CPU"}],
                },
                {
                    "question_type": "matching",
                    "question_text": "Match each term.",
                    "options": ["CPU", "RAM"],
                    "correct_answer": {"CPU": "Processor", "RAM": "Memory"},
                    "source_citations": [{"source_type": "note", "source_id": "note-plan", "quote": "CPU"}],
                },
            ]
        }

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_generate_llm)

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-plan"}],
        num_questions=2,
        question_plan=[
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            {"question_type": "matching", "count": 1, "pair_count": 2},
        ],
    )

    assert "Planned question requirements" in str(captured["prompt"])
    assert captured["max_tokens"] == 2000
    assert [question["question_type"] for question in result["questions"]] == ["multiple_choice", "matching"]
    assert len(result["questions"][0]["options"]) == 5
    assert len(result["questions"][1]["correct_answer"]) == 2


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_planned_shortfall_fails_before_persist(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "0")
    note_id = quizzes_db.add_note(title="Shortfall Note", content="A note with enough evidence for generation.")

    async def fake_generate_llm(**_kwargs):
        return {
            "questions": [
                {
                    "question_type": "multiple_choice",
                    "question_text": f"Question {index}?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": 0,
                    "source_citations": [{"source_type": "note", "source_id": str(note_id), "quote": "evidence"}],
                }
                for index in range(4)
            ]
        }

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_generate_llm)

    with pytest.raises(ValueError, match="expected 5, got 4"):
        await generate_quiz_from_sources(
            db=quizzes_db,
            media_db=media_db,
            sources=[{"source_type": "note", "source_id": note_id}],
            num_questions=5,
            question_plan=[{"question_type": "multiple_choice", "count": 5, "option_count": 4}],
        )

    assert quizzes_db.list_quizzes(limit=10, offset=0)["count"] == 0


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_uses_test_mode_fallback_for_metadata_only_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [{"source_type": "note", "source_id": "note-meta", "text": ""}],
    )
    monkeypatch.setattr(
        quiz_generator,
        "_resolve_generated_quiz_metadata",
        lambda **_kwargs: ("Metadata Fallback Quiz", "Generated from metadata-only evidence."),
    )
    monkeypatch.setattr(
        quiz_generator,
        "_persist_generated_quiz",
        lambda **kwargs: {
            "quiz": {
                "title": kwargs["quiz_title"],
                "description": kwargs["quiz_description"],
                "workspace_tag": kwargs["workspace_tag"],
            },
            "questions": kwargs["questions"],
        },
    )

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-meta"}],
        num_questions=1,
        question_types=["multiple_choice"],
        workspace_tag="workspace:test",
    )

    assert result["quiz"]["workspace_tag"] == "workspace:test"
    assert result["questions"][0]["source_citations"][0]["quote"] == "Study point from note:note-meta."


def test_assertion_reasoning_constants_define_canonical_scale() -> None:
    assert quiz_generator.ASSERTION_REASONING_TAG == "assertion_reasoning"
    assert quiz_generator.ASSERTION_REASONING_OPTIONS == ASSERTION_REASONING_OPTIONS


def test_question_tag_normalization_preserves_non_assertion_slash_tags() -> None:
    tags = quiz_generator._coerce_question_tags(
        ["domain/topic", "domain_topic"],
        generation_profile="standard_recall",
    )

    assert tags == ["domain/topic", "domain_topic"]


@pytest.mark.parametrize(
    ("generation_profile", "raw_tags", "expected_tags"),
    [
        ("standard_recall", ["Cardiology", "cardiology"], ["Cardiology"]),
        (
            "best_of_five",
            ["cardiology", "bof", "best-of-five"],
            ["cardiology", "best_of_five"],
        ),
        ("emq", ["diagnosis", "Diagnosis"], ["diagnosis"]),
    ],
)
def test_question_tag_normalization_preserves_existing_profile_behavior(
    generation_profile: str,
    raw_tags: list[str],
    expected_tags: list[str],
) -> None:
    assert (
        quiz_generator._coerce_question_tags(
            raw_tags,
            generation_profile=generation_profile,
        )
        == expected_tags
    )


@pytest.mark.parametrize(
    ("generation_profile", "expected_tags"),
    [
        ("standard_recall", ["cardiology", "hard", "topic/area"]),
        ("mixed_assessment", ["cardiology", "hard", "topic/area"]),
        (
            "best_of_five",
            ["cardiology", "hard", "topic/area", "best_of_five"],
        ),
        ("emq", ["cardiology", "hard", "topic/area"]),
    ],
)
def test_non_assertion_profiles_strip_assertion_reasoning_reserved_tags(
    generation_profile: str,
    expected_tags: list[str],
) -> None:
    tags = quiz_generator._coerce_question_tags(
        [
            "cardiology",
            "hard",
            "assertion_reasoning",
            "assertion reasoning",
            "assertion/reasoning",
            "Assertion / Reasoning",
            "topic/area",
        ],
        generation_profile=generation_profile,
    )

    assert tags == expected_tags


def test_normalize_assertion_reasoning_owns_options_and_canonicalizes_one_subtype_tag() -> None:
    question = _normalize_assertion_reasoning_question(
        tags=[
            "cardiology",
            "Assertion Reasoning",
            "assertion-reasoning",
            "assertion_reasoning",
            "ASSERTION/REASONING",
            "Assertion / Reasoning",
        ]
    )

    assert question["options"] == list(ASSERTION_REASONING_OPTIONS)
    assert question["tags"] == ["cardiology", "assertion_reasoning"]
    assert question["tags"].count("assertion_reasoning") == 1
    assert question["group_id"] is None
    assert question["group_prompt"] is None


@pytest.mark.parametrize("raw_answer", range(5))
def test_normalize_assertion_reasoning_accepts_zero_based_integer_answers(raw_answer: int) -> None:
    question = _normalize_assertion_reasoning_question(correct_answer=raw_answer)

    assert question["correct_answer"] == raw_answer


@pytest.mark.parametrize(
    ("raw_answer", "expected_index"),
    [(letter, index) for index, letter in enumerate((" A ", " b ", " C ", " d ", " E "))],
)
def test_normalize_assertion_reasoning_accepts_trimmed_case_insensitive_letters(
    raw_answer: str,
    expected_index: int,
) -> None:
    question = _normalize_assertion_reasoning_question(correct_answer=raw_answer)

    assert question["correct_answer"] == expected_index


@pytest.mark.parametrize(
    ("raw_answer", "expected_index"),
    [(f"  {label.swapcase()}  ", index) for index, label in enumerate(ASSERTION_REASONING_OPTIONS)],
)
def test_normalize_assertion_reasoning_accepts_trimmed_case_insensitive_labels(
    raw_answer: str,
    expected_index: int,
) -> None:
    question = _normalize_assertion_reasoning_question(correct_answer=raw_answer)

    assert question["correct_answer"] == expected_index


@pytest.mark.parametrize(
    "raw_answer",
    [
        "0",
        "1",
        "2",
        "3",
        "4",
        "A.",
        "a.",
        True,
        False,
        0.0,
        4.0,
        -1,
        5,
        "unknown outcome",
        None,
    ],
)
def test_normalize_assertion_reasoning_rejects_noncanonical_answers(raw_answer: object) -> None:
    with pytest.raises(ValueError, match="correct_answer"):
        _normalize_assertion_reasoning_questions(correct_answer=raw_answer)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("assertion", None),
        ("assertion", "   "),
        ("reason", None),
        ("reason", "   "),
        ("explanation", None),
        ("explanation", "   "),
        ("assertion", "a" * 2001),
        ("reason", "r" * 2001),
        ("explanation", "e" * 2001),
    ],
)
def test_normalize_assertion_reasoning_requires_bounded_text_fields(
    field: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        _normalize_assertion_reasoning_questions(**{field: invalid_value})


def test_normalize_assertion_reasoning_accepts_exact_text_limits() -> None:
    question = _normalize_assertion_reasoning_question(
        assertion="a" * 2000,
        reason="r" * 2000,
        explanation="e" * 2000,
    )

    assert question["question_text"] == (f"**Assertion:** {'a' * 2000}\n\n**Reason:** {'r' * 2000}")
    assert question["explanation"] == "e" * 2000


def test_normalize_assertion_reasoning_rejects_non_mcq_question() -> None:
    with pytest.raises(ValueError, match="multiple_choice"):
        _normalize_assertion_reasoning_questions(
            question_type="true_false",
            correct_answer="true",
        )


def test_normalize_assertion_reasoning_discards_raw_and_unknown_fields() -> None:
    question = _normalize_assertion_reasoning_question(
        question_text="LLM-supplied text must not win.",
        reasoning_steps=["private step"],
        chain_of_thought="private reasoning",
        unknown_payload={"private": True},
    )

    assert question["question_text"] == (
        "**Assertion:** The review board meets every Friday.\n\n"
        "**Reason:** The selected note states that Friday review boards are required."
    )
    assert not {
        "assertion",
        "reason",
        "reasoning_steps",
        "chain_of_thought",
        "unknown_payload",
    }.intersection(question)


def test_build_test_mode_assertion_reasoning_is_source_grounded() -> None:
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-ar",
                "text": "Friday review boards verify every cited claim.",
            }
        ],
        normalized_sources=[{"source_type": "note", "source_id": "note-ar"}],
        num_questions=1,
        question_types=None,
        generation_profile="assertion_reasoning",
    )

    question = questions[0]
    assert question["question_type"] == "multiple_choice"
    assert question["question_text"].startswith("**Assertion:**")
    assert "Friday review boards verify every cited claim." in question["question_text"]
    assert "**Reason:**" in question["question_text"]
    assert question["options"] == list(ASSERTION_REASONING_OPTIONS)
    assert question["correct_answer"] == 0
    assert question["tags"] == ["assertion_reasoning"]
    assert question["explanation"]
    assert question["source_citations"] == [
        {
            "source_type": "note",
            "source_id": "note-ar",
            "label": "Source 1",
            "quote": "Friday review boards verify every cited claim.",
        }
    ]


@pytest.mark.asyncio
async def test_generate_quiz_persists_assertion_reasoning_payload_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Assertion reasoning note",
        content="Friday review boards verify every cited claim.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=1,
        generation_profile="assertion_reasoning",
    )

    question = result["questions"][0]
    assert question["tags"] == ["assertion_reasoning"]
    assert question["question_text"].startswith("**Assertion:**")
    assert "**Reason:**" in question["question_text"]
    assert question["explanation"]
    assert question["source_citations"][0]["source_id"] == note_id


@pytest.mark.asyncio
async def test_generate_quiz_normalizes_assertion_reasoning_llm_payload_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_llm(**_kwargs):
        return {
            "questions": [
                _valid_assertion_reasoning_question(
                    tags=["evidence", "assertion reasoning", "assertion_reasoning"],
                    reasoning_steps=["hidden"],
                    chain_of_thought="hidden",
                    unknown_payload="hidden",
                )
            ]
        }

    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [
            {
                "source_type": "note",
                "source_id": "note-ar",
                "text": "Review boards meet every Friday.",
            }
        ],
    )
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_llm)
    monkeypatch.setattr(quiz_generator, "extract_response_content", lambda raw: raw)
    monkeypatch.setattr(
        quiz_generator,
        "_persist_generated_quiz",
        lambda **kwargs: captured.update(kwargs) or {"quiz": {"id": 1}, "questions": kwargs["questions"]},
    )

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-ar"}],
        num_questions=1,
        generation_profile="assertion_reasoning",
    )

    question = result["questions"][0]
    assert captured["questions"] == result["questions"]
    assert question["tags"] == ["evidence", "assertion_reasoning"]
    assert question["options"] == list(ASSERTION_REASONING_OPTIONS)
    assert question["question_text"].startswith("**Assertion:**")
    assert question["explanation"]
    assert question["source_citations"][0]["source_id"] == "note-ar"
    assert not {"reasoning_steps", "chain_of_thought", "unknown_payload"}.intersection(question)


@pytest.mark.asyncio
async def test_generate_quiz_revalidates_test_mode_assertion_reasoning_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    invalid_question = {
        "question_type": "multiple_choice",
        "question_text": "**Assertion:** A\n\n**Reason:** B",
        "group_id": None,
        "group_prompt": None,
        "options": list(ASSERTION_REASONING_OPTIONS),
        "correct_answer": "0",
        "explanation": "Evidence-backed explanation.",
        "source_citations": [{"source_type": "note", "source_id": "note-ar", "quote": "Evidence."}],
        "tags": ["assertion_reasoning"],
        "points": 1,
    }
    persistence = Mock()
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [{"source_type": "note", "source_id": "note-ar", "text": "Evidence."}],
    )
    monkeypatch.setattr(
        quiz_generator,
        "_build_test_mode_questions",
        lambda **_kwargs: [invalid_question],
    )
    monkeypatch.setattr(quiz_generator, "_persist_generated_quiz", persistence)

    with pytest.raises(ValueError, match="correct_answer"):
        await generate_quiz_from_sources(
            db=Mock(),
            media_db=Mock(),
            sources=[{"source_type": "note", "source_id": "note-ar"}],
            num_questions=1,
            generation_profile="assertion_reasoning",
        )

    persistence.assert_not_called()


@pytest.mark.asyncio
async def test_generate_quiz_revalidates_llm_assertion_reasoning_immediately_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_llm(**_kwargs):
        return {"questions": [_valid_assertion_reasoning_question()]}

    def mutate_after_provenance(questions, _selected_sources):
        questions[0]["correct_answer"] = "0"

    persistence = Mock()
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [
            {
                "source_type": "note",
                "source_id": "note-ar",
                "text": "Review boards meet every Friday.",
            }
        ],
    )
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_llm)
    monkeypatch.setattr(quiz_generator, "extract_response_content", lambda raw: raw)
    monkeypatch.setattr(quiz_generator, "_validate_strict_provenance", mutate_after_provenance)
    monkeypatch.setattr(quiz_generator, "_persist_generated_quiz", persistence)

    with pytest.raises(ValueError, match="correct_answer"):
        await generate_quiz_from_sources(
            db=Mock(),
            media_db=Mock(),
            sources=[{"source_type": "note", "source_id": "note-ar"}],
            num_questions=1,
            generation_profile="assertion_reasoning",
        )

    persistence.assert_not_called()
