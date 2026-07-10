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
async def test_generate_quiz_validates_emq_after_limiting_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = getattr(quiz_generator, "_limit_questions_by_profile", None)
    validator = getattr(quiz_generator, "_validate_emq_groups", None)
    assert limiter is not None
    assert validator is not None

    events: list[tuple[str, int]] = []

    def recording_validator(questions):
        events.append(("validate", len(questions)))
        return validator(questions)

    def recording_limiter(questions, *, num_questions, generation_profile):
        events.append(("limit", len(questions)))
        return limiter(
            questions,
            num_questions=num_questions,
            generation_profile=generation_profile,
        )

    original_provenance_validator = quiz_generator._validate_strict_provenance

    def recording_provenance(questions, selected_sources):
        events.append(("provenance", len(questions)))
        return original_provenance_validator(questions, selected_sources)

    def recording_persistence(**kwargs):
        events.append(("persist", len(kwargs["questions"])))
        return {"quiz": {"id": 1}, "questions": kwargs["questions"]}

    async def fake_llm(**_kwargs):
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
    monkeypatch.setattr(quiz_generator, "_validate_emq_groups", recording_validator)
    monkeypatch.setattr(quiz_generator, "_limit_questions_by_profile", recording_limiter)
    monkeypatch.setattr(quiz_generator, "_validate_strict_provenance", recording_provenance)
    monkeypatch.setattr(quiz_generator, "_persist_generated_quiz", recording_persistence)

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-emq"}],
        num_questions=1,
        generation_profile="emq",
    )

    assert len(result["questions"]) == 2
    assert events == [
        ("validate", 2),
        ("limit", 2),
        ("validate", 2),
        ("provenance", 2),
        ("persist", 2),
    ]


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
