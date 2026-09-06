from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core import exceptions as core_exceptions
from tldw_Server_API.app.core.DB_Management import Prompts_DB as prompts_db_module
from tldw_Server_API.app.core.Prompt_Management import service_prompts as service_prompts_module
from tldw_Server_API.app.core.Prompt_Management.service_prompts import (
    ServicePromptCorruptOverride,
    ServicePromptValidationError,
    UnknownServicePromptDefinition,
    get_service_prompt_definition,
    list_service_prompt_definitions,
    render_service_prompt_part,
    resolve_service_prompt,
    validate_service_prompt_parts,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

EXPECTED_REGISTRY = {
    "study.assistant.explain": {
        "label": "Study explanation",
        "description": "Controls study response guidance. Grounding instructions, study context and provider settings remain fixed.",
        "parts": (("guidance", "Guidance", "literal", ()),),
        "workflows": (
            ("study.assistant.flashcard", "Flashcard Study Assistant"),
            ("study.assistant.quiz", "Quiz Study Assistant"),
        ),
    },
    "study.assistant.mnemonic": {
        "label": "Study mnemonic",
        "description": "Controls study response guidance. Grounding instructions, study context and provider settings remain fixed.",
        "parts": (("guidance", "Guidance", "literal", ()),),
        "workflows": (
            ("study.assistant.flashcard", "Flashcard Study Assistant"),
            ("study.assistant.quiz", "Quiz Study Assistant"),
        ),
    },
    "study.assistant.followup": {
        "label": "Study follow-up",
        "description": "Controls study response guidance. Grounding instructions, study context and provider settings remain fixed.",
        "parts": (("guidance", "Guidance", "literal", ()),),
        "workflows": (
            ("study.assistant.flashcard", "Flashcard Study Assistant"),
            ("study.assistant.quiz", "Quiz Study Assistant"),
        ),
    },
    "study.assistant.freeform": {
        "label": "Study freeform response",
        "description": "Controls study response guidance. Grounding instructions, study context and provider settings remain fixed.",
        "parts": (("guidance", "Guidance", "literal", ()),),
        "workflows": (
            ("study.assistant.flashcard", "Flashcard Study Assistant"),
            ("study.assistant.quiz", "Quiz Study Assistant"),
        ),
    },
    "chat.rag.answer": {
        "label": "RAG answer",
        "description": "Controls how retrieved context and the current question are presented to the model.",
        "parts": (("template", "Template", "template", ("context", "question")),),
        "workflows": (
            ("chat.main.rag", "Main chat RAG"),
            ("chat.tab.rag", "Tab chat RAG"),
            ("chat.document.rag", "Document chat RAG"),
            ("chat.sidepanel.rag", "Sidepanel RAG"),
        ),
    },
    "chat.rag.question_rewrite": {
        "label": "RAG follow-up rewrite",
        "description": "Controls how a conversational follow-up is rewritten into a standalone retrieval query.",
        "parts": (("template", "Template", "template", ("chat_history", "question")),),
        "workflows": (
            ("chat.main.rag", "Main chat RAG"),
            ("chat.document.rag", "Document chat RAG"),
            ("chat.sidepanel.rag", "Sidepanel RAG"),
        ),
    },
    "chat.web_search.answer": {
        "label": "Web-search answer",
        "description": "Controls how normalized web-search results are presented for the final answer.",
        "parts": (("template", "Template", "template", ("current_date_time", "search_results")),),
        "workflows": (
            ("chat.main.web_search", "Main chat web search"),
            ("chat.compare.web_search", "Compare web search"),
        ),
    },
    "chat.title.generation": {
        "label": "Conversation title",
        "description": "Controls the instruction used to generate automatic conversation titles.",
        "parts": (("user_template", "User template", "template", ("query",)),),
        "workflows": (("chat.title.generation", "Automatic conversation titles"),),
    },
    "image.prompt.refinement": {
        "label": "Image prompt refinement",
        "description": "Controls the semantic instructions used to refine image-generation prompt drafts.",
        "parts": (
            ("system_semantics", "Refinement guidance", "literal", ()),
            ("rewrite_semantics", "Rewrite guidance", "literal", ()),
        ),
        "workflows": (("image.prompt.refinement", "Image prompt refinement"),),
    },
    "media.document.summarization": {
        "label": "Document summarization",
        "description": "Controls system instructions for synchronous document analysis. Without a saved override, server defaults apply.",
        "parts": (("system", "System instructions", "literal", ()),),
        "workflows": (("media.document.summarization", "Synchronous document analysis"),),
    },
    "media.pdf.summarization": {
        "label": "PDF summarization",
        "description": "Controls system instructions for synchronous PDF analysis. Without a saved override, server defaults apply.",
        "parts": (("system", "System instructions", "literal", ()),),
        "workflows": (("media.pdf.summarization", "Synchronous PDF analysis"),),
    },
    "media.ebook.summarization": {
        "label": "EPUB summarization",
        "description": "Controls system instructions for synchronous EPUB analysis. Without a saved override, server defaults apply.",
        "parts": (("system", "System instructions", "literal", ()),),
        "workflows": (("media.ebook.summarization", "Synchronous EPUB analysis"),),
    },
    "media.email.summarization": {
        "label": "Email summarization",
        "description": "Controls system instructions for synchronous email analysis. Without a saved override, server defaults apply.",
        "parts": (("system", "System instructions", "literal", ()),),
        "workflows": (("media.email.summarization", "Synchronous email analysis"),),
    },
    "media.audio.analysis": {
        "label": "Audio summarization",
        "description": "Controls system and user instructions for synchronous audio analysis. Without a saved override, server defaults apply.",
        "parts": (("system", "System instructions", "literal", ()), ("user", "User instructions", "literal", ())),
        "workflows": (("media.audio.analysis", "Synchronous audio analysis"),),
    },
    "media.video.summarization": {
        "label": "Video summarization",
        "description": "Controls system instructions and recursive final-summary instructions for synchronous video analysis. Without a saved override, server defaults apply.",
        "parts": (
            ("system", "System instructions", "literal", ()),
            ("final_summary", "Final-summary instructions", "literal", ()),
        ),
        "workflows": (("media.video.summarization", "Synchronous video analysis"),),
    },
    "media.web.summarization": {
        "label": "Web article summarization",
        "description": "Controls summary instructions for synchronous web scraping and web-content ingestion. Reset restores each scraping engine's existing defaults; the displayed defaults are the deployed web-article prompts.",
        "parts": (("system", "System instructions", "literal", ()), ("user", "User instructions", "literal", ())),
        "workflows": (("media.web.summarization", "Synchronous web scraping and ingestion"),),
    },
    "media.document.insights": {
        "label": "Document Insights",
        "description": "Controls analysis and presentation guidance for document insights. JSON output requirements and requested categories remain fixed.",
        "parts": (
            ("analysis_guidance", "Analysis guidance", "literal", ()),
            ("presentation_guidance", "Presentation guidance", "literal", ()),
        ),
        "workflows": (("media.document.insights", "Document workspace insights"),),
    },
    "media.text.translation": {
        "label": "Text translation",
        "description": "Controls the visible instructions used by synchronous text translation.",
        "parts": (
            ("system", "System instructions", "literal", ()),
            ("user_template", "User template", "template", ("target_language", "text")),
        ),
        "workflows": (("media.text.translation", "Text translation"),),
    },
    "notes.title.generate": {
        "label": "Notes title",
        "description": "Controls the wording used by LLM-backed automatic Notes titles.",
        "parts": (
            ("system", "System instructions", "literal", ()),
            ("title_instruction", "Title instruction", "literal", ()),
        ),
        "workflows": (("notes.title.generate", "Automatic Notes titles"),),
    },
}


def test_service_prompt_exceptions_have_one_canonical_definition() -> None:
    for name in (
        "UnknownServicePromptDefinition",
        "ServicePromptValidationError",
        "ServicePromptCorruptOverride",
    ):
        canonical = getattr(core_exceptions, name)
        assert canonical.__module__ == core_exceptions.__name__
        assert getattr(service_prompts_module, name) is canonical

    assert prompts_db_module.DatabaseError is core_exceptions.PromptsDatabaseError
    assert prompts_db_module.ConflictError is core_exceptions.PromptsConflictError
    assert prompts_db_module.ServicePromptRevisionConflict is core_exceptions.ServicePromptRevisionConflict
    assert issubclass(
        core_exceptions.ServicePromptRevisionConflict,
        prompts_db_module.ConflictError,
    )


@dataclass(frozen=True)
class _OverrideRow:
    definition_id: str
    parts_json: str
    revision: str


class _FakePromptsDatabase:
    def __init__(self, row: _OverrideRow | None):
        self.row = row
        self.requested_definition_id: str | None = None

    def get_service_prompt_override(self, definition_id: str) -> _OverrideRow | None:
        self.requested_definition_id = definition_id
        return self.row


def _valid_parts(definition_id: str) -> dict[str, str]:
    return dict(get_service_prompt_definition(definition_id).default_parts)


def _assert_validation_error(definition_id: str, parts: Any, field: str) -> ServicePromptValidationError:
    definition = get_service_prompt_definition(definition_id)
    with pytest.raises(ServicePromptValidationError) as captured:
        validate_service_prompt_parts(definition, parts)
    assert field in captured.value.field_errors
    return captured.value


def test_registry_contains_exact_locked_metadata_and_workflows() -> None:
    definitions = list_service_prompt_definitions()

    assert tuple(definition.id for definition in definitions) == tuple(EXPECTED_REGISTRY)
    for definition in definitions:
        expected = EXPECTED_REGISTRY[definition.id]
        assert definition.label == expected["label"]
        assert definition.description == expected["description"]
        assert (
            tuple((part.key, part.label, part.mode, part.required_variables) for part in definition.parts)
            == expected["parts"]
        )
        assert (
            tuple((workflow.id, workflow.label) for workflow in definition.affected_workflows) == expected["workflows"]
        )


def test_registry_and_resolved_mappings_are_immutable() -> None:
    definition = get_service_prompt_definition("chat.rag.answer")
    resolved = resolve_service_prompt(_FakePromptsDatabase(None), definition.id)

    with pytest.raises(TypeError):
        definition.default_parts["template"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved.parts["template"] = "changed"  # type: ignore[index]


def test_fixture_defaults_equal_registry_defaults_byte_for_byte() -> None:
    actual = {definition.id: dict(definition.default_parts) for definition in list_service_prompt_definitions()}

    assert actual == FIXTURE["defaults"]


def test_every_packaged_default_is_accepted_by_the_import_time_validator() -> None:
    for definition in list_service_prompt_definitions():
        assert validate_service_prompt_parts(definition, definition.default_parts) == dict(definition.default_parts)


def test_unknown_definition_is_rejected() -> None:
    with pytest.raises(UnknownServicePromptDefinition):
        get_service_prompt_definition("chat.not-registered")


def test_missing_part_is_rejected() -> None:
    _assert_validation_error("chat.rag.answer", {}, "template")


def test_extra_part_is_rejected_without_echoing_the_untrusted_key() -> None:
    secret = "PROMPT_BODY_MUST_NOT_APPEAR"
    parts = _valid_parts("chat.rag.answer") | {secret: "extra"}

    error = _assert_validation_error("chat.rag.answer", parts, "_parts")

    assert secret not in str(error)
    assert secret not in repr(error.field_errors)


@pytest.mark.parametrize("value", [None, 42, ["not", "text"], {"not": "text"}])
def test_non_string_part_is_rejected(value: object) -> None:
    _assert_validation_error("chat.rag.answer", {"template": value}, "template")


@pytest.mark.parametrize("value", ["", " ", "\t\n", "\u2003"])
def test_blank_part_is_rejected(value: str) -> None:
    _assert_validation_error("chat.rag.answer", {"template": value}, "template")


def test_part_longer_than_20000_unicode_code_points_is_rejected() -> None:
    _assert_validation_error("chat.rag.answer", {"template": "😀" * 20_001}, "template")


def test_exactly_20000_unicode_code_points_is_accepted() -> None:
    template = "{context}{question}" + ("😀" * (20_000 - len("{context}{question}")))

    assert len(template) == 20_000
    assert validate_service_prompt_parts(get_service_prompt_definition("chat.rag.answer"), {"template": template}) == {
        "template": template
    }


@pytest.mark.parametrize(
    ("name", "template"),
    [
        ("attribute traversal", "{context.value} {question}"),
        ("indexing", "{context[0]} {question}"),
        ("numeric field", "{0} {question}"),
        ("conversion", "{context!r} {question}"),
        ("format specification", "{context:>20} {question}"),
        ("explicitly empty format specification", "{context:} {question}"),
        ("unmatched opening brace", "{context} {question"),
        ("unmatched closing brace", "{context} } {question}"),
        ("nested opening brace", "{context{value}} {question}"),
        ("empty field", "{} {context} {question}"),
        ("unknown variable", "{context} {other}"),
        ("missing variable", "{context}"),
        ("repeated variable", "{context} {question} {question}"),
    ],
)
def test_invalid_template_syntax_is_rejected(name: str, template: str) -> None:
    del name
    _assert_validation_error("chat.rag.answer", {"template": template}, "template")


def test_literal_part_does_not_parse_braces() -> None:
    definition = get_service_prompt_definition("media.text.translation")
    authored_text = "Literal {unmatched and }} braces stay unchanged"
    parts = {
        "system": authored_text,
        "user_template": "Translate to {target_language}: {text}",
    }

    assert validate_service_prompt_parts(definition, parts) == parts
    assert render_service_prompt_part(definition, "system", authored_text, {}) == authored_text


@pytest.mark.parametrize("case", FIXTURE["render_cases"], ids=lambda case: case["name"])
def test_shared_single_pass_render_cases(case: dict[str, Any]) -> None:
    definition = get_service_prompt_definition(case["definition_id"])

    actual = render_service_prompt_part(
        definition,
        case["part_key"],
        case["authored_text"],
        case["values"],
    )

    assert actual == case["expected"]


def test_unknown_render_part_is_rejected_without_prompt_text_in_the_error() -> None:
    secret = "PROMPT_BODY_MUST_NOT_APPEAR"
    definition = get_service_prompt_definition("chat.rag.answer")

    with pytest.raises(ServicePromptValidationError) as captured:
        render_service_prompt_part(definition, "not-registered", secret, {})

    assert secret not in str(captured.value)
    assert secret not in repr(captured.value.field_errors)


def test_validation_errors_do_not_include_prompt_text() -> None:
    secret = "PROMPT_BODY_MUST_NOT_APPEAR"
    error = _assert_validation_error(
        "chat.rag.answer",
        {"template": f"{secret} {{context}}"},
        "template",
    )

    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in repr(error.field_errors)
    with pytest.raises(TypeError):
        error.field_errors["template"] = "changed"  # type: ignore[index]


def test_resolver_uses_packaged_default_when_override_is_absent() -> None:
    db = _FakePromptsDatabase(None)

    resolved = resolve_service_prompt(db, "chat.rag.answer")

    assert db.requested_definition_id == "chat.rag.answer"
    assert resolved.definition is get_service_prompt_definition("chat.rag.answer")
    assert resolved.parts == resolved.definition.default_parts
    assert resolved.source == "packaged"
    assert resolved.revision is None


def test_resolver_prefers_a_valid_user_override() -> None:
    parts = {"template": "Context={context}; Question={question}"}
    row = _OverrideRow("chat.rag.answer", json.dumps(parts), "revision-1")
    db = _FakePromptsDatabase(row)

    resolved = resolve_service_prompt(db, "chat.rag.answer")

    assert resolved.parts == parts
    assert resolved.source == "user"
    assert resolved.revision == "revision-1"
    with pytest.raises(TypeError):
        resolved.parts["template"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "parts_json",
    [
        pytest.param('{"template": "PROMPT_BODY_MUST_NOT_APPEAR"', id="malformed-json"),
        pytest.param(
            json.dumps({"template": "PROMPT_BODY_MUST_NOT_APPEAR"}),
            id="semantically-invalid",
        ),
        pytest.param("[]", id="non-object"),
        pytest.param("9" * 5_000, id="over-limit-integer"),
        pytest.param(("[" * 1_100) + ("]" * 1_100), id="excessive-nesting"),
    ],
)
def test_resolver_rejects_corrupt_override_without_falling_back(parts_json: str) -> None:
    row = _OverrideRow("chat.rag.answer", parts_json, "revision-corrupt")

    with pytest.raises(ServicePromptCorruptOverride) as captured:
        resolve_service_prompt(_FakePromptsDatabase(row), "chat.rag.answer")

    assert captured.value.revision == "revision-corrupt"
    assert "PROMPT_BODY_MUST_NOT_APPEAR" not in str(captured.value)
    assert "PROMPT_BODY_MUST_NOT_APPEAR" not in repr(captured.value)
