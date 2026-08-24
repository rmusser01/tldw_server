"""Static definitions, validation, rendering, and resolution for Service Prompts."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from tldw_Server_API.app.core.exceptions import (
    ServicePromptCorruptOverride,
    ServicePromptValidationError,
    UnknownServicePromptDefinition,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase


_MAX_PART_CODE_POINTS = 20_000
_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORMATTER = Formatter()


@dataclass(frozen=True)
class ServicePromptWorkflow:
    """One stable workflow affected by a Service Prompt definition."""

    id: str
    label: str


@dataclass(frozen=True)
class ServicePromptPart:
    """Metadata and variable contract for one editable prompt part."""

    key: str
    label: str
    mode: Literal["literal", "template"]
    required_variables: tuple[str, ...]


@dataclass(frozen=True)
class ServicePromptDefinition:
    """Packaged metadata and defaults for one curated Service Prompt."""

    id: str
    label: str
    description: str
    parts: tuple[ServicePromptPart, ...]
    default_parts: Mapping[str, str]
    affected_workflows: tuple[ServicePromptWorkflow, ...]


@dataclass(frozen=True)
class ResolvedServicePrompt:
    """The effective immutable parts and their non-sensitive provenance."""

    definition: ServicePromptDefinition
    parts: Mapping[str, str]
    source: Literal["user", "packaged"]
    revision: str | None


_RAG_ANSWER_DEFAULT = (
    "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say you don't know. DO NOT try to make up an answer. If the question "
    "is not related to the context, politely respond that you are tuned to only answer questions that are related "
    "to the context.  {context}  Question: {question} Helpful answer:"
)

_RAG_QUESTION_REWRITE_DEFAULT = (
    "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone "
    "question.   Chat History: {chat_history} Follow Up Input: {question} Standalone question:"
)

_WEB_SEARCH_ANSWER_DEFAULT = """You are an AI model who is expert at searching the web and answering user's queries.

Generate a response that is informative and relevant to the user's query based on provided search results. the current date and time are {current_date_time}.

`search-results` block provides knowledge from the web search results. You can use this information to generate a meaningful response.

<search-results>
 {search_results}
</search-results>
"""

_TITLE_GENERATION_USER_TEMPLATE_DEFAULT = (
    "Here is the query:\n\n--------------\n\n{query}\n\n--------------\n\n"
    "Create a concise, 3-5 word phrase as a title for the previous query. Avoid quotation marks or special "
    "formatting. RESPOND ONLY WITH THE TITLE TEXT. ANSWER USING THE SAME LANGUAGE AS THE QUERY.\n\n\n"
    "Examples of titles:\n\nStellar Achievement Celebration\nFamily Bonding Activities\n🇫🇷 Voyage à Paris\n"
    "🍜 Receta de Ramen Casero\nShakespeare Analyse Literarische\n日本の春祭り体験\nДревнегреческая Философия Обзор\n\n"
    "Response:"
)

_IMAGE_PROMPT_REFINEMENT_SYSTEM_DEFAULT = (
    "You refine image-generation prompts. Preserve intent while improving clarity, visual specificity, and composition."
)
_IMAGE_PROMPT_REFINEMENT_REWRITE_DEFAULT = (
    "Rewrite the prompt to be concise, concrete, and generation-ready."
)

_TRANSLATION_SYSTEM_DEFAULT = """You are an expert translator. Your task is to provide accurate,
natural-sounding translations that preserve the original meaning, tone, and formatting.
Do not add explanations or notes - only provide the translation."""

_TRANSLATION_USER_TEMPLATE_DEFAULT = """Translate the following text to {target_language}.
Preserve the original formatting, meaning, and tone.
Only output the translation, no explanations, notes, or additional text.

Text to translate:
{text}"""

_NOTES_TITLE_SYSTEM_DEFAULT = "You are a helpful assistant that writes concise document titles."
_NOTES_TITLE_INSTRUCTION_DEFAULT = "Write a descriptive title"

_DEFINITION_SEQUENCE = (
    ServicePromptDefinition(
        id="chat.rag.answer",
        label="RAG answer",
        description="Controls how retrieved context and the current question are presented to the model.",
        parts=(
            ServicePromptPart(
                key="template",
                label="Template",
                mode="template",
                required_variables=("context", "question"),
            ),
        ),
        default_parts=MappingProxyType({"template": _RAG_ANSWER_DEFAULT}),
        affected_workflows=(
            ServicePromptWorkflow(id="chat.main.rag", label="Main chat RAG"),
            ServicePromptWorkflow(id="chat.tab.rag", label="Tab chat RAG"),
            ServicePromptWorkflow(id="chat.document.rag", label="Document chat RAG"),
            ServicePromptWorkflow(id="chat.sidepanel.rag", label="Sidepanel RAG"),
        ),
    ),
    ServicePromptDefinition(
        id="chat.rag.question_rewrite",
        label="RAG follow-up rewrite",
        description="Controls how a conversational follow-up is rewritten into a standalone retrieval query.",
        parts=(
            ServicePromptPart(
                key="template",
                label="Template",
                mode="template",
                required_variables=("chat_history", "question"),
            ),
        ),
        default_parts=MappingProxyType({"template": _RAG_QUESTION_REWRITE_DEFAULT}),
        affected_workflows=(
            ServicePromptWorkflow(id="chat.main.rag", label="Main chat RAG"),
            ServicePromptWorkflow(id="chat.document.rag", label="Document chat RAG"),
            ServicePromptWorkflow(id="chat.sidepanel.rag", label="Sidepanel RAG"),
        ),
    ),
    ServicePromptDefinition(
        id="chat.web_search.answer",
        label="Web-search answer",
        description="Controls how normalized web-search results are presented for the final answer.",
        parts=(
            ServicePromptPart(
                key="template",
                label="Template",
                mode="template",
                required_variables=("current_date_time", "search_results"),
            ),
        ),
        default_parts=MappingProxyType({"template": _WEB_SEARCH_ANSWER_DEFAULT}),
        affected_workflows=(
            ServicePromptWorkflow(id="chat.main.web_search", label="Main chat web search"),
            ServicePromptWorkflow(id="chat.compare.web_search", label="Compare web search"),
        ),
    ),
    ServicePromptDefinition(
        id="chat.title.generation",
        label="Conversation title",
        description="Controls the instruction used to generate automatic conversation titles.",
        parts=(
            ServicePromptPart(
                key="user_template",
                label="User template",
                mode="template",
                required_variables=("query",),
            ),
        ),
        default_parts=MappingProxyType(
            {"user_template": _TITLE_GENERATION_USER_TEMPLATE_DEFAULT}
        ),
        affected_workflows=(
            ServicePromptWorkflow(
                id="chat.title.generation",
                label="Automatic conversation titles",
            ),
        ),
    ),
    ServicePromptDefinition(
        id="image.prompt.refinement",
        label="Image prompt refinement",
        description="Controls the semantic instructions used to refine image-generation prompt drafts.",
        parts=(
            ServicePromptPart(
                key="system_semantics",
                label="Refinement guidance",
                mode="literal",
                required_variables=(),
            ),
            ServicePromptPart(
                key="rewrite_semantics",
                label="Rewrite guidance",
                mode="literal",
                required_variables=(),
            ),
        ),
        default_parts=MappingProxyType(
            {
                "system_semantics": _IMAGE_PROMPT_REFINEMENT_SYSTEM_DEFAULT,
                "rewrite_semantics": _IMAGE_PROMPT_REFINEMENT_REWRITE_DEFAULT,
            }
        ),
        affected_workflows=(
            ServicePromptWorkflow(
                id="image.prompt.refinement",
                label="Image prompt refinement",
            ),
        ),
    ),
    ServicePromptDefinition(
        id="media.text.translation",
        label="Text translation",
        description="Controls the visible instructions used by synchronous text translation.",
        parts=(
            ServicePromptPart(
                key="system",
                label="System instructions",
                mode="literal",
                required_variables=(),
            ),
            ServicePromptPart(
                key="user_template",
                label="User template",
                mode="template",
                required_variables=("target_language", "text"),
            ),
        ),
        default_parts=MappingProxyType(
            {
                "system": _TRANSLATION_SYSTEM_DEFAULT,
                "user_template": _TRANSLATION_USER_TEMPLATE_DEFAULT,
            }
        ),
        affected_workflows=(ServicePromptWorkflow(id="media.text.translation", label="Text translation"),),
    ),
    ServicePromptDefinition(
        id="notes.title.generate",
        label="Notes title",
        description="Controls the wording used by LLM-backed automatic Notes titles.",
        parts=(
            ServicePromptPart(
                key="system",
                label="System instructions",
                mode="literal",
                required_variables=(),
            ),
            ServicePromptPart(
                key="title_instruction",
                label="Title instruction",
                mode="literal",
                required_variables=(),
            ),
        ),
        default_parts=MappingProxyType(
            {
                "system": _NOTES_TITLE_SYSTEM_DEFAULT,
                "title_instruction": _NOTES_TITLE_INSTRUCTION_DEFAULT,
            }
        ),
        affected_workflows=(
            ServicePromptWorkflow(
                id="notes.title.generate",
                label="Automatic Notes titles",
            ),
        ),
    ),
)

_DEFINITIONS: Mapping[str, ServicePromptDefinition] = MappingProxyType(
    {definition.id: definition for definition in _DEFINITION_SEQUENCE}
)


class _TemplateSyntaxError(ValueError):
    """Internal error carrying only a safe validation message."""


def _check_template_braces(authored_text: str) -> None:
    """Reject malformed braces and field modifiers without echoing authored text."""

    index = 0
    while index < len(authored_text):
        character = authored_text[index]
        if character == "{":
            if index + 1 < len(authored_text) and authored_text[index + 1] == "{":
                index += 2
                continue

            index += 1
            while index < len(authored_text) and authored_text[index] != "}":
                if authored_text[index] == "{":
                    raise _TemplateSyntaxError("Template has malformed braces.")
                if authored_text[index] in (":", "!"):
                    raise _TemplateSyntaxError("Template fields cannot use conversions or format specifications.")
                index += 1
            if index == len(authored_text):
                raise _TemplateSyntaxError("Template has malformed braces.")
            index += 1
            continue

        if character == "}":
            if index + 1 < len(authored_text) and authored_text[index + 1] == "}":
                index += 2
                continue
            raise _TemplateSyntaxError("Template has malformed braces.")

        index += 1


def _parse_template(
    part: ServicePromptPart,
    authored_text: str,
) -> tuple[tuple[str, str | None, str | None, str | None], ...]:
    """Parse one template and enforce its exact registered variable contract."""

    _check_template_braces(authored_text)
    try:
        parsed = tuple(_FORMATTER.parse(authored_text))
    except ValueError:
        raise _TemplateSyntaxError("Template has malformed braces.") from None

    fields: list[str] = []
    for _, field, format_spec, conversion in parsed:
        if field is None:
            continue
        if not _FIELD_NAME_PATTERN.fullmatch(field):
            raise _TemplateSyntaxError("Template fields must be simple ASCII identifiers.")
        if conversion is not None or format_spec:
            raise _TemplateSyntaxError("Template fields cannot use conversions or format specifications.")
        fields.append(field)

    if Counter(fields) != Counter(part.required_variables):
        raise _TemplateSyntaxError("Template variables must match the registered variables exactly once.")
    return parsed


def get_service_prompt_definition(definition_id: str) -> ServicePromptDefinition:
    """Return one registered definition or raise a safe domain error."""

    try:
        return _DEFINITIONS[definition_id]
    except KeyError:
        raise UnknownServicePromptDefinition(definition_id) from None


def list_service_prompt_definitions() -> tuple[ServicePromptDefinition, ...]:
    """Return all registered definitions in stable display order."""

    return _DEFINITION_SEQUENCE


def validate_service_prompt_parts(
    definition: ServicePromptDefinition,
    parts: Mapping[str, object],
) -> dict[str, str]:
    """Validate and copy a complete set of parts for one definition."""

    if not isinstance(parts, Mapping):
        raise ServicePromptValidationError({"_parts": "Parts must be an object."})

    expected_keys = {part.key for part in definition.parts}
    provided_keys = set(parts)
    field_errors: dict[str, str] = {}
    for part in definition.parts:
        if part.key not in provided_keys:
            field_errors[part.key] = "Part is required."
    if provided_keys - expected_keys:
        field_errors["_parts"] = "Parts contain one or more unregistered keys."
    if field_errors:
        raise ServicePromptValidationError(field_errors)

    validated: dict[str, str] = {}
    for part in definition.parts:
        value = parts[part.key]
        if not isinstance(value, str):
            field_errors[part.key] = "Part must be a string."
            continue
        if not value.strip():
            field_errors[part.key] = "Part must contain non-whitespace text."
            continue
        if len(value) > _MAX_PART_CODE_POINTS:
            field_errors[part.key] = "Part must be at most 20000 Unicode code points."
            continue
        if part.mode == "template":
            try:
                _parse_template(part, value)
            except _TemplateSyntaxError as exc:
                field_errors[part.key] = str(exc)
                continue
        validated[part.key] = value

    if field_errors:
        raise ServicePromptValidationError(field_errors)
    return validated


def render_service_prompt_part(
    definition: ServicePromptDefinition,
    part_key: str,
    authored_text: str,
    values: Mapping[str, str],
) -> str:
    """Render one validated template once, inserting runtime values literally."""

    part = next((candidate for candidate in definition.parts if candidate.key == part_key), None)
    if part is None:
        raise ServicePromptValidationError({"_parts": "Part key is not registered."})
    if part.mode == "literal":
        return authored_text

    try:
        parsed = _parse_template(part, authored_text)
    except _TemplateSyntaxError as exc:
        raise ServicePromptValidationError({part.key: str(exc)}) from None

    rendered: list[str] = []
    for literal, field, _, _ in parsed:
        rendered.append(literal)
        if field is not None:
            try:
                value = values[field]
            except KeyError:
                raise ServicePromptValidationError(
                    {part.key: "Render values are missing a required variable."}
                ) from None
            if not isinstance(value, str):
                raise ServicePromptValidationError({part.key: "Render values must be strings."})
            rendered.append(value)
    return "".join(rendered)


def resolve_service_prompt(
    db: PromptsDatabase,
    definition_id: str,
) -> ResolvedServicePrompt:
    """Resolve a valid saved override, otherwise the packaged default."""

    definition = get_service_prompt_definition(definition_id)
    override = db.get_service_prompt_override(definition_id)
    if override is None:
        return ResolvedServicePrompt(
            definition=definition,
            parts=definition.default_parts,
            source="packaged",
            revision=None,
        )

    try:
        decoded = json.loads(override.parts_json)
        if not isinstance(decoded, Mapping):
            raise ServicePromptValidationError({"_parts": "Parts must be an object."})
        validated = validate_service_prompt_parts(definition, decoded)
    except (ValueError, TypeError, RecursionError):
        raise ServicePromptCorruptOverride(override.revision) from None

    return ResolvedServicePrompt(
        definition=definition,
        parts=MappingProxyType(validated),
        source="user",
        revision=override.revision,
    )


for _definition in _DEFINITION_SEQUENCE:
    validate_service_prompt_parts(_definition, _definition.default_parts)
del _definition


__all__ = [
    "ResolvedServicePrompt",
    "ServicePromptCorruptOverride",
    "ServicePromptDefinition",
    "ServicePromptPart",
    "ServicePromptValidationError",
    "ServicePromptWorkflow",
    "UnknownServicePromptDefinition",
    "get_service_prompt_definition",
    "list_service_prompt_definitions",
    "render_service_prompt_part",
    "resolve_service_prompt",
    "validate_service_prompt_parts",
]
