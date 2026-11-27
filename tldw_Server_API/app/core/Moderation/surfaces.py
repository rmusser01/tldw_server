"""Moderation surface registry (TASK-13950.27). Pure data.

A surface id names one place the server calls an LLM; the gateway keys moderation on it
(``llm_surface=``) and ``[Moderation] surface.<id> = true|false`` can narrow it.

- ``CALLER_MODERATED_SURFACES``: the caller already moderates (chat); the gateway does not repeat it.
- ``EXEMPT_SURFACES``: never moderated, each with its reason.
- ``MODERATED_SURFACES``: moderated at the gateway.
- ``UNCLASSIFIED_SURFACE``: the id of a call that named no surface; moderated by default.
- ``KNOWN_SURFACES``: the union of the four; the only ids a config key or a literal may use.
"""

from __future__ import annotations

CALLER_MODERATED_SURFACES: frozenset[str] = frozenset({"chat.completions"})

EXEMPT_SURFACES: dict[str, str] = {
    "internal.byok_probe": "fixed max_tokens=1 health probe",
    "internal.first_chat_verifier": "fixed setup prompt",
    "internal.eval_judge": "owner 2026-11-26: measurement; moderating judge I/O changes scores",
    "moderation.classifier": "it is the moderation (AC5 interpretation)",
}

UNCLASSIFIED_SURFACE = "unclassified"

MODERATED_SURFACES: frozenset[str] = frozenset(
    {
        # chat
        "chat.completions.router",
        "chat.prompt_improvement",
        "chat.document_generation",
        "chat.macros",
        "chat.dialogue_workflows",
        "sharing.workspace_chat",
        # messages and character
        "messages",
        "character.completion",
        "character.roleplay_resume",
        "character.router",
        "character.memory_extraction",
        # voice
        "voice.speech_chat",
        "voice.chat_stream",
        "voice.realtime",
        "voice.assistant",
        # workflows, RAG and research
        "workflows.step",
        "rag.generation",
        "rag.auxiliary",
        "research",
        # study and prompts
        "study.assistant",
        "study.packs",
        "study.quizzes",
        "prompt_studio",
        # media and audio
        "media.summarization",
        "media.document_insights",
        "audio.insights",
        # automation, documents and generation
        "scheduled_tasks.automation",
        "chunking.boundaries",
        "claims.extraction",
        "notes.graph_suggestions",
        "rpg.rules",
        "skills.execution",
        "slides.generation",
        "slides.standalone_html",
        # web and writing
        "web.search",
        "web.extraction",
        "writing.manuscript",
        "writing.note_title",
        # everything else
        "vn_play",
        "mcp.model_completion",
        "evaluations",
        "data_tables.generation",
        "explainer",
    }
)

KNOWN_SURFACES: frozenset[str] = (
    CALLER_MODERATED_SURFACES
    | frozenset(EXEMPT_SURFACES)
    | MODERATED_SURFACES
    | frozenset({UNCLASSIFIED_SURFACE})
)
