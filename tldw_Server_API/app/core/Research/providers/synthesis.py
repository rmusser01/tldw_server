"""LLM-backed synthesis provider for deep research."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content

ChatFn = Callable[..., Awaitable[Any]]


def _render_sources(source_registry: list[Any], *, max_items: int = 12) -> str:
    lines: list[str] = []
    for source in source_registry[:max_items]:
        title = getattr(source, "title", "") or "Untitled source"
        source_id = getattr(source, "source_id", "")
        focus_area = getattr(source, "focus_area", "")
        snippet = getattr(source, "snippet", "") or ""
        lines.append(f"- {source_id} | {focus_area} | {title} | {snippet}")
    return "\n".join(lines)


def _render_notes(evidence_notes: list[Any], *, max_items: int = 20) -> str:
    lines: list[str] = []
    for note in evidence_notes[:max_items]:
        note_id = getattr(note, "note_id", "")
        source_id = getattr(note, "source_id", "")
        focus_area = getattr(note, "focus_area", "")
        text = getattr(note, "text", "") or ""
        lines.append(f"- {note_id} | {source_id} | {focus_area} | {text}")
    return "\n".join(lines)


class SynthesisProvider:
    """Generate structured synthesis payloads from collected research evidence."""

    def __init__(self, *, chat_fn: ChatFn = perform_chat_api_call_async) -> None:
        self._chat_fn = chat_fn

    async def summarize(
        self,
        *,
        plan: Any,
        source_registry: list[Any],
        evidence_notes: list[Any],
        collection_summary: dict[str, Any] | None,
        config: dict[str, Any],
    ) -> Any:
        provider = str(config.get("provider") or "").strip()
        model = str(config.get("model") or "").strip()
        if not provider or not model:
            raise ValueError("missing synthesis provider configuration")

        system_message = (
            "You are a careful research synthesis engine. "
            "Return JSON only and only cite source_ids and note_ids that are present in the prompt."
        )
        user_prompt = (
            f"Question: {plan.query}\n"
            f"Focus areas: {', '.join(plan.focus_areas)}\n"
            f"Available sources:\n{_render_sources(source_registry)}\n\n"
            f"Available evidence notes:\n{_render_notes(evidence_notes)}\n\n"
            f"Collection summary: {collection_summary or {}}\n\n"
            "Return a JSON object with keys: outline_sections, claims, report_sections, "
            "unresolved_questions, summary.\n"
            "Each outline section must contain title, focus_area, source_ids, note_ids.\n"
            "Each claim must contain text, focus_area, source_ids, citations, confidence.\n"
            "Each report section must contain title and markdown."
        )
        thinking = config.get("llamacpp_enable_thinking")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError("llamacpp_enable_thinking must be a boolean or unset")
        extensions = {}
        if thinking is not None and get_registry().resolve_provider_name(provider) == "llama.cpp":
            extensions["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
        temperature = config.get("temperature")
        response = await self._chat_fn(
            messages=[{"role": "user", "content": user_prompt}],
            api_provider=provider,
            model=model,
            system_message=system_message,
            temperature=float(0.2 if temperature is None else temperature),
            max_tokens=int(config.get("max_tokens", 2500) or 2500),
            **extensions,
        )
        choices = response.get("choices") if isinstance(response, dict) else None
        if (
            isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            and choices[0].get("finish_reason") == "length"
        ):
            raise ValueError("synthesis_output_truncated")
        return extract_openai_content(response) or response


__all__ = ["SynthesisProvider"]
