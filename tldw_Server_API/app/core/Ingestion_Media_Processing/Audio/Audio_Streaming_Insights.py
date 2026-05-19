"""
Audio_Streaming_Insights.py
-------------------------------------------------
Live meeting insight generation for streaming transcription sessions.

This module augments the real-time transcription pipeline with dynamic
meeting summaries, action items, and decision tracking similar to
granola-style assistants. It consumes finalized transcript segments and,
optionally, the full transcript at the end of a session to produce
structured insights via the existing chat LLM abstraction layer.
"""

from __future__ import annotations

import asyncio
import configparser
import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    get_adapter_or_raise,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)

LIVE_INSIGHTS_RUNTIME_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
)

LIVE_SYSTEM_PROMPT = (
    "You are a meticulous meeting notes assistant. "
    "Given a chronological transcript excerpt, identify the most important new information "
    "and express it as objective, factual notes."
)

LIVE_PROMPT_TEMPLATE = """
You are producing real-time insights for an ongoing meeting.

Instructions:
- Work only with the transcript excerpt below (chronological order).
- Focus on NEW information since the previous summary.
- Use concise, professional language.
- If a field should be empty, return an empty array.

Return a JSON object with the following keys:
  "summary": array of 1-5 bullet strings summarizing the excerpt.
  "action_items": array of objects with keys "description" and "owner".
                  Owner should be null if not specified in the transcript.
  "decisions": array of concise statements describing decisions or approvals.
  "topics": array of short topic labels (strings).

{action_instruction}
{decision_instruction}
{topics_instruction}

Transcript excerpt:
{transcript}
"""

FINAL_PROMPT_TEMPLATE = """
You are producing the definitive meeting notes for the entire session.

Summarize the full meeting transcript below using the same JSON schema:
  "summary": array of concise bullet strings describing the overall meeting.
  "action_items": array of objects with keys "description" and "owner".
  "decisions": array of concise statements describing final decisions or agreements.
  "topics": array of high-level topic labels covering the main areas discussed.

Ensure the notes are comprehensive, well-organized, and avoid redundancy.
If a category has no content, return an empty array.

Transcript:
{transcript}
"""


@dataclass
class LiveInsightSettings:
    """Configuration for live insight generation."""

    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    summary_interval_seconds: float = 90.0
    context_window_segments: int = 6
    max_context_chars: int = 6000
    final_summary_max_chars: int = 12000
    generate_action_items: bool = True
    generate_decisions: bool = True
    emit_topics: bool = True
    live_updates: bool = True
    final_summary: bool = True
    response_format_json: bool = True
    temperature: float = 0.2

    @classmethod
    def from_client_payload(cls, payload: dict[str, Any]) -> LiveInsightSettings:
        """Build settings from a client-supplied configuration dictionary."""
        settings = cls()
        if not isinstance(payload, dict):
            return settings

        if "enabled" in payload:
            settings.enabled = bool(payload.get("enabled"))
        else:
            settings.enabled = True  # payload present implies explicit opt-in

        if payload.get("provider"):
            settings.provider = str(payload["provider"]).strip()
        if payload.get("model"):
            settings.model = str(payload["model"]).strip()

        def _optional_float(key: str, default: float) -> float:
            value = payload.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        settings.summary_interval_seconds = max(
            0.0, _optional_float("summary_interval_seconds", settings.summary_interval_seconds)
        )
        settings.max_context_chars = max(
            0, int(payload.get("max_context_chars", settings.max_context_chars) or settings.max_context_chars)
        )
        settings.final_summary_max_chars = max(
            0,
            int(payload.get("final_summary_max_chars", settings.final_summary_max_chars) or settings.final_summary_max_chars),
        )
        settings.context_window_segments = max(
            1,
            int(payload.get("context_window_segments", settings.context_window_segments) or settings.context_window_segments),
        )
        settings.temperature = max(
            0.0, _optional_float("temperature", settings.temperature)
        )

        for key, attr in [
            ("generate_action_items", "generate_action_items"),
            ("generate_decisions", "generate_decisions"),
            ("emit_topics", "emit_topics"),
            ("live_updates", "live_updates"),
            ("final_summary", "final_summary"),
            ("response_format_json", "response_format_json"),
        ]:
            if key in payload:
                setattr(settings, attr, bool(payload.get(key)))

        return settings


class LiveMeetingInsights:
    """Generate live and final meeting summaries from streaming transcription segments."""

    def __init__(
        self,
        websocket,
        settings: LiveInsightSettings,
        *,
        chat_call: Callable[..., Any] | None = None,
    ):
        self.websocket = websocket
        self.settings = settings
        self._chat_call = chat_call or self._adapter_chat_call
        self._app_config = ensure_app_config()
        self._segments: list[dict[str, Any]] = []
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self._closed = False
        self._last_summary_segment_id = 0
        self._last_summary_end = 0.0
        self._insight_id = 0

        provider, model = self._resolve_provider_and_model()
        self.provider = provider
        self.model = model
        self.api_endpoint = provider.lower()

    @staticmethod
    def _adapter_chat_call(
        *,
        api_endpoint: str,
        messages_payload: list[dict[str, Any]],
        model: str | None,
        temp: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        app_config: dict[str, Any] | None = None,
        **extra_kwargs: Any,
    ) -> Any:
        provider_name = normalize_provider(api_endpoint)
        if not provider_name:
            raise ChatConfigurationError(provider=api_endpoint, message="LLM provider is required.")
        cfg = ensure_app_config(app_config)
        resolved_model = model or resolve_provider_model(provider_name, cfg)
        if not resolved_model:
            raise ChatConfigurationError(provider=provider_name, message="Model is required for provider.")
        system_message, cleaned_messages = split_system_message(messages_payload or [])
        request: dict[str, Any] = {
            "messages": cleaned_messages,
            "system_message": system_message,
            "model": resolved_model,
            "api_key": resolve_provider_api_key_from_config(provider_name, cfg),
            "temperature": temp,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "app_config": cfg,
        }
        request.update(extra_kwargs)
        return get_adapter_or_raise(provider_name).chat(request)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        """Return a snapshot of the active insight configuration."""
        return {
            "enabled": self.settings.enabled,
            "provider": self.provider,
            "model": self.model,
            "summary_interval_seconds": self.settings.summary_interval_seconds,
            "context_window_segments": self.settings.context_window_segments,
            "max_context_chars": self.settings.max_context_chars,
            "final_summary_max_chars": self.settings.final_summary_max_chars,
            "live_updates": self.settings.live_updates,
            "final_summary": self.settings.final_summary,
        }

    async def on_transcript(self, segment: dict[str, Any]) -> None:
        """Ingest a finalized transcript segment."""
        if not self.settings.enabled or not segment or not segment.get("is_final", False):
            return

        self._segments.append(segment)

        if not self.settings.live_updates or self._closed:
            return

        if self._should_emit_live(segment):
            self._schedule_task(self._emit_live_update)

    async def on_commit(self, full_transcript: str | None) -> None:
        """Generate the final meeting summary when the stream is committed."""
        if not self.settings.enabled or not self.settings.final_summary or self._closed:
            return

        await self._drain_pending()
        async with self._lock:
            segments = self._select_segments(stage="final")
            transcript_override = self._truncate_text(
                full_transcript or self._build_transcript_text(segments),
                self.settings.final_summary_max_chars,
            )
            if not transcript_override.strip():
                return
            await self._generate_and_send(
                segments,
                stage="final",
                transcript_override=transcript_override,
            )

    async def close(self) -> None:
        """Cancel outstanding insight tasks and mark the generator as closed."""
        self._closed = True
        await self._drain_pending(cancel=True)

    async def reset(self) -> None:
        """Reset accumulated state for a new live session."""
        await self._drain_pending()
        self._segments.clear()
        self._last_summary_segment_id = 0
        self._last_summary_end = 0.0
        self._insight_id = 0

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _resolve_provider_and_model(self) -> tuple[str, str]:
        """Resolve provider/model defaults from config when not supplied by client."""
        provider = (self.settings.provider or "").strip().lower()
        model = (self.settings.model or "").strip()
        config = None
        try:
            config = load_comprehensive_config()
        except (OSError, RuntimeError, ValueError, TypeError, configparser.Error) as err:
            logger.debug(f"LiveMeetingInsights: unable to load config for defaults: {err}")

        if not provider:
            candidate = None
            if config is not None:
                try:
                    candidate = config.get("Chat-API", "default_chat_provider", fallback=None)
                except (configparser.Error, AttributeError, TypeError, ValueError):
                    candidate = None
            provider = (candidate or "openai").lower()

        if not model and config is not None:
            section_map = {
                "openai": "openai_api",
                "anthropic": "anthropic_api",
                "groq": "groq_api",
                "cohere": "cohere_api",
                "mistral": "mistral_api",
                "google": "google_api",
                "qwen": "qwen_api",
            }
            section = section_map.get(provider)
            if section and config.has_section(section):
                with contextlib.suppress(configparser.Error, AttributeError, TypeError, ValueError):
                    model = config.get(section, "model", fallback=model)

        if not model:
            default_models = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-4.5",
                "groq": "mixtral-8x7b-32768",
                "cohere": "command-r-plus",
                "mistral": "mistral-large-latest",
                "google": "gemini-2.5-flash",
                "qwen": "qwen2-72b-instruct",
            }
            model = default_models.get(provider, "gpt-4o-mini")

        return provider, model

    def _schedule_task(self, coro_func) -> None:
        if self._closed:
            return
        task = self._loop.create_task(coro_func())
        self._pending.add(task)
        def _on_done(done_task: asyncio.Task) -> None:
            self._pending.discard(done_task)
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            except (asyncio.InvalidStateError, RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - defensive
                logger.debug(f"LiveMeetingInsights: task exception check failed: {exc}")
                return
            if exc:
                logger.debug(f"LiveMeetingInsights: background task failed: {exc}")

        task.add_done_callback(_on_done)

    async def _emit_live_update(self) -> None:
        async with self._lock:
            segments = self._select_segments(stage="live")
            if not segments:
                return
            await self._generate_and_send(segments, stage="live")

    def _select_segments(self, stage: str) -> list[dict[str, Any]]:
        if not self._segments:
            return []

        if stage == "live":
            new_segments = [
                seg
                for seg in self._segments
                if int(seg.get("segment_id", 0)) > self._last_summary_segment_id
            ]
            if not new_segments:
                return []
            window = max(1, self.settings.context_window_segments)
            limited = new_segments[-window:]
            return self._truncate_segments_by_chars(limited, self.settings.max_context_chars)

        # Final summary uses the entire session (with optional truncation)
        return self._truncate_segments_by_chars(list(self._segments), self.settings.final_summary_max_chars)

    def _truncate_segments_by_chars(
        self,
        segments: Sequence[dict[str, Any]],
        limit_chars: int,
    ) -> list[dict[str, Any]]:
        if limit_chars <= 0:
            return list(segments)

        total = 0
        selected: list[dict[str, Any]] = []
        for segment in reversed(segments):
            text = str(segment.get("text") or "")
            total += len(text)
            selected.append(segment)
            if total >= limit_chars:
                break
        return list(reversed(selected))

    async def _generate_and_send(
        self,
        segments: list[dict[str, Any]],
        *,
        stage: str,
        transcript_override: str | None = None,
    ) -> None:
        transcript = transcript_override or self._build_transcript_text(segments)
        transcript = self._truncate_text(
            transcript,
            self.settings.max_context_chars if stage == "live" else self.settings.final_summary_max_chars,
        )
        if not transcript.strip():
            return

        try:
            response_payload = await self._call_llm(transcript, stage=stage)
        except (ChatConfigurationError, *LIVE_INSIGHTS_RUNTIME_EXCEPTIONS) as exc:
            logger.error(f"LiveMeetingInsights: LLM request failed ({stage}): {exc}")
            await self._safe_send(
                {
                    "type": "insight_error",
                    "stage": stage,
                    "message": "Live insights request failed",
                }
            )
            return

        message = self._build_insight_message(response_payload, segments, stage)
        if message:
            await self._safe_send(message)
            if segments:
                last_seg = segments[-1]
                self._last_summary_end = float(last_seg.get("segment_end") or self._last_summary_end)
                self._last_summary_segment_id = int(
                    last_seg.get("segment_id") or self._last_summary_segment_id
                )

    async def _call_llm(self, transcript_text: str, *, stage: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": LIVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_prompt(transcript_text, stage=stage),
            },
        ]
        response_format = {"type": "json_object"} if self.settings.response_format_json else None
        max_tokens = 800 if stage == "final" else 512

        def _invoke():
            return self._chat_call(
                api_endpoint=self.api_endpoint,
                messages_payload=messages,
                model=self.model,
                temp=self.settings.temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                app_config=self._app_config,
            )

        raw_response = await self._loop.run_in_executor(None, _invoke)
        content = extract_response_content(raw_response)
        if content is None:
            content = json.dumps(raw_response) if isinstance(raw_response, dict) else str(raw_response)
        return {"raw": raw_response, "content": content}

    def _build_prompt(self, transcript_text: str, *, stage: str) -> str:
        if stage == "final":
            return FINAL_PROMPT_TEMPLATE.format(transcript=transcript_text)

        action_instruction = (
            "Identify concrete action items and assign the owner if the transcript names someone."
            if self.settings.generate_action_items
            else "If no explicit tasks are present, return an empty array for action_items."
        )
        decision_instruction = (
            "Document any clear decisions, approvals, or conclusions."
            if self.settings.generate_decisions
            else "If no decisions are mentioned, return an empty array for decisions."
        )
        topics_instruction = (
            "Provide short topic labels capturing the themes in the excerpt."
            if self.settings.emit_topics
            else "Return an empty array for topics."
        )
        return LIVE_PROMPT_TEMPLATE.format(
            action_instruction=action_instruction,
            decision_instruction=decision_instruction,
            topics_instruction=topics_instruction,
            transcript=transcript_text,
        )

    def _build_transcript_text(self, segments: Sequence[dict[str, Any]]) -> str:
        parts: list[str] = []
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            start = seg.get("segment_start")
            end = seg.get("segment_end")
            speaker = seg.get("speaker")
            prefix_parts = []
            if start is not None and end is not None:
                prefix_parts.append(f"[{float(start):0.1f}s-{float(end):0.1f}s]")
            if speaker:
                prefix_parts.append(f"{speaker}:")
            prefix = " ".join(prefix_parts)
            if prefix:
                parts.append(f"{prefix} {text}")
            else:
                parts.append(text)
        return "\n".join(parts)

    def _build_insight_message(
        self,
        response_payload: dict[str, Any],
        segments: Sequence[dict[str, Any]],
        stage: str,
    ) -> dict[str, Any] | None:
        raw_text = response_payload.get("content") or ""
        parsed = self._parse_json_response(raw_text)
        if parsed is None:
            parsed = {
                "summary": [raw_text.strip()] if raw_text.strip() else [],
                "action_items": [],
                "decisions": [],
                "topics": [],
            }

        self._insight_id += 1
        message = {
            "type": "insight",
            "insight_id": self._insight_id,
            "stage": stage,
            "provider": self.provider,
            "model": self.model,
            "summary": self._coerce_string_list(parsed.get("summary")),
            "action_items": self._coerce_action_items(parsed.get("action_items")),
            "decisions": self._coerce_string_list(parsed.get("decisions")),
            "topics": self._coerce_string_list(parsed.get("topics")),
            "source": self._source_metadata(segments),
        }
        return message

    def _parse_json_response(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None
        except (TypeError, ValueError):
            return None
        return None

    def _coerce_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        result.append(item)
                elif item is not None:
                    result.append(str(item))
            return result
        return [str(value)]

    def _coerce_action_items(self, value: Any) -> list[dict[str, Any]]:
        if not value:
            return []
        items: list[dict[str, Any]] = []
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    description = str(item.get("description") or item.get("text") or "").strip()
                    owner_raw = item.get("owner") or item.get("assignee")
                    owner = str(owner_raw).strip() if owner_raw else None
                else:
                    description = str(item).strip()
                    owner = None
                if description:
                    items.append({"description": description, "owner": owner})
        return items

    def _source_metadata(self, segments: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not segments:
            return {}
        first = segments[0]
        last = segments[-1]
        return {
            "segment_range": [
                int(first.get("segment_id", 0)),
                int(last.get("segment_id", 0)),
            ],
            "start": float(first.get("chunk_start") or first.get("segment_start") or 0.0),
            "end": float(last.get("chunk_end") or last.get("segment_end") or 0.0),
            "total_segments": len(self._segments),
        }

    async def _safe_send(self, payload: dict[str, Any]) -> None:
        try:
            await self.websocket.send_json(payload)
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.error(f"LiveMeetingInsights: failed to send payload: {exc}")

    async def _drain_pending(self, cancel: bool = False) -> None:
        tasks = list(self._pending)
        if cancel:
            for task in tasks:
                task.cancel()
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                logger.debug(f"LiveMeetingInsights: background task finished with {res}")

    def _should_emit_live(self, latest_segment: dict[str, Any]) -> bool:
        if self.settings.summary_interval_seconds <= 0:
            return True
        end_time = float(latest_segment.get("segment_end") or latest_segment.get("chunk_end") or 0.0)
        if end_time - self._last_summary_end >= self.settings.summary_interval_seconds:
            return True
        current_id = int(latest_segment.get("segment_id", 0))
        return (current_id - self._last_summary_segment_id) >= self.settings.context_window_segments

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return text[-limit:]


__all__ = ["LiveInsightSettings", "LiveMeetingInsights"]
