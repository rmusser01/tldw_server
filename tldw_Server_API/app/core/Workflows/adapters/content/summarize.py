"""Summarization adapter.

This module includes the summarization adapter.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.content._config import SummarizeConfig


@registry.register(
    "summarize",
    category="content",
    description="Summarize text content",
    parallelizable=True,
    tags=["content", "summarization"],
    config_model=SummarizeConfig,
)
async def run_summarize_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Summarize text using LLM with optional chunking strategies.

    Config:
      - text: Optional[str] (templated, defaults to last.text)
      - custom_prompt: Optional[str] (templated) - additional instructions
      - api_name: Optional[str] - LLM provider (defaults to 'openai')
      - system_message: Optional[str] (templated) - system message override
      - temperature: float = 0.7
      - recursive_summarization: bool = False
      - chunked_summarization: bool = False
      - chunk_options: Optional[Dict] - chunking configuration
    Output:
      - {"summary": str, "text": str, "api_name": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # Template rendering for text
    text_t = str(config.get("text") or "").strip()
    if text_t:
        text = apply_template_to_string(text_t, context) or text_t
    else:
        # Default to last.text
        text = None
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                text = str(last.get("text") or last.get("content") or last.get("summary") or "")
        except Exception as text_context_error:
            logger.debug("Summarize adapter failed to read text from context fallback", exc_info=text_context_error)
    text = text or ""

    if not text.strip():
        return {"error": "missing_text", "summary": "", "text": ""}

    # Template other fields
    custom_prompt = None
    custom_prompt_t = config.get("custom_prompt")
    if custom_prompt_t:
        custom_prompt = apply_template_to_string(str(custom_prompt_t), context) or str(custom_prompt_t)

    system_message = None
    system_message_t = config.get("system_message")
    if system_message_t:
        system_message = apply_template_to_string(str(system_message_t), context) or str(system_message_t)

    api_name = str(config.get("api_name") or "openai").strip().lower()
    temperature = float(config.get("temperature") or 0.7)
    recursive_summarization = bool(config.get("recursive_summarization"))
    chunked_summarization = bool(config.get("chunked_summarization"))
    chunk_options = config.get("chunk_options")

    # Test mode simulation
    if is_test_mode():
        # Simulate summarization by truncating
        simulated_summary = text[:200] + "..." if len(text) > 200 else text
        simulated_summary = f"[Summary of {len(text)} chars] {simulated_summary}"
        return {
            "summary": simulated_summary,
            "text": simulated_summary,  # Alias for chaining
            "api_name": api_name,
            "input_length": len(text),
            "output_length": len(simulated_summary),
            "simulated": True,
        }

    try:
        from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze

        # analyze is synchronous, wrap with asyncio.to_thread
        result = await asyncio.to_thread(
            analyze,
            api_name=api_name,
            input_data=text,
            custom_prompt_arg=custom_prompt,
            api_key=None,
            system_message=system_message,
            temp=temperature,
            streaming=False,  # Don't use streaming in workflow context
            recursive_summarization=recursive_summarization,
            chunked_summarization=chunked_summarization,
            chunk_options=chunk_options,
        )

        # Check for error
        if isinstance(result, str) and result.startswith("Error:"):
            return {"error": result, "summary": "", "text": ""}

        summary = str(result) if result else ""
        return {
            "summary": summary,
            "text": summary,  # Alias for chaining
            "api_name": api_name,
            "input_length": len(text),
            "output_length": len(summary),
        }

    except Exception:
        logger.exception("Summarize adapter error")
        return {"error": "summarize_error"}
