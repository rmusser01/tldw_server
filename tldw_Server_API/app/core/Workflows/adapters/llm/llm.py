"""LLM adapters: llm, llm_with_tools, llm_compare, llm_critique.

These adapters handle LLM chat completion operations.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.llm._config import (
    LLMCompareConfig,
    LLMConfig,
    LLMCritiqueConfig,
    LLMWithToolsConfig,
)


@registry.register(
    "llm",
    category="llm",
    description="Execute an LLM chat completion via the adapter registry",
    parallelizable=True,
    tags=["core", "ai"],
    config_model=LLMConfig,
)
async def run_llm_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Execute an LLM chat completion via the adapter registry.

    Config (subset; additional keys passed through):
      - provider/api_provider/api_endpoint: str
      - model: str (optional for local providers)
      - prompt: str (templated) or messages: list[dict] (templated)
      - system_message/system/system_prompt: str (templated)
      - temperature, top_p, max_tokens, stop, tools, tool_choice, response_format, seed
      - stream: bool (optional)
      - include_response: bool (default false)
    Output:
      - text: str
      - metadata: token_usage/cost if available
      - response: raw provider response (optional)
    """
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    def _render_str(val: Any) -> Any:
        if isinstance(val, str):
            try:
                return _tmpl(val, context) or val
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("LLM adapter: template rendering failed")
                return val
        return val

    def _render_message(msg: Any) -> dict[str, Any] | None:
        if isinstance(msg, dict):
            out = dict(msg)
            if isinstance(out.get("content"), str):
                out["content"] = _render_str(out["content"])
            return out
        if isinstance(msg, str):
            return {"role": "user", "content": _render_str(msg)}
        return None

    provider_raw = (
        config.get("provider")
        or config.get("api_provider")
        or config.get("api_endpoint")
        or DEFAULT_LLM_PROVIDER
    )
    provider = str(_render_str(provider_raw) or "").strip().lower()
    if not provider:
        raise AdapterError("missing_provider")

    model = config.get("model") or config.get("model_id")
    model = _render_str(model) if model is not None else None

    system_message = (
        config.get("system_message")
        or config.get("system")
        or config.get("system_prompt")
    )
    system_message = _render_str(system_message) if system_message is not None else None
    if isinstance(system_message, str) and not system_message.strip():
        system_message = None

    messages_cfg = config.get("messages") or config.get("messages_payload")
    prompt = config.get("prompt") or config.get("input") or config.get("template")
    messages: list[dict[str, Any]] = []

    if messages_cfg is None:
        if not prompt:
            raise AdapterError("missing_prompt")
        rendered_prompt = _render_str(str(prompt))
        messages = [{"role": "user", "content": rendered_prompt}]
    elif isinstance(messages_cfg, str):
        raw = _render_str(messages_cfg)
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                rendered = _render_message(item)
                if rendered:
                    messages.append(rendered)
        elif str(raw).strip():
            messages = [{"role": "user", "content": raw}]
        if not messages:
            raise AdapterError("missing_messages")
    elif isinstance(messages_cfg, list):
        for item in messages_cfg:
            rendered = _render_message(item)
            if rendered:
                messages.append(rendered)
        if not messages:
            raise AdapterError("missing_messages")
    else:
        raise AdapterError("invalid_messages")

    # Short-circuit in tests to avoid outbound LLM calls
    if is_test_mode():
        preview = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                preview = msg["content"]
                break
        if not preview:
            try:
                preview = str(messages[-1].get("content") or "")
            except (AttributeError, IndexError, TypeError, ValueError):
                preview = ""
        return {
            "text": preview,
            "provider": provider,
            "model": model,
            "simulated": True,
        }

    stream = bool(config.get("stream", False))
    include_response = bool(config.get("include_response", False))

    call_args: dict[str, Any] = {
        "api_endpoint": provider,
        "messages_payload": messages,
        "system_message": system_message,
        "model": model,
        "stream": stream,
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "max_tokens": config.get("max_tokens"),
        "max_completion_tokens": config.get("max_completion_tokens"),
        "stop": config.get("stop"),
        "tools": config.get("tools"),
        "tool_choice": config.get("tool_choice"),
        "response_format": config.get("response_format"),
        "seed": config.get("seed"),
        "n": config.get("n"),
        "logit_bias": config.get("logit_bias"),
        "user": config.get("user") or context.get("user_id"),
        "api_key": _render_str(config.get("api_key")) if config.get("api_key") is not None else None,
    }
    # Drop None values for cleaner adapter inputs
    call_args = {k: v for k, v in call_args.items() if v is not None}

    if stream:
        stream_iter = await perform_chat_api_call_async(**call_args)
        text = ""
        async for line in stream_iter:
            if not line:
                continue
            raw = line.decode("utf-8", errors="replace") if isinstance(line, (bytes, bytearray)) else str(line)
            raw = raw.strip()
            if not raw:
                continue
            if raw.lower() == "data: [done]":
                break
            if raw.startswith("data:"):
                payload = raw[5:].strip()
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    choices = data.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        chunk = delta.get("content")
                        if isinstance(chunk, str) and chunk:
                            text += chunk
                            try:
                                if callable(context.get("append_event")):
                                    context["append_event"]("llm_stream", {"delta": chunk})
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                logger.debug("LLM stream event dispatch failed")
                    continue
            # Fallback: treat as plain text chunk
            text += raw
        return {"text": text, "streamed": True}

    response = await perform_chat_api_call_async(**call_args)
    text = extract_openai_content(response) or ""
    out: dict[str, Any] = {"text": text}
    metadata: dict[str, Any] = {}
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            metadata["token_usage"] = usage
        if "cost_usd" in response:
            metadata["cost_usd"] = response.get("cost_usd")
    if metadata:
        out["metadata"] = metadata
    if include_response:
        out["response"] = response
    return out


@registry.register(
    "llm_with_tools",
    category="llm",
    description="LLM call that can invoke defined tools",
    parallelizable=False,
    tags=["ai", "tools", "agentic"],
    config_model=LLMWithToolsConfig,
)
async def run_llm_with_tools_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """LLM call that can invoke defined tools.

    Config:
      - prompt: str
      - tools: list of tool definitions
      - auto_execute: bool (default True)
      - max_tool_calls: int (default 5)
      - provider: str
      - model: str
      - system_message: str
    Output: { "text": str, "tool_results": list, "iterations": int }
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    prompt = config.get("prompt") or ""
    if isinstance(prompt, str):
        prompt = _tmpl(prompt, context) or prompt

    if not prompt:
        prev = context.get("prev") or context.get("last") or {}
        prompt = prev.get("text") or prev.get("prompt") or "" if isinstance(prev, dict) else ""

    tools = config.get("tools") or []
    auto_execute = config.get("auto_execute", True)
    max_tool_calls = int(config.get("max_tool_calls", 5))
    provider = config.get("provider")
    model = config.get("model")
    system_message = config.get("system_message") or "You are a helpful assistant with access to tools."

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        messages = [{"role": "user", "content": prompt}]
        tool_results = []
        final_response = None
        iteration = 0

        for _iteration in range(max_tool_calls + 1):
            if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                return {"__status__": "cancelled"}

            response = await perform_chat_api_call_async(
                messages=messages,
                api_provider=provider,
                model=model,
                system_message=system_message,
                tools=tools if tools else None,
            )

            # Check for tool calls in response
            tool_calls = None
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls")
                    if not tool_calls:
                        final_response = extract_openai_content(response)
                        break

            if not tool_calls or not auto_execute:
                final_response = extract_openai_content(response)
                break

            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name")
                tool_args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                # Try to execute via MCP
                try:
                    from tldw_Server_API.app.core.MCP_unified.manager import get_mcp_manager
                    manager = get_mcp_manager()
                    result = await manager.execute_tool(tool_name, tool_args, context=None)
                    tool_results.append({"tool": tool_name, "result": result})

                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, default=str)})
                except (AdapterError, AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
                    tool_results.append({"tool": tool_name, "error": "tool_execution_error"})
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": "Error: tool_execution_error"})

        return {"text": final_response or "", "tool_results": tool_results, "iterations": iteration + 1}

    except (AdapterError, AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        logger.exception("LLM with tools error")
        return {"error": "llm_with_tools_error", "text": "", "tool_results": []}


@registry.register(
    "llm_compare",
    category="llm",
    description="Run same prompt through multiple LLMs and compare",
    parallelizable=True,
    tags=["ai", "comparison"],
    config_model=LLMCompareConfig,
)
async def run_llm_compare_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Run same prompt through multiple LLMs and compare.

    Config:
      - prompt: str - Prompt to send to all LLMs
      - providers: list[dict] - List of {provider, model} pairs
      - system_message: str - Optional system message
    Output:
      - responses: list[dict] - Responses from each provider
      - comparison: dict - Comparison metadata
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    prompt = config.get("prompt") or ""
    if isinstance(prompt, str):
        prompt = _tmpl(prompt, context) or prompt

    if not prompt:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            prompt = prev.get("text") or prev.get("prompt") or ""

    if not prompt:
        return {"responses": [], "error": "missing_prompt"}

    providers = config.get("providers") or []
    if not providers:
        return {"responses": [], "error": "missing_providers"}

    system_message = config.get("system_message")

    responses = []
    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        for p in providers:
            if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                return {"__status__": "cancelled"}

            provider = p.get("provider")
            model = p.get("model")
            start_time = time.time()

            try:
                messages = [{"role": "user", "content": prompt}]
                response = await perform_chat_api_call_async(
                    messages=messages,
                    api_provider=provider,
                    model=model,
                    system_message=system_message,
                )
                text = extract_openai_content(response) or ""
                elapsed_ms = (time.time() - start_time) * 1000

                responses.append({
                    "provider": provider,
                    "model": model,
                    "text": text,
                    "elapsed_ms": elapsed_ms,
                    "char_count": len(text),
                })
            except (AdapterError, AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
                responses.append({
                    "provider": provider,
                    "model": model,
                    "error": "llm_provider_error",
                    "elapsed_ms": (time.time() - start_time) * 1000,
                })

        return {
            "responses": responses,
            "comparison": {
                "provider_count": len(providers),
                "successful": sum(1 for r in responses if "text" in r),
                "failed": sum(1 for r in responses if "error" in r),
            },
        }

    except (AdapterError, AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        logger.exception("LLM compare error")
        return {"responses": [], "error": "llm_compare_error"}


@registry.register(
    "llm_critique",
    category="llm",
    description="Run LLM critique on content (Constitutional AI pattern)",
    parallelizable=True,
    tags=["ai", "evaluation"],
    config_model=LLMCritiqueConfig,
)
async def run_llm_critique_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Run LLM critique on content (Constitutional AI pattern).

    Config:
      - content: str - Content to critique
      - criteria: list[str] - Criteria to evaluate
      - revise: bool (default True) - Whether to generate revised version
      - provider: str
      - model: str
    Output: { "critique": str, "revised": str, "criteria": list }
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    content = config.get("content") or ""
    if isinstance(content, str):
        content = _tmpl(content, context) or content

    if not content:
        prev = context.get("prev") or context.get("last") or {}
        content = prev.get("text") or prev.get("content") or "" if isinstance(prev, dict) else ""

    if not content:
        return {"error": "missing_content", "critique": "", "revised": ""}

    criteria = config.get("criteria") or ["accuracy", "clarity", "completeness"]
    revise = config.get("revise", True)
    provider = config.get("provider")
    model = config.get("model")

    criteria_str = ", ".join(criteria)

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        # Step 1: Critique
        critique_prompt = f"""Critique the following content based on these criteria: {criteria_str}

Content:
{content[:6000]}

Provide specific, actionable feedback."""

        messages = [{"role": "user", "content": critique_prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=provider,
            model=model,
            system_message="You are a critical reviewer providing constructive feedback.",
            max_tokens=1500,
            temperature=0.5,
        )

        critique = extract_openai_content(response) or ""

        revised = ""
        if revise and critique:
            # Step 2: Revise
            revise_prompt = f"""Original content:
{content[:5000]}

Critique:
{critique}

Revise the content to address the critique while maintaining the original intent."""

            messages = [{"role": "user", "content": revise_prompt}]
            response = await perform_chat_api_call_async(
                messages=messages,
                api_provider=provider,
                model=model,
                system_message="You revise content based on feedback.",
                max_tokens=2000,
                temperature=0.5,
            )

            revised = extract_openai_content(response) or ""

        return {"critique": critique, "revised": revised, "criteria": criteria}

    except (AdapterError, AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        logger.exception("LLM critique error")
        return {"error": "llm_critique_error", "critique": "", "revised": ""}
