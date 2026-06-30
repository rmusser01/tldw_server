# prompt_executor.py
# Prompt execution engine for Prompt Studio

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts import (
    StructuredPromptAssemblyError,
    assemble_prompt_definition,
)

########################################################################################################################
# Prompt Executor

class PromptExecutor:
    """Executes prompts with various LLM providers."""

    # Template variable constraints
    MAX_VARIABLE_VALUE_LENGTH = 100000  # 100K chars max per variable
    MAX_TOTAL_PROMPT_LENGTH = 500000    # 500K chars max total
    _VALID_TEMPLATE_KEY = re.compile(r"^[A-Za-z0-9_]+$")
    _PLACEHOLDER_PATTERN = re.compile(
        r"\{\{\s*(?P<double>[A-Za-z0-9_]+)\s*\}\}"
        r"|\{(?P<brace>[A-Za-z0-9_]+)\}"
        r"|\$(?P<dollar>[A-Za-z0-9_]+)\b"
        r"|<(?P<angle>[A-Za-z0-9_]+)>"
    )

    # Normalize common aliases to canonical provider ids used by the adapter registry.
    PROVIDER_ALIASES = {
        "llama": "llama.cpp",
        "oobabooga": "ooba",
        "tabby": "tabbyapi",
        "custom": "custom-openai-api",
        "custom_openai": "custom-openai-api",
        "custom-openai": "custom-openai-api",
        "custom_openai_2": "custom-openai-api-2",
        "custom-openai-2": "custom-openai-api-2",
    }

    _PARAM_ALIASES = {
        "top_p": ("top_p", "topp", "maxp"),
        "top_k": ("top_k", "topk"),
        "min_p": ("min_p", "minp"),
        "stop": ("stop",),
        "response_format": ("response_format",),
        "tools": ("tools",),
        "tool_choice": ("tool_choice",),
        "seed": ("seed",),
        "logit_bias": ("logit_bias",),
        "logprobs": ("logprobs",),
        "top_logprobs": ("top_logprobs",),
        "presence_penalty": ("presence_penalty",),
        "frequency_penalty": ("frequency_penalty",),
        "n": ("n",),
    }

    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize PromptExecutor.

        Args:
            db: Database instance
        """
        self.db = db
        self.client_id = db.client_id

    ####################################################################################################################
    # Prompt Execution

    async def execute_prompt(self, prompt_id: int, test_inputs: dict[str, Any],
                             model_config: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a prompt with given inputs and model configuration.

        Args:
            prompt_id: Prompt ID
            test_inputs: Input values for the prompt
            model_config: Model configuration (provider, model, parameters)

        Returns:
            Execution result with output, metrics, and metadata
        """
        start_time = time.time()

        try:
            # Get prompt details
            prompt = self._get_prompt(prompt_id)
            if not prompt:
                raise ValueError(f"Prompt {prompt_id} not found")

            # Get signature if linked
            signature = None
            if prompt.get("signature_id"):
                signature = self._get_signature(prompt["signature_id"])

            # Build the final prompt or canonical message list
            prompt_request = self._build_prompt_request(prompt, signature, test_inputs)

            # Execute with LLM
            provider = model_config.get("provider", "openai")
            model = model_config.get("model", "gpt-3.5-turbo")

            result = await self._call_llm(
                provider=provider,
                model=model,
                prompt=prompt_request.get("prompt"),
                messages=prompt_request.get("messages"),
                system_prompt=prompt_request.get("system_prompt"),
                parameters=model_config.get("parameters", {})
            )

            # Parse output based on signature
            parsed_output = self._parse_output(result["content"], signature)

            # Calculate metrics
            execution_time = (time.time() - start_time) * 1000  # ms

            return {
                "success": True,
                "prompt_id": prompt_id,
                "inputs": test_inputs,
                "raw_output": result["content"],
                "parsed_output": parsed_output,
                "model": model,
                "provider": provider,
                "execution_time_ms": execution_time,
                "tokens_used": result.get("tokens", 0),
                "cost_estimate": self._estimate_cost(
                    provider, model, result.get("tokens", 0)
                ),
                "metadata": {
                    "assembled_messages": prompt_request.get("assembled_messages", []),
                    "temperature": model_config.get("parameters", {}).get("temperature"),
                    "max_tokens": model_config.get("parameters", {}).get("max_tokens"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Prompt execution failed: {e}")
            execution_time = (time.time() - start_time) * 1000

            return {
                "success": False,
                "prompt_id": prompt_id,
                "inputs": test_inputs,
                "error": str(e),
                "model": model_config.get("model"),
                "provider": model_config.get("provider"),
                "execution_time_ms": execution_time,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    async def execute_batch(self, prompt_id: int, test_cases: list[dict[str, Any]],
                           model_configs: list[dict[str, Any]],
                           max_concurrent: int = 5) -> list[dict[str, Any]]:
        """
        Execute a prompt with multiple test cases and model configurations.

        Args:
            prompt_id: Prompt ID
            test_cases: List of test cases with inputs
            model_configs: List of model configurations
            max_concurrent: Maximum concurrent executions

        Returns:
            List of execution results
        """
        results = []

        # Create all execution tasks
        tasks = []
        for test_case in test_cases:
            for model_config in model_configs:
                task = self.execute_prompt(
                    prompt_id=prompt_id,
                    test_inputs=test_case.get("inputs", {}),
                    model_config=model_config
                )
                tasks.append((test_case, model_config, task))

        # Execute in batches
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(
                *[task for _, _, task in batch],
                return_exceptions=True
            )

            # Process results
            for (test_case, model_config, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch execution error: {result}")
                    results.append({
                        "success": False,
                        "test_case_id": test_case.get("id"),
                        "error": str(result),
                        "model": model_config.get("model"),
                        "provider": model_config.get("provider")
                    })
                else:
                    result["test_case_id"] = test_case.get("id")
                    result["test_case_name"] = test_case.get("name")
                    results.append(result)

        return results

    ####################################################################################################################
    # LLM Integration

    @classmethod
    def _normalize_provider(cls, provider: str) -> str:
        provider_lower = (provider or "").strip().lower()
        return cls.PROVIDER_ALIASES.get(provider_lower, provider_lower)

    def _build_adapter_request(
        self,
        *,
        provider: str,
        model: Optional[str],
        messages: list[dict[str, Any]],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        provider_name = normalize_provider(self._normalize_provider(provider))
        if not provider_name:
            raise ChatConfigurationError(provider=provider, message="LLM provider is required.")
        app_config = ensure_app_config()
        resolved_model = model or resolve_provider_model(provider_name, app_config)
        if not resolved_model:
            raise ChatConfigurationError(provider=provider_name, message="Model is required for provider.")

        system_message = system_prompt
        request_messages = messages
        if not system_message:
            system_message, request_messages = split_system_message(messages)

        request: dict[str, Any] = {
            "messages": request_messages,
            "system_message": system_message,
            "model": resolved_model,
            "api_key": resolve_provider_api_key_from_config(provider_name, app_config),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "app_config": app_config,
        }

        for canonical, aliases in self._PARAM_ALIASES.items():
            for key in aliases:
                if key in params and params[key] is not None:
                    request[canonical] = params[key]
                    break

        return request

    @staticmethod
    def _coerce_llm_response(response: Any) -> tuple[str, int]:
        if response is None:
            return "", 0
        if isinstance(response, tuple) and len(response) == 2:
            content, tokens = response
            try:
                return str(content or ""), int(tokens or 0)
            except Exception:
                return str(content or ""), 0
        if isinstance(response, str):
            return response, int(len(response.split()) * 1.3)
        if isinstance(response, list) and response:
            if isinstance(response[0], str):
                content = response[0]
                return content, int(len(content.split()) * 1.3)
            if isinstance(response[0], dict):
                return PromptExecutor._coerce_llm_response(response[0])
        if isinstance(response, dict):
            content = None
            choices = response.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message") or {}
                    msg_content = message.get("content")
                    if isinstance(msg_content, list):
                        parts = [part.get("text", "") for part in msg_content if isinstance(part, dict)]
                        msg_content = "".join(parts)
                    if isinstance(msg_content, str):
                        content = msg_content
                        break
                    delta = choice.get("delta") or {}
                    delta_content = delta.get("content")
                    if isinstance(delta_content, list):
                        parts = [part.get("text", "") for part in delta_content if isinstance(part, dict)]
                        delta_content = "".join(parts)
                    if isinstance(delta_content, str):
                        content = delta_content
                        break
            if content is None:
                raw_content = response.get("content")
                if isinstance(raw_content, str):
                    content = raw_content
            if content is None:
                content = str(response)
            tokens = 0
            usage = response.get("usage")
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                if isinstance(total_tokens, int):
                    tokens = total_tokens
                else:
                    prompt_tokens = usage.get("prompt_tokens") or 0
                    completion_tokens = usage.get("completion_tokens") or 0
                    if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
                        tokens = int(prompt_tokens) + int(completion_tokens)
            if tokens == 0 and isinstance(content, str):
                tokens = int(len(content.split()) * 1.3)
            return content, tokens
        return str(response), 0

    async def _call_llm(
        self,
        provider: str,
        model: str,
        prompt: Optional[str] = None,
        *,
        messages: Optional[list[dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Call the appropriate LLM provider.

        Args:
            provider: Provider name
            model: Model name
            prompt: User prompt
            messages: Canonical chat messages
            system_prompt: System prompt
            parameters: Additional parameters

        Returns:
            LLM response
        """
        # Prepare parameters
        params = parameters or {}
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 1000)

        api_endpoint = self._normalize_provider(provider)
        request_messages = list(messages or [])
        if not request_messages:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            request_messages = [{"role": "user", "content": prompt}]

        # Backoff + retry for transient/provider limit errors
        last_exc = None
        for attempt in range(3):
            try:
                adapter = get_registry().get_adapter(normalize_provider(api_endpoint))
                if adapter is None:
                    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call as _legacy_call
                    response = await asyncio.to_thread(
                        _legacy_call,
                        api_endpoint=api_endpoint,
                        messages_payload=request_messages,
                        system_message=system_prompt,
                        temp=temperature,
                        max_tokens=max_tokens,
                        model=model,
                        streaming=False,
                    )
                else:
                    request = self._build_adapter_request(
                        provider=api_endpoint,
                        model=model,
                        messages=request_messages,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        params=params,
                    )
                    response = await asyncio.to_thread(
                        adapter.chat,
                        request,
                    )
                content, tokens = self._coerce_llm_response(response)
                return {"content": content, "tokens": tokens}
            except Exception as e:
                last_exc = e
                # Basic 429/backoff detection
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower():
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                logger.error(f"LLM call failed for {provider}/{model}: {e}")
                raise
        # If we exhausted retries
        logger.error(f"LLM call failed after retries for {provider}/{model}: {last_exc}")
        raise last_exc if last_exc else RuntimeError("LLM call failed")

    ####################################################################################################################
    # Helper Methods

    def _get_prompt(self, prompt_id: int) -> Optional[dict[str, Any]]:
        """Get prompt details from database."""
        prompt = self.db.get_prompt(prompt_id)
        if prompt and prompt.get("deleted"):
            return None
        return prompt

    def _get_signature(self, signature_id: int) -> Optional[dict[str, Any]]:
        """Get signature details from database."""
        signature = self.db.get_signature(signature_id)
        if signature and signature.get("deleted"):
            return None
        return signature

    def _build_prompt_request(
        self,
        prompt: dict[str, Any],
        signature: Optional[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        prepared_inputs = self._prepare_prompt_inputs(inputs)
        if prompt.get("prompt_format") == "structured" and prompt.get("prompt_definition"):
            return self._build_structured_prompt_request(prompt, signature, prepared_inputs)

        final_prompt = self._build_prompt(prompt, signature, prepared_inputs)
        return {
            "prompt": final_prompt,
            "messages": None,
            "system_prompt": prompt.get("system_prompt"),
            "assembled_messages": self._build_legacy_assembled_messages(
                final_prompt,
                prompt.get("system_prompt"),
            ),
        }

    def _build_prompt(self, prompt: dict[str, Any], signature: Optional[dict[str, Any]],
                     inputs: dict[str, Any]) -> str:
        """
        Build the final prompt by substituting variables.

        Args:
            prompt: Prompt data
            signature: Optional signature data
            inputs: Input values

        Returns:
            Final prompt string
        """
        # Prompt Studio stores system and user prompts separately; use user_prompt as the template
        template = (prompt.get("user_prompt") or "")

        # Replace variables in template with length validation.
        rendered_inputs: dict[str, str] = {}
        for key, value in inputs.items():
            # Validate key name to prevent template injection
            # Keys should be alphanumeric with underscores only
            if not key or not isinstance(key, str):
                logger.warning(f"Skipping invalid variable key: {key!r}")
                continue
            if not self._VALID_TEMPLATE_KEY.fullmatch(key):
                logger.warning(f"Skipping invalid variable key: {key!r}")
                continue

            # Convert to string and validate length
            str_value = str(value) if value is not None else ""
            if len(str_value) > self.MAX_VARIABLE_VALUE_LENGTH:
                logger.warning(
                    f"Variable '{key}' value truncated from {len(str_value)} to {self.MAX_VARIABLE_VALUE_LENGTH} chars"
                )
                str_value = str_value[:self.MAX_VARIABLE_VALUE_LENGTH] + "... [truncated]"
            rendered_inputs[key] = str_value

        def _replace_placeholder(match: re.Match[str]) -> str:
            key = next((value for value in match.groupdict().values() if value is not None), "")
            if key in rendered_inputs:
                return rendered_inputs[key]
            return match.group(0)

        template = self._PLACEHOLDER_PATTERN.sub(_replace_placeholder, template)

        # Validate total prompt length
        if len(template) > self.MAX_TOTAL_PROMPT_LENGTH:
            logger.warning(
                f"Final prompt truncated from {len(template)} to {self.MAX_TOTAL_PROMPT_LENGTH} chars"
            )
            template = template[:self.MAX_TOTAL_PROMPT_LENGTH] + "\n... [prompt truncated due to length]"

        # Add signature instructions if present
        if signature:
            template = self._apply_signature_to_text(template, signature)

        return template

    def _build_structured_prompt_request(
        self,
        prompt: dict[str, Any],
        signature: Optional[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            assembly = assemble_prompt_definition(
                prompt["prompt_definition"],
                inputs,
                extras={
                    "few_shot_examples": prompt.get("few_shot_examples"),
                    "modules_config": prompt.get("modules_config"),
                },
            )
        except StructuredPromptAssemblyError:
            raise
        except Exception as exc:
            raise StructuredPromptAssemblyError(
                "prompt_assembly_failed",
                f"Failed to assemble structured prompt: {exc}",
            ) from exc

        messages = self._apply_signature_to_messages(assembly.messages, signature)
        messages = self._enforce_messages_length(messages)
        return {
            "prompt": None,
            "messages": messages,
            "system_prompt": None,
            "assembled_messages": [message.copy() for message in messages],
        }

    def _build_legacy_assembled_messages(
        self,
        final_prompt: str,
        system_prompt: Optional[str],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": final_prompt})
        return messages

    def _prepare_prompt_inputs(self, inputs: dict[str, Any]) -> dict[str, str]:
        prepared_inputs: dict[str, str] = {}
        for key, value in inputs.items():
            if not key or not isinstance(key, str):
                logger.warning(f"Skipping invalid variable key: {key!r}")
                continue

            str_value = str(value) if value is not None else ""
            if len(str_value) > self.MAX_VARIABLE_VALUE_LENGTH:
                logger.warning(
                    f"Variable '{key}' value truncated from {len(str_value)} to {self.MAX_VARIABLE_VALUE_LENGTH} chars"
                )
                str_value = str_value[:self.MAX_VARIABLE_VALUE_LENGTH] + "... [truncated]"

            prepared_inputs[key] = str_value

        return prepared_inputs

    @staticmethod
    def _apply_signature_to_messages(
        messages: list[dict[str, Any]],
        signature: Optional[dict[str, Any]],
    ) -> list[dict[str, str]]:
        rendered_messages = [
            {"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")}
            for message in messages
        ]
        if not signature:
            return rendered_messages

        target_index = next(
            (index for index in range(len(rendered_messages) - 1, -1, -1)
             if rendered_messages[index]["role"] == "user"),
            None,
        )

        target_content = ""
        if target_index is not None:
            target_content = rendered_messages[target_index]["content"]

        updated_content = PromptExecutor._apply_signature_to_text(target_content, signature)
        if target_index is None:
            rendered_messages.append({"role": "user", "content": updated_content})
        else:
            rendered_messages[target_index]["content"] = updated_content

        return rendered_messages

    @staticmethod
    def _apply_signature_to_text(text: str, signature: dict[str, Any]) -> str:
        updated_text = text
        sig_instruction = signature.get("instruction", "")
        if sig_instruction:
            updated_text = f"{sig_instruction}\n\n{updated_text}" if updated_text else sig_instruction

        output_instruction = PromptExecutor._render_output_schema_instruction(signature.get("output_schema"))
        if output_instruction:
            updated_text = f"{updated_text}\n\n{output_instruction}" if updated_text else output_instruction

        return updated_text

    @staticmethod
    def _render_output_schema_instruction(output_schema: Any) -> str:
        if not output_schema:
            return ""

        if isinstance(output_schema, str):
            try:
                output_schema = json.loads(output_schema)
            except Exception:
                return ""

        if not isinstance(output_schema, list):
            return ""

        schema_shape = {
            field["name"]: f"<{field.get('type', 'string')}>"
            for field in output_schema
            if isinstance(field, dict) and field.get("name")
        }
        if not schema_shape:
            return ""

        return (
            "Please format your response as JSON with the following structure:\n"
            + json.dumps(schema_shape, indent=2)
        )

    def _enforce_messages_length(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        rendered_messages = [message.copy() for message in messages]
        total_length = sum(len(message.get("content", "")) for message in rendered_messages)
        if total_length <= self.MAX_TOTAL_PROMPT_LENGTH or not rendered_messages:
            return rendered_messages

        logger.warning(
            f"Structured prompt truncated from {total_length} to {self.MAX_TOTAL_PROMPT_LENGTH} chars"
        )

        target_index = next(
            (index for index in range(len(rendered_messages) - 1, -1, -1)
             if rendered_messages[index]["role"] == "user"),
            len(rendered_messages) - 1,
        )
        preserved_length = total_length - len(rendered_messages[target_index].get("content", ""))
        remaining_budget = max(self.MAX_TOTAL_PROMPT_LENGTH - preserved_length, 0)
        notice = "\n... [prompt truncated due to length]"

        if remaining_budget <= 0:
            rendered_messages[target_index]["content"] = ""
            return rendered_messages

        if remaining_budget > len(notice):
            trimmed_length = remaining_budget - len(notice)
            rendered_messages[target_index]["content"] = (
                rendered_messages[target_index].get("content", "")[:trimmed_length] + notice
            )
            return rendered_messages

        rendered_messages[target_index]["content"] = (
            rendered_messages[target_index].get("content", "")[:remaining_budget]
        )
        return rendered_messages

    # Compatibility alias used by tests
    async def execute(self, prompt_id: int, inputs: dict[str, Any], provider: str = "openai", model: str = "gpt-3.5-turbo",
                      parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Execute prompt using simplified signature.

        Args:
            prompt_id: Prompt ID
            inputs: Input values
            provider: Provider name
            model: Model name
            parameters: Additional parameters

        Returns:
            Execution result dict used in tests.
        """
        model_config = {
            "provider": provider,
            "model": model,
            "parameters": parameters or {}
        }
        return await self.execute_prompt(prompt_id, inputs, model_config)

    def _parse_output(self, output: str, signature: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Parse LLM output based on signature schema.

        Args:
            output: Raw LLM output
            signature: Optional signature with output schema

        Returns:
            Parsed output
        """
        if not signature or not signature.get("output_schema"):
            return {"raw": output}

        # Try to parse as JSON
        try:
            # Look for JSON in the output
            import re
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except Exception as e:
            logger.debug(f"Failed to parse JSON from LLM output for signature-guided parsing: error={e}")

        # Try to extract fields from text
        parsed = {}
        for field in signature.get("output_schema", []):
            if isinstance(field, dict):
                field_name = field.get("name")
                if field_name:
                    # Simple extraction (can be improved)
                    # Use a raw f-string to avoid invalid escape warnings (e.g., "\s")
                    pattern = rf"{field_name}[:\s]+(.*?)(?:\n|$)"
                    match = re.search(pattern, output, re.IGNORECASE)
                    if match:
                        parsed[field_name] = match.group(1).strip()

        if not parsed:
            parsed = {"raw": output}

        return parsed

    def _estimate_cost(self, provider: str, model: str, tokens: int) -> float:
        """
        Estimate cost based on provider and token usage.

        Args:
            provider: Provider name
            model: Model name
            tokens: Token count

        Returns:
            Estimated cost in USD
        """
        # Approximate cost estimates per 1K tokens (blended input/output average).
        # NOTE: Actual pricing differs significantly between input and output tokens
        # (output is typically 3-5x more expensive). These are rough estimates for
        # cost tracking purposes. For accurate billing, use provider-specific APIs.
        # Prices as of late 2024 - may be outdated.
        cost_per_1k = {
            "openai": {
                "gpt-4o": 0.005,
                "gpt-4o-mini": 0.00015,
                "gpt-4-turbo": 0.01,
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.0005,
                "o1": 0.015,
                "o1-mini": 0.003,
            },
            "anthropic": {
                # Claude 4.x series
                "claude-opus-4-5": 0.015,
                "claude-sonnet-4-5": 0.003,
                # Claude 3.x series
                "claude-3-5-sonnet": 0.003,
                "claude-3-5-haiku": 0.001,
                "claude-3-opus": 0.015,
                "claude-3-sonnet": 0.003,
                "claude-3-haiku": 0.00025,
            },
            "groq": {
                "llama-3.3-70b": 0.0006,
                "llama-3.1-70b": 0.0006,
                "llama-3.1-8b": 0.00006,
                "mixtral-8x7b": 0.0002,
            },
            "mistral": {
                "mistral-large": 0.002,
                "mistral-small": 0.0002,
                "codestral": 0.0003,
            },
            "deepseek": {
                "deepseek-chat": 0.00014,
                "deepseek-coder": 0.00014,
            },
            "google": {
                "gemini-3-pro-preview": 0.007,
                "gemini-3-flash-preview": 0.00175,
                "gemini-2.5-pro": 0.005625,
                "gemini-2.5-flash": 0.0014,
                "gemini-2.5-flash-lite": 0.00025,
                "gemini-2.5-flash-lite-preview-09-2025": 0.00025,
                "gemini-2.5-computer-use-preview-10-2025": 0.005625,
                "gemini-2.0-flash": 0.00025,
                "gemini-2.0-flash-lite": 0.0001875,
            }
        }

        # Get cost rate
        provider_costs = cost_per_1k.get(provider.lower(), {})

        # Try exact model match first
        cost_rate = provider_costs.get(model.lower(), 0)

        # If not found, try partial match
        if cost_rate == 0:
            for model_key, rate in provider_costs.items():
                if model_key in model.lower() or model.lower() in model_key:
                    cost_rate = rate
                    break

        # Default to very small cost if unknown
        if cost_rate == 0:
            cost_rate = 0.0001

        # Calculate cost
        return (tokens / 1000.0) * cost_rate

########################################################################################################################
# Prompt Validator

class PromptValidator:
    """Validates prompts and signatures before execution."""

    @staticmethod
    def validate_prompt(prompt: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a prompt.

        Args:
            prompt: Prompt data

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Use user_prompt (Prompt Studio schema) as the primary template content
        user_text = prompt.get("user_prompt") or ""
        if not user_text:
            return False, "Prompt content is required"

        if len(user_text) > 50000:
            return False, "Prompt content exceeds maximum length"

        # Check for required variables
        import re
        variables = re.findall(r'\{(\w+)\}|\$(\w+)|<(\w+)>', user_text)
        flat_vars = [v for group in variables for v in group if v]

        if len(set(flat_vars)) > 20:
            return False, "Too many variables (max 20)"

        return True, None

    @staticmethod
    def validate_signature(signature: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a signature.

        Args:
            signature: Signature data

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate input schema
        if signature.get("input_schema"):
            try:
                input_schema = json.loads(signature["input_schema"]) if isinstance(signature["input_schema"], str) else signature["input_schema"]
                if not isinstance(input_schema, list):
                    return False, "Input schema must be a list"

                for field in input_schema:
                    if not isinstance(field, dict):
                        return False, "Each input field must be an object"
                    if not field.get("name"):
                        return False, "Each input field must have a name"
            except Exception as e:
                logger.debug(f"Invalid input schema format in signature: error={e}")
                return False, "Invalid input schema format"

        # Validate output schema
        if signature.get("output_schema"):
            try:
                output_schema = json.loads(signature["output_schema"]) if isinstance(signature["output_schema"], str) else signature["output_schema"]
                if not isinstance(output_schema, list):
                    return False, "Output schema must be a list"

                for field in output_schema:
                    if not isinstance(field, dict):
                        return False, "Each output field must be an object"
                    if not field.get("name"):
                        return False, "Each output field must have a name"
            except Exception as e:
                logger.debug(f"Invalid output schema format in signature: error={e}")
                return False, "Invalid output schema format"

        return True, None

    @staticmethod
    def validate_test_inputs(inputs: dict[str, Any], signature: Optional[dict[str, Any]]) -> tuple[bool, Optional[str]]:
        """
        Validate test inputs against signature schema.

        Args:
            inputs: Test input values
            signature: Optional signature with schema

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not signature or not signature.get("input_schema"):
            return True, None

        try:
            input_schema = signature["input_schema"]
            if isinstance(input_schema, str):
                input_schema = json.loads(input_schema)

            # Check required fields
            for field in input_schema:
                if isinstance(field, dict):
                    field_name = field.get("name")
                    if field.get("required", True) and field_name not in inputs:
                        return False, f"Required input field missing: {field_name}"

                    # Type validation (basic)
                    if field_name in inputs:
                        value = inputs[field_name]
                        field_type = field.get("type", "string")

                        if field_type == "integer" and not isinstance(value, int):
                            return False, f"Field {field_name} must be an integer"
                        elif field_type == "boolean" and not isinstance(value, bool):
                            return False, f"Field {field_name} must be a boolean"
                        elif field_type == "array" and not isinstance(value, list):
                            return False, f"Field {field_name} must be an array"
                        elif field_type == "object" and not isinstance(value, dict):
                            return False, f"Field {field_name} must be an object"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"
