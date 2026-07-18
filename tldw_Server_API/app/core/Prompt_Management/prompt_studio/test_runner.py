# test_runner.py
# Runs test cases against prompts

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from loguru import logger

from ....core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    is_runtime_issued_provider_call_credentials,
)
from ....core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    await_bounded_sync_call,
    await_owned_worker,
)
from ....core.Chat.Chat_Deps import ChatConfigurationError
from ....core.Chat.streaming_utils import (
    normalize_provider_stream_error,
    provider_result_contains_error,
    sanitized_provider_stream_exception,
)
from ....core.exceptions import raise_detached_error
from ....core.LLM_Calls.adapter_registry import get_registry
from ....core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    get_adapter_or_raise,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from .program_evaluator import ProgramEvaluator
from .prompt_executor import PromptExecutor


class _BoundedProviderResponseError(RuntimeError):
    """Carry one canonical provider code through the legacy RuntimeError seam."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("Provider returned an error response")


class TestRunner:
    """Runs test cases against prompts using LLM."""

    MAX_PERSISTED_OUTPUT_CHARS = 4000

    def __init__(self, db_manager):
        """
        Initialize test runner.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def _call_adapter(
        self,
        *,
        provider: str,
        model: Optional[str],
        messages_payload: list[dict[str, Any]],
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        app_config: Optional[dict[str, Any]] = None,
        api_key_override: Optional[str] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        provider_name = normalize_provider(provider)
        if not provider_name:
            raise ChatConfigurationError(provider=provider, message="LLM provider is required.")
        if provider_credentials is not None:
            if not is_runtime_issued_provider_call_credentials(
                provider_credentials,
                provider=provider_name,
            ):
                raise ChatConfigurationError(
                    provider=provider_name,
                    message="Provider credential context is invalid.",
                )
            cfg = provider_credentials.app_config or {}
            resolved_api_key = provider_credentials.api_key
            resolved_credentials = True
        else:
            cfg = (
                (app_config or {})
                if credentials_resolved
                else ensure_app_config(app_config)
            )
            resolved_api_key = (
                api_key_override
                if credentials_resolved
                else api_key_override
                or resolve_provider_api_key_from_config(provider_name, cfg)
            )
            resolved_credentials = credentials_resolved
        resolved_model = model or resolve_provider_model(provider_name, cfg)
        if not resolved_model:
            raise ChatConfigurationError(provider=provider_name, message="Model is required for provider.")
        sys_msg = system_message
        cleaned_messages = messages_payload
        if not sys_msg:
            sys_msg, cleaned_messages = split_system_message(messages_payload)
        request: dict[str, Any] = {
            "messages": cleaned_messages,
            "system_message": sys_msg,
            "model": resolved_model,
            "api_key": resolved_api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "app_config": cfg,
            "credentials_resolved": resolved_credentials,
        }
        if provider_credentials is not None and (
            get_registry().is_local_provider_name(provider_name)
            or provider_name.startswith("custom-openai-api")
        ):
            request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = provider_credentials
        return get_adapter_or_raise(provider_name).chat(
            request,
            timeout=timeout_seconds,
        )

    async def run_test_case(
        self,
        prompt_id: int,
        test_case_id: int,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        provider: str = "openai",
        app_config: Optional[dict[str, Any]] = None,
        api_key_override: Optional[str] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
        timeout_seconds: Optional[float] = None,
        persist_run: bool = True,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
    ) -> dict[str, Any]:
        """
        Run a single test case against a prompt.

        Args:
            prompt_id: ID of the prompt
            test_case_id: ID of the test case
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens

        Returns:
            Test run result
        """
        start_time = time.time()
        _log = logger.bind(ps_component="test_runner", prompt_id=prompt_id, test_case_id=test_case_id, model=model)
        _log.info("PS testrun.start temperature={} max_tokens={}", temperature, max_tokens)
        prompt = self.db.get_prompt(prompt_id)
        if not prompt or prompt.get("deleted"):
            raise ValueError(f"Prompt {prompt_id} not found")

        test_case = self.db.get_test_case(test_case_id)
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")

        inputs = test_case.get("inputs") or {}
        expected = test_case.get("expected_outputs") or {}

        # Call LLM
        try:
            signature = None
            if prompt.get("signature_id"):
                signature = self.db.get_signature(prompt["signature_id"])
                if signature and signature.get("deleted"):
                    signature = None

            prompt_request = PromptExecutor(self.db)._build_prompt_request(prompt, signature, inputs)
            messages_payload = prompt_request.get("messages") or [
                {"role": "user", "content": prompt_request.get("prompt", "")}
            ]

            async def _dispatch_and_extract() -> str:
                adapter_kwargs: dict[str, Any] = {
                    "provider": provider,
                    "model": model,
                    "messages_payload": messages_payload,
                    "system_message": prompt_request.get("system_prompt"),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "app_config": app_config,
                    "api_key_override": api_key_override,
                    "credentials_resolved": credentials_resolved,
                    "timeout_seconds": timeout_seconds,
                }
                if provider_credentials is not None:
                    adapter_kwargs["provider_credentials"] = provider_credentials
                response = await await_bounded_sync_call(
                    lambda: self._call_adapter(**adapter_kwargs),
                    pool=SYNC_ADAPTER_CALL_POOL,
                    exhaustion_message="Prompt Studio adapter capacity is exhausted",
                )
                return self._extract_response_text(response)

            response_text = await await_owned_worker(
                _dispatch_and_extract(),
                on_cancel_success=on_provider_success,
            )
            if on_provider_success is not None:
                await await_owned_worker(on_provider_success())

            actual_output = {"response": response_text}
            _log.info("PS testrun.llm.done time_ms={}", int((time.time() - start_time) * 1000))

        except Exception as e:
            safe_error = sanitized_provider_stream_exception(e)
            _log.error(
                "PS testrun.llm.error error_code={} time_ms={}",
                safe_error.code,
                int((time.time() - start_time) * 1000),
            )
            if strict_provider_errors:
                raise_detached_error(safe_error)
            actual_output = {
                "error": str(safe_error),
                "error_code": safe_error.code,
            }

        execution_time_ms = int((time.time() - start_time) * 1000)
        tokens_used = None

        stored_run: dict[str, Any] = {}
        if persist_run:
            stored_run = self.db.create_test_run(
                project_id=prompt.get("project_id") or test_case.get("project_id"),
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_name=model,
                model_params={
                    "provider": provider,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                inputs=inputs,
                outputs=actual_output,
                expected_outputs=expected,
                execution_time_ms=execution_time_ms,
                tokens_used=tokens_used,
                client_id=self.db.client_id,
            )
            _log.info("PS testrun.persisted id={} exec_ms={}", stored_run.get("id"), execution_time_ms)

        return {
            "id": stored_run.get("id"),
            "test_case_id": stored_run.get("test_case_id", test_case_id) if stored_run else test_case_id,
            "prompt_id": stored_run.get("prompt_id", prompt_id) if stored_run else prompt_id,
            "inputs": stored_run.get("inputs", inputs) if stored_run else inputs,
            "expected": stored_run.get("expected_outputs", expected) if stored_run else expected,
            "actual": stored_run.get("outputs", actual_output) if stored_run else actual_output,
            "model": stored_run.get("model_name", model) if stored_run else model,
            "execution_time_ms": execution_time_ms,
            "tokens_used": tokens_used,
        }

    @staticmethod
    def _truncate_value(value: Any, max_chars: int) -> Any:
        if isinstance(value, str):
            if len(value) <= max_chars:
                return value
            return value[:max_chars] + "...(truncated)"
        if isinstance(value, dict):
            return {k: TestRunner._truncate_value(v, max_chars) for k, v in value.items()}
        if isinstance(value, list):
            return [TestRunner._truncate_value(v, max_chars) for v in value]
        return value

    @staticmethod
    def _provider_response_error(response: Any) -> BaseException | None:
        if not provider_result_contains_error(response, legacy_error_prefix=True):
            return None

        seen: set[int] = set()

        def find_normalized(item: Any) -> Any:
            normalized = normalize_provider_stream_error(item)
            if normalized is not None:
                return normalized
            if isinstance(item, (list, tuple)):
                identity = id(item)
                if identity in seen:
                    return None
                seen.add(identity)
                for nested_item in item:
                    nested = find_normalized(nested_item)
                    if nested is not None:
                        return nested
                return None
            if not isinstance(item, dict):
                return None

            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            for field in ("choices", "message", "delta", "content"):
                if field in item:
                    nested = find_normalized(item[field])
                    if nested is not None:
                        return nested
            if item.get("type") in {"text", "output_text"} and "text" in item:
                return find_normalized(item["text"])
            return None

        normalized = find_normalized(response)
        return _BoundedProviderResponseError(
            normalized.code if normalized is not None else "provider_unavailable"
        )

    @staticmethod
    def _is_sse_framed_response(value: str) -> bool:
        return any(
            line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip().startswith(
                ("data:", "event:", "id:", "retry:", ":")
            )
            for line in value.splitlines()
            if line.strip()
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        provider_error = TestRunner._provider_response_error(response)
        if provider_error is not None:
            raise provider_error from None

        text: str | None = None
        if isinstance(response, str):
            if response.lstrip().lower().startswith("error:"):
                raise _BoundedProviderResponseError("provider_unavailable")
            if TestRunner._is_sse_framed_response(response):
                raise _BoundedProviderResponseError("provider_unavailable")
            text = response
        if isinstance(response, list) and response:
            if isinstance(response[0], (str, dict)):
                return TestRunner._extract_response_text(response[0])
        if isinstance(response, dict):
            if response.get("error") not in (None, False, ""):
                raise _BoundedProviderResponseError("provider_unavailable")
            choices = response.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if not isinstance(message, dict):
                        message = {}
                    content = message.get("content")
                    if isinstance(content, list):
                        parts = [
                            part["text"]
                            for part in content
                            if isinstance(part, dict) and isinstance(part.get("text"), str)
                        ]
                        content = "".join(parts)
                    if isinstance(content, str):
                        text = content
                        break
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        delta = {}
                    delta_content = delta.get("content")
                    if isinstance(delta_content, list):
                        parts = [
                            part["text"]
                            for part in delta_content
                            if isinstance(part, dict) and isinstance(part.get("text"), str)
                        ]
                        delta_content = "".join(parts)
                    if isinstance(delta_content, str):
                        text = delta_content
                        break
            if text is None:
                content = response.get("content")
                if isinstance(content, list):
                    parts = [
                        part["text"]
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    ]
                    content = "".join(parts)
                if isinstance(content, str):
                    text = content
        if isinstance(text, str) and text.strip():
            if (
                normalize_provider_stream_error(text) is not None
                or text.lstrip().lower().startswith("error:")
            ):
                raise _BoundedProviderResponseError("provider_unavailable")
            return text
        raise RuntimeError("Provider returned an empty or malformed response")

    async def run_single_test(
        self,
        *,
        prompt_id: int,
        test_case_id: int,
        model_config: dict[str, Any],
        metrics: Optional[list[Any]] = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
    ) -> dict[str, Any]:
        """Compatibility wrapper used by optimizers.

        Executes a single test case using model_config and returns a dict that
        includes a simple 'scores' map for downstream selection logic.
        """
        params = (model_config or {}).get("parameters", {})
        model = (model_config or {}).get("model", "gpt-3.5-turbo")
        provider = (model_config or {}).get("provider") or (model_config or {}).get("api_name") or "openai"
        api_key_override = (model_config or {}).get("api_key")
        app_config = (model_config or {}).get("app_config")
        credentials_resolved = (model_config or {}).get("credentials_resolved") is True
        timeout_seconds = (model_config or {}).get("timeout_seconds") or params.get(
            "timeout_seconds"
        )
        temperature = float(params.get("temperature", 0.7)) if params is not None else 0.7
        max_tokens = int(params.get("max_tokens", 1000)) if params is not None else 1000

        _log = logger.bind(ps_component="test_runner", prompt_id=prompt_id, test_case_id=test_case_id, model=model)
        _log.info("PS single_test.start provider={} temperature={} max_tokens={}", provider, temperature, max_tokens)
        t0 = time.perf_counter()
        result = await self.run_test_case(
            prompt_id=prompt_id,
            test_case_id=test_case_id,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            app_config=app_config,
            api_key_override=api_key_override,
            credentials_resolved=credentials_resolved,
            provider_credentials=provider_credentials,
            timeout_seconds=timeout_seconds,
            persist_run=False,
            on_provider_success=on_provider_success,
            strict_provider_errors=strict_provider_errors,
        )
        # Determine if this is a program test case and if evaluator is enabled
        try:
            test_case = self.db.get_test_case(test_case_id)
        except Exception:
            test_case = None

        runner_hint = None
        if isinstance(test_case, dict):
            runner_hint = (
                (test_case.get("expected_outputs") or {}).get("runner")
                or (test_case.get("expected_outputs") or {}).get("_runner")
                or (test_case.get("inputs") or {}).get("runner")
                or (test_case.get("inputs") or {}).get("_runner")
            )

        eval_res = None
        reward = None
        run_success = "error" not in (result.get("actual") or {})
        if str(runner_hint or "").lower() == "python":
            # Sandboxed program evaluator (feature-gated per project)
            pe = ProgramEvaluator()
            try:
                project_id = (self.db.get_prompt(prompt_id) or {}).get("project_id")
            except Exception:
                project_id = None
            eval_res = pe.evaluate(
                project_id=project_id,
                db=self.db,
                llm_output=result.get("actual", {}).get("response", ""),
                spec=(self.db.get_test_case(test_case_id) or {}).get("expected_outputs") or {},
            )
            reward = eval_res.reward
            score = max(0.0, min(1.0, reward / 10.0))
            run_success = bool(eval_res.success)
            _log.info("PS single_test.code_eval reward={} score={}", round(reward, 3), round(score, 3))
        else:
            # Provide a basic aggregate score based on expected vs actual overlap
            expected = result.get("expected", {}) or {}
            actual = result.get("actual", {}) or {}
            exp = str(expected.get("response", "")).lower()
            act = str(actual.get("response", "")).lower()
            if not exp and not act:
                score = 0.0
            elif exp == act:
                score = 1.0
            elif exp and act and (exp in act or act in exp):
                score = 0.5
            else:
                # very rough token overlap
                ew = set(exp.split())
                aw = set(act.split())
                score = (len(ew & aw) / max(1, len(ew))) if ew else 0.0

        outputs_for_storage = result.get("actual") or {}
        if eval_res is not None:
            outputs_for_storage = {
                "program_eval": {
                    "success": bool(eval_res.success),
                    "return_code": eval_res.return_code,
                    "reward": float(eval_res.reward),
                    "error": eval_res.error,
                    "stdout": eval_res.stdout,
                    "stderr": eval_res.stderr,
                    "metrics": eval_res.metrics,
                },
                "response_preview": (result.get("actual") or {}).get("response", ""),
            }
        outputs_for_storage = self._truncate_value(outputs_for_storage, self.MAX_PERSISTED_OUTPUT_CHARS)

        scores_payload: dict[str, Any] = {"aggregate_score": float(score)}
        if reward is not None:
            scores_payload["reward"] = float(reward)
        if eval_res is not None:
            scores_payload["program_eval_success"] = bool(eval_res.success)

        try:
            prompt_row = self.db.get_prompt(prompt_id) or {}
        except Exception:
            prompt_row = {}
        project_id = prompt_row.get("project_id") or (test_case or {}).get("project_id")
        if project_id is None:
            raise ValueError(f"Prompt/test case project_id unavailable for test_case_id={test_case_id}")

        stored_run = self.db.create_test_run(
            project_id=project_id,
            prompt_id=prompt_id,
            test_case_id=test_case_id,
            model_name=model,
            model_params={
                "provider": provider,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            inputs=result.get("inputs") or {},
            outputs=outputs_for_storage,
            expected_outputs=result.get("expected") or {},
            scores=scores_payload,
            execution_time_ms=result.get("execution_time_ms"),
            tokens_used=result.get("tokens_used"),
            client_id=self.db.client_id,
        )

        result = dict(result)
        result["id"] = stored_run.get("id")
        result["success"] = bool(run_success)
        result["scores"] = {"aggregate_score": float(score)}
        if reward is not None:
            result["scores"]["reward"] = float(reward)
        if eval_res is not None:
            result["program_eval"] = {
                "success": bool(eval_res.success),
                "return_code": eval_res.return_code,
                "stdout": eval_res.stdout,
                "stderr": eval_res.stderr,
                "metrics": eval_res.metrics,
                "reward": float(eval_res.reward),
                "error": eval_res.error,
            }
        _log.info(
            "PS single_test.done aggregate_score={} success={} duration_ms={} test_run_id={}",
            round(float(score), 3),
            bool(run_success),
            int((time.perf_counter() - t0) * 1000),
            stored_run.get("id"),
        )
        return result

    async def run_multiple_tests(
        self,
        prompt_id: int,
        test_case_ids: list[int],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        parallel: bool = False,
        provider: str = "openai",
        app_config: Optional[dict[str, Any]] = None,
        api_key_override: Optional[str] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
        timeout_seconds: Optional[float] = None,
        strict_provider_errors: bool = False,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run multiple test cases.

        Args:
            prompt_id: ID of the prompt
            test_case_ids: List of test case IDs
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            parallel: Run tests in parallel

        Returns:
            List of test run results
        """
        if parallel:
            # Run tests concurrently
            _log = logger.bind(ps_component="test_runner", prompt_id=prompt_id, total_tests=len(test_case_ids))
            _log.info("PS testrun.multi.start mode=parallel total_tests={}", len(test_case_ids))
            tasks = [
                self.run_test_case(
                    prompt_id=prompt_id,
                    test_case_id=test_id,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                    app_config=app_config,
                    api_key_override=api_key_override,
                    credentials_resolved=credentials_resolved,
                    provider_credentials=provider_credentials,
                    timeout_seconds=timeout_seconds,
                    persist_run=True,
                    on_provider_success=on_provider_success,
                    strict_provider_errors=strict_provider_errors,
                )
                for test_id in test_case_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    if strict_provider_errors:
                        raise result
                    logger.error(f"Test case {test_case_ids[i]} failed: {result}")
                    processed_results.append({
                        "test_case_id": test_case_ids[i],
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            _log.info("PS testrun.multi.done mode=parallel ok={} failed={}", sum(1 for r in processed_results if not isinstance(r, dict) or not r.get("error")), sum(1 for r in processed_results if isinstance(r, dict) and r.get("error")))
            return processed_results
        else:
            # Run tests sequentially
            results = []
            _log = logger.bind(ps_component="test_runner", prompt_id=prompt_id, total_tests=len(test_case_ids))
            _log.info("PS testrun.multi.start mode=sequential total_tests={}", len(test_case_ids))
            for test_id in test_case_ids:
                try:
                    result = await self.run_test_case(
                        prompt_id=prompt_id,
                        test_case_id=test_id,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        provider=provider,
                        app_config=app_config,
                        api_key_override=api_key_override,
                        credentials_resolved=credentials_resolved,
                        provider_credentials=provider_credentials,
                        timeout_seconds=timeout_seconds,
                        persist_run=True,
                        on_provider_success=on_provider_success,
                        strict_provider_errors=strict_provider_errors,
                    )
                    results.append(result)
                except Exception as e:
                    if strict_provider_errors:
                        raise
                    _log.error("PS testrun.multi.error test_case_id={} error={}", test_id, e)
                    results.append({
                        "test_case_id": test_id,
                        "error": str(e)
                    })
            ok = sum(1 for r in results if isinstance(r, dict) and not r.get("error"))
            failed = sum(1 for r in results if isinstance(r, dict) and r.get("error"))
            _log.info("PS testrun.multi.done mode=sequential ok={} failed={}", ok, failed)
            return results
