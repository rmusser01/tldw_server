# response_quality_evaluator.py - Response Quality Evaluation Module
"""
Evaluation module for assessing the quality of generated responses.

Evaluates:
- Relevance to prompt
- Completeness
- Accuracy
- Format compliance
- Clarity and coherence
- Custom criteria
"""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    DaemonCapacityError,
    await_bounded_sync_call,
    await_owned_worker,
)
from tldw_Server_API.app.core.Evaluations.circuit_breaker import llm_circuit_breaker
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
    SummaryProviderError,
    analyze,
)


async def _run_bounded_quality_analyze(*args: Any, **kwargs: Any) -> Any:
    """Run a sync quality adapter inside the breaker's single deadline."""
    return await await_bounded_sync_call(
        partial(analyze, *args, **kwargs),
        pool=SYNC_ADAPTER_CALL_POOL,
        exhaustion_message="Response quality provider capacity is exhausted",
    )


class ResponseQualityEvaluator:
    """Evaluator for response quality assessment"""

    async def evaluate(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None,
        custom_criteria: Optional[dict[str, str]] = None,
        api_name: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate the quality of a generated response.

        Args:
            prompt: Original prompt
            response: Generated response
            expected_format: Expected response format/structure
            custom_criteria: Additional evaluation criteria
            api_name: LLM API to use
            app_config: Provider configuration captured with the API key
            credentials_resolved: Whether the key/config snapshot is authoritative

        Returns:
            Evaluation results with metrics and suggestions
        """
        results = {
            "metrics": {},
            "overall_quality": 0.0,
            "format_compliance": True,
            "issues": [],
            "improvements": []
        }

        # Core quality metrics
        task_factories: list[Callable[[], Awaitable[tuple]]] = [
            lambda: self._evaluate_relevance(
                prompt, response, api_name, api_key, model, app_config,
                credentials_resolved, provider_credentials
            ),
            lambda: self._evaluate_completeness(
                prompt, response, api_name, api_key, model, app_config,
                credentials_resolved, provider_credentials
            ),
            lambda: self._evaluate_clarity(
                response, api_name, api_key, model, app_config,
                credentials_resolved, provider_credentials
            ),
            lambda: self._evaluate_accuracy(
                prompt, response, api_name, api_key, model, app_config,
                credentials_resolved, provider_credentials
            ),
        ]

        # Format compliance check
        if expected_format:
            task_factories.append(
                lambda: self._check_format_compliance(
                    response,
                    expected_format,
                    api_name,
                    api_key,
                    model,
                    app_config,
                    credentials_resolved,
                    provider_credentials,
                )
            )

        # Custom criteria evaluation
        if custom_criteria:
            for criterion_name, criterion_desc in custom_criteria.items():
                task_factories.append(
                    lambda criterion_name=criterion_name, criterion_desc=criterion_desc: (
                        self._evaluate_custom_criterion(
                            prompt,
                            response,
                            criterion_name,
                            criterion_desc,
                            api_name,
                            api_key,
                            model,
                            app_config,
                            credentials_resolved,
                            provider_credentials,
                        )
                    )
                )

        # Authenticate with one metric before fanning out. A rejected key must
        # cause exactly one provider request and no successful partial result.
        first_result = await task_factories[0]()
        remaining_results = await await_owned_worker(
            asyncio.gather(
                *(factory() for factory in task_factories[1:]),
                return_exceptions=True,
            )
        )
        for result in remaining_results:
            if isinstance(result, BaseException):
                raise result
        evaluation_results = [
            first_result,
            *remaining_results,
        ]

        # Process results
        format_scores = []
        for result in evaluation_results:
            if isinstance(result, tuple):
                metric_name, metric_data = result
                if metric_name == "format_compliance":
                    results["format_compliance"] = metric_data["compliant"]
                    if not metric_data["compliant"]:
                        results["issues"].extend(metric_data.get("issues", []))
                else:
                    results["metrics"][metric_name] = metric_data
                    if metric_name not in ["custom_" + k for k in (custom_criteria or {})]:
                        format_scores.append(metric_data["score"])

        # Calculate overall quality
        if format_scores:
            results["overall_quality"] = sum(format_scores) / len(format_scores)

        # Generate improvement suggestions
        results["improvements"] = self._generate_improvements(results)

        return results

    async def _evaluate_relevance(
        self,
        prompt: str,
        response: str,
        api_name: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> tuple:
        """Evaluate how relevant the response is to the prompt"""
        evaluation_prompt = f"""
        Evaluate how relevant and appropriate the following response is to the given prompt.

        Prompt: {prompt}

        Response: {response}

        Rate relevance on a scale of 1-5 where:
        1 = Completely off-topic or unrelated
        2 = Minimally relevant with significant deviation
        3 = Partially relevant but missing key aspects
        4 = Mostly relevant with minor gaps
        5 = Highly relevant and directly addresses the prompt

        Provide only the numeric score.
        """

        try:
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_bounded_quality_analyze,
                api_name,  # First param
                response,  # input_data
                evaluation_prompt,  # custom_prompt_arg
                api_key,   # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=app_config,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                raise_on_error=True,
            )

            score = float(score_str.strip()) / 5.0

            return ("relevance", {
                "name": "relevance",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "How well the response addresses the prompt"
            })

        except (SummaryProviderError, DaemonCapacityError):
            raise
        except Exception as e:
            logger.error("Relevance evaluation failed error_type={}", type(e).__name__)
            return ("relevance", {
                "name": "relevance",
                "score": 0.0,
                "explanation": "Provider evaluation failed."
            })

    async def _evaluate_completeness(
        self,
        prompt: str,
        response: str,
        api_name: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> tuple:
        """Evaluate if the response is complete"""
        evaluation_prompt = f"""
        Evaluate the completeness of the following response to the given prompt.

        Prompt: {prompt}

        Response: {response}

        Rate completeness on a scale of 1-5 where:
        1 = Severely incomplete, missing most required information
        2 = Incomplete with several missing elements
        3 = Partially complete with some gaps
        4 = Mostly complete with minor omissions
        5 = Fully complete and comprehensive

        Provide only the numeric score.
        """

        try:
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_bounded_quality_analyze,
                api_name,  # First param
                response,  # input_data
                evaluation_prompt,  # custom_prompt_arg
                api_key,   # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=app_config,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                raise_on_error=True,
            )

            score = float(score_str.strip()) / 5.0

            return ("completeness", {
                "name": "completeness",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "How complete and comprehensive the response is"
            })

        except (SummaryProviderError, DaemonCapacityError):
            raise
        except Exception as e:
            logger.error("Completeness evaluation failed error_type={}", type(e).__name__)
            return ("completeness", {
                "name": "completeness",
                "score": 0.0,
                "explanation": "Provider evaluation failed."
            })

    async def _evaluate_clarity(
        self,
        response: str,
        api_name: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> tuple:
        """Evaluate clarity and coherence of the response"""
        evaluation_prompt = f"""
        Evaluate the clarity, coherence, and readability of the following response.

        Response: {response}

        Rate clarity on a scale of 1-5 where:
        1 = Very unclear, confusing, or incoherent
        2 = Somewhat unclear with significant issues
        3 = Moderately clear with some confusing parts
        4 = Mostly clear with minor issues
        5 = Very clear, well-structured, and easy to understand

        Provide only the numeric score.
        """

        try:
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_bounded_quality_analyze,
                api_name,  # First param
                response,  # input_data
                evaluation_prompt,  # custom_prompt_arg
                api_key,   # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=app_config,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                raise_on_error=True,
            )

            score = float(score_str.strip()) / 5.0

            return ("clarity", {
                "name": "clarity",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "Clarity and coherence of the response"
            })

        except (SummaryProviderError, DaemonCapacityError):
            raise
        except Exception as e:
            logger.error("Clarity evaluation failed error_type={}", type(e).__name__)
            return ("clarity", {
                "name": "clarity",
                "score": 0.0,
                "explanation": "Provider evaluation failed."
            })

    async def _evaluate_accuracy(
        self,
        prompt: str,
        response: str,
        api_name: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> tuple:
        """Evaluate factual accuracy of the response"""
        evaluation_prompt = f"""
        Evaluate the factual accuracy and correctness of the following response.

        Prompt: {prompt}

        Response: {response}

        Rate accuracy on a scale of 1-5 where:
        1 = Contains significant factual errors or misinformation
        2 = Some factual errors that affect reliability
        3 = Mix of accurate and questionable information
        4 = Mostly accurate with minor uncertainties
        5 = Highly accurate and factually correct

        Provide only the numeric score.
        """

        try:
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_bounded_quality_analyze,
                api_name,  # First param
                response,  # input_data
                evaluation_prompt,  # custom_prompt_arg
                api_key,   # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=app_config,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                raise_on_error=True,
            )

            score = float(score_str.strip()) / 5.0

            return ("accuracy", {
                "name": "accuracy",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "Factual accuracy of the response"
            })

        except (SummaryProviderError, DaemonCapacityError):
            raise
        except Exception as e:
            logger.error("Accuracy evaluation failed error_type={}", type(e).__name__)
            return ("accuracy", {
                "name": "accuracy",
                "score": 0.0,
                "explanation": "Provider evaluation failed."
            })

    async def _check_format_compliance(
        self,
        response: str,
        expected_format: str,
        api_name: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> tuple:
        """Check if response matches expected format"""
        evaluation_prompt = f"""
        Check if the following response matches the expected format.

        Expected Format: {expected_format}

        Response: {response}

        Evaluate:
        1. Does the response follow the expected format? (yes/no)
        2. List any format violations or deviations (or "none" if compliant)

        Format your response as:
        COMPLIANT: yes/no
        ISSUES: [list issues or "none"]
        """

        try:
            result = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_bounded_quality_analyze,
                api_name,  # First param
                response,  # input_data
                evaluation_prompt,  # custom_prompt_arg
                api_key,   # api_key (None to load from config)
                "You are a format compliance checker. Be precise and systematic.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=app_config,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                raise_on_error=True,
            )

            # Parse result
            compliant = "yes" in result.lower().split("compliant:")[1].split("\n")[0]

            issues = []
            if "issues:" in result.lower():
                issues_text = result.lower().split("issues:")[1].strip()
                if "none" not in issues_text:
                    issues = [issue.strip() for issue in issues_text.split(",") if issue.strip()]

            return ("format_compliance", {
                "compliant": compliant,
                "issues": issues
            })

        except (SummaryProviderError, DaemonCapacityError):
            raise
        except Exception as e:
            logger.error("Format compliance check failed error_type={}", type(e).__name__)
            return ("format_compliance", {
                "compliant": False,
                "issues": ["Provider format evaluation failed."]
            })

    async def _evaluate_custom_criterion(
        self,
        prompt: str,
        response: str,
        criterion_name: str,
        criterion_desc: str,
        api_name: str,
        api_key: Optional[str],
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        credentials_resolved: bool = False,
        provider_credentials: ProviderCallCredentials | None = None,
    ) -> tuple:
        """Evaluate a custom criterion"""
        evaluation_prompt = f"""
        Evaluate the following response based on this custom criterion:

        Criterion: {criterion_name}
        Description: {criterion_desc}

        Original Prompt: {prompt}

        Response: {response}

        Rate on a scale of 1-5 how well the response meets this criterion.
        Provide only the numeric score.
        """

        try:
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                _run_bounded_quality_analyze,
                api_name,  # First param
                response,  # input_data
                evaluation_prompt,  # custom_prompt_arg
                api_key,   # api_key (None to load from config)
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1,       # temp
                model_override=model,
                app_config=app_config,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                raise_on_error=True,
            )

            score = float(score_str.strip()) / 5.0

            return (f"custom_{criterion_name}", {
                "name": criterion_name,
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": criterion_desc
            })

        except (SummaryProviderError, DaemonCapacityError):
            raise
        except Exception as e:
            logger.error("Custom criterion evaluation failed error_type={}", type(e).__name__)
            return (f"custom_{criterion_name}", {
                "name": criterion_name,
                "score": 0.0,
                "explanation": "Provider evaluation failed."
            })

    def _generate_improvements(self, results: dict[str, Any]) -> list[str]:
        """Generate improvement suggestions based on evaluation results"""
        improvements = []

        metrics = results.get("metrics", {})

        # Check relevance
        if "relevance" in metrics and metrics["relevance"]["score"] < 0.7:
            improvements.append("Improve response relevance by better understanding the prompt intent")

        # Check completeness
        if "completeness" in metrics and metrics["completeness"]["score"] < 0.7:
            improvements.append("Ensure all aspects of the prompt are addressed comprehensively")

        # Check clarity
        if "clarity" in metrics and metrics["clarity"]["score"] < 0.7:
            improvements.append("Enhance clarity by using simpler language and better structure")

        # Check accuracy
        if "accuracy" in metrics and metrics["accuracy"]["score"] < 0.7:
            improvements.append("Verify factual information and reduce speculative content")

        # Check format compliance
        if not results.get("format_compliance", True):
            improvements.append("Follow the specified format requirements more closely")

        # Overall quality
        if results.get("overall_quality", 0) < 0.6:
            improvements.append("Consider regenerating the response with refined prompting")

        return improvements
