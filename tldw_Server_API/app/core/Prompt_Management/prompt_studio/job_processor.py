# job_processor.py
# Job processing handlers for Prompt Studio

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Logging.log_context import (
    log_context,
    new_request_id,
)

from ..optimization_model_config import (
    normalize_durable_optimization_config,
    reconcile_optimization_strategy,
    strip_sensitive_optimization_config,
)
from .job_types import JobType
from .test_case_generator import TestCaseGenerator
from .test_case_manager import TestCaseManager

########################################################################################################################
# Job Processor

class JobProcessor:
    """Processes different types of Prompt Studio jobs."""

    def __init__(self, db: PromptStudioDatabase, job_manager: Optional[object] = None):
        """
        Initialize JobProcessor.

        Args:
            db: Database instance
            job_manager: Legacy job manager (ignored; retained for compatibility)
        """
        self.db = db
        self.job_manager = job_manager
        self.test_manager = TestCaseManager(db)
        self.test_generator = TestCaseGenerator(self.test_manager)
        self._handlers: dict[str, Any] = {}
        self._register_handlers()

    async def process_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Process a single job dict (core Jobs or legacy shape)."""
        job_type = str(job.get("job_type") or "").lower()
        handler = self._handlers.get(job_type)
        if not handler:
            raise ValueError(f"No handler registered for job type {job_type}")

        payload = job.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}

        entity_id = self._resolve_entity_id(job_type, job, payload)
        return await handler(payload, entity_id)

    def _register_handlers(self):
        """Register job handlers."""
        self._handlers[JobType.GENERATION.value] = self.process_generation_job
        self._handlers[JobType.EVALUATION.value] = self.process_evaluation_job
        self._handlers[JobType.OPTIMIZATION.value] = self.process_optimization_job

    def _resolve_entity_id(self, job_type: str, job: dict[str, Any], payload: dict[str, Any]) -> int:
        if job_type == JobType.OPTIMIZATION.value:
            value = payload.get("optimization_id") or payload.get("entity_id") or job.get("entity_id")
        elif job_type == JobType.EVALUATION.value:
            value = payload.get("evaluation_id") or payload.get("entity_id") or job.get("entity_id")
        elif job_type == JobType.GENERATION.value:
            value = payload.get("project_id") or payload.get("entity_id") or job.get("entity_id")
        else:
            value = payload.get("entity_id") or job.get("entity_id")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _positive_id(value: Any, *, label: str) -> int:
        """Return one positive resource ID or fail closed."""

        if isinstance(value, bool):
            raise ValueError(f"{label} is invalid")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{label} is invalid")
        try:
            resource_id = int(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{label} is invalid") from None
        if resource_id <= 0:
            raise ValueError(f"{label} is invalid")
        return resource_id

    @classmethod
    def _positive_id_list(cls, value: Any, *, label: str) -> list[int]:
        """Return a validated ordered resource snapshot."""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if not isinstance(value, list):
            raise ValueError(f"{label} snapshot is invalid")
        resource_ids = [
            cls._positive_id(candidate, label=label) for candidate in value
        ]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError(f"{label} snapshot is invalid")
        return resource_ids

    @staticmethod
    def _model_config_list(value: Any, *, label: str) -> list[dict[str, Any]]:
        """Return a validated ordered model-config snapshot."""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if isinstance(value, dict):
            value = [value]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(candidate, dict) for candidate in value)
        ):
            raise ValueError(f"{label} snapshot is invalid")
        return [dict(candidate) for candidate in value]

    def _require_active_project(self, project_id: Any) -> dict[str, Any]:
        """Load one non-deleted project in the active tenant scope."""

        selected_project_id = self._positive_id(project_id, label="project")
        project = self.db.get_project(
            selected_project_id,
            include_deleted=True,
        )
        if not project or project.get("deleted"):
            raise ValueError(f"Project {selected_project_id} not found")
        return project

    def _require_prompt_in_project(
        self,
        prompt_id: Any,
        project_id: Any,
    ) -> dict[str, Any]:
        """Load one non-deleted prompt and enforce its persisted project."""

        selected_prompt_id = self._positive_id(prompt_id, label="prompt")
        selected_project_id = self._positive_id(project_id, label="project")
        self._require_active_project(selected_project_id)
        prompt = self.db.get_prompt_with_project(
            selected_prompt_id,
            include_deleted=True,
        )
        if not prompt or prompt.get("deleted"):
            raise ValueError(f"Prompt {selected_prompt_id} not found")
        if self._positive_id(
            prompt.get("project_id"),
            label="prompt project",
        ) != selected_project_id:
            raise ValueError("Prompt does not belong to the optimization project")
        return prompt

    def _require_test_case_in_project(
        self,
        test_case_id: Any,
        project_id: Any,
    ) -> dict[str, Any]:
        """Load one non-deleted test case and enforce its persisted project."""

        selected_case_id = self._positive_id(test_case_id, label="test case")
        selected_project_id = self._positive_id(project_id, label="project")
        test_case = self.db.get_test_case(
            selected_case_id,
            include_deleted=True,
        )
        if not test_case or test_case.get("deleted"):
            raise ValueError(f"Test case {selected_case_id} not found")
        if self._positive_id(
            test_case.get("project_id"),
            label="test case project",
        ) != selected_project_id:
            raise ValueError(
                "Test case does not belong to the optimization project"
            )
        return test_case

    def _validated_optimization_resources(
        self,
        optimization: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[int, list[int]]:
        """Validate persisted resources and reject a stale queued snapshot."""

        project_id = self._positive_id(
            optimization.get("project_id"),
            label="project",
        )
        prompt_id = self._positive_id(
            optimization.get("initial_prompt_id"),
            label="prompt",
        )
        test_case_ids = self._positive_id_list(
            optimization.get("test_case_ids"),
            label="test case",
        )
        if "initial_prompt_id" in payload:
            queued_prompt_id = self._positive_id(
                payload.get("initial_prompt_id"),
                label="queued prompt",
            )
            if queued_prompt_id != prompt_id:
                raise ValueError(
                    "Queued prompt does not match the persisted optimization prompt"
                )
        if "test_case_ids" in payload:
            queued_test_case_ids = self._positive_id_list(
                payload.get("test_case_ids"),
                label="queued test case",
            )
            if queued_test_case_ids != test_case_ids:
                raise ValueError(
                    "Queued test case snapshot does not match the persisted optimization"
                )

        self._require_prompt_in_project(prompt_id, project_id)
        for test_case_id in test_case_ids:
            self._require_test_case_in_project(test_case_id, project_id)
        return prompt_id, test_case_ids

    @staticmethod
    def _completed_optimization_result(
        optimization_id: int,
        row: dict[str, Any],
        raw_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return completed fields from the authoritative Prompt row."""

        def _metrics(value: Any) -> dict[str, Any]:
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return {}
                return dict(parsed) if isinstance(parsed, dict) else {}
            return {}

        def _score(metrics: dict[str, Any]) -> Any:
            return next(
                (
                    metrics[key]
                    for key in ("score", "accuracy", "best_metric")
                    if metrics.get(key) is not None
                ),
                None,
            )

        initial_metrics = _metrics(row.get("initial_metrics"))
        final_metrics = _metrics(row.get("final_metrics"))
        iterations = int(row.get("iterations_completed") or 0)
        optimized_prompt_id = (
            row.get("optimized_prompt_id") or row.get("initial_prompt_id")
        )
        improvement_percentage = row.get("improvement_percentage")
        result = {
            "optimization_id": optimization_id,
            "status": str(row.get("status") or "completed").lower(),
            "optimized_prompt_id": optimized_prompt_id,
            "best_prompt_id": optimized_prompt_id,
            "iterations": iterations,
            "iterations_completed": iterations,
            "initial_score": _score(initial_metrics),
            "final_score": _score(final_metrics),
            "best_metric": _score(final_metrics),
            "improvement": (
                improvement_percentage / 100
                if isinstance(improvement_percentage, (int, float))
                else None
            ),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "total_tokens": row.get("total_tokens"),
            "total_cost": row.get("total_cost"),
        }
        for key in ("provider_dispatches", "scorer_provider_dispatched"):
            if raw_result is not None and key in raw_result:
                result[f"_{key}"] = raw_result[key]
        return result

    ####################################################################################################################
    # Generation Jobs

    async def process_generation_job(self, payload: dict[str, Any], entity_id: int) -> dict[str, Any]:
        """
        Process a test case generation job.

        Args:
            payload: Job payload with generation parameters
            entity_id: Project ID

        Returns:
            Generation results
        """
        try:
            project_id = entity_id
            generation_type = payload.get("type", "description")
            req_id = payload.get("request_id") or new_request_id()
            with log_context(
                ps_component="job_processor",
                ps_job_kind="generation",
                request_id=req_id,
                project_id=project_id,
                generation_type=generation_type,
                job_id=payload.get("job_id"),
            ):
                logger.info(
                    "PS generation.start project_id={} type={}",
                    project_id,
                    generation_type,
                )

                if generation_type == "diverse":
                    # Generate diverse test cases
                    generated = await self._generate_diverse_cases(project_id, payload)
                elif generation_type == "description":
                    # Generate from description
                    generated = await self._generate_from_description(project_id, payload)
                elif generation_type == "data":
                    # Generate from existing data
                    generated = await self._generate_from_data(project_id, payload)
                else:
                    raise ValueError(f"Unknown generation type: {generation_type}")

                result = {
                    "generated_count": len(generated),
                    "test_case_ids": [tc["id"] for tc in generated],
                    "generation_type": generation_type,
                    "timestamp": datetime.utcnow().isoformat()
                }

                logger.info(
                    "PS generation.done project_id={} type={} generated_count={} timestamp={}",
                    project_id,
                    generation_type,
                    len(generated),
                    result["timestamp"],
                )
                return result

        except Exception as e:
            logger.error(
                "PS generation.error project_id={} type={} error={}",
                payload.get("project_id") or entity_id,
                payload.get("type"),
                e,
            )
            raise

    async def _generate_diverse_cases(self, project_id: int, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate diverse test cases."""
        signature_id = payload.get("signature_id")
        num_cases = payload.get("num_cases", 5)

        if not signature_id:
            raise ValueError("signature_id required for diverse generation")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.test_generator.generate_diverse_cases,
            project_id, signature_id, num_cases
        )

    async def _generate_from_description(self, project_id: int, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate test cases from description."""
        description = payload.get("description")
        num_cases = payload.get("num_cases", 5)
        signature_id = payload.get("signature_id")
        prompt_id = payload.get("prompt_id")

        if not description:
            raise ValueError("description required for description-based generation")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.test_generator.generate_from_description,
            project_id, description, num_cases, signature_id, prompt_id
        )

    async def _generate_from_data(self, project_id: int, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate test cases from existing data."""
        source_data = payload.get("source_data", [])
        signature_id = payload.get("signature_id")

        if not source_data:
            raise ValueError("source_data required for data-based generation")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.test_generator.generate_from_existing_data,
            project_id, source_data, signature_id
        )

    ####################################################################################################################
    # Evaluation Jobs

    async def process_evaluation_job(self, payload: dict[str, Any], entity_id: int) -> dict[str, Any]:
        """
        Process an evaluation job.

        Args:
            payload: Job payload with evaluation parameters
            entity_id: Evaluation ID

        Returns:
            Evaluation results
        """
        try:
            evaluation_id = entity_id
            prompt_id = payload.get("prompt_id")
            test_case_ids = payload.get("test_case_ids", [])
            model_configs = payload.get("model_configs", [])
            req_id = payload.get("request_id") or new_request_id()
            with log_context(
                ps_component="job_processor",
                ps_job_kind="evaluation",
                request_id=req_id,
                evaluation_id=evaluation_id,
                prompt_id=prompt_id,
                job_id=payload.get("job_id"),
            ):
                logger.info(
                    "PS evaluation.start evaluation_id={} prompt_id={} test_cases={} models={}",
                    evaluation_id,
                    prompt_id,
                    len(test_case_ids),
                    len(model_configs),
                )

                evaluation = self.db.get_evaluation(evaluation_id)
                if not evaluation:
                    raise ValueError(f"Evaluation {evaluation_id} not found")
                project_id = self._positive_id(
                    evaluation.get("project_id"),
                    label="project",
                )
                persisted_prompt_id = self._positive_id(
                    evaluation.get("prompt_id"),
                    label="prompt",
                )
                if prompt_id is not None and self._positive_id(
                    prompt_id,
                    label="queued prompt",
                ) != persisted_prompt_id:
                    raise ValueError(
                        "Queued prompt does not match the persisted evaluation prompt"
                    )
                prompt_id = persisted_prompt_id
                persisted_test_case_ids = self._positive_id_list(
                    evaluation.get("test_case_ids"),
                    label="test case",
                )
                if "test_case_ids" in payload:
                    queued_test_case_ids = self._positive_id_list(
                        payload.get("test_case_ids"),
                        label="queued test case",
                    )
                    if queued_test_case_ids != persisted_test_case_ids:
                        raise ValueError(
                            "Queued test case snapshot does not match the "
                            "persisted evaluation"
                        )
                test_case_ids = persisted_test_case_ids

                persisted_model_configs = self._model_config_list(
                    evaluation.get("model_configs"),
                    label="model config",
                )
                if "model_configs" in payload:
                    queued_model_configs = self._model_config_list(
                        payload.get("model_configs"),
                        label="queued model config",
                    )
                    if queued_model_configs != persisted_model_configs:
                        raise ValueError(
                            "Queued model config snapshot does not match the "
                            "persisted evaluation"
                        )
                model_configs = persisted_model_configs

                self._require_prompt_in_project(prompt_id, project_id)
                for test_case_id in test_case_ids:
                    self._require_test_case_in_project(test_case_id, project_id)

                self.db.update_evaluation(
                    evaluation_id,
                    {
                        "status": "running",
                        "started_at": datetime.now(timezone.utc),
                    },
                )

                # Process test cases
                test_runs = []
                total_tokens = 0
                total_cost = 0.0

                for test_case_id in test_case_ids:
                    for model_config in model_configs:
                        # Simulate test execution (would call actual LLM here)
                        test_run = await self._execute_test_case(
                            prompt_id, test_case_id, model_config
                        )
                        test_runs.append(test_run)
                        total_tokens += test_run.get("tokens_used", 0)
                        total_cost += test_run.get("cost_estimate", 0.0)

                    # Add small delay to avoid rate limiting
                    await asyncio.sleep(0.1)

                # Calculate aggregate metrics
                aggregate_metrics = self._calculate_aggregate_metrics(test_runs)

                self.db.update_evaluation(
                    evaluation_id,
                    {
                        "status": "completed",
                        "completed_at": datetime.now(timezone.utc),
                        "test_run_ids": [tr["id"] for tr in test_runs],
                        "aggregate_metrics": aggregate_metrics,
                        "total_tokens": total_tokens,
                        "total_cost": total_cost,
                    },
                )

                result = {
                    "evaluation_id": evaluation_id,
                    "test_runs": len(test_runs),
                    "aggregate_metrics": aggregate_metrics,
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "status": "completed"
                }

                logger.info(
                    "PS evaluation.done evaluation_id={} runs={} tokens={} cost={}",
                    evaluation_id,
                    len(test_runs),
                    total_tokens,
                    total_cost,
                )
                return result

        except Exception as e:
            logger.error(
                "PS evaluation.error evaluation_id={} error={}",
                entity_id,
                e,
            )

            # Update evaluation status to failed
            self.db.update_evaluation(
                entity_id,
                {
                    "status": "failed",
                    "error_message": "Prompt Studio evaluation job failed",
                    "completed_at": datetime.now(timezone.utc),
                },
            )

            raise

    async def _execute_test_case(self, prompt_id: int, test_case_id: int,
                                 model_config: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a single test case (simulation).

        In production, this would call the actual LLM API.
        """
        import random
        import uuid

        # Simulate execution delay
        await asyncio.sleep(random.uniform(0.5, 1.5))

        # Resolve both persisted resources before creating a dependent run.
        test_case = self.test_manager.get_test_case(test_case_id)
        if not test_case or test_case.get("deleted"):
            raise ValueError(f"Test case {test_case_id} not found")
        test_case_project_id = self._positive_id(
            test_case.get("project_id"),
            label="test case project",
        )
        self._require_prompt_in_project(prompt_id, test_case_project_id)

        # Simulate test run result
        test_run = {
            "id": random.randint(1000, 9999),
            "uuid": str(uuid.uuid4()),
            "prompt_id": prompt_id,
            "test_case_id": test_case_id,
            "model_name": model_config.get("model", "gpt-3.5-turbo"),
            "inputs": test_case["inputs"],
            "outputs": {
                "result": f"Simulated output for {test_case.get('name', 'test')}"
            },
            "expected_outputs": test_case.get("expected_outputs"),
            "scores": {
                "accuracy": random.uniform(0.7, 1.0),
                "relevance": random.uniform(0.6, 1.0)
            },
            "execution_time_ms": random.randint(100, 2000),
            "tokens_used": random.randint(50, 500),
            "cost_estimate": random.uniform(0.001, 0.01)
        }

        # Store test run in database
        tc_project_id = test_case_project_id

        persisted = self.db.create_test_run(
            project_id=tc_project_id,
            prompt_id=prompt_id,
            test_case_id=test_case_id,
            model_name=test_run["model_name"],
            model_params=model_config,
            inputs=test_run["inputs"],
            outputs=test_run["outputs"],
            expected_outputs=test_run["expected_outputs"],
            scores=test_run["scores"],
            execution_time_ms=test_run["execution_time_ms"],
            tokens_used=test_run["tokens_used"],
            cost_estimate=test_run["cost_estimate"],
            client_id=self.db.client_id,
        )

        test_run["id"] = persisted.get("id", test_run.get("id"))
        test_run["uuid"] = persisted.get("uuid", test_run.get("uuid"))

        return test_run

    def _calculate_aggregate_metrics(self, test_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate metrics from test runs."""
        if not test_runs:
            return {}

        # Calculate averages
        total_accuracy = sum(tr.get("scores", {}).get("accuracy", 0) for tr in test_runs)
        total_relevance = sum(tr.get("scores", {}).get("relevance", 0) for tr in test_runs)

        metrics = {
            "total_runs": len(test_runs),
            "avg_accuracy": total_accuracy / len(test_runs),
            "avg_relevance": total_relevance / len(test_runs),
            "avg_execution_time_ms": sum(tr.get("execution_time_ms", 0) for tr in test_runs) / len(test_runs),
            "total_tokens": sum(tr.get("tokens_used", 0) for tr in test_runs),
            "total_cost": sum(tr.get("cost_estimate", 0) for tr in test_runs)
        }

        return metrics

    ####################################################################################################################
    # Optimization Jobs

    async def process_optimization_job(
        self,
        payload: dict[str, Any],
        entity_id: int,
        *,
        runtime_model_config: dict[str, Any] | None = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        runtime_scorer_model_config: dict[str, Any] | None = None,
        scorer_provider_credentials: ProviderCallCredentials | None = None,
        on_scorer_provider_success: Callable[[], Awaitable[None]] | None = None,
        before_finalize: Callable[[], Awaitable[bool]] | None = None,
        before_completion: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        manage_failure_status: bool = True,
    ) -> dict[str, Any]:
        """
        Process an optimization job.

        Args:
            payload: Job payload with optimization parameters
            entity_id: Optimization ID

        Returns:
            Optimization results
        """
        optimization_id = entity_id
        try:
            initial_prompt_id = payload.get("initial_prompt_id")
            optimizer_type = payload.get("optimizer_type", "basic")
            max_iterations = payload.get("max_iterations", 20)

            req_id = payload.get("request_id") or new_request_id()
            with log_context(
                ps_component="job_processor",
                ps_job_kind="optimization",
                request_id=req_id,
                optimization_id=optimization_id,
                optimizer_type=optimizer_type,
                job_id=payload.get("job_id"),
            ):
                logger.info(
                    "Processing optimization job {} with optimizer '{}' (max_iterations={})",
                    optimization_id,
                    optimizer_type,
                    max_iterations,
                )

            optimization = self.db.get_optimization(optimization_id, include_deleted=True)
            if optimization is None:
                raise ValueError(f"Optimization {optimization_id} not found")

            try:
                max_iterations = int(
                    payload.get("max_iterations")
                    or optimization.get("max_iterations")
                    or max_iterations
                )
            except (TypeError, ValueError):
                max_iterations = 20

            optimization_status = str(optimization.get("status") or "").lower()
            if optimization_status == "cancelled":
                final_metrics = optimization.get("final_metrics")
                if isinstance(final_metrics, str):
                    try:
                        final_metrics = json.loads(final_metrics)
                    except Exception:
                        final_metrics = {}
                if not isinstance(final_metrics, dict):
                    final_metrics = {}

                best_metric = (
                    final_metrics.get("score")
                    or final_metrics.get("accuracy")
                    or final_metrics.get("best_metric")
                )
                logger.info(
                    "PS optimization.skip_cancelled optimization_id={} strategy={} status=cancelled",
                    optimization_id,
                    optimizer_type,
                )
                return {
                    "optimization_id": optimization_id,
                    "iterations_completed": int(optimization.get("iterations_completed") or 0),
                    "best_prompt_id": optimization.get("optimized_prompt_id")
                    or optimization.get("initial_prompt_id"),
                    "best_metric": best_metric,
                    "status": "cancelled",
                }
            if optimization_status == "completed":
                return self._completed_optimization_result(
                    optimization_id,
                    optimization,
                )

            initial_prompt_id, _ = (
                self._validated_optimization_resources(
                    optimization,
                    payload,
                )
            )

            # Reconcile every persisted/queued strategy source before allowing
            # a queued snapshot to update the authoritative row.
            row_cfg = optimization.get("optimization_config")
            if isinstance(row_cfg, str):
                try:
                    row_cfg = json.loads(row_cfg)
                except Exception:
                    row_cfg = {}
            if not isinstance(row_cfg, dict):
                row_cfg = {}

            payload_config = payload.get("optimization_config")
            if isinstance(payload_config, str):
                try:
                    payload_config = json.loads(payload_config)
                except Exception:
                    payload_config = None
            normalized_payload_config: dict[str, Any] | None = None
            if isinstance(payload_config, dict) and payload_config:
                try:
                    normalized_payload_config = (
                        normalize_durable_optimization_config(
                            payload_config,
                            reject_sensitive=False,
                        )
                    )
                except ValueError:
                    scrubbed_config = strip_sensitive_optimization_config(
                        payload_config
                    )
                    self.db.update_optimization(
                        optimization_id,
                        {"optimization_config": scrubbed_config},
                    )
                    raise

            strategy = reconcile_optimization_strategy(
                optimization.get("optimizer_type"),
                row_cfg.get("optimizer_type"),
                row_cfg.get("strategy"),
                payload.get("optimizer_type"),
                (
                    normalized_payload_config.get("optimizer_type")
                    if normalized_payload_config is not None
                    else None
                ),
                (
                    normalized_payload_config.get("strategy")
                    if normalized_payload_config is not None
                    else None
                ),
            )

            updates: dict[str, Any] = {}
            if normalized_payload_config is not None:
                normalized_payload_config["optimizer_type"] = strategy
                normalized_payload_config.pop("strategy", None)
                updates["optimization_config"] = normalized_payload_config
            if updates:
                with log_context(
                    ps_component="job_processor",
                    ps_job_kind="optimization",
                    request_id=req_id,
                    optimization_id=optimization_id,
                ):
                    optimization = self.db.update_optimization(optimization_id, updates)

            # Normalization above rejects unknown strategies. Every accepted
            # strategy executes through the provider-bound durable engine;
            # compatibility strategies are mapped inside that engine.
            with log_context(
                ps_component="job_processor",
                ps_job_kind="optimization",
                request_id=req_id,
                optimization_id=optimization_id,
                optimizer_type=strategy,
                job_id=payload.get("job_id"),
            ):
                logger.info(
                    "Routing optimization job {} to OptimizationEngine (strategy={})",
                    optimization_id,
                    strategy,
                )
                from .optimization_engine import OptimizationEngine

                engine = OptimizationEngine(self.db)
                engine_kwargs: dict[str, Any] = {
                    "runtime_model_config": runtime_model_config,
                    "provider_credentials": provider_credentials,
                    "on_provider_success": on_provider_success,
                    "runtime_scorer_model_config": runtime_scorer_model_config,
                    "scorer_provider_credentials": scorer_provider_credentials,
                    "on_scorer_provider_success": on_scorer_provider_success,
                    "manage_failure_status": manage_failure_status,
                    "emit_completion_event": False,
                }
                if before_finalize is not None:
                    engine_kwargs["before_finalize"] = before_finalize
                if before_completion is not None:
                    engine_kwargs["before_completion"] = before_completion
                engine_result = await engine.optimize(
                    optimization_id,
                    **engine_kwargs,
                )

            latest = self.db.get_optimization(optimization_id, include_deleted=True) or {}
            result = dict(engine_result or {})
            if str(latest.get("status") or "").lower() == "completed":
                result = self._completed_optimization_result(
                    optimization_id,
                    latest,
                    result,
                )
            else:
                result.setdefault("optimization_id", optimization_id)
                result.setdefault("status", str(latest.get("status") or "completed"))
                result.setdefault(
                    "iterations_completed",
                    int(
                        latest.get("iterations_completed")
                        or result.get("iterations")
                        or 0
                    ),
                )
                if (
                    result.get("best_prompt_id") is None
                    and result.get("optimized_prompt_id") is not None
                ):
                    result["best_prompt_id"] = result.get(
                        "optimized_prompt_id"
                    )
                if (
                    result.get("best_metric") is None
                    and result.get("final_score") is not None
                ):
                    result["best_metric"] = result.get("final_score")

            logger.info(
                "PS optimization.engine_done optimization_id={} strategy={} status={} iterations={}",
                optimization_id,
                strategy,
                result.get("status"),
                result.get("iterations_completed"),
            )
            return result

        except Exception as e:  # noqa: BLE001
            logger.error(
                "PS optimization.error optimization_id={} error_type={}",
                locals().get("optimization_id"),
                type(e).__name__,
            )

            if manage_failure_status:
                try:
                    self.db.set_optimization_status(
                        optimization_id,
                        "failed",
                        error_message=(
                            str(e)
                            if isinstance(e, ValueError)
                            else "Optimization provider execution failed"
                        ),
                        mark_completed=True,
                    )
                except Exception as status_exc:  # noqa: BLE001
                    logger.warning(
                        "PS optimization.mark_failed_failed optimization_id={} error_type={}",
                        optimization_id,
                        type(status_exc).__name__,
                    )

            raise
