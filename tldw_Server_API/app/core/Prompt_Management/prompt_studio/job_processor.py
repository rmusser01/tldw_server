# job_processor.py
# Job processing handlers for Prompt Studio

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Logging.log_context import (
    log_context,
    new_request_id,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import (
    EventBroadcaster,
)

from .job_types import JobType
from .test_case_generator import TestCaseGenerator
from .test_case_manager import TestCaseManager

try:
    # Import connection manager used by WebSocket endpoints
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
        connection_manager as ws_connection_manager,
    )
except Exception:
    ws_connection_manager = None

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

    def _ensure_ps_prompt_exists(self, prompt_id: Optional[int], project_id: Optional[int]) -> None:
        """Ensure a minimal prompt exists in prompt_studio_prompts for the given IDs.

        Some tests insert evaluation/optimization rows referencing a prompt_id
        that was not previously created. This guard creates a stub prompt row
        to prevent foreign key failures when inserting dependent rows like test_runs.
        """
        if not prompt_id or not project_id:
            return

        try:
            self.db.ensure_prompt_stub(
                prompt_id=prompt_id,
                project_id=project_id,
                name=f"Auto-Created Prompt {prompt_id}",
            )
        except Exception as exc:  # noqa: BLE001 - defensive guard for legacy sqlite paths
            logger.warning(
                "PS ensure_prompt_stub failed: prompt_id={} project_id={} error={}",
                prompt_id,
                project_id,
                exc,
            )

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
                if evaluation:
                    self._ensure_ps_prompt_exists(
                        evaluation.get("prompt_id"),
                        evaluation.get("project_id"),
                    )

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

                evaluation = self.db.get_evaluation(evaluation_id)
                if evaluation:
                    self._ensure_ps_prompt_exists(
                        evaluation.get("prompt_id"),
                        evaluation.get("project_id"),
                    )

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

        # Get test case
        test_case = self.test_manager.get_test_case(test_case_id)

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
        tc_project_id = test_case.get("project_id")
        self._ensure_ps_prompt_exists(prompt_id, tc_project_id)

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

    async def process_optimization_job(self, payload: dict[str, Any], entity_id: int) -> dict[str, Any]:
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

            project_id = optimization.get("project_id")
            if initial_prompt_id is None:
                initial_prompt_id = optimization.get("initial_prompt_id")

            self._ensure_ps_prompt_exists(initial_prompt_id, project_id)

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
                    "best_prompt_id": optimization.get("optimized_prompt_id") or initial_prompt_id,
                    "best_metric": best_metric,
                    "status": "cancelled",
                }

            # Keep optimization row in sync with queued payload fields so
            # OptimizationEngine receives the runtime test set/config.
            payload_test_case_ids = payload.get("test_case_ids")
            payload_config = payload.get("optimization_config")
            if isinstance(payload_config, str):
                try:
                    payload_config = json.loads(payload_config)
                except Exception:
                    payload_config = None
            updates: dict[str, Any] = {}
            if isinstance(payload_test_case_ids, list):
                updates["test_case_ids"] = payload_test_case_ids
            if isinstance(payload_config, dict) and payload_config:
                updates["optimization_config"] = payload_config
            if updates:
                with log_context(
                    ps_component="job_processor",
                    ps_job_kind="optimization",
                    request_id=req_id,
                    optimization_id=optimization_id,
                ):
                    optimization = self.db.update_optimization(optimization_id, updates)

            row_cfg = optimization.get("optimization_config")
            if isinstance(row_cfg, str):
                try:
                    row_cfg = json.loads(row_cfg)
                except Exception:
                    row_cfg = {}
            if not isinstance(row_cfg, dict):
                row_cfg = {}

            strategy = str(
                payload.get("optimizer_type")
                or row_cfg.get("optimizer_type")
                or row_cfg.get("strategy")
                or optimization.get("optimizer_type")
                or "mipro"
            ).strip().lower()

            # Stage-1 production wiring: execute real optimization engine for
            # supported strategies, including MCTS.
            if strategy in {"mipro", "bootstrap", "mcts"}:
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
                    engine_result = await engine.optimize(optimization_id)

                latest = self.db.get_optimization(optimization_id, include_deleted=True) or {}
                result = dict(engine_result or {})
                result.setdefault("optimization_id", optimization_id)
                result.setdefault("status", str(latest.get("status") or "completed"))
                result.setdefault(
                    "iterations_completed",
                    int(latest.get("iterations_completed") or result.get("iterations") or 0),
                )
                if result.get("best_prompt_id") is None and result.get("optimized_prompt_id") is not None:
                    result["best_prompt_id"] = result.get("optimized_prompt_id")
                if result.get("best_metric") is None and result.get("final_score") is not None:
                    result["best_metric"] = result.get("final_score")

                logger.info(
                    "PS optimization.engine_done optimization_id={} strategy={} status={} iterations={}",
                    optimization_id,
                    strategy,
                    result.get("status"),
                    result.get("iterations_completed"),
                )
                return result

            logger.info(
                "PS optimization.legacy_fallback optimization_id={} strategy={}",
                optimization_id,
                strategy,
            )
            return await self._process_optimization_job_legacy(
                optimization_id=optimization_id,
                initial_prompt_id=initial_prompt_id,
                project_id=project_id,
                max_iterations=max_iterations,
            )

        except Exception as e:  # noqa: BLE001
            logger.error(
                "PS optimization.error optimization_id={} error={}",
                locals().get("optimization_id"),
                e,
            )

            try:
                self.db.set_optimization_status(
                    optimization_id,
                    "failed",
                    error_message=str(e),
                    mark_completed=True,
                )
            except Exception as status_exc:  # noqa: BLE001
                logger.warning(
                    "PS optimization.mark_failed_failed optimization_id={} error={}",
                    optimization_id,
                    status_exc,
                )

            raise

    async def _process_optimization_job_legacy(
        self,
        *,
        optimization_id: int,
        initial_prompt_id: Optional[int],
        project_id: Optional[int],
        max_iterations: int,
    ) -> dict[str, Any]:
        """Legacy optimization simulation path for unsupported strategies."""
        initial_row = self.db.get_optimization(optimization_id, include_deleted=True) or {}
        if str(initial_row.get("status") or "").lower() == "cancelled":
            logger.info(
                "PS optimization.legacy_skip_cancelled optimization_id={}",
                optimization_id,
            )
            return {
                "optimization_id": optimization_id,
                "iterations_completed": int(initial_row.get("iterations_completed") or 0),
                "best_prompt_id": initial_row.get("optimized_prompt_id") or initial_prompt_id,
                "best_metric": None,
                "improvement_percentage": 0.0,
                "status": "cancelled",
            }

        self.db.set_optimization_status(
            optimization_id,
            "running",
            mark_started=True,
        )

        best_prompt_id = initial_prompt_id
        best_metric = 0.5
        iterations: list[dict[str, Any]] = []
        total_tokens = 0
        total_cost = 0.0
        iteration_limit = max(1, min(max_iterations, 5))
        was_cancelled = False

        for iteration_index in range(1, iteration_limit + 1):
            current = self.db.get_optimization(optimization_id, include_deleted=True) or {}
            if str(current.get("status") or "").lower() == "cancelled":
                logger.info(
                    "PS optimization.legacy_cancelled optimization_id={} iteration={}",
                    optimization_id,
                    iteration_index,
                )
                was_cancelled = True
                break

            iteration_result = await self._run_optimization_iteration(
                optimization_id,
                initial_prompt_id,
                iteration_index,
            )
            iterations.append(iteration_result)
            total_tokens += iteration_result.get("tokens_used", 0)
            total_cost += iteration_result.get("cost", 0.0)

            try:
                self.db.record_optimization_iteration(
                    optimization_id,
                    iteration_number=iteration_result.get("iteration", iteration_index),
                    prompt_variant=None,
                    metrics={"metric": iteration_result.get("metric")},
                    tokens_used=iteration_result.get("tokens_used"),
                    cost=iteration_result.get("cost"),
                    note=None,
                )
            except Exception as iteration_exc:  # noqa: BLE001
                logger.warning(
                    "PS optimization.iteration_record_failed optimization_id={} iteration={} error={}",
                    optimization_id,
                    iteration_index,
                    iteration_exc,
                )

            metric_value = iteration_result.get("metric", 0.0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_prompt_id = iteration_result.get("prompt_id", initial_prompt_id)

            if best_metric > 0.95:
                logger.info(
                    "PS optimization.early_stop optimization_id={} iteration={} metric={}",
                    optimization_id,
                    iteration_index,
                    round(best_metric, 3),
                )
                break

            try:
                if ws_connection_manager:
                    broadcaster = EventBroadcaster(ws_connection_manager, self.db)
                    await broadcaster.broadcast_optimization_iteration(
                        optimization_id=optimization_id,
                        iteration=iteration_index,
                        max_iterations=max_iterations,
                        current_metric=metric_value,
                        best_metric=best_metric,
                    )
            except Exception as broadcast_exc:  # noqa: BLE001
                logger.warning(
                    "PS optimization.broadcast_failed optimization_id={} iteration={} error={}",
                    optimization_id,
                    iteration_index,
                    broadcast_exc,
                )

            await asyncio.sleep(0.5)

        initial_metric = 0.5
        improvement = ((best_metric - initial_metric) / initial_metric) * 100

        if best_prompt_id:
            self._ensure_ps_prompt_exists(best_prompt_id, project_id)

        if was_cancelled:
            self.db.update_optimization(
                optimization_id,
                {
                    "optimized_prompt_id": best_prompt_id,
                    "iterations_completed": len(iterations),
                    "initial_metrics": {"accuracy": initial_metric},
                    "final_metrics": {"accuracy": best_metric},
                    "improvement_percentage": improvement,
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                },
                set_completed_at=True,
            )
        else:
            self.db.complete_optimization(
                optimization_id,
                optimized_prompt_id=best_prompt_id,
                iterations_completed=len(iterations),
                initial_metrics={"accuracy": initial_metric},
                final_metrics={"accuracy": best_metric},
                improvement_percentage=improvement,
                total_tokens=total_tokens,
                total_cost=total_cost,
            )

        result = {
            "optimization_id": optimization_id,
            "iterations_completed": len(iterations),
            "best_prompt_id": best_prompt_id,
            "best_metric": best_metric,
            "improvement_percentage": improvement,
            "status": "cancelled" if was_cancelled else "completed",
        }

        logger.info(
            "PS optimization.done optimization_id={} iterations={} best_metric={} improvement_pct={} tokens={} cost={}",
            optimization_id,
            len(iterations),
            round(best_metric, 3),
            round(improvement, 1),
            total_tokens,
            total_cost,
        )
        return result

    async def _run_optimization_iteration(self, optimization_id: int,
                                         prompt_id: Optional[int], iteration: int) -> dict[str, Any]:
        """
        Run a single optimization iteration (simulation).

        In production, this would implement actual optimization logic.
        """
        import random

        # Simulate iteration processing
        await asyncio.sleep(random.uniform(1, 2))

        # Simulate metric improvement
        metric = 0.5 + (iteration * 0.1) + random.uniform(-0.05, 0.1)
        metric = min(1.0, max(0.0, metric))  # Clamp to [0, 1]

        return {
            "iteration": iteration,
            "prompt_id": prompt_id,
            "metric": metric,
            "tokens_used": random.randint(100, 1000),
            "cost": random.uniform(0.01, 0.1)
        }
