# unified_evaluation_service.py - Unified evaluation service combining all evaluation capabilities
"""
Unified evaluation service that combines OpenAI-compatible evaluation framework
with tldw-specific evaluation features.

This service provides:
- OpenAI Evals compatible evaluation management
- G-Eval, RAG, and response quality evaluations
- Dataset management
- Run management with async processing
- Webhook notifications
- Advanced metrics and monitoring
"""

import asyncio
import time
import uuid
from contextlib import suppress
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

_UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)

# Import database components
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_evaluations_database as _create_evals_db,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.audit_adapter import (
    log_dataset_created,
    log_dataset_created_async,
    log_dataset_deleted,
    log_dataset_deleted_async,
    log_evaluation_created,
    log_evaluation_created_async,
    log_evaluation_deleted,
    log_evaluation_deleted_async,
    log_evaluation_updated,
    log_evaluation_updated_async,
    log_run_cancelled,
    log_run_cancelled_async,
    log_run_started,
    log_run_started_async,
)
from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreaker
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope

# Import support services
from tldw_Server_API.app.core.Evaluations.metrics_advanced import advanced_metrics
from tldw_Server_API.app.core.Evaluations.persona_telemetry_metrics import (
    get_persona_telemetry_metrics_summary,
)
from tldw_Server_API.app.core.Evaluations.run_state import (
    can_transition_run_status,
    normalize_run_status,
)

# Import evaluation engines
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
from tldw_Server_API.app.core.Evaluations.webhook_identity import webhook_user_id_from_value
from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookEvent
from tldw_Server_API.app.core.testing import is_test_mode


def _await_webhook_inline_in_test_mode() -> bool:
    return is_test_mode()


class EvaluationType(str, Enum):
    """Supported evaluation types"""
    MODEL_GRADED = "model_graded"
    EXACT_MATCH = "exact_match"
    INCLUDES = "includes"
    GEVAL = "geval"
    RAG = "rag"
    RESPONSE_QUALITY = "response_quality"
    PROPOSITION_EXTRACTION = "proposition_extraction"
    QA3 = "qa3"
    OCR = "ocr"
    LABEL_CHOICE = "label_choice"
    NLI_FACTCHECK = "nli_factcheck"
    CUSTOM = "custom"


class UnifiedEvaluationService:
    """
    Unified service for all evaluation operations.

    Combines OpenAI-compatible evaluation framework with tldw-specific features
    into a single, cohesive service.
    """

    def __init__(
        self,
        db_path: str = str(DatabasePaths.get_evaluations_db_path(DatabasePaths.get_single_user_id())),
        *,
        enable_webhooks: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize the unified evaluation service.

        Args:
            db_path: Path to the evaluations database
            enable_webhooks: Toggle webhook delivery for evaluation lifecycle events
            enable_caching: Toggle optional caching layers (reserved for future use)
        """
        # Feature flags
        self.enable_webhooks = enable_webhooks
        self.enable_caching = enable_caching

        # Lifecycle flag
        self._initialized = False

        # Initialize database (allow override via env for tests)
        import os as _os
        _override_db = _os.getenv("EVALUATIONS_TEST_DB_PATH")
        effective_db_path = _override_db or db_path
        # Use backend-aware factory so Postgres content backend is honored
        self.db = _create_evals_db(db_path=effective_db_path)

        # Initialize evaluation runner for async processing (lazy import)
        from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner
        self.runner = EvaluationRunner(effective_db_path)

        # Initialize evaluation engines (lazy loading)
        self._rag_evaluator = None
        self._quality_evaluator = None
        self._ocr_evaluator = None

        # Initialize circuit breaker for resilience
        from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreakerConfig
        self.circuit_breaker = CircuitBreaker(
            name="evaluation_service",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=Exception
            )
        )

        # Audit logger shim for backward compatibility in tests
        class _AuditShim:
            def evaluation_created(self, *, user_id: str, eval_id: str, name: str, eval_type: str) -> None:
                log_evaluation_created(user_id=user_id, eval_id=eval_id, name=name, eval_type=eval_type)

            def evaluation_updated(self, *, user_id: str, eval_id: str, updates: dict[str, Any]) -> None:
                log_evaluation_updated(user_id=user_id, eval_id=eval_id, updates=updates)

            def evaluation_deleted(self, *, user_id: str, eval_id: str) -> None:
                log_evaluation_deleted(user_id=user_id, eval_id=eval_id)

            def run_started(self, *, user_id: str, run_id: str, eval_id: str, target_model: str) -> None:
                log_run_started(user_id=user_id, run_id=run_id, eval_id=eval_id, target_model=target_model)

            def run_cancelled(self, *, user_id: str, run_id: str) -> None:
                log_run_cancelled(user_id=user_id, run_id=run_id)

            def dataset_created(self, *, user_id: str, dataset_id: str, name: str, samples: int) -> None:
                log_dataset_created(user_id=user_id, dataset_id=dataset_id, name=name, samples=samples)

            def dataset_deleted(self, *, user_id: str, dataset_id: str) -> None:
                log_dataset_deleted(user_id=user_id, dataset_id=dataset_id)

        self.audit_logger = _AuditShim()

        # Initialize per-service webhook manager bound to this DB only if enabled
        self.webhook_manager = None
        if self.enable_webhooks:
            try:
                from tldw_Server_API.app.core.Evaluations.db_adapter import create_adapter_from_backend
                from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookManager

                backend = getattr(self.db, "backend", None)
                backend_type = getattr(self.db, "backend_type", None)
                backend_type_value = getattr(backend_type, "value", backend_type)

                if backend is not None and backend_type_value == "postgresql":
                    self.webhook_manager = WebhookManager(adapter=create_adapter_from_backend(backend))
                else:
                    self.webhook_manager = WebhookManager(db_path=effective_db_path)
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                self.webhook_manager = None

        logger.info("Unified Evaluation Service initialized")

    async def initialize(self) -> None:
        """Async initializer to align with test fixtures and future setup steps."""
        if self._initialized:
            return

        # Placeholder for future async setup (e.g., warming caches, migrations)
        self._initialized = True
        logger.debug("Unified Evaluation Service async initialization complete")

    async def shutdown(self) -> None:
        """Gracefully shut down background activity and release resources."""
        if not self._initialized:
            return

        runner_shutdown = getattr(self.runner, "shutdown", None)
        if callable(runner_shutdown):
            try:
                result = runner_shutdown()
                if asyncio.iscoroutine(result):
                    await result
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Evaluation runner shutdown encountered an error: {exc}")
        else:
            # Fallback: cancel any tracked tasks directly
            tasks_map = getattr(self.runner, "running_tasks", None)
            if isinstance(tasks_map, dict):
                for task in list(tasks_map.values()):
                    if not task or task.done():
                        continue
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                tasks_map.clear()

        self._initialized = False
        logger.debug("Unified Evaluation Service shutdown complete")

    # ============= Lazy Initialization =============

    def get_rag_evaluator(self) -> RAGEvaluator:
        """Get or create RAG evaluator instance"""
        if self._rag_evaluator is None:
            self._rag_evaluator = RAGEvaluator()
        return self._rag_evaluator

    def get_quality_evaluator(self) -> ResponseQualityEvaluator:
        """Get or create quality evaluator instance"""
        if self._quality_evaluator is None:
            self._quality_evaluator = ResponseQualityEvaluator()
        return self._quality_evaluator

    def get_ocr_evaluator(self):
        """Get or create OCR evaluator instance"""
        if self._ocr_evaluator is None:
            from tldw_Server_API.app.core.Evaluations.ocr_evaluator import OCREvaluator
            self._ocr_evaluator = OCREvaluator()
        return self._ocr_evaluator

    # ============= Evaluation Management =============

    async def create_evaluation(
        self,
        name: str,
        eval_type: str,
        eval_spec: dict[str, Any],
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
        created_by: str = "system"
    ) -> dict[str, Any]:
        """
        Create a new evaluation definition.

        Supports both OpenAI-style evaluation definitions and tldw-specific types.

        Args:
            name: Unique evaluation name
            eval_type: Type of evaluation (see EvaluationType enum)
            eval_spec: Evaluation specification
            description: Optional description
            dataset_id: ID of existing dataset
            dataset: Inline dataset (creates new dataset if provided)
            metadata: Optional metadata
            created_by: User/system creating the evaluation

        Returns:
            Created evaluation object
        """
        try:
            # Normalize eval type for CRUD-created evals
            eval_type_value = getattr(eval_type, "value", eval_type)
            eval_type_value = str(eval_type_value)
            eval_spec_payload = dict(eval_spec or {})
            if eval_type_value in {"geval", "rag", "response_quality"}:
                sub_type_map = {
                    "geval": "summarization",
                    "rag": "rag",
                    "response_quality": "response_quality",
                }
                eval_spec_payload.setdefault("sub_type", sub_type_map[eval_type_value])
                eval_type_value = "model_graded"

            # If inline dataset provided, create it first
            if dataset and not dataset_id:
                dataset_id = self.db.create_dataset(
                    name=f"{name}_dataset",
                    samples=dataset,
                    description=f"Dataset for {name}",
                    created_by=created_by
                )

            # Create evaluation
            eval_id = self.db.create_evaluation(
                name=name,
                description=description,
                eval_type=eval_type_value,
                eval_spec=eval_spec_payload,
                dataset_id=dataset_id,
                created_by=created_by,
                metadata=metadata
            )

            # Unified audit
            await log_evaluation_created_async(user_id=created_by, eval_id=eval_id, name=name, eval_type=eval_type_value)

            # Get and return created evaluation
            evaluation = self.db.get_evaluation(eval_id)
            if not evaluation:
                raise ValueError("Failed to retrieve created evaluation")

            return evaluation

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create evaluation: {e}")
            raise

    async def list_evaluations(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        eval_type: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> tuple[list[dict], bool]:
        """
        List evaluations with pagination and filtering.

        Args:
            limit: Maximum number of results
            after: Cursor for pagination
            eval_type: Filter by evaluation type
            created_by: Filter by creator

        Returns:
            Tuple of (evaluations list, has_more flag)
        """
        try:
            return self.db.list_evaluations(
                limit=limit,
                after=after,
                eval_type=eval_type,
                created_by=created_by
            )
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to list evaluations: {e}")
            raise

    async def get_evaluation(self, eval_id: str, *, created_by: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get evaluation by ID"""
        try:
            return self.db.get_evaluation(eval_id, created_by=created_by)
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get evaluation {eval_id}: {e}")
            raise

    async def update_evaluation(
        self,
        eval_id: str,
        updates: dict[str, Any],
        updated_by: str = "system",
        created_by: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Update evaluation definition and return the updated record.

        The underlying DB method returns a boolean. For API correctness and
        caller convenience, fetch and return the updated evaluation object
        when the update succeeds, or None if not found/updated.
        """
        try:
            success = self.db.update_evaluation(eval_id, updates, created_by=created_by)

            if not success:
                return None

            # Audit on success (mandatory)
            await log_evaluation_updated_async(user_id=updated_by, eval_id=eval_id, updates=updates)

            # Return the updated evaluation
            updated = self.db.get_evaluation(eval_id, created_by=created_by)
            return updated

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to update evaluation {eval_id}: {e}")
            raise

    async def delete_evaluation(
        self,
        eval_id: str,
        deleted_by: str = "system",
        created_by: Optional[str] = None,
    ) -> bool:
        """Soft delete an evaluation"""
        try:
            success = self.db.delete_evaluation(eval_id, created_by=created_by)

            if success:
                await log_evaluation_deleted_async(user_id=deleted_by, eval_id=eval_id)

            return success

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to delete evaluation {eval_id}: {e}")
            raise

    # ============= Run Management =============

    async def create_run(
        self,
        eval_id: str,
        target_model: str,
        config: Optional[dict] = None,
        dataset_override: Optional[dict] = None,
        webhook_url: Optional[str] = None,
        created_by: str = "system",
        webhook_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create and start an evaluation run.

        Args:
            eval_id: ID of evaluation to run
            target_model: Model to evaluate
            config: Run configuration
            dataset_override: Optional dataset override
            webhook_url: Optional webhook for notifications
            created_by: User creating the run

        Returns:
            Run object with status and ID
        """
        try:
            # Get evaluation
            evaluation = await self.get_evaluation(eval_id, created_by=created_by)
            if not evaluation:
                raise ValueError(f"Evaluation {eval_id} not found")

            run_id = f"run_{uuid.uuid4().hex[:12]}"

            # Unified audit (mandatory)
            await log_run_started_async(
                user_id=created_by,
                run_id=run_id,
                eval_id=eval_id,
                target_model=target_model,
            )

            # Create run record after mandatory audit persistence succeeds
            run_id = self.db.create_run(
                eval_id=eval_id,
                target_model=target_model,
                config=config or {},
                webhook_url=webhook_url,
                run_id=run_id,
            )

            # Send webhook for run started after the durable run row exists
            if webhook_url and self.enable_webhooks and self.webhook_manager:
                effective_webhook_user = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(created_by) or created_by
                await self.webhook_manager.send_webhook(
                    user_id=effective_webhook_user,
                    event=WebhookEvent.EVALUATION_STARTED,
                    evaluation_id=run_id,
                    data={
                        "eval_id": eval_id,
                        "target_model": target_model,
                        "eval_type": evaluation["eval_type"]
                    }
                )

            # Prepare evaluation config
            eval_config = {
                "eval_type": evaluation["eval_type"],
                "eval_spec": evaluation["eval_spec"],
                "dataset_id": evaluation.get("dataset_id"),
                "dataset_override": dataset_override,
                "config": config or {},
                "webhook_url": webhook_url,
                "created_by": created_by,
            }

            # Start async evaluation only after the mandatory audit and row persistence succeed
            asyncio.create_task(
                self._run_evaluation_async(
                    run_id=run_id,
                    eval_id=eval_id,
                    eval_config=eval_config,
                    created_by=created_by,
                    webhook_user_id=webhook_user_id,
                )
            )

            # Return run info
            run = self.db.get_run(run_id)
            return run

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create run: {e}")
            raise

    async def _run_evaluation_async(
        self,
        run_id: str,
        eval_id: str,
        eval_config: dict,
        created_by: str,
        webhook_user_id: Optional[str] = None,
    ):
        """Run evaluation asynchronously"""
        current_task = asyncio.current_task()
        if current_task:
            # Register the running task so cancel_run can stop it
            self.runner.running_tasks[run_id] = current_task
        try:
            # Run evaluation with circuit breaker protection
            await self.circuit_breaker.call(
                self.runner.run_evaluation,
                run_id=run_id,
                eval_id=eval_id,
                eval_config=eval_config,
                background=False
            )

            # Send completion webhook
            if eval_config.get("webhook_url") and self.enable_webhooks and getattr(self, "webhook_manager", None):
                effective_webhook_user = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(created_by) or created_by
                run = self.db.get_run(run_id)
                await self.webhook_manager.send_webhook(
                    user_id=effective_webhook_user,
                    event=WebhookEvent.EVALUATION_COMPLETED,
                    evaluation_id=run_id,
                    data={
                        "run_id": run_id,
                        "status": run["status"],
                        "results": run.get("results")
                    }
                )

        except asyncio.CancelledError:
            # Cancellation path: persist cancelled status and notify listeners
            run = self.db.get_run(run_id, created_by=created_by)
            current_status = normalize_run_status(run.get("status") if run else None)
            transitioned_to_cancelled = can_transition_run_status(current_status, "cancelled")
            if transitioned_to_cancelled:
                self.db.update_run_status(run_id, "cancelled")
            if (
                transitioned_to_cancelled
                and eval_config.get("webhook_url")
                and self.enable_webhooks
                and getattr(self, "webhook_manager", None)
            ):
                effective_webhook_user = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(created_by) or created_by
                await self.webhook_manager.send_webhook(
                    user_id=effective_webhook_user,
                    event=WebhookEvent.EVALUATION_CANCELLED,
                    evaluation_id=run_id,
                    data={"run_id": run_id}
                )
            raise

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Evaluation run {run_id} failed: {e}")

            # Update run status
            self.db.update_run_status(run_id, "failed", error_message=str(e))

            # Send failure webhook
            if eval_config.get("webhook_url") and self.enable_webhooks and getattr(self, "webhook_manager", None):
                effective_webhook_user = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(created_by) or created_by
                await self.webhook_manager.send_webhook(
                    user_id=effective_webhook_user,
                    event=WebhookEvent.EVALUATION_FAILED,
                    evaluation_id=run_id,
                    data={"error": str(e)}
                )
        finally:
            # Ensure the task registry is cleaned up
            self.runner.running_tasks.pop(run_id, None)

    async def get_run(self, run_id: str, *, created_by: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get run by ID"""
        try:
            return self.db.get_run(run_id, created_by=created_by)
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            raise

    async def list_runs(
        self,
        eval_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        after: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> tuple[list[dict], bool]:
        """List runs with filtering"""
        try:
            runs, has_more = self.db.list_runs(
                eval_id=eval_id,
                status=status,
                limit=limit,
                after=after,
                return_has_more=True,
                created_by=created_by,
            )
            return runs, has_more
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to list runs: {e}")
            raise

    async def cancel_run(self, run_id: str, cancelled_by: str = "system", *, created_by: Optional[str] = None) -> bool:
        """Cancel a running evaluation"""
        try:
            run = self.db.get_run(run_id, created_by=created_by) if created_by else self.db.get_run(run_id)
            if not run:
                return False

            current_status = normalize_run_status(run.get("status") if isinstance(run, dict) else None)
            task_was_registered = run_id in getattr(self.runner, "running_tasks", {})

            # Try to cancel via runner
            success = self.runner.cancel_run(run_id)
            task_cleaned_up = task_was_registered and run_id not in getattr(self.runner, "running_tasks", {})

            if task_cleaned_up and not success:
                return True

            if not success:
                if not can_transition_run_status(current_status, "cancelled"):
                    return False
                # Update status directly if not in runner
                self.db.update_run_status(run_id, "cancelled")

            await log_run_cancelled_async(user_id=cancelled_by, run_id=run_id)

            return True

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to cancel run {run_id}: {e}")
            raise

    # ============= Specialized Evaluations =============

    async def evaluate_geval(
        self,
        source_text: str,
        summary: str,
        metrics: Optional[list[str]] = None,
        api_name: str = "openai",
        api_key: Optional[str] = None,
        user_id: str = "system",
        webhook_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run G-Eval summarization evaluation.

        Args:
            source_text: Original text
            summary: Summary to evaluate
            metrics: Metrics to compute (default: all)
            api_name: LLM API to use
            api_key: Optional API key
            user_id: User running evaluation

        Returns:
            Evaluation results with metrics
        """
        try:
            start_time = time.time()

            # Track metrics
            # Lazy import to avoid heavy chat stack at module import time
            from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval

            if advanced_metrics.enabled:
                with advanced_metrics.track_sli_request("/evaluations/geval"):
                    result = await asyncio.to_thread(
                        run_geval,
                        transcript=source_text,
                        summary=summary,
                        api_key=api_key,
                        api_name=api_name,
                        save=False
                    )
            else:
                result = await asyncio.to_thread(
                    run_geval,
                    transcript=source_text,
                    summary=summary,
                    api_key=api_key,
                    api_name=api_name,
                    save=False
                )

            # Parse and structure results
            evaluation_time = time.time() - start_time

            # Store in database
            eval_id = await self._store_evaluation_result(
                evaluation_type="geval",
                input_data={
                    "source_text": source_text[:1000],  # Truncate for storage
                    "summary": summary[:500]
                },
                results=result,
                metadata={
                    "api_name": api_name,
                    "evaluation_time": evaluation_time,
                    "user_id": user_id
                }
            )

            # Emit webhook for completion (await in TEST_MODE for deterministic tests)
            if self.enable_webhooks:
                try:
                    import asyncio as _asyncio
                    effective_user_id = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(user_id) or user_id
                    if effective_user_id == "single_user":
                        try:
                            from tldw_Server_API.app.core.config import settings as _app_settings
                            effective_user_id = f"user_{_app_settings.get('SINGLE_USER_FIXED_ID', '1')}"
                        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                            effective_user_id = "user_1"
                    if self.webhook_manager and _await_webhook_inline_in_test_mode():
                        await self.webhook_manager.send_webhook(
                            user_id=effective_user_id,
                            event=WebhookEvent.EVALUATION_COMPLETED,
                            evaluation_id=eval_id,
                            data={
                                "evaluation_type": "geval",
                                "average_score": result.get("average_score", 0.0),
                                "processing_time": evaluation_time
                            }
                        )
                    elif self.webhook_manager:
                        _asyncio.create_task(self.webhook_manager.send_webhook(
                            user_id=effective_user_id,
                            event=WebhookEvent.EVALUATION_COMPLETED,
                            evaluation_id=eval_id,
                            data={
                                "evaluation_type": "geval",
                                "average_score": result.get("average_score", 0.0),
                                "processing_time": evaluation_time
                            }
                        ))
                except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                    # Never fail the evaluation due to webhook issues
                    pass

            return {
                "evaluation_id": eval_id,
                "results": result,
                "evaluation_time": evaluation_time
            }

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"G-Eval evaluation failed: {e}")
            raise

    async def evaluate_rag(
        self,
        query: str,
        contexts: list[str],
        response: str,
        ground_truth: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        api_name: str = "openai",
        api_key: Optional[str] = None,
        user_id: str = "system",
        webhook_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run RAG system evaluation.

        Args:
            query: User query
            contexts: Retrieved contexts
            response: Generated response
            ground_truth: Optional expected answer
            metrics: Metrics to compute
            api_name: LLM API to use
            user_id: User running evaluation

        Returns:
            Evaluation results with metrics
        """
        try:
            start_time = time.time()

            # Run evaluation
            evaluator = self.get_rag_evaluator() if not api_key else RAGEvaluator(api_key=api_key)
            results = await evaluator.evaluate(
                query=query,
                contexts=contexts,
                response=response,
                ground_truth=ground_truth,
                metrics=metrics,
                api_name=api_name
            )

            evaluation_time = time.time() - start_time

            # Store in database
            eval_id = await self._store_evaluation_result(
                evaluation_type="rag",
                input_data={
                    "query": query,
                    "num_contexts": len(contexts),
                    "response_length": len(response)
                },
                results=results,
                metadata={
                    "api_name": api_name,
                    "evaluation_time": evaluation_time,
                    "user_id": user_id
                }
            )

            # Emit webhook for completion (await in TEST_MODE for deterministic tests)
            if self.enable_webhooks:
                try:
                    import asyncio as _asyncio
                    effective_user_id = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(user_id) or user_id
                    if effective_user_id == "single_user":
                        try:
                            from tldw_Server_API.app.core.config import settings as _app_settings
                            effective_user_id = f"user_{_app_settings.get('SINGLE_USER_FIXED_ID', '1')}"
                        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                            effective_user_id = "user_1"
                    if self.webhook_manager and _await_webhook_inline_in_test_mode():
                        await self.webhook_manager.send_webhook(
                            user_id=effective_user_id,
                            event=WebhookEvent.EVALUATION_COMPLETED,
                            evaluation_id=eval_id,
                            data={
                                "evaluation_type": "rag",
                                "overall_score": results.get("overall_score", 0.0),
                                "processing_time": evaluation_time
                            }
                        )
                    elif self.webhook_manager:
                        _asyncio.create_task(self.webhook_manager.send_webhook(
                            user_id=effective_user_id,
                            event=WebhookEvent.EVALUATION_COMPLETED,
                            evaluation_id=eval_id,
                            data={
                                "evaluation_type": "rag",
                                "overall_score": results.get("overall_score", 0.0),
                                "processing_time": evaluation_time
                            }
                        ))
                except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                    pass

            return {
                "evaluation_id": eval_id,
                "results": results,
                "evaluation_time": evaluation_time
            }

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"RAG evaluation failed: {e}")
            raise

    async def evaluate_response_quality(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None,
        custom_criteria: Optional[dict] = None,
        api_name: str = "openai",
        api_key: Optional[str] = None,
        user_id: str = "system",
        webhook_user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Evaluate response quality.

        Args:
            prompt: Original prompt
            response: Generated response
            expected_format: Expected response format
            custom_criteria: Custom evaluation criteria
            api_name: LLM API to use
            user_id: User running evaluation

        Returns:
            Evaluation results with quality metrics
        """
        try:
            start_time = time.time()

            # Run evaluation
            results = await self.get_quality_evaluator().evaluate(
                prompt=prompt,
                response=response,
                expected_format=expected_format,
                custom_criteria=custom_criteria,
                api_name=api_name,
                api_key=api_key,
            )

            evaluation_time = time.time() - start_time

            # Store in database
            eval_id = await self._store_evaluation_result(
                evaluation_type="response_quality",
                input_data={
                    "prompt": prompt[:500],
                    "response_length": len(response),
                    "expected_format": expected_format
                },
                results=results,
                metadata={
                    "api_name": api_name,
                    "evaluation_time": evaluation_time,
                    "user_id": user_id
                }
            )

            # Emit webhook for completion (await in TEST_MODE for deterministic tests)
            if self.enable_webhooks:
                try:
                    import asyncio as _asyncio
                    effective_user_id = webhook_user_id_from_value(webhook_user_id) or webhook_user_id_from_value(user_id) or user_id
                    if effective_user_id == "single_user":
                        try:
                            from tldw_Server_API.app.core.config import settings as _app_settings
                            effective_user_id = f"user_{_app_settings.get('SINGLE_USER_FIXED_ID', '1')}"
                        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                            effective_user_id = "user_1"
                    if self.webhook_manager and _await_webhook_inline_in_test_mode():
                        await self.webhook_manager.send_webhook(
                            user_id=effective_user_id,
                            event=WebhookEvent.EVALUATION_COMPLETED,
                            evaluation_id=eval_id,
                            data={
                                "evaluation_type": "response_quality",
                                "overall_quality": results.get("overall_quality", 0.0),
                                "processing_time": evaluation_time
                            }
                        )
                    elif self.webhook_manager:
                        _asyncio.create_task(self.webhook_manager.send_webhook(
                            user_id=effective_user_id,
                            event=WebhookEvent.EVALUATION_COMPLETED,
                            evaluation_id=eval_id,
                            data={
                                "evaluation_type": "response_quality",
                                "overall_quality": results.get("overall_quality", 0.0),
                                "processing_time": evaluation_time
                            }
                        ))
                except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                    pass

            return {
                "evaluation_id": eval_id,
                "results": results,
                "evaluation_time": evaluation_time
            }

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Response quality evaluation failed: {e}")
            raise

    async def evaluate_ocr(
        self,
        items: list[dict[str, Any]],
        ocr_options: Optional[dict[str, Any]] = None,
        metrics: Optional[list[str]] = None,
        thresholds: Optional[dict[str, float]] = None,
        user_id: str = "system",
    ) -> dict[str, Any]:
        """Evaluate OCR effectiveness for provided documents.

        Each item supports keys: id, pdf_path|pdf_bytes|extracted_text, ground_truth_text
        """
        try:
            start_time = time.time()

            results = await self.get_ocr_evaluator().evaluate(
                items=items,
                metrics=metrics,
                ocr_options=ocr_options,
                thresholds=thresholds,
            )

            evaluation_time = time.time() - start_time

            eval_id = await self._store_evaluation_result(
                evaluation_type="ocr",
                input_data={
                    "count": len(items),
                    "metrics": metrics or ["cer", "wer", "coverage", "page_coverage"],
                    "thresholds": thresholds or {},
                },
                results=results,
                metadata={
                    "evaluation_time": evaluation_time,
                    "user_id": user_id,
                },
            )

            return {
                "evaluation_id": eval_id,
                "results": results,
                "evaluation_time": evaluation_time,
            }
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"OCR evaluation failed: {e}")
            raise

    async def evaluate_qa3(
        self,
        items: list[dict[str, Any]],
        allowed_labels: Optional[list[str]] = None,
        label_mapping: Optional[dict[str, str]] = None,
        generate_predictions: bool = False,
        api_name: str = "openai",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 3,
        user_id: str = "system"
    ) -> dict[str, Any]:
        """Evaluate tri-label QA accuracy and PRF per label.

        If generate_predictions is True, uses LLM to produce predictions.
        Otherwise expects 'prediction' on each item.
        """
        allowed = [l.upper() for l in (allowed_labels or ["SUPPORTED","REFUTED","NEI"])]
        qa3_timeout = 30.0

        def norm_label(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = str(x).strip()
            if label_mapping and s in label_mapping:
                s = label_mapping[s]
            s = s.upper()
            # normalize common variants
            aliases = {
                "TRUE": "SUPPORTED",
                "FALSE": "REFUTED",
                "NOT_ENTAILED": "NEI",
                "NOT_ENTAILMENT": "NEI",
            }
            s = aliases.get(s, s)
            return s

        def parse_prediction(text: str) -> Optional[str]:
            t = (text or "").upper()
            for lab in allowed:
                if lab in t:
                    return lab
            # try exact tokenization
            for lab in allowed:
                if t.strip() == lab:
                    return lab
            return None

        # Prompt builder
        def build_prompt(q: str, allowed_str: str, ctx: Optional[str]) -> str:
            p = (
                "You are a strict grader. Read the question and answer with exactly one token from the set.\n"
                f"Allowed labels: {allowed_str}. Respond with only one of these labels.\n\n"
            )
            if ctx:
                p += f"Context (optional):\n{ctx}\n\n"
            p += f"Question:\n{q}\n\nAnswer (one token):"
            return p

        # LLM call helper
        import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
        def call_llm(prompt: str) -> str:
            try:
                # use analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, streaming=False, recursive_summarization=False, chunked_summarization=False)
                result = sgl.analyze(api_name, prompt, None, api_key, "You output one token only.", temperature, False, False, False, None)
                if isinstance(result, tuple) and result:
                    return str(result[0])
                return str(result)
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"LLM call failed in QA3: {e}")
                return ""

        async def call_llm_async(prompt: str) -> str:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(call_llm, prompt),
                    timeout=qa3_timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM call timed out in QA3 after {qa3_timeout}s")
                return ""

        # Compute predictions
        results = []
        for it in items:
            q = it.get("question", "")
            gold = norm_label(it.get("label"))
            pred = norm_label(it.get("prediction"))
            if generate_predictions or not pred:
                prompt = build_prompt(q, ", ".join(allowed), it.get("context"))
                raw = await call_llm_async(prompt)
                parsed = parse_prediction(raw) or "NEI"
                pred = parsed
                results.append({"id": it.get("id"), "question": q, "gold": gold, "pred": pred, "raw": raw})
            else:
                results.append({"id": it.get("id"), "question": q, "gold": gold, "pred": pred})

        # Metrics
        cm: dict[str, dict[str, int]] = {g: dict.fromkeys(allowed, 0) for g in allowed}
        total = 0
        correct = 0
        for r in results:
            g = r.get("gold") or "NEI"
            p = r.get("pred") or "NEI"
            if g not in cm:
                cm[g] = dict.fromkeys(allowed, 0)
            if p not in cm[g]:
                for lab in cm:
                    cm[lab][p] = cm[lab].get(p, 0)
            cm[g][p] = cm[g].get(p, 0) + 1
            total += 1
            if g == p:
                correct += 1

        accuracy = (correct / total) if total else 0.0
        per_label = {}
        macro_f1 = 0.0
        for lab in allowed:
            tp = cm.get(lab, {}).get(lab, 0)
            fp = sum(cm[g].get(lab, 0) for g in cm if g != lab)
            fn = sum(cm[lab].get(p, 0) for p in cm[lab] if p != lab) if lab in cm else 0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_label[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(cm.get(lab, {}).values())}
            macro_f1 += f1
        macro_f1 = macro_f1 / len(allowed) if allowed else 0.0

        # Pack results
        payload = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_label": per_label,
            "confusion_matrix": cm,
            "results": results,
        }

        # Store
        eval_id = await self._store_evaluation_result(
            evaluation_type="qa3",
            input_data={"count": len(items), "generate": generate_predictions, "allowed_labels": allowed},
            results=payload,
            metadata={"user_id": user_id}
        )

        return {"evaluation_id": eval_id, "results": payload}

    async def evaluate_propositions(
        self,
        extracted: list[str],
        reference: list[str],
        method: str = "semantic",
        threshold: float = 0.7,
        user_id: str = "system"
    ) -> dict[str, Any]:
        """Evaluate proposition extraction against a reference set."""
        try:
            from tldw_Server_API.app.core.Chunking.utils.proposition_eval import evaluate_propositions as eval_props
            start = time.time()
            result = eval_props(extracted=extracted, reference=reference, method=method, threshold=threshold)
            evaluation_time = time.time() - start

            # Structure results as a dict
            results = {
                "metrics": {
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "claim_density_per_100_tokens": result.claim_density_per_100_tokens,
                    "avg_prop_len_tokens": result.avg_prop_len_tokens,
                    "dedup_rate": result.dedup_rate,
                },
                "counts": {
                    "matched": result.matched,
                    "total_extracted": result.total_extracted,
                    "total_reference": result.total_reference,
                },
                "details": result.details,
            }

            # Store result
            eval_id = await self._store_evaluation_result(
                evaluation_type="proposition_extraction",
                input_data={
                    "method": method,
                    "threshold": threshold,
                    "extracted_size": len(extracted),
                    "reference_size": len(reference),
                },
                results=results,
                metadata={
                    "user_id": user_id,
                    "evaluation_time": evaluation_time
                }
            )

            return {
                "evaluation_id": eval_id,
                "results": results,
                "evaluation_time": evaluation_time
            }

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Proposition evaluation failed: {e}")
            raise

    # ============= Dataset Management =============

    async def create_dataset(
        self,
        name: str,
        samples: list[dict],
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new dataset"""
        try:
            dataset_id = self.db.create_dataset(
                name=name,
                description=description,
                samples=samples,
                created_by=created_by,
                metadata=metadata
            )

            await log_dataset_created_async(user_id=created_by, dataset_id=dataset_id, name=name, samples=len(samples))

            return dataset_id

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    async def list_datasets(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        offset: int = 0,
        created_by: Optional[str] = None,
    ) -> tuple[list[dict], bool]:
        """List datasets with pagination"""
        try:
            return self.db.list_datasets(limit=limit, after=after, offset=offset, created_by=created_by)
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to list datasets: {e}")
            raise

    async def get_dataset(
        self,
        dataset_id: str,
        *,
        created_by: Optional[str] = None,
        include_samples: bool = True,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Optional[dict[str, Any]]:
        """Get dataset by ID"""
        try:
            return self.db.get_dataset(
                dataset_id,
                created_by=created_by,
                include_samples=include_samples,
                sample_limit=limit,
                sample_offset=offset,
            )
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e}")
            raise

    async def delete_dataset(
        self,
        dataset_id: str,
        deleted_by: str = "system",
        created_by: Optional[str] = None,
    ) -> bool:
        """Delete a dataset"""
        try:
            success = self.db.delete_dataset(dataset_id, created_by=created_by)

            if success:
                await log_dataset_deleted_async(user_id=deleted_by, dataset_id=dataset_id)

            return success

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            raise

    async def get_evaluation_history(
        self,
        user_id: str,
        evaluation_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        Get evaluation history for a user.

        Args:
            user_id: User identifier
            evaluation_type: Optional filter by evaluation type
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of evaluation records
        """
        try:
            evaluations = self.db.list_evaluations_filtered(
                limit=limit,
                offset=offset,
                eval_type=evaluation_type,
                created_by=user_id,
                start_date=start_date,
                end_date=end_date,
            )

            return evaluations

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get evaluation history: {e}")
            raise

    async def count_evaluations(
        self,
        user_id: str,
        evaluation_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Count evaluations matching criteria.

        Args:
            user_id: User identifier
            evaluation_type: Optional filter by evaluation type
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Total count of matching evaluations
        """
        try:
            return self.db.count_evaluations_filtered(
                eval_type=evaluation_type,
                created_by=user_id,
                start_date=start_date,
                end_date=end_date,
            )

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to count evaluations: {e}")
            raise

    # ============= Helper Methods =============

    @staticmethod
    def _user_id_variants(user_id: Optional[str]) -> set[str]:
        if not user_id:
            return set()
        raw = str(user_id).strip()
        if not raw:
            return set()
        variants = {raw}
        if raw.startswith("user_"):
            core = raw[5:]
            if core:
                variants.add(core)
        elif raw.isdigit():
            variants.add(f"user_{raw}")
        return variants

    @staticmethod
    def _extract_created_ts(record: dict[str, Any]) -> Optional[int]:
        value = record.get("created_at")
        if value is None:
            value = record.get("created")
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                ts = value.replace("Z", "+00:00")
                return int(datetime.fromisoformat(ts).timestamp())
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                return None
        return None

    async def _store_evaluation_result(
        self,
        evaluation_type: str,
        input_data: dict,
        results: Any,
        metadata: dict
    ) -> str:
        """Store evaluation result in database using unified approach"""
        try:
            # Generate evaluation ID
            eval_id = f"eval_{evaluation_type}_{uuid.uuid4().hex[:12]}"

            # Store using unified method if available (best-effort)
            if hasattr(self.db, 'store_unified_evaluation'):
                try:
                    success = self.db.store_unified_evaluation(
                        evaluation_id=eval_id,
                        name=f"{evaluation_type}_{int(time.time())}",
                        evaluation_type=evaluation_type,
                        input_data=input_data,
                        results=results if isinstance(results, dict) else {"result": results},
                        status="completed",
                        user_id=metadata.get("user_id", "system"),
                        metadata=metadata,
                        embedding_provider=metadata.get("embedding_provider"),
                        embedding_model=metadata.get("embedding_model")
                    )
                    if success:
                        logger.info(f"Stored evaluation {eval_id} in unified table")
                except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                    pass

            # Fallback to standard approach (ensure evals are runnable via CRUD)
            eval_type_value = evaluation_type
            eval_spec_payload = {"type": evaluation_type, "input": input_data}
            if metadata.get("api_name"):
                eval_spec_payload["api_name"] = metadata.get("api_name")
            if metadata.get("model"):
                eval_spec_payload["model"] = metadata.get("model")
            if evaluation_type in {"geval", "rag", "response_quality"}:
                sub_type_map = {
                    "geval": "summarization",
                    "rag": "rag",
                    "response_quality": "response_quality",
                }
                eval_type_value = "model_graded"
                eval_spec_payload.setdefault("sub_type", sub_type_map[evaluation_type])

            eval_id = self.db.create_evaluation(
                name=f"{evaluation_type}_{int(time.time())}",
                eval_type=eval_type_value,
                eval_spec=eval_spec_payload,
                created_by=metadata.get("user_id", "system"),
                metadata=metadata,
                eval_id=eval_id,
            )

            # Create and complete a run with results
            run_id = self.db.create_run(
                eval_id=eval_id,
                target_model=metadata.get("api_name", "unknown"),
                config={}
            )

            # Update run with results (atomically sets status and completed_at)
            self.db.store_run_results(
                run_id,
                results if isinstance(results, dict) else {"result": results},
                usage=metadata.get("usage"),
            )

            return eval_id

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to store evaluation result: {e}")
            return f"temp_{int(time.time())}"

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Get evaluation metrics summary"""
        try:
            summary: dict[str, Any]
            if advanced_metrics.enabled and hasattr(advanced_metrics, "get_summary"):
                advanced_summary = advanced_metrics.get_summary()
                summary = dict(advanced_summary) if isinstance(advanced_summary, dict) else {}
                summary.setdefault("metrics_enabled", True)
            else:
                summary = {"metrics_enabled": False}

            summary["persona_telemetry"] = get_persona_telemetry_metrics_summary()
            return summary
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get metrics summary: {e}")
            # Do not expose internal error details to external clients
            return {"error": "Failed to collect metrics"}

    async def health_check(self) -> dict[str, Any]:
        """Check service health"""
        try:
            # Check database connectivity with a lightweight probe
            db_healthy = False
            try:
                with self.db.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    _ = cur.fetchone()
                    db_healthy = True
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as db_err:
                logger.error(f"DB health probe failed: {db_err}")

            # Check circuit breaker
            from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitState
            cb_healthy = self.circuit_breaker.state != CircuitState.OPEN

            return {
                "status": "healthy" if (db_healthy and cb_healthy) else "degraded",
                "database": "connected" if db_healthy else "disconnected",
                "circuit_breaker": "closed" if cb_healthy else "open",
                "uptime": time.time(),
                "version": "1.0.0"
            }

        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_service_instance = None
_service_instances_lock = None

# LRU cache for per-user services to bound memory in long-lived servers
try:
    from collections import OrderedDict
except ImportError:  # pragma: no cover - stdlib guard
    OrderedDict = dict  # type: ignore

_MAX_SERVICE_INSTANCES = 128
_service_instances_by_user: "OrderedDict[str, UnifiedEvaluationService]" = OrderedDict()  # type: ignore[name-defined]


def get_unified_evaluation_service(db_path: Optional[str] = None) -> UnifiedEvaluationService:
    """
    Get or create the unified evaluation service singleton.

    Args:
        db_path: Optional database path override

    Returns:
        UnifiedEvaluationService instance
    """
    global _service_instance

    if _service_instance is None:
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
        _default_path = str(_DP.get_evaluations_db_path(_DP.get_single_user_id()))
        _service_instance = UnifiedEvaluationService(db_path or _default_path)

    return _service_instance


def get_unified_evaluation_service_for_user(user_id: str | int) -> UnifiedEvaluationService:
    """Get or create a per-user unified evaluation service bound to that user's DB.

    Uses the canonical evaluations user scope so tenant-style string ids bind to
    their own per-user DB instead of falling back to the default single-user DB.
    """
    # Lazy init lock to avoid import-time issues
    global _service_instances_lock
    if _service_instances_lock is None:
        import threading as _threading
        _service_instances_lock = _threading.Lock()

    with _service_instances_lock:
        uid_key = canonical_evaluations_user_scope(user_id)
        legacy_numeric_key: int | None = None
        try:
            legacy_numeric_key = int(uid_key)
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
            legacy_numeric_key = None
        # Return existing and mark as recently used
        if uid_key in _service_instances_by_user:
            svc = _service_instances_by_user.pop(uid_key)
            # If tests override the DB via env, ensure the cached instance matches
            try:
                import os as _os
                override_path = _os.getenv("EVALUATIONS_TEST_DB_PATH")
                if override_path and getattr(getattr(svc, "db", None), "db_path", None) != override_path:
                    # Replace with a new instance bound to the override path
                    svc = UnifiedEvaluationService(db_path=override_path)
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                pass
            _service_instances_by_user[uid_key] = svc
            return svc
        # Temporary migration path: older in-process callers cached services under numeric
        # user ids before canonical string scopes landed. Remove this branch after all
        # callers and long-lived workers have been restarted on the string-scope contract.
        if legacy_numeric_key is not None and legacy_numeric_key in _service_instances_by_user:  # type: ignore[operator]
            svc = _service_instances_by_user.pop(legacy_numeric_key)  # type: ignore[arg-type]
            _service_instances_by_user[uid_key] = svc
            return svc

        # Create new service for this user
        db_path = str(DatabasePaths.get_evaluations_db_path(uid_key))
        svc = UnifiedEvaluationService(db_path=db_path)
        _service_instances_by_user[uid_key] = svc

        # Evict least-recently-used if over capacity
        if hasattr(_service_instances_by_user, "popitem") and len(_service_instances_by_user) > _MAX_SERVICE_INSTANCES:  # type: ignore[attr-defined]
            try:
                old_user_id, old_svc = _service_instances_by_user.popitem(last=False)  # type: ignore[arg-type]
                # Best effort shutdown without blocking
                try:
                    import asyncio as _aio
                    if hasattr(old_svc, "shutdown"):
                        _aio.create_task(old_svc.shutdown())
                except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                    pass
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS:
                pass
        return svc
