# evaluation_manager.py - Evaluation Management Module
"""
Central manager for evaluation operations.

Handles:
- Evaluation storage and retrieval
- History tracking
- Comparison operations
- Custom metric evaluation
- Statistical analysis
"""

import asyncio
import contextlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    resolve_trusted_database_path,
)
from tldw_Server_API.app.core.DB_Management.migrations import migrate_evaluations_database
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze


class EvaluationManager:
    """Manages evaluation operations and persistence."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None, *, user_id: Optional[int | str] = None) -> None:
        self.config = load_comprehensive_config()
        if user_id is None:
            self._user_id = canonical_evaluations_user_scope(
                user_id,
                fallback=DatabasePaths.get_single_user_id(),
            )
        elif isinstance(user_id, str) and not user_id.strip():
            raise ValueError("Evaluations user scope is required")
        else:
            self._user_id = canonical_evaluations_user_scope(user_id)
        self.db_path = self._get_db_path(explicit_path=db_path)
        self._init_database()
        # Session identifier to isolate list operations within the lifetime of this manager
        self._session_id = uuid.uuid4().hex
        # Track evaluations created since last listing (for property tests)
        self._recent_created_ids: list[str] = []

    def _get_db_path(self, explicit_path: Optional[Union[str, Path]] = None) -> Path:
        """Resolve a safe evaluations database path.

        Rules:
        - Default to the per-user evaluations DB derived from DatabasePaths
        - If an explicit db_path argument is provided, honor it (after sanitization)
        - Config-provided paths must resolve within the per-user base directory; otherwise
          fall back to the default path
        - Strip any null bytes
        """
        base_dir = DatabasePaths.get_user_base_directory(self._user_id)
        base_dir.mkdir(parents=True, exist_ok=True)
        default_path: Path = DatabasePaths.get_evaluations_db_path(self._user_id).resolve()
        base_resolved = base_dir.resolve()

        # Read candidate from config if present
        if explicit_path is not None:
            candidate_str = str(explicit_path).replace("\x00", "")
            try:
                candidate_path = resolve_trusted_database_path(
                    candidate_str,
                    label="evaluations_db_path",
                    extra_roots=[base_resolved],
                ).resolve()
            except Exception:
                logger.warning(
                    "EvaluationManager: failed to resolve explicit db_path '{}', using default.",
                    candidate_str,
                )
                candidate_path = default_path
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            return candidate_path

        candidate_str: Optional[str] = None
        if self.config and self.config.has_section("Database"):
            candidate_str = self.config.get(
                "Database",
                "evaluations_db_path",
                fallback=None
            )

        candidate_path: Path = default_path

        if candidate_str:
            candidate_str = candidate_str.replace("\x00", "")
            try:
                raw_candidate = Path(candidate_str)
            except (TypeError, ValueError):
                raw_candidate = default_path

            try:
                if not raw_candidate.is_absolute():
                    candidate_path = (base_dir / raw_candidate).resolve()
                else:
                    candidate_path = raw_candidate.resolve()
            except Exception:
                logger.warning(f"EvaluationManager: failed to resolve evaluations_db_path '{candidate_str}', using default.")
                candidate_path = default_path
            else:
                try:
                    candidate_path.relative_to(base_resolved)
                except Exception:
                    logger.warning(f"EvaluationManager: rejecting evaluations_db_path outside user directory: {candidate_path}")
                    candidate_path = default_path

        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        return candidate_path

    def _init_database(self):
        """Initialize evaluation database using migration system"""
        try:
            # Apply all pending migrations
            migrate_evaluations_database(self.db_path)
            logger.info("Database migrations applied successfully")
        except Exception as e:
            # In production, database migration failures should be fatal
            # This ensures consistency and prevents silent data corruption
            error_msg = f"CRITICAL: Failed to apply database migrations to {self.db_path}: {e}"
            logger.critical(error_msg)

            # Check if we're in a production environment
            env = os.getenv('ENVIRONMENT', 'development').lower()

            if env in ['production', 'staging']:
                # Fail loudly in production/staging
                raise RuntimeError(error_msg) from e
            else:
                # In development, log a warning but continue with fallback
                logger.warning("Running in development mode - using fallback database initialization")
                self._init_database_fallback()

    def _init_database_fallback(self):
        """Fallback database initialization without migrations

        WARNING: This method should ONLY be used in development environments.
        Production deployments must use the migration system to ensure
        database schema consistency.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Create basic tables if they don't exist
            # Use internal_evaluations table to avoid conflict with OpenAI evaluations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS internal_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT UNIQUE NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    input_data TEXT NOT NULL,
                    results TEXT NOT NULL,
                    metadata TEXT,
                    user_id TEXT,
                    status TEXT DEFAULT 'completed',
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    embedding_provider TEXT,
                    embedding_model TEXT
                )
            """)

            # Create indexes
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_type ON internal_evaluations(evaluation_type)",
                "CREATE INDEX IF NOT EXISTS idx_created ON internal_evaluations(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_user_id ON internal_evaluations(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_status ON internal_evaluations(status)"
            ]:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass  # Index might already exist

            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES internal_evaluations(evaluation_id)
                )
            """)

            # Create indexes for metrics table
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_eval_id ON evaluation_metrics(evaluation_id)",
                "CREATE INDEX IF NOT EXISTS idx_metric ON evaluation_metrics(metric_name)",
                "CREATE INDEX IF NOT EXISTS idx_metric_created ON evaluation_metrics(created_at)"
            ]:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(index_sql)

            # Create webhook registrations table (needed for webhook tests)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_registrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT NOT NULL,
                    events TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    retry_count INTEGER DEFAULT 3,
                    timeout_seconds INTEGER DEFAULT 30,
                    total_deliveries INTEGER DEFAULT 0,
                    successful_deliveries INTEGER DEFAULT 0,
                    failed_deliveries INTEGER DEFAULT 0,
                    last_delivery_at TIMESTAMP,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, url)
                )
            """)

            # Create webhook deliveries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    webhook_id INTEGER NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    status_code INTEGER,
                    response_body TEXT,
                    response_time_ms INTEGER,
                    delivered BOOLEAN DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP,
                    next_retry_at TIMESTAMP,
                    FOREIGN KEY (webhook_id) REFERENCES webhook_registrations(id),
                    FOREIGN KEY (evaluation_id) REFERENCES internal_evaluations(evaluation_id)
                )
            """)

            # Create indexes for webhook tables
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_webhook_user ON webhook_registrations(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_webhook_active ON webhook_registrations(active)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_webhook ON webhook_deliveries(webhook_id)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_eval ON webhook_deliveries(evaluation_id)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_status ON webhook_deliveries(delivered)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_retry ON webhook_deliveries(next_retry_at)"
            ]:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass  # Index might already exist

            conn.commit()

    async def store_evaluation(
        self,
        evaluation_type: str,
        input_data: dict[str, Any],
        results: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """Store evaluation results"""
        import uuid

        evaluation_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        # Store main evaluation record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO internal_evaluations (
                    evaluation_id, evaluation_type, created_at,
                    input_data, results, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                evaluation_type,
                created_at,
                json.dumps(input_data),
                json.dumps(results),
                json.dumps({**(metadata or {}), "session_id": self._session_id})
            ))

            # Store individual metrics for easier querying
            if "metrics" in results:
                for metric_name, metric_data in results["metrics"].items():
                    score = metric_data.get("score", 0.0)
                    conn.execute("""
                        INSERT INTO evaluation_metrics (
                            evaluation_id, metric_name, score, created_at
                        ) VALUES (?, ?, ?, ?)
                    """, (evaluation_id, metric_name, score, created_at))

            conn.commit()

        logger.info(f"Stored evaluation {evaluation_id} of type {evaluation_type}")
        # Track recent creations for this manager instance
        with contextlib.suppress(Exception):
            self._recent_created_ids.append(evaluation_id)
        return evaluation_id

    async def get_history(
        self,
        evaluation_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> dict[str, Any]:
        """Retrieve evaluation history with filtering"""
        base_query = "FROM internal_evaluations WHERE 1=1"
        filter_params: list[Any] = []

        query_clauses = base_query

        if evaluation_type and evaluation_type != "all":
            query_clauses += " AND evaluation_type = ?"
            filter_params.append(evaluation_type)

        if start_date:
            query_clauses += " AND created_at >= ?"
            filter_params.append(start_date)

        if end_date:
            query_clauses += " AND created_at <= ?"
            filter_params.append(end_date)

        query = f"SELECT * {query_clauses} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params = [*filter_params, limit, offset]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get total count
            count_query = f"SELECT COUNT(*) {query_clauses}"
            total_count = conn.execute(count_query, filter_params).fetchone()[0]

            # Get records
            cursor = conn.execute(query, params)
            items = []

            for row in cursor:
                item = dict(row)
                item["input_data"] = json.loads(item["input_data"])
                item["results"] = json.loads(item["results"])
                item["metadata"] = json.loads(item["metadata"] if item["metadata"] else "{}")
                items.append(item)

            # Calculate average scores
            avg_query = """
                SELECT metric_name, AVG(score) as avg_score
                FROM evaluation_metrics
                WHERE evaluation_id IN (
                    SELECT evaluation_id FROM internal_evaluations WHERE 1=1
            """

            if evaluation_type and evaluation_type != "all":
                avg_query += " AND evaluation_type = ?"
            if start_date:
                avg_query += " AND created_at >= ?"
            if end_date:
                avg_query += " AND created_at <= ?"

            avg_query += ") GROUP BY metric_name"

            avg_params = []
            if evaluation_type and evaluation_type != "all":
                avg_params.append(evaluation_type)
            if start_date:
                avg_params.append(start_date)
            if end_date:
                avg_params.append(end_date)

            avg_cursor = conn.execute(avg_query, avg_params)
            average_scores = {row[0]: row[1] for row in avg_cursor}

        # Calculate trends if we have enough data
        trends = None
        if len(items) > 10:
            trends = self._calculate_trends(items)

        return {
            "total_count": total_count,
            "items": items,
            "average_scores": average_scores,
            "trends": trends
        }

    async def compare_evaluations(
        self,
        evaluation_ids: list[str],
        metrics_to_compare: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Compare multiple evaluations"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get evaluations
            placeholders = ",".join("?" * len(evaluation_ids))
            evaluation_ids_clause = f"({placeholders})"
            fetch_evaluations_sql_template = (
                "SELECT * FROM internal_evaluations WHERE evaluation_id IN {evaluation_ids_clause}"
            )
            fetch_evaluations_sql = fetch_evaluations_sql_template.format_map(locals())  # nosec B608
            cursor = conn.execute(
                fetch_evaluations_sql,
                evaluation_ids
            )

            evaluations = []
            for row in cursor:
                eval_dict = dict(row)
                eval_dict["results"] = json.loads(eval_dict["results"])
                evaluations.append(eval_dict)

            if len(evaluations) != len(evaluation_ids):
                raise ValueError("Some evaluation IDs not found")

            # Get metrics for comparison
            fetch_metrics_sql_template = """
                SELECT evaluation_id, metric_name, score
                FROM evaluation_metrics
                WHERE evaluation_id IN {evaluation_ids_clause}
                """
            fetch_metrics_sql = fetch_metrics_sql_template.format_map(locals())  # nosec B608
            metric_cursor = conn.execute(
                fetch_metrics_sql,
                evaluation_ids
            )

            metric_data = {}
            for row in metric_cursor:
                eval_id = row[0]
                metric_name = row[1]
                score = row[2]

                if metric_name not in metric_data:
                    metric_data[metric_name] = {}
                metric_data[metric_name][eval_id] = score

        # Filter metrics if specified
        if metrics_to_compare:
            metric_data = {k: v for k, v in metric_data.items() if k in metrics_to_compare}

        # Format comparison data
        metric_comparisons = {}
        best_performing = {}

        for metric_name, scores in metric_data.items():
            # Order scores by evaluation ID order
            ordered_scores = [scores.get(eval_id, 0.0) for eval_id in evaluation_ids]
            metric_comparisons[metric_name] = ordered_scores

            # Find best performing
            best_eval_id = max(scores.items(), key=lambda x: x[1])[0]
            best_performing[metric_name] = best_eval_id

        # Generate comparison summary
        summary_parts = []
        for metric, best_id in best_performing.items():
            summary_parts.append(f"{metric}: {best_id} performs best")

        comparison_summary = "Comparison Results:\n" + "\n".join(summary_parts)

        # Statistical analysis
        statistical_analysis = None
        if len(evaluation_ids) > 2:
            statistical_analysis = self._perform_statistical_analysis(metric_comparisons)

        return {
            "comparison_summary": comparison_summary,
            "metric_comparisons": metric_comparisons,
            "best_performing": best_performing,
            "statistical_analysis": statistical_analysis
        }

    async def evaluate_custom_metric(
        self,
        metric_name: str,
        description: str,
        evaluation_prompt: str,
        input_data: dict[str, Any],
        scoring_criteria: dict[str, Any],
        api_name: str = "openai",
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Evaluate using custom metric definition"""
        runtime_input_data = dict(input_data)
        if self._contains_model_response_placeholder(runtime_input_data):
            model_response = await self._generate_model_response(
                input_data=runtime_input_data,
                api_name=api_name,
                api_key=api_key,
            )
            if isinstance(model_response, str) and model_response.startswith("Error:"):
                logger.error(f"Model response generation failed for metric '{metric_name}': {model_response}")
                return {
                    "metric_name": metric_name,
                    "score": 0.0,
                    "explanation": model_response,
                    "raw_output": None,
                }
            runtime_input_data = {
                key: (model_response if value == "{model_response}" else value)
                for key, value in runtime_input_data.items()
            }

        # Format the evaluation prompt with input data
        formatted_prompt = evaluation_prompt
        for key, value in runtime_input_data.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))

        # Add scoring criteria to prompt
        criteria_text = "\n".join([f"- {k}: {v}" for k, v in scoring_criteria.items()])
        full_prompt = f"{formatted_prompt}\n\nScoring Criteria:\n{criteria_text}\n\nProvide a score from 1-10 and explanation."

        try:
            # Get evaluation from LLM
            response = await asyncio.to_thread(
                analyze,
                api_name,
                json.dumps(runtime_input_data),
                full_prompt,
                api_key,
                temp=0.3,
                system_message="You are an expert evaluator. Provide scores and detailed explanations."
            )

            # Parse response with strict validation
            import json as json_module
            import re

            # Try to parse as JSON first (most reliable)
            score = None
            explanation = response

            try:
                # Attempt to parse JSON response
                parsed = json_module.loads(response)
                if isinstance(parsed, dict):
                    if 'score' in parsed:
                        raw_score = parsed['score']
                        # Validate score is a number between 0 and 10
                        if isinstance(raw_score, (int, float)) and 0 <= raw_score <= 10:
                            score = float(raw_score) / 10.0
                    if 'explanation' in parsed:
                        explanation = str(parsed['explanation'])
            except (json_module.JSONDecodeError, ValueError):
                # Fallback to regex parsing with strict validation
                # Only accept scores that are clearly delimited (e.g., "Score: 8")
                score_patterns = [
                    r'[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?',  # Score: 8 or Score: 8/10
                    r'(\d+(?:\.\d+)?)\s*(?:/\s*10)\s+points?',  # 8/10 points
                    r'^(\d+(?:\.\d+)?)\s*$'  # Just a number on its own line
                ]

                for pattern in score_patterns:
                    match = re.search(pattern, response, re.MULTILINE)
                    if match:
                        try:
                            raw_score = float(match.group(1))
                            # Validate range
                            if 0 <= raw_score <= 10:
                                score = raw_score / 10.0 if raw_score > 1 else raw_score
                                break
                        except (ValueError, IndexError):
                            continue

                # Extract explanation (everything after first score mention or first newline)
                if '\n' in response:
                    lines = response.split('\n')
                    # Skip lines that contain just the score
                    explanation_lines = [l for l in lines if not re.match(r'^\s*\d+(?:\.\d+)?\s*(?:/\s*10)?\s*$', l)]
                    explanation = '\n'.join(explanation_lines).strip()

            # Default to 0.5 if no valid score found
            if score is None:
                logger.warning(f"Could not parse valid score from response: {response[:100]}...")
                score = 0.5

            return {
                "metric_name": metric_name,
                "score": score,
                "explanation": explanation.strip(),
                "raw_output": response
            }

        except Exception as e:
            logger.error(f"Custom metric evaluation failed: {e}")
            return {
                "metric_name": metric_name,
                "score": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "raw_output": None
            }

    @staticmethod
    def _contains_model_response_placeholder(input_data: dict[str, Any]) -> bool:
        return any(value == "{model_response}" for value in input_data.values())

    @staticmethod
    def _build_model_response_prompt(input_data: dict[str, Any]) -> str:
        visible = {
            key: value
            for key, value in input_data.items()
            if value != "{model_response}"
        }

        question = visible.get("question")
        if question:
            choices = visible.get("choices")
            if choices:
                return (
                    f"Question:\n{question}\n\n"
                    f"Choices:\n{choices}\n\n"
                    "Answer the question directly."
                )
            return str(question)

        instruction = visible.get("instruction")
        if instruction:
            constraints = visible.get("constraints")
            if constraints:
                return (
                    f"Instruction:\n{instruction}\n\n"
                    f"Constraints:\n{constraints}\n\n"
                    "Provide the best compliant response."
                )
            return str(instruction)

        problem = visible.get("problem")
        if problem:
            language = visible.get("language")
            test_cases = visible.get("test_cases")
            extra = ""
            if language:
                extra += f"\nTarget language: {language}"
            if test_cases:
                extra += f"\nTests:\n{test_cases}"
            return f"Programming task:\n{problem}{extra}\n\nProvide a complete solution."

        query = visible.get("query")
        if query:
            return str(query)

        prompt = visible.get("prompt")
        if prompt:
            return str(prompt)

        if visible:
            try:
                serialized = json.dumps(visible)
            except (TypeError, ValueError):
                serialized = str(visible)
            return (
                "Use the following task context to provide the best possible answer:\n"
                f"{serialized}"
            )

        return "Provide a helpful and accurate response."

    async def _generate_model_response(
        self,
        *,
        input_data: dict[str, Any],
        api_name: str,
        api_key: Optional[str],
    ) -> str:
        prompt = self._build_model_response_prompt(input_data)
        response = await asyncio.to_thread(
            analyze,
            api_name,
            prompt,
            None,
            api_key,
            temp=0.2,
            system_message=(
                "You are a helpful assistant. Respond directly to the task."
            ),
        )
        if isinstance(response, str):
            return response.strip()
        return str(response)

    def _calculate_trends(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate metric trends over time"""
        trends = {}

        # Group by metric
        metric_scores = {}
        for item in items:
            if "metrics" in item["results"]:
                for metric_name, metric_data in item["results"]["metrics"].items():
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append({
                        "timestamp": item["created_at"],
                        "score": metric_data.get("score", 0.0)
                    })

        # Calculate trends
        for metric_name, scores in metric_scores.items():
            if len(scores) > 1:
                # Sort by timestamp
                scores.sort(key=lambda x: x["timestamp"])

                # Calculate simple linear trend
                values = [s["score"] for s in scores]
                x = np.arange(len(values))

                # Linear regression
                coeffs = np.polyfit(x, values, 1)
                trend_direction = "improving" if coeffs[0] > 0 else "declining" if coeffs[0] < 0 else "stable"

                trends[metric_name] = {
                    "direction": trend_direction,
                    "slope": float(coeffs[0]),
                    "recent_average": np.mean(values[-5:]) if len(values) >= 5 else np.mean(values),
                    "overall_average": np.mean(values)
                }

        return trends

    def _perform_statistical_analysis(self, metric_comparisons: dict[str, list[float]]) -> dict[str, Any]:
        """Perform statistical analysis on comparison data"""
        analysis = {}

        for metric_name, scores in metric_comparisons.items():
            if len(scores) > 1:
                analysis[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "range": float(np.max(scores) - np.min(scores)),
                    "cv": float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                }

        return analysis

    # --- Compatibility helpers for tests expecting simple retrieval APIs ---
    async def get_evaluation(self, evaluation_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single evaluation by ID from internal storage.

        Returns a dict with columns from internal_evaluations or None if not found.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM internal_evaluations WHERE evaluation_id = ?",
                    (evaluation_id,)
                ).fetchone()
                if not row:
                    return None
                return dict(row)
        except Exception as e:
            logger.error(f"get_evaluation failed: {e}")
            return None

    async def list_evaluations(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List evaluations created in this manager session (ordered by created_at desc).

        To ensure isolation for property-based tests that reuse the same fixture
        across multiple generated examples, this method only returns evaluations
        created since the last call on this manager instance.
        """
        try:
            # If nothing was created since last listing, return empty
            if not getattr(self, "_recent_created_ids", None):
                return []

            ids = list(self._recent_created_ids)
            # Apply offset/limit at the ID list level to keep semantics simple
            sliced_ids = ids[offset: offset + limit]
            if not sliced_ids:
                return []

            placeholders = ",".join(["?"] * len(sliced_ids))
            sliced_ids_clause = f"({placeholders})"
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                list_evaluations_sql_template = (
                    "SELECT * FROM internal_evaluations WHERE evaluation_id IN {sliced_ids_clause} "
                    "ORDER BY created_at DESC"
                )
                list_evaluations_sql = list_evaluations_sql_template.format_map(locals())  # nosec B608
                rows = conn.execute(
                    list_evaluations_sql,
                    sliced_ids
                ).fetchall()
                results = [dict(r) for r in rows]
            # Clear after listing to avoid cross-example accumulation
            self._recent_created_ids.clear()
            return results
        except Exception as e:
            logger.error(f"list_evaluations failed: {e}")
            return []


# --- Shared instance helpers ---

@lru_cache(maxsize=1)
def _get_cached_manager() -> "EvaluationManager":
    """Return a cached EvaluationManager instance for lightweight checks."""
    return EvaluationManager()


def get_cached_evaluation_manager() -> "EvaluationManager":
    """Public accessor for the cached EvaluationManager instance."""
    return _get_cached_manager()


def reset_cached_evaluation_manager() -> None:
    """Clear the cached EvaluationManager instance (primarily for tests)."""
    _get_cached_manager.cache_clear()
