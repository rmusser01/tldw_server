# optimization_engine.py
# MIPRO-style optimization engine for Prompt Studio

import json
import random
import uuid
from enum import Enum
from typing import Any, Optional

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Logging.log_context import log_context

from .evaluation_metrics import EvaluationMetrics
from .mcts_optimizer import MCTSOptimizer
from .prompt_executor import PromptExecutor
from .test_runner import TestRunner
from .types_common import MetricType

########################################################################################################################
# Metric Types
# Moved to types_common.py to avoid circular imports. Re-imported here for API stability.

########################################################################################################################
# Optimization Strategies

class OptimizationStrategy(str, Enum):
    """Optimization strategies available."""

    MIPRO = "mipro"  # Multi-Instruction Prompt Optimization
    BOOTSTRAP = "bootstrap"  # Bootstrap few-shot examples
    ITERATIVE = "iterative"  # Iterative refinement
    GENETIC = "genetic"  # Genetic algorithm
    BAYESIAN = "bayesian"  # Bayesian optimization
    GRID_SEARCH = "grid_search"  # Grid search
    RANDOM_SEARCH = "random_search"  # Random search


def _next_variant_version(cursor, project_id: int, variant_name: str, base_version: Any) -> int:
    cursor.execute(
        """
            SELECT COALESCE(MAX(version_number), 0)
            FROM prompt_studio_prompts
            WHERE project_id = ? AND name = ?
        """,
        (project_id, variant_name),
    )
    row = cursor.fetchone()
    existing_max = int(row[0] or 0) if row else 0
    return max(existing_max, int(base_version or 0)) + 1

########################################################################################################################
# MIPRO Optimizer

class MIPROOptimizer:
    """
    Multi-Instruction Prompt Optimization (MIPRO) implementation.
    Based on DSPy's MIPRO approach for optimizing prompt instructions.
    """

    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """
        Initialize MIPRO optimizer.

        Args:
            db: Database instance
            test_runner: Test runner for evaluations
        """
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        self.metrics = EvaluationMetrics()
        # Optional context populated by callers (e.g., JobProcessor)
        self.optimization_id: Optional[int] = None
        # Model config for internal LLM calls (set during optimize())
        self._internal_model_config: dict[str, Any] = {}

    async def optimize(self, initial_prompt_id: int, test_case_ids: list[int],
                       model_config: dict[str, Any], max_iterations: int = 20,
                       target_metric: MetricType = MetricType.ACCURACY,
                       min_improvement: float = 0.01) -> dict[str, Any]:
        """
        Optimize a prompt using MIPRO strategy.

        Args:
            initial_prompt_id: Starting prompt ID
            test_case_ids: Test cases for evaluation
            model_config: Model configuration
            max_iterations: Maximum optimization iterations
            target_metric: Metric to optimize
            min_improvement: Minimum improvement to continue

        Returns:
            Optimization results
        """
        with log_context(ps_component="opt_engine", strategy="mipro", prompt_id=initial_prompt_id, optimization_id=getattr(self, "optimization_id", None)):
            logger.info("Starting MIPRO optimization for prompt {}", initial_prompt_id)

        # Store model_config for use in internal LLM calls (instruction generation)
        self._internal_model_config = model_config

        # Initialize
        current_prompt_id = initial_prompt_id
        best_prompt_id = initial_prompt_id
        best_score = await self._evaluate_prompt(
            initial_prompt_id, test_case_ids, model_config, target_metric
        )

        iteration_history = []
        no_improvement_count = 0

        for iteration in range(max_iterations):
            logger.info("MIPRO iteration {}/{}", iteration + 1, max_iterations)

            # Generate instruction candidates
            candidates = await self._generate_instruction_candidates(
                current_prompt_id, best_score, iteration
            )

            # Evaluate candidates
            candidate_scores = []
            for candidate in candidates:
                # Create new prompt with candidate instruction
                new_prompt_id = await self._create_prompt_variant(
                    current_prompt_id, candidate
                )

                # Evaluate
                score = await self._evaluate_prompt(
                    new_prompt_id, test_case_ids, model_config, target_metric
                )

                candidate_scores.append((new_prompt_id, score, candidate))

            # Select best candidate
            if candidate_scores:
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                new_prompt_id, new_score, best_candidate = candidate_scores[0]

                # Track iteration
                iteration_data = {
                    "iteration": iteration + 1,
                    "prompt_id": new_prompt_id,
                    "score": new_score,
                    "improvement": new_score - best_score,
                    "instruction": best_candidate
                }
                iteration_history.append(iteration_data)

                # Check for improvement
                if new_score > best_score + min_improvement:
                    logger.info(f"Improvement found: {best_score:.3f} -> {new_score:.3f}")
                    best_score = new_score
                    best_prompt_id = new_prompt_id
                    current_prompt_id = new_prompt_id
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                    # Early stopping
                    if no_improvement_count >= 3:
                        logger.info("Early stopping: No improvement for 3 iterations")
                        break

            # Check convergence
            if await self._check_convergence(iteration_history):
                logger.info("Convergence detected")
                break

        # Return optimization results
        return {
            "initial_prompt_id": initial_prompt_id,
            "optimized_prompt_id": best_prompt_id,
            "initial_score": iteration_history[0]["score"] if iteration_history else best_score,
            "final_score": best_score,
            "improvement": best_score - (iteration_history[0]["score"] if iteration_history else best_score),
            "iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "strategy": "MIPRO"
        }

    async def _generate_instruction_candidates(self, prompt_id: int,
                                              current_score: float,
                                              iteration: int) -> list[str]:
        """
        Generate instruction candidates for MIPRO.

        Args:
            prompt_id: Current prompt ID
            current_score: Current best score
            iteration: Current iteration number

        Returns:
            List of instruction candidates
        """
        # Get current prompt
        prompt = self._get_prompt(prompt_id)
        current_instruction = prompt.get("system_prompt", "")

        candidates = []

        # Strategy 1: Rephrase current instruction
        rephrased = await self._rephrase_instruction(current_instruction)
        if rephrased:
            candidates.append(rephrased)

        # Strategy 2: Add clarifying details
        detailed = await self._add_details(current_instruction, current_score)
        if detailed:
            candidates.append(detailed)

        # Strategy 3: Simplify instruction
        if iteration > 5:  # Try simplification after some iterations
            simplified = await self._simplify_instruction(current_instruction)
            if simplified:
                candidates.append(simplified)

        # Strategy 4: Add examples
        with_examples = await self._add_examples(current_instruction)
        if with_examples:
            candidates.append(with_examples)

        # Strategy 5: Combine successful patterns
        if iteration > 10:
            combined = await self._combine_patterns(current_instruction)
            if combined:
                candidates.append(combined)

        return candidates[:5]  # Limit to 5 candidates per iteration

    async def _rephrase_instruction(self, instruction: str) -> Optional[str]:
        """Rephrase instruction using LLM."""
        prompt = f"""Rephrase the following instruction to be clearer and more effective.
Keep the same intent but improve clarity and specificity.

Original instruction:
{instruction}

Rephrased instruction:"""

        try:
            # Use configured provider/model, falling back to defaults
            provider = self._internal_model_config.get("provider", "openai")
            model = self._internal_model_config.get("model", "gpt-3.5-turbo")
            result = await self.executor._call_llm(
                provider=provider,
                model=model,
                prompt=prompt,
                parameters={"temperature": 0.7, "max_tokens": 500}
            )
            return result["content"].strip()
        except Exception as e:
            logger.warning(f"_rephrase_instruction failed to call LLM: error={e}")
            return None

    async def _add_details(self, instruction: str, current_score: float) -> Optional[str]:
        """Add clarifying details to instruction."""
        prompt = f"""The following instruction achieves {current_score:.1%} accuracy.
Add specific details and constraints to improve its effectiveness.

Current instruction:
{instruction}

Enhanced instruction with more details:"""

        try:
            # Use configured provider/model, falling back to defaults
            provider = self._internal_model_config.get("provider", "openai")
            model = self._internal_model_config.get("model", "gpt-3.5-turbo")
            result = await self.executor._call_llm(
                provider=provider,
                model=model,
                prompt=prompt,
                parameters={"temperature": 0.8, "max_tokens": 500}
            )
            return result["content"].strip()
        except Exception as e:
            logger.warning(f"_add_details failed to call LLM: error={e}")
            return None

    async def _simplify_instruction(self, instruction: str) -> Optional[str]:
        """Simplify instruction by removing unnecessary complexity."""
        prompt = f"""Simplify the following instruction while keeping its core requirements.
Remove unnecessary words and complexity.

Complex instruction:
{instruction}

Simplified instruction:"""

        try:
            # Use configured provider/model, falling back to defaults
            provider = self._internal_model_config.get("provider", "openai")
            model = self._internal_model_config.get("model", "gpt-3.5-turbo")
            result = await self.executor._call_llm(
                provider=provider,
                model=model,
                prompt=prompt,
                parameters={"temperature": 0.5, "max_tokens": 300}
            )
            return result["content"].strip()
        except Exception as e:
            logger.warning(f"_simplify_instruction failed to call LLM: error={e}")
            return None

    async def _add_examples(self, instruction: str) -> Optional[str]:
        """Add few-shot examples to instruction."""
        # This would ideally use actual successful examples from test runs
        enhanced = f"""{instruction}

Here are some examples of the expected format:
- Example 1: [Input] -> [Expected Output]
- Example 2: [Input] -> [Expected Output]

Follow these examples for consistency."""

        return enhanced

    async def _combine_patterns(self, instruction: str) -> Optional[str]:
        """Combine successful patterns from history."""
        # This would analyze successful prompts and combine patterns
        # For now, return a variation
        return f"""Follow these guidelines:
1. {instruction}
2. Be precise and consistent
3. Validate your output before responding"""

    async def _evaluate_prompt(self, prompt_id: int, test_case_ids: list[int],
                              model_config: dict[str, Any],
                              target_metric: MetricType) -> float:
        """Evaluate a prompt and return target metric score."""
        scores = []

        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config,
                metrics=[target_metric]
            )

            if result.get("success") and "scores" in result:
                score = result["scores"].get(target_metric.value, 0)
                scores.append(score)

        return np.mean(scores) if scores else 0.0

    async def _create_prompt_variant(self, base_prompt_id: int,
                                    new_instruction: str) -> int:
        """Create a new prompt variant with updated instruction."""
        # Get base prompt
        prompt = self._get_prompt(base_prompt_id)

        # Create new prompt variant: update system_prompt (instructions), preserve user_prompt
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            variant_name = f"{prompt['name']} (Optimized)"
            version_number = _next_variant_version(
                cursor,
                int(prompt["project_id"]),
                variant_name,
                prompt.get("version_number"),
            )

            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name, system_prompt,
                    user_prompt, version_number, parent_version_id, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"opt-{uuid.uuid4()}",
                prompt["project_id"],
                prompt.get("signature_id"),
                variant_name,
                new_instruction,
                prompt.get("user_prompt"),
                version_number,
                base_prompt_id,
                self.db.client_id
            ))

            new_prompt_id = cursor.lastrowid

        return new_prompt_id

    async def _check_convergence(self, history: list[dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        if len(history) < 5:
            return False

        # Check if last 5 scores are very similar
        recent_scores = [h["score"] for h in history[-5:]]
        std_dev = np.std(recent_scores)

        return std_dev < 0.001

    def _get_prompt(self, prompt_id: int) -> dict[str, Any]:
        """Get prompt from database."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
            row = cursor.fetchone()

            if row:
                return self.db._row_to_dict(cursor, row)
        return {}

########################################################################################################################
# Bootstrap Optimizer

class BootstrapOptimizer:
    """
    Bootstrap optimization for few-shot learning.
    Automatically selects best examples from test runs.
    """

    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize Bootstrap optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        self.optimization_id: Optional[int] = None

    async def optimize(self, prompt_id: int, test_case_ids: list[int],
                       model_config: dict[str, Any],
                       num_examples: int = 3,
                       selection_strategy: str = "diverse") -> dict[str, Any]:
        """
        Optimize prompt by bootstrapping few-shot examples.

        Args:
            prompt_id: Base prompt ID
            test_case_ids: Test cases to use
            model_config: Model configuration
            num_examples: Number of examples to include
            selection_strategy: How to select examples (best, diverse, random)

        Returns:
            Optimization results
        """
        with log_context(ps_component="opt_engine", strategy="bootstrap", prompt_id=prompt_id, optimization_id=getattr(self, "optimization_id", None)):
            logger.info("Starting Bootstrap optimization for prompt {}", prompt_id)

        # Run initial evaluation to get examples
        test_runs = []
        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )
            test_runs.append(result)

        # Select best examples
        examples = self._select_examples(test_runs, num_examples, selection_strategy)

        # Create new prompt with examples
        new_prompt_id = await self._create_prompt_with_examples(prompt_id, examples)

        # Evaluate new prompt
        new_scores = []
        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=new_prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )

            if result.get("success") and "scores" in result:
                new_scores.append(result["scores"].get("aggregate_score", 0))

        # Calculate improvement
        original_scores = [
            run.get("scores", {}).get("aggregate_score", 0)
            for run in test_runs if run.get("success")
        ]

        original_mean = np.mean(original_scores) if original_scores else 0
        new_mean = np.mean(new_scores) if new_scores else 0

        return {
            "initial_prompt_id": prompt_id,
            "optimized_prompt_id": new_prompt_id,
            "initial_score": original_mean,
            "final_score": new_mean,
            "improvement": new_mean - original_mean,
            "num_examples": len(examples),
            "strategy": "Bootstrap"
        }

    def _select_examples(self, test_runs: list[dict[str, Any]],
                        num_examples: int,
                        strategy: str) -> list[dict[str, Any]]:
        """Select examples based on strategy."""
        # Filter successful runs with good scores
        good_runs = [
            run for run in test_runs
            if run.get("success") and
            run.get("scores", {}).get("aggregate_score", 0) > 0.8
        ]

        if not good_runs:
            good_runs = [run for run in test_runs if run.get("success")]

        if strategy == "best":
            # Select top scoring examples
            good_runs.sort(
                key=lambda x: x.get("scores", {}).get("aggregate_score", 0),
                reverse=True
            )
            return good_runs[:num_examples]

        elif strategy == "diverse":
            # Select diverse examples
            selected = []
            remaining = good_runs.copy()

            while len(selected) < num_examples and remaining:
                # Pick one at random
                if not selected:
                    # Non-security sampling for diverse few-shot example selection.
                    idx = random.randint(0, len(remaining) - 1)  # nosec B311
                    selected.append(remaining.pop(idx))
                else:
                    # Pick most different from selected
                    max_diff = -1
                    max_idx = 0

                    for i, candidate in enumerate(remaining):
                        diff = self._calculate_diversity(candidate, selected)
                        if diff > max_diff:
                            max_diff = diff
                            max_idx = i

                    selected.append(remaining.pop(max_idx))

            return selected

        else:  # random
            random.shuffle(good_runs)
            return good_runs[:num_examples]

    def _calculate_diversity(self, candidate: dict[str, Any],
                           selected: list[dict[str, Any]]) -> float:
        """Calculate diversity score for candidate."""
        # Simple diversity based on input/output differences
        diversity_scores = []

        for s in selected:
            # Compare inputs
            input_diff = 0
            for key in candidate.get("inputs", {}):
                if str(candidate["inputs"][key]) != str(s.get("inputs", {}).get(key)):
                    input_diff += 1

            # Compare outputs
            output_diff = 0
            for key in candidate.get("actual_output", {}):
                if str(candidate["actual_output"][key]) != str(s.get("actual_output", {}).get(key)):
                    output_diff += 1

            diversity_scores.append(input_diff + output_diff)

        return min(diversity_scores) if diversity_scores else float('inf')

    async def _create_prompt_with_examples(self, base_prompt_id: int,
                                          examples: list[dict[str, Any]]) -> int:
        """Create new prompt with few-shot examples."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()

            # Get base prompt
            cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (base_prompt_id,))
            row = cursor.fetchone()
            prompt = self.db._row_to_dict(cursor, row)

            # Build examples text
            examples_text = "Here are some examples:\n\n"
            for i, example in enumerate(examples, 1):
                examples_text += f"Example {i}:\n"
                examples_text += f"Input: {json.dumps(example.get('inputs', {}), indent=2)}\n"
                examples_text += f"Output: {json.dumps(example.get('actual_output', {}), indent=2)}\n\n"

            # Add examples to user prompt (preserve system_prompt)
            base_user_prompt = prompt.get("user_prompt") or ""
            new_user_prompt = f"{examples_text}{base_user_prompt}"

            # Create new prompt
            variant_name = f"{prompt['name']} (Bootstrap)"
            version_number = _next_variant_version(
                cursor,
                int(prompt["project_id"]),
                variant_name,
                prompt.get("version_number"),
            )
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name, system_prompt,
                    user_prompt, version_number, parent_version_id, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"bootstrap-{uuid.uuid4()}",
                prompt["project_id"],
                prompt.get("signature_id"),
                variant_name,
                prompt.get("system_prompt"),
                new_user_prompt,
                version_number,
                base_prompt_id,
                self.db.client_id
            ))

            new_prompt_id = cursor.lastrowid

        return new_prompt_id

########################################################################################################################
# Main Optimization Engine

class OptimizationEngine:
    """Main optimization engine coordinating different strategies."""

    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize OptimizationEngine.

        Args:
            db: Database instance
        """
        self.db = db
        self.test_runner = TestRunner(db)

        # Initialize optimizers
        self.mipro = MIPROOptimizer(db, self.test_runner)
        self.bootstrap = BootstrapOptimizer(db, self.test_runner)
        self.mcts = MCTSOptimizer(db, self.test_runner)

    @staticmethod
    def _coerce_json_value(value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                value = bytes(value).decode("utf-8")
            except Exception:
                return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return default
        return default

    async def optimize(self, optimization_id: int) -> dict[str, Any]:
        """
        Run optimization based on configuration.

        Args:
            optimization_id: Optimization ID from database

        Returns:
            Optimization results
        """
        # Get optimization config
        optimization = self._get_optimization(optimization_id)
        if not optimization:
            raise ValueError(f"Optimization {optimization_id} not found")

        # Parse configuration
        config = self._coerce_json_value(
            optimization.get("optimization_config") or optimization.get("optimizer_config"),
            {},
        )
        if str(optimization.get("status") or "").lower() == "cancelled":
            return {
                "optimization_id": optimization_id,
                "optimized_prompt_id": optimization.get("optimized_prompt_id")
                or optimization.get("initial_prompt_id"),
                "iterations": int(optimization.get("iterations_completed") or 0),
                "status": "cancelled",
            }
        # Support both legacy "strategy" and new "optimizer_type" fields
        strategy = (
            config.get("strategy") or optimization.get("optimizer_type") or config.get("optimizer_type") or "mipro"
        )
        strategy = str(strategy).lower()

        test_case_ids = self._coerce_json_value(
            optimization.get("test_case_ids")
            if optimization.get("test_case_ids") is not None
            else config.get("test_case_ids"),
            [],
        )
        if not isinstance(test_case_ids, list):
            test_case_ids = []

        model_config = self._coerce_json_value(
            optimization.get("model_config")
            or config.get("model_config")
            or config.get("model_configuration"),
            {},
        )
        if isinstance(model_config, list):
            model_config = model_config[0] if model_config else {}
        if not isinstance(model_config, dict):
            model_config = {}

        # Update status
        self._update_optimization_status(optimization_id, "running")

        try:
            # Run optimization based on strategy
            if strategy == "mipro":
                results = await self.mipro.optimize(
                    initial_prompt_id=optimization["initial_prompt_id"],
                    test_case_ids=test_case_ids,
                    model_config=model_config,
                    max_iterations=optimization["max_iterations"],
                    target_metric=MetricType(config.get("target_metric", "accuracy"))
                )

            elif strategy == "bootstrap":
                results = await self.bootstrap.optimize(
                    prompt_id=optimization["initial_prompt_id"],
                    test_case_ids=test_case_ids,
                    model_config=model_config,
                    num_examples=config.get("num_examples", 3),
                    selection_strategy=config.get("selection_strategy", "diverse")
                )

            elif strategy == "mcts":
                results = await self.mcts.optimize(
                    initial_prompt_id=optimization["initial_prompt_id"],
                    optimization_id=optimization_id,
                    test_case_ids=test_case_ids,
                    model_config=model_config,
                    # Treat max_iterations as fallback simulations; allow override via strategy_params
                    max_iterations=optimization.get("max_iterations", 20),
                    target_metric=MetricType(config.get("target_metric", "accuracy")),
                    strategy_params=config.get("strategy_params", {}),
                )

            else:
                raise ValueError(f"Unknown optimization strategy: {strategy}")

            latest = self._get_optimization(optimization_id) or {}
            if str(latest.get("status") or "").lower() == "cancelled":
                self._update_cancelled_optimization_results(optimization_id, results)
            else:
                # Update optimization with results
                self._update_optimization_results(optimization_id, results)

            return results

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self._update_optimization_status(optimization_id, "failed", str(e))
            raise

    def _get_optimization(self, optimization_id: int) -> Optional[dict[str, Any]]:
        """Get optimization from database."""
        return self.db.get_optimization(optimization_id)

    def _update_optimization_status(self, optimization_id: int, status: str,
                                   error_message: Optional[str] = None):
        """Update optimization status."""
        mark_started = status == "running"
        mark_completed = status in {"failed", "cancelled"}
        self.db.set_optimization_status(
            optimization_id,
            status,
            error_message=error_message,
            mark_started=mark_started,
            mark_completed=mark_completed,
        )

    @staticmethod
    def _build_final_metrics(results: dict[str, Any]) -> dict[str, Any]:
        final_metrics: dict[str, Any] = {"score": results.get("final_score", 0)}
        if isinstance(results.get("final_metrics"), dict):
            try:
                # Do not overwrite score if present in extra metrics without intent
                extra = dict(results.get("final_metrics") or {})
                if "score" in extra and extra["score"] is None:
                    extra.pop("score")
                final_metrics.update(extra)
            except Exception as metric_merge_error:
                logger.debug("Optimization engine failed to merge extra metric payload", exc_info=metric_merge_error)
        return final_metrics

    def _update_optimization_results(self, optimization_id: int, results: dict[str, Any]):
        """Update optimization with results."""
        final_metrics = self._build_final_metrics(results)

        self.db.complete_optimization(
            optimization_id,
            optimized_prompt_id=results.get("optimized_prompt_id"),
            iterations_completed=results.get("iterations", 1),
            initial_metrics={"score": results.get("initial_score", 0)},
            final_metrics=final_metrics,
            improvement_percentage=results.get("improvement", 0) * 100,
            total_tokens=results.get("total_tokens"),
            total_cost=results.get("total_cost"),
        )

    def _update_cancelled_optimization_results(self, optimization_id: int, results: dict[str, Any]) -> None:
        """Persist partial metrics for a cancelled optimization without reviving it."""
        updates: dict[str, Any] = {
            "optimized_prompt_id": results.get("optimized_prompt_id"),
            "iterations_completed": results.get("iterations", 0),
            "initial_metrics": {"score": results.get("initial_score", 0)},
            "final_metrics": self._build_final_metrics(results),
            "improvement_percentage": results.get("improvement", 0) * 100,
            "total_tokens": results.get("total_tokens"),
            "total_cost": results.get("total_cost"),
        }
        updates = {k: v for k, v in updates.items() if v is not None}
        self.db.update_optimization(
            optimization_id,
            updates,
            set_completed_at=True,
        )
