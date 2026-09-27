# optimization_strategies.py
# Additional optimization strategies for Prompt Studio

import contextlib
import random
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Optional

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Logging.log_context import log_context

from .prompt_executor import PromptExecutor
from .test_runner import TestRunner

########################################################################################################################
# Hyperparameter Optimizer

class HyperparameterOptimizer:
    """
    Optimize model hyperparameters (temperature, max_tokens, etc.).
    Uses Bayesian optimization approach.
    """

    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize hyperparameter optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        self.optimization_id: Optional[int] = None

        # Define parameter search spaces
        self.param_spaces = {
            "temperature": (0.0, 2.0),
            "max_tokens": (50, 2000),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (0.0, 2.0),
            "presence_penalty": (0.0, 2.0)
        }

    async def optimize(
        self,
        prompt_id: int,
        test_case_ids: list[int],
        base_model_config: dict[str, Any],
        max_iterations: int = 20,
        params_to_optimize: Optional[list[str]] = None,
        optimization_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.

        Args:
            prompt_id: Prompt to optimize
            test_case_ids: Test cases for evaluation
            base_model_config: Base model configuration
            max_iterations: Maximum iterations
            params_to_optimize: Specific parameters to optimize

        Returns:
            Optimization results with best parameters
        """
        # Populate optional context for logging/observability
        with contextlib.suppress(Exception):
            self.optimization_id = optimization_id
        with log_context(ps_component="opt_strategies", strategy="hyperparams", prompt_id=prompt_id, optimization_id=getattr(self, "optimization_id", None)):
            logger.info("Starting hyperparameter optimization for prompt {}", prompt_id)

            if params_to_optimize is None:
                params_to_optimize = ["temperature", "max_tokens", "top_p"]

            # Initialize with random samples
            observations = []

            # Random exploration phase
            for i in range(min(5, max_iterations)):
                params = self._sample_random_params(params_to_optimize)
                score = await self._evaluate_params(
                    prompt_id, test_case_ids, base_model_config, params
                )
                observations.append((params, score))
                logger.info("Random sample {}: score={:.3f}, params={}", i + 1, score, params)

            # Bayesian optimization phase
            best_params = observations[0][0]
            best_score = observations[0][1]

            for i in range(5, max_iterations):
                # Use Gaussian Process to predict next point
                next_params = self._get_next_params(observations, params_to_optimize)

                # Evaluate
                score = await self._evaluate_params(
                    prompt_id, test_case_ids, base_model_config, next_params
                )

                observations.append((next_params, score))
                logger.info("Iteration {}: score={:.3f}, params={}", i + 1, score, next_params)

                # Update best
                if score > best_score:
                    best_score = score
                    best_params = next_params
                    logger.info("New best score: {:.3f}", best_score)

                # Early stopping if converged
                if self._has_converged(observations):
                    logger.info("Convergence detected")
                    break

            return {
                "best_params": best_params,
                "best_score": best_score,
                "iterations": len(observations),
                "all_observations": observations,
                "improvement": best_score - observations[0][1]
            }

    def _sample_random_params(self, params_to_optimize: list[str]) -> dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}

        for param in params_to_optimize:
            if param in self.param_spaces:
                min_val, max_val = self.param_spaces[param]

                if param in ["max_tokens"]:
                    # Integer parameters
                    params[param] = random.randint(int(min_val), int(max_val))
                else:
                    # Float parameters
                    params[param] = random.uniform(min_val, max_val)

        return params

    def _get_next_params(self, observations: list[tuple[dict, float]],
                        params_to_optimize: list[str]) -> dict[str, Any]:
        """
        Use acquisition function to get next parameters to try.
        Simplified Expected Improvement approach.
        """
        if len(observations) < 10:
            # Not enough data, sample randomly
            return self._sample_random_params(params_to_optimize)

        # Extract parameter values and scores
        X = []
        y = []

        for params, score in observations:
            x = [params.get(p, 0) for p in params_to_optimize]
            X.append(x)
            y.append(score)

        X = np.array(X)
        y = np.array(y)

        # Simple acquisition: explore areas with high uncertainty
        # Sample candidates
        candidates = []
        for _ in range(100):
            candidate = self._sample_random_params(params_to_optimize)

            # Calculate distance to observed points
            x_cand = [candidate.get(p, 0) for p in params_to_optimize]
            distances = np.linalg.norm(X - x_cand, axis=1)
            min_distance = np.min(distances)

            # Prefer points far from observations (exploration)
            exploration_score = min_distance

            # Also consider exploitation (near good points)
            best_idx = np.argmax(y)
            exploitation_score = -distances[best_idx]

            # Balance exploration and exploitation
            acquisition_score = 0.5 * exploration_score + 0.5 * exploitation_score

            candidates.append((candidate, acquisition_score))

        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def _evaluate_params(self, prompt_id: int, test_case_ids: list[int],
                              base_model_config: dict[str, Any],
                              params: dict[str, Any]) -> float:
        """Evaluate parameters on test cases."""
        # Merge parameters
        model_config = base_model_config.copy()
        model_config["parameters"] = params

        # Run evaluation
        scores = []
        for test_case_id in test_case_ids[:5]:  # Limit for speed
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )

            if result.get("success") and "scores" in result:
                scores.append(result["scores"].get("aggregate_score", 0))

        return np.mean(scores) if scores else 0.0

    def _has_converged(self, observations: list[tuple[dict, float]]) -> bool:
        """Check if optimization has converged."""
        if len(observations) < 10:
            return False

        # Check if recent scores are stable
        recent_scores = [score for _, score in observations[-5:]]
        return np.std(recent_scores) < 0.01

########################################################################################################################
# Iterative Refinement Optimizer

class IterativeRefinementOptimizer:
    """
    Iteratively refine prompts based on error analysis.
    """

    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize iterative refinement optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        self.optimization_id: Optional[int] = None

    async def optimize(
        self,
        prompt_id: int,
        test_case_ids: list[int],
        model_config: dict[str, Any],
        max_iterations: int = 10,
        optimization_id: Optional[int] = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Iteratively refine prompt based on errors.

        Args:
            prompt_id: Initial prompt ID
            test_case_ids: Test cases
            model_config: Model configuration
            max_iterations: Maximum refinement iterations

        Returns:
            Optimization results
        """
        # Populate optional context for logging/observability
        with contextlib.suppress(Exception):
            self.optimization_id = optimization_id
        with log_context(ps_component="opt_strategies", strategy="iterative", prompt_id=prompt_id, optimization_id=getattr(self, "optimization_id", None)):
            logger.info("Starting iterative refinement for prompt {}", prompt_id)

            current_prompt_id = prompt_id
            iteration_history = []

            for iteration in range(max_iterations):
                logger.info(f"Refinement iteration {iteration + 1}")

                # Run evaluation
                test_runs = []
                for test_case_id in test_case_ids:
                    runner_kwargs = {
                        "prompt_id": current_prompt_id,
                        "test_case_id": test_case_id,
                        "model_config": model_config,
                    }
                    if provider_credentials is not None:
                        runner_kwargs["provider_credentials"] = provider_credentials
                    if on_provider_success is not None:
                        runner_kwargs.update(
                            strict_provider_errors=True,
                            on_provider_success=on_provider_success,
                        )
                    result = await self.test_runner.run_single_test(**runner_kwargs)
                    test_runs.append(result)

                # Analyze errors
                errors = self._analyze_errors(test_runs)

                if not errors:
                    logger.info("No errors found, optimization complete")
                    break

                # Generate refinement based on errors
                refinement = await self._generate_refinement(
                    current_prompt_id,
                    errors,
                    model_config=model_config,
                    provider_credentials=provider_credentials,
                    on_provider_success=on_provider_success,
                )

                if not refinement:
                    logger.warning("Could not generate refinement")
                    break

                # Create refined prompt
                new_prompt_id = await self._create_refined_prompt(
                    current_prompt_id, refinement
                )

                # Evaluate refined prompt
                new_score = await self._evaluate_prompt(
                    new_prompt_id,
                    test_case_ids,
                    model_config,
                    provider_credentials=provider_credentials,
                    on_provider_success=on_provider_success,
                )

                iteration_history.append({
                    "iteration": iteration + 1,
                    "prompt_id": new_prompt_id,
                    "score": new_score,
                    "errors_addressed": len(errors),
                    "refinement": refinement
                })

                current_prompt_id = new_prompt_id

                # Check if errors are decreasing
                if len(errors) <= 2:
                    logger.info("Few errors remaining, stopping refinement")
                    break

            # Calculate final improvement
            initial_score = await self._evaluate_prompt(
                prompt_id,
                test_case_ids,
                model_config,
                provider_credentials=provider_credentials,
                on_provider_success=on_provider_success,
            )
            final_score = iteration_history[-1]["score"] if iteration_history else initial_score

            return {
                "initial_prompt_id": prompt_id,
                "optimized_prompt_id": current_prompt_id,
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": final_score - initial_score,
                "iterations": len(iteration_history),
                "iteration_history": iteration_history
            }

    def _analyze_errors(self, test_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze test runs to identify error patterns."""
        errors = []

        for run in test_runs:
            if not run.get("success") or run.get("scores", {}).get("aggregate_score", 0) < 0.8:
                error_info = {
                    "test_case": run.get("test_case_name"),
                    "input": run.get("inputs"),
                    "expected": run.get("expected_outputs"),
                    "actual": run.get("actual_output"),
                    "score": run.get("scores", {}).get("aggregate_score", 0),
                    "error": run.get("error")
                }
                errors.append(error_info)

        return errors

    async def _generate_refinement(
        self,
        prompt_id: int,
        errors: list[dict[str, Any]],
        *,
        model_config: dict[str, Any],
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
    ) -> Optional[str]:
        """Generate refinement based on error analysis."""
        # Get current prompt
        prompt = self._get_prompt(prompt_id)

        # Build error summary
        error_summary = "Common errors found:\n"
        for i, error in enumerate(errors[:5], 1):  # Limit to 5 errors
            error_summary += f"{i}. Input: {error['input']}\n"
            error_summary += f"   Expected: {error['expected']}\n"
            error_summary += f"   Got: {error['actual']}\n\n"

        # Generate refinement
        refinement_prompt = f"""Analyze these errors and suggest how to refine the prompt to address them.

Current prompt (user section):
{prompt.get('user_prompt', '')}

{error_summary}

Suggest specific refinements to fix these errors:"""

        try:
            parameters = dict(model_config.get("parameters") or {})
            parameters.update({"temperature": 0.7, "max_tokens": 500})
            result = await self.executor._call_llm(
                provider=str(model_config.get("provider") or "openai"),
                model=str(model_config.get("model") or "gpt-3.5-turbo"),
                prompt=refinement_prompt,
                parameters=parameters,
                api_key_override=model_config.get("api_key"),
                app_config=model_config.get("app_config"),
                credentials_resolved=(
                    model_config.get("credentials_resolved") is True
                ),
                provider_credentials=provider_credentials,
                timeout_seconds=parameters.get("timeout_seconds"),
                on_provider_success=on_provider_success,
            )
            return result["content"].strip()
        except Exception as e:
            if on_provider_success is not None:
                raise
            logger.debug(
                "_generate_refinement failed to call LLM; error_type={}",
                type(e).__name__,
            )
            return None

    async def _create_refined_prompt(self, base_prompt_id: int,
                                    refinement: str) -> int:
        """Create refined version of prompt."""
        # Get base prompt
        prompt = self._get_prompt(base_prompt_id)

        # Apply refinement: augment system instructions; preserve user_prompt
        base_system = prompt.get("system_prompt") or ""
        new_system = (base_system + ("\n\n" if base_system and refinement else "") + (refinement or "")).strip()

        # Create new prompt
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, name, system_prompt,
                user_prompt, version_number, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"refined-{datetime.utcnow().timestamp()}",
            prompt["project_id"],
            prompt.get("signature_id"),
            f"{prompt['name']} (Refined)",
            new_system,
            prompt.get("user_prompt"),
            (prompt.get("version_number") or 0) + 1,
            base_prompt_id,
            self.db.client_id
        ))

        new_prompt_id = cursor.lastrowid
        conn.commit()

        return new_prompt_id

    async def _evaluate_prompt(self, prompt_id: int, test_case_ids: list[int],
                              model_config: dict[str, Any],
                              *,
                              provider_credentials: ProviderCallCredentials | None = None,
                              on_provider_success: Callable[[], Awaitable[None]] | None = None,
                              ) -> float:
        """Evaluate prompt and return average score."""
        scores = []

        for test_case_id in test_case_ids:
            runner_kwargs = {
                "prompt_id": prompt_id,
                "test_case_id": test_case_id,
                "model_config": model_config,
            }
            if provider_credentials is not None:
                runner_kwargs["provider_credentials"] = provider_credentials
            if on_provider_success is not None:
                runner_kwargs.update(
                    strict_provider_errors=True,
                    on_provider_success=on_provider_success,
                )
            result = await self.test_runner.run_single_test(**runner_kwargs)

            if result.get("success") and "scores" in result:
                scores.append(result["scores"].get("aggregate_score", 0))

        if not scores and on_provider_success is not None:
            raise ValueError("Optimization requires one validated baseline result")
        return np.mean(scores) if scores else 0.0

    def _get_prompt(self, prompt_id: int) -> dict[str, Any]:
        """Get prompt from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
        row = cursor.fetchone()

        if row:
            return self.db._row_to_dict(cursor, row)
        return {}

########################################################################################################################
# Genetic Algorithm Optimizer

class GeneticOptimizer:
    """
    Use genetic algorithm to evolve prompts.
    """

    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize genetic optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)

    async def optimize(self, prompt_id: int, test_case_ids: list[int],
                       model_config: dict[str, Any],
                       population_size: int = 10,
                       generations: int = 10,
                       mutation_rate: float = 0.1) -> dict[str, Any]:
        """
        Optimize prompt using genetic algorithm.

        Args:
            prompt_id: Initial prompt ID
            test_case_ids: Test cases
            model_config: Model configuration
            population_size: Population size
            generations: Number of generations
            mutation_rate: Mutation probability

        Returns:
            Optimization results
        """
        logger.info(f"Starting genetic optimization for prompt {prompt_id}")

        # Initialize population
        population = await self._initialize_population(prompt_id, population_size)

        best_individual = None
        best_score = -1
        generation_history = []

        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")

            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = await self._evaluate_fitness(
                    individual, test_case_ids, model_config, prompt_id
                )
                fitness_scores.append((individual, score))

            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Track best
            if fitness_scores[0][1] > best_score:
                best_score = fitness_scores[0][1]
                best_individual = fitness_scores[0][0]

            generation_history.append({
                "generation": generation + 1,
                "best_score": fitness_scores[0][1],
                "avg_score": np.mean([s for _, s in fitness_scores]),
                "population_size": len(population)
            })

            # Check convergence
            if generation > 5 and self._has_converged_genetic(generation_history):
                logger.info("Population converged")
                break

            # Selection and reproduction
            new_population = []

            # Elitism: keep top 2
            new_population.extend([ind for ind, _ in fitness_scores[:2]])

            # Crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)

                # Crossover
                child = await self._crossover(parent1, parent2)

                # Mutation
                if random.random() < mutation_rate:
                    child = await self._mutate(child)

                new_population.append(child)

            population = new_population

        # Create final prompt from best individual
        final_prompt_id = await self._individual_to_prompt(best_individual, prompt_id)

        return {
            "initial_prompt_id": prompt_id,
            "optimized_prompt_id": final_prompt_id,
            "best_score": best_score,
            "generations": len(generation_history),
            "generation_history": generation_history
        }

    async def _initialize_population(self, prompt_id: int,
                                    size: int) -> list[dict[str, Any]]:
        """Initialize population with variations of base prompt."""
        prompt = self._get_prompt(prompt_id)
        population = []

        # Add original (use user_prompt field)
        population.append({
            "user_prompt": prompt.get("user_prompt", ""),
            "system_prompt": prompt.get("system_prompt", "")
        })

        # Generate variations
        for _ in range(size - 1):
            variation = await self._generate_variation(prompt)
            population.append(variation)

        return population

    async def _generate_variation(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Generate a variation of the prompt."""
        variation_prompt = f"""Create a variation of this user prompt that maintains the same goal but uses different wording or structure:

{prompt.get('user_prompt', '')}

Variation:"""

        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=variation_prompt,
                parameters={"temperature": 0.9, "max_tokens": 500}
            )

            return {
                "user_prompt": result["content"].strip(),
                "system_prompt": prompt.get("system_prompt", "")
            }
        except Exception as e:
            # Fallback to original with minor changes
            logger.debug(f"_generate_variation failed to call LLM; using fallback variation. error={e}")
            return {
                "user_prompt": (prompt.get("user_prompt", "") + "\nBe precise and clear.").strip(),
                "system_prompt": prompt.get("system_prompt", "")
            }

    async def _evaluate_fitness(self, individual: dict[str, Any],
                               test_case_ids: list[int],
                               model_config: dict[str, Any],
                               base_prompt_id: Optional[int]) -> float:
        """Evaluate fitness of an individual."""
        # Create temporary prompt (link to base prompt's project when available)
        prompt_id = await self._individual_to_prompt(individual, base_prompt_id)

        # Evaluate
        scores = []
        for test_case_id in test_case_ids[:3]:  # Limit for speed
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )

            if result.get("success") and "scores" in result:
                scores.append(result["scores"].get("aggregate_score", 0))

        return np.mean(scores) if scores else 0.0

    def _tournament_select(self, fitness_scores: list[tuple[dict, float]],
                          tournament_size: int = 3) -> dict[str, Any]:
        """Tournament selection."""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    async def _crossover(self, parent1: dict[str, Any],
                        parent2: dict[str, Any]) -> dict[str, Any]:
        """Crossover two individuals."""
        # Simple approach: combine parts of prompts
        crossover_prompt = f"""Combine the best aspects of these two user prompts into a new one:

Prompt 1 (user):
{parent1.get('user_prompt', '')}

Prompt 2 (user):
{parent2.get('user_prompt', '')}

Combined user prompt:"""

        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=crossover_prompt,
                parameters={"temperature": 0.7, "max_tokens": 500}
            )

            return {
                "user_prompt": result["content"].strip(),
                "system_prompt": parent1.get("system_prompt", "")
            }
        except Exception as e:
            # Fallback: return parent1
            logger.debug(f"_crossover failed to call LLM; returning parent1. error={e}")
            return parent1.copy()

    async def _mutate(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Mutate an individual."""
        mutation_prompt = f"""Make a small random change to this user prompt while keeping its core purpose:

{individual.get('user_prompt', '')}

Mutated user prompt:"""

        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=mutation_prompt,
                parameters={"temperature": 1.0, "max_tokens": 500}
            )

            return {
                "user_prompt": result["content"].strip(),
                "system_prompt": individual.get("system_prompt", "")
            }
        except Exception as e:
            # Fallback: add random instruction
            logger.debug(f"_mutate failed to call LLM; using random instruction. error={e}")
            mutations = [
                "\nBe concise.",
                "\nProvide detailed explanations.",
                "\nFocus on accuracy.",
                "\nConsider edge cases."
            ]

            return {
                "user_prompt": (individual.get("user_prompt", "") + random.choice(mutations)).strip(),
                "system_prompt": individual.get("system_prompt", "")
            }

    def _has_converged_genetic(self, history: list[dict[str, Any]]) -> bool:
        """Check if population has converged."""
        if len(history) < 5:
            return False

        # Check if best scores are stable
        recent_best = [h["best_score"] for h in history[-5:]]
        return np.std(recent_best) < 0.001

    async def _individual_to_prompt(self, individual: dict[str, Any],
                                   base_prompt_id: Optional[int]) -> int:
        """Convert individual to prompt in database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Get project ID if base prompt provided
        project_id = 1  # Default
        if base_prompt_id:
            cursor.execute(
                "SELECT project_id FROM prompt_studio_prompts WHERE id = ?",
                (base_prompt_id,)
            )
            row = cursor.fetchone()
            if row:
                project_id = row[0]

        # Build a unique name to avoid UNIQUE(project_id, name, version_number) collisions
        # when creating many temporary prompts during optimization/evaluation.
        # Keep a readable base name while guaranteeing uniqueness.
        unique_suffix = f"{int(datetime.utcnow().timestamp()*1000)}-{random.randint(0, 99999):05d}"
        prompt_name = f"Genetic Prompt {unique_suffix}"

        # Create prompt
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, name, system_prompt,
                user_prompt, version_number, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"genetic-{datetime.utcnow().timestamp()}",
            project_id,
            prompt_name,
            individual.get("system_prompt"),
            individual.get("user_prompt", ""),
            1,
            base_prompt_id,
            self.db.client_id
        ))

        prompt_id = cursor.lastrowid
        conn.commit()

        return prompt_id

    def _get_prompt(self, prompt_id: int) -> dict[str, Any]:
        """Get prompt from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
        row = cursor.fetchone()

        if row:
            return self.db._row_to_dict(cursor, row)
        return {}
