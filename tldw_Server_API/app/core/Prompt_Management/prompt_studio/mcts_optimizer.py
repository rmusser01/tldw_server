"""
mcts_optimizer.py
Full MCTS optimizer for Prompt Studio (tree search, UCT, contextual generation,
optional feedback refinement, and WS progress broadcasts).
"""

import hashlib
import json
import math
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError, PromptStudioDatabase
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
    strip_sensitive_durable_mapping,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import (
    EventBroadcaster,
    EventType,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import (
    prompt_studio_metrics,
)

from .prompt_decomposer import PromptDecomposer
from .prompt_executor import PromptExecutor
from .prompt_quality import PromptQualityScorer
from .test_runner import TestRunner
from .types_common import MetricType

_MCTS_IMPORT_EXCEPTIONS = (ImportError, OSError, RuntimeError)
_MCTS_DURABLE_CACHE_VERSION = "v2"
_MCTS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    sqlite3.Error,
    DatabaseError,
)

try:
    # Optional: shared WS connection manager if WS endpoints loaded
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
        connection_manager as ws_connection_manager,
    )
except _MCTS_IMPORT_EXCEPTIONS:  # pragma: no cover - optional in minimal builds
    ws_connection_manager = None
import contextlib

from tldw_Server_API.app.core.testing import is_truthy


class MCTSOptimizer:
    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        self.scorer = PromptQualityScorer(executor=self.executor)
        self.decomposer = PromptDecomposer()
        # Simple in-memory caches (bounded by usage patterns)
        self._rephrase_cache: dict[tuple[str, str, str], str] = {}
        self._eval_cache: dict[str, float] = {}
        try:
            from .optimization_strategies import IterativeRefinementOptimizer  # noqa: WPS433
            self._refiner_cls = IterativeRefinementOptimizer
        except _MCTS_IMPORT_EXCEPTIONS:  # pragma: no cover
            self._refiner_cls = None

    class _Node:
        __slots__ = (
            "parent",
            "children",
            "children_by_bin",
            "segment_index",
            "system_text",
            "q_sum",
            "n_visits",
            "score_bin",
        )

        def __init__(self, *, parent: Optional["MCTSOptimizer._Node"], segment_index: int, system_text: str, score_bin: Optional[int] = None):
            self.parent = parent
            self.children: list[MCTSOptimizer._Node] = []
            self.children_by_bin: dict[int, MCTSOptimizer._Node] = {}
            self.segment_index = segment_index
            self.system_text = system_text
            self.q_sum = 0.0
            self.n_visits = 0
            self.score_bin = score_bin if score_bin is not None else -1

        def uct(self, *, exploration_c: float) -> float:
            if self.n_visits == 0:
                return float("inf")
            parent_visits = self.parent.n_visits if self.parent is not None else max(1, self.n_visits)
            exploitation = self.q_sum / max(1, self.n_visits)
            exploration = exploration_c * math.sqrt(math.log(max(1, parent_visits)) / self.n_visits)
            return exploitation + exploration

    def _optimization_cancelled(self, optimization_id: Optional[int]) -> bool:
        """Read cancellation state without allowing provider work to fail open."""

        if optimization_id is None:
            return False
        optimization = self.db.get_optimization(optimization_id)
        if not optimization:
            raise RuntimeError(
                f"Optimization {optimization_id} disappeared while MCTS was running"
            )
        return str(optimization.get("status") or "").lower() == "cancelled"

    async def optimize(
        self,
        *,
        initial_prompt_id: int,
        optimization_id: Optional[int] = None,
        test_case_ids: list[int],
        model_config: dict[str, Any],
        max_iterations: int = 20,
        target_metric: MetricType = MetricType.ACCURACY,
        strategy_params: Optional[dict[str, Any]] = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        scorer_model_config: dict[str, Any] | None = None,
        scorer_provider_credentials: ProviderCallCredentials | None = None,
        on_scorer_provider_success: Callable[[], Awaitable[None]] | None = None,
        emit_completion_event: bool = True,
    ) -> dict[str, Any]:
        def _coerce_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(int(value))
            return is_truthy(str(value))

        params = strategy_params or {}
        n_sims = int(params.get("mcts_simulations") or max_iterations or 20)
        early_no_improve = int(params.get("early_stop_no_improve") or 5)
        min_quality = float(params.get("min_quality") or 0.0)
        exploration_c = float(params.get("mcts_exploration_c") or 1.4)
        max_depth = int(params.get("mcts_max_depth") or 4)
        k_candidates = int(params.get("prompt_candidates_per_node") or 3)
        score_bin_size = float(params.get("score_dedup_bin") or 0.1)
        token_budget = int(params.get("token_budget") or 0)  # 0 => unlimited
        scorer_model = params.get("scorer_model")
        feedback_enabled = _coerce_bool(params.get("feedback_enabled"), True)
        feedback_threshold = float(params.get("feedback_threshold", 6.0))
        feedback_max_retries = int(params.get("feedback_max_retries", 2))
        strict_provider_errors = (
            on_provider_success is not None
            or provider_credentials is not None
            or scorer_provider_credentials is not None
        )
        provider_call_kwargs = (
            {"provider_credentials": provider_credentials}
            if provider_credentials is not None
            else {}
        )
        scorer_dispatch_state = {"dispatched": False}
        ws_throttle_every = int(params.get("ws_throttle_every") or max(1, int(n_sims // 50) or 1))
        trace_top_k = int(params.get("trace_top_k") or 3)
        # Debugging/observability of decisions
        import os as _os
        debug_decisions = is_truthy(_os.getenv("PROMPT_STUDIO_MCTS_DEBUG_DECISIONS", "false"))

        # Configure scorer
        if scorer_model:
            with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                self.scorer.set_model(str(scorer_model))

        # Token accounting
        self._tokens_spent = 0
        def _add_tokens(n: int):
            with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                self._tokens_spent += int(n or 0)
        self.scorer.set_token_callback(_add_tokens)

        logger.info(
            'MCTS starting: prompt={} sims={} depth={} c={}',
            initial_prompt_id,
            n_sims,
            max_depth,
            exploration_c,
        )

        base_prompt = self._get_prompt(initial_prompt_id)
        base_system = (base_prompt.get("system_prompt") or "").strip()
        base_user = base_prompt.get("user_prompt") or ""
        segments = self.decomposer.decompose_text(base_system + ("\n\n" + base_user if base_user else ""))

        root = self._Node(parent=None, segment_index=0, system_text=base_system, score_bin=None)

        # Baseline evaluation
        best_prompt_id = initial_prompt_id
        if not strict_provider_errors:
            best_score = await self._evaluate_prompt(
                initial_prompt_id,
                test_case_ids,
                model_config,
                target_metric,
                **provider_call_kwargs,
            )
        else:
            best_score = await self._evaluate_prompt(
                initial_prompt_id,
                test_case_ids,
                model_config,
                target_metric,
                on_provider_success=on_provider_success,
                strict_provider_errors=strict_provider_errors,
                **provider_call_kwargs,
            )
        initial_score = best_score

        iteration_history: list[dict[str, Any]] = []
        no_improve_streak = 0
        nodes_created = 0
        edges_created = 0
        parent_ids: set = set()
        t_start = datetime.utcnow()

        # Error/observability counters
        self._counters = {
            "prune_low_quality": 0,
            "prune_dedup": 0,
            "scorer_failures": 0,
            "evaluator_timeouts": 0,
        }
        # Collect top scored candidates per depth when debugging
        self._debug_top_by_depth: dict[int, list[dict[str, Any]]] = {} if debug_decisions else None

        broadcaster = None
        if ws_connection_manager is not None and optimization_id is not None:
            broadcaster = EventBroadcaster(ws_connection_manager, self.db)
            with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                await broadcaster.broadcast_event(
                    event_type=EventType.OPTIMIZATION_STARTED,
                    data={
                        "optimization_id": optimization_id,
                        "strategy": "mcts",
                        "max_iterations": n_sims,
                    },
                    project_id=base_prompt.get("project_id"),
                )

        for sim in range(1, n_sims + 1):
            if self._optimization_cancelled(optimization_id):
                logger.info("MCTS detected cancellation before provider dispatch")
                break
            if token_budget and self._tokens_spent >= token_budget:
                logger.info("MCTS token budget exhausted: {} >= {}", self._tokens_spent, token_budget)
                break
            # Selection & Expansion
            path: list[MCTSOptimizer._Node] = [root]
            node = root
            while True:
                depth = node.segment_index
                if depth >= len(segments) or depth >= max_depth:
                    break
                if len(node.children) < k_candidates:
                    expansion_kwargs: dict[str, Any] = {
                        "segment": segments[depth],
                        "base_user": base_user,
                        "k_candidates": k_candidates,
                        "score_bin_size": score_bin_size,
                        "min_quality": min_quality,
                    }
                    if any(
                        value is not None
                        for value in (
                            provider_credentials,
                            on_provider_success,
                            scorer_model_config,
                            scorer_provider_credentials,
                            on_scorer_provider_success,
                        )
                    ):
                        expansion_kwargs.update(
                            model_config=model_config,
                            scorer_model=(
                                str(scorer_model)
                                if scorer_model is not None
                                else str(model_config.get("model") or "")
                            ),
                            provider_credentials=provider_credentials,
                            on_provider_success=on_provider_success,
                            scorer_model_config=scorer_model_config,
                            scorer_provider_credentials=scorer_provider_credentials,
                            on_scorer_provider_success=on_scorer_provider_success,
                            scorer_dispatch_state=scorer_dispatch_state,
                            strict_provider_errors=strict_provider_errors,
                        )
                    child = await self._expand_node(node, **expansion_kwargs)
                    if child is not None:
                        node = child
                        path.append(node)
                        nodes_created += 1
                        edges_created += 1
                        with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                            parent_ids.add(id(node.parent))
                        continue
                if node.children:
                    # Log selection decision (UCT) for observability
                    try:
                        if debug_decisions:
                            scored_children = [
                                (ch, ch.uct(exploration_c=exploration_c)) for ch in node.children
                            ]
                            chosen, chosen_uct = max(scored_children, key=lambda p: p[1])
                            logger.debug(
                                'mcts.select depth={} chose_child_bin={} uct={}',
                                node.segment_index,
                                getattr(chosen, "score_bin", None),
                                float(chosen_uct),
                            )
                            node = chosen
                        else:
                            node = max(node.children, key=lambda ch: ch.uct(exploration_c=exploration_c))
                    except _MCTS_NONCRITICAL_EXCEPTIONS:
                        node = max(node.children, key=lambda ch: ch.uct(exploration_c=exploration_c))
                    path.append(node)
                    continue
                break

            # Simulation/Evaluation at leaf
            eval_system = node.system_text
            score, prompt_id = await self._evaluate_with_feedback(
                base_prompt=base_prompt,
                system_text=eval_system,
                user_text=base_user,
                test_case_ids=test_case_ids,
                model_config=model_config,
                target_metric=target_metric,
                feedback_enabled=feedback_enabled,
                feedback_threshold=feedback_threshold,
                feedback_max_retries=feedback_max_retries,
                optimization_id=optimization_id,
                provider_credentials=provider_credentials,
                on_provider_success=on_provider_success,
                strict_provider_errors=strict_provider_errors,
            )

            # Backpropagate
            for p in path:
                p.n_visits += 1
                p.q_sum += float(score)

            # Update best and record
            # Compact system trace info
            import hashlib
            sys_hash = hashlib.sha256((eval_system or "").encode("utf-8", errors="ignore")).hexdigest()
            sys_preview = (eval_system or "")[:160]
            iter_entry = {
                "simulation": sim,
                "prompt_id": prompt_id,
                "score": score,
                "improvement": score - best_score,
                "system_hash": sys_hash,
                "system_preview": sys_preview,
            }
            iteration_history.append(iter_entry)
            improved = score > best_score
            if improved:
                best_score = score
                best_prompt_id = prompt_id
                no_improve_streak = 0
            else:
                no_improve_streak += 1
            stop_for_no_improvement = no_improve_streak >= early_no_improve

            # Throttled WS + per-iteration persistence
            do_broadcast = (
                (sim == 1)
                or (sim == n_sims)
                or improved
                or (sim % ws_throttle_every == 0)
            )
            if broadcaster and do_broadcast:
                with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                    await broadcaster.broadcast_optimization_iteration(
                        optimization_id=optimization_id,
                        iteration=sim,
                        max_iterations=n_sims,
                        current_metric=float(score),
                        best_metric=float(best_score),
                        extra_data={
                            "strategy": "mcts",
                            "sim_index": sim,
                            "depth": int(node.segment_index),
                            "reward": float(score),
                            "best_reward": float(best_score),
                            "token_spend_so_far": int(self._tokens_spent),
                            "trace_summary": {
                                "prompt_id": prompt_id,
                                "system_hash": sys_hash,
                            },
                        },
                    )

            # Persist iteration record (throttled similarly to WS)
            if optimization_id is not None and do_broadcast:
                with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                    self.db.record_optimization_iteration(
                        optimization_id,
                        iteration_number=sim,
                        prompt_variant={
                            "prompt_id": prompt_id,
                            "system_hash": sys_hash,
                            "system_preview": sys_preview,
                        },
                        metrics={
                            "score": float(score),
                            "best_metric": float(best_score),
                        },
                        tokens_used=int(self._tokens_spent),
                        note="mcts-iteration",
                    )

            if self._optimization_cancelled(optimization_id):
                logger.info("MCTS detected cancellation; exiting loop")
                break
            if stop_for_no_improvement:
                logger.info("MCTS early stop: no improvement for {} sims", early_no_improve)
                break

        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000.0
        parents_used = len(parent_ids) or 1
        avg_branching = float(edges_created) / float(parents_used)

        # Record metrics and error counters
        with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
            prompt_studio_metrics.record_mcts_summary(
                sims_total=len(iteration_history),
                tree_nodes=nodes_created,
                avg_branching=avg_branching,
                best_reward=float(best_score),
                tokens_spent=self._tokens_spent,
                duration_ms=duration_ms,
            )
        # Emit error counters
        try:
            for key, val in (self._counters or {}).items():
                if not val:
                    continue
                # Map internal keys to error labels
                label = {
                    "prune_low_quality": "prune_low_quality",
                    "prune_dedup": "prune_dedup",
                    "scorer_failures": "scorer_failure",
                    "evaluator_timeouts": "evaluator_timeout",
                }.get(key, key)
                prompt_studio_metrics.record_mcts_error(error=label, count=int(val))
        except _MCTS_NONCRITICAL_EXCEPTIONS:
            pass

        if (
            broadcaster
            and emit_completion_event
            and not self._optimization_cancelled(optimization_id)
        ):
            with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                await broadcaster.broadcast_event(
                    event_type=EventType.OPTIMIZATION_COMPLETED,
                    data={
                        "optimization_id": optimization_id,
                        "strategy": "mcts",
                        "iterations": len(iteration_history),
                        "final_score": float(best_score),
                        "tokens_spent": int(self._tokens_spent),
                    },
                    project_id=base_prompt.get("project_id"),
                )

        # Build compact final trace: best path + top-K candidates
        top_candidates = sorted(iteration_history, key=lambda e: e.get("score", 0.0), reverse=True)[: max(1, trace_top_k)]
        final_trace = {
            "best_path": {
                "prompt_id": best_prompt_id,
                "system_hash": (top_candidates[0]["system_hash"] if top_candidates else None),
                "system_preview": (top_candidates[0]["system_preview"] if top_candidates else None),
                "depth": None,  # unknown without tracking full path; kept for schema stability
            },
            "top_candidates": [
                {
                    "simulation": tc.get("simulation"),
                    "prompt_id": tc.get("prompt_id"),
                    "score": tc.get("score"),
                    "system_hash": tc.get("system_hash"),
                    "system_preview": tc.get("system_preview"),
                }
                for tc in top_candidates
            ],
            "sims_total": len(iteration_history),
        }
        if debug_decisions and isinstance(self._debug_top_by_depth, dict):
            final_trace["debug_top_scores_by_depth"] = self._debug_top_by_depth

        result = {
            "initial_prompt_id": initial_prompt_id,
            "optimized_prompt_id": best_prompt_id,
            "initial_score": initial_score,
            "final_score": best_score,
            "improvement": best_score - initial_score,
            "iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "strategy": "MCTS",
            "scorer_provider_dispatched": scorer_dispatch_state["dispatched"],
            "total_tokens": self._tokens_spent,
            "duration_ms": duration_ms,
            # Extra metrics/traces for engine to persist
            "final_metrics": {
                "score": float(best_score),
                "best_reward": float(best_score),
                "tree_nodes": nodes_created,
                "avg_branching": avg_branching,
                "tokens_spent": int(self._tokens_spent),
                "duration_ms": duration_ms,
                "trace": final_trace,
                "errors": dict(self._counters or {}),
                "applied_params": {
                    "mcts_max_depth": max_depth,
                    "prompt_candidates_per_node": k_candidates,
                    "mcts_exploration_c": exploration_c,
                    "mcts_simulations": n_sims,
                    "min_quality": min_quality,
                    "score_dedup_bin": score_bin_size,
                    "token_budget": token_budget,
                    "scorer_model": str(scorer_model) if scorer_model is not None else None,
                    "feedback_enabled": bool(feedback_enabled),
                    "feedback_threshold": feedback_threshold,
                    "feedback_max_retries": feedback_max_retries,
                    "ws_throttle_every": ws_throttle_every,
                    "trace_top_k": trace_top_k,
                },
            },
        }

        # Reset counters holder
        self._counters = None
        return result

    async def _expand_node(
        self,
        node: "MCTSOptimizer._Node",
        *,
        segment: str,
        base_user: str,
        k_candidates: int,
        score_bin_size: float,
        min_quality: float,
        model_config: dict[str, Any] | None = None,
        scorer_model: str | None = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        scorer_model_config: dict[str, Any] | None = None,
        scorer_provider_credentials: ProviderCallCredentials | None = None,
        on_scorer_provider_success: Callable[[], Awaitable[None]] | None = None,
        scorer_dispatch_state: dict[str, bool] | None = None,
        strict_provider_errors: bool = False,
    ) -> Optional["MCTSOptimizer._Node"]:
        # Support both async and sync monkeypatching for _propose_candidates in tests
        try:
            if model_config is None:
                maybe = self._propose_candidates(
                    node.system_text,
                    segment,
                    k_candidates,
                )
            else:
                maybe = self._propose_candidates(
                    node.system_text,
                    segment,
                    k_candidates,
                    model_config=model_config,
                    provider_credentials=provider_credentials,
                    on_provider_success=on_provider_success,
                    strict_provider_errors=strict_provider_errors,
                )
            if hasattr(maybe, "__await__") or hasattr(maybe, "__aiter__"):
                candidates = await maybe  # type: ignore[assignment]
            else:
                candidates = maybe  # type: ignore[assignment]
        except TypeError:
            if model_config is not None:
                raise
            # Fallback to direct await if attribute detection failed
            candidates = await self._propose_candidates(
                node.system_text,
                segment,
                k_candidates,
            )
        if not candidates:
            return None
        best_existing: Optional[MCTSOptimizer._Node] = None
        best_existing_score = -1.0
        new_child: Optional[MCTSOptimizer._Node] = None
        scored: list[tuple[str, float, int]] = []
        quality_model_config = scorer_model_config or model_config
        quality_success_callback = (
            on_scorer_provider_success or on_provider_success
        )
        quality_provider_credentials = scorer_provider_credentials
        if (
            quality_provider_credentials is None
            and scorer_model_config is None
            and (
                scorer_model is None
                or str(scorer_model)
                == str((model_config or {}).get("model") or "")
            )
        ):
            quality_provider_credentials = provider_credentials
        for cand_system in candidates:
            # DB-backed scorer cache (optional)
            cache_allowed = (
                quality_model_config is None and scorer_model is None
            ) or self._provider_result_cache_allowed(quality_model_config or {})
            try:
                behavior_fingerprint = self._model_behavior_fingerprint(
                    quality_model_config or {}
                )
                cache_user = (
                    f"{behavior_fingerprint}\0{base_user}"
                    if quality_model_config is not None
                    else f"heuristic\0{base_user}"
                )
                key = "scorer:" + self.scorer._cache_key(
                    cand_system,
                    cache_user,
                    scorer_model,
                )
                cached = self._db_cache_get(key) if cache_allowed else None
            except _MCTS_NONCRITICAL_EXCEPTIONS:
                cached = None
            if cached is not None:
                q = float(cached)
            else:
                try:
                    if (
                        scorer_dispatch_state is not None
                        and on_scorer_provider_success is not None
                    ):
                        scorer_dispatch_state["dispatched"] = True
                    if quality_model_config is None:
                        q = await self.scorer.score_prompt_async(
                            system_text=cand_system,
                            user_text=base_user,
                        )
                    else:
                        scorer_kwargs: dict[str, Any] = {
                            "system_text": cand_system,
                            "user_text": base_user,
                            "model_config": quality_model_config,
                            "scorer_model": scorer_model,
                            "on_provider_success": quality_success_callback,
                            "strict_provider_errors": strict_provider_errors,
                            "cache_scope": behavior_fingerprint,
                            "use_cache": cache_allowed,
                        }
                        if quality_provider_credentials is not None:
                            scorer_kwargs["provider_credentials"] = (
                                quality_provider_credentials
                            )
                        q = await self.scorer.score_prompt_async(**scorer_kwargs)
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    if strict_provider_errors:
                        raise
                    q = 0.0
                    try:
                        if hasattr(self, "_counters") and isinstance(self._counters, dict):
                            self._counters["scorer_failures"] = self._counters.get("scorer_failures", 0) + 1
                    except _MCTS_NONCRITICAL_EXCEPTIONS:
                        pass
                if cache_allowed:
                    with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                        self._db_cache_set(key, q, ttl_sec=1800)
            try:
                bin_idx = PromptQualityScorer.score_to_bin(q, score_bin_size)
                scored.append((cand_system, q, bin_idx))
            except _MCTS_NONCRITICAL_EXCEPTIONS:
                bin_idx = PromptQualityScorer.score_to_bin(q, score_bin_size)
            if q < min_quality:
                try:
                    if hasattr(self, "_counters") and isinstance(self._counters, dict):
                        self._counters["prune_low_quality"] = self._counters.get("prune_low_quality", 0) + 1
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    pass
                continue
            if bin_idx in node.children_by_bin:
                ch = node.children_by_bin[bin_idx]
                if q > best_existing_score:
                    best_existing = ch
                    best_existing_score = q
                try:
                    if hasattr(self, "_counters") and isinstance(self._counters, dict):
                        self._counters["prune_dedup"] = self._counters.get("prune_dedup", 0) + 1
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    pass
                continue
            # Create at most one child per expansion
            if new_child is None:
                child = self._Node(parent=node, segment_index=node.segment_index + 1, system_text=cand_system, score_bin=bin_idx)
                node.children.append(child)
                node.children_by_bin[bin_idx] = child
                new_child = child
            # Debug: record top scored candidates for this depth
            try:
                if isinstance(self._debug_top_by_depth, dict):
                    depth = node.segment_index
                    top = sorted(scored, key=lambda t: t[1], reverse=True)[:3]
                    self._debug_top_by_depth[depth] = [
                        {"score": float(s[1]), "bin": int(s[2]), "system_preview": (s[0] or "")[:160]}
                        for s in top
                    ]
            except _MCTS_NONCRITICAL_EXCEPTIONS:
                pass
        # If no child added, still capture debug top scored at this depth
        try:
            if isinstance(self._debug_top_by_depth, dict) and scored:
                depth = node.segment_index
                top = sorted(scored, key=lambda t: t[1], reverse=True)[:3]
                self._debug_top_by_depth[depth] = [
                    {"score": float(s[1]), "bin": int(s[2]), "system_preview": (s[0] or "")[:160]}
                    for s in top
                ]
        except _MCTS_NONCRITICAL_EXCEPTIONS:
            pass
        return new_child or best_existing

    async def _propose_candidates(
        self,
        system_so_far: str,
        segment_text: str,
        k: int,
        *,
        model_config: dict[str, Any] | None = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
    ) -> list[str]:
        proposals: list[str] = []
        if model_config is None and on_provider_success is None:
            improved = await self._rephrase_segment(system_so_far, segment_text)
        else:
            improved = await self._rephrase_segment(
                system_so_far,
                segment_text,
                model_config=model_config,
                provider_credentials=provider_credentials,
                on_provider_success=on_provider_success,
                strict_provider_errors=strict_provider_errors,
            )
        if improved:
            proposals.append(improved)
        suffix = "\n\nEnsure outputs strictly follow the required format and constraints."
        proposals.append((system_so_far + suffix).strip())
        suffix2 = "\n\nBefore responding, validate that all required fields are present."
        proposals.append((system_so_far + suffix2).strip())
        seen = set()
        uniq: list[str] = []
        for p in proposals:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
            if len(uniq) >= k:
                break
        return uniq

    async def _rephrase_segment(
        self,
        system_text: str,
        segment_text: str,
        *,
        model_config: dict[str, Any] | None = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
    ) -> Optional[str]:
        if not system_text or not segment_text:
            return None
        selected_config = model_config or {}
        cache_allowed = self._provider_result_cache_allowed(selected_config)
        behavior_fingerprint = self._model_behavior_fingerprint(selected_config)
        cache_key = (
            behavior_fingerprint,
            system_text,
            segment_text,
        )
        if cache_allowed and cache_key in self._rephrase_cache:
            return self._rephrase_cache[cache_key]
        # DB cache
        db_key = "rephrase:" + self._hash_pair(
            behavior_fingerprint + "\0" + system_text,
            segment_text,
        )
        if cache_allowed:
            try:
                cached = self._db_cache_get(db_key)
                if isinstance(cached, str) and cached:
                    self._rephrase_cache[cache_key] = cached
                    return cached
            except _MCTS_NONCRITICAL_EXCEPTIONS:
                pass
        prompt = (
            "You are improving a system prompt for an assistant.\n"
            "Focus on the following segment, enhancing clarity, specificity, and constraint adherence,"
            " without changing the overall intent. Return the full revised system prompt.\n\n"
            f"Current system prompt:\n{system_text}\n\nSegment to improve:\n{segment_text}\n\n"
            "Revised system prompt:"
        )
        try:
            parameters = dict(selected_config.get("parameters") or {})
            parameters.update({"temperature": 0.5, "max_tokens": 600})
            result = await self.executor._call_llm(
                provider=str(selected_config.get("provider") or "openai"),
                model=str(selected_config.get("model") or "gpt-3.5-turbo"),
                prompt=prompt,
                parameters=parameters,
                api_key_override=selected_config.get("api_key"),
                app_config=selected_config.get("app_config"),
                credentials_resolved=(
                    selected_config.get("credentials_resolved") is True
                ),
                provider_credentials=provider_credentials,
                timeout_seconds=parameters.get("timeout_seconds"),
                on_provider_success=on_provider_success,
            )
            content = (result or {}).get("content", "").strip()
            with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                self._tokens_spent += int((result or {}).get("tokens", 0) or 0)
            if content:
                if cache_allowed:
                    self._rephrase_cache[cache_key] = content
                    with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                        self._db_cache_set(db_key, content, ttl_sec=3600)
            return content or None
        except _MCTS_NONCRITICAL_EXCEPTIONS:
            if strict_provider_errors:
                raise
            return None

    async def _evaluate_with_feedback(
        self,
        *,
        base_prompt: dict[str, Any],
        system_text: str,
        user_text: str,
        test_case_ids: list[int],
        model_config: dict[str, Any],
        target_metric: MetricType,
        feedback_enabled: bool,
        feedback_threshold: float,
        feedback_max_retries: int,
        optimization_id: Optional[int] = None,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
    ) -> tuple[float, int]:
        # Caching by content to reduce repeated evaluations
        provider_call_kwargs = (
            {"provider_credentials": provider_credentials}
            if provider_credentials is not None
            else {}
        )
        test_case_fingerprint = self._test_case_behavior_fingerprint(test_case_ids)
        cache_allowed = (
            self._provider_result_cache_allowed(model_config)
            and test_case_fingerprint is not None
        )
        eval_cache_key = self._make_eval_cache_key(
            system_text,
            user_text,
            model_config,
            test_case_ids,
            target_metric,
            test_case_fingerprint=test_case_fingerprint,
        )
        cached = self._eval_cache.get(eval_cache_key) if cache_allowed else None
        prompt_id = self._create_ephemeral_prompt_version(
            base_prompt=base_prompt,
            system_text=system_text,
            user_text=user_text,
        )
        if cached is not None:
            score = cached
        else:
            # DB cache (rollout)
            db_key = "eval:" + eval_cache_key
            if cache_allowed:
                try:
                    cached_db = self._db_cache_get(db_key)
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    cached_db = None
            else:
                cached_db = None
            if cached_db is not None:
                score = float(cached_db)
            else:
                if not strict_provider_errors:
                    score = await self._evaluate_prompt(
                        prompt_id,
                        test_case_ids,
                        model_config,
                        target_metric,
                        **provider_call_kwargs,
                    )
                else:
                    score = await self._evaluate_prompt(
                        prompt_id,
                        test_case_ids,
                        model_config,
                        target_metric,
                        on_provider_success=on_provider_success,
                        strict_provider_errors=strict_provider_errors,
                        **provider_call_kwargs,
                    )
                if cache_allowed:
                    self._eval_cache[eval_cache_key] = score
                    with contextlib.suppress(_MCTS_NONCRITICAL_EXCEPTIONS):
                        self._db_cache_set(db_key, score, ttl_sec=3600)
        scaled = score * 10.0
        if not feedback_enabled or scaled >= feedback_threshold or not self._refiner_cls:
            return score, prompt_id
        refiner = self._refiner_cls(self.db, self.test_runner)
        best_score = score
        best_prompt_id = prompt_id
        for _ in range(max(0, feedback_max_retries)):
            try:
                refiner_kwargs: dict[str, Any] = {
                    "prompt_id": best_prompt_id,
                    "test_case_ids": test_case_ids,
                    "model_config": model_config,
                    "max_iterations": 1,
                    "optimization_id": optimization_id,
                }
                if on_provider_success is not None:
                    refiner_kwargs["on_provider_success"] = on_provider_success
                if provider_credentials is not None:
                    refiner_kwargs["provider_credentials"] = provider_credentials
                result = await refiner.optimize(
                    **refiner_kwargs,
                )
                cand_id = int(result.get("optimized_prompt_id", best_prompt_id))
                if not strict_provider_errors:
                    cand_score = await self._evaluate_prompt(
                        cand_id,
                        test_case_ids,
                        model_config,
                        target_metric,
                        **provider_call_kwargs,
                    )
                else:
                    cand_score = await self._evaluate_prompt(
                        cand_id,
                        test_case_ids,
                        model_config,
                        target_metric,
                        on_provider_success=on_provider_success,
                        strict_provider_errors=strict_provider_errors,
                        **provider_call_kwargs,
                    )
                if cand_score > best_score:
                    best_score = cand_score
                    best_prompt_id = cand_id
                if best_score * 10.0 >= feedback_threshold:
                    break
            except _MCTS_NONCRITICAL_EXCEPTIONS:
                if strict_provider_errors:
                    raise
                break
        return best_score, best_prompt_id

    def _create_ephemeral_prompt_version(self, *, base_prompt: dict[str, Any], system_text: str, user_text: str) -> int:
        # Compute next version number for the same prompt name within the project to avoid collisions
        new_name = f"{base_prompt['name']} (MCTS)"
        select_sql = """
            SELECT COALESCE(MAX(version_number), 0)
            FROM prompt_studio_prompts
            WHERE project_id = ? AND name = ?
        """
        insert_sql = """
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, name, system_prompt,
                user_prompt, version_number, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """
        with self.db.transaction() as conn:
            cursor = self.db._cursor_exec(conn, select_sql, (base_prompt["project_id"], new_name))
            row = cursor.fetchone()
            if row is not None and row[0] is not None:
                next_version = int(row[0]) + 1
            else:
                next_version = int(base_prompt.get("version_number") or 0) + 1

            cursor = self.db._cursor_exec(
                conn,
                insert_sql,
                (
                    f"mcts-{datetime.utcnow().timestamp()}",
                    base_prompt["project_id"],
                    base_prompt.get("signature_id"),
                    new_name,
                    system_text,
                    user_text,
                    next_version,
                    base_prompt.get("id"),
                    self.db.client_id,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                new_id = getattr(cursor, "lastrowid", None)
            elif isinstance(row, dict):
                new_id = row.get("id")
            else:
                new_id = row[0]
            if new_id is None:
                raise RuntimeError("Failed to create MCTS prompt version")
            return int(new_id)

    async def _evaluate_prompt(
        self,
        prompt_id: int,
        test_case_ids: list[int],
        model_config: dict[str, Any],
        target_metric: MetricType,
        *,
        provider_credentials: ProviderCallCredentials | None = None,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
        strict_provider_errors: bool = False,
    ) -> float:
        scores: list[float] = []
        metric_key = getattr(target_metric, "value", str(target_metric))
        for tc_id in test_case_ids:
            try:
                runner_kwargs = {
                    "prompt_id": prompt_id,
                    "test_case_id": tc_id,
                    "model_config": model_config,
                    "metrics": (
                        [target_metric]
                        if hasattr(target_metric, "value")
                        else None
                    ),
                }
                if on_provider_success is not None or strict_provider_errors:
                    runner_kwargs.update(
                        strict_provider_errors=strict_provider_errors,
                        on_provider_success=on_provider_success,
                    )
                if provider_credentials is not None:
                    runner_kwargs["provider_credentials"] = provider_credentials
                result = await self.test_runner.run_single_test(**runner_kwargs)
            except _MCTS_NONCRITICAL_EXCEPTIONS as e:
                if strict_provider_errors:
                    raise
                # Count timeouts; keep conservative by substring match
                msg = str(e).lower()
                if "timeout" in msg or "timed out" in msg:
                    try:
                        if hasattr(self, "_counters") and isinstance(self._counters, dict):
                            self._counters["evaluator_timeouts"] = self._counters.get("evaluator_timeouts", 0) + 1
                    except _MCTS_NONCRITICAL_EXCEPTIONS:
                        pass
                continue
            if result.get("success") and "scores" in result:
                score = result["scores"].get(metric_key)
                if score is None:
                    score = result["scores"].get("aggregate_score", 0.0)
                scores.append(float(score))
        if not scores and strict_provider_errors:
            raise ValueError("Optimization requires one validated baseline result")
        return sum(scores) / len(scores) if scores else 0.0

    async def _rephrase_instruction(self, instruction: str) -> Optional[str]:
        if not instruction:
            return None
        prompt = (
            "Rephrase these system instructions to be clearer and more precise, "
            "keeping the same intent.\n\n" + instruction + "\n\nRephrased:"
        )
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=prompt,
                parameters={"temperature": 0.5, "max_tokens": 300},
            )
            return (result or {}).get("content", "").strip() or None
        except _MCTS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"mcts: rephrase failed: {e}")
            return None

    def _get_prompt(self, prompt_id: int) -> dict[str, Any]:
        p = self.db.get_prompt(prompt_id)
        if not p:
            raise ValueError(f"Prompt {prompt_id} not found")
        return p

    @staticmethod
    def _provider_result_cache_allowed(model_config: Mapping[str, Any]) -> bool:
        """Cache provider results only from an authoritative runtime snapshot."""

        return model_config.get("credentials_resolved") is True

    @staticmethod
    def _model_behavior_fingerprint(model_config: Mapping[str, Any]) -> str:
        """Hash only provider behavior, excluding request-scoped credentials."""

        safe_config = strip_sensitive_durable_mapping(model_config)
        raw_app_config = model_config.get("app_config")
        if isinstance(raw_app_config, Mapping):
            safe_config["app_config"] = strip_sensitive_durable_mapping(
                raw_app_config
            )

        provider = canonical_provider_name(
            str(model_config.get("provider") or model_config.get("api_name") or "")
        )
        if provider:
            safe_config["provider"] = provider
        model = str(
            model_config.get("model") or model_config.get("model_name") or ""
        ).strip()
        if model:
            safe_config["model"] = model

        encoded = json.dumps(
            safe_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _test_case_behavior_fingerprint(
        self,
        test_case_ids: list[int],
    ) -> str | None:
        """Hash the authoritative test-case records used by an evaluation.

        Durable result caching fails closed when the requested cases cannot be
        read exactly. Hashing the normalized records, rather than timestamps
        alone, also invalidates the cache when multiple edits occur within a
        database timestamp's resolution.
        """

        try:
            requested_ids = [int(test_case_id) for test_case_id in test_case_ids]
            if not requested_ids or len(set(requested_ids)) != len(requested_ids):
                return None
            records = self.db.get_test_cases_by_ids(
                requested_ids,
                include_deleted=False,
            )
            if not isinstance(records, list):
                return None
            records_by_id = {
                int(record["id"]): record
                for record in records
                if isinstance(record, Mapping) and record.get("id") is not None
            }
            if set(records_by_id) != set(requested_ids):
                return None
            encoded = json.dumps(
                [records_by_id[test_case_id] for test_case_id in sorted(requested_ids)],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()
        except _MCTS_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _make_eval_cache_key(
        system_text: str,
        user_text: str,
        model_config: dict[str, Any],
        test_case_ids: list[int],
        target_metric: MetricType | str | None = None,
        *,
        test_case_fingerprint: str | None = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(system_text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(user_text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(MCTSOptimizer._model_behavior_fingerprint(model_config).encode("ascii"))
        h.update(b"\0")
        h.update(
            (",".join(str(int(x)) for x in sorted(test_case_ids or []))).encode("utf-8")
        )
        h.update(b"\0")
        h.update(str(test_case_fingerprint or "").encode("ascii", errors="ignore"))
        h.update(b"\0")
        metric_value = getattr(target_metric, "value", target_metric)
        h.update(str(metric_value or "").encode("utf-8", errors="ignore"))
        return h.hexdigest()

    # --- Simple DB-backed cache via sync_log ---
    def _durable_cache_supported(self) -> bool:
        """Return whether this backend owns the sync_log cache schema."""

        backend_type = getattr(self.db, "backend_type", None)
        backend_name = str(getattr(backend_type, "value", backend_type) or "")
        # Prompt Studio's PostgreSQL schema does not own sync_log. Other
        # subsystems may create an incompatible table with the same name.
        return backend_name.strip().lower() not in {"postgres", "postgresql"}

    def _scoped_db_cache_key(self, key: str) -> str | None:
        """Namespace durable cache entries by tenant without storing tenant PII."""

        tenant_identity = getattr(self.db, "tenant_user_id", None) or getattr(
            self.db,
            "user_id",
            None,
        )
        if not tenant_identity:
            backend_type = getattr(self.db, "backend_type", None)
            backend_name = str(getattr(backend_type, "value", backend_type) or "")
            if backend_name.strip().lower() == "sqlite":
                tenant_identity = getattr(self.db, "client_id", None)
        if not tenant_identity:
            return None
        tenant_hash = hashlib.sha256(
            str(tenant_identity).encode("utf-8", errors="ignore")
        ).hexdigest()
        return f"{_MCTS_DURABLE_CACHE_VERSION}:{tenant_hash}:{key}"

    def _db_cache_get(self, key: str) -> Optional[Any]:
        try:
            if not self._durable_cache_supported():
                return None
            scoped_key = self._scoped_db_cache_key(key)
            if scoped_key is None:
                return None
            with self.db.transaction() as conn:
                cursor = self.db._cursor_exec(
                    conn,
                    "SELECT payload, timestamp FROM sync_log WHERE entity = ? AND entity_uuid = ? ORDER BY timestamp DESC LIMIT 1",
                    ("prompt_studio_cache", scoped_key),
                )
                row = cursor.fetchone()
            if not row:
                return None

            payload_raw = None
            if isinstance(row, dict):
                payload_raw = row.get("payload")
            else:
                try:
                    payload_raw = row.get("payload")
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    payload_raw = row[0] if isinstance(row, (list, tuple)) or hasattr(row, "__getitem__") else None

            if isinstance(payload_raw, (bytes, bytearray, memoryview)):
                try:
                    payload_raw = bytes(payload_raw).decode("utf-8")
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    payload_raw = None

            import datetime
            import json
            if isinstance(payload_raw, str):
                payload = json.loads(payload_raw)
            elif isinstance(payload_raw, dict):
                payload = payload_raw
            else:
                return None

            expires = payload.get("expires_at") if isinstance(payload, dict) else None
            if expires:
                try:
                    if datetime.datetime.fromisoformat(expires) < datetime.datetime.utcnow():
                        return None
                except _MCTS_NONCRITICAL_EXCEPTIONS:
                    pass
            return payload.get("value") if isinstance(payload, dict) else None
        except _MCTS_NONCRITICAL_EXCEPTIONS:
            return None

    def _db_cache_set(self, key: str, value: Any, *, ttl_sec: int = 3600) -> None:
        try:
            if not self._durable_cache_supported():
                return
            import datetime
            scoped_key = self._scoped_db_cache_key(key)
            if scoped_key is None:
                return
            expires_at = (datetime.datetime.utcnow() + datetime.timedelta(seconds=int(ttl_sec))).isoformat()
            payload = {"value": value, "expires_at": expires_at}
            self.db._log_sync_event(
                entity="prompt_studio_cache",
                entity_uuid=scoped_key,
                operation="update",
                payload=payload,
            )
        except _MCTS_NONCRITICAL_EXCEPTIONS:
            pass

    @staticmethod
    def _hash_pair(a: str, b: str) -> str:
        h = hashlib.sha256()
        h.update(a.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(b.encode("utf-8", errors="ignore"))
        return h.hexdigest()
