"""Runtime bounded candidate exploration for persona websocket planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import time
from typing import Any

from tldw_Server_API.app.core.Persona.dialogue_tree_pruners import (
    PruneDecision,
    PruneSeverity,
    duplicate_low_diversity_pruner,
    malformed_candidate_pruner,
    persona_boundary_violation_pruner,
    prompt_injection_pressure_pruner,
    unsafe_tool_plan_pruner,
)
from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
    AggregateScoreResult,
    ScoreResult,
    aggregate_scores,
    grounding_style_score,
    persona_consistency_score,
    policy_score,
    refusal_quality_score,
    tool_plan_score,
    usefulness_score,
)

RuntimeCandidate = Mapping[str, Any]
CandidateGenerator = Callable[[Mapping[str, Any]], Any]
RuntimeScorer = Callable[[RuntimeCandidate], ScoreResult]


class ExplorationFallback(str, Enum):
    DISABLED = "disabled"
    SOFT_EXISTING_BEHAVIOR = "soft_existing_behavior"
    HARD_SAFE_DENIAL = "hard_safe_denial"
    CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True)
class RuntimeExplorerConfig:
    enabled: bool = False
    max_depth: int = 1
    max_branching: int = 2
    max_provider_calls: int = 1
    timeout_ms: int = 750
    max_tokens: int = 256
    llm_judges_enabled: bool = False
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 30.0
    safe_denial_text: str = (
        "I cannot safely proceed with that request. I can help with a safer alternative instead."
    )

    def __post_init__(self) -> None:
        for field_name in (
            "max_depth",
            "max_branching",
            "max_provider_calls",
            "timeout_ms",
            "max_tokens",
            "circuit_breaker_failure_threshold",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.circuit_breaker_cooldown_seconds < 0:
            raise ValueError("circuit_breaker_cooldown_seconds must be >= 0")


@dataclass(frozen=True)
class RuntimeBudgetUsage:
    provider_calls: int = 0
    candidates_considered: int = 0
    hard_prunes: int = 0
    soft_prunes: int = 0
    elapsed_ms: int = 0


@dataclass(frozen=True)
class RuntimeExplorationResult:
    selected_candidate: dict[str, Any] | None
    fallback: ExplorationFallback | None
    safe_denial: str | None
    budget: RuntimeBudgetUsage
    score: AggregateScoreResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    circuit_open: bool = False


class PersonaRuntimeExplorer:
    def __init__(
        self,
        *,
        config: RuntimeExplorerConfig,
        candidate_generator: CandidateGenerator,
        scorers: Sequence[RuntimeScorer] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.candidate_generator = candidate_generator
        self.scorers = list(scorers or _DEFAULT_SCORERS)
        self._clock = clock or time.monotonic
        self._consecutive_runtime_failures = 0
        self._circuit_opened_at: float | None = None
        self._generator_lock = threading.Lock()
        self._timed_out_generator_in_flight = False

    def explore(self, context: Mapping[str, Any]) -> RuntimeExplorationResult:
        started_at = self._clock()
        if not self.config.enabled:
            return RuntimeExplorationResult(
                selected_candidate=None,
                fallback=ExplorationFallback.DISABLED,
                safe_denial=None,
                budget=RuntimeBudgetUsage(elapsed_ms=self._elapsed_ms(started_at)),
            )

        if self._circuit_is_open(started_at):
            return RuntimeExplorationResult(
                selected_candidate=None,
                fallback=ExplorationFallback.CIRCUIT_OPEN,
                safe_denial=None,
                budget=RuntimeBudgetUsage(elapsed_ms=self._elapsed_ms(started_at)),
                diagnostics={"reason": "runtime_explorer_circuit_open"},
                circuit_open=True,
            )

        max_provider_calls = min(
            max(0, int(self.config.max_provider_calls)),
            max(0, int(self.config.max_depth)),
        )
        if max_provider_calls <= 0:
            return RuntimeExplorationResult(
                selected_candidate=None,
                fallback=ExplorationFallback.SOFT_EXISTING_BEHAVIOR,
                safe_denial=None,
                budget=RuntimeBudgetUsage(elapsed_ms=self._elapsed_ms(started_at)),
                diagnostics={"reason": "provider_call_budget_exhausted"},
            )

        provider_calls = 0
        raw_candidates: list[Any] = []
        try:
            for call_index in range(max_provider_calls):
                elapsed_ms = self._elapsed_ms(started_at)
                remaining_timeout_ms = max(0, self.config.timeout_ms - elapsed_ms)
                if remaining_timeout_ms <= 0:
                    raise TimeoutError("runtime explorer exceeded timeout budget")
                call_context = dict(context)
                call_context.update(
                    {
                        "runtime_depth": call_index + 1,
                        "runtime_max_depth": self.config.max_depth,
                        "runtime_provider_call_index": call_index,
                        "runtime_max_provider_calls": self.config.max_provider_calls,
                    }
                )
                provider_calls += 1
                raw_candidates.extend(
                    _coerce_candidate_list(
                        self._generate_candidates_with_timeout(
                            call_context,
                            timeout_ms=remaining_timeout_ms,
                        )
                    )
                )
        except _GeneratorBusyError as exc:
            return self._runtime_failure_result(
                started_at=started_at,
                provider_calls=0,
                error=exc,
                reason="candidate_generation_busy",
            )
        except TimeoutError as exc:
            return self._runtime_failure_result(
                started_at=started_at,
                provider_calls=provider_calls,
                error=exc,
                reason="candidate_generation_timeout",
            )
        except Exception as exc:
            return self._runtime_failure_result(
                started_at=started_at,
                provider_calls=provider_calls,
                error=exc,
                reason="candidate_generation_error",
            )

        elapsed_ms = self._elapsed_ms(started_at)
        if elapsed_ms > self.config.timeout_ms:
            return self._runtime_failure_result(
                started_at=started_at,
                provider_calls=provider_calls,
                error=TimeoutError("runtime explorer exceeded timeout budget"),
                reason="runtime_timeout",
            )

        candidates = raw_candidates[: self.config.max_branching]
        usage = _MutableBudget(provider_calls=provider_calls, elapsed_ms=elapsed_ms)
        hard_violation_seen = False
        scored_candidates: list[tuple[float, int, dict[str, Any], AggregateScoreResult]] = []
        existing_signatures: set[str] = set()

        for index, candidate in enumerate(candidates):
            usage.candidates_considered += 1
            checked_candidate = _candidate_with_derived_tool_plan(candidate)
            prune_decisions = _run_runtime_pruners(
                checked_candidate,
                existing_signatures=existing_signatures,
            )
            hard_prunes = [decision for decision in prune_decisions if decision.severity == PruneSeverity.HARD]
            soft_prunes = [decision for decision in prune_decisions if decision.severity == PruneSeverity.SOFT]
            usage.hard_prunes += len(hard_prunes)
            usage.soft_prunes += len(soft_prunes)
            if hard_prunes:
                hard_violation_seen = True
                continue
            if soft_prunes or not isinstance(checked_candidate, Mapping):
                continue

            aggregate = self._score_candidate(checked_candidate)
            if aggregate.failed_results:
                usage.soft_prunes += 1
                continue
            scored_candidates.append((aggregate.overall_score, index, dict(checked_candidate), aggregate))

        budget = usage.freeze(elapsed_ms=self._elapsed_ms(started_at))
        if scored_candidates:
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            selected_score, _index, selected_candidate, aggregate = scored_candidates[0]
            self._reset_runtime_failures()
            return RuntimeExplorationResult(
                selected_candidate=selected_candidate,
                fallback=None,
                safe_denial=None,
                budget=budget,
                score=aggregate,
                diagnostics={"selected_score": selected_score},
            )

        self._reset_runtime_failures()
        if hard_violation_seen:
            return RuntimeExplorationResult(
                selected_candidate=None,
                fallback=ExplorationFallback.HARD_SAFE_DENIAL,
                safe_denial=self.config.safe_denial_text,
                budget=budget,
                diagnostics={"reason": "hard_prune_without_safe_candidate"},
            )

        return RuntimeExplorationResult(
            selected_candidate=None,
            fallback=ExplorationFallback.SOFT_EXISTING_BEHAVIOR,
            safe_denial=None,
            budget=budget,
            diagnostics={"reason": "no_safe_candidate"},
        )

    def _score_candidate(self, candidate: RuntimeCandidate) -> AggregateScoreResult:
        results: list[ScoreResult] = []
        for scorer in self.scorers:
            try:
                results.append(scorer(candidate))
            except Exception as exc:
                results.append(
                    ScoreResult.skipped_result(
                        scorer=getattr(scorer, "__name__", "runtime_scorer"),
                        reason=f"scorer_error:{type(exc).__name__}",
                    )
                )
        return aggregate_scores(results)

    def _generate_candidates_with_timeout(self, context: Mapping[str, Any], *, timeout_ms: int | None = None) -> Any:
        with self._generator_lock:
            if self._timed_out_generator_in_flight:
                raise _GeneratorBusyError("previous runtime explorer generator is still running")
            self._timed_out_generator_in_flight = True

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _run_generator() -> None:
            try:
                result_queue.put_nowait(("ok", self.candidate_generator(dict(context))))
            except Exception as exc:
                with _suppress_queue_full():
                    result_queue.put_nowait(("error", exc))
            finally:
                with self._generator_lock:
                    self._timed_out_generator_in_flight = False

        thread = threading.Thread(
            target=_run_generator,
            name="persona-runtime-explorer-generator",
            daemon=True,
        )
        thread.start()
        try:
            effective_timeout_ms = self.config.timeout_ms if timeout_ms is None else timeout_ms
            status, payload = result_queue.get(timeout=max(0.0, effective_timeout_ms / 1000.0))
        except queue.Empty as exc:
            with self._generator_lock:
                self._timed_out_generator_in_flight = thread.is_alive()
            raise TimeoutError("runtime explorer exceeded timeout budget") from exc
        if status == "error":
            raise payload
        return payload

    def _runtime_failure_result(
        self,
        *,
        started_at: float,
        provider_calls: int,
        error: Exception,
        reason: str,
    ) -> RuntimeExplorationResult:
        circuit_open = self._record_runtime_failure()
        return RuntimeExplorationResult(
            selected_candidate=None,
            fallback=ExplorationFallback.SOFT_EXISTING_BEHAVIOR,
            safe_denial=None,
            budget=RuntimeBudgetUsage(
                provider_calls=provider_calls,
                elapsed_ms=self._elapsed_ms(started_at),
            ),
            diagnostics={"reason": reason, "error_type": type(error).__name__},
            circuit_open=circuit_open,
        )

    def _record_runtime_failure(self) -> bool:
        self._consecutive_runtime_failures += 1
        threshold = max(1, self.config.circuit_breaker_failure_threshold)
        if self._consecutive_runtime_failures >= threshold:
            self._circuit_opened_at = self._clock()
            return True
        return False

    def _reset_runtime_failures(self) -> None:
        self._consecutive_runtime_failures = 0
        self._circuit_opened_at = None

    def _circuit_is_open(self, now: float) -> bool:
        if self._circuit_opened_at is None:
            return False
        if now - self._circuit_opened_at >= self.config.circuit_breaker_cooldown_seconds:
            self._reset_runtime_failures()
            return False
        return True

    def _elapsed_ms(self, started_at: float) -> int:
        return max(0, int((self._clock() - started_at) * 1000))


@dataclass
class _MutableBudget:
    provider_calls: int = 0
    candidates_considered: int = 0
    hard_prunes: int = 0
    soft_prunes: int = 0
    elapsed_ms: int = 0

    def freeze(self, *, elapsed_ms: int) -> RuntimeBudgetUsage:
        return RuntimeBudgetUsage(
            provider_calls=self.provider_calls,
            candidates_considered=self.candidates_considered,
            hard_prunes=self.hard_prunes,
            soft_prunes=self.soft_prunes,
            elapsed_ms=elapsed_ms,
        )


_DEFAULT_SCORERS: tuple[RuntimeScorer, ...] = (
    policy_score,
    tool_plan_score,
    persona_consistency_score,
    refusal_quality_score,
    usefulness_score,
    grounding_style_score,
)

_RISKY_PLAN_ACTIONS: frozenset[str] = frozenset(
    {
        "delete",
        "remove",
        "write",
        "update",
        "edit",
        "execute",
        "shell",
        "http_post",
    }
)


def _coerce_candidate_list(raw_candidates: Any) -> list[Any]:
    if raw_candidates is None:
        return []
    if isinstance(raw_candidates, Mapping):
        return [raw_candidates]
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, (str, bytes, bytearray)):
        return list(raw_candidates)
    return [raw_candidates]


def _candidate_with_derived_tool_plan(candidate: Any) -> Any:
    if not isinstance(candidate, Mapping):
        return candidate
    derived_tool_plan = _derive_tool_plan_from_candidate_plan(candidate.get("plan"))
    if not derived_tool_plan:
        return candidate
    enriched = dict(candidate)
    enriched["tool_plan"] = derived_tool_plan
    return enriched


def _derive_tool_plan_from_candidate_plan(plan: Any) -> dict[str, Any] | None:
    if not isinstance(plan, Mapping):
        return None
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes, bytearray)):
        return None

    first_action: str | None = None
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            continue
        action = _infer_step_action(raw_step)
        if not action:
            continue
        if action in _RISKY_PLAN_ACTIONS:
            return {"action": action, "authorized": False, "source": "candidate_plan"}
        if first_action is None:
            first_action = action

    if first_action is None:
        return None
    return {"action": first_action, "authorized": True, "source": "candidate_plan"}


def _infer_step_action(step: Mapping[str, Any]) -> str | None:
    step_type = str(step.get("step_type") or "").strip().casefold()
    tool = str(step.get("tool") or "").strip().casefold()
    if step_type in {"final_answer", "rag_query"}:
        return step_type
    normalized_tool = tool.replace(".", "_").replace("-", "_")
    for risk_action in _RISKY_PLAN_ACTIONS:
        if normalized_tool == risk_action or normalized_tool.startswith(f"{risk_action}_"):
            return risk_action
        if f"_{risk_action}" in normalized_tool or f"{risk_action}_" in normalized_tool:
            return risk_action
    return normalized_tool or step_type or None


def _run_runtime_pruners(candidate: Any, *, existing_signatures: set[str]) -> list[PruneDecision]:
    decisions = [malformed_candidate_pruner(candidate)]
    if decisions[-1].severity == PruneSeverity.HARD or not isinstance(candidate, Mapping):
        return decisions

    decisions.extend(
        [
            prompt_injection_pressure_pruner(candidate),
            persona_boundary_violation_pruner(candidate),
            unsafe_tool_plan_pruner(candidate),
        ]
    )
    duplicate_decision = duplicate_low_diversity_pruner(
        candidate,
        existing_signatures=existing_signatures,
    )
    decisions.append(duplicate_decision)
    signature = str(duplicate_decision.metadata.get("signature") or "").strip()
    if signature:
        existing_signatures.add(signature)
    return decisions


class _suppress_queue_full:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return exc_type is queue.Full


class _GeneratorBusyError(TimeoutError):
    pass


__all__ = [
    "CandidateGenerator",
    "ExplorationFallback",
    "PersonaRuntimeExplorer",
    "RuntimeBudgetUsage",
    "RuntimeCandidate",
    "RuntimeExplorationResult",
    "RuntimeExplorerConfig",
    "RuntimeScorer",
]
