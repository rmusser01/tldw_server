from __future__ import annotations

import threading
import time

from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import ScoreResult, ScoreSeverity


def test_runtime_explorer_disabled_falls_back_without_generator_call() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    calls: list[dict] = []
    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=False),
        candidate_generator=lambda context: calls.append(dict(context)) or [],
    )

    result = explorer.explore({"user_message": "hello"})

    assert result.fallback == ExplorationFallback.DISABLED
    assert result.selected_candidate is None
    assert calls == []


def test_runtime_explorer_soft_timeout_falls_back_without_hard_denial() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    def _slow_generator(_context: dict) -> list[dict]:
        raise TimeoutError("slow")

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, timeout_ms=1, max_provider_calls=1),
        candidate_generator=_slow_generator,
    )

    result = explorer.explore({"user_message": "hello"})

    assert result.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert result.selected_candidate is None
    assert result.safe_denial is None
    assert result.budget.provider_calls == 1


def test_runtime_explorer_timeout_budget_interrupts_blocked_generator() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    release_generator = threading.Event()

    def _blocked_generator(_context: dict) -> list[dict]:
        release_generator.wait(timeout=1.0)
        return []

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, timeout_ms=10, max_provider_calls=1),
        candidate_generator=_blocked_generator,
    )

    started_at = time.monotonic()
    try:
        result = explorer.explore({"user_message": "hello"})
    finally:
        release_generator.set()
    elapsed_ms = int((time.monotonic() - started_at) * 1000)

    assert result.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert result.diagnostics["reason"] == "candidate_generation_timeout"
    assert elapsed_ms < 200


def test_runtime_explorer_does_not_spawn_new_generator_while_timeout_is_in_flight() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    release_generator = threading.Event()
    calls = 0

    def _blocked_generator(_context: dict) -> list[dict]:
        nonlocal calls
        calls += 1
        release_generator.wait(timeout=1.0)
        return []

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, timeout_ms=10, max_provider_calls=1),
        candidate_generator=_blocked_generator,
    )

    first = explorer.explore({"user_message": "one"})
    second = explorer.explore({"user_message": "two"})
    release_generator.set()

    assert first.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert first.diagnostics["reason"] == "candidate_generation_timeout"
    assert second.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert second.diagnostics["reason"] == "candidate_generation_busy"
    assert second.budget.provider_calls == 0
    assert calls == 1


def test_runtime_explorer_selects_highest_scoring_safe_candidate() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    def _score(candidate: dict) -> ScoreResult:
        return ScoreResult(
            scorer="test_score",
            score=float(candidate["metadata"]["score"]),
            severity=ScoreSeverity.PASS,
        )

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, max_branching=3, max_provider_calls=1),
        candidate_generator=lambda _context: [
            {
                "action_type": "plan",
                "text": "lower quality but safe answer with enough detail",
                "plan": {"steps": [{"idx": 0, "step_type": "final_answer", "tool": "summarize"}]},
                "metadata": {"score": 0.2},
            },
            {
                "action_type": "plan",
                "text": "highest quality safe answer with enough detail",
                "plan": {"steps": [{"idx": 0, "step_type": "rag_query", "tool": "rag_search"}]},
                "metadata": {"score": 0.9},
            },
        ],
        scorers=[_score],
    )

    result = explorer.explore({"user_message": "compare notes"})

    assert result.fallback is None
    assert result.selected_candidate is not None
    assert result.selected_candidate["plan"]["steps"][0]["tool"] == "rag_search"
    assert result.score is not None
    assert result.score.overall_score == 0.9


def test_runtime_explorer_prunes_unsafe_nested_plan_without_tool_plan_metadata() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, max_provider_calls=1),
        candidate_generator=lambda _context: [
            {
                "action_type": "plan",
                "text": "This looks benign but carries an unsafe nested step.",
                "tool_plan": {"action": "read", "authorized": True},
                "plan": {
                    "steps": [
                        {
                            "idx": 0,
                            "step_type": "mcp_tool",
                            "tool": "notes.delete",
                            "args": {"note_id": "n1"},
                        }
                    ]
                },
            }
        ],
    )

    result = explorer.explore({"user_message": "delete note"})

    assert result.fallback == ExplorationFallback.HARD_SAFE_DENIAL
    assert result.selected_candidate is None
    assert result.budget.hard_prunes == 1


def test_runtime_explorer_hard_policy_candidate_returns_safe_denial() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, max_provider_calls=1),
        candidate_generator=lambda _context: [
            {
                "action_type": "tool_plan",
                "text": "I will delete that for you.",
                "tool_plan": {"action": "delete", "authorized": False},
                "plan": {"steps": [{"idx": 0, "step_type": "mcp_tool", "tool": "notes.delete"}]},
            }
        ],
    )

    result = explorer.explore({"user_message": "delete everything"})

    assert result.fallback == ExplorationFallback.HARD_SAFE_DENIAL
    assert result.selected_candidate is None
    assert result.safe_denial is not None
    assert result.budget.hard_prunes == 1


def test_runtime_explorer_opens_circuit_after_three_runtime_failures() -> None:
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, max_provider_calls=1),
        candidate_generator=lambda _context: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )

    first = explorer.explore({"user_message": "one"})
    second = explorer.explore({"user_message": "two"})
    third = explorer.explore({"user_message": "three"})
    fourth = explorer.explore({"user_message": "four"})

    assert first.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert second.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert third.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert third.circuit_open is True
    assert fourth.fallback == ExplorationFallback.CIRCUIT_OPEN
    assert fourth.budget.provider_calls == 0
