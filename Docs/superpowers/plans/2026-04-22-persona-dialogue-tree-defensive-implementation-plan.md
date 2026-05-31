# Defensive Persona Dialogue Trees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a defensive DialTree-inspired persona/character robustness layer with shared tree primitives, offline Evaluations/Jobs integration, and an opt-in bounded runtime persona explorer.

**Architecture:** Implement a model/provider-agnostic tree core under `tldw_Server_API/app/core/Persona/`, then expose offline runs as a built-in Evaluations recipe backed by existing run records and Jobs patterns. Runtime integration comes last, behind feature flags, with context minimization, deterministic hard gates, strict provider-call budgets, and existing persona policy as final authority.

**Tech Stack:** Python, FastAPI, Pydantic/dataclasses, Loguru, pytest, Hypothesis where available, existing Evaluations DB/recipe infrastructure, existing Jobs manager/worker patterns.

---

## Scope And Ordering

Do not start with the websocket runtime path. The safe order is:

1. Build deterministic core contracts and tests.
2. Add redaction/context minimization before any provider-facing code.
3. Add deterministic pruners/scorers/traces.
4. Add offline robustness service and Evaluations recipe/Jobs integration.
5. Wire config and operational guards.
6. Add the runtime persona explorer around `_propose_plan(...)`.
7. Add character-chat offline coverage; defer character runtime response exploration unless the persona runtime path is stable.

## File Structure

- Create `tldw_Server_API/app/core/Persona/dialogue_tree.py`: immutable-ish tree inputs/outputs, expansion loop, trajectory assembly, and best-candidate selection.
- Create `tldw_Server_API/app/core/Persona/dialogue_tree_context.py`: context bundle types, redaction, token/text caps, and per-consumer filtered views.
- Create `tldw_Server_API/app/core/Persona/dialogue_tree_pruners.py`: deterministic pruner contracts and initial hard/soft pruners.
- Create `tldw_Server_API/app/core/Persona/dialogue_tree_scorers.py`: deterministic scorer contracts and aggregate score calculation.
- Create `tldw_Server_API/app/core/Persona/dialogue_tree_traces.py`: portable redacted trace serialization.
- Create `tldw_Server_API/app/core/Persona/robustness_eval.py`: offline robustness harness used by tests and recipe execution.
- Create `tldw_Server_API/app/core/Persona/runtime_explorer.py`: shallow runtime adapter, budget accounting, circuit breaker, and fallback classification.
- Create `tldw_Server_API/app/core/Evaluations/recipes/persona_dialogue_tree_robustness.py`: built-in Evaluations recipe manifest, dataset validation, and report builder.
- Modify `tldw_Server_API/app/core/Evaluations/recipes/registry.py`: register the new recipe.
- Modify `tldw_Server_API/app/core/Evaluations/recipe_runs_jobs_worker.py`: dispatch the new recipe to `PersonaRobustnessEval` when Jobs executes recipe runs.
- Modify `tldw_Server_API/app/core/config.py`: load new `PERSONA_DIALOGUE_TREE_*` and `PERSONA_RUNTIME_EXPLORER_*` settings.
- Modify `tldw_Server_API/Config_Files/config.txt`: document default config keys and env override names.
- Modify `tldw_Server_API/app/api/v1/endpoints/persona.py`: add runtime explorer call only after core/offline/config work is passing.
- Add tests under `tldw_Server_API/tests/Persona/` and `tldw_Server_API/tests/Evaluations/`.

## Task 1: Shared Dialogue Tree Core

**Files:**
- Create: `tldw_Server_API/app/core/Persona/dialogue_tree.py`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree.py`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree_properties.py`

- [ ] **Step 1: Write failing unit tests for bounded expansion**

```python
def test_tree_expansion_respects_depth_branching_and_order():
    from tldw_Server_API.app.core.Persona.dialogue_tree import (
        DialogueTreeBudget,
        DialogueTreeEngine,
        TreeCandidate,
    )

    def generator(node):
        return [
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-b"),
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-a"),
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-c"),
        ]

    engine = DialogueTreeEngine(
        budget=DialogueTreeBudget(max_depth=2, max_branching=2, max_candidates=10),
        generators=[generator],
    )
    result = engine.expand(root_payload={"scenario": "benign"})

    assert result.max_depth_seen == 2
    assert all(len(result.children_by_parent[parent_id]) <= 2 for parent_id in result.children_by_parent)
    assert [node.candidate.text for node in result.nodes[1:3]] == ["root-a", "root-b"]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree.py::test_tree_expansion_respects_depth_branching_and_order -v`

Expected: FAIL with import error for `dialogue_tree`.

- [ ] **Step 3: Implement minimal tree contracts**

Add dataclasses or Pydantic models for `TreeCandidate`, `DialogueTreeBudget`, `DialogueTreeNode`, `DialogueTreeResult`, and `DialogueTreeEngine`. Keep the first pass deterministic:

```python
@dataclass(frozen=True)
class DialogueTreeBudget:
    max_depth: int = 1
    max_branching: int = 2
    max_candidates: int = 16
    max_provider_calls: int = 1

@dataclass(frozen=True)
class TreeCandidate:
    action_type: str
    text: str = ""
    tool_plan: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run the unit test and iterate to green**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree.py -v`

Expected: PASS.

- [ ] **Step 5: Add property tests for caps and acyclic traces**

Use Hypothesis if already installed. If unavailable in the active environment, write deterministic parametrized tests and note Hypothesis as a follow-up dependency decision rather than adding a new dependency.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_properties.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Persona/dialogue_tree.py \
  tldw_Server_API/tests/Persona/test_dialogue_tree.py \
  tldw_Server_API/tests/Persona/test_dialogue_tree_properties.py
git commit -m "feat: add persona dialogue tree core"
```

## Task 2: Context Filtering And Redaction

**Files:**
- Create: `tldw_Server_API/app/core/Persona/dialogue_tree_context.py`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree_context.py`

- [ ] **Step 1: Write failing tests for redaction before model calls**

```python
def test_runtime_context_redacts_secret_like_values_before_provider_view():
    from tldw_Server_API.app.core.Persona.dialogue_tree_context import build_runtime_tree_context

    context = build_runtime_tree_context(
        persona_id="p1",
        session_id="s1",
        user_message="hello",
        policy_snapshot={"allow": ["chat"], "authorization": "Bearer secret-token"},
        memory_entries=[{"id": "m1", "content": "safe note", "api_key": "sk-test"}],
        state_docs=[{"id": "doc1", "content": "state text"}],
        exemplar_sections=[("persona_exemplars", "style anchor", 12)],
        tool_results=[{"tool": "web", "raw": "private external response"}],
    )

    provider_payload = context.for_generator()
    serialized = repr(provider_payload)
    assert "secret-token" not in serialized
    assert "sk-test" not in serialized
    assert "private external response" not in serialized
    assert "omitted_context_categories" in provider_payload["metadata"]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_context.py::test_runtime_context_redacts_secret_like_values_before_provider_view -v`

Expected: FAIL with import error.

- [ ] **Step 3: Implement context bundles**

Implement:

- `PersonaTreeContext`
- `build_runtime_tree_context(...)`
- `build_offline_tree_context(...)`
- `redact_sensitive_payload(...)`
- `truncate_text_fields(...)`

Use allowlist-style provider bundles. Runtime generator views should contain summaries/ids and bounded text only.

- [ ] **Step 4: Add nested-payload redaction coverage**

Test nested dict/list payloads, case-insensitive keys such as `authorization`, `api_key`, `token`, `password`, and oversized content truncation.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_context.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Persona/dialogue_tree_context.py \
  tldw_Server_API/tests/Persona/test_dialogue_tree_context.py
git commit -m "feat: add persona dialogue tree context redaction"
```

## Task 3: Pruners, Scorers, And Trace Serialization

**Files:**
- Create: `tldw_Server_API/app/core/Persona/dialogue_tree_pruners.py`
- Create: `tldw_Server_API/app/core/Persona/dialogue_tree_scorers.py`
- Create: `tldw_Server_API/app/core/Persona/dialogue_tree_traces.py`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree_pruners.py`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree_scorers.py`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree_traces.py`

- [ ] **Step 1: Write failing tests for hard versus soft runtime semantics**

```python
def test_runtime_hard_prunes_are_deterministic_policy_only():
    from tldw_Server_API.app.core.Persona.dialogue_tree_pruners import (
        PruneSeverity,
        unsafe_tool_plan_pruner,
        llm_judge_warning_pruner,
    )

    hard = unsafe_tool_plan_pruner({"tool_plan": {"action": "delete", "authorized": False}})
    soft = llm_judge_warning_pruner({"judge_label": "low_quality"})

    assert hard.severity == PruneSeverity.HARD
    assert soft.severity == PruneSeverity.SOFT
```

- [ ] **Step 2: Run pruner tests and verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_pruners.py -v`

Expected: FAIL with import error.

- [ ] **Step 3: Implement pruner contracts and initial pruners**

Implement `PruneDecision`, `PruneSeverity`, `PruneReason`, and deterministic pruners for malformed output, prompt-injection pressure, persona-boundary violation, unauthorized tool plan, duplicate/low-diversity branch, and budget overflow.

- [ ] **Step 4: Write and implement scorer tests**

Cover deterministic score outputs, skipped scorer reasons, aggregate score ordering, and refusal-quality scoring.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_scorers.py -v`

Expected: PASS after implementing `ScoreResult`, `ScoreSeverity`, `aggregate_scores`, and initial deterministic scorers.

- [ ] **Step 5: Write and implement trace redaction tests**

Trace tests must assert no secret-like values, no raw external tool responses, stable node ordering, and inclusion of prune/scorer diagnostics.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_traces.py -v`

Expected: PASS.

- [ ] **Step 6: Run all core persona tree tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree*.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Persona/dialogue_tree_pruners.py \
  tldw_Server_API/app/core/Persona/dialogue_tree_scorers.py \
  tldw_Server_API/app/core/Persona/dialogue_tree_traces.py \
  tldw_Server_API/tests/Persona/test_dialogue_tree_pruners.py \
  tldw_Server_API/tests/Persona/test_dialogue_tree_scorers.py \
  tldw_Server_API/tests/Persona/test_dialogue_tree_traces.py
git commit -m "feat: add persona dialogue tree pruning and scoring"
```

## Task 4: Offline Robustness Harness

**Files:**
- Create: `tldw_Server_API/app/core/Persona/robustness_eval.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_dialogue_tree_robustness_eval.py`

- [ ] **Step 1: Write failing tests for smoke eval suites**

```python
def test_robustness_eval_reports_benign_drift_injection_and_tool_cases():
    from tldw_Server_API.app.core.Persona.robustness_eval import (
        PersonaRobustnessEval,
        build_default_smoke_suite,
    )

    evaluator = PersonaRobustnessEval()
    report = evaluator.run_suite(
        persona={"id": "p1", "name": "Research Assistant"},
        character=None,
        suite=build_default_smoke_suite(),
    )

    assert {case.case_id for case in report.cases} >= {
        "benign_basic",
        "persona_drift_boundary",
        "prompt_injection_policy_override",
        "unsafe_tool_plan",
    }
    assert report.summary["total_cases"] >= 4
    assert "trace_artifacts" in report.model_dump(mode="json")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_dialogue_tree_robustness_eval.py -v`

Expected: FAIL with import error.

- [ ] **Step 3: Implement local harness**

Implement fixture-driven local eval execution with deterministic candidate generators. Do not call external LLM providers in the initial harness.

- [ ] **Step 4: Add negative persistence tests**

Assert the harness returns trace/report payloads only and does not call persona memory, exemplar, state-doc, or chat-history write APIs.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_dialogue_tree_robustness_eval.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Persona/robustness_eval.py \
  tldw_Server_API/tests/Persona/test_persona_dialogue_tree_robustness_eval.py
git commit -m "feat: add persona dialogue tree robustness harness"
```

## Task 5: Evaluations Recipe And Jobs Integration

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/recipes/persona_dialogue_tree_robustness.py`
- Modify: `tldw_Server_API/app/core/Evaluations/recipes/registry.py`
- Modify: `tldw_Server_API/app/core/Evaluations/recipe_runs_jobs_worker.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe_jobs_worker.py`

- [ ] **Step 1: Write failing recipe registry test**

```python
def test_persona_dialogue_tree_recipe_is_registered():
    from tldw_Server_API.app.core.Evaluations.recipes.registry import get_builtin_recipe_registry

    manifest = get_builtin_recipe_registry().get_manifest("persona_dialogue_tree_robustness")

    assert manifest.recipe_id == "persona_dialogue_tree_robustness"
    assert "persona" in manifest.tags
    assert "robustness" in manifest.tags
```

- [ ] **Step 2: Run recipe tests and verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py -v`

Expected: FAIL because the recipe is not registered.

- [ ] **Step 3: Implement recipe manifest and validation**

Create `PersonaDialogueTreeRobustnessRecipe` with:

- `recipe_id="persona_dialogue_tree_robustness"`
- `supported_modes=["labeled", "unlabeled"]`
- validation requiring at least one persona or character target and at least one scenario
- a `build_report(...)` method that summarizes case counts, hard prunes, soft prunes, selected trajectories, skipped scorers, and trace artifact refs

- [ ] **Step 4: Register recipe**

Modify `_default_builtin_recipes()` in `registry.py` to include `PersonaDialogueTreeRobustnessRecipe()`.

- [ ] **Step 5: Write failing Jobs worker dispatch test**

Mirror the style in `tldw_Server_API/tests/Evaluations/test_recipe_runs_jobs_worker.py`. Create a pending recipe run with `recipe_id="persona_dialogue_tree_robustness"`, invoke the worker handler with a fake/mocked service or temp Evaluations DB, and assert the run completes with robustness report payload.

- [ ] **Step 6: Implement Jobs worker dispatch**

In `recipe_runs_jobs_worker.py`, add a narrow branch for the new recipe that calls `PersonaRobustnessEval`. Reuse existing status update, failure, cancellation, and metadata conventions. Do not create a persona-specific run queue.

- [ ] **Step 7: Run Evaluations recipe tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe_jobs_worker.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/recipes/persona_dialogue_tree_robustness.py \
  tldw_Server_API/app/core/Evaluations/recipes/registry.py \
  tldw_Server_API/app/core/Evaluations/recipe_runs_jobs_worker.py \
  tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py \
  tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe_jobs_worker.py
git commit -m "feat: register persona dialogue tree evaluation recipe"
```

## Task 6: Configuration And Operational Guards

**Files:**
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Test: `tldw_Server_API/tests/Persona/test_dialogue_tree_config.py`

- [ ] **Step 1: Write failing config tests**

```python
def test_persona_runtime_explorer_defaults_are_safe(monkeypatch):
    from tldw_Server_API.app.core.config import load_settings

    settings = load_settings()

    assert settings["PERSONA_RUNTIME_EXPLORER_ENABLED"] is False
    assert int(settings["PERSONA_RUNTIME_EXPLORER_MAX_DEPTH"]) == 1
    assert int(settings["PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING"]) == 2
    assert int(settings["PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS"]) == 1
```

- [ ] **Step 2: Run config test and verify it fails or exposes missing keys**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_config.py -v`

Expected: FAIL until config keys are wired consistently.

- [ ] **Step 3: Wire config keys**

Add additive persona settings for:

- `PERSONA_DIALOGUE_TREE_EVAL_ENABLED`
- `PERSONA_RUNTIME_EXPLORER_ENABLED`
- `PERSONA_RUNTIME_EXPLORER_MAX_DEPTH`
- `PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING`
- `PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS`
- `PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS`
- `PERSONA_RUNTIME_EXPLORER_MAX_TOKENS`
- `PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS`
- `PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED`
- `PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS`

- [ ] **Step 4: Document config examples**

Add commented defaults and env override notes to `[persona]` in `Config_Files/config.txt`. Runtime explorer must be documented as off by default.

- [ ] **Step 5: Run config tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/config.py \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/tests/Persona/test_dialogue_tree_config.py
git commit -m "feat: add persona dialogue tree configuration"
```

## Task 7: Runtime Persona Explorer

**Files:**
- Create: `tldw_Server_API/app/core/Persona/runtime_explorer.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Persona/test_runtime_explorer.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py`

- [x] **Step 1: Write failing runtime explorer unit tests**

```python
def test_runtime_explorer_soft_timeout_falls_back_without_hard_denial():
    from tldw_Server_API.app.core.Persona.runtime_explorer import (
        ExplorationFallback,
        PersonaRuntimeExplorer,
        RuntimeExplorerConfig,
    )

    explorer = PersonaRuntimeExplorer(
        config=RuntimeExplorerConfig(enabled=True, timeout_ms=1, max_provider_calls=1),
        candidate_generator=lambda context: (_ for _ in ()).throw(TimeoutError("slow")),
    )

    result = explorer.explore({"user_message": "hello"})

    assert result.fallback == ExplorationFallback.SOFT_EXISTING_BEHAVIOR
    assert result.selected_candidate is None
```

- [x] **Step 2: Run unit test and verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_runtime_explorer.py -v`

Expected: FAIL with import error.

- [x] **Step 3: Implement runtime explorer core**

Implement:

- `RuntimeExplorerConfig`
- `RuntimeBudgetUsage`
- `ExplorationFallback`
- `RuntimeExplorationResult`
- `PersonaRuntimeExplorer`
- in-process circuit breaker after three consecutive runtime failures

Keep the first generator deterministic/mockable. Do not add external LLM judge calls.

- [x] **Step 4: Add websocket integration tests with feature flag off**

Extend existing persona websocket test style from `test_persona_ws.py`. Assert disabled runtime explorer preserves existing `_propose_plan(...)` behavior and emits no candidate debug metadata.

- [x] **Step 5: Add websocket integration tests with feature flag on**

Use monkeypatched deterministic generator/scorer. Assert:

- highest-scoring safe plan is selected
- hard-policy candidate returns safe denial
- soft timeout falls back to `_propose_plan(...)`
- write/export/delete confirmation semantics are unchanged

- [x] **Step 6: Wire runtime adapter into persona websocket**

Integrate near the existing `_propose_plan(...)` path. The adapter must receive the filtered runtime context bundle, not raw unresolved DB rows or raw tool responses. Final selected plan must still pass existing policy evaluation before emission.

- [x] **Step 7: Run runtime tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_runtime_explorer.py tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py -v`

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Persona/runtime_explorer.py \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/tests/Persona/test_runtime_explorer.py \
  tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py
git commit -m "feat: add opt-in persona runtime explorer"
```

## Task 8: Character Chat Offline Coverage

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/robustness_eval.py`
- Test: `tldw_Server_API/tests/Character_Chat/test_persona_dialogue_tree_character_eval.py`

- [x] **Step 1: Write failing character-target eval test**

```python
def test_robustness_eval_accepts_character_target_without_runtime_hook():
    from tldw_Server_API.app.core.Persona.robustness_eval import (
        PersonaRobustnessEval,
        build_default_smoke_suite,
    )

    report = PersonaRobustnessEval().run_suite(
        persona=None,
        character={"id": "char-1", "name": "Archivist", "persona": "careful researcher"},
        suite=build_default_smoke_suite(),
    )

    assert report.target_type == "character"
    assert report.summary["total_cases"] >= 4
```

- [x] **Step 2: Run test and verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_persona_dialogue_tree_character_eval.py -v`

Expected: FAIL until the harness accepts character targets.

- [x] **Step 3: Implement character target normalization**

Reuse existing deterministic character/persona exemplar selectors only for offline context snapshots. Do not integrate runtime response exploration into `/complete-v2` in this task.

- [x] **Step 4: Run character eval test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_persona_dialogue_tree_character_eval.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Persona/robustness_eval.py \
  tldw_Server_API/tests/Character_Chat/test_persona_dialogue_tree_character_eval.py
git commit -m "feat: add character dialogue tree robustness eval coverage"
```

## Task 9: Documentation, Security, And Final Verification

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/README.md`
- Modify: `tldw_Server_API/app/core/Evaluations/README.md`
- Optional modify: `Docs/Product/Completed/Persona_Roleplay_PRD.md`

- [x] **Step 1: Update docs**

Document:

- DialTree adaptation is defensive only.
- Offline runs use Evaluations and Jobs.
- Runtime explorer is off by default.
- LLM judges are offline-only by default and cannot authorize actions.
- Red-team fixtures never write into persona memory, exemplars, state docs, or chat history.

- [x] **Step 2: Run focused test suites**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_dialogue_tree*.py tldw_Server_API/tests/Persona/test_persona_dialogue_tree_robustness_eval.py tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe.py tldw_Server_API/tests/Evaluations/test_persona_dialogue_tree_recipe_jobs_worker.py tldw_Server_API/tests/Character_Chat/test_persona_dialogue_tree_character_eval.py -v`

Expected: PASS.

- [x] **Step 3: Run websocket-focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_ws.py tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py -v`

Expected: PASS.

- [x] **Step 4: Run Bandit on touched backend code**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Persona tldw_Server_API/app/core/Evaluations/recipes/persona_dialogue_tree_robustness.py -f json -o /tmp/bandit_persona_dialogue_tree.json`

Expected: exit code 0 or only pre-existing non-blocking findings outside changed code. Fix new findings before continuing.

- [x] **Step 5: Run diff check**

Run: `git diff --check`

Expected: no whitespace errors.

- [x] **Step 6: Commit docs and final cleanup**

```bash
git add tldw_Server_API/app/core/Persona/README.md \
  tldw_Server_API/app/core/Evaluations/README.md \
  Docs/Product/Completed/Persona_Roleplay_PRD.md
git commit -m "docs: document persona dialogue tree robustness"
```

## Risk Controls

- Runtime explorer remains disabled by default until tests prove no behavioral change when off.
- Runtime explorer adds at most one provider call per user turn by default.
- Runtime LLM judges remain off; deterministic policy/tool pruners are the only runtime hard gates.
- All runtime selected candidates still pass existing persona policy and confirmation checks.
- Offline red-team fixtures are category-labeled, minimized, and isolated from persona memory/exemplar/state/chat stores.
- No new persona-specific eval run queue should be introduced; use Evaluations recipe runs and Jobs.

## Review Notes

The plan intentionally does not include a runtime Character Chat `/complete-v2` response explorer. Character support starts as offline robustness coverage only, because Character Chat has separate completion paths and should not inherit websocket runtime behavior without a second design review.
