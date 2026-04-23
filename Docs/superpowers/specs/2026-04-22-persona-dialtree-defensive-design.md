# Defensive Persona Dialogue Trees Design

**Date:** 2026-04-22

## Goal

Adapt the useful ideas from DialTree for the tldw_server persona and character systems as a defensive robustness and candidate-selection layer.

The system should improve persona/character behavior by exploring multiple safe dialogue or plan candidates, pruning unsafe or low-quality branches, and scoring complete trajectories for persona consistency, policy adherence, grounding, and usefulness.

This design covers both:

- Offline defensive evals for personas and characters.
- Runtime bounded exploration for live persona behavior.

## Source Paper

Paper reviewed:

- [Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks](https://arxiv.org/html/2510.02286v2)
- [OpenReview entry](https://openreview.net/forum?id=El37o7iBjX)

The paper introduces DialTree, a tree-search reinforcement-learning framework for multi-turn red-teaming. The relevant transferable ideas are:

- Dialogue tree rollout instead of single linear rollouts.
- Branch pruning for malformed, off-topic, or low-quality branches.
- Trajectory-level scoring rather than isolated turn scoring.
- Stable structure preservation during optimization.

The offensive RL and jailbreak-training parts of the paper are not part of this design.

## Current Project Context

The repo already has adjacent but not equivalent systems:

- `tldw_Server_API/app/core/Persona/` persists persona profiles, sessions, policy rules, state docs, memories, exemplars, and websocket interactions.
- `tldw_Server_API/app/core/Persona/exemplar_retrieval.py` performs deterministic persona exemplar selection.
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py` performs character-scoped exemplar selection with scoring, MMR diversification, safety gating, and token-budget packing.
- `tldw_Server_API/app/core/Persona/policy_evaluator.py` enforces persona/session/tool policy.
- `tldw_Server_API/app/api/v1/endpoints/persona.py` has the live persona websocket plan-confirm-act loop.
- `Docs/Product/Completed/Persona_Roleplay_PRD.md` defines persona roleplay exemplars, policy-first boundaries, and IOO/IOR/LCS diagnostics.

What is missing:

- No tree rollout over alternate multi-turn persona trajectories.
- No branch-level pruning for persona drift, prompt injection progression, or unsafe tool-plan evolution.
- No trajectory-level robustness score across several turns.
- No shared offline/runtime tree engine.
- No runtime candidate explorer that ranks multiple safe plans or responses before showing one.

## Recommended Approach

Build a shared defensive dialogue-tree core, then use it in both offline evals and runtime exploration.

Implementation should proceed in one design family:

1. Shared tree engine and pruner/scorer contracts.
2. Offline robustness eval harness.
3. Runtime shallow explorer behind feature flags.

This avoids duplicate systems and gives the runtime path a safer validation base.

## Non-Goals

- Do not implement offensive jailbreak training.
- Do not implement GRPO, DAPO, or model fine-tuning.
- Do not optimize for eliciting harmful content.
- Do not store a jailbreak strategy library in normal runtime data.
- Do not let tree scoring override existing persona policy enforcement.
- Do not auto-mutate persona prompts, memories, state docs, or exemplars based on eval failures.

## Architecture

Add a new defensive tree layer beside the existing persona/character systems.

### Core Modules

Proposed backend modules:

- `tldw_Server_API/app/core/Persona/dialogue_tree.py`
- `tldw_Server_API/app/core/Persona/dialogue_tree_pruners.py`
- `tldw_Server_API/app/core/Persona/dialogue_tree_scorers.py`
- `tldw_Server_API/app/core/Persona/dialogue_tree_traces.py`
- `tldw_Server_API/app/core/Persona/robustness_eval.py`
- `tldw_Server_API/app/core/Persona/runtime_explorer.py`

Names can be adjusted during planning, but the boundaries should remain clear.

### PersonaDialogueTree

Shared engine for:

- Root state creation.
- Candidate branch expansion.
- Branch pruning.
- Trajectory assembly.
- Scoring.
- Best-candidate selection.
- Trace serialization.

The engine should be model/provider agnostic. It accepts candidate generators and scorer/pruner callables.

### PersonaTreePruners

Pruners return structured decisions with:

- `pruned: bool`
- `severity: soft | hard`
- `reason_code`
- `reason`
- optional diagnostics

Initial pruners:

- Malformed candidate output.
- Off-topic or task-drift detection.
- Prompt-injection pressure.
- Persona-boundary violation.
- Unsafe or unauthorized tool plan.
- Duplicate or low-diversity branch.
- Token, depth, branch, and latency budget limits.
- Exemplar over-copying risk.

Hard prunes should block fallback when the candidate itself would violate policy or safety.

### PersonaTrajectoryScorers

Scorers return independent structured scores so the aggregate remains debuggable.

Initial scorers:

- Persona consistency.
- Character voice/style adherence.
- Policy adherence.
- Tool-plan safety.
- Memory/state grounding.
- Exemplar use quality.
- Refusal quality.
- User-goal usefulness.
- Drift resistance over multiple turns.

Scores should be numeric where useful but keep explanations and skipped reasons. Deterministic scorers should be the default baseline. LLM-judge scorers may be optional and explicitly marked as skipped when unavailable.

### PersonaRobustnessEval

Offline harness that runs tree evals against persona and character profiles.

It should support:

- Benign interaction scenarios.
- Persona-drift scenarios.
- Prompt-injection scenarios.
- Tool misuse scenarios.
- Boundary-pressure scenarios.
- Exemplar over-copying scenarios.
- Multi-turn escalation scenarios.

The eval harness should produce structured trace artifacts and summary reports, not automatic profile rewrites.

### PersonaRuntimeExplorer

Runtime adapter for live persona behavior.

It should:

- Run only behind feature flags.
- Use shallow trees and strict budgets.
- Generate candidate plans or responses.
- Apply the same pruners and scorers as offline evals.
- Select the highest-scoring safe candidate.
- Fall back to existing behavior when exploration is unavailable, times out, or all candidates are soft-pruned.
- Emit a safe denial when all candidates are hard-policy violations.

The initial runtime integration should target persona websocket plan proposal near the existing `_propose_plan(...)` path. Character chat runtime response exploration can follow after the persona path is stable.

## Data Flow

### Offline Eval Flow

1. Select a persona or character plus an eval suite.
2. Build a root state with profile metadata, policy snapshot, state docs, relevant memory/exemplar context, and scenario goal.
3. Expand candidate user/assistant/tool-plan branches using configured generators.
4. Apply pruners at each node.
5. Assemble surviving root-to-leaf trajectories.
6. Score trajectories.
7. Persist trace artifact and report summary metrics.
8. Surface failing trajectories and scorer breakdowns for human review.

### Runtime Flow

1. Persona websocket receives `user_message`.
2. Existing context resolution runs: session policy, memory, companion context, persona state docs, and exemplar prompt sections.
3. If runtime exploration is disabled, existing behavior continues unchanged.
4. If enabled, a shallow tree generates candidate plans or candidate assistant answers.
5. Candidates are pruned and scored.
6. The best safe candidate is emitted.
7. If exploration fails or exceeds budget, fallback behavior depends on failure type:
   - soft failure: existing `_propose_plan(...)`
   - hard policy/safety violation: safe denial or denied tool-plan event

### Trace Shape

Trace records should be portable JSON-like objects.

Required fields:

- `root`: persona/character ids, session mode, scenario, policy snapshot id, model/provider metadata, feature flags.
- `nodes`: node id, parent id, turn index, candidate source, visible dialogue summary, hidden scoring metadata, prune status.
- `edges`: action type, candidate text or summarized tool plan, generator metadata.
- `trajectory_scores`: per-scorer output and aggregate score.
- `decision`: selected node, fallback reason, or failure class.

Privacy constraints:

- Do not store API keys, secrets, auth headers, or raw credentials.
- Do not persist full external tool responses unless explicitly requested for a test fixture.
- Summarize tool results using existing retention-summary patterns.
- Keep adversarial eval traces separate from normal persona memory, exemplars, and chat history.

## Error Handling

Runtime handling must be conservative:

- Candidate generation failure falls back to existing single-path behavior.
- Scoring failure discards the affected candidate and records a skipped scorer reason.
- LLM judge unavailability falls back to deterministic scorers.
- Budget exhaustion stops exploration and falls back if no hard violation occurred.
- Hard policy/safety violation emits safe denial rather than fallback to a potentially unsafe path.
- Trace persistence failure should not break live chat; emit diagnostics and continue.

Offline handling:

- Eval failures should be reported as failed test cases, not thrown away.
- Judge/scorer errors should be represented as skipped or errored scorer entries.
- Missing optional model providers should skip provider-dependent evals.

## Safety Boundaries

- Runtime exploration never executes tools during candidate generation.
- Existing persona policy remains final authority before any tool call.
- Runtime exploration may reject, rank, or choose candidates; it may not grant capabilities.
- External LLM judges classify candidates but cannot authorize actions.
- Red-team fixtures remain isolated from user-facing memory, exemplars, and state docs.
- Harmful request text in fixtures should be minimized, category-labeled, and avoid procedural harmful details.

## Configuration

Initial feature flags and settings should be additive.

Candidate config keys:

- `PERSONA_DIALOGUE_TREE_EVAL_ENABLED`
- `PERSONA_RUNTIME_EXPLORER_ENABLED`
- `PERSONA_RUNTIME_EXPLORER_MAX_DEPTH`
- `PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING`
- `PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS`
- `PERSONA_RUNTIME_EXPLORER_MAX_TOKENS`
- `PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS`

Defaults:

- Offline eval flag may default off unless explicitly invoked.
- Runtime explorer must default off.
- Runtime depth should initially be `1`.
- Runtime branching should initially be small, for example `2`.
- Runtime timeout should be short enough to preserve websocket responsiveness.

## API And UI Surface

Initial implementation can be backend-first.

Optional follow-on API endpoints:

- `POST /api/v1/persona/evals/dialogue-tree/run`
- `GET /api/v1/persona/evals/dialogue-tree/runs/{run_id}`
- `GET /api/v1/persona/evals/dialogue-tree/runs/{run_id}/trace`

Runtime websocket additions:

- Optional diagnostic `notice` when runtime exploration is enabled, skipped, timed out, or falls back.
- Optional debug metadata for selected candidate id and scorer summary when requested by an authenticated developer/debug flag.

No UI is required for the first implementation beyond existing logs/API results. A later UI can show trace trees and scorer breakdowns.

## Testing Strategy

### Core Unit Tests

- Tree expansion respects max depth, max branching, candidate caps, and deterministic ordering.
- Pruners classify malformed candidates, off-topic drift, prompt injection pressure, unsafe tool plans, duplicate branches, and over-budget branches.
- Scorers return stable structured outputs for normal and malformed inputs.
- Aggregate scoring handles skipped/failed scorer results.
- Trace serialization redacts or summarizes sensitive fields.

### Offline Eval Tests

- Benign scenario produces at least one passing trajectory.
- Persona-drift scenario fails persona-consistency scoring.
- Prompt-injection scenario is pruned or scored below threshold.
- Risky tool-plan scenario is denied by policy scoring.
- Reports include failing trajectories, scorer breakdowns, and skipped scorer reasons.

### Runtime Tests

- Feature flag off preserves persona websocket behavior.
- Feature flag on emits the highest-scoring safe plan when candidates are available.
- Candidate generation timeout falls back to existing `_propose_plan(...)`.
- All hard-policy candidates produce safe denial rather than fallback.
- Existing confirmation semantics are unchanged for write/export/delete tools.

### Verification

Use targeted pytest suites:

- `tldw_Server_API/tests/Persona/`
- relevant `tldw_Server_API/tests/Character_Chat/` tests when character integration is touched

Run Bandit on touched backend code before implementation completion.

## Rollout

Stage 1: Shared Core And Offline Harness

- Add tree, pruner, scorer, and trace contracts.
- Add deterministic initial pruners/scorers.
- Add offline robustness eval harness.
- Add tests for benign, drift, injection, and unsafe-tool scenarios.

Stage 2: Runtime Persona Explorer

- Add shallow runtime adapter behind `PERSONA_RUNTIME_EXPLORER_ENABLED`.
- Integrate around persona websocket planning.
- Preserve fallback to existing behavior.
- Add telemetry/diagnostic notices.

Stage 3: Character Chat Integration

- Use the offline evaluator for character/persona prompt robustness checks.
- Optionally apply runtime response exploration to character chat after persona runtime proves stable.

Stage 4: Trace Review UI

- Add optional UI to inspect tree traces, pruned branches, and scorer breakdowns.

## Acceptance Criteria

- Offline eval can run multi-turn defensive tree scenarios for at least one persona and one character.
- Eval traces include branch/prune/score decisions and redact sensitive data.
- Runtime explorer can be enabled without changing behavior when disabled.
- Runtime explorer falls back safely on timeout or soft failure.
- Hard policy/safety violations are not routed through unsafe fallback.
- Existing persona policy confirmation behavior remains unchanged.
- Tests cover core tree behavior, eval reports, and websocket runtime fallback.

## Open Questions

- Should trace artifacts live in ChaChaNotes DB, the existing eval DB, or filesystem-backed run artifacts?
- Which deterministic scorer thresholds should block runtime candidates versus only lowering score?
- Should LLM judge scorers be allowed in runtime at all, or reserved for offline eval only?
- What is the preferred first eval suite: persona drift, prompt injection, unsafe tool plans, or all three in a small smoke set?
