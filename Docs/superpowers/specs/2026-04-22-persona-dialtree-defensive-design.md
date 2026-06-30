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
- The Evaluations module already provides evaluation definitions, run records, recipes, result history, and Jobs-backed long-running execution through `tldw_Server_API/app/core/Evaluations/` and `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py`.
- Project guidance says user-visible long-running work should use Jobs rather than inventing a parallel queue/status system.

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
- `tldw_Server_API/app/core/Persona/dialogue_tree_context.py`
- `tldw_Server_API/app/core/Persona/dialogue_tree_pruners.py`
- `tldw_Server_API/app/core/Persona/dialogue_tree_scorers.py`
- `tldw_Server_API/app/core/Persona/dialogue_tree_traces.py`
- `tldw_Server_API/app/core/Persona/robustness_eval.py`
- `tldw_Server_API/app/core/Persona/runtime_explorer.py`
- `tldw_Server_API/app/core/Evaluations/recipes/persona_dialogue_tree_robustness.py`

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

### PersonaDialogueTreeContext

Shared context filtering and redaction layer for offline and runtime candidate generation.

It should:

- Accept resolved persona/character context from existing services.
- Produce per-consumer context bundles for generators, pruners, scorers, judges, and traces.
- Strip API keys, credentials, auth headers, private tool outputs, and raw external responses before any model call.
- Bound memory, state-doc, companion-context, and exemplar text independently before assembly.
- Preserve only stable ids, summaries, category labels, and explicit safe excerpts when full text is unnecessary.
- Treat runtime model calls as stricter than offline eval calls.

Candidate generators and LLM judges must declare which filtered bundle they consume. The default runtime generator receives only the same or less sensitive context than the current single-path `_propose_plan(...)` prompt.

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

Runtime hard prunes are restricted to deterministic gates:

- Existing persona policy or RBAC denial.
- Unsafe or unauthorized tool plan.
- Prompt-injection instruction that attempts to override system, policy, or tool authority.
- Malformed candidate that cannot be safely parsed.
- Redaction failure before a provider call.

LLM judges may never create a runtime hard allow or runtime hard deny. They can only add soft-prune decisions, warnings, or ranking inputs.

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

Runtime scoring must keep deterministic policy and tool-safety scorers authoritative. LLM-judge scorers are disabled in runtime by default and, when explicitly enabled later, may only lower rank or emit diagnostics. They cannot authorize a tool call, bypass confirmation, or override a deterministic policy denial.

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

The harness should integrate with the existing Evaluations architecture:

- Register as a built-in evaluation recipe, initially `persona_dialogue_tree_robustness`.
- Persist run status and summary results in the Evaluations DB.
- Store full redacted trace payloads as run artifacts or results payloads attached to the evaluation run, not in ChaChaNotes, persona memory, exemplars, or chat history.
- Use Jobs for async user-visible runs and reuse existing recipe-run status, idempotency, cancellation, and history patterns.
- Avoid new top-level persona eval run APIs unless they are thin wrappers around the unified Evaluations endpoints.

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
- Track added latency, provider-call count, token budget, and fallback reason for every explored turn.
- Open a circuit breaker for runtime exploration after repeated timeout/provider failures and continue with the existing single-path behavior while open.

The initial runtime integration should target persona websocket plan proposal near the existing `_propose_plan(...)` path. Character chat runtime response exploration can follow after the persona path is stable.

## Data Flow

### Offline Eval Flow

1. Select a persona or character plus an eval suite.
2. Build a root state with profile metadata, policy snapshot, state docs, relevant memory/exemplar context, and scenario goal.
3. Create or reuse an Evaluations recipe run and enqueue it through Jobs when run asynchronously.
4. Filter the root state through `PersonaDialogueTreeContext` before any generator, scorer, judge, or trace serialization step.
5. Expand candidate user/assistant/tool-plan branches using configured generators.
6. Apply pruners at each node.
7. Assemble surviving root-to-leaf trajectories.
8. Score trajectories.
9. Persist redacted trace artifacts and summary metrics to the Evaluations run record or artifact location.
10. Surface failing trajectories and scorer breakdowns through existing Evaluations run history.

### Runtime Flow

1. Persona websocket receives `user_message`.
2. Existing context resolution runs: session policy, memory, companion context, persona state docs, and exemplar prompt sections.
3. If runtime exploration is disabled, existing behavior continues unchanged.
4. If enabled, `PersonaDialogueTreeContext` builds a runtime-safe filtered bundle before any additional provider call.
5. A shallow tree generates candidate plans or candidate assistant answers within max depth, branching, provider-call, token, and timeout budgets.
6. Candidates are pruned and scored.
7. The best safe candidate is passed through the existing final policy/confirmation path before it is emitted.
8. If exploration fails or exceeds budget, fallback behavior depends on failure type:
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
- Apply the same redaction rules before model calls, not only before persistence.
- Record secret-redaction counts and omitted-context categories without recording the omitted values.

## Error Handling

Runtime handling must be conservative:

- Candidate generation failure falls back to existing single-path behavior.
- Scoring failure discards the affected candidate and records a skipped scorer reason.
- LLM judge unavailability falls back to deterministic scorers.
- Budget exhaustion stops exploration and falls back if no hard violation occurred.
- Hard policy/safety violation emits safe denial rather than fallback to a potentially unsafe path.
- Trace persistence failure should not break live chat; emit diagnostics and continue.
- Runtime exploration opens a circuit breaker after repeated provider failures or timeouts and remains disabled for a short cooldown.
- Fallback output still goes through the existing persona policy and confirmation checks.

Offline handling:

- Eval failures should be reported as failed test cases, not thrown away.
- Judge/scorer errors should be represented as skipped or errored scorer entries.
- Missing optional model providers should skip provider-dependent evals.
- Jobs enqueue failure should mark the Evaluations run failed instead of leaving it pending.

## Safety Boundaries

- Runtime exploration never executes tools during candidate generation.
- Existing persona policy remains final authority before any tool call.
- Runtime exploration may reject, rank, or choose candidates; it may not grant capabilities.
- External LLM judges classify candidates but cannot authorize actions.
- Red-team fixtures remain isolated from user-facing memory, exemplars, and state docs.
- Harmful request text in fixtures should be minimized, category-labeled, and avoid procedural harmful details.
- Context minimization is required before any runtime candidate-generation or judge provider call.
- Candidate generation can propose summarized tool plans, but execution remains exclusively in the existing plan-confirm-act path.

## Configuration

Initial feature flags and settings should be additive.

Candidate config keys:

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

Defaults:

- Offline eval flag may default off unless explicitly invoked.
- Runtime explorer must default off.
- Runtime depth should initially be `1`.
- Runtime branching should initially be small, for example `2`.
- Runtime provider-call budget should initially be `1` additional provider call per user turn.
- Runtime timeout should initially be no more than `750ms` added wall-clock time in mocked tests.
- Runtime p95 added latency target should initially be no more than `1000ms` in integration-style tests with mocked providers.
- Runtime LLM judges should default off.
- The runtime circuit breaker should open after three consecutive runtime exploration failures in one process.

Implementation must wire these settings through the existing persona config loader in `tldw_Server_API/app/core/config.py`, document them in `tldw_Server_API/Config_Files/config.txt`, and expose debug/capability metadata only where existing authenticated config/status surfaces already support it.

## API And UI Surface

Initial implementation can be backend-first.

Primary API surface should reuse the existing Evaluations endpoints and recipe-run patterns:

- Register `persona_dialogue_tree_robustness` as an Evaluations recipe.
- Create, enqueue, inspect, cancel, and list runs through existing `/api/v1/evaluations/...` recipe/run endpoints.
- Persist summary results and redacted traces in Evaluations run storage or artifact references.

Optional follow-on persona endpoints may be added only as thin convenience wrappers over Evaluations:

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
- Context filtering removes secrets, raw credentials, raw external tool responses, and oversized context before model calls.

### Property And Fuzz Tests

- Tree generation never exceeds configured depth, branch, candidate, token, or provider-call caps.
- Serialized tree traces contain no cycles and preserve deterministic node ordering.
- Redaction is idempotent and removes known secret-key patterns from arbitrary nested payloads.
- Adversarial fixture text is never written to persona memory, exemplars, state docs, or chat history.
- Fallback classification remains deterministic for hard versus soft prune combinations.

### Offline Eval Tests

- Benign scenario produces at least one passing trajectory.
- Persona-drift scenario fails persona-consistency scoring.
- Prompt-injection scenario is pruned or scored below threshold.
- Risky tool-plan scenario is denied by policy scoring.
- Reports include failing trajectories, scorer breakdowns, and skipped scorer reasons.
- Recipe run records are created in the Evaluations DB and queued through Jobs for async runs.
- Jobs enqueue failure marks the run failed with a diagnostic reason.

### Runtime Tests

- Feature flag off preserves persona websocket behavior.
- Feature flag on emits the highest-scoring safe plan when candidates are available.
- Candidate generation timeout falls back to existing `_propose_plan(...)`.
- All hard-policy candidates produce safe denial rather than fallback.
- Existing confirmation semantics are unchanged for write/export/delete tools.
- Runtime exploration uses no more than the configured additional provider-call budget.
- Mocked slow providers trigger timeout fallback and circuit-breaker behavior.
- Runtime LLM judges are skipped by default.

### Verification

Use targeted pytest suites:

- `tldw_Server_API/tests/Persona/`
- relevant `tldw_Server_API/tests/Character_Chat/` tests when character integration is touched

Run Bandit on touched backend code before implementation completion.

## Rollout

Stage 1: Shared Core And Offline Harness

- Add tree, pruner, scorer, and trace contracts.
- Add context filtering and redaction contracts.
- Add deterministic initial pruners/scorers.
- Add offline robustness eval harness as a local service.
- Add tests for benign, drift, injection, and unsafe-tool scenarios.

Stage 2: Evaluations And Jobs Integration

- Register `persona_dialogue_tree_robustness` as a built-in Evaluations recipe.
- Persist summaries and redacted trace artifacts through Evaluations run records.
- Queue async runs through Jobs using existing recipe-run patterns.
- Add API tests against existing Evaluations recipe/run endpoints.

Stage 3: Configuration And Operational Guards

- Wire feature flags and budgets through `config.py`.
- Add `config.txt` examples and env override notes.
- Add runtime budget, circuit-breaker, and provider-call accounting tests.

Stage 4: Runtime Persona Explorer

- Add shallow runtime adapter behind `PERSONA_RUNTIME_EXPLORER_ENABLED`.
- Integrate around persona websocket planning.
- Preserve fallback to existing behavior.
- Add telemetry/diagnostic notices.

Stage 5: Character Chat Integration

- Use the offline evaluator for character/persona prompt robustness checks.
- Optionally apply runtime response exploration to character chat after persona runtime proves stable.

Stage 6: Trace Review UI

- Add optional UI to inspect tree traces, pruned branches, and scorer breakdowns.

## Acceptance Criteria

- Offline eval can run multi-turn defensive tree scenarios for at least one persona and one character.
- Eval traces include branch/prune/score decisions and redact sensitive data.
- Offline eval runs are discoverable through the existing Evaluations run history.
- Async offline evals use Jobs and do not create a separate persona-specific run queue.
- Runtime explorer can be enabled without changing behavior when disabled.
- Runtime explorer falls back safely on timeout or soft failure.
- Hard policy/safety violations are not routed through unsafe fallback.
- Runtime provider-call and latency budgets are enforced and tested.
- Runtime candidate generation and judge calls use minimized, redacted context.
- Existing persona policy confirmation behavior remains unchanged.
- Tests cover core tree behavior, eval reports, and websocket runtime fallback.

## Open Questions

- What filesystem or DB artifact layout should large redacted traces use once Evaluations run result payloads become too large?
- What exact deterministic score thresholds should the first smoke suite use after baseline fixtures are written?
