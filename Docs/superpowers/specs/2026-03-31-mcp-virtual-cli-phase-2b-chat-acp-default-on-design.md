# MCP Virtual CLI Phase 2b Chat + ACP Default-On Design

Date: 2026-03-31
Status: Approved in conversation; written for user review
Owner: Codex brainstorming session

## Summary

Phase 2b promotes the existing `run(command)` run-first presentation from a gated experiment to the default agent-facing posture for the same stable `chat + ACP` cohort proven in phase 2a.

This is a rollout-posture change, not a runtime redesign.

The main change is operational posture:

- `run(command)` becomes the normal presentation for the stable cohort
- typed tools remain visible and executable as fallback
- the rollout control remains as a rollback and override mechanism
- the phase 2a telemetry contract stays in place with only a low-cardinality posture refinement
- out-of-cohort sessions do not widen automatically in this phase

The design preserves phase 1 runtime behavior, current governance, approval semantics, and typed fallback. It changes the default surface posture for a known-good cohort without changing the underlying execution model.

## Problem

Phase 2a introduced run-first presentation for `chat + ACP` behind a gated rollout and telemetry contract. That is the right first step, but it is still an experimental posture.

Once a stable provider and runtime cohort has been validated, continuing to treat run-first as only a gated experiment creates unnecessary operational drag:

- the default experience for the stable cohort remains split between experimental and control postures
- prompt and tool-surface behavior remain coupled to rollout enrollment rather than the normal product path
- rollback and cohort-ineligibility become harder to distinguish from intentional control traffic
- wider rollout decisions are delayed by the lack of a clean “default-on for proven cohort” state

Phase 2b exists to promote the successful phase 2a behavior into the normal `chat + ACP` experience for the already-proven cohort, while preserving a clean rollback path and keeping broader expansion out of scope.

## User-Approved Decisions

Validated during brainstorming:

1. Phase 2b stays on `chat + ACP`.
2. `run(command)` should become default-on only for the same stable provider and runtime cohort already proven in phase 2a.
3. Typed tools should remain visible and executable as fallback.
4. The telemetry contract should stay mostly the same; stronger judged outcome evaluation is a later slice.
5. Phase 2b should be a posture-promotion slice, not a cohort-expansion slice.
6. The recommended phase 2b approach is default-on for the proven cohort with a rollback gate still in place.

## Goals

- Make run-first presentation the normal `chat + ACP` posture for the already-proven stable cohort.
- Preserve typed-tool visibility and executability as fallback.
- Keep rollback and targeted disablement operationally simple.
- Reuse the phase 2a presenter, config, and telemetry seams instead of adding a new surface architecture.
- Maintain clear separation between stable default-on traffic and out-of-cohort traffic.

## Non-Goals

- Widening the provider/runtime cohort in this phase.
- Hiding typed tools completely.
- Adding new command families or changing the phase 1 `run(command)` runtime.
- Reworking governance, approvals, or execution semantics.
- Redesigning the telemetry model or adding judged outcome evaluation in this phase.
- Expanding run-first behavior to personas or workflows in this design.

## Current Repo Fit

Phase 2a seams already exist in this worktree, so phase 2b should be an incremental extension of those surfaces rather than a new rollout architecture.

### Chat already has a dedicated presenter and telemetry seam

Chat resolves run-first presentation in [run_first_presentation.py](tldw_Server_API/app/core/Chat/run_first_presentation.py) and applies it in [chat_service.py](tldw_Server_API/app/core/Chat/chat_service.py). Chat metrics already expose rollout, first-tool, fallback-after-run, and completion-proxy counters in [chat_metrics.py](tldw_Server_API/app/core/Chat/chat_metrics.py).

That means phase 2b does not need a new chat surface abstraction. It needs a clearer default posture model on top of the existing presenter and metric context.

### ACP already has a parallel presenter and metric context

ACP resolves session-aware run-first presentation in [mcp_tool_presentation.py](tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_tool_presentation.py), wires it through [mcp_adapter.py](tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_adapter.py), and emits matching telemetry through [mcp_runners.py](tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_runners.py) and [metrics.py](tldw_Server_API/app/core/Agent_Client_Protocol/metrics.py).

Phase 2b should reuse those seams and keep `chat` and ACP behavior symmetric.

### Config currently models only `off` and `gated`

The current rollout helpers in [config.py](tldw_Server_API/app/core/config.py) resolve run-first mode as `off` or `gated`, with a provider allowlist and presentation variant for each surface.

Phase 2b needs one explicit posture addition so the stable cohort can be modeled as normal default-on behavior rather than as perpetual experiment enrollment.

The operational distinction must also be explicit:

- resolver support expands to `off | gated | default_on`
- shipped `chat` and ACP surface defaults for the stable cohort move to `default_on`
- code-level safe fallback remains `off` when no config or profile default is present

That keeps the product posture distinct without turning missing configuration into an implicit enablement.

### Phase 2b depends on phase 2a surfaces landing first

If phase 2a is not yet merged everywhere this doc is consumed, phase 2b should branch on top of the phase 2a surface work or wait for it to land. Phase 2b is not specified as a standalone replacement for phase 2a.

## Approaches Considered

### Approach 1: Keep gating and harden evaluation

Do not change default posture yet. Keep `run(command)` gated and spend phase 2b on richer dashboards, judged evals, or automation.

Pros:

- lowest rollout risk
- no default-behavior change

Cons:

- phase 2a already covered the experiment posture
- delays the product transition for the proven cohort
- keeps operational semantics noisier than they need to be

### Approach 2: Default-on for the proven cohort with rollback

Make run-first presentation the normal behavior for the stable `chat + ACP` cohort, keep typed fallback visible, and retain the rollout control as an override and rollback switch.

Pros:

- cleanest continuation of phase 2a
- changes only one variable: posture, not cohort width
- preserves rollback without hiding fallback tools

Cons:

- still leaves broader cohort expansion for a later phase

### Approach 3: Default-on plus cohort expansion

Promote run-first to default-on and widen the provider/runtime cohort in the same phase.

Pros:

- faster rollout reach

Cons:

- mixes two variables at once
- makes regressions harder to attribute
- raises risk without new architectural benefit

## Recommendation

Use Approach 2.

Phase 2b should make run-first presentation the default posture for the already-proven stable `chat + ACP` cohort, preserve typed-tool fallback, and keep a rollback switch in place. It should not widen the cohort or redesign telemetry in the same phase.

## Proposed Design

### 1. Scope And Boundary

Phase 2b is a rollout-posture change for `chat + ACP`.

It changes:

- default presentation posture for the stable cohort
- rollout mode semantics
- low-cardinality posture labeling in telemetry

It does not change:

- the phase 1 command runtime
- the command family set
- typed fallback visibility
- approval behavior
- surface-specific prompt fragments beyond preserving the existing run-first contract
- the eligible cohort itself

The central rule is:

`run(command)` becomes the normal first-choice presentation for the proven cohort, while typed tools remain visible fallback tools.

### 2. Rollout Mode And Cohort Model

Phase 2b should make the rollout mode explicit instead of overloading `gated`.

Recommended mode vocabulary for both `chat` and ACP:

- `off`
- `gated`
- `default_on`

Semantics:

- `off`: disable run-first presentation entirely for that surface
- `gated`: preserve the phase 2a experiment posture for explicitly gated traffic
- `default_on`: treat run-first presentation as the normal path for in-cohort sessions, with out-of-cohort sessions remaining on the non-default path

Operational source of truth:

- phase 2b changes the shipped `chat` and ACP config or profile defaults for this feature to `default_on`
- resolver fallback in code remains `off`
- the existing env and explicit-argument overrides still take precedence over config defaults

The stable cohort selector in phase 2b is explicitly the existing provider allowlist match on the normalized `provider:model` key. Earlier discussion used “provider/runtime” as shorthand, but phase 2b does not widen the selector contract beyond what phase 2a actually implemented.

That means:

- `provider:model` allowlist membership is the eligibility boundary
- chat `streaming` remains a telemetry and reporting dimension, not an eligibility selector
- any richer runtime-specific cohorting is deferred to a later expansion phase

This avoids a hidden scope increase. Cohort modeling changes belong to a later expansion phase, not the default-on promotion slice.

### 3. Chat Surface Design

For `chat`, phase 2b should promote the current phase 2a run-first presentation to the normal behavior for in-cohort sessions.

#### Default behavior

For sessions matching the stable cohort:

- `run(command)` is ordered first
- the existing chat-specific run-first prompt fragment is injected
- typed tools remain listed after `run`
- typed descriptions continue to be framed as fallback or specialized tools
- provider-facing `tool_choice` remains unset or `auto`

For sessions outside the stable provider-model cohort:

- chat should not auto-expand run-first default behavior
- the surface should fall back to the non-default presentation path for that session

#### Rollback behavior

Chat keeps the run-first config switch as an operational override:

- operators can set `off` to disable run-first default behavior quickly
- phase 2a-style `gated` can remain available for internal testing or controlled experimentation
- `default_on` is the production posture for the stable cohort

#### Effective tool set invariant

Phase 2b must preserve the phase 2a invariant that one resolved effective tool set drives both:

- the `llm_tools` surface shown to the model
- the local auto-exec and allow-catalog eligibility used after tool selection

Promoting run-first to default-on must not reintroduce divergence between what the model sees and what chat will later permit.

### 4. ACP Surface Design

ACP should mirror the same posture promotion for the same stable cohort.

#### Default behavior

For in-cohort ACP sessions:

- the session-aware ACP presenter applies run-first ordering by default
- the ACP-specific run-first prompt fragment is attached by default
- typed MCP-derived tools remain visible and executable after `run`

For out-of-cohort ACP sessions:

- ACP should remain on the non-default posture rather than widening automatically

#### Rollback behavior

ACP keeps the rollout control as an override:

- `off` disables run-first presentation
- `gated` preserves controlled experiment posture when needed
- `default_on` is the normal stable-cohort posture

This override changes only the LLM-facing surface, not ACP governance, approvals, or tool execution semantics.

### 5. Telemetry And Posture Labels

Phase 2b should keep the same metrics and event boundaries from phase 2a:

- rollout exposure
- first tool selected
- fallback after `run`
- completion proxy

The recommended change is label refinement, not metric redesign.

Phase 2b should keep the existing `cohort` label name for continuity in both chat and ACP metrics, and broaden its value set rather than introducing a new `posture` label in this phase.

The rollout context carried with these metrics should distinguish at least:

- `default_on`
- `gated`
- `override_off`
- `out_of_cohort`

Recommended mapping:

- `default_on`: mode is `default_on` and the session is in the stable provider-model allowlist
- `out_of_cohort`: mode is `default_on` and the session is outside the stable provider-model allowlist
- `gated`: mode is `gated`
- `override_off`: mode is `off`

This keeps the telemetry model stable while making phase 2b behavior separable from phase 2a experimentation.

### 6. Error Handling And Operational Safety

Phase 2b should continue using safe defaults:

- unknown rollout mode values should resolve to `off`
- sessions where `run` is absent after filtering should remain ineligible for run-first presentation
- provider or cohort mismatches should fall into explicit out-of-cohort or ineligible labeling, not silent fallback

Run-first presentation failures remain non-fatal surface behavior:

- typed fallback stays available
- surface shaping must never bypass governance
- telemetry failures should remain observable but non-fatal

Phase 2b should explicitly bring chat telemetry emission up to that standard. ACP already logs non-fatal metric emission failures; chat currently suppresses them silently. The phase 2b implementation should replace chat-side silent suppression with warning or debug logging that preserves execution flow while keeping telemetry failures visible.

### 7. Testing And Verification

Phase 2b should add focused coverage around posture promotion rather than re-testing all phase 2a behavior from scratch.

Required test coverage:

- config resolver tests for `default_on`
- chat presenter tests for in-cohort `default_on`, out-of-cohort fallback, and explicit `off`
- ACP presenter tests for the same posture matrix
- chat and ACP telemetry tests verifying refined `cohort` label values
- chat telemetry helper tests verifying non-fatal metric failures are logged rather than silently suppressed
- integration coverage showing that stable-cohort sessions take the default-on run-first path without forcing `tool_choice`

Verification should include:

- targeted pytest coverage for chat and ACP presentation/config/metrics paths
- Bandit on touched runtime files
- `git diff --check`

## Risks And Mitigations

### Risk: Config ambiguity between experiment and default posture

If `gated` and `default_on` are not clearly separated, operators will have difficulty understanding whether a surface is in experimental mode or normal production posture.

Mitigation:

- explicit mode vocabulary
- explicit shipped config defaults for `default_on`
- explicit posture labels in telemetry
- docs and config examples that show `default_on` as the stable-cohort production posture

### Risk: Hidden cohort expansion

If phase 2b quietly changes the stable cohort selector while promoting default behavior, regressions will be hard to attribute.

Mitigation:

- keep the existing cohort selector exactly as phase 2a used it
- explicitly define that selector as provider-model only in this phase
- defer any broader cohort modeling to a later phase

### Risk: Surface divergence between chat and ACP

If chat and ACP interpret `default_on` differently, rollout posture becomes harder to reason about and metrics become harder to compare.

Mitigation:

- use one shared posture model
- keep prompt fragments surface-specific but behaviorally equivalent
- keep the event boundaries aligned across chat and ACP

## Acceptance Criteria

Phase 2b is successful when:

- `chat` and ACP both support `off`, `gated`, and `default_on` run-first modes
- in-cohort stable sessions use run-first presentation by default
- typed tools remain visible and executable fallback tools
- out-of-cohort sessions do not automatically widen into default-on behavior
- rollback to `off` is immediate and does not require changing the underlying `run` tool availability
- telemetry can distinguish default-on traffic from gated, override-off, and out-of-cohort traffic
