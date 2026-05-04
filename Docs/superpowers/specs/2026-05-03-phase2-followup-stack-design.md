# Phase 2 Follow-Up Stack Design

- Date: 2026-05-03
- Issue: #1116
- Backlog: TASK-7
- Topic: Remaining Phase 2 follow-up extraction work after the Phase 4 roadmap stack closed
- Status: Approved design

## Goal

Finish the remaining #1116 earlier-phase extraction debt as a deliberate stack of small, independently reviewable PRs.

The Phase 4 roadmap stack is closed. This design intentionally does not fold in PR #1237 or any OpenAPI raw-response-contract cleanup. That work remains a separate lane. The scope here is only the remaining Phase 2 follow-up work called out by #1116:

- Phase 2.1 lifecycle and initialization cleanup
- Phase 2.2 complex conditional router group cleanup
- Phase 2.3 ChaChaNotes PersonaStateStore/facade delegation and conservative monolith shrink
- Phase 2.4 typed config-section follow-up only when it directly unblocks later refactors

## Current State

The refreshed #1116 tracker says the original Phase 2 redux PRs merged:

- #1123 landed startup validation and auth helper extraction for Phase 2.1.
- #1122 landed core/content/admin router grouping for Phase 2.2.
- #1115 landed continued ChaChaNotes store extraction and conservative duplicate monolith-method shrink for Phase 2.3.
- #1121 landed the first typed config sections for Phase 2.4.
- #1120 landed the unified DB error hierarchy for Phase 2.5.

The remaining work is follow-up extraction debt, not a continuation of the old stale branches. New work should start from current `origin/dev` in fresh `.worktrees/...` worktrees.

Important current surfaces:

- `tldw_Server_API/app/main.py` remains large and still owns lifecycle and startup orchestration.
- `tldw_Server_API/app/services/lifecycle_workers.py` already provides the current `WorkerRegistry` facade, `ManagedWorker`, phased inventory publication, and stop-event shutdown helpers.
- `tldw_Server_API/app/api/v1/router_groups/` exists, but `core.py`, `content.py`, `admin.py`, and `minimal.py` still contain complex conditional router import and registration logic.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` remains a large compatibility facade, while focused stores under `tldw_Server_API/app/core/DB_Management/chacha/` now include `persona_state_store.py`.

## Design Principles

Use conservative, test-covered tranches.

Each PR should have one primary behavioral invariant:

- Phase 2.1 tranches must preserve startup order, enablement flags, app-state publication, and shutdown behavior.
- Phase 2.2 tranches must preserve public route paths, tags, route keys, lazy-import behavior, and minimal-test-app gating.
- Phase 2.3 tranches must preserve the public `ChaChaNotes_DB` facade while moving only covered methods or pure helpers behind focused stores.
- Phase 2.4 tranches must be pulled only when typed config data removes duplication or startup fragility needed by another stage.

Avoid broad cleanups:

- Do not combine lifecycle cleanup, router extraction, and ChaChaNotes movement in one PR.
- Do not start with `chat`, `audio`, or `media` router families unless a smaller router tranche has already proven the pattern.
- Do not delete uncovered ChaChaNotes monolith branches.
- Do not change endpoint payloads, response envelope behavior, pagination contracts, or OpenAPI raw-response documentation in this stack.

## Staged Plan

### Stage 0: Inventory And Work Order Setup

Create a current-state inventory branch from `origin/dev`.

Recommended branch:

- `codex/phase2-followup-work-order`

Deliverables:

- This design/spec.
- Backlog tasks for the first implementation tranche once selected.
- A short remaining-work matrix mapping candidate files, risk, existing tests, and proposed first PR boundaries.

Success criteria:

- #1237 is documented as separate.
- The first implementation tranche is selected from the current codebase, not from stale PR diffs.
- No production code changes are included.

### Stage 1: Phase 2.1 Lifecycle/Init Cleanup

Start with the smallest lifecycle tranche that extends existing seams instead of inventing a new registry.

Recommended first branch:

- `codex/phase2-1-worker-lifecycle-cleanup-a`

Initial candidate scope:

- Use the existing `lifecycle_workers.py` model.
- Migrate or remove one remaining duplicated worker lifecycle/deprecated-code path.
- Prefer a worker path that already has a single stop event, direct task ownership, and focused tests.
- Keep ResourceGovernor and heavy startup extraction out of the first PR unless inventory proves a tiny pure-helper extraction is lower risk.

Out of scope for the first 2.1 PR:

- broad `ResourceGovernor` ownership changes
- moving all heavy startup logic
- scheduler redesign
- worker migration tracking that belongs in #1114

Verification gate:

- focused lifecycle worker tests
- startup/import guard tests touching `main.py`
- any route/minimal-app smoke tests affected by startup changes
- Bandit on touched Python source
- `git diff --check`

### Stage 2: Phase 2.2 Router Conditional Groups

Extract conditional router registration in medium, domain-bounded tranches after Stage 1 establishes that current `dev` is stable.

Recommended first branch:

- `codex/phase2-2-router-conditionals-a`

Recommended order:

1. `sandbox` or `ACP` router conditionals first because they are smaller and easier to verify than `chat`, `audio`, or `media`.
2. `chat` conditionals next only after the first router helper pattern is proven.
3. `media` and `audio` last because they have heavier optional imports, test-runtime gates, and provider-compatible behavior concerns.

Design shape:

- Add small helper modules under `tldw_Server_API/app/api/v1/router_groups/` only when they reduce repeated conditional import logic.
- Keep `RouterSpec` as the contract boundary.
- Preserve route keys and lazy factories exactly.
- Preserve `MINIMAL_TEST_APP` and audio import gating semantics.
- Add regression tests around the generated specs, not only route existence.

Verification gate:

- router group unit tests for selected family
- minimal test app route-gating tests
- generated OpenAPI/client-path verification if route registration changes can affect docs
- Bandit on touched source
- `git diff --check`

### Stage 3: Phase 2.3 ChaChaNotes PersonaStateStore/Facade Delegation

Continue shrinking `ChaChaNotes_DB.py` only where fresh tests prove behavior.

Recommended first branch:

- `codex/phase2-3-chacha-persona-delegation-a`

Initial candidate scope:

- Select one cohesive PersonaStateStore-backed method family.
- Add or strengthen tests against the public `ChaChaNotes_DB` facade first.
- Move only the covered implementation into `PersonaStateStore` or a small helper behind it.
- Keep compatibility aliases and public method signatures stable.

Out of scope for the first 2.3 PR:

- broad SQL rewrites
- schema migrations
- changing sync logging semantics
- deleting uncovered monolith methods
- moving unrelated note/conversation/message/character behavior

Verification gate:

- focused ChaChaNotes/persona tests proving facade compatibility
- direct PersonaStateStore tests when helper logic is added
- any relevant property tests already covering serialization or normalization
- Bandit on touched source
- `git diff --check`

### Stage 4: Optional Phase 2.4 Typed Config Sections

Treat Phase 2.4 as an enabling lane, not a default workstream.

Open a 2.4 tranche only when one of these is true:

- Phase 2.1 startup cleanup needs a typed config section to remove repeated parsing safely.
- Phase 2.2 router gating needs typed config to remove duplicated route-default logic.
- A current test exposes drift between config defaults and runtime parsing.

Recommended branch if needed:

- `codex/phase2-4-config-followup-a`

Verification gate:

- direct typed-loader tests
- startup validation tests
- any affected feature flag or route-gating tests
- Bandit on touched source
- `git diff --check`

### Stage 5: Tracker Closeout

After each tranche merges:

- update #1116 with the PR link, scope, verification, and remaining Phase 2 follow-up debt
- update Backlog tasks with verification and final summary
- prune only clean merged worktrees when asked or when repo policy allows

Do not mark #1116 closed until the owner decides whether remaining lower-priority Phase 2.4 or new Phase 5 scope should stay on the tracker.

## Branch And PR Policy

Use fresh worktrees from current `origin/dev`:

```bash
git fetch origin
git worktree add .worktrees/phase2-1-worker-lifecycle-cleanup-a -b codex/phase2-1-worker-lifecycle-cleanup-a origin/dev
```

Keep one reviewable unit per PR. Prefer smaller PRs over preserving numeric phase ordering when risk argues for a smaller first slice.

Recommended merge order:

1. Stage 0 planning/inventory
2. Stage 1 first 2.1 lifecycle cleanup
3. Stage 2 first 2.2 router conditional helper tranche
4. Stage 3 first 2.3 ChaChaNotes persona delegation tranche
5. Additional 2.1/2.2/2.3 iterations based on current remaining-work matrix
6. Optional 2.4 only when it is an enabling dependency

## Review Checklist

Each implementation PR should include:

- human-authored `Change summary` before merge
- explicit behavior-preservation statement
- focused tests that fail before the implementation where practical
- touched-source Bandit result or documented non-code skip
- `git diff --check`
- #1116 status comment after merge

## Risks And Mitigations

Risk: touching `main.py` creates broad startup regressions.

Mitigation: start with existing lifecycle seams, keep startup policy in `main.py`, and move only one worker lifecycle pattern at a time.

Risk: router group extraction changes route availability or OpenAPI output.

Mitigation: preserve `RouterSpec` values, test spec generation, and use generated OpenAPI/client-path verification for route movement.

Risk: ChaChaNotes facade movement changes persistence semantics.

Mitigation: write public-facade tests before movement, keep signatures stable, and do not remove uncovered branches.

Risk: Phase 2.4 grows into unrelated config cleanup.

Mitigation: require an explicit dependency from Stage 1 or Stage 2 before opening a 2.4 tranche.

## Non-Goals

- fixing or merging #1237
- changing response envelopes or endpoint payloads
- continuing Phase 3.2 pagination migration
- reopening Phase 4 decomposition work
- broad DB schema cleanup
- large all-at-once `main.py` or `ChaChaNotes_DB.py` rewrites
