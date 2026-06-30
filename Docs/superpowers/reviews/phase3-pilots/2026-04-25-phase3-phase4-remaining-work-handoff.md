# Phase 3 And Phase 4 Remaining Work Handoff

**Date:** 2026-04-25

**Status:** Handoff complete; runtime implementation remains blocked until gates are accepted.

## Purpose

Provide one queue for the remaining roadmap work after Phase 2 closeout, PR `#1125` stabilization, Phase 3 planning, and Phase 4 parking-lot planning. This is a docs-only coordination artifact. It does not approve starting runtime code.

Maintainer decision checklist:

- `Docs/superpowers/reviews/phase3-pilots/2026-04-25-maintainer-decision-checklist.md`

## Current Rule

Do not start broad runtime implementation from this dirty workspace.

Before starting code:

- create a clean worktree from the accepted base;
- recheck current PR status;
- confirm maintainers accept the specific phase gate;
- run the focused baseline tests before edits;
- keep each implementation PR narrowly scoped.

## Required PR Status Refresh

Before implementation starts, rerun:

```bash
gh pr view 1115 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1120 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1121 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1122 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1123 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
gh pr view 1125 --repo rmusser01/tldw_server --json state,isDraft,mergeStateStatus,statusCheckRollup
```

Do not rely on the 2026-04-25 snapshot after new pushes land.

## Immediate Blocked Item

PR `#1125` CI fixes are tracked but should stay with the PR-fixing fork unless maintainers explicitly redirect that work here.

Tracked fixes:

- MediaWiki safe error category restoration.
- SBOM fallback for `pyproject.toml`.
- Mobile attach-image accessible control.
- Wizard teardown SQLite isolation failure.

If this fork is asked to take over PR `#1125` fixes:

- create a clean PR-head worktree;
- avoid this dirty workspace;
- keep fixes separate from Phase 3/4 planning docs;
- rerun the PR-specific focused tests and Bandit before pushing.

## Phase 3 Implementation Queue

### 1. Shared Response Envelope Helpers

Owner artifact:

- `Docs/superpowers/plans/2026-04-25-phase3-1-standard-response-envelope-implementation-plan.md`
- `Docs/superpowers/reviews/api-response-envelope/2026-04-25-helper-contract-spec.md`

Entry gate:

- PR `#1125` stable or accepted.
- Maintainers accept legacy-default `v1` and header opt-in.

First PR shape:

- helper schemas/builders only;
- no endpoint behavior changes;
- helper unit tests only.

### 2. Shared Pagination Helpers

Owner artifact:

- `Docs/superpowers/plans/2026-04-25-phase3-2-pagination-standardization-implementation-plan.md`
- `Docs/superpowers/reviews/api-pagination/2026-04-25-helper-contract-spec.md`

Entry gate:

- Phase 3.1 metadata location accepted, or maintainers accept pagination helpers independently.
- Legacy pagination aliases remain accepted.

First PR shape:

- schemas and helper functions only;
- backwards-compatible `Link` header builder;
- no route-wide alias rejection.

### 3. Auth Dependency Aliases And Contract Tests

Owner artifact:

- `Docs/superpowers/plans/2026-04-25-phase3-4-auth-dependency-standardization-implementation-plan.md`
- `Docs/superpowers/reviews/auth-dependencies/2026-04-25-helper-contract-spec.md`

Entry gate:

- AuthNZ-adjacent Phase 2 bases stable.
- Maintainers accept alias naming and `require_token_scope(...)` guard behavior.

First PR shape:

- alias/helper exports;
- request-state contract tests;
- no endpoint migration unless helper PR is already green.

### 4. `skills` Pilot

Owner artifact:

- `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-execution-packet.md`

Entry gate:

- shared helpers are green;
- frontend owner accepts opt-in header and client-side unwrap;
- OpenAPI caveat or contract approach accepted.

First PR shape:

- opt-in envelope on selected JSON `skills` routes;
- canonical pagination metadata on opt-in list route;
- legacy default responses unchanged;
- export zip and `204` delete routes exempt.

## Phase 4 Implementation Queue

Phase 4 implementation remains behind the Phase 4 readiness gate:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-phase4-readiness-gate.md`

Recommended order after the gate opens:

1. Accept API versioning policy.
2. Stabilize OpenAPI contract decisions.
3. Measure coverage baseline.
4. Execute deployment docs refresh.
5. Start `Prompts_DB.py` decomposition.
6. Start `storage.py` endpoint decomposition.

### Phase 4.1 Coverage Ratchet

Owner artifacts:

- `Docs/superpowers/plans/2026-04-25-phase4-1-coverage-ratchet-baseline-plan.md`
- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-coverage-ratchet-measurement-packet.md`

Next step:

- run backend coverage baseline with `--cov-fail-under=0` from a clean accepted base.

Do not:

- raise directly to 25%;
- mix threshold changes with runtime refactors.

### Phase 4.2 Deployment Docs

Owner artifacts:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-deployment-docs-inventory.md`
- `Docs/superpowers/plans/2026-04-25-phase4-2-deployment-docs-refresh-plan.md`

Next step:

- get docs owner decision on source/published flow, HA guide status, and monitoring publishing shape.

Do not:

- manually edit generated `Docs/Published` mirrors outside the accepted refresh flow.

### Phase 4.3 DB Decomposition

Owner artifacts:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-db-hotspot-inventory.md`
- `Docs/superpowers/plans/2026-04-25-phase4-3-prompts-db-decomposition-plan.md`

Next step:

- get maintainer acceptance for `Prompts_DB.py` as the first DB target.

Do not:

- start with `ChaChaNotes_DB.py`;
- touch `Collections_DB.py` from this dirty workspace.

### Phase 4.4 Endpoint Decomposition

Owner artifacts:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-endpoint-hotspot-inventory.md`
- `Docs/superpowers/plans/2026-04-25-phase4-4-storage-endpoint-decomposition-plan.md`

Next step:

- get maintainer acceptance for `storage.py` user-owned JSON routes as the first endpoint target.

Do not:

- move file-download or admin quota routes in the first split;
- change auth dependencies or response shapes during route movement.

### Phase 4.5 API Versioning

Owner artifacts:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md`
- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md`

Next step:

- maintainers accept or amend the five policy decisions.

Do not:

- make standard envelopes default in `v1` before policy acceptance.

### Phase 4.6 OpenAPI Contract Testing

Owner artifact:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-openapi-contract-testing-plan.md`

Next step:

- wait for Phase 3.1/3.2 helper schema names and pagination schema names to stabilize.

Do not:

- make strict OpenAPI mode required while reviewed exceptions remain.

## Clean Worktree Rule

Use this workspace for planning only unless maintainers explicitly accept its dirty state.

Implementation should use one worktree per target:

- one for Phase 3 shared helpers;
- one for `skills` pilot;
- one for coverage ratchet;
- one for deployment docs refresh;
- one for DB decomposition;
- one for endpoint decomposition.

Do not share a runtime worktree between PR `#1125` fixes and Phase 3/4 implementation.

## Final Remaining Decisions

- Are Phase 2 PR heads stable enough to use as bases?
- Is PR `#1125` stable enough to use as the sanitized-error base?
- Do maintainers accept the Phase 3 legacy-default/header-opt-in contract?
- Do maintainers accept `skills` as the first Phase 3 pilot?
- Do maintainers accept the Phase 4 readiness gate and recommended order?
- Which Phase 4 item gets the first implementation slot after Phase 3 stabilizes?

Use `Docs/superpowers/reviews/phase3-pilots/2026-04-25-maintainer-decision-checklist.md` as the review checklist for these decisions.
