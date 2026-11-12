# UAT389 Content Review acceptance repair

Backlog: TASK-13260.278.2. Scope: supported Quick Ingest draft creation, local edit/diff/revision persistence, AI proposal and canonical commit. Auth seeding is harness setup, not fresh-login acceptance. No browser API success stubs.

## Stage 1: Trace and reproduce
**Goal**: Identify the false-pass paths and real ingest/commit contracts.
**Success Criteria**: Required effects and prerequisites are explicit; causal regressions fail first.
**Tests**: Page readiness rejects absent content; db_id-only add response reaches final content update.
**Status**: Complete

## Stage 2: Repair bounded harness and commit identity
**Goal**: Own input files/batches, require exact edits/diffs and versions after reload, require successful AI/commit results. Fix UAT395 db_id extraction under its child task.
**Success Criteria**: Missing draft/action/API result fails; successful commits identify and reload the same saved source. Separate empty-state test.
**Tests**: Targeted component and page-object regressions; registered Playwright workflows.
**Status**: Complete

## Stage 3: Verify and report remaining acceptance
**Goal**: Run independent checks and record integrated runtime dependencies.
**Success Criteria**: Counts, lint/type diagnostics and Bandit applicability are reported without claiming native UAT.
**Tests**: Vitest regressions, Playwright discovery, touched ESLint, TypeScript diagnostics, Bandit applicability.
**Status**: In Progress

Native obligations: run new workflows against an owned built candidate with real API and initialized SQLite and official PostgreSQL profiles; configure a working text model for AI fix. B-09 saved-source analysis/reanalysis, failure/fallback, partial batch, attachment-loss, conflict and duplicate-click recovery remain separately required.

## Verified repairs and remaining execution

- TASK-13260.278.2.1 / UAT395: actual add response uses db_id; one typed fallback lets reviewed content update the saved source. Two causal component regressions pass, including failed final update recovery.
- TASK-13260.278.2.2 / UAT396: WebUI Quick Ingest used an extension options URL. One existing-runtime branch correction routes to the shipped pages/content-review.tsx. Three new route cases plus seven existing hook cases pass. The acceptance fixture now requires the real automatic batch/draft handoff before proceeding.
- UAT389 readiness regression proves missing UI rejects (old helper resolved). Four browser cases are registered; they require real owned ingestion, AI and canonical server reads and install no API fulfillment. UAT397 repairs the shared auth fixture so it never fabricates extension identity; the UAT389-only workaround was removed.
- Final focused Vitest after UAT397: 16 passed across 4 files. UAT389 harness/new tests ESLint: no warnings/errors. Shared auth helper: 3 existing warnings, no errors. Shared code: 38 existing warnings, no errors; no added warning. Full frontend typecheck had four concurrent release-collection test errors outside this scope; root handles the combined rerun.
- Bandit /tmp/bandit_uat389_uat395_uat396.json: zero Python lines and no errors; TS/TSX security is outside its coverage.
- No browser test was executed against a native candidate: owned integrated runtime is unavailable. Run the four registered cases against built SQLite and official PostgreSQL profiles with a working text model and TLDW_LIVE_TIER_UAT=1. B-09 bulk/partial commit and saved-source reanalysis/fallback remain separate obligations.

- TASK-13260.278.2.3 / UAT397: the actual seedAuth callback reproduced two false extension IDs for absent/empty runtimes; the genuine extension case already passed. Removing runtime identity fabrication preserves storage/auth seeding and real IDs. Three focused regression scenarios added; normal UI login remains a separate full-UAT fixture obligation.
