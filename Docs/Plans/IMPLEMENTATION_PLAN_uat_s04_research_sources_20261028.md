# S04 Research and source workspaces UAT

Task: TASK-13260.278.18.83.45. S03 is accepted; selected acceptance40/43. Historical Kanban180s incident remains OPEN/UNATTRIBUTED by human choice. Finish S04 before S05. Frozen dev a412f6580b7ed0cf48fd7d222a47a20cfc0dde92 includes3309/3310/3311; no current live-dev or release-wide claim. Qualified local product tree57985eca0aed3396b7ca1a978e6a6aa93250f430, private backend /private/tmp/tldw-s03-document-20261028/site and frontend BUILD BJwzzv2nGHfhY5bChe4bj are reused only after exact source/build guards.

Spec: Docs/Development/RELEASE_UAT_PLAYBOOK.md S-04 and apps/tldw-frontend/scripts/live-tier-uat/release-catalog-data.mjs. Execute inline with existing executing-plans, debugging/TDD, scoped security, independent actual/staged review and normal hooks. Reuse this checkout, current backend package and frontend build until a proven change requires replacement.

ADR check: no new architecture decision for validation. Preserve [ADR007](../ADR/007-research-workspace-canonical-first-slice-shell.md), [ADR008](../ADR/008-workspace-split-key-persistence-and-indexeddb-offload.md), [ADR066](../ADR/066-explicit-web-capture-and-refresh-snapshots.md) and [ADR059](../ADR/059-backlog-py-task-editor-cutover.md). Assess any proven repair before implementation.

## Contract and source map

- Research Workspace: apps/packages/ui/src/routes/option-research-workspace.tsx and components/Option/ResearchWorkspace/{ResearchWorkspaceRouteGate,index,ChatPane/index,StudioPane/QuickNotesSection}. Create a workspace, ingest controlled Rowan text plus a controlled second source, perform a real bounded cited query, save a resulting Note/report, reload and compare exact workspace/source/result IDs, text, citation provenance and account scope.
- Import variants: SourcesPane/AddSourceModal.tsx advertises Upload, My Media, URL, Paste and Search Server. Map each active path and its backend before execution; preserve each real receipt and independently compare imported identity/content. Existing A05 format checks do not substitute for this integration.
- Deep Research: apps/tldw-frontend/pages/research.tsx and lib/api/researchRuns. Inventory advertised run/checkpoint/bundle-to-workspace behavior and enabled providers before assigning its native oracle.
- Sources: apps/packages/ui/src/routes/option-sources{,-new}.tsx and components/Option/Sources/{SourcesAvailabilityGate,SourceForm}. Inventory actual supported source adapters and authority before a real connector action. No operator path/source access outside the private fixture.
- The four /connectors routes deliberately show ConnectorRoutePlaceholder. Verify explicit unavailable state and valid recovery links; do not count placeholders as successful connectors or silently implement an unrequested integration.
- Recovery: real provider refusal then one legitimate Retry preserves saved sources/Note/provenance; invalid or unavailable connector action gives truthful error without replacing prior work. No fake provider or browser responses. External-network failure gets its own result instead of unrelated substitute data.
- Exact second source: Rowan field report says the dome opens at17:30 and visitors use the east gate. Ask which telescope, tour time, opening time and entrance; expected answer contains Selene, Friday18:00,17:30,east gate with correct source citations. Do not accept invented facts or title-only output.

## Runtime and preservation

Existing all-necessary diagnostics/retries/repairs/build/runtime authorization persists. Reassess after three failures using new evidence and reviewed corrections; no repeated permission question for already-authorized scope. One private Chrome window/tab, normal app connect/login, no auth-store injection. Protect original API50010/frontend50863/Chrome54716, all user tabs/data/keys, current unknown draft and Splash8010. Original DOM remains unverified/untouched; retain canonical/owner/target guards without stale B02 questions. Official restricted isolated PostgreSQL15492 only; no DIY database/operator SQL. Heartbeat paused; no GitHub/fetch/publication/queue/merge. No installs/downloads/models unless necessary and expressly within retained scope.

Caps:600s workflow/build,120s startup,180s generation,300s test case,15s operations/graceful cleanup plus independent handback reserve. Retain actual request completion, ASGI/process release and exact fixture disposal guards. No proof bundles; remove completed private profiles/exports/review images after acceptance. Keep current reusable package/build only.

## Stage1: Inventory and pin cases
**Goal**: Resolve every advertised workspace/import variant to active controls, API/fixture/provider and an exact oracle before writes.
**Success Criteria**: Bounded manifest records capability/refusal, own-positive/foreign-denial cases, source text/provenance, native driver ownership, cleanup and request budgets.
**Tests**: Small checks for new operator admission/result/privacy logic; source inspection of existing ResearchWorkspace/AddSourceModal/QuickNotes and Research run/source adapter tests. No unchanged broad suite replay.
**Status**: In Progress

## Stage2: Validate and repair sequentially
**Goal**: Complete the canonical workspace and its import variants, then other active advertised research/source paths, one unit at a time.
**Success Criteria**: Real query/citations/save/reload, recovery and no prior-work loss, SQLite/PostgreSQL single-/multi-user ownership checks, independent canonical/export comparisons. Causal red precedes minimal shared production repair; affected client/backend tests, scoped lint/compile/type/security and source/staged review pass before packaging changed source.
**Tests**: Existing adjacent component/API/DB fixtures selected from the pinned case manifest; real restricted PostgreSQL when touched; exact baseline/current Bandit for Python scope. No claim that mocked tests are live UAT.
**Status**: Not Started

## Stage3: Accept and dispose
**Goal**: Settled screenshots and human UX, final independent review and truthful source/native qualification, original preservation and completed owned cleanup.
**Success Criteria**: All named variants have actual results or explicit owner disposition, all private requests/processes retire, current user session untouched, final task notes retain limits and historical open incidents. Remove only this own plan when complete; acceptance41/43 only after the family gate passes.
**Tests**: Actual saved identity/text/provenance, foreign refusals, strict receipts/release and guarded disposal. No routine tracking-only publication.
**Status**: Not Started
