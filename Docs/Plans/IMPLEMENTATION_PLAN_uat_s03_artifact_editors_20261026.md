# S03 Artifact editors UAT

Task: TASK-13260.278.18.83.45. S02 is closed at local33bbfd6ca54085ef40351ead362cda8b6d8022aa; selected39/43. Existing production backendc607 and private BUILD9NhGqJCpJaJdEQO9ZztbS/site `/private/tmp/tldw-s02-pg-jobs-20261026/site` compose on frozen deva412f6580b7ed0cf48fd7d222a47a20cfc0dde92, including3309/3310/3311. No fresh live-dev or release-wide claim.

ADR assessment: frontend save/cancellation repair requires no new architecture decision. The completion/publication protocol requires proposed [ADR-083](../ADR/083-data-table-completion-before-publication.md) and its [bounded design](../Design/Data_Tables_Completed_Job_Publication_20261027.md). Preserve ADR003 Jobs, ADR008 workspace persistence, ADR023 Media-owned tables/exports, ADR058 locked completion identity and ADR059 task editing. Governing accepted rationale stays immutable; proposed backend design is not implementation acceptance.

## Contract and order

Source: RELEASE_UAT_PLAYBOOK.md S-03 and release-catalog-data.mjs. Finish this family before S04, in this order:

1. Data Tables: real generation from the fixed Rowan source;3 rows by2columns; change one cell, save/reload and export. Independently parse every exported cell and compare with saved canonical content. Failure/cancellation must preserve the prior saved table.
2. Presentations: real3-slide generation; exact deterministic text edit and one reorder, save/reload and export. Independently open output and verify slide count, text and order. Failed/cancelled generation preserves saved slides/order.
3. Kanban:2lists/3cards; move one card once, save/reload/export and independently verify every list/card/order. Map the actual shipped generation controls before claiming the catalog generation-cancel case; never substitute UUID generation for model generation.
4. Document Workspace: controlled2-page document, one annotation, persisted/reloaded exact text/annotation and independently opened export. Map actual shipped generation/export/recovery controls before claiming unsupported operations.

Human UX requires settled screenshots; acceptance is separate from functionality. Each advertised named variant and its recovery gets an explicit result/disposition. No title-only pass, fake provider/browser response, silent exemption or implication that one native variant covers all modes. Extra unadvertised features are outside scope.

## Runtime and preservation

Human all-necessary diagnostics/retries/fixes/runtime authorization persists. Reassess after three failures with new evidence, tested operator corrections and independent review; no repetitive permission questions. Original API50010/frontend50863/Chrome54716, its five tabs/unique retained Chat/five message IDs/draft, profiles/data/keys and Splash8010 remain protected. One owned private Chrome window/tab; normal API-key connect/login, no auth-store injection. Reuse current tools/dependencies/build, change only proven affected source, and dispose completed owned payloads. No force, shared-service/model changes, install/download without need, GitHub/remote/publication/queue action or heartbeat change. Heartbeat stays paused.

Caps:600s host/workflow/build,120s startup,180s generation,300s test case,15s operation/graceful cleanup, with independent handback reserve. Original canonical/DOM/owner checks and actual request/lifespan release are required. Use official isolated restricted PostgreSQL fixtures and database abstractions; no DIY database/operator SQL. Never read old logs, caches/models/auth stores or recreate proof bundles.

## Stage 1: Map shipped controls and freeze exact oracles
**Goal**: Pin source/test/caller/fixture/config/export paths and supported behavior for all four variants before live writes.
**Success Criteria**: Explicit fixtures, exact saved/exported comparisons, actual generation/recovery mapping, current package/build/principal binding, reviewed private operators and cleanup guards.
**Tests**: Small refusal/privacy/deadline/result-preservation checks only for new operator logic; no unchanged passing-control replay.
**Status**: In Progress

## Stage 2: Validate and repair sequentially
**Goal**: Execute each real native workflow, establish causal red for actual defects, apply the smallest shared fix and qualify affected callers.
**Success Criteria**: Real provider/Jobs outcomes, exact edit/order/save/reload/export/independent readers and recovery preservation, explicit SQLite/PostgreSQL single-/multi-user composition with own-positive/foreign-refusal controls.
**Tests**: Genuinely affected backend/frontend modules, normal lint/compile/scoped baseline-current Bandit, independent actual/staged reviews and normal hooks. Package/build only when changed source requires it.
**Status**: In Progress

## Stage 3: Accept, preserve and dispose
**Goal**: Human screenshot UX, final independent review, original handback, completed owned cleanup and concise durable closure.
**Success Criteria**: All named variants/recoveries have truthful results or explicit owner dispositions, private processes/requests retired, originals preserved, completed files disposed; only this owned plan removed after completion.
**Tests**: Independent canonical/export checks, actual source/request/process/DOM guards, exact-scope normal hooks and guarded disposal.
**Status**: Not Started

## Current inventory

Data Tables uses SourceSelector, GenerationPanel, TablePreview, SaveTablePanel, EditableDataTable/TableDetailModal and ExportMenu; Media DB APIs/Jobs worker and tests/DataTables govern storage/export/cancellation. Existing Splash8010 is reachable and reports unsloth/Qwen3.8-27B-GGUF:Q8_0; no generation occurred in preflight. Kanban currently exposes normal board/import controls in the inspected index; generation mapping remains incomplete. Document Workspace page exposes document loading and Highlights & Notes; generation/export mapping remains incomplete. No S03 native check or acceptance yet.


## Authorized capability additions and Data Tables causal unit

Human explicitly authorized adding all three missing capabilities: AI Kanban board/list/card generation with Cancel, document-page generation with Cancel, and a combined annotated-document export. Neither generation-cancel row may be silently marked unavailable. New-feature design/ADR assessment precedes their implementation; the current Data Tables repair remains first.

Causal UI red: all five original component cases fail naturally in7.012s, proving unsaved edited cells, missing content-write failure protection and false success for artifact/preview/local IDs. The separate connected row-deletion case fails expected3 versus2 in3.065s; no prior red replay. Real Zustand/Antd/QueryClient are retained, with mocked API boundaries only. This is component evidence, not live or real-store acceptance. The minimal repair reuses saveDataTableContent before metadata and library success, takes canonical saved counts, and refuses false server saves for serverless previews while preserving their draft and export. Existing caller/serializer and independent test review CLEAR. Save green:7focused cases in each configured client, including real-shaped name-keyed rows, stale-count correction, content failure and metadata failure after content ACK. Both writes are separate; metadata failure does not roll back content. Actual native workflow remains pending.


## Data Tables frontend qualified unit

Shared generation lifecycle now owns the exact start receipt, captured server/account scope, polling, final read and Cancel for both Preview and Detail. Cancel before start ACK retires the accepted job; cancelled/unmounted/replaced operations cannot publish. Errors remain visible and accepted nonterminal jobs retain Cancel; real scope invalidation retires even a retained/pending Cancel. Successful Cancel aborts the actual lease signal before releasing its listeners. Previous preview/edit state is retained until replacement success; drawer detail GET is also fenced on switch/close.

Evidence composition: initial cancellation8failed; API/scope-policy15failed6passed; connected current cohort265passed in UI28.206s/WebUI21.262s. Review-discovered poll/Cancel and stale-detail races reproduced4failed then corrected18passed in each client. Additional real-lease retained/Cancel/abort-order controls reproduced individually before correction. Final changed cancellation15+real-lease6 PASS UI21/13.110s and WebUI21/10.180s. Earlier focused save7 passes in both clients remain valid. These are separate scoped cohorts, not one full-suite run or live acceptance.

Touched ESLint baseline/current introduces0new findings; inherited API517/media62/TablePreview2 diagnostics and the global Next pages warning remain. Touched syntax and diff checks pass; no whole-project TypeScript claim. Bandit is inapplicable to this frontend-only unit. Independent actual/staged review and normal hooks/local commit gate the unit. No frontend build or S03 runtime action yet.

## Backend publication repair open

Real SQLite dedicated Cancel, direct manager cancellation and admin batch cancellation each acknowledge cancellation after handler generation but before SDK completion, while saved columns/rows have already changed:3 genuine preservation failures. The first lookup-error fixture used an uncaught Media DatabaseError and does not prove fail-open. Its corrected caught SQLite OperationalError-only run fails DID NOT RAISE. No PostgreSQL/live qualification or backend repair yet.

Use the proposed completion-first design: durable stored Jobs result and exact completion precede Media replacement; callbacks plus authenticated job-read replay recover publication, with atomic owner/version/stable-ID idempotence and explicit public publication state. Never expose staged rows/source snapshots or use progress telemetry as authority. Backend tests remain a separate unfinished unit; do not build or claim Cancel data safety until that unit and real workflow checks pass.
