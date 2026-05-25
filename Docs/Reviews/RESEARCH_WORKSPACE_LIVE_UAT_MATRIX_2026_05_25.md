# Research Workspace Live UAT Matrix

Date: 2026-05-25

This matrix is the current acceptance ledger for the Research Workspace remediation
stream. It is intentionally tied to live backend/WebUI validation, not only code
inspection. Rows owned by completed child tasks retain their recorded live evidence;
rows marked `Current run` were rechecked during TASK-478.13.

## Validation Rules

- Behavior claims require a live backend plus WebUI run using Playwright/CDP.
- Static code review can justify `Not covered` or `Gap`; it cannot justify `Pass`.
- `/research-workspace` is the only active WebUI route. `/workspace-playground`
  must remain removed with no alias and no redirect.
- Extension handoff remains blocked until a current extension build is available.
- MCP, ACP, and Sandbox are part of the canonical workspace model. If a behavior
  is only documented as a handoff contract and not yet exposed as a working UI/API
  flow, it must be marked `Partial` or `Gap`.

## Status Legend

| Status | Meaning |
| --- | --- |
| Pass | Works in the live product or focused regression suite, with evidence recorded. |
| Partial | Core contract exists, but part of the user workflow is not live or not fully covered. |
| Blocked | Cannot be validated until an external prerequisite is available. |
| Gap | Known missing behavior or endpoint/UI surface. |
| Watch | Works for the tested path, but needs continued matrix coverage because it is high-risk. |

## Matrix

| ID | Scenario | Owner | Surface | Status | Evidence | Regression coverage | Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-UAT-001 | Backend starts and authenticated workspace APIs respond | TASK-478.3 | FastAPI Workspaces API | Pass | TASK-478.3 live validation created a workspace, added an idempotent source, saw one `workspace_source_ingest` job, and read `/sources/status` job/error details. | `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`; `test_workspaces_api.py`; media ingest worker tests. | Keep status rows updated when Jobs projection changes. |
| RW-UAT-002 | Canonical `/research-workspace` route loads | TASK-478.7, TASK-478.13 | WebUI route | Pass | TASK-478.13 live probe booted `/research-workspace`, found `#workspace-main-content` and `workspace-workspaces-button`, and recorded screenshot `/private/tmp/task47813-live-matrix-research-workspace.png`. | `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`. | Keep this as a high-risk smoke check for future route metadata work. |
| RW-UAT-003 | Old `/workspace-playground` route is removed with no redirect | TASK-478.7, TASK-478.13 | WebUI route | Pass | TASK-478.13 live probe returned HTTP 404 for `/workspace-playground`, retained path `/workspace-playground`, and observed no `Location` header. Focused Playwright regression passed. | `research-workspace.real-backend.spec.ts` checks 404, retained path, then canonical route boot. | Never add route aliases or redirects for compatibility. |
| RW-UAT-004 | Model catalog loads selectable configured models | TASK-478.1 | Model selector | Pass | Live CDP selected configured `Ollama / gemma3:1b`; fresh tab reported 0 console errors after Add Sources -> My Media. | `modelSelectorUtils.test.ts`; `ChatPane.stage2.test.tsx`. | Maintain provider metadata normalization across shared selectors. |
| RW-UAT-005 | Missing-model and failed-response states are recoverable | TASK-478.2 | Chat composer/RAG send | Pass | Live CDP confirmed missing-model sends made zero chat requests, draft was preserved, invalid provider rendered recoverable 503 error, empty stream rendered `No response was returned.` | Focused chat/RAG frontend tests for missing model, failed submit, empty stream, and request normalization. | Continue checking provider-specific failure copy during model/provider changes. |
| RW-UAT-006 | First-class workspace source ingestion/indexing status exists | TASK-478.3 | Workspaces API, Jobs | Pass | Live backend validation exposed workspace-source Jobs in `/api/v1/jobs/list` and `/sources/status`, including progress/error details for missing media. | Workspace status API and media ingest worker tests passed in TASK-478.3. | Need future live matrix rows for long-running vector completion with real embeddings enabled. |
| RW-UAT-007 | Source cards do not claim fully ready when API says partial | TASK-478.3 | Sources pane | Pass | TASK-478.3 live CDP confirmed source cards aligned with status projection after backend projection fix. | Frontend regression prevents legacy media fallback from overriding authoritative partial status. | Recheck when vector/indexing providers change. |
| RW-UAT-008 | Individual, bulk, and persisted source selection share one contract | TASK-478.4 | Sources pane, Workspaces API, RAG | Pass | Live CDP seeded two media documents, selected one, saw `/sources/selection` and `/sources/status` agree, and saw `/api/v1/rag/search` include only the selected media ID. | Store/API-first selection tests and server reconciliation tests. | Keep selection intent separate from queryable media IDs for processing sources. |
| RW-UAT-009 | Grounded selected-source RAG Q&A returns evidence/citations | TASK-478.5 | Chat/RAG | Pass | Live WebUI returned `/api/v1/rag/search` documents with `include_media_ids [8,7]`, visible answer containing `PASTE-EVIDENCE-ORION`, and expanded citations titled with source names. | `ragMode.sanitization.test.ts` plus chat-mode tests. | Continue verifying with at least one configured local provider per release. |
| RW-UAT-010 | Studio enables and generates from selected sources | TASK-478.6 | Studio pane | Pass | Live Playwright/CDP generated a completed summary artifact with local provider response HTTP 200; screenshot `/private/tmp/task4786-studio-summary-live.png`. | StudioPane focused suites and ResearchWorkspace stage tests. | Recheck saved artifact persistence when workspace artifact storage changes. |
| RW-UAT-011 | Add Sources URL validation is explicit and recoverable | TASK-478.8 | Add Sources modal | Pass | Live CDP validated invalid URL inline feedback with 0 console errors and no silent stall. | `AddSourceModal.stage2.intake.test.tsx`. | Extend to duplicate/partial batch ingestion when backend receipts are finalized. |
| RW-UAT-012 | My Media search handles exact matches and already-attached media | TASK-478.8 | Add Sources modal | Pass | Live CDP exact search for `research-workspace-uat-source.md` showed the already-in-workspace explanation instead of unrelated drift. | `AddSourceModal.stage2.intake.test.tsx`. | Add pagination regression if media result sets become large. |
| RW-UAT-013 | Paste/file acquisition creates visible workspace sources | TASK-478.8, TASK-478.9 | Add Sources modal, Sources pane | Pass | TASK-478.8 live CDP paste created `Gate D Paste Smoke`; TASK-478.9 uploaded `/private/tmp/task4789-live-source.md` and saw source creation return 201. | AddSourceModal and source-preview tests. | Keep tied to status projection so acquisition and readiness do not diverge. |
| RW-UAT-014 | Source preview shows captured content/evidence without dumping unbounded payloads | TASK-478.9 | Source preview modal, API | Pass | Live validation opened preview for uploaded source, saw captured content, chunk snippets, citation-ready state, and persisted browser-local annotation; screenshot `/private/tmp/task4789-source-preview-live.png`. | 13 backend tests, 55 frontend tests, OpenAPI guard for preview/status capability. | Add server-persisted annotations if browser-local is no longer acceptable. |
| RW-UAT-015 | Context envelope exposes page capability/readiness/service state | TASK-478.9 | Workspaces API, UI warnings | Pass | TASK-478.9 kept page context envelope separate from bounded preview endpoint and surfaced partial errors as compact Sources-pane warnings. | Workspace API status/capability tests. | Keep envelope small; do not move source previews into page bootstrap payloads. |
| RW-UAT-016 | Responsive layout remains usable at desktop and 390px mobile | TASK-478.10 | Workspace shell | Pass | Live backend/WebUI validation passed at 1365x900 and 390x844; screenshots `/private/tmp/task47810-desktop-source-advanced-after.png` and `/private/tmp/task47810-mobile-source-advanced-after.png`. | Desktop layout and SourcesPane transfer/layout tests. | Continue to include advanced filters open state in responsive checks. |
| RW-UAT-017 | Keyboard/focus paths are reachable for dense workspace controls | TASK-478.10 | Workspace shell | Partial | TASK-478.10 covered labels/focus for primary workspace controls and no page-level scroll; full keyboard-only source-to-chat walkthrough is not yet a dedicated E2E. | Focused UI tests plus live layout assertions. | Add a keyboard-only UAT row before final release gate. |
| RW-UAT-018 | First-run copy gives next actions without extra banner clutter | TASK-478.11 | Empty states, Sources pane, Chat pane | Pass | Live CDP asserted local/self-hosted copy present, Sources pane wording present, missing-model copy present, and rejected `workspace trust`/`left panel` copy absent. | Source-location copy, SourcesPane, ChatPane tests. | Keep local-first copy contextual at source/model decisions, not as persistent bars. |
| RW-UAT-019 | Start tour and Settings > Replay tour open visible walkthrough | TASK-478.11 | Guided tour | Pass | Live CDP against `/research-workspace` saw `tooltipCount: 1` and `overlayCount: 1` for both first-run Start tour and Settings Replay tour; screenshots `/private/tmp/task47811-first-run-tour.png`, `/private/tmp/task47811-settings-replay-tour.png`. | WebLayout runner and Research Workspace copy tests. | Recheck after any WebLayout or tutorial runner changes. |
| RW-UAT-020 | Research Workspace aligns with canonical Shared Workspaces model | TASK-478.7 | Architecture/API/UI contract | Partial | Contract documented in `Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md`; live backend confirmed `/api/v1/research-workspace/capabilities` 200 and old Research Studio capabilities 404. | Capability endpoint/derivation tests; route metadata and extension route tests. | Remaining live UI affordances for Shared Workspaces/MCP/ACP/Sandbox are tracked separately. |
| RW-UAT-021 | MCP Hub Shared Workspaces is included in the workspace model | TASK-478.7 | MCP hub | Partial | Contract marks MCP Hub Shared Workspaces as canonical path/tool trust registry. Prior May 24 matrix saw hub route availability, but no end-to-end Research Workspace -> MCP workspace set binding was validated. | Backend MCP/capability tests from TASK-478.7. | Add a live workspace-set binding test when MCP UI/API fixture is stable. |
| RW-UAT-022 | ACP canonical bridge uses Research Workspace IDs/source labels | TASK-478.7 | ACP | Partial | Contract defines `/api/v1/agent-orchestration/workspaces/canonical-bridge` with `canonical_workspace_source: research_workspace`; focused ACP canonical tests passed. | Agent orchestration canonical/workspace DB/artifact promotion tests. | Add live ACP run history/filter UAT keyed by canonical workspace ID. |
| RW-UAT-023 | Sandbox diagnostics/admission are part of workspace handoff model | TASK-478.7 | Sandbox | Gap | Contract defines sandbox ownership and deferred diagnostics filters; no live Research Workspace -> sandbox admission/diagnostics flow is exposed yet. | None specific to a live Research Workspace flow. | Create a sandbox handoff task before claiming agent/tool workspace completeness. |
| RW-UAT-024 | Browser extension capture targets canonical workspace/source IDs | TASK-478.12 | Browser extension, WebUI | Blocked | TASK-478.12 is To Do and depends on a current extension build. No current extension CDP handoff run is available. | Existing extension route tests were updated in TASK-478.7, but live capture is not verified. | Resume TASK-478.12 when extension build issue is resolved. |
| RW-UAT-025 | Migration/import/export recovery is clear and resumable | TASK-478.3, migration API work | Migration APIs/UI | Partial | Migration API idempotency and conflict handling were fixed in prior PR review cycles; current WebUI migration wizard/recovery walkthrough is not part of this live pass. | Backend migration tests from earlier workspace migration work. | Add a dedicated migration recovery UAT row/task if migration UI is in release scope. |
| RW-UAT-026 | Maintained matrix and regression gate exist | TASK-478.13 | Docs, E2E | Pass | Current matrix exists in this document. Live probe passed route, empty-state copy, rejected-copy absence, tour overlay, and critical console/page-error checks. Focused Playwright route regression passed: `1 passed (2.4s)`. | `research-workspace.real-backend.spec.ts` route contract test. | TASK-478.12 still owns extension handoff once the extension build is available. |

## Current High-Risk Remainders

1. Browser extension handoff is still blocked by build availability and must not
   be represented as working until TASK-478.12 runs CDP validation.
2. MCP/ACP/Sandbox are correctly part of the workspace model, but several live
   user flows are still `Partial` or `Gap`; the current contract is not the same
   as full workflow completion.
3. Full frontend TypeScript verification has been blocked in multiple child
   tasks by unrelated pre-existing Watchlists/e2e errors. Focused tests passed,
   but this remains a release hygiene risk outside TASK-478.13.
4. Long-running vector indexing with real embedding completion should remain a
   Watch row even though the Jobs/status projection is now first-class.

## How To Update This Matrix

1. Add or update one row per user-visible workflow, not one row per component.
2. Record the owning Backlog task and exact live evidence. Use screenshot paths
   only as supporting evidence; prefer DOM/API assertions or test names.
3. Mark extension, MCP, ACP, Sandbox, or provider-dependent rows as `Blocked`,
   `Partial`, or `Gap` unless a current live run proves the complete workflow.
4. If a row regresses, update the status immediately and create/link a child
   Backlog task before editing product code.
5. Keep `/workspace-playground` removal as a permanent regression check: 404,
   no `Location` header, no client-side redirect, and no route alias.
