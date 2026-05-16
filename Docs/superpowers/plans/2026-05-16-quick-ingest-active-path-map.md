# Quick Ingest Active Path Map

Date: 2026-05-16
Backlog: TASK-393, TASK-394.1

## Active Launch Paths

| Surface | Trigger | Modal/Runtime | Evidence |
|---|---|---|---|
| Web app shell | Command palette `onIngestPage` dispatches `tldw:open-quick-ingest` | `QuickIngestModalHost` lazy-loads `QuickIngestWizardModal` | `apps/tldw-frontend/components/layout/WebLayout.tsx:30` imports `QuickIngestModalHost`; `:267` dispatches `tldw:open-quick-ingest`; `:543` renders `<QuickIngestModalHost />`. `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx:13-15` imports `../Common/QuickIngestWizardModal`; `:139` listens for `tldw:open-quick-ingest`; `:366` renders `<QuickIngestModal ... />`. |
| Shared layout shell | Mounted event-only host | `QuickIngestModalHost` lazy-loads active wizard | `apps/packages/ui/src/components/Layouts/Layout.tsx:25` imports `QuickIngestModalHost`; `:556` renders `<QuickIngestModalHost />`. `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx:376-386` defines host render path using lazy `QuickIngestModal`. |
| Header/add-content button | Click `data-testid="open-quick-ingest"` | Same `QuickIngestButton` state/session host and active wizard | `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx:313-317` button calls `openQuickIngest()` and exposes `data-testid="open-quick-ingest"`; `:65-74` shows/resumes existing session or creates a draft session. |
| Chat sidepanel | Chat form renders modal directly from sidepanel session state | Active `QuickIngestWizardModal` imported as `QuickIngestModal` | `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx:84` imports `QuickIngestWizardModal as QuickIngestModal`; `:3591-3594` renders it with `open={ingestOpen}` and `autoProcessQueued`. |
| Extension sidepanel command palette | Dispatches `tldw:open-quick-ingest` | Extension uses same host/modal path when mounted | `apps/tldw-frontend/extension/routes/sidepanel-chat.tsx:1947-1949` dispatches `tldw:open-quick-ingest`; `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx:139` is the listener. |
| Extension background runtime | Quick ingest modal session messages and cancellation | Background tracks modal sessions, abort controllers, and cancellation | `apps/packages/ui/src/entries/background.ts:279-283` defines `QuickIngestModalSession`; `:442` stores `quickIngestModalSessions`; `:467-478` looks up/cancel-checks sessions; `:2410-2413` marks a modal session cancelled. |
| Chat sidebar shortcut | Event action `quick-ingest` | Dispatches active host event | `apps/packages/ui/src/components/Common/ChatSidebar/shortcut-actions.ts:40-46` maps `quick-ingest` to `eventName: "tldw:open-quick-ingest"`. |

## Legacy Reachability Decision

| File/Test | Current status | Decision | Rationale |
|---|---|---|---|
| `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx` | Active wizard modal | Keep as current target | `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1550` exports `QuickIngestWizardModal`; `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx:13-15` lazy-loads this component for layout launches. |
| `apps/packages/ui/src/components/Common/QuickIngestModal.tsx` | Legacy monolithic modal | Treat as legacy/unreachable from current launch paths unless imported directly by legacy tests | `apps/packages/ui/src/components/Common/QuickIngestModal.tsx:131` exports `QuickIngestModal`, but active launch imports resolve to `QuickIngestWizardModal` in `QuickIngestButton.tsx:13-15` and sidepanel `form.tsx:84`. |
| `apps/packages/ui/src/components/Common/__tests__/QuickIngestModal.session-cancel.test.tsx` | Legacy unit coverage | Stale for active UX remediation; migrate or replace before relying on it | `apps/packages/ui/src/components/Common/__tests__/QuickIngestModal.session-cancel.test.tsx:177` imports `QuickIngestModal` from the legacy file; `:294-310` drives `quick-ingest-run` / `quick-ingest-cancel` on the legacy component, not active wizard launch. |
| `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx` | Active wizard session runtime coverage | Keep and extend for wizard runtime/session changes | `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx:240` imports `QuickIngestWizardModal`; later tests render it directly. |
| `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx` | Active wizard flow coverage | Keep and extend for step/selector regressions | `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx:397` describes `QuickIngestWizardModal - full wizard flow integration`. |
| `apps/packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx` | Active host/session launch coverage | Keep for shell trigger/session resume behavior | `apps/packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx:52-53` mocks `QuickIngestWizardModal`; `:110` clicks `open-quick-ingest`; `:169-185` renders `QuickIngestModalHost`. |

## Test Classification

| Test file/helper | Current wizard | Legacy reachable | Stale selector | Missing coverage |
|---|---:|---:|---:|---|
| `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx` | Yes | No | No | Likely needs targeted assertions for remediation-specific UX states, but it is the right unit/integration home. |
| `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx` | Yes | No | No | Good session-runtime home; extend here for active cancel/resume/session edge cases. |
| `apps/packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx` | Yes | No | No | Covers shell open/resume host behavior; does not cover full wizard body interactions. |
| `apps/packages/ui/src/components/Common/__tests__/QuickIngestModal.session-cancel.test.tsx` | No | Yes | Mixed | Imports legacy modal at `:177`; selector names overlap active `ProcessButton.tsx:88` and `:104`, so passing selectors would not prove active wizard behavior. |
| `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts` | Yes | No | Mixed | Quick Ingest block starts at `:420`; active helpers are imported at `:20-29`. Earlier file/upload/URL cases use generic page selectors and are not wizard coverage. |
| `apps/tldw-frontend/e2e/utils/journey-helpers.ts` | Yes | No | Mixed | Opens through `open-quick-ingest` at `:94`, dispatch fallback at `:127`/`:144`, and `quick-ingest-run` at `:164`/`:341`/`:753`; helper fallback paths can mask missing visible triggers. |
| `apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts` | Yes | No | No | Verifies active result CTA via `quick-ingest-open-media-primary` at `:101-104`; narrow result-summary coverage only. |
| `apps/extension/tests/e2e/quick-ingest-cancel.spec.ts` | Yes | No | No | Uses active selectors `quick-ingest-run` at `:132` and `quick-ingest-cancel` at `:136`; ensure it opens through active host before relying on it for launch UX. |
| `apps/extension/tests/e2e/quick-ingest-workflows.spec.ts` | Yes | No | Mixed | Uses visible `open-quick-ingest` at `:33`; classify as active launch smoke, not deep wizard coverage. |
| `apps/extension/tests/e2e/quick-ingest-file-upload.spec.ts` | Yes | No | Mixed | Opens via `open-quick-ingest` at `:232` with dispatch fallback at `:242`; useful for active file path, but fallback can hide trigger regressions. |
| `apps/extension/tests/e2e/quick-ingest.spec.ts` | Yes | No | Mixed | Uses `open-quick-ingest` at `:69` and `:78`; active launch smoke, likely too shallow for remediation acceptance. |
| `apps/extension/tests/e2e/live-ux-workflows.spec.ts` | Yes | No | Mixed | Uses `open-quick-ingest` at `:191` and `quick-ingest-run` at `:235`; live UX breadth, not a focused active wizard contract. |
| `apps/extension/tests/e2e/live-ux-review.spec.ts` | Yes | No | Mixed | Uses `quick-ingest-run` at `:183`; review/audit oriented and should not be the primary regression contract. |

## Follow-Up Notes

- Active product launch paths should be treated as `QuickIngestButton` / `QuickIngestModalHost` / `QuickIngestWizardModal`, not legacy `QuickIngestModal`.
- `quick-ingest-run`, `quick-ingest-cancel`, and `quick-ingest-open-media-primary` are active selectors in shared wizard pieces: `ProcessButton.tsx:88`, `ProcessButton.tsx:104`, and `ResultsPanel.tsx:324`.
- Legacy tests that import `QuickIngestModal` can still pass against overlapping selectors, but they do not validate the active wizard runtime.
- For the next implementation task, prefer extending active wizard tests before changing product code; only migrate legacy tests if the changed behavior needs their scenario coverage.
