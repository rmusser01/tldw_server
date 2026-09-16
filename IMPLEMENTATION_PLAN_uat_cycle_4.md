# UAT cycle4 implementation plan

> For agentic workers: use systematic-debugging and test-driven-development. Apply dispatching-parallel-agents only to the independent file ownership listed below, with independent review before committing each repair unit. Continue authorized repairs without a new permission checkpoint.

**Goal:** Repair cycle4 findings and confirmed scope gaps, then repeat the authoritative fresh single/multi workflow matrices.
**Architecture:** Retain current UI/store/API boundaries; repair missing acknowledgements, competing state writers, stale configuration and inaccurate failure outcomes at their owners.
**Tech stack:** Next.js, shared React/Zustand/Dexie, Vitest/Playwright, FastAPI/pytest/SQLite and real llama.cpp.
**Spec:** [Cycle4 repair design](Docs/Design/2026-09-16-uat-cycle-4-repairs.md).

## Global constraints

- Preserve authentication, role permissions, account isolation, source classification and external restrictions.
- Preserve unsaved drafts, prior successful source/analysis and canonical identity; no broad text deduplication or store reset.
- No live credentials in logs, tests, commits or retained artifacts. No mocked response counts as real-model acceptance.
- Every repository edit belongs to an existing Backlog task, updated only through official MCP/CLI.
- Use current libraries and actual component/store/service boundaries. Reuse retained failing probes as evidence, then add permanent behavioral regressions before production edits.
- Activate `.venv` before Python/pytest/Bandit. Run Bandit on touched Python and resolve new findings.
- Commit reviewable units, preserve unrelated work, and retain exact command/results. Existing TypeScript/lint baselines must be compared, not called clean.
- Independent review and targeted native verification precede another full fresh single/multi run. External blocked rows remain explicit.

## Stage1: Preserve the frozen run and update dev
**Goal:** Make the tested source and current repair baseline unambiguous.
**Success criteria:** Complete reports/hash checks committed; freshly fetched dev is an ancestor; affected imported checks pass or report explicit host limits.
**Tests:** Evidence hash/credential scan, Git ancestry and changed sandbox regression suites.
**Status:** Complete

- [x] Preserve both FINAL_REPORT matrices,203single captures,62diagnosis artifacts and independent audit; checkpoint `b2595a4edb`.
- [x] TASK13260.32: merge fresh dev59049e094e via `f4e9f954d7`, zero dev-only commits.
- [x] Imported regression verification:170pass/1hostskip initially, same two `ps`-denied cases pass with required process-inspection permission.172passing cases total; retained `output/playwright/cycle4-repair-verification-2026-09-16/dev-integration/`.

## Stage2: Repair state and outcome boundaries
**Goal:** Fix the major defects with permanent reproductions and bounded interfaces.
**Success criteria:** Each failing behavior becomes green while its draft/account/positive controls remain green.
**Tests:** Actual pipeline/component/store tests and backend caller tests specified per unit.
**Status:** In Progress

Reviewed code checkpoints: Task1/4 `14f33af27c`; Task2 `0dcca3c032`; Task3 `8128c93c1b`. Permanent regression and independent review evidence is in the tracker repair table. Native acceptance remains pending, so combined acceptance/commit bullets below stay incomplete.

Targeted native setup at03:37UTC reopened Task2/UAT106: the browser has no API key during the permitted local first-run flow, so the protected model service returns an empty catalog. A real successful same-target first-chat response and server completion are incorrectly blocked by the client catalog-match requirement. Correct the handoff using the authoritative verified pair with existing canonical-provider/owner guards; retain protected catalog authentication. This correction requires an actual model-service credential-boundary regression and independent review before the next full run.

Correction`8097d672d5` is independently reviewed and now passes a second fresh native setup→ordinary Chat without an API restart. Native103canonical reload/111Note backlink and104Minimize also pass. Targeted checks exposed two remaining boundaries before the full run:105projects a saved-source Warning as generic failure in the frontend, and108Retry sends correct context but duplicates the failed user row in canonical server persistence. Existing tasks46/49 own these bounded corrections. Their independent file ownership permits parallel implementation; root continues native analysis/UI checks. No full fresh run begins before these are corrected and reviewed.

### Task1: Canonical user acknowledgement — TASK13260.44/51, UAT103/111
**Files:** `apps/packages/ui/src/models/ChatTldw.ts`, `hooks/chat-modes/chatModePipeline.ts`, `hooks/chat/useChatActions.ts`, `hooks/chat-helper/index.ts`, their existing tests; `tldw_Server_API/app/core/Chat/chat_service.py`, streaming metadata helper and associated Chat tests as needed.
**Inputs:** `read-only-diagnoses/uat103-read-only-diagnosis.md`, `uat111-read-only-diagnosis.md`, retained RED probes; paths relative to cycle4 evidence root.
**Interface:** Add optional canonical user acknowledgement beside existing assistant metadata. Consume it at local persistence with the captured owner. Character path passes its already-created canonical user ID.
- [ ] Port the real-boundary RED: successful two turns → server5 rows → reload5; identical unsent draft remains; Note backlink accepts saved history.
- [ ] Trace additive acknowledgement through normal streaming/nonstreaming and character persistence; keep legacy/error/cancel compatibility.
- [ ] Implement minimal acknowledgement and defensible legacy recovery; test exact repeated turns, delayed owners, partial persistence and ambiguous drafts.
- [ ] Run existing affected Chat suites, backend metadata tests and scoped Bandit; independent review, targeted live normal/character reload and backlink; commit with tasks.

### Task2: Consolidated model handoff — TASK13260.47/55, UAT106/115
**Files:** `components/Option/Onboarding/UnifiedSetupWizard.tsx`, `steps/FirstChatStep.tsx`, `hooks/chat/useSelectedModel.ts`, `components/Media/AnalysisModal.tsx` and their existing tests (all under shared UI).
**Inputs:** retained `uat106-readonly-diagnosis.md`, `uat-cycle4-model-and-connections-diagnosis.md`, two failing actual-storage probes.
**Interface:** Existing consolidated selected-model setter owns store and durable storage; one-time setup publication precedes completion publication.
- [ ] Permanent RED controls: fresh verified custom model wins over first Ollama; deliberate/newer selection survives; actual Media+shared owner sends explicitly chosen model.
- [ ] Resolve canonical provider-qualified identity and publish only for the captured eligible setup owner; update Media's explicit setter.
- [ ] Verify missing/ambiguous catalog, stored normalization, cancel/error, delayed completion and account changes; run focused suites, review and native request-body check; commit.

### Task3: Safe summarization outcome — TASK13260.46, UAT105
**Files:** `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`; `tests/LLM_Calls/test_summarization_adapter.py`, `tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py`, existing persistence/job test fixtures.
**Input:** retained `uat105-readonly-diagnosis.md` and offline actual-caller RED.
**Interface:** Valid nonblank answer remains text; invalid/length-truncated output uses sanitized existing failure contract, which preserves source with Warning/no bad analysis.
- [ ] Add RED empty/null/whitespace/malformed/reasoning-only/length cases, good envelope and plain-text controls, typed/legacy errors and sentinel privacy checks.
- [ ] Remove arbitrary response stringification; reject truncated analysis without leaking response metadata.
- [ ] Exercise real summarizer→plaintext→persistence/job outcome with provider transport mocked; source survives, no raw envelope in analysis/chunk metadata, terminal warning retained.
- [ ] Run targeted pytest/Bandit, independently review and verify native ingestion warning/success paths; commit.

### Task4: Refresh model inventory — TASK13260.48, UAT107
**Files:** `tldw_Server_API/app/core/Chat/chat_service.py`, setup/config invalidation seam if necessary; existing configured-model/setup tests.
**Dependency:** After Task1 releases chat_service.py. No concurrent edits of that module.
**Input:** retained `uat107-108-read-only-diagnosis.md` and AST reproduction.
- [ ] RED: import/warm old inventory → save advertised custom model via actual setup boundary → ordinary strict validation succeeds without restart; unknown model still fails.
- [ ] Read current effective config and invalidate/key/remove stale inventory cache; preserve aliases, env precedence and numbered slots.
- [ ] Run affected backend setup/Chat tests and Bandit, review and native setup→ordinary Chat without restart; commit.

## Stage3: Repair recovery and background behavior
**Goal:** Make failed or incomplete requests truthful and keep navigation/network usable.
**Success criteria:** Recovery acts on the intended turn/request, local failure UI contains expected errors, and hidden tabs release streams.
**Tests:** Actual request projection, stream cancellation, QA completed-request states, visibility transitions and native controls.
**Status:** In Progress

Task7 reviewed code committed `e4552e4764` (111 independent tests); native six-tab acceptance pending. Task6 reviewed code committed `3c159b219b` (248 independent covering tests; author359 broader cases). Task5 committed `f83a8ba456` after independent11Retry/173remaining controls and combined1299frontend tests/84suites. Whole TypeScript matches90existing diagnostics with none added. All code checkpoints are reviewed; targeted native acceptance follows before the full fresh run.

### Task5: Chat retry, character route and missing final answer — TASK13260.49/53/57, UAT108/113/117
**Files:** `utils/generate-history.ts`, `hooks/chat/useChatActions.ts`, `components/Option/Playground/PlaygroundForm.tsx`, `components/Option/Playground/Playground.tsx`, session persistence and existing tests.
**Dependency:** Task1 stable; coordinate any model form changes with Task2.
- [ ] RED actual Retry projection excludes recognized assistant display error, contains failed user exactly once and preserves user-quoted/malformed markers and partial assistant prose.
- [ ] Use existing failed-turn regeneration/history override; do not broadly deduplicate request text.
- [ ] RED character entry→save→normal reload restores saved rows; canonicalize consumed route while preserving explicit same/different new-character entry and account guards.
- [ ] RED closed/unclosed think-only and structured-reasoning-only completion yields recoverable missing-final-answer status; good final answer still succeeds and history/reasoning survive.
- [ ] Run affected Chat/session/message tests, independent review and native Retry/character reload/empty-answer checks; commit separately where reviewable.

### Task6: Contain failures and clarify QA — TASK13260.50/56, UAT109/110/116
**Files:** `services/background-proxy.ts`, KnowledgeQA provider, `AnswerPanel.tsx`, `panels/AnswerWorkspace.tsx`, `components/Media/AnalysisModal.tsx`, affected tests.
**Dependency:** Task2 releases AnalysisModal. Task5 owns Chat files only.
- [ ] RED reader.cancel rejection is owned and expected timeout/provider errors show local recovery without overlay; cancel is distinguished and prior analysis preserved.
- [ ] RED requested generation with empty answer keeps sources and shows missing-output retry guidance; disabled generation keeps its own guidance.
- [ ] Bind guidance to completed request settings; clear stale assertive timeout text after recovery; cover failure/insufficient-evidence/normal-answer controls.
- [ ] Run actual provider/panel/stream tests, independent review and native timeout/recovery; commit.

### Task7: Bound hidden-tab streams — TASK13260.54, UAT114
**Files:** `apps/tldw-frontend/components/notifications/NotificationLifecycleProvider.tsx`, existing lifecycle tests.
- [ ] RED hidden mount opens no stream; visible→hidden cancels; return visible catches up without duplicate events.
- [ ] Add visibility-aware lifecycle using existing subscription/cleanup ownership, preserving rotation/cursor/account/read permission and bounded polling.
- [ ] Run StrictMode/no-overlap/rotation/A→B→A regressions, independent review and actual six-tab requests plus one Prompt save; commit.

## Stage4: Close presentation gaps
**Goal:** Repair remaining visible/accessible interaction defects and classify secondary observations.
**Success criteria:** Minimize dismisses, actions have stable names, context feedback/title/form lifecycle has no newly observed warnings.
**Tests:** Actual modal, AntD feedback/form lifecycle, title owner and accessibility behavior.
**Status:** In Progress

Task8 code reviewed/committed: feedback102 andMinimize104 `6d2f3abc55`, accessibility112 `db141cbd90`; native controls pending. Task9 titles/form committed `eb8782e5fd` after independent33title/3focused form controls and author135cases/8files. Review Markdown060 correction committed `74f5a0ee1e` after independent33-test review: existing safe renderer reused in the analysis reading pane, preserving raw copy/edit/export data. Combined backend353pass/1existing skip and production Bandit0; combined frontend/compiler checks follow Task5 freeze.

### Task8: Minimize and feedback/accessibility — TASK13260.45/43/52, UAT104/102/112
**Files:** shared `components/Common/QuickIngest/ProcessingStep.tsx`, `QuickIngestWizardModal.tsx`; actual admin creation and Prompt sync feedback owners; `components/Flashcards/tabs/ManageTab.tsx` and affected action/loading components; existing tests.
- [ ] RED actual processing button→modal owner closes, background job survives and Resume restores; use existing dismiss callback, not cancel.
- [ ] RED actual admin success and Prompt sync failure preserve behavior without static-context feedback warning; use existing App context.
- [ ] RED FAB has stable meaningful accessible name and completed enabled buttons exclude decorative loading labels; preserve pending/disabled behavior.
- [ ] Focused test/lint review and targeted native controls; separate commits by Backlog unit.

### Task9: Titles, character form and Review triage — TASK13260.20/15 plus parent13260
**Files:** `apps/tldw-frontend/pages/prompts.tsx`, `pages/characters.tsx`, actual title owner and New-character form; route/form tests.
- [ ] RED actual Prompts/Characters titles and late Chat completion across Settings reconnect; repair route-owned publication without stale metadata.
- [ ] Reproduce exact New-character disconnected useForm warning, then fix actual mount/unmount access boundary and preserve create/cancel/reopen.
- [ ] Compare Review Markdown presentation to intended renderer/nearby tests. Record reasoned defect/expected classification; create/find child task before any newly scoped repair.
- [ ] Run affected tests/lint, independent review and native titles/form checks; commit.

## Stage5: Integrated verification and another full fresh UAT
**Goal:** Verify repaired behavior together, then execute both full named matrices on frozen code.
**Success criteria:** Identified product issues have reviewed fixes and targeted native evidence; new fresh matrices report every required row with explicit limits.
**Tests:** Combined affected frontend/backend suites, TypeScript baseline comparison, scoped lint/Bandit, independent review, real browser/model workflows.
**Status:** Not Started

- [ ] Update tracker per issue with code checkpoint, targeted evidence and any remaining gap; do not rewrite historical failures into passes.
- [ ] Run combined checks once interfaces settle; compare known90 TypeScript signatures and known lint warnings/errors, resolve new findings.
- [ ] Independent integration review; correct confirmed findings with regressions and scoped rereview.
- [ ] Freeze reviewed source. Start new empty configuration/data/browser profiles for both modes; preserve old profiles/evidence. Reuse dependencies transparently.
- [ ] Run exact named journeys/shared workflows: setup, normal/character Chat, file+Wikipedia ingestion, QA/citations/source handoffs, Notes/Prompt/cards/Study, Media analysis/Review/delete/restore, roles/isolation/offline and token refresh controls. Record every encountered issue immediately.
- [ ] Retain credential-scanned evidence and independent report review; continue repairs for confirmed issues. Remove only this plan when its work is actually complete.

## Ownership and sequencing

Initial independent units: Task1 Chat acknowledgement, Task2 model selection, Task3 summarization; controller owns evidence/dev checks and Task8 Quick Ingest only. Task4 waits for Task1's backend file. Task5 waits for Task1; Task6 waits for Task2's AnalysisModal. Task7 and presentation units can run with those once a slot is free. No concurrent writers to the same file, no worker-owned broad staging/commits; controller integrates after review. Native inference is serialized and runtimes remain unchanged until a coordinated rebuild.
