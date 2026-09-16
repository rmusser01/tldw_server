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

Frontend105 correction is reviewed and committed`d94077c354`:359focused cases plus independent173permanent/2poller probes pass. The results UI separates saved warnings from clean successes and retains Media navigation. A review-discovered pre-existing nested unknown/cancelled classification gap is covered by a narrow guard. Native115analysis selection/save and110local failure preservation passed;113new Character reload passed in preserved multi. Browser checks are paused during108source edits/HMR, with both targeted Next servers stopped;060Review/109QA/116 and the corrected105native check resume after freeze.114native hidden-tab coverage remains explicitly unverified due harness/native-control limits.

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

Final108source review is clear after an actual successful-Regenerate probe found an overly broad local ACK update; explicit failed-only intent now gates it. Independent113frontend/76backend controls pass. Combined1467frontend/367backend+1existing skip passed before that final frontend-only correction, and final compiler retains exactly90baseline diagnostics. Native105/108/060/109/116 controls resume on intentionally restarted isolated APIs; the existing107no-restart proof is preserved.

Targeted native105/060/109/116 now pass: source-preserving warning and Media link, Review Markdown reload, real stream timeout/local recovery, and generation-enabled missing-answer guidance.108 follow-up `7dcee3d72c` remains incomplete: after the initial502, the loader materializes another unacknowledged user before Retry; Retry then persists both. A permanent mounted action+loader+mirror RED reproduces this. TASK13260.49 adds durable identity correlation and rejection of Retry requests that retain extra user rows, with independent review. Both Next frontends are stopped for this repair. No full fresh matrix has started.

Independent targeted-evidence audit confirmed179 retained hashes and the105/060/109/116 claims. Before the full run, finish three remaining native boundaries alongside the108 recheck:102 Prompt sync-failure feedback,058 late Chat completion/title ownership during Settings navigation/reconnect, and117 a newly completed reasoning-only response. Restored117 transcript presentation remains a narrower positive control.

The ordinary-text108 correlation follow-up passes independent review and combined 1,346 UI / 383 backend cases (one existing skip); final loader safety controls pass separately. Bandit is clear and the 90-diagnostic compiler baseline is unchanged. Native verification is still required.

### Additional repair: failed image-bearing Chat identity — TASK-13260.58, UAT-118

**Goal:** Preserve and recover unchanged user image attachments through the actual canonical listing and Retry boundary.
**Success criteria:** No duplicate visible or canonical user for an unchanged failed image-bearing turn; changed or ambiguous local work stays intact; account ownership remains enforced.
**Tests:** Actual endpoint/listing/client/mapper controls for text + image, image-only, changed content, intentional repeats and legacy data; independent review and targeted native verification.
**Status:** In Progress — approved design is recorded in `Docs/Design/2026-09-16-uat-image-recovery.md`. Native checks are paused and the author has a disjoint ten-file production window. Bounded hunks in `normalChatMode.ts` and `messageHandlers.ts` preserve qualified image data URLs and permit image-only Retry without trimming the user's content: mounted regressions exposed PNG-to-JPEG local-history relabeling and a text-only Retry gate. Both loader history projections also retain the existing image field. Exact MIME/byte matching stays strict. Opt-in bounded attachment reads, conservative synthetic-placeholder projection and exact failed-Retry matching are being verified at actual endpoint/client/loader boundaries; independent source review follows the freeze. The single-image frontend history contract is not represented as general multiple-image Retry support; ambiguous extra attachments must remain intact.

### Additional presentation repair — TASK-13260.59, UAT-119

**Goal:** Give the full Prompt editor and mobile preview opaque backgrounds using the existing `bg-bg` theme token.
**Evidence:** Native screenshot and computed style show the current unsupported `bg-background` leaves the library and header visible through the editor.
**Tests:** Existing editor controls, scoped lint and dark/light native screenshots with computed background checks. This class-only correction does not need an implementation-mirroring unit test.
**Status:** Reviewed and committed as `a8e5430d76`: four existing theme-token replacements,10 editor tests passing, generated opaque Tailwind CSS and unchanged lint. Native dark/light acceptance follows the remaining source freeze.

## Stage5: Integrated verification and another full fresh UAT
Additional independent repair ownership: TASK13260.60 / UAT120 belongs to the Prompt sync service and its existing behavioral tests; `review_ingest065` implements, then a separate reviewer verifies. Actual normalized transport failure must persist Pending without weakening structured-recipe uncertainty protections. TASK13260.51 / UAT111 is reopened by a real no-navigation Note backlink; `account_access` investigates the callback/router boundary. Neither overlaps root's Prompt background file or the Chat attachment unit. All source edits wait until the current native window closes.

The native window closed after058 Settings title/disconnect/reconnect passed and117 produced a valid final answer. Missing Character token settings are new121 / TASK13260.61, owned by `account_access` after the separate111 repair. Source edits are now released with disjoint modules/tests; coordinate before touching another owner's tests. Root's119 four-class correction is reviewed/committed `a8e5430d76`; native dark/light acceptance waits for the next stable window. No full fresh cycle5 has started.

Follow-up independent review is clear for111 and120. Root repeated136 tests/4 suites, including24 Note cases after adding image-draft preservation and the actual normalized Prompt network-failure/remount/acknowledgement boundary. Commits `142cc2f1cc` and `28dc9ddffa` retain code, task notes and evidence.121's bounded four-setting Character mapping is independently verified and committed045a953883.118 source/tests are frozen and independently reviewed; combined456 backend checks pass with2 skips. Shared UI2547 passed with one UAT120 fixture failure, fixed/verified separately at17ae5f2292. Bandit8production paths is clear. API fingerprint/types were regenerated with existing local package source paths and the drift check passes. Targeted native acceptance remains pending. Current-chat settings retain their existing in-memory lifetime, so no persistence feature is added for a blank post-reload field.

**Goal:** Verify repaired behavior together, then execute both full named matrices on frozen code.
**Success criteria:** Identified product issues have reviewed fixes and targeted native evidence; new fresh matrices report every required row with explicit limits.
**Tests:** Combined affected frontend/backend suites, TypeScript baseline comparison, scoped lint/Bandit, independent review, real browser/model workflows.
**Status:** In Progress — integrated review and checks complete with the explicit baseline/skips above; targeted native checks and full fresh run remain.

- [ ] Update tracker per issue with code checkpoint, targeted evidence and any remaining gap; do not rewrite historical failures into passes.
- [ ] Run combined checks once interfaces settle; compare known90 TypeScript signatures and known lint warnings/errors, resolve new findings.
- [ ] Independent integration review; correct confirmed findings with regressions and scoped rereview.
- [ ] Freeze reviewed source. Start new empty configuration/data/browser profiles for both modes; preserve old profiles/evidence. Reuse dependencies transparently.
- [ ] Run exact named journeys/shared workflows: setup, normal/character Chat, file+Wikipedia ingestion, QA/citations/source handoffs, Notes/Prompt/cards/Study, Media analysis/Review/delete/restore, roles/isolation/offline and token refresh controls. Record every encountered issue immediately.
- [ ] Retain credential-scanned evidence and independent report review; continue repairs for confirmed issues. Remove only this plan when its work is actually complete.

## Ownership and sequencing

Initial independent units: Task1 Chat acknowledgement, Task2 model selection, Task3 summarization; controller owns evidence/dev checks and Task8 Quick Ingest only. Task4 waits for Task1's backend file. Task5 waits for Task1; Task6 waits for Task2's AnalysisModal. Task7 and presentation units can run with those once a slot is free. No concurrent writers to the same file, no worker-owned broad staging/commits; controller integrates after review. Native inference is serialized and runtimes remain unchanged until a coordinated rebuild.

## Additional send-boundary repair — TASK13260.62 / UAT122

Native image118 verification found the composer/local transcript retaining an image while actual transport and canonical persistence drop it for supportsMultimodal=false. Root recorded the exact request and canonical IDs. Chat author owns read-only minimal design initially; source edits wait until both Next runtimes are paused. Actual formatter/model/action tests must cover unsupported handling, supported MIME preservation, image-only input and Retry. Full fresh cycle5 stays pending.

119/120 native checks pass with the explicit offline-readiness-gate adaptation.111/121/actual new117 negative+reload pass. Separate direct-route/prior-mode observations are being classified before the next source window.

## Additional saved-conversation mode repair — TASK13260.63 / UAT123

Actual History/Note navigation loads an ordinary canonical chat while persisted new-chat Character mode keeps the wrong banner/gating active. Account-access author owns Playground.tsx/coordinator regression scope; canonical loaded metadata governs existing-chat mode, with explicit fresh Character and true unsaved drafts preserved. The manually constructed bare chatId URL is unsupported by current product producers and is excluded from implementation scope. Both Next runtimes paused; APIs/data unchanged.

### Current repair verification stages

1. Reproduce/design — Complete: native122 image omission and123 stale canonical mode; approved122 design in Docs/Design/2026-09-16-uat-image-input-validation.md,123 bounded derivation recorded above. Task62/63 own disjoint code.
2. Implement/regress — Complete: frozen122 three production files plus tests/test alias; frozen123 one production file plus existing coordinator/Notes tests. Author and independent12293tests pass; root12374tests pass. Owner, Retry provenance, unsaved-work and fresh Character controls retained.
3. Review/targeted — In Progress: independent source reviews clear; scoped lint no new findings; full compiler exactly90 baseline diagnostics. Broader affected frontend run passed2593tests/95files. Commit and native no-loss refusal/History/Note mode checks follow.
4. Full fresh cycle5 — Not Started: only after these repairs and preserved evidence, use prepared fresh protocol. Existing vision/Postgres/hidden-tab limitations remain explicit.

122 independent review refinement: actual backend proof requires distinct local failed-user identity and server failed-turn reuse. Approved narrow chatModePipeline provenance flag (thirdproductionpath), no backend relaxation. Current OCR remains intentional text; old historical OCR context was already lost, and its unproven image history will now fail visibly rather than be stripped. No re-OCR or global history exemption. Approved test-only Vitest alias to existing installed pa-tesseract, no dependency change.
