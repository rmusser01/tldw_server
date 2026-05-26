# Flashcards UX Fix List

Status: closeout update after Phase 0 through Phase 5 remediation plus F06 task-first split follow-up.

Scope: `/flashcards` plus directly connected WebUI and extension flashcard workflows. This file is the master UX audit and fix-list source referenced by `Docs/superpowers/plans/2026-05-25-flashcards-ux-fixes-implementation-plan.md`.

## Evidence Summary

The original audit was based on a browser-observed WebUI pass with the backend running, plus route/component inspection for directly connected flashcards workflows.

Observed and inspected surfaces:

- `/flashcards`
- `/flashcards?tab=review&deck_id=1`
- Study, Manage, Create & Import, Templates, and Scheduler tabs
- Extension sidepanel flashcards route
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
- `apps/packages/ui/src/components/Flashcards/components/DeckStudyDashboard.tsx`
- `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`
- `apps/extension/docs/features/flashcards.md`
- `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`

Test setup assumptions from the original audit:

- Local single-user backend was available.
- Temporary flashcard deck/card data was used for live workflow probing and then removed.
- Multi-user sharing, workspace collaboration, and scheduler algorithm correctness were non-goals.

## Workflow Map

Actual flow after Phase 0-5 remediation plus the F06 follow-up:

1. Entry point: user opens `/flashcards`.
2. Empty first-run state lands on Study instead of a dense import utility screen.
3. Study provides onboarding and deck-level next actions when data is available.
4. Create/import/generate/export work is available through the visible `Create & Import` tab label while preserving the `importExport` route key.
5. `Create & Import` now starts with a task selector so Create cards, Import file, and Export backup workflows are separated without changing existing panel APIs.
6. Manual creation can be tested through stable Playwright selectors; failed create keeps user input and shows a visible error.
7. Review supports keyboard reveal/rate, visible undo/re-rate, recall-first assistant disclosure, and clearer completion actions.
8. Progress copy distinguishes scheduled due cards from the current available study queue.
9. Recent sessions show user-facing deck/mode/count/timing labels where data is available.
10. Deck dashboard rows expose direct Review, Cram, Edit, Scheduler, and Export actions.
11. Generated-card save recovery distinguishes success, partial success, failure, fatal validation errors, and retry state.
12. Extension sidepanel offers explicit full Flashcards and Generate from page selection actions, preserving page URL/title as supported manual source references.
13. Documentation describes the stabilized WebUI/extension handoff using current tab names.

## Phase Coverage

| Phase | Task | Findings covered | Result |
| --- | --- | --- | --- |
| Phase 0: Verification harness | TASK-477 | F05, F20 | Completed. Added create-drawer, failed-create, and keyboard-only review coverage. F05 was not retained as a product bug after e2e proof. |
| Phase 0 review fixes | TASK-503 | Phase 0 PR feedback | Completed. Addressed review feedback after PR #2064. |
| Phase 1: First-run defaults and labels | TASK-506, TASK-507 | F03, F04, F14, F15, F17, F18 | Completed. Study is first-run default, transfer limits render real values, Create & Import label is visible, Scheduler/Quiz/Manage empty states are clarified. |
| Phase 2: Review recovery | TASK-508 | F02, F07, F08, F10, F16, F19 | Completed. Added visible undo/re-rate, completion actions, assistant disclosure, available-now copy, shortcut parity, and clearer completed-session labels. |
| Phase 3A: Recent sessions | TASK-509 | F09, F19 support | Completed. Recent sessions use deck/mode/count/timing labels and API exposes reviewed counts. |
| Phase 3B: Deck dashboard | TASK-510, TASK-511 | F11 | Completed. Existing analytics data supports a deck-first dashboard with direct actions. Review fixes preserved session close behavior and dashboard switching. |
| Phase 4: Import/generate recovery | TASK-512 | F01 support, F06 support | Completed for generated-card save recovery. The later F06 follow-up completes the task-first IA split. |
| Phase 5: Extension capture and docs | TASK-513 | F12 support, F13 | Completed as an extension bridge and WebUI generate handoff. A fully native extension deck-picker/save flow remains deferred. |
| Closeout source restoration | TASK-514 | Planning traceability | Completed. Restores this tracked source file on `dev`. |
| F06 follow-up: Task-first Create & Import split | TASK-515 | F06 | Completed. Adds Create cards, Import file, and Export backup task workspaces while preserving existing route keys and panel handoffs. |

## Severity-Ranked Findings

| ID | Severity | Category | Evidence | User impact | Root UX cause | Remediation status |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | High | Error recovery | Partial/fatal create/import/generate operations could leave unclear success or failure state. | Users could believe cards were saved when only part of the work succeeded. | Async outcomes were not consistently surfaced as success, partial, retryable failure, or fatal validation. | Addressed for generated-card save recovery in TASK-512; keep broader import normalization as future hardening if new evidence appears. |
| F02 | High | User control | Review advertised undo/re-rate through shortcuts but did not keep a visible recovery action after rating/completion. | Users could not confidently recover from wrong spaced-repetition ratings. | Recovery existed as a hidden accelerator instead of visible control. | Addressed in TASK-508. |
| F03 | High | Visual/copy defect | Transfer summary rendered literal `{{cards}} cards / {{bytes}} bytes`. | Broken copy undermined trust in limits and import/export state. | Default copy relied on unresolved placeholders. | Addressed in TASK-506. |
| F04 | Medium | Onboarding IA | Empty `/flashcards` opened the setup utility path instead of Study/Home. | First-time users entered a dense screen before understanding the product. | Default tab optimized for implementation setup, not learning flow. | Addressed in TASK-506. |
| F05 | Medium | Creation reliability | Original browser automation could not confirm manual create submit result. | Users might be blocked at first card creation if the issue reproduced. | Missing proof around create drawer success/error behavior. | Verified and covered in TASK-477; no retained product bug. |
| F06 | Medium | IA overload | Create/import/generate/export/image occlusion lived together in one task area. | Users had to parse too many setup concepts at once. | Workflows were grouped by implementation area. | Addressed in TASK-515 with task-specific Create cards, Import file, and Export backup workspaces. |
| F07 | Medium | Review focus | Study assistant competed with recall before answer reveal. | Review loop slowed and primary recall action lost focus. | Assistant was promoted as peer workflow instead of on-demand aid. | Addressed in TASK-508. |
| F08 | Medium | Completion next step | Completion reported caught-up state without clear repeat/practice/edit next actions. | Users lacked a clear path for more practice or deck maintenance. | Completion summarized status but did not map to actions. | Addressed in TASK-508. |
| F09 | Medium | History comprehension | Recent sessions used labels like `Session #1`, `Deck 1`, or raw scope keys. | Progress/history was present but hard to trust. | Backend identifiers leaked into user-facing history. | Addressed in TASK-509. |
| F10 | Medium | Progress semantics | Due/new/current queue labels could appear contradictory. | Users could misunderstand whether cards were available to study. | Scheduler labels were technically accurate but not explained in queue context. | Addressed in TASK-508. |
| F11 | Medium | Expert workflow | Manage was card-first with no deck-first dashboard for quick study decisions. | Experienced users spent time filtering cards instead of selecting a deck action. | The model privileged card management over deck review. | Addressed in TASK-510/TASK-511. |
| F12 | Medium | Extension workflow | Extension sidepanel behaved mainly as a link-out path. | Capturing web content into cards was slower and could lose context. | Extension was treated as navigation, not capture workflow. | Addressed as a bridge/generate handoff in TASK-513; native in-extension deck picker/save remains deferred. |
| F13 | Medium | Docs mismatch | Extension and flashcards docs lagged current UI tab names and workflows. | Users could not rely on docs to understand current behavior. | Docs had not tracked UI evolution. | Addressed in TASK-513. |
| F14 | Low | Empty-state hierarchy | Manage empty state showed expert filters before any cards existed. | New users saw advanced management chrome before first action. | Empty state inherited full management layout. | Addressed in TASK-506. |
| F15 | Low | Scheduler discoverability | Scheduler was hidden until a deck existed. | New users could not learn scheduling exists until after setup. | Progressive disclosure hid a core concept too completely. | Addressed in TASK-506. |
| F16 | Low | Shortcut consistency | Shortcut hints did not always pair with visible controls. | Keyboard/power users relied on recall for recovery actions. | Accelerators were not consistently mirrored by UI. | Addressed in TASK-508. |
| F17 | Low | Global handoff | Test with Quiz was globally visible without context. | Users could leave flashcards without knowing what would be tested. | Cross-surface handoff was global rather than state-aware. | Addressed in TASK-506. |
| F18 | Low | Density and labeling | Import/Export/LLM naming implied normal import/export was an LLM feature. | Users might miss normal import/export. | Label described implementation capability rather than user task. | Addressed in TASK-506. |
| F19 | Low | Session resumption | Snapshot/continue labels did not clearly distinguish active vs completed sessions. | Users might not know whether they were resuming or viewing history. | History and active-session states were blurred in copy. | Addressed in TASK-508 and TASK-509. |
| F20 | Low | A11y and keyboard audit gap | Original pass did not complete keyboard-only create/review/recovery coverage. | Keyboard users could encounter untested traps. | Shortcut features lacked observed keyboard journey coverage. | Addressed for create/review/rate/completion in TASK-477; deeper browser a11y audits can continue separately. |

## First-Time User Assessment

The remediated first-time path is materially stronger than the audited baseline. Users now land in a learning-oriented Study entry instead of a dense transfer surface, can see scheduler and quiz affordances in context, and can reach Create & Import without decoding an LLM-oriented tab name. Manual create has explicit success/error coverage, and empty Manage no longer leads with expert filters.

Create & Import now separates setup into task-specific workspaces, so first-time users no longer have to parse import, export, generation, study-pack, and image-occlusion controls all at once.

## Power-User Assessment

The strongest power-user improvement is the deck dashboard. It gives experienced users deck-level counts and direct Review/Cram/Edit/Scheduler/Export actions, replacing a card-filter-first starting point for common study decisions. Review recovery and recent-session labels also make repeat review safer and easier to resume.

Remaining weakness: extension capture is still a bridge to full Flashcards generation, not a full in-extension save flow with deck picker and direct draft editing. That deeper workflow is intentionally deferred.

## Improvement Backlog

### Completed Quick Wins

- Fix unresolved `{{cards}}` and `{{bytes}}` placeholders.
- Default no-card users to Study instead of Import/Export.
- Add visible Undo/Re-rate after rating.
- Add completion CTAs for continued practice and card creation.
- Replace recent session internal labels with user-facing labels.
- Rename/clarify Import/Export as Create & Import.
- Clarify due/new/current study queue labels.
- Reduce Manage filters in the no-card state.

### Completed Medium Changes

- Collapse/defer Study assistant until the user asks for help.
- Add create drawer verification and failed-create recovery coverage.
- Add deck-first dashboard with direct actions.
- Make shortcut affordances map to visible controls.
- Make Quiz handoff state-aware.
- Add first-time/review/create keyboard e2e coverage.
- Add generated-card save recovery states with retry.
- Split Create & Import into task-specific Create cards, Import file, and Export backup workspaces.

### Deferred Larger Product Improvements

- Build a fully native extension capture flow with deck picker, generated drafts, edit, save, and open-in-WebUI continuation.
- Add broader import result normalization if future evidence shows unresolved partial/fatal import ambiguity outside generated-card save.
- Run a full browser accessibility audit beyond the focused keyboard e2e coverage.

## Ideal Target Workflow

### First-Time User

1. User opens `/flashcards` and lands on Study.
2. Empty state explains the value of flashcards and offers create/import/generate actions.
3. Manual create opens a focused drawer with required fields first.
4. Import/generate flows confirm how many cards will be created and where.
5. Review shows front, answer reveal, rating controls, and visible undo/re-rate.
6. Completion explains what changed and offers practice/create/scheduler next actions.
7. Reloading `/flashcards` points to the next available study action.

### Experienced Power User

1. User opens `/flashcards` and sees deck-level study status.
2. Dashboard rows expose Review, Cram, Edit, Scheduler, and Export.
3. Review supports keyboard reveal/rate/undo/edit flows with visible equivalents.
4. Completion supports repeat workflows.
5. History uses meaningful session labels instead of raw ids.
6. Extension selected-text capture opens the Create & Import generate flow with page provenance preserved.

## Open Questions And Non-Goals

- Scheduler algorithm correctness and backend spaced-repetition math were not audited beyond visible UX effects.
- Quiz surfaces were out of scope except for the direct Flashcards handoff.
- Multi-user permission, workspace sharing, and collaboration states need separate testing.
- Native extension flashcard save/edit remains a future product improvement, not a Phase 5 deliverable.

## Master Checklist

### Reliability And Verification

- [x] F01 Generated-card save recovery states implemented for success, partial failure, full failure, fatal validation, and retry. Broader import hardening remains evidence-driven future work.
- [x] F02 Visible Undo/Re-rate added after rating.
- [x] F03 Transfer summary placeholder copy fixed.
- [x] F05 Manual create drawer submit path verified and failed-create recovery covered.
- [x] F20 Keyboard-only create/review/rate/completion coverage added for the scoped journey.

### First-Time Flow

- [x] F04 Empty `/flashcards` users default to Study.
- [x] F06 Task-first Create & Import split completed for WebUI Create cards, Import file, and Export backup workflows.
- [x] F07 Study assistant collapsed/deferred.
- [x] F08 Completion CTAs added.
- [x] F10 Progress labels clarified with current study availability.
- [x] F14 Manage no-card chrome reduced.
- [x] F15 Scheduler made discoverable before deck configuration.
- [x] F18 Import/Export label clarified as Create & Import.

### Power-User Flow

- [x] F09 Recent session labels improved.
- [x] F11 Deck-first dashboard added.
- [x] F16 Shortcut hints paired with visible controls.
- [x] F19 Active/completed session copy clarified.

### Connected Workflows

- [x] F12 Extension selected-text bridge added with page provenance into GeneratePanel saves.
- [ ] F12 Native extension deck-picker/edit/save workflow remains deferred.
- [x] F13 WebUI and extension flashcards docs updated.
- [x] F17 Quiz handoff made state-aware.

### Planning Follow-Up

- [x] Convert checklist into implementation phases.
- [x] Link each phase to Backlog.md tasks.
- [x] Add product acceptance criteria for first-time setup, review completion, undo recovery, power-user deck selection, and extension capture.
- [x] Add regression tests for fixed copy, tab defaults, review undo, completion CTAs, session labels, import/generate recovery, and extension handoff.
