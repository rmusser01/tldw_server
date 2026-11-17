# A08 Notes lifecycle, search and export implementation plan

**Task:** TASK-13260.278.18.83.45; related historical repair TASK-13260.281.2.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md A-08 and release-catalog-data.mjs A-08. Execute inline with executing-plans; no new agents.
**Goal:** Validate and causally repair actual Notes workflow failures before A09.
**Architecture:** Use normal Notes UI and existing owner-scoped versioned API. Inspect actual responses and exported bytes; preserve content, metadata, draft ownership and optimistic conflict rejection. No new subsystem or dependency.
**ADR assessment:** No new decision for executing this existing contract; ADR-031 Notes capability Sync domains and ADR-059 task editing govern. Reassess any demonstrated behavior repair.
**Candidate:** frozen523ef323e9f56bc31559f6c5b8e506c27e599dcb+preserved WIP, unpublished/not latest dev. Current client build k5Td_JQ4JA4b-imz6ENU2; original SQLite-multi API25443/owner25437/18906, frontend23497/23486/18985; one owned Chrome window/CDP18987.
**Bounds:** ACK2s, navigation/save15s, background600s; one declared recovery per actual check, maximum3 attempts per issue. No unchanged replay, model/install/assets/cache/log/proof work, raw operator SQL, auth-store/cookie/header injection or unrelated signal. Existing all43 authorization includes a temporary second tab in the same owned window for the required two-client conflict; close it afterward. All other checks keep one tab. Normal role/account UI only. Fresh PG cells require exact released-owner guards and official restricted fixtures.

**Exact F-BIOLOGY:**
```text
The mitochondria is the powerhouse of the cell.

DNA stands for deoxyribonucleic acid.

Photosynthesis converts light energy into chemical energy.

The human body has 206 bones.

Water boils at 100 degrees Celsius at sea level.
```
**Controlled names:** uat-20261014-a08-biology; uat-20261014-a08-orchid; uat-20261014-a08-harbor. Tags a08-biology/a08-review. Only these new notes may be edited/deleted/restored; all foreign/existing content preserved.

## Stage1: Contract and principal preflight
**Goal:** Verify current candidate/principal/empty controlled library and public save/search/export/conflict/lifecycle contracts.
**Success Criteria:** Ordinary User7 confirmed, exact field/control selectors and actual API response identities known. Select JSON export, Markdown editor; do not invent Media lifecycle assumptions.
**Tests:** Existing Notes UI; notes endpoint/hook source and governed version fields; no note mutation until plan recorded.
**Status:** Complete
- [x] Record actual owned list response and visible empty/new-note state.
- [x] Identify title/body/tag fields, Save completion, expected-version/conflict controls, Trash/Restore and selected JSON export.

## Stage2: Real Notes lifecycle and roundtrip
**Goal:** Create three controlled notes and verify complete persisted content/metadata, edits/search/filter and export.
**Success Criteria:** Correct canonical IDs/owner/title/normalized full text/tags/version after save and reload; title/body search/tag filter exact intended membership; one actual selected JSON download parses to the same note identity/content/tags; delete/restore preserves exact record.
**Tests:** Normal UI plus actual POST/PUT/GET/search/keyword/DELETE/restore responses. No fabricated responses or direct DB mutation; arm observers before actions. Inspect full canonical bytes, not just spinners/list labels.
**Status:** Complete
- [x] Create biology with exact text/tags and two unique short notes; await each actual acknowledged save within15.
- [x] Edit title/body/tags; verify exact canonical version and normal reload; run distinct title/body queries and tag filter.
- [x] Export selected JSON once; consume actual download stream in memory and parse intended metadata/text; no proof copy/bundle.
- [x] Delete only controlled harbor note; verify active/Trash; normal Restore and full canonical equality except documented lifecycle/version timestamps.

## Stage3: Recovery, contention and applicable cells
**Goal:** Preserve drafts and reject silent overwrite across two actual editors; validate controlled input/error recovery and database/auth variants.
**Success Criteria:** Blank/over-limit rejection truthful; one exact controlled save transport failure preserves draft; legitimate retry saves intended revision; unsaved switch/reload makes explicit save/discard decision; stale second editor receives documented conflict or safely preserves its draft and resolves intentionally. SQLite/PG single/multi results explicit, no automatic adoption.
**Tests:** Public UI and wire/version observations; one temporary same-window second tab for conflict, quiesce/close finally. Any exact request interruption uses existing bounded browser transport tools and always removes interception; no provider/model or unrelated network control. Official restricted PG fixture if new cells required; 600/120/15 lifecycle guards.
**Status:** Complete
- [x] Dismiss any actual first-use UI once; test blank/over-limit inputs against current schemas.
- [x] Hold two editors at same base version, save distinct revision in first, attempt second; preserve stale draft/correct ID/owner and resolve normally.
- [x] Distinct controlled failed-save/Retry and unsaved navigation/reload checks.
- [x] Record selected remaining SQLite-single/PG-single/PG-multi cells with reviewed source/driver/host before runtime transitions; no hidden retries.

## Stage4: Repair, human UX and selected acceptance
**Goal:** Causal repairs with affected checks/scoped security/source review and one screenshot for human UX, then accurate selected-scope closure.
**Success Criteria:** Actual demonstrated defects fixed once in shared consumers; causal RED/GREEN and current affected verification; human UX acceptance separate; failed histories remain failed and common engineering/latest-dev/publication/release limits stay open.
**Tests:** Only genuinely affected configured checks; Bandit touched Python scope (TS-only N/A); reuse permitted read-only reviewer. One screenshot at /private/tmp/uat-current-ux-review.png, no duplicates.
**Status:** In Progress
- [ ] For each real failure: retain exact symptom, trace current callers, test RED, minimal fix GREEN, affected checks and review before live rebuild/retry.
- [x] Show actual Notes screenshot and obtain human usability review while independent functional variants continue.
- [x] Record selected scope/remaining risks in official task/master before starting A09; retain plan while common engineering gates remain open.

## A08 first lifecycle defect and bounded source repair
- User7 normal Notes GET returned an empty owned library. Biology create/reload: UUID 2f425399-8320-4924-879d-a27e32a3bd4d, exact227-character F-BIOLOGY, title/tag, owner7/version1 verified. Bare reload did not restore last selection by design; normal row reopen returned the exact canonical note.
- Orchid create UUID ff2457b4-dce8-47e9-942a-a041030b675b/version1 succeeded. Starting New note during its pending save refresh left the old editor writable: intended Harbor title persisted into the Orchid UUID/version2 with Orchid body. No Harbor create was acknowledged; no blind resubmit or foreign record mutation occurred.
- First bounded repair locks draft fields only during New/Save-and-new transitions. Two held-refresh regressions RED2, then GREEN. Review found owner retirement needed while a confirmation waits; second repair uses the existing captured owner/token, retires lock on authority changes, and fences held Discard/Retry callbacks. Deterministic owner regressions RED2 then GREEN; old DOM-modal observer approach STOP after3 probes, replaced with the existing Modal.confirm spy pattern after reassessment.
- Third bounded repair: reviewer found old settlePendingEdits could resume after logout and save next-session draft. Held-refresh/logout/new-draft regression causally RED1 (unexpected second POST), then GREEN13 full page cases after existing owner guards inside drain loop. Final WebUI/ownership/types/lint and source review pending; no fourth speculative source repair.
- Existing shortcut suite partial-mock setup corrected to retain actual safeStorageSerde; original assertions/functions retained, corrected6passed. Earlier reliability12passed, canonical-save/hook97passed were scoped to their then-current source; final changed-loop qualification recorded separately. TS-only production Bandit N/A. Python operator checks scoped Ruff/Bandit0 before final label/phase delta; final delta review pending.
- First A08 client build/serve prepared but unexecuted. Live wrong-record symptom remains historical/current old build; no runtime fix or A08 acceptance claimed. Biology preserved; only controlled A08 records may change. A09 not started.

- Final third-source verification: full page13 UI +13 WebUI, canonical-save/hook97, configured WebUI TypeScript0, scoped ESLint0errors108warnings; restored pre-change baseline also0errors108warnings (count comparison, not an exhaustive signature comparison). All244 check bindings restored. Independent final source/operator review CLEAR; touched3 Python operator Ruff0/Bandit0findings0errors. First A08 build running; live repair not yet claimed.

## Current original SQLite-multi results
- A08 first production build natural0/365.243s, compiler/token-sync/budget0, exact55 bindings restored; buildFawItKlHIjSLY8iQJ6AyU. Initial frontend replacement guard refused absolute-vs-relative monitor path before action; exact relative command+cwd correction reviewed. One TERM then exceeded15s and stayed a failed-stop outcome. Normal saved tab quiescence allowed natural exit0 of old23497/23486, with no further signal. Startup-only recovery retained failed history, guarded actual integer exit0/absent owners/free port/locks/API, reviewed CLEAR; frontend54353/54348 ready2.596s, monitor9889. API25443/25437 untouched. No rebuild.
- First live transition harness incorrectly expected detailGET after an existing-version PUT. It timed out, restored title only and removed all routing; no New/create replay. One corrected create recovery on new Harbor POST201/detailGET200 passed2.314s: fields locked during real held response, empty new draft after release, distinct HarborUUIDb8afc53a-c043-433b-a80f-25f92a1e094a/v1/owner7/full63chars; Orchid previousUUID unchanged canonicalv3/body57.
- Actual overlimit256title rejected422/max255 with full draft retained. Corrected Orchid title/body with PAVO-842 and a08-review tag persistedv5; first capturedv4 was intermediate title autosave, not final-body failure. Tag removalv6 and complete reload exact. Unique title-only reviewed search/body-only PAVO search each200/total1/correctID/UIcount1; tag filter same1, clear3.
- One selected Biology JSON export; automated stream0/missing path was a CDP download-location mismatch. Actual single Chrome Downloads file /Users/macbook-dev/Downloads/uat-20261014-a08-biology.json read in memory:380bytes/exactID/title/full227text/tag. No second export or proof copy.
- Two actual editors exactv6/owner7; first PUT200/v7/ALPHA, stale expected-version6 PUT409 retained fullBETA draft; explicit Use their version GET200 matched complete first canonical and renderedSaved. Temporary second tab closed; one remains.
- Harbor DELETE204 removed active row/list2; actual Trash row exactID/title/full63body; RestorePOST200 expected_version2 and detailGET200 full original fields preserved, active/deletedfalse/v3/count3. TrashAPI first observer watched wrong path; recovery hit ambiguous Notes selector before action; no third read replay. Full Trash state used actual row plus versioned Restore, not a captured TrashAPI body.
- One exact Harbor PUT transport abort retained draft/visible kept-on-device notice; explicit Retry200/v4 intended full text. Unsaved navigation flush200/v5 and full roundtrip. Return shortcut selector corrected after ambiguity without replaying save. Unsaved reload actual beforeunload dismissed retained draft; normal Save+confirmed reload canonical exactv7 (intermediate autosaves counted honestly).
- First screenshot caught transient loading banner; passive loading-hidden wait then overwrote same screenshot with settled UI. Human A08 UX question pending; remaining new-mode cells not run. All broader engineering/latest-dev/native/release gates remain open. No A09.

## First fresh A08 mode and filter diagnosis
- Reviewed one-use host/transition/restoration operators, shared mutex/owner admission and 600/120/15 limits; scoped13 Python and3 Node operator checks passed. Original API25443/25437 stop consumed/exited. First SQLite-single API65311/owner65234 canonical ready20.464s; real empty library, three canonical distinct notes/full text/tag/owner, edit/reload and title/body searches passed. Tag-filter stage TimeoutError; exact suboperation was not captured. No app defect or mode acceptance is established.
- One existing tab quiesced about:blank; failed finish retained. API -15, actual ASGI shutdown complete0.706s/overCapfalse; host exited/returned, pytest1failed15deselected67.25s/70.002outer. Other2 fresh PG hosts unstarted. Original profile/data/keys and failed SQLite profile/results preserved.
- Source shows tag suggestion search has300ms debounce; driver immediately filled/Enter/Escape. Changed only operator to wait for the actual visible exact tag suggestion and click it, and split selection/response/rows/clear stages for diagnosis. Deferred-suggestion operator check1passed; prior3 Node controls excluded from unchanged replay. Timing is a hypothesis, not a confirmed application cause; no app source edit or rebuild for this timeout. One distinct SQLite recovery remains to be declared after reviewing the changed driver. Human A08 UX answer pending; A09 not started.

## PostgreSQL results and remaining-only recoveries
- PGsingle API68002/owner67972 ready44.865s; three full canonical creates/owner/tag, edit/reload, title/body searches, exact real tag membership+clear, canonical Trash/Delete/Restore and exact Biology passed. API -15/ASGIshutdown0.645s;1passed15deselected83.85pytest87.187outer/natural0. Changed visible-suggestion selector independently reviewed CLEAR. This success does not prove the cause of the earlier SQLite timeout.
- PGmulti API68523/owner68375 ready52.092s, ordinaryAlice11; creates/edit/reload/search/tag checks passed. Two-editor conflict block failed AssertionError; exact assertion was omitted by the operator. No application cause/fix claim. Failed finish/one tababout:blank, API -15/ASGIshutdown3.195s;1failed15deselected131.37pytest134.698outer.
- Declare exactly one distinct remaining-only fresh recovery for each failed cell: sqlite-single-recovery runs necessary3-note fixture setup, failed tag filter and remaining lifecycle; pg-multi-recovery runs necessary3-note setup, conflict/lifecycle only. Already-passed edit/reload/search and PGtag checks excluded. Same official host/600-120-15 limits/failed-first guards/newprofile/one-use receipts/shared mutex; never reset originals. Recovery initialization/start still unexecuted.
- Narrow operator review caught missing optional-recovery receipt with existing profile could be skipped; both host admission and original restoration now reject orphan recovery profiles. New negative guard and two source-bound checks3passed. Four earlier recovery checks passed before exact2-cell generalization; affected3one-use/released-first checks rerun afterward. Ruff0/Nodeparse0/currentstaticBandit0findings0errors. Existing warning noise retained; /dev config-path unit invocation attempted denied pytestcache writes, then later changed checks used existing project configuration; no cache cleanup/workaround or acceptance claim.
- Conflict driver only adds exact substage, own literal AssertionError message and response status/version metadata; no application patch, fabricated response or transport hold. Two stage-label lines initially landed after module end; self-read caught and corrected before import/runtime. Final independent recovery review pending. A08 UX pending; A09 not started.

## Selected A08 functional completion
- Final independent recovery operator review CLEAR, including both orphan-profile guards, exact two recovery selectors/failed-first release gates, remaining-only branches and distinct ordinaryAlice12. Both receipts initialized once; neither original failed receipt/profile reset.
- SQLite-single recovery API72755/owner72716 ready36.877s, actual exact tag membership/clear3, canonical Trash/Delete/Restore sameID/full63text and preserved227 Biology passed. Necessary three-note fixture setup only; already-passed edit/reload/search excluded. API -15/ASGI1.000s;1passed23deselected70.28pytest72.904outer/natural0.
- PGmulti recovery API73985/owner73751 ready50.288s, actual ordinaryAlice12/fresh empty library. Basev1 firstwriterPUT200/v2/ALPHA; staleexpected-version1 PUT409 retained fullBETA draft. Explicit UseTheirVersion GETexactfirstcanonical and Saved; temporarysecondtabclosed. Canonical HarborDelete/Trash/Restore sameID/full63text/v3 and Biology227unchanged. API -15/ASGI2.709s/overCapfalse;1passed23deselected143.88pytest147.212outer/natural0.
- Selected functional matrix is complete by explicit composition: originalSQLite-multi complete lifecycle/export/recovery/2editor, PGsingle complete selected mode, firstSQLite/PGmulti completed subchecks plus distinct remaining-only recovery results. This is not one complete all-mode rerun. First SQLite tag Timeout and PG conflict AssertionError remain failed with causes not conclusively established; changed-driver success is not an application root-cause/fix claim. The actual wrong-record draft-transition defect retains its causal source/live repair qualification above.
- OriginalSQLite-multi restoration initiated once as apiA08CellsRestore after all five authoritative hosts released/absent, ASGIshutdown/noovercap and orphan/owner/port/source/profile checks. API75730/owner75726 starting; readiness pending at this entry. Existing build/frontend54353/54348/oneChrome retained. HumanA08UX pending; selected A08 acceptance not yet granted, A09 not started,7accepted36open. No rebase/publication/currentdev/native/release acceptance.

- Restoration outcome: original API75730/owner75726 canonical ready53.233s within120, marker consumed/monitor56487 running. Fresh normal UI restored ordinaryUser7; actual NotesGET200 total3/BiologydetailGET200 exact227characters/owner7; one existing Chrome tab at18985/notes, Saved and loadinghidden. Sole screenshot overwritten/displayed for pending humanA08UX. No extra frontend build/start/window, app patch or model action. Original source/profile/data/keys/OCR/provider preserved; all five fixture hosts released.

## Human selected-scope acceptance
Human explicitly accepted A08 Notes UX on2026-10-14. Selected functional scope plus UX ACCEPTED by the recorded four-mode composition. Counts8accepted35open; A09next. Failed first runs remain failed/causesundetermined; frozen523+WIP/current-dev/rebase/publication/common engineering/native/release limits retain. Stage4 remains InProgress for those common final gates; retain this owned plan.
