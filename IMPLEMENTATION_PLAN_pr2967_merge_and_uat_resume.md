# PR2967 integration and UAT continuation

Tracking: TASK13260.219. User requested merge of the accumulated repair PR into `dev`, then UAT resumption starting with TestBot261. This changes the execution order, not the exact-output criterion or the full SQLite/PostgreSQL single-user/multi-user workflow scope.261 remains open.

## Stage 1: establish the merge candidate and failing checks
**Goal:** Include the reviewed local repairs and identify actual CI/review blockers.
**Success Criteria:** Current origin/dev and PR revisions recorded; failed job logs retained; independent integration review started; findings separated by cause.
**Tests:** Inspect current GitHub Actions job conclusions/logs, source diff and retained regression evidence. No skipped job is a test pass.
**Status:** Complete.

Initial base59049e094e0845a4611ea725ae19b7c1754ea709 is an ancestor of local91be80160ec9a8962daba5a4b5fcbd0688dfcd03. DraftPR2967 currently points to3e57d8887ed08e911808d56648731cec71953e27. Eighteen failed job conclusions include aggregate jobs; determine distinct causes before assigning repairs. Current local branch includes native276277 acceptance.

User requested removing excessive generated Playwright evidence from the PR. The 11,712 branch-added files under `output/playwright/` account for 3,254,093 added lines. Preserve every local file, remove only Git index entries, and use the existing `/output` ignore rule. Keep the tracker, matrix and concise verification results; historical evidence references become local archive references. No branch history rewrite.

## Stage 2: repair and verify merge blockers
**Goal:** Fix confirmed failures with causal evidence and minimal changes.
**Success Criteria:** Required CI passes on the reviewed candidate; substantive review findings resolved; associated tasks and running tracker current; no bypass of hooks, tests or branch protection.
**Tests:** Reproduce failing tests, retain baseline comparisons where needed, run relevant package/backend suites and actual official-fixture PostgreSQL controls for database changes. Run scoped lint/security checks and final candidate CI.
**Status:** Complete.

Generated evidence removal is committed and pushed at `6bc0b1c13368d241461c35257a2a322b67f89b12`. Confirmed integration defects are World Book PostgreSQL ownership (278), rejected Chat Retry connection cleanup (279), and World Book secondary operations/LIKE placeholder conversion (280). Retry has independent review and 110 passing focused tests; World Book verification is expanding to RAG and conversation consumers found during review. CI repairs include realistic frontend fixtures, stable OpenAPI dependency/schema generation, ingestion opt-out, backend shard assignment, onboarding and extension persistence fixtures. Final candidate CI and the human-authored Change summary remain required.

World Book ownership/consumer/transaction repairs now have independent review and official PostgreSQL/SQLite verification. Notes remediation passes32 tests, published docs pass52 tests plus three macOS controls, and the final World Book attachment fixture passes10 tests. Additional frontend readiness changes preserve all original assertions; exact CI Node20.20.2 local verification passes46 cases and times out in four under unrelated worktree CPU saturation. At pushed1c4ff92226, hosted CI confirms all repaired areas except two Settings assertions and the original extension fixture. The user's approved single trace-enabled extension replay passes with zero skips/retries; source review then moves temporary-chat setup before the first history seed. The final refinement requires hosted CI. Its Axe report also reveals a named generic content container, tracked as UAT281/TASK13260.219.9 for a minimal semantic repair and regression. These are merge gates, not completed UAT acceptance.

At fa50b25ca4, final hosted extension acceptance passes36 scenarios with zero skips/unexpected/flaky outcomes, and Axe has zero violations/incomplete observations. Onboarding also passes; TASK13260.219.6/.9 are Done and281 verified. The Settings radio repair passes hosted CI, but the first ordinary load/save assertion still sees an empty field. Independent source review finds no missed Form registration/reset path. The reviewed helper candidate separates real textbox arrival from its unchanged exact-value readiness;10 focused tests pass on CI Node20.20.2. That final candidate still needs hosted confirmation; the earlier timing cause is not claimed as established. The human-authored Change summary has been requested and remains pending.

## Stage 3: complete PR requirements and merge
**Goal:** Integrate the verified repair branch into dev.
**Success Criteria:** PR description represents final scope and261 limitation; human-requester Change summary required by repository policy is present; PR ready and required checks/review clear; normal merge succeeds and remote merge commit verified.
**Tests:** Fresh PR status/head/base/check review immediately before merge; inspect remote merged state afterward.
**Status:** Complete.

Requester checkpoint2026-09-18: the user supplied their own Change summary and explicitly requested latest-dev rebase, remediation of all Qodo findings/comments, then merge. The exact summary is published in PR2967. Fresh origin/dev remains59049e094e0845a4611ea725ae19b7c1754ea709. Rebase replayed327 non-merge commits, preserving the previous8eca6 head in codex/pr2967-before-rebase-20260918. Historical tracker/task conflicts retain their accepted records; the original cycle3 plan detail is restored. Application/test files match the pre-rebase candidate. Generated output/playwright files remain excluded and their local archive is preserved. Before pushing, integrate and verify the final CI fixture correction below; then mark ready for Qodo review and address every finding before normal merge.

At8eca6, all eight frontend unit shards pass; shard2 has31 frontend and563 UI tests, including all10 Settings timeout/form cases. Final frontend-required then fails the immediate session-credential assertion in manual-api-key-persistence.spec.ts before reload, on all three hosted attempts. Independent review confirms that the helper checks any local configuration even though session saving writes local metadata before the separate session credential. Wait for the exact credential record in its selected storage area; retain all reload/reopen/auth and local-secret-absence assertions. Production credential-loss behavior is not established by this failure. Relevant local validation and final hosted CI remain required.

## Stage 4: resume UAT from261, then the full workflow scope
**Goal:** Continue the original UAT → review → fix loop on the integrated code.
**Success Criteria:** First resumed scenario is original TestBot character flow, all outcomes retained; subsequent fresh four-configuration workflow matrix proceeds with261 still open until genuinely resolved. No green-only retries or acceptance weakening.
**Tests:** Characters → TestBot → Chat, exact public question and instruction, real completion/canonical reload; then the retained twelve-row protocol across SQLite/PostgreSQL and single/multi-user setups, including actual image attachment.
**Status:** In Progress.

The first corrected local browser run passes device and legacy persistence but fails session readiness after the unchanged15-second wait, with retries0. Thus the original weak readiness check does not explain the full failure. Retain that failed attempt and trace actual save/storage behavior before another candidate. Local evidence: .tmp/pr2967-merge-20260918/manual-api-key-e2e/. The rebased branch may be reviewed by Qodo while this known gate is diagnosed; it is not merge-ready.

## Qodo remediation checkpoint, 2026-09-18

Rebase and force-with-lease push are complete at19919ea; the PR is ready for review. The human summary is published verbatim. Qodo posted14 findings; all are tracked in [the review ledger](Docs/Reviews/PR2967_QODO_REVIEW_2026_09_18.md). Qodo accepted the actual TypeScript parser evidence and dismissed its Flashcards syntax finding. TASK13260.219.11 owns Notes durable offline saving and title authority recovery; .12 owns the PostgreSQL upgrade, compatible grounded legacy Chatbook saves, DB probe and core claim construction; .13 owns the remaining frontend transport, translations, timer and page-boundary findings. Existing and new focused regressions remain intact, including failed intermediate runs.

UAT282 is now reproduced as a real stale initializer deleting a concurrently written session credential. TASK13260.219.10 owns the mutation serialization and causal race tests. An initial isolated browser candidate passes3/3, but independent review finds the full clear/write sequence still needs serialization and broader existing cleanup controls. That candidate is not considered final. Final reviewed source must pass the required hosted checks after publication.

Fresh dev rulesets require strict CI and permit merge commits only. Once final review and checks pass, use a normal merge with an exact head guard; do not squash or bypass branch protection. Then resume261 before the full fresh UAT matrix.


## Final local review acceptance, 2026-09-18

All 14 Qodo findings now have a verified fix or accepted disposition. The reviewed source is committed in b852f644a0 (complete credential mutation serialization), 3420b66228 (PostgreSQL upgrade, legacy Chatbook compatibility and DB/core boundaries), 9344f3ae95 (frontend transport, localization, timers and web route boundaries), and a55e86167f (durable offline Notes saves and owner-safe title recovery).

The final credential repair passes 113 focused and adjacent tests and all three real browser lifecycle cases with zero skips or retries. Backend verification passes 70 combined controls, then 35 focused controls after the final refinements, including four actual PostgreSQL cases with no skips. Frontend review verification passes 46 focused UI, 12 route-title and seven Login/navigation tests. Notes/title verification passes 18 focused UI and 18 existing web title tests; the final test-only lint correction also passes all three provenance cases. Independent reviews accept the resulting changes. Failed intermediate attempts remain retained.

The canonical OpenAPI fingerprint is 6dcb5357a7a6d13b636b6d1cae6a7275a432796b6a2105745fcde7e7dc89bc06. Only the two legacy flashcard field descriptions change; path and schema counts remain 2097 and 3207. Nine touched backend production files have zero Bandit findings.

Source repair is complete locally. Publication, individual Qodo replies and final required hosted CI remain merge gates. These results do not close UAT261 or certify the pending full fresh-install matrix. The requester-provided Change summary is already published, and the latest fetch still identifies dev as 59049e094e0845a4611ea725ae19b7c1754ea709.


Publication checkpoint: all reviewed source fixes are pushed at 38b686f0b2. All 14 Qodo threads are resolved, and the updated review reports zero open bugs, rule violations or cross-repo conflicts. Individual replies include fix commits and verification evidence. Final required hosted CI remains pending.


## Verified merge and UAT resumption

PR2967 merged normally at 2026-09-18T21:06:54Z as `3cff7962721a60b768464221c1f7fe2a8b25e4d5`, with exact tree parity to tested head 35e46b7a14. All required checks and eight frontend shards pass; actual WebUI, extension and cookie lifecycle results are 3, 3 and 1 passes respectively. All 14 Qodo threads are resolved. The new `codex/postmerge-uat-20260918` branch starts at this merged dev revision. Targeted preparation preserves the original PostgreSQL single-user TestBot profile and controls. One exact original question followed by canonical reload is first; the full fresh matrix remains pending.


Post-merge resumption checkpoint: one original PG single-user TestBot submission completed, persisted200 and reloaded as exactly two canonical rows. Its final `BEEP BOOP` lacks the required period, so261 remains open. Independent review confirms the request count, unchanged card/model/profile and visible/canonical agreement; it makes no inference about uncaptured internal provider inputs. The requester explicitly authorized continuing the full matrix with261 open. TASK13260.219 merge/resumption acceptance is complete; parent TASK13260 continues Stage4 through the [fresh post-merge matrix](Docs/Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_18.md).

Stage4 checkpoint2026-09-19 02:12UTC: PostgreSQL single-user, PostgreSQL multi-user, and SQLite single-user have completed frozen execution with failures and limitations retained in the matrix. SQLite multi-user remains pending. SQLite single adds UAT299 (TASK13260.236), with283/286/288/290 recurrences. Source manifest24899 entries matches3cff exactly; original provider configuration restored, browser closed, and both owned app PIDs exited. Docs/task-only checkpoint: whitespace and credential scan pass; Bandit and production test execution are not applicable to these tracking edits. Stage4 remains In Progress; no full acceptance or new PR.

Stage4 checkpoint2026-09-19 03:25UTC: all four frozen configurations now have outcomes for the12 journeys. SQLite multi completes natural expiry, reciprocal native/API ownership controls, actual image attachment, source/QA/Study, and real provider/API outages. Acceptance remains failed:261/286/288/290/293/294/299 recur, exact Wikipedia is externally blocked, hidden-tab completion is tool-blocked. Character retry preserves one user and truthful interruption state but returns no final answer; ordinary retry identity/canonical count passes, cross-tab reload fails290. All24899 source entries still match3cff; no production source changes. Original config restored byte-for-byte, independent contexts/main browser closed; owned API/frontend teardown requested. Stage4 and TASK13260 continue into repairs; full matrix results are not an all-pass claim.

Post-matrix repair checkpoint2026-09-19: UAT294 is verified in e840341be1 after124 focused tests, independent review, and reciprocal native SQLite/PostgreSQL account-switch, legacy-link, Back and reload acceptance. Official PostgreSQL fixtures used a restricted runtime role;24,919 repair archive entries match the commit. The completed UAT294-specific plan was removed under repository plan cleanup guidance; TASK13260.231 and the running tracker retain the final evidence and limits. Seventeen findings remain open. UAT293 diagnosis identifies unscoped composer drafts and retained live Chat state; repair work continues before another full matrix.

Post-matrix repair checkpoint2026-09-19 05:48UTC: UAT293 is verified in30d15d951d after231 focused tests, the additional organization control, and reciprocal native SQLite/fresh PostgreSQL Chat account isolation and owned recovery. PostgreSQL canonical foreign reads remain403;24,924 archived source entries match. All test app processes and the official fixture holder exited normally. The completed UAT293-specific plan is removed; TASK13260.230 and the running tracker retain evidence. New UAT300 separately records early navigation before login workspace assignment. Totals300 findings/283 verified/17 open; Stage4 repair work continues before another full matrix.

Post-matrix repair checkpoint2026-09-19 06:08UTC: UAT291 scoped PostgreSQL bindings and truthful retrieval failures are repaired alongside newly identified UAT301 title sorting on both engines.220 focused tests pass with real PostgreSQL and no skips; Ruff/Bandit and independent follow-up review are clear. Targeted native acceptance remains pending, so both findings stay open. Current totals301 findings/283 verified/18 open; Stage4 remains in progress.
