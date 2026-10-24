# UAT fixes on latest dev — 2026-10-23

Task: TASK-13260.278.18.83.45. Human explicitly authorizes rebase, necessary repairs/setup/retries and sequential completion of all43 workflows. X01 and X02 selected acceptances are complete by recorded composition;35accepted,8open. X03 is next, after actual incoming dev review/integration and affected qualification. Prior frozen qualification alone does not qualify this new candidate.

## Constraints and rulings

Preserve current runtime/source/profile/data/keys/build/Chrome and complete unpublished fixes. Work in the separately managed uat-latest-dev-20261023 checkout. Original523 plus765tracked/345new source-doc-test files; exclude every .next build, model/data/cache/runtime lock/secret. X02 is accepted at frozen dev17141/e388823/frontend9U. The fresh closeout handoff read is devc90e57e3824a77102939e5b4017c5c0973a00c64, not yet integrated; the earlier final X01 readback at22c9 remains historical. Refresh before any later publication. PR3084 is merged/immutable. No GitHub queue/push/merge action in this integration unit. Existing startup120s/generation180s/workflow600s/case300s/operation and gracefulcleanup15s remain. No force or weakened acceptance. Heartbeat stays paused.

ADR required:no for integration; preserve upstream decisions and their actual statuses. Proposed Media identity069 collided with existing upstream069 frontend API contract; it is now Proposed082. Exact duplicate regeneration/recovery records map to upstream077/079; their Proposed statuses remain unchanged.

Ruling: use a separate checkout despite the general reuse preference because the protected running API imports the old checkout; rebasing there would change its live modules. Preserve that source until a separately qualified transition.
Ruling: normal Git stash in the new checkout is functional rebase state, not a proof bundle; apply only its exact captured SHA and retain it until restored changes are verified. Never stash/clean/reset the original dirty checkout.

## Stage1: Preserve and rebase
**Goal**: Exact finite source transfer, committed-fix rebase onto verifieddev, unpublished changes restored.
**Success Criteria**: Original bytes/index/owners unchanged; only admitted source copied/stashed; rebased ancestry contains exactdev; all conflicts inventoried, no blanket ours/theirs.
**Tests**: File/mode/owner/path checks, exact Git state and patch-identity checks, post-transfer source comparison.
**Status**: Complete

## Stage2: Reconcile and qualify combined source
**Goal**: Preserve upstream behavior and every still-needed UAT repair.
**Success Criteria**: Keep current NotesSQLite84/PostgreSQL89 migration chain, active-borrower pool ownership, native-chat quota schema102, Notes semantic schema103 and admin indexes101 and strict stored-settings errors; combine X01 authority/progress/read-only admission. Resolve ADR collision. Satisfy exact current dependencies including pytest<9.1; affected causal/connected tests, compile/lint/scoped baseline-current Bandit and independent review pass. Normal hooks, no bypass.
**Tests**: Genuinely changed and conflict-affected suites in both configured clients, official isolated PostgreSQL/restricted roles and relevant SQLite controls. Canonical API generation only on qualified combined source where affected.
**Status**: Complete

## Stage3: Finish X01 and continue sequential UAT
**Goal**: Preserve recorded X01/X02 acceptance, integrate and qualify actual incoming dev changes, then continue X03 and the remaining workflows sequentially.
**Success Criteria**: Version-source checkpoints oldest justified plusv0.1.43/v0.1.46; all required SQLite/PostgreSQL single/multi upgrade, backup/separate restore, preserved old data/accounts/Jobs, new writes, real provider, UI/reload, malformed/recovery and complete lifecycle/handback checks. Prior accepted UX remains; screenshot for changed UX. Dispose completed functional bulk/operators promptly. Update officialtask/X01/masterindex accurately, then move one workflow at a time.
**Tests**: Reviewed actual drivers with current source/package/build binding, no model replay merely for operator diagnosis. Explicit limitations retained; no latestdev readiness or completion claim from old artifacts.
**Status**: In Progress

## 2026-10-23 integration checkpoint

Committed history rebase complete at exactbd9b4e0ac3; all34old committed intents already upstream by equivalent/newer implementations, no duplicate local commits. Exact functional stash b47f59b22e4ee68042fd1be42cc4971e8a190d06 retained and restored. Initial180unmerged paths; reviewed24Admin/API/type paths selected upstream, one cold-start test selected unpublished source to retain two new cases. Three backend conflicts syntactically combined; AuthNZ MCP discovery ID101 collides with upstream101–103 and deliberately awaits a real regression/renumbering to104. No latest-dev qualification. All auto-merges still require review, especially active-borrower cleanup, strict stored settings and source/lease authority. Canonical task carries both histories without changing criteria/DoD/status/summary.

## Combined-source checkpoint

All180initial conflicts resolved. Qualified separate Python3.12.13/pytest9.0.3/exact editableNEW and133applicable requirements, plus frozen Bun3576packages. Migration collision3red ->3green(100/103/unique ordered registry) after only own104renumber. Display8red/10pass ->28pass in each client, preserving provider history and native metadata isolation. Existing pool/legacy/fallback controls40pass1optional modern-poolskip; new private-pool lifetime and native-quota readiness cases cover review-found integration gaps. Full source/affected checks/security/final review/runtime qualification are not complete. Original checkout/runtime/session/data/keys/build remain protected; X01 soleactive.

## Review repairs and current validation

Real official PostgreSQL quota/readiness and private/shared/legacy pool checks59passed37warnings148.69pytest154.487outer (modern pool3.3.3). Workflows deferrable-arbiter causal2failed/positive1passed; require immediate unique arbiters, then37connected runtime/store/index cases natural0/7.928outer. Admin merge duplication removed only after AST equality of both copies to upstream; file now byte-identicalHEAD. MCP overlapping services reproduced stale outcome publication in both directions2failed; shared worker-local weak-value async lock now covers snapshot through executor/registry and persistence, with cancellation-release test. Independent corrective source review CLEAR, no cross-worker catalog consistency claim. MCP/runtime/upgrade/AuthNZ validation ongoing; no installed/new-build/live qualification or new UAT closure. Original runtime/session protected.

## Latest-dev scoped qualification and packaging

Remaining AuthNZ66passed, Media/runtime/RLS92passed, Collections/WorldBook/UserDB/MCPfilters/Notes141passed, AuthNZ/Workflows/revocation/PGdiscovery212passed, ACP/runtime/startup/upgrade/package237passed; MCP179passed. Existing focused migration3/quota-pool59/Workflows37 results retain scope. All natural0 within unchanged caps. Final remaining frontend2modules31passed perclient; earlier display28perclient, catalog/focus/credential287perclient, timeout96UI/146WebUI retained. No full-suite/live UAT acceptance.

113Python compile; Ruff71baseline66current/no new, ESLint17files320baseline318current/no new/no ignored. Bandit1169baseline1163current before one narrowly explained changed-assert annotation; AST unchanged, target final42inherited/0new/0errors, combined1162current/0new/0errors. Operator filename/path-scope errors corrected without treating initial incomplete outputs as qualification. Canonical normal export/types22.730s/2140paths3316schemas; committed fingerprint/pathunion byte-identicalHEAD. Fresh remote dev remains exactbd9b4e0ac3.

Existing packaging operator now uses current declared3816package/data inputs rather than old3775inventory; previous site remains byte-protected.9operator guards/staticsecurity0/0/Ruffclear and independent preparation reviewCLEAR. Actual fresh private wheel/install in progress; no new API/frontend/browser startup yet. X01 soleactive; Stage2 still needs final staged review/normalhooks/commit, Stage3 actual installed/runtime qualification remains open.

## Source freeze and isolated artifacts

Normal local04f73263923e2f3d371ffbe9d5a97f73a2ab31e0 contains139reviewedpaths10431insertions497deletions onbd9. Committed patch equals staged SHA25628202e2eb30bacc18451f7ef93fde3b30e523ddd2d543213de0b4142062d831a exactly; tracked clean at postcommit, unpublished. Normal139filehooks passed3.397s; wizard-only formatter hooks correctly skipped unrelated scope.

Private installed current3816file artifact qualified9.150s/3roots/337dependencyNodes722edges/cleanup0. Normal isolated NEW frontend compile qualified135.275s/7731sourcebytes unchanged/compiled77s/token-sync and593.4KB budgetpass; buildnaXUk1FRiQ0MLFiz-np_d, all5rewrites18908. Inherited TypeScript build skip disclosed, no fullTS claim. Originalowners/source/env/session unchanged. No private API/frontend/browser startup or liveUAT acceptance yet. Stage2Complete, Stage3InProgress. Futurecandidate drivers need new site/env/build bindings and prospective strict retirement metadata; noSend diagnosis before mode reruns. Dispose completed dormant OLD build roots only after full no-active-binding/source/ownership/link guards.

## Latest-dev login repair — component checks

Actual browser diagnostic caught ReferenceError:isAbsolute is not defined before login dispatch. The shared request executor contained an obsolete duplicate CSRF block referring to variables now owned by resolveTldwRequestPlan. Existing absolute same-page regression reproduced1failed/12excluded/1.167s. Delete only duplicate14lines; preserve samePageOriginRequest CSRF/auth/origin/recipe protections. Connected6modules86passed in each configured client (12.826s UI/12.380s WebUI). Changed module has0diagnostics under both actual configured TypeScript programs; no fullprojectTS claim. Configured lint0errors/6identical inheritedany warnings/no ignored. TS-only Bandit N/A; functional Python operators0findings0errors. Final review/normalcommit/build/live validation pending. Python installed bytes remain unchanged, so no redundant wheel/install. Separate private frontend shutdown27.786s exceeded15s and remains unqualified. Original session/runtime/source/archive/config preserved.

Latest live dev advanced during validation to4e7be41029b18881e55aa4260dfcb83baf92c45c. Current bd9-based results retain exact scope. Preserve reviewed login repair locally, review incoming delta, rebase to4e and qualify genuinely affected source before the next build. Stage1/2 will be reopened for this new delta; Stage3/X01 remains active. Original runtime remains preserved.

Exact4e integration complete: normal rebase of both local commits atb6e13804f25c8ca7bd59c2c3c67d6fb69f39c75e, no conflicts; originaltrackedsource/owners preserved. Incoming13path production/tests reviewed; one incoming test constructor-cleanup correction now pending qualification. Old bd9 installedsite/build/results retain exact scope; newPython package and affected checks required. Frontend/login source unchanged by newincoming, so no repeated86case controls. X01 remains active.

Incoming4e qualification completed:35sequence/constructor controls passed47.003s outer;82MediaFiles/restrictedRAG passed322warnings208.32s pytest;18admission/ABA passed6warnings24.43s pytest26.906s outer. All selected cases passed, normal official PostgreSQL isolation required.7Python modules compile; Ruff0baseline/0current; Bandit474identical/0errors/0new. Protected owners unchanged; oldpytest-temp warnings untouched. New test constructor cleanup reviewCLEAR, behavior/assertions unchanged. Source qualification is separate from pending current4e package/frontend build and actual X01 live/lifecycle acceptance.

## Current4e artifacts and live check

Normal local ec3b5ddb8e810f080e224f5bfc6b002ae9700b75 contains the reviewed incoming MediaFiles fixture cleanup after all affected35+82+18checks, static/security and normal hooks passed; exact committed patch SHA256bf94739161ad185befce183da921ad759d8a652c033de22c4b4346dfc65b007a. New private Python3816file package qualified8.210s/3roots/337dependencyNodes722edges/cleanup0. New direct frontend build THttd0p00MlSSxPE1Yyre qualified84.444s on ec3,7731inputs unchanged, exact5APIrewrites18908, token-sync and593.3KB budgetpass. Inherited build TypeScript skip disclosed. Build operator preserves primary failures, revokes late/interrupted publication and explicitly guards the canonical tracked extension assets symlink; focused controls and independent review CLEAR. Initial symlink preflight stopped before compilation and was corrected without weakening other guards. Original3owners unchanged. Saved-only0132 login/Chat/reload/Disconnect/handback check is running; no outcome or X01 closure inferred.


## Current22c dev integration

Latest dev refreshed to22c9eac5cb9eb59cea3b9aea0448ee4da0965ebc (PR3306):10commits/33paths,13production Python+catalog,7test paths,ordinary docs/foreign tracking; no dependency/frontend/OpenAPI-shape change. Independent incoming source review complete with foreign narrative coverage explicitly partial; historical verification/prices not adopted. Only owned overlap llm_provider_overrides; normal six-commit rebase conflict-free, all six range-diff identical, newHEAD8dfd1e2b87d0d4ba849b542bd4c60b6d35a1fa6b; protected original tracked source/owners unchanged. All12 upgrade matrix cells now pass on4e-based artifacts (one explicitly hybrid .32PGsingle). Latest .43full1passed1warning46.27s/47.048s outer,API0.229/frontend0.005; .46full1passed1warning44.48s/45.293s,API0.241/frontend0.005,zero-work/wholeoriginalpreservation/cleanup0. New22c qualification pending, not accepted from old artifact. One actionable incoming compatibility regression: removed saved claude-haiku-3.5/claude-sonnet-3.7 spellings map to invalid generic3-5/3-7 rather than new dated catalog IDs. Add real normal-module causal tests, two explicit shared aliases, affected tests/static/security/review; preserve strict final-dispatch require_model and normal budgets. No new ADR: compatibility mapping within existing provider-ID boundary.


## Current22c source qualification

Latest22c combined qualification complete: four normal-module legacy alias/adapter cases4failed28deselected3warnings6.85s pytest12.040s outer before the fix, then4passed6.47s/11.063s. Two explicit shared3.x aliases retain price, strict selection, inference, allowlists, native payload and actual adapter IDs; generic mappings and final-dispatch require_model unchanged. Complete seven incoming plus15connected modules744passed1skipped8907warnings299.65s pytest306.632s outer, normal600s/300s guards; sole inherited TestClient streaming skip remains disclosed, no actual-provider price verification.20Pythoncompile; Bandit22c baseline836/current835/0new/0errors (one narrow changed-pytest annotation), Ruff24identical inherited/0new. Independent actual-source review CLEAR, six rebased intents unchanged. All8runtime pins/fresh packager target latest-22c9/site,3817inputs admitted, unchanged TH frontend. Four released superseded packages removed16751entries311915747bytes after process/env/open-handle/owner/no-model guards; qualified profile-reads candidate and originalsession preserved. Fresh22c packaging and selected current persisted-config/old-route recovery remain; no repeat whole-mode/model controls justified.


## X01 current recovery qualification

X01 final current-package recovery PASS: one case/one warning43.43s pytest44.222s outer, current310642/dev22c9 and3817-file installed package/unchangedTH frontend. Genuine v0.1.43-source PostgreSQL single-user profile: old seed -> whole backup -> separate restore -> owner migrations -> restricted serving; actual stored retired URL becomes canonical after reload with Multi User/manual authority and no cookie config unchanged; normal single-user key Save, profile200/retainedJob, old two-message Chat, legacy /chat/settings -> /settings/chat200 and visible Chat behavior, return/reload retained IDs/content all pass. Strict Disconnect/receipts/zero tasks, graceful API0.174s/frontend0.006s/EOF/group release and full original canonical/five messages/draft/URL/owners/source/config/archive preservation pass; cleanupErrors0. No generation or original runtime cycle. Fresh remote dev remains22c9 exactly. Independent requirement review supports selected X01 closure by explicit twelve-cell4e + affected22c qualification/new package/unchanged frontend/current recovery composition. Prior failures remain failed, no auth-expiry guidance claim. Final documentation reconciliation and guarded disposal precede closure; X02 not started.


## X01 closeout

X01 SELECTED FUNCTIONAL + HUMAN UX ACCEPTED by reviewed twelve-cell4e/current22c composition and nine catalog dispositions in Docs/Plans/IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#x01-supported-upgrade-selected-closure. Fresh dev22c9 unchanged; current310642 package/recovery qualified. Original session preserved. Completed disposal removed five historical/superseded roots plus28 retired operator/check files,181807entries5387693120bytes, after process/environment/open-handle/UID/inode/no-follow guards. Initial model-suffix guard refused before mutation; only registered dependency assets inside the two private historical venvs were admitted by their RECORD metadata; unrelated models/shared BASE/.venv untouched. Current22c site/build/env and original restoration dependency chain remain. Both completed owned X01 plans removed; their earlier task references are historical and point to the retained master closure. ADR082 related-plan pointer updated only, Proposed status/rationale unchanged. Selected tally34/43accepted9open; X02next, notyetexecuted. Overall task/AC/DoD remain open; no publication or release-wide acceptance.


## 2026-10-24 current Chat redesign integration

User explicitly identified the complete redesign PRs3309/3310/3311. Fetched dev17141 includes3309 calm composer and3313 message views;3310/3311 remain open future references. All18 existing UAT commits rebased conflict-free from22c9; every range-diff entry identical. Notes stash2ba4a30 retained/restored. Current-source incoming/backend/frontend qualifications and minimal test-fixture repairs are recorded in the active X02 plan. Original running checkout/session remains preserved. Fresh API3817-file package qualified; current private frontend and X02 live drain/in-flight checks still required. X01 stays closed by its selected composition; no full43-case requalification or release-wide completion claim.


## 2026-10-24 X02 closeout and next dev integration

X02 selected functional and human UX acceptance is complete by the [retained master closure](IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#x02-account-and-browser-isolation-selected-closure):35/43 accepted, eight open. Its own completed plan and released payloads/operators are removed; detailed failed outcomes remain in the official task. The earlier active-plan and pending-X02 notes are historical. X02 qualifies e388823/dev17141/frontend9U by explicit composition. New handoff devc90 contains merged PR3310 head0ae8; open PR3311 advanced to fad63/basec90. Neither is covered by the earlier reference-only c0cd/8d369 reviews or current candidate. Actual source review, preserved-fix rebase and genuinely affected checks precede X03; Stage3 remains In Progress. Original source/runtime/session and all retained limits remain protected.
