# UAT fixes on latest dev — 2026-10-23

Task: TASK-13260.278.18.83.45. Human explicitly authorizes rebase, necessary repairs/setup/retries and sequential completion of all43 workflows. X01 remains sole active;33prior accepted,10open. Prior frozen qualification does not qualify this new candidate.

## Constraints and rulings

Preserve current runtime/source/profile/data/keys/build/Chrome and complete unpublished fixes. Work in the separately managed uat-latest-dev-20261023 checkout. Original523 plus765tracked/345new source-doc-test files; exclude every .next build, model/data/cache/runtime lock/secret. Latest verified dev 4e7be41029b18881e55aa4260dfcb83baf92c45c; refresh exact remote before final qualification/publication. PR3084 is merged/immutable. No GitHub queue/push/merge action in this integration unit. Existing startup120s/generation180s/workflow600s/case300s/operation and gracefulcleanup15s remain. No force or weakened acceptance. Heartbeat stays paused.

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
**Goal**: Actual installed combined candidate passes X01 modes and recovery before X02.
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
