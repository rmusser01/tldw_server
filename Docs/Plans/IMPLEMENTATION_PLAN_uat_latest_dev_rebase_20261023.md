# UAT fixes on latest dev — 2026-10-23

Task: TASK-13260.278.18.83.45. Human explicitly authorizes rebase, necessary repairs/setup/retries and sequential completion of all43 workflows. X01 remains sole active;33prior accepted,10open. Prior frozen qualification does not qualify this new candidate.

## Constraints and rulings

Preserve current runtime/source/profile/data/keys/build/Chrome and complete unpublished fixes. Work in the separately managed uat-latest-dev-20261023 checkout. Original523 plus765tracked/345new source-doc-test files; exclude every .next build, model/data/cache/runtime lock/secret. Latest verified dev bd9b4e0ac3f65f3b4fcc4796842288e5a451b3a3; refresh exact remote before final qualification/publication. PR3084 is merged/immutable. No GitHub queue/push/merge action in this integration unit. Existing startup120s/generation180s/workflow600s/case300s/operation and gracefulcleanup15s remain. No force or weakened acceptance. Heartbeat stays paused.

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
**Status**: In Progress

## Stage3: Finish X01 and continue sequential UAT
**Goal**: Actual installed combined candidate passes X01 modes and recovery before X02.
**Success Criteria**: Version-source checkpoints oldest justified plusv0.1.43/v0.1.46; all required SQLite/PostgreSQL single/multi upgrade, backup/separate restore, preserved old data/accounts/Jobs, new writes, real provider, UI/reload, malformed/recovery and complete lifecycle/handback checks. Prior accepted UX remains; screenshot for changed UX. Dispose completed functional bulk/operators promptly. Update officialtask/X01/masterindex accurately, then move one workflow at a time.
**Tests**: Reviewed actual drivers with current source/package/build binding, no model replay merely for operator diagnosis. Explicit limitations retained; no latestdev readiness or completion claim from old artifacts.
**Status**: Not Started

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
