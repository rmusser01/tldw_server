# PR3084 CI repair and merge plan

Task: TASK-13260.278.18.83.45. The requester authorized fixing CI and merging PR3084 into dev. Existing checkpoint and live UAT are separate. Implement serially in the clean checkpoint worktree.

ADR required: no. Fixture, documentation publication and baseline-location corrections preserve existing behavior and architecture. ADR059 governs task edits. Never increase ratchet offense counts.

## Stage 1: Repair failing UI and documentation fixtures
**Goal**: Align Notes Dock request ownership and Playground history capability fixtures with current contracts; publish the missing proposed ADRs.
**Success Criteria**: Original cache invalidation, pending-history, handoff and docs assertions pass without weaker guards.
**Tests**: Current failing Notes Dock and Playground search cases, published-doc refresh parity and strict documentation builds.
**Status**: Complete

## Stage 2: Clear backend ratchets
**Goal**: Fix new test-quality offenses with actual body/state/output assertions and appropriate real boundary fixtures; refresh only moved conditional-predicate locations after verifying unchanged semantics/counts.
**Success Criteria**: No new or worse quality offenses; no added tenant predicate baseline allowances.
**Tests**: Scope and test-quality ratchets, affected test modules and touched-scope lint/Bandit.
**Status**: Complete

## Stage 3: Publish and finish integration
**Goal**: Publish reviewed fixes through normal hooks and get the final head merged into dev.
**Success Criteria**: Current-head required CI green, valid review feedback settled, verified human-summary disposition, current dev incorporated and normal merge confirmed.
**Tests**: Final diff/source review, current-head checks/reviews, GitHub MERGED state and merge ancestry on dev.
**Status**: In Progress

## First verified publication batch

- Notes/history:33unique UI cases qualified across32passing full-run cases and one corrected focused refusal case. No single33-case green run is claimed. Three originally failing documentation checks pass under supported pytest9.0.3.
- SQLite Jobs now binds JSON-array type membership in static SQL, retaining both candidate and future-scheduled membership guards.72Jobs/SQL guard cases pass;44PostgreSQL cases were deselected. Existing json_each use means no new architecture decision.
- All13uncovered test files are assigned in the existing five CI matrices; baseline/ignore unchanged, coverage guard new_uncovered=0. All28conditional-predicate guard ASTs match dev exactly; only baseline line locations changed.
-66authentication/prompt/MCP contract cases pass. Canonical-cookie expectations were corrected after connected independent review; explicit bad-header, owner, scope and RBAC denials remain intact.26focused registration/backup/grader/Flashcards cases pass;73scope/required-workflow cases complete exit0.
-16Python files compile; Ruff adds no diagnostics; Bandit87baseline/86current, zero new AST-aligned findings/errors. Notes/history ESLint counts3/3and19/19, no new diagnostics. Independent two-round source review is clear for this batch, not whole-PR merge approval.

## Second repair batch

- Monitoring now hides stale cached health after an error and supersedes pending generations on explicit refresh/retry. Its new imperative shared-query waiter test confirms replacement preserves waiters and ignores late old results. All21 Monitoring and4 shared-cache cases passed.
- Playground no longer resets twice after the shared reset helper. Shared restoration accounts for the explicit image ownership increment. Both chat surfaces honor the existing B03 no-admission flag before restoring a failed structured result; skipped and separately tested raw dispatch-rejection recovery remain. Existing positive assertions now use an explicit preflight refusal, and both surfaces have unflagged-failure negatives. Final connected recovery/draft/offline cohort98/98 passes. Across these cohorts121 unique frontend cases are qualified; no single121-case run is claimed.
- The six originally remaining backend quality files are repaired through owned real HTTP fixtures, real deterministic token accounting, exact configured-model prompt transport, scoped override restoration, real cache lifecycle and one consolidated cold offline accounting scenario. Character1 passes; initial affected cohort87passed1failed/88 plus corrected B02 focused case exits0; offline42 passes. No single88-case green run is claimed.
- Six Python files compile and Ruff reports0 diagnostics. Bandit119baseline/119current,0new AST-aligned findings/errors. Eight frontend files add no ESLint diagnostics; Sidepanel32 and Playground17 inherited diagnostics remain. Independent connected reviews are clear for this batch only.

## Third repair batch

The11further quality-file violations are cleared without baseline changes. Added assertions check real response/state/error propagation; shared backup workspace and opt-in audio cache restart fixtures own actual environment/cache lifetimes. The transcription route fixture retains real principal admission. Global test-quality scanner5199files exits0. The strengthened outcome cohort112/112 passes. The ledger/guard/PG cohort has16passes and3failures caused by unavailable local pg_dump/pg_restore executables; those three real backup/restore scenarios remain unqualified. Linux CI provisions the clients. Thirteen Python files compile; Ruff adds0diagnostics; Bandit95baseline95current with0newAST-alignedfindings/errors. Independent source review is clear for this batch only.

## Fourth repair batch

The shared composer now counts an explicit image setter mutation once: it updates the existing previous-image ref before the layout fallback. Sequential drops retain the second image while reset/raw-setValues/restore invalidation and synchronous owner/ABA/latest-read fences remain. Model Playground's local-input fixture mocks its current catalog boundary; Setup asserts both the exact safe progress payload and captured request scope. All146affected image/account/recovery/draft/Model cases pass; focused Setup1passes19deselected. All3changedUI files have0ESLint diagnostics; TypeScript-only Bandit is not applicable. Independent connected source review supports this scope.

## Fifth repair batch

Recipe Pull now acquires the existing exact reconciliation lease before copying content and validates captured owner/link/schema at each actual local mutation. Denied reconciliation preserves the exact original row; acquired write/release failures retain scoped evidence. Eleven new concurrent owner/link/schema replacement controls reproduced red before repair. Direct/extension settlement observers execute the real mutator once and hold its acknowledgement. The final recipe/uncertainty/structured/auto-sync cohort passes1182/1182. The preceding auth/structured/auto-sync cohort passes1206/1206; those cohorts overlap and are not one combined run. Auth tests preserve raw credentials/session state, require the documented opaque boolean invalidation marker and withheld effective tokens, and cover collection versionsPOST positively with itemPOST denied. Four UI ESLint scopes add0diagnostics (background10 inherited); no Python scope for Bandit. Independent connected review is clear for this batch. No new ADR: this restores the existing exact-authority ownership contract without changing APIs, persistence schema or architecture rules.

## Sixth repair batch

Opening a peer WebUI tab no longer deletes the shared cookie configuration before verification. Only the existing seed's locked exact-revision cleanup can remove a current failed marker; document-local invalidation and initial/recovery joining remain. Exact recovery fixtures include the existing UUID verificationRevision and target only the intended removal failure. Held-profile tests use their captured graph's real nonblocking canonical reader for local-null assertions and await an already-started client getter after its actual probe settles. All64runtime-bootstrap cases pass, then all8cross-tab cases pass;72unique cases across those runs, not one72greenrun. Three UI scopes add0ESLint diagnostics (one inherited source warning). TypeScript-only Bandit is inapplicable. ADR044 and the existing cross-tab plan govern; this restores the accepted contract and needs no new ADR.

## Seventh repair batch

OpenUI's documented standalone SDK actions/session stay outside the child composer provider. Its explicit Scene adapter accepts only a null or idle ownerless controller and checks the actual child account lease/scope; owned/non-idle H1 capture remains production-owned. Canonical user1/profile1 and user2/profile2 fixtures retain exact identity assertions. The actual server metadata loader supplies readiness from owned neutral network records; stable assistant fixture exports and two awaited loaded-state setups remove fixture races. All53unique OpenUI cases qualify across51full-cohort passes plus2focused passes, not one53greenrun. All347original expect-call AST chains remain.

Sidepanel Keep now exercises the actual native capability/settlement service through a finally-restored per-test capture spy. Owned local clicks await the actual initial tab; expansion awaits its qualified action. All5diagnosed cases qualify across3passes and2focused passes, not a272-case suite. All826original expect-call AST chains remain. OpenUI18/18 and Sidepanel0/0lint diagnostics are unchanged. No Python Bandit scope and no production/persistence/ownership/architecture rule change; no new ADR.

## Remaining work

All previously identified frontend failure paths now have scoped local qualification. Current-head hosted CI must qualify final source including the three locally unavailable PostgreSQL-tool scenarios. Quality baselines remain unchanged; global ratchet passed before frontend-only batches. Inspect exact current-head failures and remaining private-FastAPI review feedback. Verify trusted original human-summary waiver evidence before relying on it. Review clearance remains batch-scoped until final whole-PR review.

Auto-merge is disabled. Latest verified published head624a207a0b1a3dd25c329e50ea11fc4ce8eeb64c is OPEN and behind currentdev. The necessary serial integration is local and unpublished. MERGE_QUEUE was verified absent; check again before final integration. The10-minute watch remains active until verified merge into dev. Main UAT worktree/runtime and older paused heartbeat remain untouched. Broader runtime/UAT/release acceptance remains outside these CI results.

## Serial current-dev integration — 2026-10-21

Pinned dev ad946dd6877d5d31fd7ff5ca2cfdfd6fbdd641e9 adds ten commits across eight ChaCha PostgreSQL paths. The topology-preserving rebase retained both intended merge nodes, but its content check caught missing historical manual resolutions. Those reviewed bytes were restored from the clean independently computed three-way combined tree; both ADR rename destinations are retained. Before applying the task-only tracking stash, the index matched f64d64911dad56d847bc8de81423d41cc510dcde exactly. Original624a backup remains codex/pr3084-before-dev-rebase-20261021. An ordinary corrective commit records this recovery; no new final-content decisions beyond incoming dev. Focused PostgreSQL checks and independent connected integration review remain in progress. Required hosted statuses must qualify the eventual published head.

The outdated private FastAPI suggestion was independently evaluated: keep the real classification-cache admission assertions because they reproduce the causal retired-app retention, together with real route registration and weakref collection. No source repair is warranted; the outdated thread is now resolved. Original expanded-summary waiver evidence is still being verified.

## Incoming fallback-pool repair

Integration testing exposed that the default PostgreSQL fallback pool closed another live thread's checkout. After cohort-reporting/setup reassessment, a minimal new unit regression reproduced the real closure bug red. Shutdown now snapshots only locked idle connections; existing closed-pool return/discard closes borrowers after their use. A native-only optional import follows its unchanged existing skip check. Connected independent review is clear for these three paths, including return/close/concurrent-connect races. Global test-quality enforcement passes5202files with baseline unchanged. Ruff0; Bandit55/55zero new-context/errors. The final affected pool/real-PG cohort had51passes1failure1skip/53; its exact direct/fallback factory pair then passed2with21deselected after passing existing dict_row at direct factory admission. This qualifies52unique cases plus one unchanged optional-pool dependency skip, not one53green run. Connected follow-up review is clear. This restores existing borrower authority, with no new architecture decision or dependency changes.

Fresh dev advanced to6f226a48f239ac7c7fa835d0df767657158a4d28 during qualification, adding18caller-operation paths through PR3295. Preserve the local qualified ad946 integration before incorporating this next pinned delta. Normal publication and final-head hosted gates remain open.

## Second pinned integration qualification

The complete committed tree at486ed3bab6 equals canonical2eff32e319fe8f3ccbce0fb648c8cbcc652d6935 exactly before reapplying the task-only note. Dev6f226 is an ancestor and both historical merge nodes survive. All22new caller-operation/connected character-adapter checks pass with official PostgreSQL fixtures. Global quality enforcement5208files passes unchanged baseline. Incoming15Python Bandit229/229zero new-context/errors; Ruff69/20zero new. Independent connected composition review is clear for this integration. Upstream TASK13896 event-loop blocking and TASK13897 detached-workflow operation inheritance remain recorded dev limits; no live acceptance or closure is added. Earlier pool qualification remains52unique passes plus one unchanged optional-pool skip across full/focused runs.

Current candidate includes pinned dev6f226 and is prepared for normal final hooks and exact624a-lease publication. Final published-head hosted statuses, trusted original expanded-summary waiver evidence and merge verification remain open. No main UAT/runtime or older heartbeat changes.

## Current-head CI follow-up

Published923c reports two Sidepanel fixture failures and a production-extension build OOM before UX tests start. The Sidepanel repair supplies its standalone async storage writer and awaits actual initial/selected tab readiness before injecting the focus failure. All272cases pass in one full default-reporter run; original assertions remain. The exact E2Ebuild artifact identifies WXT0.20.13/Vite5.4.21 chunk generation exhausting the default4GB V8heap. Build resource/root-cause repair remains in progress; controlled environment isolation stays intact. These changes are local and unpublished; other current-head jobs are still running.

## Eighth bounded CI repair qualification

Sidepanel272/272, composer20/20 and DSRintake/regeneration28/28 pass in complete scoped runs. Exact composer snapshots include the existing image filename. DSR history uses the separately tested shared admin formatter and proves requested_at precedence over a distinct created_at. The exact regeneration payload retains its ownership data and no-copy assertions; the composed callback now has causal current/retired/current and real draft-retirement checks with store restoration.

The isolated extension child receives a fixed6144MiB heap without inheriting arbitrary ambient Node flags. All runtime command environments remain filtered. The actual Node20/WXT0.20.13/Vite5.4.21 Chrome production build exited0 in98.151s with5456527360bytes maximumRSS. The first setup-only127 invocation did not compile; existing dependency packages were linked locally afterward without shared changes. The strengthened environment case passes; the complete runner file has57passes and one retained local shared-dependency containment refusal, not58green. Six JS/TS lint scopes add0diagnostics; PythonBandit is inapplicable. No baseline/allowance/production ownership guard changed and no new ADR is needed. Final connected batch review and normal hooks precede publication.

Current923c hosted CI has further frontend/coverage and UXsmoke failures. Local-history fixtures need complete real store/H1/profile setup; selected-history Regenerate smoke expectations require connected contract diagnosis. Those issues and final required statuses remain open. This bounded qualification does not approve the wholePR or broader liveUAT/release.
