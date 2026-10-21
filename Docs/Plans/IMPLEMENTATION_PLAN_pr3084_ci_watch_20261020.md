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

## Ninth bounded fixture qualification

Published eighth batch is a33e947d74dd351c8e95a6c6e81193b4b6914a4c, verified OPEN/dev with auto-merge disabled after a normal fast-forward push. Configured checks passed its eight paths explicitly.

Native status regressions now assert exact canonical interrupted/stopped flags while excluding hostile native local_history presentation/recovery hints. Metadata/raw immutability and genuine local-owner positives remain. Both real message surfaces assert passive accessible saved status, exact saved-content copy, retained answer/server id and no local recovery controls; lazy content is awaited. Projection/helpers18/18 and API-send13/13 pass. Message73unique cases qualify across71passing full-run cases and4focused passes, not one73green run. The API transport fixture retains the actual storage serializer and its intentional read override. Four lint scopes add0diagnostics, PythonBandit is inapplicable and independent connected review is clear for this batch only. No architecture rule, production guard, allowance, package version or lockfile changed.

Further hosted frontend fixture failures and selected-native regeneration E2E qualification remain actionable. Accepted ADR049 and current TASK13559/D7 CM01 permit qualified selected-native regeneration; preserve nonready/streaming/no-automatic-replay negatives and verify an explicit branch against actual selected ancestry instead of simply changing the old blanket refusal assertion. Whole-PR required gates, trusted expanded-summary disposition and verified merge remain open.

## Handoff reassessment checkpoint

The coupled local-history fixture has reached three attempts and is not qualified:20standalone passes, then2mounted negative passes, while the positive retains its original callback assertion failure despite a real H1-ready probe passing. No further unchanged coupled rerun. Incoming option/session snapshots, MemoryDexie profile/readability guards and actual H1/draft guards remain.

Reassessment compared the independent LocalChatList positive and cockpit fallback, their shared loadConversation/onLoaded/captureChatDraftLoad chain, and existing core CAS/New/ABA controls. Phase1 now isolates actual H1 to actual draft acceptance without sidebar/query/model, observing callback count/result, fences, address/revision and load return. Phase2 traces the caller only after that causal boundary is established. Phase3 applies the smallest reviewed root repair with replacement negatives. No guard is mocked true.

Independent Maintenance render isolation1passes with the current user-ID/email API fields. Media-budget WIP remains unqualified after its third bounded attempt: the current timeout oracle is not observed during isolated bootstrap, despite current exact scope/timeout request expectations and a stable stored-connection snapshot. Preserve the retry/count/timeout assertions; reassess actual query state instead of more blind retries.

## Qualified profile-local opening repair

The minimized direct H1/draft CAS control passed, while the actual local caller reproduced the original fresh-address/IDs-null refusal. Connected review confirmed a production defect: profile qualification awaits before open increments its epoch, so capturing the counter after calling async open records the old epoch. The shared local loader now captures its expected next epoch before open. All result/epoch/owner/current checks and profile-preflight ordering remain; failed preflight does not retire a valid prior view.

All261cases pass in one connected run:26local-load,172controller,19draft-transition and44actual on-device sidebar tests. That includes the original local-load assertions, real H1/draft receipt acceptance, a retired-reader negative and existing cancellation/profile/navigation/ABA controls. The separate Maintenance render-isolation case1passes with exact current user-ID/email allowlist fields. Lint adds0diagnostics across these3paths (H1source3inherited, other2zero). PythonBandit does not apply to TypeScript. No new architecture decision, API, schema, dependency or quality allowance is introduced. Final connected review and normal checks precede publication.

The independent unqualified cockpit fallback remains open; this qualified on-device epoch defect does not explain its unqualified opening path. Media-budget WIP is also unqualified and excluded from this commit; its timeout fixture reached the bounded-attempt stop and requires actual query-state reassessment. Current source qualification is not whole-PR or live-UAT acceptance.

## Qualified admin query fixture repair

The timeout query and workspace receive the correct error, but the initial async canonical mount needs to settle inside React act before the original message/retry oracle. That controlled fixture change passes without changing production error handling, transport or cancellation. Both budget selectors now await the real card after connection hydration. Exact request checks include the existing canonical snapshot and abort signal while preserving the10000ms cap and two-call retry contract.

The StrictMode case retains Create-user visibility and now checks exactly two query attempts: the abandoned mount signal is aborted, the current signal is distinct/live, and a late retired result cannot replace current statistics. The two-file cohort was16passed1failed/17, followed by1passing focused card correction with11excluded;17unique cases are qualified, not one17green run. Both lint scopes report0diagnostics; PythonBandit is inapplicable. Independent connected review is clear. No production guard, architecture rule, API, schema, dependency or baseline changed; existing canonical authority/cancellation decisions govern without a new ADR.

Current published1b318b9587 remains OPEN/dev and unarmed. Other CI repairs, seven final-head required statuses, trusted expanded-summary disposition and verified merge remain open.

## Qualified Compare mirror identity repair

Admin fixture batchc585c46d3b1c4dea439404568d3d3429c4afe451 was safely published and GitHub read back OPEN/dev with auto-merge disabled.

Two original Compare snapshot regressions exposed incomplete positive fixtures: mirror helpers now require real captured admission and publish their exact local alias while preserving server identity; loaded saved identity is explicit. After those prerequisites, the actual production metadata fence still compared the admitted mirror ID against captured pre-mirror null. The one-line repair initializes Compare metadata identity from the already admitted enhanced chat parameters. Exact store-ID equality and every Scene/draft/account/session/dispatch fence remain.

Causal red was2original failures plus2passing New/same-ID replacement controls with73excluded. The final complete cohort passes77/77, including every original assertion and both replacement controls. ESLint adds0diagnostics across source22/22 and test2/2 inherited findings; PythonBandit is inapplicable. Independent connected root/final reviews are clear. This restores the existing captured-identity contract without a new durable architecture rule, API, schema, dependency or quality allowance; no new ADR is needed. Normal configured checks precede publication.

Other valid CI failures, final-head required statuses, trusted expanded-summary disposition and verified merge remain open.

## Qualified Persona fixture ownership repair

Compare root batch115f35eb4f5594fd85e7dbcb34595b0f0601b2fa was safely published and GitHub read back OPEN/dev with auto-merge disabled. All current review threads are resolved; no new human PR comment supplies the pending summary disposition.

The shared SDK Scene fixture now copies the supplied metadata-loaded flag into real core state. Existing-conversation Persona positives explicitly declare loaded identity, while a new unloaded-identity control retains the draft and refuses dispatch. The obsolete implicit character-to-Persona test contradicted canonical saved identity: the real picker detaches before replacement. It now proves refusal without clearing or dispatch. All13original clearing oracles remain at the real reset-helper boundary, with all resets ordered before creation. Fresh SDK positives retain the exact new-Persona payload, final identity, provider parameters and saved-turn signal identity. Production identity guards remain untouched.

The connected run was83passed35failed/118: all27Persona and48reset-helper cases passed; exact saved-signal positives subsequently2passed25excluded. Actual Form detachment1passed220excluded. The three sibling SDK files also fail35/43 using the exact pre-edit fixture, with every failure name identical; this introduces no sibling regression and does not qualify that scope. Three changed TS files report0lint diagnostics; PythonBandit is inapplicable. Independent connected final review is clear. No new architecture rule, dependency, schema or baseline is introduced; no new ADR is needed. Normal configured checks precede publication.

The35sibling failures and other remaining CI scopes, final-head statuses, summary disposition and normal verified merge remain open.

## Qualified sibling SDK fixture repair

Persona fixture batch59944a2dc03a8ed9f01958f26f00805db2cd628a was safely published and GitHub read back OPEN/unarmed. The previously identical35sibling failures are now resolved in their bounded scope.

Image and dynamic-action fixtures explicitly declare their loaded neutral saved identity; the fresh image case explicitly remains unloaded. Overlay metadata-retention tests now admit a loaded neutral chat and deliver canonical character metadata only after the real local save is held. Every original final title/character/version/meta/history-ID assertion and retired-owner control remains. A separate known-character overlay case refuses with retained draft and no dispatch/clearing. No production guard is bypassed or changed.

All44cases pass in one complete run (original43plus1refusal). Three lint scopes add0findings: dynamic12/12,image21/21,overlay0/0 inherited. PythonBandit is inapplicable. Independent connected final review is clear; no new architecture rule, dependency, schema or baseline is introduced and no new ADR is needed. Normal configured checks precede publication. Other CI scopes, final-head required statuses, trusted summary disposition and verified normal merge remain open.

## Qualified Character fixture readiness repair

Sibling SDK batch35e90fb83d271ac7db9f40e36cfc523642202b83 was safely published and GitHub read back OPEN/unarmed.

The separate private Character Scene fixture likewise omitted saved identity readiness. It now declares the tracked saved-chat prerequisite and fills missing state while preserving an explicit currentfalse and freshfalse. Two new refusal controls cover unloaded props and unloaded current state. No original assertion, production guard or H1 authority changes.

The initial84case run failed81/passed3; the first focused repair passed the five original progress/greeting cases, with two new controls requiring a missing value-import correction. Final complete qualification passes86/86 (original84plus2controls), covering existing auth, logout, timeout, retry, branch, attachment, greeting and cancellation assertions. ESLint25/25 inherited,0new; PythonBandit is inapplicable. Independent connected final review is clear. No durable architecture rule, API, schema, dependency or baseline changed; no new ADR is needed. Normal configured checks precede publication. Other CI, required final-head statuses, trusted summary disposition and verified merge remain open.


## Preparation retirement repair

Character fixture batch b299e13f583c96cf644b3735d088877ad0811089 was safely published and GitHub read back OPEN/dev/unarmed.

Two original New-during-held-preparation tests exposed a production lock retained after cancellation. Each preparation now carries its owning submission Symbol, and both retirement sites release only that token. Existing draft/account/context/scene admission, abort, identity and finalizer comparisons remain. No interface, schema, durable architecture rule or baseline changed; no new ADR is needed.

The original document19 and draft34 cases pass53/53 in one run. One added late-old/held-new control passes separately with19excluded, qualifying54unique, not one54green run. The separate extracted compact command fixture now supplies its missing revision/owner normalization adapter and neutral intent/context prerequisites; all12original scene/New/edit/Undo/model/tool/file assertions pass. That adapter does not qualify context-mode/account changes. ESLint source17/17,test8/8,Scene0/0 inherited,0new; PythonBandit is inapplicable. Independent connected final review is clear for this batch only. Normal configured checks precede publication.

Connected Sidepanel harness and two history fixture repairs remain separate WIP. The original disconnected readonly assertion stays intact pending the existing human question; final-head required statuses, expanded-summary disposition and normal verified merge remain open.


## History fixture receipt qualification

Cold-session fixture exports the current local binding dependency with an explicit unexpected-binding failure, preserving real H1/lease execution for already-owned rows. The explicit imported-consent fake controller supplies a matching local profile/owner and a typed valid view/opened receipt; the real draft transition still decides admission. Every original pre-binding refusal, readable-row, metadata, prompt, model, file and title assertion stays unchanged.

Initial complete48+21history cohort passes69/69. Independent review identified an invalid fixture cursor/missing session ID hidden by existing any; corrected to a typed HistoryLoadReceipt view with a fixed session and after_message/image-question. The exact consent case passes again1with20excluded. This qualifies69unique cases across full/focused runs, not one final69green run. Initial ESLint14/14and17/17 inherited,0new; final typed-file lint and configured checks precede publication. No production logic, architecture rule, dependency or baseline changes; no new ADR or PythonBandit scope. Connected final receipt review precedes commit.

Preparation batch caa155f739b828ebcfc822ff802cbbf91898d00c is committed locally. Connected Sidepanel harness remains WIP and readonly disposition is pending. Fresh b299 UX Smoke log119203461533 confirms the obsolete final blanket selected-history Regenerate-disabled expectation; it requires contract-preserving qualification. Other final-head gates and verified merge remain open.


## Prompt component fixture qualification

The published head2b53466117be1770e4fdb84abca0297292e24244 contains the reviewed preparation/history batches and was read back OPEN/dev/unarmed. Three remaining Prompt regressions reproduce on that source.

ProjectSelector unit setup now supplies a fixed authenticated owner, verifies the exact existing owner policy alongside the original create payload/onSelect/invalidation, and adds an unavailable-owner refusal. PromptSelect UX setup follows the existing server-library fixture with matching authority/resolver/transport and owned local data; its original local/server-list assertions remain. The clean-save/Back regression now proves its old create editor has detached and clicks the current saved-ID editor; original no-confirm/closed/library assertions remain. No production or durable architecture rule changes; no new ADR is needed.

All5focused cases pass: complete3ProjectSelector plus1server-library and1clean-save, with80other PromptBody cases excluded. ESLint9/9,0/0,20/20 inherited,0new. PythonBandit is inapplicable. Independent connected review and normal configured checks precede publication. Separate Sidepanel readonly and live smoke qualification remain open, as do final-head statuses, trusted expanded-summary disposition and verified merge.


## Saved-history smoke and inventory qualification

Prompt batch0f361224cf0abdc65e29650207cc22c77439090a was safely published/read back OPEN/dev/unarmed; its five configured paths passed. The current smoke correction replaces the obsolete blanket native-history regeneration refusal with exact saved-pair/H1 admission, one explicit branch, immutable original rows/versions, no duplicated user POST and retained unsent text. Initial/no-response/no-automatic-replay assertions remain. The bounded-attempt reassessment identified the new WIP header wait as incompatible with existing1024-character moderation holdback; exact saved-input admission now precedes unchanged strict Stop, response errors are checked afterward, and null requires an actual Stop click. A late captured settled child is handled by explicit UI reselection while retaining the leaf assertion. No moderation, production, timeout or allowance changes.

One fresh official isolated dev:webpack/Chromium/SQLite run passes1/1 in1.2m with mandatory Stop evidence and temporary250ms/one-word provider chunks. Owned services stopped and the runner exited0. This qualifies that exact smoke only, not all Stop paths, hosted UX or broader live UAT. Three earlier executed/setup outcomes and their limits remain in the task; no repeated broad suite.

Palette fixtures now provide their resolved synthetic owner and owner-stamped rows while the real inventory filter remains active; all20original inventory/UUID/invalidation/disabled checks plus1foreign-owner refusal pass21/21. The media static publication-order oracle now names the actual private publishSelected setter, retaining its order assertion; its2cases plus3real permalink hydration cases pass5/5. ESLint smoke53/53,palette0/0,media0/0 inherited,0new. TypeScript-only scope needs no PythonBandit; existing ownership/history decisions govern and no new ADR is needed. Independent final connected review and configured checks precede publication.

Backup's real Popconfirm preparation needed the opening event to settle inside awaited async act before the original visibility assertion. The candidate geometry shim did not resolve it and was removed. All original22restore/overwrite/cancel/exact-scope/invalid-metadata/refusal/admin-guard cases pass in one final complete run; lint0/0. Independent final connected review is clear across these four test paths, excluding Sidepanel WIP. Six configured path checks precede safe fast-forward publication.

Fresh exact0f361 hosted diagnostics retain further frontend/coverage failures; its UX job failed Playwright browser-download DNS before tests. No required-gate dispatch or infrastructure allowance is added. Sidepanel fixtures, other valid current CI failures, seven final-head required statuses, trusted summary disposition and verified normal merge remain open.


## Connected owner fixture completion

Nineteenth batch26c5cdc1e01db5b9faf21f951e8f7ad88f5f20c9 was safely published/read back OPEN/dev/unarmed after all six configured path checks passed. Five owned generated Next build directories from stopped isolated runs were removed; external reports and shared dependencies remain.

The human directly approved the pending disconnected-composer contract correction. Its original retained text, disabled Send and zero queue/provider assertions remain, while the single obsolete readonly assertion now checks editable input. The real Sidepanel harness supplies H1/storage and captured Alice transport prerequisites without bypassing ownership. All25draft/model cases pass in one final cohort. This approval does not answer the separate expanded-summary question.

Composer's missing shared-local inventory fixture now uses the real ownership predicate and deleted filter, preserving intended unowned/unlinked recipe data and all persistence-owner/credential/uncertainty assertions. All69composer cases pass; the unchanged Kanban owner-lifetime file also passes all15cases in the same84case run, so no speculative Kanban fix is added. Notes UX receives the same matching synthetic canonical config/active user/real graph hash as existing Notes fixtures; all5original cases pass, including tag retention and added expected-user/version header checks. These84,25and5cohorts are separate runs, not one114case pass.

No production guard, API, architecture rule, dependency or allowance changes. No new ADR or PythonBandit scope is needed. Final lint, connected review and seven configured path checks precede publication. Further valid CI failures, final-head required statuses, trusted expanded-summary disposition and verified normal merge remain open.


## History, Scene and persistence fixture qualification

Twentieth batch b8b71d138b0d86eee45f64b26db769268f43b978 was safely published and read back OPEN/dev/unarmed. The original human summary and fresh Cubic suffix were preserved byte-exact during the technical description update. The human subsequently approved the separate Workspace caption correction and directly stated “all approvals granted”; this confirms the pending expanded-summary waiver without attributing AI prose to the human. Normal merge authorization is reaffirmed.

Cockpit supplies the same fail-closed binding export used by its already-owned H1 path, preserving original assertions; all67cases pass. Coordinator seeds its existing authoritative transcript map for Cedar/ordinary/Robot; the original four B02 variants pass with76excluded. Persistence supplies its missing mandatory real metadata-loaded setter. Its30cases and Workspace38cases pass together68/68. The approved Workspace test requires Server connected and rejects Ready/Healthy claims while retaining the degraded registry control.

Scene preserves actual system-prompt utility exports and explicitly opens the real Scene notes blade after a source change. The original exact held-reset value assertion remains. The full112case run passed111 and failed that navigation-dependent case; a wait-only attempt also failed. The final exact case passes1with111excluded, qualifying112unique across runs, not one final112green run. No broad replay is claimed.

Independent connected review is clear for these five fixture paths. Final ESLint adds0findings: Scene0/0,cockpit4/4,coordinator0/0,persistence2/2,Workspace4/4 inherited. No production guard, architecture rule, dependency or allowance changes; no new ADR or PythonBandit scope. Normal configured checks precede publication. Recipe/capture WIP, recurring hosted Kanban Undo, new scoped CI failures, seven final-head statuses and verified normal merge remain open.


## Scoped recipe, Notes freshness and quota fixture repairs

Twenty-first batch ffa06b54593d49dd83a41da29097052bef02855f safely fast-forwarded to the remote branch. The immediate GitHub readback was stale; a later exact readback verified ffa OPEN/dev/unarmed.

Recipe bridge mocks now supply owner-verified project defaults, owned local creation, real shared-local ownership/deletedAt filtering and the owner-filtered exact-pull collection. The system picker waits for its actual resolved owner query. Direct ambiguity keeps scoped quarantine; extension response loss requires the stronger global unknown-owner banner. Owner B cannot select A's recipe or dispatch its explicit pull; B's built-in Apply remains local. Recovery returns to A before actual Forget and preserves the original B-clear/A-scoped final registry assertions. These human-authorized contract corrections retain production privacy and all mutation/receipt/reconciliation guards. After the bounded failed-attempt reassessment, the last two failures were new waits using a guessed cache key; current source established getAllPromptsForSelect. Two prior full-run successes plus two final focused successes qualify4unique cases, not one final4green run.

Capture's shared scoped policy and real content/analysis editors legitimately permit POST to media versions. The obsolete negative now rejects PUT; a new exact POST positive checks principal, bearer credential, body and one direct dispatch. All79cases pass. Notes stale-version tests explicitly advance remote state and only the30second interval clock, retaining conflict, reload, original Version1/2, no stale PUT and no modal checks; both pass.

Quota resolver unit fixtures now include soft/hard status fields and versioned-delete orchestration. A distinct connection exists only inside the transaction context; capture/delete/touch require it and the same floor in exact order. This is a recording unit boundary, not actual profile-gateway SQL or PostgreSQL qualification. All11pass on final formatted source. Bandit20baseline/20current, zero errors or non-assert findings; added test-only assertions have explicit B101 regression annotations. Ruff1baseline/0current. Global quality5208passed unchanged baseline and final changed-file ratchet passes. Three TS lint scopes0/0. Independent connected final review is clear; no production rule, dependency, baseline or architecture change, so no new ADR. Six configured path checks precede publication.

Further hosted failures, seven actual final-head required statuses, current-dev integration at landing and verified normal merge remain open. The trusted direct all-approvals instruction resolves the expanded-summary waiver; preserve original human summary bytes.


## Hosted Kanban toast admission regression

Current d489 hosted shared-ui-coverage2/16 job119271340956 reproduces the original Undo DOM absence after the archive request and message.open succeed:2,084other cases pass. Earlier unchanged normal and V8 local scopes did not reproduce it, so those runs do not close the hosted failure.

The fixture now awaits the actual asynchronous archive event within React act before observing AntD's separately mounted static holder. Installed AntD source defers holder rendering and instance admission to a microtask; the original synchronous fireEvent wrapper did not await that work. All original visible Undo, retained callback, retired-owner and no-unarchive assertions remain unchanged, as do production code and timeouts.

Complete Kanban15 plus existing native static-root5 pass20/20 with V8 coverage. This qualifies the connected local scope, not the hosted165-file shard or aggregate coverage. ESLint0/0. No architecture rule, dependency or allowance changes; no new ADR or PythonBandit scope. Independent connected review and three configured path checks precede publication. The human explicitly requested continued active work through merge; required pending statuses are not a reason for an early status-only handoff.


## Current-head ownership, architecture and backup-tool repairs

Current307 hosted coverage10/12/14/16 and full-suite core-utils/admin-backup/admin-bundle/auth-runtime failures were read from exact completed job logs. The repair preserves original recent-note deletion, exact route warning, Admin Error/retry, stale task/activity, auth refusal, restore rollback and retry assertions. Notes list/races now supply actual canonical ownership; race keywords use the existing real React state pattern and explicitly await the old request before retiring it. A bounded reassessment caught the no-op keyword setter retiring the request before activity; no guard or timeout was relaxed. Admin fixtures own queued mock behavior and query-client cleanup. Setup transport accepts the actual provider-progress POST and checks its nonsecret body.

The human all-approvals instruction covers the obsolete prompt insertion contract correction: the real hook awaits usage recording inside captured ownership, then revalidates before publishing. Tests now assert no Chat publication while usage is held, and retain original final exact values/navigation and once-only usage calls. Existing retirement regressions remain unchanged.

Two remaining core imports now use the already owning core schemas instead of their API reexports. The manual terminal SSE frame uses the existing identical sse_done bytes. A transport double accepts and asserts the existing dict_row argument. PG18 backup lanes previously used pg_dump16; the existing wait action can now select a validated optional client major using installed binaries or signed official PGDG packages and bounded apt calls. Only the two backup matrix shards request18. Local selector checks cover installed tools and invalid input; actual Ubuntu installation remains hosted qualification. Existing Homebrew18.3 tools were discovered and used with official restricted fixtures, without dependency changes. The interruption fixture now puts its archive within its permitted backup base and verifies the file before the original real timeout/rollback/retry. The evaluations negative explicitly disables alternate TEST_MODE flags to isolate TESTING-only absence.

Qualification: Notes list2/2 V8; earlier four-file128 cohort127passed1failed followed by complete prompt4/4 qualifies128unique, not one128green. Later63 assertions passed but one unhandled rejection invalidated the run; the other two files50 cases passed there, and final Notes13/13 V8 exited0 after reassessment. Backend scope62/62; action/contracts72/72 plus finalselector2/2; real PG3 earlier plus interruption1 final qualify4unique, not one4green. Evaluations11 with interruption passed12/12. Six Python compile; Ruff1/1 no new; Bandit47/47 AST-aligned no new/errors; sevenTS ESLint57/57 no new. Global quality5208 passes unchanged baseline. Independent connected review and configured hooks precede publication. Governing architecture/stream/CI decisions remain unchanged; no new ADR, production ownership/security rule, dependency or allowance. This is scoped CI qualification, not full-suite/whole-PR/live-UAT acceptance.


## Serial integration with dev5c2 semantic Notes migrations

Before batch24 publication, dev advanced via PR2862 to5c2e2b4c804045aa01dc74e2854f53fb902a4849 (170paths). Qualified local1996fabd836f3b448c9bd9747ddc910882dc89cb is retained at codex/pr3084-qualified-batch24-20261021. Topology-preserving rebase retained both historical merge nodes; comparison with predicted raw merge-tree5f6e433626fead32497d377448d647da3a219e5a differed only in the three known conflict paths. No dropped historical source resolution was found.

The merge exposed an actual migration-number collision: published semanticPG86–88 and PR-only dictionary86 shared a method name. The composition preserves published SQLite82–84 and PG86–88, moves dictionary ownership to88→89 with both dispatch paths, CAS and verifier threshold, and removes the duplicate method/dispatch. Under the existing bootstrap lock, a verified old dictionary86 layout with none of all seven semantic relations receives unchanged semantic86 SQL while retaining its version86 stamp. Any existing semantic relation requires the real semantic verifier; missing dictionary forced-RLS refuses. Normal87/88/89 then run. No version rewind, owner adoption, guard reduction or new architecture rule. Existing migration decisions govern this compatibility repair, so no new ADR is required.

Real legacy86 retention/partial/unsafe regressions first reproduced1failure2passes; complete final dictionary/semanticPG/v67PG cohort42/42 passes, including dev88 retained configuration. Historical target86/87/88 controls remain. SQLite semantic/Jobs/architecture/SSE cohort93/93 and privilege/scope23/23 pass. Existing28complete guarded If ASTs match pinneddev exactly; only baseline locations regenerated. Combined canonical OpenAPI2140paths3316schemas fingerprint2894b02f6a25027752aa9cccddd29f10a773904b8ac04db82a7c9d4cf8f2acff; committed path union matches. Later hosted stale privilege snapshot now records existing expected-user dependencies on three routes across six buckets, with no removals. Later Docker failure is a mirror.gcr.io quota refusal before tests, not source acceptance.

Independent final connected review clear for composition. FourPython Bandit95parent/93current,0newAST-aligned/errors; Ruff0. Global quality and all configured integration path hooks precede publication. This qualifies the affected integration scope and retains the earlier batch evidence; it does not claim full-suite/whole-PR/live-UAT approval. Final-head hosted gates and fresh dev ancestry remain required.


## Hosted Skills initial render admission

Published105 shared-ui-coverage15/16 job119323013821 fails one initial Select shared-skill lookup while1,915other cases pass. This happens before the server-switch event. The unchanged full Manager file83/83 passes under V8, so there is no local red reproduction. Connected source review found valid single-user scope/config prerequisites and no evidenced earlier getConfig one-shot leak; speculative production, mock-reset and query-client edits were rejected.

The minimal fixture correction awaits async React act around that exact initial render so the real scope/query updates settle before selection. Every original checkbox, drawer, count2, scope retirement, foreign-row absence and timeout assertion remains. Complete candidate83/83 passes with V8; ESLint18/18 no new. Independent read-only review clear; three configured path hooks precede safe fast-forward publication. PythonBandit is not applicable to this TS-only fixture change. No architecture decision, production guard, dependency or allowance changed. Hosted closure remains pending until the new head's full shard succeeds.


## Exact Back-navigation completion observation

Current cc02 shared-ui-coverage2/16 job119344249482 fails the hash Back case with zero confirmations while2,089other cases pass. The unchanged complete16-case navigation file passes locally under V8, so no local red reproduction is claimed. Its final substring root-route wait also accepts the old /programmatic route, allowing router disposal before the asynchronous POP/blocker confirmation.

The minimal correction requires the exact root route before unmount. All original confirmation-count, error, disposal and timeout checks remain, as does production code. The complete candidate16/16 passes with V8; ESLint0/0. Independent connected review and normal configured path hooks precede publication. No new architecture rule, dependency, baseline or allowance; no new ADR or PythonBandit scope. Hosted shard closure and all seven actual final-head required statuses remain pending.


## WXT build child memory admission

Current c99 Extension Research Workspace Parity job119351225684 fails before tests when production Chrome WXT exhausts the default approximately4GiB heap. Both shared profile wrappers inherit caller env, and zip independently performs a production build. The frontend8192 setting belongs to coverage reporting, not the extension build job; the earlier hypothesis was corrected after source review.

Both WXT children now default absent NODE_OPTIONS to6144MiB while retaining explicit caller options, including empty, and leaving parent env unchanged. The same one-line default in each existing child boundary avoids a new abstraction and keeps the existing isolated live-tier environment policy. No runtime ownership/security rule, dependency, quality/coverage allowance or durable architecture decision changes; no new ADR or PythonBandit scope.

The absent-option regressions first fail2with11passing; final complete connected build/zip/profile scope16/16passes. The actual Chrome production build with no parent NODE_OPTIONS exits0 in55.6seconds and exports50.45MB. Both scripts pass node syntax; ESLint build0/zip2inherited/test0, no new findings. Independent final connected review and five-path normal hooks precede publication. The heap cap does not bound totalRSS; Firefox/Edge/archive and actual parity browser scenarios remain hosted qualification, as do all seven final-head required statuses.


## Admin stats Error observation after asynchronous mounting

Current eae coverage12/16 job119358142862 fails the original stats Error/Retry case after1443ms while1,824other cases pass. The DOM still shows the transitional Empty stats panel. Earlier mock reset and owned query-client cleanup remain but did not close this recurrence. The unchanged complete12-case file passes locally with V8, so no local red reproduction is claimed.

The real page first awaits canonical config, mounts the heavy AntD workspace, receives a timer-batched query notification, then copies query error through an effect. The existing one-second Error lookup began from the resolving-config placeholder. The minimal candidate awaits React async act around this exact initial render before the unchanged Error lookup. It retains the immediate rejection, original visible Error/Retry/count2 and every timeout; no production, mock, cache, ownership or guard change.

Complete candidate12/12 passes with V8; ESLint0/0. Independent connected review and normal three-path checks precede publication. Hosted closure remains required, and a further recurrence needs renewed phase evidence rather than another guessed wait. No new architecture rule, dependency, baseline or allowance; no new ADR or PythonBandit scope. Six required contexts have passed on eae, but the known frontend failure still blocks merge and all statuses must qualify the final head.


## RBAC assignment recovery queries in the owning role card

Current2e01 coverage1/16 job119383875127 times out the original missing-role/refusal/fresh-selection journey at15seconds while1,875other cases pass. Both hosted AntD message warnings show the path reached refusal and successful replacement. There is no connected evidence of a stalled mutation, a production guard problem or a need for a longer timeout.

A temporary exact-case V8 phase profile passes1with14excluded in3.37seconds; original bytes were restored exactly. The body-wide initial/final cell waits consume582/1037ms locally. Pinned Testing Library computes accessible names across matching cells before visibility filtering, including other permission tables and retained tabs. The minimal fixture scopes the same three exact cell queries and Add Role control to the selected user's real Assigned Roles card. This strengthens the correct-table assertion; original global guard, effective permission, dialog, mutation-order and15second checks remain.

Complete candidate15/15passes with V8, target4.892seconds in that full run. The focused baseline/full candidate are not a comparable speed benchmark, and no local red or throughput improvement is claimed. ESLint0/0; independent connected final review clear because the owning card persists through data reload for selected user11. Normal three-path checks precede publication. No architecture rule, dependency, guard or allowance changes; no new ADR or PythonBandit scope. Actual hosted closure remains required.

A separate current-head admin-reset-backups job119384338261 fails before tests because the PostgreSQL18 mirror rejects pulls for quota. No source workaround or acceptance is claimed; the next necessary source push naturally retries that lane. The six required successes on2e01 do not transfer to a new head; frontend/known failures still prevent merge.


## Chat initial-setup complete composer fixtures

Current0a66 final frontend-required job119421588630 fails only the two fresh-input restoration cases;2,453other cases pass. The unchanged complete file reproduces2failures/12passes locally. The real composer initializes message, image and imageFilename, merges partial values, and captures/restores the complete form. Both fixture objects omitted the existing empty filename.

The minimal repair adds imageFilename:"" to those two objects. Exact whole-form equality, metadata restoration, address retirement, null chat IDs, no unresolved registry storage and no Chat request checks remain unchanged. Complete candidate14/14passes with WebUI V8; the configured local report shows0percent and does not qualify aggregate coverage. ESLint0/0; independent final connected review clear. Normal three-path hooks precede publication. No production, timeout, dependency, architecture rule or allowance change; no new ADR or PythonBandit scope. Seven actual final-head required statuses/currentdev/reviews and verified normal merge remain open.


## Prompt Studio waits for visible canonical results

Current eba7 frontend-unit4/8 job119447691000 fails one canonical-answer visibility check while1,017other shared cases and54WebUI cases pass. The original findByText observes badge presence before the immediate visibility assertion. Installed dialog preparation still runs with motion:false, and AntD applies opacity0 until that preparation clears. The unchanged full15-case file passes locally, so no local red is claimed.

The minimal one-line correction waits for the original exact1ungraded badge visibility assertion with the unchanged default timeout. Exact twoanswers,0.00s,noPass and all provider/refusal contracts remain. Complete candidate15/15passes in the owning normal UI runtime; ESLint0/0 and independent connected review clear. No production, mock, dependency, allowance or architecture rule change; no new ADR or PythonBandit scope. Normal configured path checks precede publication. Remaining current-head outcomes are collected before another necessary push; actual hosted closure, all seven final-head statuses and verified normal merge remain open.


## Media filter completion observes rows and count together

The same eba7 frontend-unit5/8 job119447690934 fails only the final total2 check after cached unfiltered rows[1,2] return;1,295other shared cases and9WebUI cases pass. The actual query key restores its initial cached rows immediately, while the separate local total updates in the asynchronous refetch. The unchanged complete18-case file passes locally, so no local red is claimed.

Both original exact final assertions now run together in the existing default waitFor, requiring rows[1,2] and total2 in the same observation. No assertion, timeout, production, cache or mock contract changes. Complete candidate18/18passes in the normal owning UI runtime; ESLint0/0 and independent connected proposal review clear. These18 and the separate Studio15 are not one33-case green run. Four-path normal hooks and final source review precede publication. No new ADR, PythonBandit scope, dependency or allowance. Remaining actual hosted outcomes are collected before a grouped necessary push; final-head required statuses/currentdev/normal merge remain open.


## Runtime log timeout uses an isolated real held lock

Current eba7 core-server-smoke job119452793948 fails only the runtime log timeout refusal;227other cases pass. Its fixture created a plain lock before reload/settings/config and enabled queued writes before switching the lock implementation. Stale-file expiry and queued-writer interference are source-supported risks; the hosted log does not prove which occurred. The unchanged complete logging file passes21/21 locally.

The test now sets the same env timeout0.2 before lazily requesting the existing isolated fallback fixture, acquires a real outer holder, then times the inner contender using its configured default timeout. Original RuntimeError and elapsed<0.5 remain; all original assertion ASTs are identical. The shared fixture, production lock and stale recovery remain unchanged. Complete candidate21/21passes under Python3.12/retained pytest9.0.3. Compile passes, Ruff0/0; Bandit59/59 has no new finding class/count or scan error. Global test-quality ratchet passes unchanged baseline. Independent final connected review clear; five-path hooks precede publication. No new ADR, dependency or allowance. The three separate full-file results Studio15/Media18/Logging21 are not one54-case run, and actual hosted closure/final required gates remain open.


## Character tests use the real application style context

The final eba7 shared-coverage13/16 job119447691464 passes all3,180assertions with1skip but exits124 at the900second process deadline before coverage finishes. Its Character first-use104-case file takes855.828seconds. This is not coverage acceptance. A native style-cost profile finds414getComputedStyle calls use8,993.65ms of a9,558ms greeting case. Temporary instrumentation is removed exactly. An initial motion-disabled wrapper yields33.765s versus33.315s for three matched V8 cases and is rejected.

Both real app surfaces already use StyleProvider hashPriorityhigh; the isolated fixture used cssinjs defaultlow :where selectors. Its minimal RTL wrapper now supplies the exact production style context across initial renders and rerenders. Native motion, lazy held forms, real drawers, style/accessibility checks, all assertions and every timeout remain. The same three V8 cases pass in17.759s versus33.315s, a scoped single comparison. Complete final Character104/104passes under V8 in209.40stest/222.95swall; this local full-file time is not a controlled comparison with the hosted shard. ESLint122/122 inherited, zero new findings. Independent connected review clear. No new dependency, architecture rule, production behavior, quality/coverage allowance or deadline change; no new ADR.

The grouped batch has four separately qualified complete files: Studio15, Media18, Logging21 and Character104, not one158-case green run. Six-path normal configured checks precede publication. All seven actual final-head statuses, actual hosted budget closure, current-dev/review/queue verification and normal merge remain required. Broader live UAT stays open.
