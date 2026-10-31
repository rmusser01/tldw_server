# C10 managed model/runtime administration to real inference

Task: TASK-13260.278.18.83.45. Continue the existing sequential 43-family UAT; C09 is closed. 30 selected acceptances, 13 open. C11 stays unopened until C10 selected closure.
Spec: Docs/Development/RELEASE_UAT_PLAYBOOK.md C-10 and release-catalog-data.mjs C-10.
ADR assessment: no new durable rule for qualification. Retain ADR-030 configured-local trusted-origin boundaries, ADR-043 supervisor-owned runtime boundaries and ADR-059 task editing. Reassess any substantiated behavior repair before editing production.

## Constraints and entry state

- Frozen523ef323e9f56bc31559f6c5b8e506c27e599dcb plus unpublished WIP; not latest dev. Heartbeat PAUSED. No Git/ref/index/fetch/stage/commit/push/rebase/merge.
- Original API52868/owner52843, frontend50832/owner50736/build9fW2FVTzfI054S_pnJB8P and same Chrome58150/owner58135/CDP18987/context/window/tab. Private profile/data/keys and exact five-message Character7 Chat/Interrupted/unsent draft plus specified saved records remain protected. No new window/tab or unrelated process control.
- Existing user service8010 is a configured inference dependency. C10 needs a separately declared run-owned instance before start/stop qualification. No lifecycle request, model file access, model download, install or broad inventory scan yet. Never inspect or reuse older partial Whisper/model assets, shared caches or foreign proof bundles.
- One initial plus at most ONE declared recovery per native case, maximum three failed attempts per issue before reassessment; no hidden retry or reset of previous STOPs. Preserve 120s startup, 180s inference, 15s request/save/graceful cleanup, 600s host/build. No force stop, timeout increase, arbitrary model download or induced system-wide OOM/disk exhaustion.
- Use normal Admin/user UI and separately authenticated API clients, scoped disposable profiles, official PostgreSQL fixtures and DB abstractions. No credential output/auth-store manipulation/raw operator SQL/ordinary-user elevation. Only necessary functional outputs; promptly remove completed bulk files, keep concise results/ownership controls.
- C09 build/API-cycle exception and all three native diagnostic slots are consumed; they do not authorize a repeated C09 run. Its qualified controls remain composed.

## Stage1: Declare runtime and map actual contracts
**Goal:** Identify the exact preapproved installed model/executable, owned process boundary and safe resource budget; map UI/API/config/provider paths.
**Success Criteria:** Absolute model/executable paths and ownership are known; active original service remains protected; each C-10 normal/recovery oracle has a supported operation and truthful scope.
**Tests:** Read-only current configuration and service metadata; scoped caller/test/source inventory. No source-only claim of native acceptance.
**Status:** Complete

- [x] Read exact playbook/catalog oracles and locate shared llama.cpp Admin panels/domain methods/config/supervisor/inventory owners.
- [x] Read current Admin config/profiles/instances without model inventory scanning: saved enabled=false, handler_configured=false, no registered models/folders; profiles/instances return503 with setup guidance.
- [x] Read existing service model IDs:8010/v1/models returns200 with unsloth/Qwen3.8-27B-GGUF:Q8_0. /props returned404; this is an unsupported endpoint, not a provider outage. Installed executable is /opt/homebrew/bin/llama-server.
- [x] Declare the existing GGUF file, intended run-owned instance and memory/context/port budget before any model start/stop/configuration mutation. The human supplied the existing Splash offline command; no question remains pending.
- [x] Complete affected source/caller/setup/permission and existing-test assessment. Build only if a causally proven repair requires it, using fresh applicable authorization.

## Stage2: Causal repairs and supported validation
**Goal:** Fix only demonstrated C10 contract defects at their shared owner.
**Success Criteria:** Connected tests reproduce each defect on unchanged source, minimal correction passes affected tests with no new security findings; independent source review clear.
**Tests:** Existing UI/WebUI llama.cpp Admin controls, actual config/supervisor/provider tests, official restricted PostgreSQL when persistence is affected, touched lint/compile/Bandit.
**Status:** Complete

- [x] Reproduce actual new failures before production edits; preserve existing assertions, auth, ownership, resource budgets and configured-local egress boundaries.
- [x] Review the complete changed functions/callers/tests, scoped security and configured clients before runtime deployment. No new agent; reuse the authorized read-only reviewer.

## Stage3: Real owned runtime and recovery
**Goal:** Normal Chat uses the selected managed instance; setting/readiness/lifecycle states match actual behavior.
**Success Criteria:** Declared installed instance starts under its owner, real inference/observable setting/reload pass, owned stop and legitimate recovery preserve failed turn identity; applicable missing/resource/config/start failures give truthful guidance.
**Tests:** One existing tab, normal principals and real provider, scoped SQLite/PostgreSQL auth composition, bounded process and source guards.
**Status:** In Progress

- [ ] Inspect meaningful installed/available/loading/empty states without arbitrary model/cache scans.
- [ ] Start only the preapproved run-owned model; require real readiness and one normal Chat response using the exact selected runtime.
- [ ] Change one supported runtime setting; save/reload and verify its observable process effect.
- [ ] Stop/restart only the run-owned instance; verify unavailable Chat, retained failed input/turn and legitimate successful recovery.
- [ ] Qualify missing model, invalid config, failed start and bounded memory/storage admission without exhausting the Mac or inventing native passes from mocks. State any service/component-only fault coverage.
- [ ] Capture one settled non-secret UX screenshot for human acceptance; restore exact original Chat/draft/records and release/remove completed disposable outputs.

## Stage4: Review, cleanup and selected closure
**Goal:** Accurate selected acceptance without widening release claims.
**Success Criteria:** Included oracles pass or have explicit human disposition; UX accepted; independent actual outcome/handback/closure review clear; completed bulk outputs removed.
**Tests:** Current ownership/release/canonical handback, task format, source security and outcome review. No unchanged broad control replay.
**Status:** Not Started

- [ ] Record failed attempts and actual source/runtime/mode limits; keep overall task In Progress and broader release/current-dev/full-type/native/installer/publication gates open.
- [ ] Update master selected count only after review; remove ONLY this owned completed plan; then open C11 sequentially.


## Declared installed runtime — 2026-10-20

The human supplied `splash serve --model unsloth/Qwen3.8-27B-GGUF:Q8_0 --port8010 --max-memory60G --max-context130K --host0.0.0.0 --allowed-host10.251.75.124 --idle-releaseoff --max-cache-disk0 --offline` (original spaced flags retained in task conversation). This answers the one model declaration question. Preserve that existing service and its network settings.

- Installed Splash1.3.2 uses its own engine. Installed `/opt/homebrew/bin/llama-server` is0.5.0/build11146/commit7fe450e19. These are different runtime implementations; no compatibility or native inference is inferred.
- Exact declared Splash selection metadata identifies one27.052GiB GGUF, `target/Qwen3.8-27B-Q8_0.gguf`, linking to its installed content-addressed HFblob. Only small model.json metadata, symlink destination and file stat were read; no model-content read/hash/copy or broad/cache scan.
- Mac memory128GiB, available38.72GiB at initial read. Prepare a small-context CPU-only run-owned instance rather than duplicate the60G/130K service. Recheck resource/ownership/ports before any launch; startup120s and normal inference180s remain.
- The existing application canonical-filename policy expects `.gguf`; canonical HFblob names have no suffix. Assess a normal run-owned `.gguf` hard link to the declared existing model as minimal fixture preparation, without copying bytes or weakening validation. Remove only that new link after release. No format-support feature or production repair is justified by metadata alone. Independent read-only assessment pending.
- No lifecycle request, model startup, runtime/config mutation, new browser or C10 native attempt yet. All earlier caps remain; C09 closure stays accepted.


### Model preparation refinement before execution
- Reject the earlier hard-link idea: Splash official assembly verification checks source ctime, so even link-count metadata changes could invalidate its next startup. No hard link was created.
- Prepare a private regular `.gguf` with macOS copy-on-write `clonefile`; no byte-copy fallback, downloads or hashes. This creates a separate inode sharing existing disk blocks while leaving source contents/size/mtime/ctime/link count unchanged. Fail on unsupported/cross-filesystem cloning. Verify tiny controlled success/no-overwrite tests and review before the real operation. Source and original Splash installation remain untouched; remove only the private clone after owned runtime release.
- Proposed mapped CPU-only settings: n_gpu_layers0, no_kv_offload, threads4, parallel1, ctx512 then1024, bounded batch/ubatch. Do not add unsupported no_warmup or enable unvalidated args. Recheck model-size plus10GiB available-memory reserve before model startup. No model lifecycle action yet.


### Verified preparation and startup resource boundary
- Tiny real copy-on-write/no-overwrite controls2passed; declared-selection controls2passed. Causal controls exposed the metadata-to-clone gap and partial completion-record publication (2failed; exact wrong-target negative passed). Identity binding plus atomic fail-closed publication passed4 affected controls. Private0600 publication refinement passed2 affected controls. These results compose; no unchanged full replay or native/model qualification is claimed. Ruff/format/compile clear; operator Bandit0findings0errors.
- Latest startup admission: model27.052GiB, available34.22GiB, required model plus10GiB reserve. No model launch is admitted. Do not lower that reserve to fit.
- Model-only APFS clone preparation can proceed independently: clonefile shares existing disk blocks, creates a separate inode and does not load model bytes into RAM. Startup still requires the unchanged memory/ownership/port guards. The exact source metadata must remain unchanged. No real clone/start or config mutation has occurred at this entry.
- Existing Splash listener45130 is protected; exact declared offline command and installed source are known. Its supported idle-only memory-release endpoint is an alternative to process pause, but neither action has been executed or authorized as a new exception.
- The application managed-runner normal cleanup sends SIGTERM, waits10s, then may SIGKILL its own process group. This applies to stop/restart and failed-start cleanup; resolve the retained no-force scope before native lifecycle testing. Do not change production solely to bypass the UAT constraint.


### C10 source candidates and causal repair scope
Model-only preparation completed once: separate27.052GiB APFS clone, full source stat identity unchanged, no byte-copy fallback/load/start or Splash action. Current resource/lifecycle launch gates remain.

1. Runner status currently labels a spawned child RUNNING before its readiness await resolves. Add original-suite deferred-readiness success/failure controls with real endpoint wiring admission; then use existing STARTING state until health.ready is true. Keep process ownership and cleanup intact.
2. Llamacpp Admin currently retains model/profile IDs when canonical account/server changes; domain lifecycle calls lack config snapshots. Establish causal component/transport controls, then reuse the existing keyed-authority/admin-request-scope pattern across actual affected callers. No speculative broader Admin rewrite.

ADR required:no new record. These repairs restore existing readiness and authority rules; ADR043 and the existing canonical scoped-request boundaries govern. No native acceptance from source or mocked tests.


### C10 verified readiness and pending-save test STOP
- Readiness causal unchanged2failed; minimal existing STARTING gate passed121connected checks/29warnings/7.50pytest9.273outer/natural0. Bothcompile/format clear; Ruff4identical inheritedproduction BLE001/test0. Banditproduction0/0,test70identical/new0/errors0. Independent source review CLEAR. No native/model qualification.
- Resource404 mutation recovery correction: corrected mocked-panel observations passed2 targeted checks. The first mocked error-text assertion was invalid, not accepted as causal proof. Real403 control passed separately; genuine load404 remains gated. Add exact selected mutation/profile-ID call assertions before complete connected qualification.
- Profile pending-save initial real causal failure: Cancel disabled=false. Minimal local save latch now blocks pending Cancel/open/secondSave and disables inputs including JSON. Three invocations consumed: second failed changed loading-button selector; third passed pending/single-write assertions but final dialog-removal assertions failed under Antd motion. STOP further pending-save execution. Prepared real ConfigProvider motion:false fixture, as existing Admin restore tests use, with original time limits/assertions. Review and a specific exception are required for one further focused run per client; no passing control replay.
- Authority-switch defect remains a separate unimplemented scoped repair. All C10 native/resource/no-force gates remain open.


### C10 connected repair qualification — composed, not native
- Four product repairs: observed readiness gates existing STARTING/RUNNING; keyed canonical-authority model/snapshot owners and captured cancellable transport; resource404 stays local while403/genuine load404 retain guards; pending profile save owns its dialog/inputs until settlement. The authority repair's StrictMode latch/catch/finally interaction was separately reproduced and corrected before broader qualification.
- Python121connected checks passed as recorded above. UI98cases compose72broad passes+6failed-only snapshot expectation corrections+15transport+3StrictMode+2pending controls. No single clean98case UI run is claimed. WebUI96passed2deselected14.297outer plus2approved pending controls3.075outer. The focused pending grants are consumed; original failures remain historical.
- The snapshot6failures were exact-once matcher sites missing the additional scope argument; original IDs/payloads/write counts remain. Two early transport test runs used invalid oracles and are NOT causal proof: explicit config snapshots retain captured authority, and fetch mocks must reject pre-aborted signals. Corrected15controls pass in both configurations.
- Eight touched TS files semantic0diagnostics using real configured ambient declarations and officialVitest setup. Initial reduced-root missing-ambient diagnostics were operator-only/unqualified. ESLint retained-functional baseline607/current607 identical warning-message signatures/0errors/0new; not whole-project qualification. Production/test Python scoped security as above; TS Bandit N/A. Independent readiness/scope/source/harness reviews CLEAR.
- No corrected frontend build, C10native/model lifecycle, original API cycle or Splash action yet. Model fixture remains private/separate inode/source metadata unchanged at preparation. Available-memory admission and no-force conflict remain open. One question specifically permitting the application's existing10s-then-own-group-kill model cleanup was asked once; answer pending. It excludes original Splash/API/frontend/unrelated processes.


### First corrected C10 build
Prepared one fresh corrected build under the standing all43fix/validation authorization; no previous C10build attempt exists. This does not reset any older exhausted issue or C09limit. Exact reviewed202source bindings preserve the prior200 (including accepted notification relay) and add the two changed llama.cpp panels; other changed transport files were already included. Prior9fW/200provenance, fresh dist/control,600s cap/TERM15/noinstall/noGit/reversiblebytes and compiled route preservation remain. Independent actual operator review CLEAR; operator Ruff/format/compile clear (format ASTidentical), Bandit0/0errors. Build output is separate .next-live-tier-c10-runtime-20261020; running original frontend/API/Chrome/Splash stay unchanged. The app-owned model forcefallback question remains pending; no native/model launch follows from a build.


### Corrected build and prepared configuration-only native cohort
First build completed naturally0/104.657s: nXso4xJvJyBTEPJFHK4xo, compiler/token-sync/bundle-budget0,202 temporary bindings restored. Existing type skipping remains disclosed; touched-file semantic checks are separate. No runtime transition has executed.

Fresh C10 exact-original transition/preservation/login operators are independently reviewed CLEAR. They retain full canonical saved-record/Chat/draft/actor/receipt/owner/notification-route guards,120s startup and15s graceful/no-force shutdown. Separate configuration-only host/browser operators use normal AuthNZ setup, official restrictedPG and real Admin forms: inventory, create512, edit1024, reload, no unready Use-in-Chat. No model start/stop/download is present. Eight closed-write-scope negative controls, syntax/compile/Ruff and operator Bandit0errors/findings pass; full host/browser source review is pending before execution.

Reviewer confirms shared lifecycle composition is source-supported: profile JSON and supervisor/runner ownership have no DB-specific start/stop/Use-in-Chat branch. Verify actual changed argv after owned stop/start; changing a running profile alone does not change its process. Other modes Chat persistence/recovery remains composed coverage, not newly established by profile-only checks. Model cleanup exception remains unanswered; no engine/Splash lifecycle action is implied.


### C10 live transition, setup failures and restored handback
- Full fresh preservation passed with zero protected mutations. OriginalAPI52868/52843 stopped gracefully; frontend50832/50736 replaced gracefully by15731/15715, buildnXso4xJvJyBTEPJFHK4xo ready0.764s.
- First SQLite-Multi setup1failed1warning/1.59pytest2.453outer, before any API/browser/model. Actual supported password-policy diagnostic confirms rejected/missing-special: operator incorrectly used URL token as a password. Prepared actual policy generator and normal-dotenv round-trip/quoted-form parsing.
- Declared setup recovery1failed1warning/0.63pytest1.609outer, again before any API/browser/model. The generator constructed multi-user Settings before AuthNZ initialization created JWT keys; exact source and absent-key/DB metadata support this cause. Corrected sequence: normal noninteractive AuthNZ initialize (skips Admin), then policy generator, explicit canonical Admin creation, Database setup. Two isolated operator sequence checks pass; they are not application/native validation.
- Initial and ONE recovery are consumed. No further native profile cohort yet. Fresh reviewed apiC10Profiles cycle is being prepared around current21278/21262 and unchanged202-binding frontend; additional SQLite attempt and original API cycle need a specific exception. Three other modes remain unattempted.
- OriginalAPI restored21278/21262, ready8.671s/session45756. Full actual handback completedtrue/failedfalse/originalChatRestoredtrue/fullRowsUnchanged/providerIdle, exact retained Chat/draft/C03/Studio/C04/C05/C06 and zero protected mutations. Frontend15731/15715/buildnXso4x remainsready. No C10 native acceptance/model load or inference claim.
- Human explicitly granted the existing normal C10-owned model SIGTERM10s/possible-own-group-kill cleanup exception. This is no longer pending and excludes original Splash/API/frontend/unrelated processes. No engine started, no kill occurred and memory admission remains open.
- Cleanup will remove only the released obsolete9fW Next output and two failed pre-browser fixture roots after exact owner/open-file guards, retaining concise outcomes/current active files. No Git, original data, model-source or unrelated cache action.


### Prepared next cohort / completed-file cleanup
Correct initialization order, actual generated credential plus normal-dotenv round-trip, current21278/21262 same-build cycle and order-fix admission independently reviewed CLEAR. Two isolated operator-order controls and21pure scope/completion/deadline controls pass; configured static checks/operator Bandit0findings0errors. These controls are not native acceptance. One specific question approving the additional SQLite-Multi plus three unattempted profile checks and one original-API stop/restoration was asked once; answer pending. Model-owned cleanup is already granted separately and is not pending. Do not repeat either question or infer cohort approval.

Cleanup completed after exact absent old owners, live new-build/complete saved-record handback and open-file guards: obsolete9fW Next dist plus the two failed pre-browser fixture roots removed. Tiny failure controls/current clone/current runtime retained. Free disk119.034GiB; no fixed physical-space saving claim. No original data, model source, Splash, unrelated cache or Git action. C10 remains open; master30accepted/13open and C11 unopened.


### Approved profile cohort: real saved-profile result and cleanup boundary
- Human granted the corrected SQLite-Multi additional check, three unattempted modes and one original-API21278/21262 cycle. Pre-preservation passed, original stopped gracefully once; unchanged C10 frontend15731/15715 and Chrome58150 retained. Model-owned cleanup grant is separate and already granted; no engine/model/Splash action.
- SQLite-Multi extra attempt consumed: actual normal initialization/login/inventory passed, then covered model-input click timed out before any profile write (1failed1warning/32.73pytest33.64outer). Real DOM hit testing and installed Antd source confirm the visible value/option selectors; corrected read-only dropdown check passed. No native replay or acceptance. Released API/browser; exact owned form Cancel/local Logout/blank cleanup completed afterward outside the failed case.
- SQLite-Single initial check consumed: create512 and edit1024 bothHTTP200 with exact completed request/response journals, edited JSON persisted after actual reload, defined/Start and no unready Chat admission verified. Failure afterward at15s Logout request drain (1failed1warning/28.20pytest28.919outer); API/browser released. Its one normal recovery is reserved for a no-profile-write finish on existing private data; original failed receipt remains. Add bounded method/path pending metadata before observer close; no longer repeat Logout after it was attempted. Nine new predecessor/no-write/deadline controls pass; syntax/Ruff clear/operatorBandit0findings0errors. Independent source review pending.
- The already-logged-out failed-case Settings page was retired to blank outside acceptance. Original API remains stopped for the cohort; full same-profile restore/record handback remains required. Two PostgreSQL initial mode grants remain unattempted; retained official cluster on15492 freshly read-only verified running, no SQL or container mutation. No latest-dev/publication/native acceptance or count change; master30accepted/13open/C11unopened.


### Single no-write recovery: failure attributed to notification endpoint, cause still open
The reviewed no-write recovery ran once on retained SQLite Single data:1failed1warning25.52pytest26.087outer. Exact pre-start inert store and normal reload GET identity passed; no initialization/config/profile write or model action. Strict15s Logout drain failed. Page/context/raw each retained exactly TWO GET/api/v1/notifications/stream, zero writes,50matching starts; current health200 had a cancelled terminal. All owned API/browser children released (API0.558s graceful), original failed receipts retained. Single initial plus one normal recovery are consumed; no third grant exists.

Do not infer a product cause from request count. Existing source stops streams on credentials, React cleanup and visibilityhidden; authenticated full-document navigation/reload may leave prior-document streams and is not separately pagehide-owned. Source-supported hypothesis only. Add bounded passive main-frame/Document-loader, notification-request loader/frame/whitelistedpath and terminal attribution to the next still-original-approved PostgreSQL check. Six operator fake-event guards pass; strict ledgers/drain/budgets stay. Both PG initial checks remain unattempted at this entry. No new build, product change, model/Splash action or latest-dev/acceptance claim.


### PG setup correction, committed-loader result and native STOP
- Original PG-Single attempt failed before API/browser/model (0.65pytest1.437outer): canonical restricted fixture yields PostgreSQLBackend, while the operator incorrectly passed it as DatabaseConfig. Corrected backend.config/get_pool().close_all() handoff matches C09. Normal recovery source reviewed CLEAR.
- PG-Single recovery1failed1warning34.35pytest35.007outer: real create512/edit1024/reload and completed two writes passed, then strict cleanup failed. Passive trace confirms two pending notification GETs belonged to the prior Settings and pre-reload Admin committed loaders; both current-document streams cancelled on Logout. Two further pending profile/instance reads were not document-attributed. No product leak root cause or repair is inferred. Official fixture,API39662/browser39751 released; no SQL/container mutation.
- Three cleanup failures now exhaust AGENTS three-attempt limit. No further native case runs under the consumed allowances. PG-Multi original remains unattempted, but a new cohort exception is required to validate the corrected cross-document procedure. No pending question has been sent yet at this entry.
- Reserved original restoration completed40850/40841 (session18266), canonicalready8.193s; exact original Chat/draft/C03/Studio/C04/C05/C06/full rows handback completedtrue/failedfalse/fullRowsUnchanged/protectedMutations0. Frontend15731/15715/buildnXso4x and Chrome58150 unchanged.

### Concrete reviewed-recipe implementation prepared; not executed
C10_NAVIGATION_REPAIR selects fresh private mode roots and one-use host/browser controls, with exact released failed predecessors for three modes and unattempted PG-Multi admission. Use real Header Llama.cpp Admin SPA route with same-document check. After edit, normal Logout and strict all-channel drain retire that document BEFORE a real logged-out Settings reload; normal reconnect plus fresh exact-ID/model/args GET and UI read preserve the durable-save oracle. All cumulative request ledgers remain; separate mid-retirement and final Logout state, each15s with final reserve retained. No route/mock/pending-filter/timeout weakening. Exact launcher selectors checked read-only in restored Chat and menu closed without navigation/profile/model action.

Seven new navigation/completion controls and12bounded document/terminal/privacy/two-Logout controls pass. Fresh apiC10Navigation exact40850/40841 cycle/preservation operators compile/self-check, are UNEXECUTED and require the new grant. Source review pending. Ruff/format/syntax clear; operator Python Bandit0findings0errors. No frontend build, product-source/model/Splash/Git action; all C10 native/runtime/UX acceptance remains open and master30accepted/13open/C11unopened.


### Final prepared-source review and one pending validation exception
Independent actual-source review CLEAR after two new failure-path repairs: the known retired/logged-out window now performs strict-zero cleanup/blanking without another Logout, and the15s mid-retirement budget is checked after helper/progress publication; a failed pre-action save leaves reconnect-attempt false. Ten navigation/failure-stage guards and12bounded document/terminal/privacy/two-Logout guards pass. Syntax/Ruff/format clear; touched Python operator Bandit0findings0errors. This is reviewed operator preparation, not a product repair or native acceptance.

One specific question was asked ONCE after the concrete review: approve four sequential mode validations and one guarded originalAPI40850/40841 stop/restoration, same tab/build,600s per case/120s startup/15s graceful cleanup, no model launch/build. Answer PENDING. Do not repeat/infer approval or execute the fresh apiC10Navigation cycle before an explicit answer. Earlier original cohort and named attempted recoveries are consumed. PG-Multi had no initial case attempt, but the shared cleanup three-failure STOP still applies. Model-owned cleanup exception is already granted separately and never pending.

Original API40850/40841/session18266 remains running after actualready8.193s and full saved-record/Chat/draft handback (completedtrue/failedfalse/fullRowsUnchanged/protectedMutations0). Frontend15731/15715/buildnXso4x and Chrome58150 retained. Four closed failed disposable fixture roots removed after exact absent-PID/open-file/no-symlink guards:17,925,687logicalbytes; concise controls retained, current private clone/original data preserved. No fixed physical-space claim. Master30accepted13open/C10open/C11unopened; no new model/Splash/latest-dev/Git/publication/merge acceptance.


### Corrected navigation cohort explicitly APPROVED
Human answered “Approve corrected C10 cohort.” The one four-mode sequential validation cohort and one originalAPI40850/40841 guarded stop/restoration are GRANTED; never pending/reask. This supersedes only the named pending question above, not other exhausted ceilings or model/resource gates. Same Chrome/tab/build/private data,600s per fixture/120s startup/15s graceful cleanup; no model launch/build/Splash action. Fresh full pre-preservation passed with no protected mutation/inference; the reserved exact-owner stop is executing. Fresh per-mode one-use controls remain mandatory; restore full original Chat/draft/records after the cohort.


### Corrected four-mode configuration cohort completed; original session restored
The explicit grant above is fully CONSUMED. All four sequential cases passed naturally: SQLite Single1passed1warning/16.10pytest16.953outer; SQLite Multi1passed1warning/19.51pytest20.193outer; PostgreSQL Single1passed1warning/22.26pytest22.966outer; PostgreSQL Multi1passed1warning/23.38pytest24.010outer. No skips reported. Each completed exactly two profile writes (create512/edit1024), a real logged-out reload/reconnection, fresh exact saved identity/args read, two normal Logouts with strict zero-pending retirement, and released API/browser children. Official restricted PostgreSQL fixtures cleaned their databases. Model starts/stops0. Independent outcome review CLEAR for configuration-only qualification; do not infer native model/inference acceptance or replay these controls.

The one reserved original API cycle completed: restored46071/monitor46060, canonicalready6.901s; frontend15731/15715/buildnXso4xJvJyBTEPJFHK4xo and Chrome58150 unchanged. Complete handback verifies original five-message Chat/draft, C03/Studio/C04/C05/C06/full saved rows unchanged, provider idle, Studio queue empty and protected mutations0. Completed disposable mode roots may be removed after absent-owner/open-file/no-symlink guards; retain only the SQLite Single root as the active owned-engine fixture, concise controls and the model clone.

Owned model startup remains NOT ADMITTED: fresh35.693GiB available versus27.052GiB model plus10GiB reserve (37.052GiB required). No model has started; do not lower the reserve or repoll unchanged. A concrete one-use idle-only Splash memory-release operator is prepared, not executed; separate human consent and source review are required because8010 is protected.409/503/transport timeout leaves the operation unconfirmed and permits no retry.200/released is only a snapshot: another request can restore weights. Fresh model+10GiB admission remains required. Model-owned normal cleanup exception is already granted and never pending.

C10 stays OPEN; master30accepted/13open and C11 unopened. Frozen candidate/unpublished WIP, ADR030/043/059 and all latest-dev/full-type/native/publication/merge limits remain. No new durable architecture decision.


### Existing-service decision and completed-output cleanup
The human chose “I’ll free RAM instead” in response to the one idle-only release proposal. No release approval is pending or granted; no Splash HTTP mutation, process stop/signal or memory release occurred. The unexecuted proposal helper and its own bytecode were removed; no one-use release control exists. Its source review had cleared the shared-deadline/fixed-buffer corrections (four changed-only operator checks, compile/Ruff/format and Bandit0findings0errors), not a native service action. Wait for the new freed-RAM state before another admission check; do not lower the37.052GiB model-plus-reserve threshold or repoll unchanged.

Completed SQLite Multi/PG Single/PG Multi roots removed after exact release/open-file/no-symlink guards:9041810/674233/926383 logical bytes. Current SQLite Single engine fixture, model clone, concise controls and restored original session remain. C10 owned-model/readiness/inference/lifecycle/recovery/UX are still open; no C11 transition or selected-count change.


### First owned-engine workflow ready for execution
Human explicitly requested live startup/inference/recovery, fixes and C10 closure before C11. After the subsequent “Another1GiB freed” confirmation, fresh65.686GiB available passes37.052GiB admission. Ruling: this new live-phase instruction covers the necessary one exact original46071/46060 API stop/restoration into the retained private SQLite Single fixture; it does not reset consumed profile-case grants or permit Splash mutation. Same frontend15731/15715/buildnXso4x and Chrome58150/58135/tab;600s host with45s reserved end cleanup,120s startup/180s inference/15s graceful requests/cleanup. Fresh memory admission is mandatory before each owned Start.

New one-use engine transition/preservation/host and interactive UI observations independently reviewed CLEAR after correction of exact release/bindings, publication-failure cleanup, deadline and actual error-body seams. Seven release/live-lease/budget/build controls,13reserve/native-contract,4mutation+9completion controls, two never-claimed/partial-lease controls pass. Shared passive observer changed to bounded cached error-body reads; all eight affected streamed/completed/base64/incomplete/aborted/error-body controls pass, no native/provider calls. Python compile/Ruff/format and Bandit0findings0errors; JS syntax clear. Correct canonical provider layers are response llama and normal Chat wire llama.cpp. This is SOURCE readiness only. No model or original transition has executed at this entry.

Live observations require real engine model ID/18908 mapping, completed answer4 and canonical persistence, a real unavailable Send retaining its user ID/input, then same-input Retry without another user row. Child argv must match actual saved1024 then512 contexts, composing with the already-qualified saved1024/reload result. All original records/Chat/draft must be handed back; native model/recovery/UX and selected closure remain open.


### Actual first Start failure and causal POSIX spawn repair
First real UI Start returned400: `unexpected kwargs: process_group` under ASGI uvloop. No model process spawned. Browser Logout/drain/detach completed; private API82489 gracefully released1.216s; original46071/46060 stays stopped within the same approved cycle. Initial failure receipts remain intact.

Real subprocess regression: baseline asyncio passes, uvloop fails with the exact exception. POSIX `start_new_session=True` preserves child PID==PGID without uvloop's unsupported keyword; Windows/cancellation/group-stop ownership unchanged. Full affected runner/supervisor/runtime API suites108passed30warnings13.14pytest15.427outer, no skips reported. Two files compile/format clear; Ruff four inherited production BLE001/test0. Bandit immediate pre-spawn WIP/current production0/0, tests70/69, no new findings/errors0. Comparator's first metadata-only fallback error corrected without test/runtime replay. Narrow changed-assertion nosec only. Independent source review CLEAR; no new ADR beyond021/030/043/059.

One normal changed-source native recovery is available after the initial product failure. Separate recovery host/browser receipts are exclusive; old failed leases must be released, and both cases gate original restoration. Same retained SQLite Single fixture/profile1024, exact build/tab,600s host/120s startup/180s generation/15s cleanup; no extra original API cycle, model download/build or Splash action. Fifteen new pure admission/release guards pass, JS syntax clear. Recovery seam review precedes execution. C10 remains OPEN;30accepted13open/C11unopened.
