# X02 account and browser isolation implementation plan

> Execute inline using the executing-plans workflow; reuse the authorized read-only reviewer.

**Goal:** Finish selected X02 before X03, proving reciprocal ownership and browser-state isolation with working owner controls.
**Architecture:** Use the qualified current3817-file installed API and a reviewed private frontend on18908/18989, alongside the protected original runtime. The GET repair requires its own build before native qualification. Reuse official PostgreSQL isolation and existing lifecycle/receipt/session restoration helpers. Use supported real API/UI paths; mocks do not qualify live isolation.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md X-02 and apps/tldw-frontend/scripts/live-tier-uat/release-catalog-data.mjs X-02.
**Task:** TASK-13260.278.18.83.45.
**ADR assessment:** No new architecture rule for validation; existing AuthNZ/tenant ownership, PostgreSQL RLS and documented revocation contracts govern. Any demonstrated repair gets its own assessment and causal red/green before product edits.

## Global constraints

- Rebased sourceff08bf on fetched dev17141; fresh private API3817-file package qualified, frontend rebuild pending. Prior private API179ab7db8e and frontend2e746c40a4/buildDvZ0FkqSyO65jyxcrcyNX qualified the recorded dev22c9 RAG/cache/usable Chat isolation with the explicit saved-history limits below. Human X02 UX is accepted; current functional browser/drain/in-flight and remaining PostgreSQL acceptance stays open. Refresh dev before later publication, not after every step.
- Preserve original API50010/49993, frontend50863/50855, Chrome54716/54710, five saved messages, exact draft/URL and full canonical objects. No original runtime cycle.
- Keep the original default Chrome context/tab untouched. X02 uses one owned temporary main context; create a second owned context only for the independent concurrent-account check, then close both owned contexts. Two tabs within the main context share storage and are not independent identities. The owned main context also bounds its disposable network pool.
-600s per workflow/run/build,120s startup/child,180s inference,300s pytest case,15s operation/graceful cleanup; no force or assertion weakening. Reassess after three failures; latest human authorization covers necessary diagnostics/retries without another approval question.
- No stored secrets in output, model/cache/proof recovery, raw operator SQL or homemade PostgreSQL setup. Actual official managed fixture provisioning only.
- Keep failed outcomes failed. Dispose completed private payloads/operators promptly, retaining current artifacts and original restoration dependencies. Heartbeat remains paused.

## Stage1: Prepare current fixtures and owner controls
**Goal:** Fresh SQLite and official PostgreSQL multi-user profiles with ordinary Alice/Bob and Admin.
**Success Criteria:** Both ordinary users can create and read their source, Note, Chat, card, draft/handoff and queued Job. Additional owned mutations, exports and completed Job results require Stage2/3 qualification. Same display names use distinct owner-scoped identities and private canary contents; numeric IDs may repeat across per-user stores. Admin is used only for setup/authorized operational controls. All stores and role bindings are explicit.
**Tests:** Qualified environment/source/package/build/owner/listener guards; normal initializer and real authenticated API creation. Any blocked resource blocks that isolation component.
**Status:** Complete

- [x] Read current endpoint/request/denial contracts and reuse existing profile/lifecycle helpers.
- [x] Add minimal disposable BASE/x02_host.py and x02_api.py; guard their finite profile, process, role and deadline boundaries with small operator controls.
- [x] Review before starting either private profile; record owned positives and resource IDs only in functional memory/private fixture.

## Stage2: Reciprocal API, retrieval and revocation
**Goal:** All ordinary-user directions reject foreign work while own work remains usable.
**Success Criteria:** Valid foreign IDs disclose no body/title/snippet/count/export/asset/job progress. Bob retrieves his private phrase; Alice retrieval candidates/citations/answer do not contain it without providing it. A real declared share/role grants only its scope; revocation follows the actual session/cache contract. Admin metadata never substitutes for ordinary controls.
**Tests:** Both backend cells; own read/write before each endpoint family, documented403/404 denials, export body/asset exclusions, Job ownership, same-name IDs, actual supported retrieval and representative real Chat.
**Status:** In Progress

- [ ] Run reviewed real clients; stop at a demonstrated failure and retain its exact contract/outcome.
- [ ] For a product defect, inspect every caller, add a normal causal regression, make the smallest shared repair, run affected checks plus compile/lint/scoped baseline-current Bandit, then independent review.
- [ ] Revalidate changed live behavior without replaying passing unrelated controls.

## Stage3: Browser switch and recovery
**Goal:** Account changes retire prior private UI state and in-flight work.
**Success Criteria:** Alice logout -> Bob login -> reload -> Back clears prior lists/drafts/handoffs/helpers/autocomplete/notifications. Switching during fetch/generation and an old tab cannot publish old-account results. Independent context remains isolated. Owned/foreign deep links, exports and background jobs retain positive/negative controls.
**Tests:** BASE/x02_browser.cjs with real login/logout/reload/Back and passive actual request receipts; current full canonical handback and zero-work graceful release. Screenshot for human UX review while remaining checks continue.
**Status:** In Progress

- [ ] Review browser driver and finite authority/write admission before action.
- [ ] Run sequential SQLite/PostgreSQL browser cells in an owned main context; use a separate context for the concurrent-account check and close every owned context.
- [ ] Capture one settled screenshot; record functional and human UX results separately.

## Stage4: Review, dispose and close
**Goal:** Selected X02 accepted before X03.
**Success Criteria:** Ten catalog variants have explicit outcomes, original session and owners preserved, current task/master agree and independent actual/final review is clear. Completed private data/operators removed after release/ownership/open-handle guards.
**Tests:** Task normalization, scoped syntax/lint/security/diff checks and normal hooks/commits when product files change; no unchanged suite replay.
**Status:** Not Started

- [ ] Reconcile both multi-user cells and all recovery dispositions; single-user reciprocal cells remain N/A.
- [ ] Record concise results, retain failures/limitations, remove only this completed own plan, then activate X03.


## Confirmed handoff storage boundary defect

Before native startup, root and reviewer found that the default upload handoff
store uses the shared installation rather than configured USER_DB_BASE_DIR,
while ownership is numeric user ID alone. Separate deployments can reuse IDs;
known foreign draft IDs and profile-wide quotas therefore cross their boundary.
The cached factory also retains the first configured store.

ADR required:no; ADR020 governs configured database paths. Add normal real SQLite
get/delete regressions across two profile roots and a same-profile worker/quota
control before modifying production. The minimal repair uses DatabasePaths and
a factory cache keyed by its resolved file. Explicit paths, TTL, quotas and
cross-worker sharing remain; never import or fall back to the unscoped old DB.
Native API/browser work remains unexecuted until causal and affected checks,
security, review and refreshed packaging are qualified.

### Handoff repair verification

- Causal unchanged source:2 failed/1 passed; the known foreign draft was readable
  and deletable with the reused numeric actor. Corrected complete store and upload
  processing modules:18 passed in2.43s pytest/4.009s outer.
- Compile2passed. Configured Ruff:production0; tests retain exactly I001/BLE001.
  Black:production inherited flag unchanged, tests0. Bandit:production0, tests11
  identical inherited findings, no new findings/errors. Three new pytest asserts
  have narrow explained B101 annotations; import ordering/comments preserve behavior.
- Independent actual source, full added tests and affected callers: CLEAR.
- Pending: final staged review/normal commit and fresh3817-file private packaging
  before the first live fixture. Frontend unchanged; no rebuild is needed.

### SQLite API progress

- Repaired package73a9a80 qualified3817files/337dependency nodes/722edges.
- Both ordinary users created/read six resource families; ten reciprocal
  Note/Chat/card/draft/Job denials and two caller-owned numeric Media controls
  passed. That first run failed only the inherited browser-stream precondition;
  actual zero-work graceful retirement and three current-session logouts passed.
- API-only monitor correction retains exact fresh dispatch timestamp/request,
  response200/final body/return/task/zero work and deadline fences. Nine rejection
  controls and seven causal final-readiness controls pass; independent review CLEAR.
- Second extension run failed an unsupported Media-list full-content expectation.
  Read-only diagnostic proves Alice zero IDs/count and Bob owned ID1/count1, with
  separate rich-content positives and no foreign canary. Corrected metadata
  oracle has seven positive/wrong-ID/count/missing/duplicate controls.
- New scope-only continuation passes real FTS RAG candidates, ordinary/Admin
  boundaries and organization grant/read/forbidden rename/revoke/immediate
  same-session denial. Independent preparation review CLEAR; actual API release
  2.176s, EOF/group gone/lifespan zero, all three sessions closed, cleanup empty.
- No original runtime cycle or browser/model action. PostgreSQL, UI deep links,
  same-browser/independent contexts, inflight/old-tab recovery, generated answer,
  completed background result, full export accounting and human UX remain open.

### PostgreSQL enqueue failure investigation

The first normal official PostgreSQL fixture run failed at ordinary-user Job
submission (HTTP207), before Bob setup. Actual private API/session cleanup and
original-owner preservation passed; the failed fixture was retired normally.
The schema-owning fixture role is NOSUPERUSER/NOBYPASSRLS, not DML-only serving.

Source hypothesis: six ingest Jobs routes acquire a cached manager without
binding the authenticated principal to its PostgreSQL RLS context. Its default
owner GUC is empty; FORCE policies deny unscoped writes and reads. Prove this
with normal real official-PG causal regressions before changing production.
Assess all six callers; preserve the manager factory/overrides, media_ingest
domain, claim-based Admin and explicit owner checks, and restore context on
normal/error/cancellation exit. ADR required:no; ADR003/018 govern unchanged
Jobs and claim-first request authority. Do not accept207 or relax RLS.


### PostgreSQL request-scope causal repair

- Normal official FORCE-policy causal baseline:3failed3passed/4.12s pytest,
  6.179s outer. Alice/Bob owned reads and claimed Admin returned404; reciprocal
  and convenience-admin-hint denials passed.
- All six media-ingest Jobs endpoints now use an async yielding manager dependency
  that binds the existing claims, media_ingest domain and actor. The cached pure
  factory/overrides and explicit owner checks remain. A ContextVar token scope
  restores the exact parent through nesting, concurrency, error and cancellation;
  streaming producers retain the actor until response completion.
- Initial nine checks pass. Eight affected modules:45passed287warnings/28.41s
  pytest33.63s outer, including actual PG producer isolation and the existing
  submit/list/cancel/event and fail-closed controls. Four sources compile.
  Production Bandit1identical inherited finding; new tests0/new findings0/errors0.
- Independent actual-source review CLEAR, including full new tests, six callers
  and installed FastAPI streaming lifetime. Configured Ruff retains exactly two
  inherited diagnostics; baseline/current Black flags remain inherited. Final
  staged review, normal hooks/commit, fresh private packaging and full native
  PostgreSQL validation remain required. Product unit
  qualification does not accept X02. Original runtime owners remain unchanged.


### Installed PostgreSQL qualification

Local22141c4223 matches the independently reviewed six-path patch; normal hooks
1.854s/commit1.172s, unpublished. Fresh jobs-scope site qualifies3817files,
3roots/337dependencyNodes/722edges in7.843s; installed production bytes match.

The reviewed normal official-PG fixture passes1test/6warnings41.97s pytest,
43.884s outer. Each actor creates/reads six resources. Twelve reciprocal404s,
eight owned-content exports, two foreign Note-update404s with victim intact,
Media search, FTS candidate privacy, ordinary/Admin metadata and organization
grant/read/forbidden rename/revoke/immediate same-session denial all pass.
Both native phases close all three current sessions and release the private API
with actual zero-work graceful receipts/cleanupempty; original owners unchanged.
This qualifies the earlier207 enqueue repair on the installed API. It does not
qualify browser/inflight/old-tab recovery, generated answers, completed background
Jobs or human UX. Stage2 remains partial and X03 remains untouched.

### Browser continuation scope

Keep the protected original tab entirely untouched. Use one temporary working
tab in its existing Chrome context at the separate private origin, with one
temporary old tab for shared-storage recovery. One temporary independent context
and page provide concurrent ordinary-user positive/negative controls. Same-context Alice logout/Bob login/reload/Back is a separate
control. Close every temporary page/context and disconnect only the acquired CDP
handle after current-session logouts and actual network/server retirement. The
latest human approval covers this necessary isolation scope; no original runtime
cycle, browser relaunch or permission question is needed. Reuse the qualified
TH build and current Jobs-scope API package; no frontend build is needed.


### Browser preparation and completed-payload disposal

BASE/x02_browser.cjs and x02_ui.py prepare normal UI-only authentication, owned
and foreign Note links, unsent Chat draft, logout/login/reload/Back, old-tab and
independent-context controls. No Send or model call is part of this cohort.
The working/old tabs share verified auth epochs; the independent context does not.
Actual JWT subject/session receipts register only current fixture sessions in a
private functional cleanup control; fallback cleanup never changes the core
Admin client or uses all-device logout. Original full canonical reads and exact
URL/draft plus process owners are guarded independently. The existing complete
server-stream guard remains, with a fresh final-ready callback fence.
Nine Node boundary controls, ten denial-cohort controls and four final-readiness
controls pass; compile2/Ruff0/Bandit0. Independent preparation review remains
pending; no browser, frontend, model or original runtime action has run.

After the successful native PG case released its official database, its profile
and the superseded X02 handoff package were removed:4251entries/81,380,979bytes.
Exact root/inode/UID/no-link/open-file and protected-owner checks passed. Current
Jobs package, unchanged frontend build and original restoration dependencies stay.


### First SQLite browser outcome and navigation diagnosis

The first bounded browser cohort failed56.715s outer. It completed Alice owned
and foreign Note controls, actual Alice logout/Bob login, old-tab private-state
clearing and Bob owned Note read. It did not complete reload/Back: its fixed
step was too coarse to distinguish those next operations. Node reported a
Logout UI timeout and one observer-drain failure; these remain failed outcomes.
All verified UI sessions nevertheless closed, all temporary handles/CDP released;
parent cleanupempty and original full canonical/URL/draft/owners preserved.
Private API actual zero-work/lifespan/graceful release and three core logouts pass.

Prepare one distinct diagnostic run on the same artifacts/profile and limits,
with explicit reload/Back steps, capped own-source line/column and pending
method/path metadata only. No raw error text, queries, credentials or body data.
Recreating the causal navigation history is necessary; no Send/model or original
runtime cycle. Review before execution. X02 and Stage3 remain open.


### Second navigation diagnostic and real-focus correction

Second run failed54.301s outer at the first Back, already at/settings/tldw;
Bob-owned Note reload completed. Eleven pending requests included four open
notification200 streams plus Auth me and Notes task-activity reads. No private
body leak or application cause is established. Actual private retirement, verified
UI/core session closure and original full canonical/URL/draft/owners pass again.

Installed Playwright1.58 crPage.js412 enables focus emulation for every new page.
The qualified existing attach helper disables it only on the original page; new
X02 pages omitted that real-focus guard. Third diagnostic will disable emulation
on every owned page with an immediately owned/detached CDP session, and bring the
actual target tab forward. Keep the Back DOMContentLoaded oracle unchanged;
record only path/readyState/visibility/hasFocus on failure. No product edit/build,
model or original cycle. Review first; do not call this an application repair.


### Third browser diagnostic and reassessment

Real-focus correction did not resolve Back: failed69.389s outer at
back-0-to-prior-chat, already/settings/tldw, readyState complete/visible/focused.
Bob-owned Note reload completed. Same eleven pending requests/four open200
notification streams; no private body leak or application cause established.
Node Logout/observer-drain failures remain failed outcomes. All UI/core sessions
closed, temporary handles released, private API zero-work/lifespan/graceful release,
parent cleanupempty and original full canonical/URL/draft/owners preserved.

Reassess before further runtime: distinguish a Back lifecycle wait mismatch
(BFCache/same-document) from real request retirement failure. Compare existing
navigation patterns and inspect request/document lifecycle metadata. Any next
diagnostic must be distinct, bounded, reviewed and retain strict request drain
and data/privacy assertions. Latest human authority covers necessary diagnostics;
no repeated permission question. Stage2/3 partial; X02 remains open.


### Passive navigation/request diagnostic preparation

Compare three mechanisms before changing behavior: new-document DOMContentLoaded,
same-document history, and BFCache restoration. Retained Chrome omits Playwright's
normal disable-back-forward-cache argument; installed Playwright clears lifecycle
state while discarding frameNavigated.type. These are source hypotheses only.

X02-only driver adds a capped passive main-frame commit/type/loader/history/path
and lifecycle journal; Back's DOMContentLoaded wait is unchanged. Track actual
terminal/status/cancellation for private-origin requests pending at Back separately.
Failure DOM metadata contains only readyState/visibility/focus, Settings/Note
visibility and dialog count; never accept/dismiss a dialog or read private bodies.
Parent projects existing ASGI notification start/task-completion receipts before
cleanup to distinguish browser ledger from actual server work. No app/protocol
change, request injection, original cycle, model or build. Syntax/compile/Ruff0;
eight navigation and seven passive-wire controls pass. Review before one bounded
distinct diagnostic; strict final request/write/privacy/lifecycle gates stay.


### Proven Back wait mismatch and corrected driver

Distinct passive diagnostic failed70.625s outer but proved actual
BackForwardCacheRestore to exact prior history index6/entry454, old loader,
Settings visible/complete and no dialog; no subsequent DOMContentLoaded event.
This is an operator wait mismatch, not an application navigation defect.
Several requests from cached loaders have no browser terminal; one older Auth me
finished after restoration. ASGI notification receipts show8starts/8task_done
only after Node cleanup, not at Back failure. Therefore no early retirement
acceptance follows. Node retains Logout/drain failures and a new CDP-detach race;
parent cleanupempty, sessionsclosed, actual API release and original full
canonical/URL/draft/owners preserved.

Correct Back to wait for commit, then require the exact prior private history
entry/index/id/path, readyState complete, route-specific Settings/Chat control
and absence of Alice text. Seven refusal/positive controls pass. Retire only
owned CDP sessions; ignore detach failure only after their own page is actually
closed. All request/write/session/privacy/ASGI drain gates remain unchanged.
Review before validation. Any remaining cached-document transport issue remains
a separate diagnosis; no product change or browser cache disabling.


### Corrected Back live progress and cleanup root cause

Corrected commit/history/DOM guard passes real reload/Back, both foreign Note
directions, unsent Chat draft clearing, old-tab clearing and concurrent independent
Alice/Bob contexts (14 functional checks). Cohort fails74.600s outer at
independent Alice's Open Settings before Logout; one dialog is visible. Source
shows foreign denial is inline, so the dialog's cause is not established.
Wait for the exact settled Linked note unavailable region before the next action;
never auto-dismiss an unknown dialog.

Parent session cleanup failed because httpx.Headers.update rejects keyword
Authorization; the existing helper used that unsupported call. Real-Headers
causal refusal plus six valid/closed/foreign/type/unauthorized/wrong-owner controls
pass after the minimal mapping fix. Two exploratory test invocations exposed this
operator bug; the third corrected control run passes. No application test claim.
A proposed broader fresh-owner revocation workaround was removed before runtime;
keep exact recorded JWT subject/session, Auth me positive and current-session
Logout all_devices=false. Add an explicit cleanup-only host phase for the retained
browser control; review before actual private startup. No UI/model/build/original
cycle. Parent original full canonical/URL/draft/owners and API graceful zero-work
release pass; browser session closure remains open until this finish succeeds.


### Exact-session finish and targeted dialog continuation

Cleanup-only installed API passes13.457s: the one remaining browser session
completes verified current Logout, private control removed, three fresh core
sessions close, API actual zero/lifespan/grace/cleanupempty and original owners
preserved. No frontend/browser/model or original-runtime action.

Next denial-browser diagnostic avoids replaying14 passing controls: one temporary
working tab, Alice login, foreign Note settled unavailable region, normal Settings
navigation/Logout. Unknown dialogs are categorized by fixed public titles/button
flags only and never dismissed. Keep strict request/write/teardown fences.
JS syntax/two Python compile/Ruff/Bandit0; independent preparation review first.


### Settled denial validation and pending Auth document recovery

Targeted check50.897s proves foreign Note unavailable view, normal Settings
navigation and current Logout with no dialog. Thus no dialog dismissal or product
change is needed. Strict final observer drain still fails on exactly two Auth me
GETs without responses/terminals. Both notification ASGI tasks complete; all UI/core
sessions close, parent cleanupempty/API gracefulzero and original data/owners pass.

Next distinct diagnostic selects the earliest private Settings entry in only the
owned temporary page's real browser history after Logout. Require exact entry ID,
actual known commit/loader, Login visible, no Logout/private canaries, and retain
strict actual request drain. This tests whether resuming that old document delivers
the Auth terminals. No history reset, cache disabling, simulated terminal, dialog
action, original-runtime cycle, model or build. Review before invocation.


### Completed PostgreSQL Job authority subunit

Source review independently finds normal/heavy/CLI media worker acquisition lacks
Jobs RLS authority; per-job content scope begins only after acquisition. Add two
normal official restricted-PG regressions (stop/cancel) using actual service/SDK
and policy, two tenant Jobs and another-domain sentinel. Stub only per-job work
and deterministic empty-queue sleep. Prove durable completion, non-admin content
owners, sentinel unchanged and exact ambient restoration before product edits.
ADR003/018 govern existing trusted-worker authority; no new durable rule.
Minimal proposed repair belongs at the converged trusted service entry, never
generic SDK elevation or a policy bypass. Actual content and native completed
result remain separate required controls.


Causal unchanged-source official-PG result:2failed/11warnings2.52s pytest,
4.313s outer. Both actual worker lifecycles reach empty acquire with no handler
call; expected queued-owner work is not acquired. No test/import/role workaround.
Implement only token-restoring Jobs service authority around SDK.run. Keep
per-job content owner, explicit configured-domain acquisition, policies, SDK,
lease/finalization and stop/watcher behavior unchanged. Green/affected checks,
scoped security, review and native qualification remain required.


Worker repair green:2passed/11warnings2.67s pytest4.735s outer. Fifteen connected
actor/SSE/real SQLite+PG content-owner/heavy/startup-gate/cancellation controls
pass, two already-green new cases excluded,11warnings5.51s pytest7.539s outer.
Two sources compile. Ruff retains exactly inherited production I001/F401, tests0;
Black production whole-file flag inherited/tests0. Baseline/current Bandit0
findings/errors in both files. Independent actual two-file/caller/full-test review
CLEAR; idle cancellation is distinguished from real ingestion content and native
completed-result acceptance. Stage/normal hooks/commit, fresh installed package and
actual background completion remain pending.


Worker repair local29a1b69eed31619bf2e236b49f9016c0d7de9e22, parent22141;
reviewed four-path38573-byte patch SHA256
ae15eb6da2f502c4abdf951ec96b32706fd44b5ea8ad6f99993871d9380d9e51
committed exactly. Normal hooks1.506s/commit1.134s; unpublished. Final staged
review CLEAR after narrowing Stage1 to fixture create/read; mutation/completion
acceptance remains later-stage. Changed function Black0; whole-file flags inherited.
Fresh worker-scope package3817files/3roots/337dependencyNodes/722edges qualifies
9.632s cleanupempty, source/environment/originalowners preserved; installed worker
bytes equal committed source. Host pins new site, unchanged TH frontend.
No new application runtime/native completed Job acceptance yet.


### Native completed-Job preparation

Current3817-file worker package matches committed29a1. The normal separately owned
media service uses the private profile and public stop_event, never other pollers
or original-runtime cycles. Two fixed dispatch/service-return receipts are not
startup readiness. Actual two-owner completed results, content canaries and reciprocal
Job denials must pass within120s; worker signal/return/exit/EOF uses15s graceful
cleanup.21 pure operator controls cover schema/results, output failure, identity
refusal without signals, collector setup and latched interruption. Compile/Ruff/
Bandit0. Final source review precedes native execution. Official PG entry adds
only prepare-jobs selection; prior complete API controls are not replayed.

The distinct browser diagnostic changes metadata only: exact history before/target/
after and fine selection/entry/commit/UI failure steps. The strict history and
request-drain assertions stay unchanged. Last83.628s run had zero pending requests
in all three browser ledgers but did not prove its history recovery; do not close
that gate or infer the frontend cleanup cause from the coarse RuntimeError.


### Actual completed-Job tenant acceptance on29a1

Final worker/operator/official-entry source review CLEAR. SQLite real background
completion passes17.261s outer/worker2.847s, both durable Success results and owned
content, reciprocal403. Worker returns through public stop_event, exit0/grace0.271s;
API graceful2.230s/actual lifespan zero, all current sessions closed and original
owners preserved. Official restricted PostgreSQL1passed6warnings25.67s pytest/
27.425s outer, worker3.619s, both content positives and reciprocal404, worker returned/
API released/shutdowncomplete/cleanupempty/currentlogoutsclosed/ownerspreserved.
No fake in-scope work, application mocks/source patches, model calls, policy changes, builds
or original runtime cycles. Existing sidecar mode stayed enabled; only media
sidecar was launched. The completed disposable PG profile was removed using exact
root/UID/no-link/closed-session/no-open-handle guards after official DB teardown.

One read-only Docker preflight stopped before pytest because existing port15492
is wildcard-bound, rather than loopback-bound. The corrected admission verifies
that retained mapping and still connects exclusively through127.0.0.1; no Docker
configuration or service change. No native failure or extra source qualification.

These checks close the selected completed-Job result component only. Browser strict
request retirement/history recovery, PostgreSQL browser, retrieval answers, switch
during fetch/generation, further UI state/export checks and UX remain open.


### Exact history passed; request retirement remains open

Metadata diagnostic83.789s passes settled foreign Note denial/current Logout and
actual known BFCache Settings history entry523 atindex1, Loginvisible/Logoutabsent/
canary-free. Strict final drain fails one GET auth/me36347.146 without status or
terminal in all3browser ledgers/cohorts. Notifications2start2done afterNodecleanup.
Frontend release30.630s errorsnone/exit0/groupEOF fails15s; API2.251s/lifespanallzero,
all current sessions closed and full original canonical/URL/draft/owners preserved.
Neither a30s configured timeout nor an application cause is established. Independent
source review finds shared transport aborts on actual account changes, but not on
document pagehide; Authme's10s timer may freeze in BFCache. This is a coverage gap,
not yet attribution of that pending ID. Next diagnostic adds only exact failure-time
readiness callback timestamps and pending request loader metadata, then retains
strict failure, cleanup and original handback. No product edits or timeout increase.


### Browser passed; own the complete disposable network context

Distinct diagnostic82.472s: all targeted browser checks PASS, including exact history,
Login/privacy and strict actual3-ledger drain. All owned handles/sessions close and
original tab unchanged. Failure-time probe is correctly unused on a passing browser.
Private frontend beforeTERM has1acceptedESTABLISHED/noAPIoutbound pluslistener; its
release60.120s/exit0/groupEOF exceeds15. API2.165s actualzero/grace/logouts and full
original canonical/owners pass. No default30s shutdown/proxy rule is inferred.

Normal isolated Node20.19.5 HTTP/TCP control0.104s confirms a no-request preconnect
can hold server.close, and normal client.end releases it. App attribution remains
limited: actual browser drain passed, the remaining failure is cleanup. Use the
existing qualified acquireOwned mechanism for a separately owned MAIN browser
context rather than borrowing default-context connection pools. Its page/history/
login/logout and strict request fences are unchanged; normal context.close owns
the disposable pool. The retained original default context is preserved. Separate
concurrency context remains genuinely independent. No product, timeout, cache,
model, build or original-runtime change. Review and actual lifecycle validation
remain required before claiming this correction.


### Owned context retires frontend; fresh Notes tour adaptation

Actual changed-context check65.841s: owned context is separate from original,
frontend beforeTERM has listeneronly, graceful0.004s/exit0/groupEOF; API2.233s
actualzero/grace/all core and parent current-session cleanup/full originalcanonical/
owners pass. This validates the context cleanup correction; no frontend timer or
product change. Work fails earlier at OpenSettings after foreign Note denial:
visible unknown dialog,2Auth requests from prior Settings loader plus Notes
notification. Actual probe200 but its strict pre-cleanup callback correlation is
inconclusive; do not claim failure-time APIzero. Requests are single starts, not
duplicated IDs; public initiators are the app bundle.

Current source NotesManagerPage2735-2790 auto-launches notes-basics1000ms after a
fresh Notes visit unless the user already interacts; TutorialRunner exposes its
normal Skip tour action. The Notes definition supplies eight fixed public step
titles. Adapt only a visibly matching Notes tutorial tooltip/title/Skip tour; no
generic dialog dismissal, storage injection or assertion/timer change. Preserve
unknown-dialog failure and all strict request/teardown fences. Review before run.


### SQLite targeted browser recovery accepted

Actual21.140s completes foreign Note denial, visibly matched normal Notes tour
Skip, current Logout, exact Settings history558/index1 knownBFCache, Login/privacy
and strict actual3-ledger drain. All owned contexts/handles and browser/core
sessions close; original tab/URL/draft/fullcanonical/owners preserved. Frontend
beforeTERM listeneronly, graceful0.007s/exit0/groupEOF; API2.179s actualzero/lifespan/
cleanupempty plus fresh final callback pass. No failure probe ran on success.
The context lifecycle and source-specific first-visit UI handling are qualified
operator corrections, with no app cancellation/time/cache change or build.

This closes selected SQLite recovery/teardown only. Four Alice exports were
qualified earlier; Bob tail is still unattempted. Prepare a bounded Bob-only
continuation that skips Alice and already-qualified scope work. Completed Jobs
now create additional owned Sources, so the old singletonsearch oracle must not
be replayed against this changed fixture. PostgreSQL browser, generation/inflight
isolation, further UI states and human UX remain open.


### Correct retained export accounting before finish

Fresh private-control inspection finds both Alice and Bob already have a message
ID. The previous claim that Bob's entire tail was unattempted was inaccurate; a
message ID alone does not qualify any export body. The first proposed continuation
correctly refused its absent-message guard before callback mutation (13.254s outer,
API graceful2.160s/actualzero/currentlogoutsclosed/originalownerspreserved). No
export acceptance resulted. Remove the unqualified resume option and preserve the
original qualified full-run behavior. The replacement finish is strictly GET-only:
Bob own Note, four owned export bodies, foreign Chat export404 and Alice own Chat
positive. It refuses PG/missing seed and performs no seed, PUT, search or scope
replay. Four operator controls pass; Ruff0/Bandit0. Final review and live finish
remain required. Earlier Alice export evidence retains only its stated scope.


### Source durability and synthetic-password failures

X02 new native failures: read-only SQLite export finish14.841s passes Bob Notes/cards/Chat bodies, then original Source download404. API grace2.264s/actualzero/all current sessions closed/originalowners preserved. Normal read-only DB/storage diagnostic confirms both actors retain active Media/file records but neither blob exists under current installed-package storage root; root is outside private profile. Factory default is anchored to replaceable package and ignores USER_DB_BASE_DIR; independent source review confirms durability and profile-isolation defect. Do not recover deleted package/proofs or treat those fixture exports accepted. Add causal real factory tests for configured profile root, reused numeric keys/profile switch, package relocation, explicit path and reset; minimally bind default to existing DatabasePaths profile root and effective-root cache. Preserve explicit override and document ownership-confirmed legacy migration; no automatic ambiguous-root import or deletion-service redirect. ADR assessment: ADR020 governs existing profile paths, no new tenancy/storage authority. Normal config wiring of optional file_storage_path is absent and must be explicitly handled if exposed. Separately, official PG browser preparation22.600s failed Bob account POST400 before browser action; actual password-service validation identifies random synthetic password's forbidden sequential/repeated characters. All sessions closed/API2.281s actualzero/originalpreserved. Reuse normal policy-validated secure generator; do not weaken policy or blindly replay failing credentials. X02 remains open.


### Storage repair source verification and password preparation

X02 storage causal baseline4failed1passed/1.01pytest2.779outer. Minimal factory repair uses profile DatabasePaths root/user_files, normalized effective-root cache and explicit FILE_STORAGE_PATH override; existing settings override/reset/relative keys preserved. Normal connected cohort78 assertions pass365warnings23.83pytest28.168outer, but one explicit-setting test's LazySettings teardown fails. Only its proxy ownership corrected; focused finish1passed6warnings0.91pytest2.592outer; no77-control replay. Initial causal test used old default outside tmp; exact synthetic Alice-private blob removed and fixture now fences both old/new package roots inside tmp. No user/runtime files touched. Compile3/Ruff0/Black0; baseline/current productionBandit0/newtests0/operator0/errors0/diffclear. Independent actual review CLEAR after documentation precision and password literal/binding corrections. Operator password preparation reuses normal installed PasswordService with bounded32 interpolation-safe generation, rejects ${ on deserialize, updates exact quoted canonical Admin field/state/env before initializer. Five initial plus4literal/bound/dotenv operator controls pass; temporary trees removed. No live repaired-package acceptance yet. Legacy blobs require ownership-confirmed preservation/migration, no automatic cross-profile fallback or deletion-service redirect; missing bytes cannot be recovered by this path fix. Original runtime unchanged; X02 remains sole active.


### Reviewed storage package ready for live checks

X02 storage repair local commit6abd503df1e1665f4643562cf48138f96c6d0b5d, parent29a1,6files293insertions33deletions. Final staged review CLEAR; normal scoped hooks/commit2.575s, exact reviewed patch37331bytes SHA256fc3e2ca64c77a3c9945407deb5805fc5442cd5898124d6ffcf7d9cf98a03b16f. No publication. New private storage-root package PASS7.810s/3817files/3roots/337nodes722edges/cleanupempty; normal wheel/install, every child capped120s and original owners/package/env/source preserved. BASE host pins new site, unchanged TH frontend. Retired failed PGprofile removed39entries1160900bytes after closed sessions/no open handles/official DB teardown. Password preparation/controls/review qualified; fresh PGbrowser validation next. Existing SQLite missing original blobs remain failed until legitimate normal reupload/download recovery, not silently reconstructed or accepted. No original runtime cycle or frontend build.


### Browser budget admission and explicit new asset recovery

X02 corrected-password PG preparation25.996s reaches browser callback after both accounts/resources created. Browser refuses before CDP at unchanged90s admission: nested250-35-75-45 allocation leaves nominal95s before preparation. No browser checks ran; cleanup wrongly attempted an attachment never started. Privatefrontend0.005/API2.322actualzero/all current logoutsclosed/full originalpreserved. Reviewed operator correction allocates280s only to browser cohort within unchanged300s pytest cap; othercases250, all local cleanup/supervisor reserves/90s admission unchanged. if(acquisition) retains every started/rejected/late CDP promise but skips never-started cleanup. Three real operator controls pass without attachment/application requests. New SQLite-only original-reupload/read phases reviewed CLEAR: one supported upload per actor with pending-intent/acknowledged IDs, exact-owned bytes; later fresh API GET-only persistence required. Five asset controls/compile/Ruff/Black/Bandit0; no live asset acceptance yet. Missing old blobs remain historical failures; new upload is not automatic restoration. Original runtime remains unchanged.


### Live storage reupload and fresh-process persistence accepted

X02 repaired storage native recovery PASS27.210s outer: one legitimate normal original-file upload per actor retains existing MediaID1, then exact owned-byte downloads; independent fresh API process GET-only downloads both pass. API graceful2.220/2.263s, actual lifespan allzero/cleanupempty/all current sessionsclosed/full originalowners preserved. This recreates known fixture bytes via supported reupload; old deleted package blobs are not recovered. Reupload callback initially returned final-read helper readOnly:true despite its earlier uploads; summary flag corrected to false, GET-only phase remains true, no behavioral change or application replay. Product source6abd/private3817package/THfrontend unchanged. Corrected PGbrowser allocation/review ready; no original runtime cycles.


### PostgreSQL visible ownership passes; old-document drain fails

X02 PG full browser71.046s: all14visible ownership/switch/oldtab/reloadBack/unsentdraft/independent-context controls pass. Strict finaldrain FAIL: anonymous main Settings26pending/independentSettings4pending/oldLogin0; all3requestledgerscohortmatch. Old-loader Auth/sessions/NotesTasks/buddies/profile/modelmetadata lack terminals; several notification streams200/no terminal. Failureprobe200 but failure-time callback correlation inconclusive;15notificationtasks complete only afterNodecleanup. No browser-cell acceptance. Front graceful0.006/API2.370actualzero/all sessionsclosed/fulloriginalcanonical/ownerspreserved. Trace shared transport/pagehide and BFCache lifecycle causally; do not weaken drain, close contexts before it or replay14passing functional controls. Existing Splash catalog read-only preflight200/exact requested model present; no inference/model changes. Generation-isolation preparation continues independently within X02. Failed official PGprofile is retired after database teardown and eligible guarded disposal.


### Direct web GET document-lifetime causal repair scope

All14 PostgreSQL visible isolation controls passed, but final old-document drain
remains failed. Source confirms direct GETs watch account changes without a
pagehide/BFCache lifetime. Exact native pending-ID attribution is still unproved.
First repair scope: direct HTTP(S) GET fetch/body/config preparation, coalesced
callers, result/rate-limit publication and restored-page fresh requests. Capture
the document signal before the first await, abort on pagehide and renew only on
pageshow; retain original coalescing eligibility and promise-identity cleanup.
No fake account transition or cancellation merely because a tab is hidden.
POST/refresh rotation, extension messaging, streams, uploads and raw bootstrap
remain unchanged and separately open. ADR assessment:no new durable rule; retain
existing request/account ownership. Causal real-request-core tests precede source
edits; preserve the spent-refresh control, check both configured clients and
affected transport consumers, then scoped security/review and private build plus
targeted native lifecycle checks. Do not replay14 passing browser controls or
claim broader lifecycle acceptance. Original runtime/session stays unchanged.


### GET proxy evidence and connected-entry causal preparation

bgRequest-direct GET lifetime causal failures confirmed for fetch/body/config,
coalescing, calls while retired, late429/500 and spent-refresh/no-replay. Fixed11
focused cases pass. Reviewer-found warm cached429 settlement fails causally in
both return modes; await/finally fence added. UI20affected modules623pass before
warm extension, currentUI237proxy controls+2warm finishes pass; WebUI19other
modules526pass plus current2proxy modules239pass. New fixture owns/restores its
listener/visibility spies explicitly; warm clock calibration excludes Event's
own timestamp. Syntax0; lint baseline/current exact15production+5test inherited
warnings/0errors/noignored; BanditN/A(TypeScript only). No native acceptance yet.
Review limits this to bgRequest-direct GETs: actual TldwApiClient config await and
direct requestWithCurrentConfig, plus apiSend, have other entry points. Before a
build, add real-client held-config/body and apiSend body/coalescing causal cases;
only confirmed gaps justify extending the same document-lifetime signal. No new
architecture rule or authority. POST/refresh rotation, streams/uploads/bootstrap
remain excluded, and original runtime/session remains unchanged.


### Connected GET repair verified; targeted native preparation

Six additional actual-client/apiSend causal failures1.488s confirm held-config,
direct-body and coalescing gaps. One HTTP(S) document-get-scope helper now binds
only GET at bgRequest/apiSend and base-client request/requestWithCurrentConfig/
fetchWithAuth. POST/refresh/stream/upload/bootstrap logic remains unchanged; the
spent rotation still completes for its original account while its retired GET
is not replayed. All19focused cases pass2.240s; final32connected modules962pass
in each client(UI31.238outer/WebUI31.348outer), no skips reported. FiveTS parse0;
exact baseline/current inherited lint15+4+517production/5test, helper0, all0errors
and noignored. BanditN/A TypeScript-only product changes. Final review/staged
review/hooks/commit remain required, then one private frontend build. APIpackage
remains6abd/3817. Prepare explicit prepare-denial-browser phase plus officialPG
case within unchanged280/300 budgets. Target denied-Note, normal Logout, exact
Settings BFCache/history and strict3-ledger drain/fresh server callback/normal
shutdown; do not replay14passing whole-browser controls. Original runtime/session
unchanged; X02 remains open and exact native pending-ID attribution is unproved.


### Final GET exclusion and disposable-package cleanup

HTTP(S)+real extension runtime.id negative control fails1.342s before helper
exclusion. Corrected shared guard preserves that extension behavior; complete
106-case lifetime/refresh module passesUI15.889outer/WebUI16.196outer. Prior
32modules962each passed before this narrow guard; no unchanged broad replay.
Final helper syntax/lint0; existing diagnostics unchanged. Operator compile/
Ruff/Bandit0errors0 and three pure phase/budget controls pass; no app/DB/browser
action during those controls. Actual+staged review, normal commit, private build
and native targeted checks remain. Retired unselected worker package removed
4190entries/78,001,469bytes after no handles/references, UID and symlink guards;
protected original owners unchanged. No native acceptance or X02 closure.


### Reviewed GET repair committed and built privately

Independent actual+staged review CLEAR; normal scoped hooks/local commitf286a3b3cb
pass2.819s. Committed7path patch exactly reviewed61131bytes SHA256
2e05b1db1f532ca905b55b6123ca1ae5a481edd760742f30261d22c8687e7b0c; unpublished.
Private normal buildPASS75.245s/fGDxU2Zf7SJMROKg0qgUg/7732inputs/5rewrites18908,
token-sync/bundle593.7KBunder600KB, no source/owner restoration errors. Existing
configured typecheck skip disclosed; no fullTS qualification. APIpackage6abd
unchanged3817files. Functional build pin verified; targeted officialPG native
denied-Note/logout/exactSettingshistory/strictdrain check now running. No native
acceptance yet; originalsession/runtime preserved, X02open/X03untouched.


### Repaired-build PostgreSQL target still has one pending session read

Actual62.758outer/61.06pytest6warnings FAIL at finalstrictdrain. Foreign Note
denial, visible known Notes tour Skip, normal navigation/logout and exactSettings
history625/index1 pass; one GET/auth/sessions request92860.152 lacks status/terminal
in all3matching ledgers. No native/browser acceptance. Notification2starts/2done
are post-Node-cleanup only; readiness failure-time correlation inconclusive.
Browser final originalTab assertion was not reached after drain; parent full
originalcanonical/owners pass. Frontgrace0.006exit0/API2.310graceful-15, actual
lifespan zero/shutdowncomplete/cleanupempty/all current sessionsclosed. Need exact
current-build loader/initiator attribution; do not infer another source cause.
Existing outer formatter omitted that navigation detail from RAM; no old log or
proof recovery. First changed-build SQLite target will retain bounded public
loader/initiator/lifecycle fields, same driver/guards/budgets, no14control replay.
Failed officialPG profile is disposable after teardown/nohandles/ownershipguard.


### SQLite identifies restored loader; prepare a transport-only failure probe

Changed-build SQLite targetFAIL52.953s after same4visible passes. OneGET/auth/me
94151.148 status/terminalnull in matching3ledgers; untruncated journal records
starts1/no previousTerminal and originalSettings loader2FDA6561C133E3C93E12F65D7743E018
restored via actualBackForwardCacheRestore. Public current _app16:12586 maps to
request-core fetch. Start relative restore remains unknown; no exact sourcecause.
Parentcanonical/owners preserved; browser finalTab assertion after drain notrun.
Front0.006/API2.234graceful-15/actualzero/lifespancomplete/cleanupempty/all sessions
closed. Failuretime server correlation still inconclusive. Shared pre-entry
Auth/current-user and connection-check awaits are hypotheses, not attribution.
Next distinct diagnostic: passive request start timestamps; at actual failure
only, bounded anonymous GET of observed auth/me or auth/sessions via both private
API and frontend, within existing14s probe cap. Fixed whitelisted paths only,
status/elapsed/byte counts in RAM, no queries/headers/bodies in output. No app
change, writes, models, build or originalcycle; strictdrain remains required.
Pure whitelist/pairing/timestamp controls and review precede one SQLite run.


### Read-only failure diagnostic reviewed

Source reviewCLEAR; whitelist11 +exact anonymous pairing/clock1 +actualNode
journal receipt/CDPclock/reused-ID/listenercleanup1 controls pass. Compile/Ruff/
currentBandit0findings0errors. Node receipt time is eventReceivedAt, not network
start; retain finite CDP networkTimestamp/networkWallTime separately and make no
BFCache start-order claim from receipt time alone. One distinct SQLite diagnostic
uses existing fGD build, same strictdrain/14s probe/2s HTTP/native caps and original
preservation. No app source/build/model/credential change or acceptance waiver.


### Three native failures reassessed; isolated history calibration

X02 latest repaired-build SQLite diagnostic: FAIL51.195outer after four visible controls passed. One GET/auth/me97147.148 has no status/terminal in all three browser ledgers; actual CDP wall start1792842337.138369 precedes Notes commit1792842337162, and originalSettings loaderE9D22B7D502B81D635385BFF1BB8AC34 is restored by BFCache. Current public request-core fetch initiator16:12586. Failure-time fresh anonymous auth/me returns expected401/66bytes: directAPI4.716ms,frontend7.057ms. Exact fresh readiness callback task_done has pending_tasks/server_tasks/queued all0 while browser remains open; frontend has LISTEN only. This does not establish phantom-browser attribution or native acceptance. Notification2starts/2done remain post-Node-cleanup observations. Front0.004/API2.212 graceful, actual lifespan zero/complete/cleanupempty/current sessionsclosed; parent original canonical objects and owners preserved. Browser final original-tab assertion after strict drain was not reached.
Three post-build native failures are reassessed rather than replayed. Product document-GET lifetime tests/build are qualified, but pre-entry await and Chrome BFCache ledger hypotheses remain unproved. Next distinct calibration uses one temporary owned context/tab in existing Chrome and an isolated ephemeral loopback HTTP fixture, two synthetic pages and one held GET. Require native fetch AbortError, pagehide/pageshow persisted, actual server response-close before finish, actual history restore and all three event-ledger snapshots before context close. First prove fixture cancellation with a real local HTTP client. Strict application acceptance/drain remains unchanged. Total60s with15s reserved cleanup/15s operations; no original API/frontend/Chrome restart, app navigation, credentials, models, installs/builds, proof artifacts or Git mutation. Later source changes require causal evidence. ADR not required: diagnostic only, existing lifecycle architecture unchanged. Existing read-only reviewer checks operator before native execution. Latest human necessary-diagnostic/retry approval applies.


### Isolated calibration guards and review

X02 isolated history-calibration preparation: normal real loopback client cancellation PASS0.228s, response-close before response finish and listener release; exact-origin/terminal/listener ledger control passes, Node syntax passes. Initial sandbox listenerEPERM was permission-only before app/browser action, corrected with authorized escalation. Read-only review found raw AssertionError message privacy risk and post-BFCache callback-order seam; metadata-only own-file frames now replace main raw message, and an existing-left bounded wait requires actual persisted pageshow plus AbortError before copying receipts. No app source/build/credentials change. Native calibration has not run; application strict drain/acceptance remains unchanged.


### Real cancellation reproduces three stale event ledgers

X02 isolated Chrome/BFCache calibration PASS0.591outer: real fetch AbortError plus pagehide.persistedtrue/pageshow.persistedtrue and actual BackForwardCacheRestore; held server response closed before finish and before browser cleanup. Despite that, page/context/CDP request ledgers each remain1 with null status/terminal after navigation and Back. All owned handles/listener closed, original tab/draft and protected owners unchanged. This causally establishes event-ledger nonterminal is not proof of a live connection; it does not attribute each prior application request or accept X02. Next one synthetic same-tab normal reload will observe whether native terminal events arrive; no ledger mutation or acceptance exception. Same isolated60s/15s reserve/15s ops scope, no app API/frontend/runtime/credentials/models/build action. If events remain absent, correct validation around real owned-context closure and exact server task/stream/zero receipts, retaining unknown event metadata and all privacy/write assertions. Review before invocation.


### Correct event-ledger measurement without inventing request completion

X02 synthetic reload also reproduces stale event records: calibration PASS0.547outer, after normal reload page/context/raw remain1 with nullstatus/terminal, while native AbortError and server response-close precede browser cleanup. Owned resources/listener/original tab/draft/owners preserved. Read-only design review: nonempty browser ledgers are not live-connection proof, but native GETs must remain UNKNOWN rather than invented cancelled/completed. Correct only the X02 driver: retain strict empty default; finite <=2 observed anonymous auth/me/auth/sessions GET residuals may defer only with matched three-ledger counts/signatures, no response/write, exact request IDs and actual original-loader navigate-away/BFCacheRestore journal. All functional/privacy assertions and successful complete auth writes remain mandatory before closure; authority/write observers remain through closure. Require exact owned contexts/pages CLOSED; parent independently requires that receipt, unchanged notification/SSE/server lifecycle, fresh zero-task readiness and bounded graceful release plus original preservation. Unknown reads cannot count as own/denial/response evidence. Pure positive/refusal/closure controls and independent source review precede changed-driver native checks. No product edit/build follows this calibration; ADR not needed because validation measurement only.


### Sequence and closure fences validated

X02 changed-driver controls now pass: unknown-history1positive33refusals; complete authwrite1positive3refusals; closure request-start1positive3refusals; actual navigationJournal same-millisecond pre-away positive/post-BFCache-restore refusal and listener cleanup. Parent exact context-closure1positive4refusals pass. Review caught timestamp-order ambiguity and passive-detach gap before native: request now captures exact navigation row count and requires later away/restore sequence; all-API request-start map epoch remains fenced through awaited owned context closure, rejecting later starts. Node syntax/Pythoncompile/Ruff pass; current operator Bandit0findings0errors. No application acceptance yet; existing strict API/SSE/zero/graceful/preservation gates unchanged. Completed isolated-calibration operator is removed after both successful native results and listener/context release; concise results remain in task/plan, no proof payload retained.


### Both native history boundaries pass; proceed to generated isolation

X02 repaired-driver native history boundary PASSES both backends on fGD/f286 frontend and6abd installed API: SQLite21.269outer; official PostgreSQL1passed6warnings28.41pytest30.081outer. Each run passes foreign-Note denial, known tour Skip, normal navigation/Logout and exact originalSettings history restore. One auth/me event record per run remains explicitly UNKNOWN/completionEvidencefalse; it is not endpoint completion evidence. All owned contexts/pages closed, no new API requests after final closure fence, all current sessionsclosed, both notification tasks2/2 complete before browser return; fresh strict API/server/SSE checkpoint passes. Frontend onlyLISTEN before graceful release0.005s; API graceful2.245SQLite/2.334PG, actual lifespan active/pending/server/connections0, no app/cleanup errors. Browser original URL/draft and full canonical objects/owners preserved. PostgreSQL disposable profile removed after official teardown/nohandles/ownership guard:61entries1,709,453bytes. No14control replay, no product/source/build change or X02 closure.
Next X02 work: actual generated RAG/representative Chat plus cache separation and in-flight switch. Prepare real cached retrieval with Bob positive, identical Alice query without Bob nonce, check candidates/citations/answer, then representative real Alice Chat on existing Splash8010. Reuse provider_config for ONLY disposable private config; bind semantic cache to that private profile through supported RAG_SEMANTIC_CACHE_DIR and leave models/services untouched. Normal authenticated public calls only, no mock generation, downloads, hidden retries or unknown-write replay. New generator requests retain180s and total600/officialcase300, normal current-session cleanup and actual zero/graceful/preservation gates. Pure driver guards, source review and operator security checks precede live use; any demonstrated product failure requires normal causal red/green. X02 remains active; X03 untouched;34/43 accepted.


### Generated request returns; cache control needs actual metadata

X02 first SQLite generated-isolation run FAIL16.962outer at require_rag BobGenerated cache-hit control. Warm Bob retrieval passed coldFalse/owned-doc/noAlice controls; Bob generation HTTP200 response returned, then expected cache_hitTrue failed. No Alice RAG/Chat dispatched and no generation/answer acceptance claimed; safe response metadata was not retained. No old-log/proof/body recovery. Actual API graceful2.182/lifespanall0/shutdowncomplete, all current sessionsclosed, protected owners unchanged. Private config only uses reviewed existing8010 external provider; catalog HTTP200/expected model was checked before run. Pure generator guards passed before a temporary-path alias setup error; canonical tempconfig-only follow-up passes; no application change. Current compile/Ruff/Black/Bandit0 and source review clear within stated scope.
Next distinct diagnostic: at most two ordinary Bob RAG retrieval-only requests with identical cache-enabled payload, no model calls, capturing safe cache_hit/retrieval_cache_hit/generation_executed booleans, document/error counts and fixed whitelisted cache-bypass reason/code only. Retain strict own-positive/noAlice checks, but cache-hit result is observed diagnosis, not passed isolation. Improve the cache-control failure message with that same safe metadata before future generation; no raw bodies, keys, queries, credentials or cache-file contents. Inspect actual current cache identity/write/authorization-scope source before any product change; no bypass of fail-closed caching. Same private API/profile/600host/15request/120startup/15graceful/current-session/zero/preservation scope. Source review precedes execution; latest human diagnostics/retries authority applies. X02 remains open and X03 untouched.


### Confirmed Balanced-preset explicit-setting defect

X02 retrieval-only diagnostic completes13.954outer: two identical Bob requests each1owned document/0errors/cacheHitfalse/retrievalCacheHitfalse/generationExecutedfalse; both explicitly report cache_bypassed.reason=secondary_retrieval_mode. No model requests in this explicit ten-feature-disabled diagnostic; actual installed publicschema/default-resolver guard verifies10falsefields against forcedtrue defaults. Earlier primary-cwd import selected primarysource and is UNQUALIFIED; corrected source/private-profile binding only, no application startup or qualification adopted from that control. Actual private config search_default_mode=balanced confirmed by public-key-only read. Current unified_pipeline2705-2770 re-enables expand_query despite explicitFalse when Balancedpreset is selected; canonical schema promises individual explicit overrides. This source behavior explains the observed intentional cache bypass; no auth/cache leak is claimed. API graceful2.238/actualallzero/sessionclosed/ownerspreserved.
Bounded repair design under existing human all-necessary-fix authorization: preserve current public schema and canonical Pydantic field-set through ResolvedRAGRequest into preset application; explicitFalse and explicitdefault values win, while omittedsettings and legacy direct-call heuristic remain. Search all five production dataclass constructors and affected standard/batch/agentic/RPG consumers. Add normal causal publicschema/resolver/realpipeline cache regression first, plus omitted/default-value controls, then minimal shared fix, affected checks, compile/lint/baseline-current scopedBandit and independent actual/staged review/normalhooks/commit. Refresh private installed package through existing build/install path only after qualification; current frontend unchanged. Revalidate genuinely affected native cache/generation/isolation only; do not change the environment/preset to hide the defect. ADR required:no; restoring existing documented request precedence, no new public authority/schema/storage/fusion rule (ADR053 fusion unaffected). X02 remains active beforeX03;34/43 accepted.


### Canonical preset repair and connected qualification

Normal causal regression on unchanged production:4failed3passed. Canonical explicit
field-set now survives request resolution; presets and compatibility aliases
cannot override explicit values. All six public preset knobs are covered; HyDE
is internal-only and its omitted quality preset remains intact, with a separate
internal canonical-False control. No public schema/API fingerprint change.
Independent review caught a scalar-plan coherence edge; two new causal failures
(3,.7) and (10,0) corrected by replacing only plan.top_k, preserving min_score.
Final eight-module qualification:101passed8warnings/2.78pytest4.825outer/natural0,
including10 new cases and real isolated scoped cache. Compile3 pass; touched
baseline/current Bandit0findings0errors; Ruff no new diagnostics; both inherited
whole-file Black flags retained, new test clean. An initial new-test C408 was
corrected; an initial test-only nonexistent-public-HyDE assumption was corrected
without a product/schema change. Actual/staged review and normal hooks/commit
precede private wheel refresh and affected native cache/generation/isolation.
No native acceptance or X02 closure yet; protected original session untouched.


### Preset package and real SQLite RAG qualified; Chat receipt unknown

Local e830669e8f committed reviewed5paths353+/12-; exact full-index patch56249bytes
SHA256286298d72c3858cdb43437ccea2b74b49234c78c29e2726f6d6c1a486f274ec4.
Fresh private preset package verified3817files/3roots/337nodes/722edges,7.380s.
Changed-source SQLite native30.999outer passed all four RAG checkpoints before
Alice complete-v2 failed HTTP/deadline guard. Bob cold retrieval, real generated
private answer with cache hit, Alice same query without Bob nonce, then retained
Bob cache after Alice all passed. Their exact safe response projections were not
retained, so claim only those enforced checkpoints. No SQLite RAG replay needed.
Alice Chat receipt/status unknown; isolation-chat intent retained, no save or
Chat acceptance. API graceful2.178/actualall0/complete/cleanupempty/current3sessions
closed/owners preserved. Next read-only supported Chat/messages reconciliation
precedes any legitimate retry; add safe completed-response status/shape metadata
to prevent another lost diagnosis. No old log/body/cache/proof recovery.
X02 stays active; PostgreSQL generated isolation and remaining browser variants
remain open, X03 untouched.


### Saved Chat reconciliation and response-contract hypothesis

Read-only reconciliation PASS13.839outer:3messages/1matching question, no character
or persona; uncertain mutation intent retained and no resend. API graceful2.252/
actualall0/complete/current sessionsclosed/owners preserved. Endpoint constructs
completion response after saving with optional conversation.character_id, while
shared response model currently requires int. Establish this exact response
failure with normal endpoint/schema regressions before changing code. Minimal
design aligns ordinary completion receipt with existing nullable Chat identity;
protected Resume's integer/identity admission remains unchanged. Review all
response callers, canonical generated artifacts and protected/ordinary clients.
No invented sentinel ID or repeat of already-saved question. ADR not required:
response representation repair, existing relationship/authority unchanged.


### Nullable ordinary completion receipt qualified

Causal2endpoint failures both reached correct provider/persistence then HTTP500.
Required character_id now permits null, matching existing optional Chat identity;
no sentinel, history change or protected Resume relaxation. Full connected check
37passed654warnings/40.06pytest44.159outer/natural0, including protected generation/
replay/concurrent-writer controls. All15old test definitions AST-identical.
Baseline canonical export matches committed contract; normal shared generation
and full JSON comparison prove only integer→integer|null for that response field.
Required list/allother values and three owned descriptions unchanged;2140paths/
3316schemas/final drift+pathcheckPASS41.374s. Fingerprint generated, path union
unchanged; active normal rich types generated with existing installed tools.
Compile2/Ruff0/Bandit72identical inherited test signatures0errors0new; production0.
Same-config Black: schema clean, test inherited whole-file flag unchanged, added
block formatted without AST changes. Final reviews/hooks/commit/package refresh
precede native receipt qualification. Existing saved3messages remain retained;
no blind resend. X02 remains active beforeX03.


### Empty saved answer is a failed result; pre-persistence guard needed

Nullable receipt repair6797e03285 committed normalhooks/reviews; private package
verified3817files7.403s. Finish-only13.324s stopped before Send on unresolved prior
intent. Distinct read-only diagnostic15.205s reports fixed answer_content reason:
old assistant placeholder is blank. No model/reasoning root attribution claimed;
no empty answer acceptance. All sessionsclosed, actualserverzero/graceful cleanup,
original owners preserved. Current nonstream path saves a space when no answer
and no toolcalls; establish causal tests and reject this before either row write,
using existing validated-toolcall path and retaining valid tool-only/stream/Resume
behavior. No schema change/new option. ADR not required. Prior linked empty pair
may be reconciled as failed/consumed while all rows remain, never called usable.
Later one distinct follow-up uses an existing normal output-limit setting within
180s; no repeated prior Send or SQLite RAG. Retired old packages removed under
exact ownership/nohandle guards:156003520bytes; current/original packages kept.
X02 active beforeX03.


### Empty-output repair qualified locally

- Causal unchanged endpoint:6failed/1valid-tool positive passed, plus4failed
  malformed-tool/emote cases. Guard follows emote cleaning and precedes both
  writes;502/empty_provider_response, no new rows. Valid bounded tool metadata,
  streaming and protected Resume retained. No schema/options/architecture change.
- Connected50passed1060warnings/71.39pytest77.038outer. Final wording and exact
  absent-tool metadata preservation:direct11passed365warnings/30.79pytest35.827outer.
  Compile2/old16testdefs unchanged/configured Ruff0/same inherited Black flags;
  production Bandit0/test72identical inherited findings/no new/errors. Canonical
  fingerprint+paths unchanged15.664s. Initial exporter flag typo failed before
  import, then corrected documented --check; no application retry. Static baseline
  path-ignore mismatch corrected with same configured stdin filename.
- Reviewed operator classifies a fully ID/role/parent-linked blank string as
  failed_empty, retains all3rows, clears only the consumed intent and never calls
  it usable. One distinct follow-up uses normal2048 output tokens within180s;
  this does not prove the earlier provider reason. Wire positive/six malformed
  refusals and blank-parent/role/nonstring controls pass; compile/Ruff/Bandit0.
- Final staged review/normal hooks/commit/private package refresh precede native
  SQLite finish; do not repeat its four passed RAG controls. PostgreSQL generated
  isolation and remaining browser variants/UX/final closeout remain open.


### Saved answer passed; denial observer finish remains

Repair179ab7db8e committed normalhooks2.864s/commit2.049s; exact reviewed4path
patch17571bytes SHA256e0975f383c019889920ff8c39751570585016140781f56fbc57f5e1490b0454a,
unpublished. Private3817file/3root/337node/722edge packagePASS8.566s, cleanupempty.
Native SQLite32.077s retained the proven failed_empty3rows and saved a distinct
nonempty follow-up with correct receipt/roles/parent/exact oldrows+2/noBobnonce.
Foreignmessages GET404/body checks passed; run FAILED only at operator denial
receipt because its selector covered Chat root, not messages. No further Send.
All3sessionsclosed/actualzero/grace2.206/originalownerspreserved. Prepare readonly
known5-ID finish and exact GETmessages observer; positive/five malformed-history
refusals/five route controls passed. PostgreSQL generation remains unexecuted.


### Generated isolation complete in both backend cells

SQLite readonlyfinishPASS19.448s:known5IDs/usablelinkedfollow-up/priorfailed3rows
retained/noBobnonce/foreignmessages404 actualreceipt, zero Sends. All3sessions
closed/freshactualzero/grace2.211/originalownerspreserved. Exact GETmessages
observer, read-only positive/five malformed refusals/five routecontrols reviewed.
Fresh official restrictedPG generatedisolationPASS1test6warnings38.95pytest/
41.128outer. Bobcold1doc0errors/cachefalse; Bobrealgeneratedanswer/cachetrue;
Alicematchingquery1owneddoc0errors/cachefalse/noBobnonce; BobcacheafterAlicetrue;
realAliceChatSaved/foreigngeneratedChatdenied/privateanswernotsupplied. All3
sessionsclosed/actualzero/grace2.421/ownerspreserved; officialteardown and exact
nohandle disposal67entries2717638bytes. Do not replay these passed controls.
Remaining:deeper UIhandoff/helper/autocomplete/notification state, switchduring
realgeneration, settledUXscreenshot/humanreview/finalcloseout. Use existingprivate
maincontext+one temporary samecontextoldtab; preserve originalruntime/session.


### Shared helper boundary investigation

QuickChat store/hook/input currently clear on close/unmount but have no explicit
account-change subscription. Source risk only until causal test/native proof.
Cover actual shared hook/store/observer and input, late stream/docs replies,
auth-preserving refresh and modal/popout consumers. Reuse watchChatAccountChanges;
minimal epoch cancellation/output fence and input reset if causal tests confirm.
No new auth rule/dependency; ADR not required. Review existing popout transfer
scope separately without opening a new window. After qualification/review, one
private frontend build and focused native browser states; no API RAG replay.


### Shared helper repair qualification — 2026-10-24

X02 shared helper repair (latest-dev candidate, 2026-10-24): causal hook controls reproduced five privacy failures (retained messages/input and late stream/RAG publication); module-level retirement additionally covers a closed helper's model/mode. Pop-out transfers now carry the existing verified opaque account owner and are consumed once; actual shared/legacy entry-point controls reproduced both StrictMode lost-restore failures and both pending-restore overwrite failures. The shared effect now revalidates its retained in-memory payload on StrictMode replay and refuses publication after unmount, account/key change or intervening helper work. Three mounted-modal controls cover unresolved-owner refusal, admitted current-store snapshot/blocked-window cleanup and boundary-before-click refusal; window.open is mocked, with no native window created. Seven connected modules pass 59 tests in each configured UI/WebUI runner (4.80s/4.52s); all original behavioral assertions retained. Twelve touched TS/config files parse clean and configured ESLint has zero errors/new diagnostics; one inherited QuickChatInput any warning and inherited Next pages-discovery diagnostic remain. Bandit N/A to this TypeScript/prose-only unit; no Python change or repeated Python scans. No new architecture decision: existing account observer, verified prompt-owner scope, store and transport cancellation are reused. Actual-source/staged review and normal hooks are required before local commit/build. These are source/component checks only: X02 live helper/account-switch and human UX acceptance remain open; X03 untouched. Failed operator cwd/static inventory/tracking-import invocations are not qualification; scoped reruns above completed normally. Original runtime/session untouched.


### Remaining browser setup and outbox correction — 2026-10-24

X02 helper source repair c95f2832b84bcc60761a2339b114c5d07620a7a6 committed locally with exact reviewed 14-path patch 04777e8c9581748129e7249fc8de0aa5c1e4051cc2d7df036c32856ac2c4ba09; normal pre-commit passed1.463s, commit1.260s. Private frontend build P86mohokazhvHKTkJczZX passed89.647s,7734 inputs/five private rewrites/token sync/593.7KB gzip budget; inherited configured typechecking skip, no whole-TypeScript claim. Original owners preserved. First remaining-edge SQLite attempt failed before browser launch at notification-bridge qualification (15.180s); all three temporary sessions closed, API actual shutdown/release clean, no original cycle. The operator failed to retain the bridge summary before validating it; that historical attempt is consumed and will not be replayed. Managed readback confirmed optional JOBS_EVENTS_OUTBOX was off, both original Jobs completed but zero corresponding completion events/notifications and bridge cursor0. An initial readback used a list where the existing RLS API requires a string, so it failed; corrected readback completed0.681s and saved fixed metadata. This is private fixture/configuration correction, not a demonstrated product notification defect. Next bounded setup: enable the normal outbox only in helper-browser private API/worker environments, create one distinct real ingest Job per actor with normal mutation intents, preserve original ids.job/results, complete via the existing real worker, then run one distinct bridge attempt on actual new events. Save its fixed receipt before qualification. Operator selection/bridge/owner guards25, compile3, Ruff and Bandit0/0 pass; source review required before execution. X02 remains open; no helper/browser/in-flight or UX credit yet.


## Prompt diagnostic and helper model routing — 2026-10-24

X02 remaining browser diagnosis: own Prompt API readbacks are correct for Alice/Bob, but the mounted picker shows No matching prompts without a Prompt-list request. Diagnostic collected; normal Logout guard then failed after anonymous navigation. All disposable contexts/current sessions closed, actual API tasks zero, private services released, protected original canonical session/owners preserved. No picker acceptance claimed. A separate source review confirms Quick Chat passes server-catalog tldw: selector values unchanged into the real Chat request without api_provider. Next bounded unit: causal actual-hook/real-service request tests, minimal reuse of existing model normalization/provider resolution with account-boundary fencing, affected tests/lint/security/independent review, then a private build and remaining native checks. Existing ADR/account-authority rules govern; no new durable architecture decision.


## Model routing and Prompt feedback qualification — 2026-10-24

X02 helper model routing and Prompt feedback qualification: five unchanged-source wire cases reproduced missing api_provider/retained UI/provider prefixes; minimal shared resolver/normalizer use plus post-await generation fence passes eight new routing/cancel cases and nine existing boundary cases. Picker initial new harness failed before causal assertions (accessible-name/legacy-I/O setup), corrected harness then reproduced two pending/refusal feedback failures. Keyed ownership failure/retry now passes pending, thrown-refusal, normal-null refusal and retry admission without weakening owner-required listing. Connected UI cohort137 passed/1 failed (local unconfigured empty-library control); localOnlyReady now follows completed config loading while authenticated authority remains guarded, and that unchanged failing case passes focused. Final WebUI all138 cases/eight modules pass18.884s natural0; UI137+focused1 retain exact stated scope. Five touched TS files parse0; configured baseline/current ESLint zero errors, one identical inherited test any warning, no added diagnostics; inherited Next pages-discovery diagnostic remains. Bandit not applicable to TypeScript/prose-only source scope. Native Prompt ownership/remote auth-write causes remain unproved; no X02 acceptance/UX claim. Capped auth-write metadata projection has four guard checks and keeps strict write-failure rejection. Independent final source/staged review, normal hooks/local commit, private build and live remaining checks still required.


## Current private build and remaining helper diagnosis — 2026-10-24

X02 local repair committed normally as2e746c40a4c651fd5d64c12f3db85df9ce2d1db4; reviewed seven-file binary patch35287bytes/SHA256268f23f6b12e799663c55f9a69aeea63ac2d38fc0e6461b4a243a15743c0d627, normal precommit natural0/1.400s and commit1.129s, tracked-clean at commit/unpublished. Private build DvZ0FkqSyO65jyxcrcyNX passed74.648s/7735inputs/token-sync/bundle593.7KBgzip below600; inherited configured typechecking skip disclosed. Original owners unchanged. Changed diagnostic48.490s observed13 auth/me GET200+finished, no list: ownership-key remount closed the disclosure; all Login/Logout writes200/completed, clean private release/original canonical preservation. No native auth failure established. One bounded normal owner-resolution reopen now requires a NEW completed200list before common search; four async operator remount/admission/failure/budget scenarios pass. Next SQLite no-Send cohort40.465s qualified actual Alice Media handoff, own Prompt suggestion and exact new Job notification ownership, then failed before helper draft/mode setup at dialog visibility wait. All actual4SSE tasks completed, contexts/current sessions closed, private API/frontend released, original canonical/owners preserved. A disposable real-component/real-Antd selector check passes1case/1.907s with exact Quick Chat Helper accessible name and unique title anchor, so native selector/product cause is UNPROVED. Prepared a helper-only capped DOM/public-chunk/categorical-error diagnostic (one open/15s observation), no Media/Prompt/Job/model replay or data writes. Operator compile3/Ruff/Nodeparse clear, Bandit0findings0errors. X02 remains open, no screenshot/UX acceptance or in-flight generation credit; X03 untouched.


## Redesigned Chat integration — 2026-10-24

Human accepts X02 UX, and reports the redesigned /chat. Functional closure remains open. Fetched exactdev17141ff73c3caee7998d93a290598d6c3ab4ec0c adds132netpaths, including selectable message views, calm composer, ChaCha transaction ownership and workflow/Claims fixes. Preserve all18 existing commits and task/plan WIP during normal isolated rebase, review combined changed callers/tests/fixtures/config/exports, run genuinely affected checks and scoped security, then refresh the inactive installed API and private frontend before native continuation. Original API/frontend/Chrome/session remain untouched. The prepared read-drain diagnostic has not run. The prior15 browser content controls reached their assertions, but final drain failed; do not promote them to whole-cohort acceptance or blindly replay them. X01 remains closed; X03 unstarted. No new ADR: integrate existing upstream UI/ADR080 rules while retaining prior UAT ownership contracts.


### Redesigned candidate qualification

- All18 rebased commits remain identical by range-diff on fetched dev17141; local sourceff08bf. Notes stash2ba4a30 is retained and restored. PR3309 composer and PR3313 message views are included. PR3310 rails and PR3311 header remain open references: their actual production patches were reviewed, but are not integrated or qualified here.
- Incoming18Python modules:399passed/864warnings/123.63pytest/129.795outer, official PostgreSQL required. Initial sharedUI32modules:827passed/2failed. Corrected AntD wait preserves its visibility/callback assertions; picker fixture now holds all6 profile loads before deliberate selection. Full corrected2modules90passed/31.117outer; fullWebUI33modules835passed/75.387outer including6 browser-helper controls.
- Causal fixture cleanup failures: tx3setup cases; semantic/FVA6failed1passed in2.79pytest/5.582outer; Persona4failed in2.33pytest/5.007outer. Five new acquisition sites now enter owning try before constructor/schema/seed and use nested independent pool cleanup; FVA compensates only committed roles. Complete corrected3modules38passed/846warnings/23.49pytest/25.482outer; Persona11passed/6warnings/11.45pytest/13.517outer. All original Python behavior assertion ASTs retained.
- Compile39; Bandit exactdev/current1050identical findings/0new/0errors; finalPersona8identical assertion-AST findings after correcting a formatter-only line signature mismatch. Four changedPython Ruff/Black pass. Proper shared-apps lint53TS files:328identical warnings/0errors/0ignored/0new, parse0. Earlier ignored-file lint invocation is unqualified. Independent incoming production/config/test/affected-owner review and current6test-path review CLEAR; long foreign plans/tasks only selected narrative coverage, no historical claim adoption.
- Fresh installed API /private/tmp/tldw-x02-dev17141-20261024/site:3817files/3roots/337dependencyNodes/722edges, qualified7.859s with original owners and old package preserved. New frontend build and remaining live checks pending. X02 UX accepted by human; functional drain, in-flight helper and remaining PostgreSQL checks OPEN. No original runtime cycle, publication or X03 work.
