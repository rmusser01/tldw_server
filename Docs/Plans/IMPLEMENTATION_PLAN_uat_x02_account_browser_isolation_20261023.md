# X02 account and browser isolation implementation plan

> Execute inline using the executing-plans workflow; reuse the authorized read-only reviewer.

**Goal:** Finish selected X02 before X03, proving reciprocal ownership and browser-state isolation with working owner controls.
**Architecture:** Use the qualified current3817-file installed API and unchanged TH frontend on private18908/18989, alongside the protected original runtime. Reuse official PostgreSQL isolation and existing lifecycle/receipt/session restoration helpers. Use supported real API/UI paths; mocks do not qualify live isolation.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md X-02 and apps/tldw-frontend/scripts/live-tier-uat/release-catalog-data.mjs X-02.
**Task:** TASK-13260.278.18.83.45.
**ADR assessment:** No new architecture rule for validation; existing AuthNZ/tenant ownership, PostgreSQL RLS and documented revocation contracts govern. Any demonstrated repair gets its own assessment and causal red/green before product edits.

## Global constraints

- Product source22141c4223 on reviewed dev22c9; current Jobs-scope package3817files, unchanged buildTHttd0p00MlSSxPE1Yyre. Refresh dev before later publication, not after every step.
- Preserve original API50010/49993, frontend50863/50855, Chrome54716/54710, five saved messages, exact draft/URL and full canonical objects. No original runtime cycle.
- One temporary independent browser context is necessary for isolation; keep the existing context/tab and close only the temporary one after validation. Never treat two shared-storage tabs as independent identities.
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
- [ ] Run sequential SQLite/PostgreSQL browser cells; use one temporary independent context and close it.
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
