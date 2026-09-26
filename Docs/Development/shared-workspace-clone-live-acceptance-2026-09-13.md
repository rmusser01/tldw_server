# Shared Workspace Clone Live Acceptance

Tracking: TASK-12020.50. Status: **In Progress, not certified**.

## Runtime

- Worktree: `.worktrees/shared-workspace-clone-ux`, branch
  `codex/shared-workspace-clone-ux`, base `c70387f496d82fcee92926bf3715bf5cd240ba88`.
- Runs include the uncommitted TASK-12020.49 UI plus the four repairs
  described below. This is not evidence for the unchanged base revision.
- Actual FastAPI app, production clone Jobs worker and SQLite databases;
  non-production multi-user JWT mode. Backend localhost port 18095,
  current Next.js WebUI port 8095, Chrome controlled through CDP port 9225.
- Synthetic owner, recipient and outsider accounts; owner and recipient are in
  one organization. A separate synthetic admin configured a custom role granting
  only `sharing.read` equally to all three subjects. Subjects are not admins.
- No mocked sharing, auth, Jobs, ingestion, clone or retrieval responses.
- Runtime data and raw artifacts: `/tmp/tldw-clone-uat-jIXQx6`. Credentials are
  confined to that temporary directory and are not part of this report.
- Vector configuration was explicitly disabled. No LLM generation was performed;
  the available llama.cpp instance at localhost 9099 was only probed for models.
- The interrupted turn stopped the original services. They were restarted with
  the same data; no CI jobs or unrelated processes were cancelled.

## Matrix

| Scenario | Result | Evidence / Qualification |
| --- | --- | --- |
| Real registration, login and organization membership | Pass | Three distinct subjects authenticated; membership created via APIs. Login refreshed after organization creation to obtain its JWT scope. |
| Markdown ingestion and chunking | Pass | `/media/add` returned Success, persisted Media ID and extracted synthetic HARBOR-COPPER-731 facts. |
| Source and note attachment | Pass | Owner workspace sources/status reports text extraction, FTS and citation readiness; note persisted. |
| Fresh default-role recipient access | Pass after permission repair | List now enforces `sharing.read`, matching recipient operations. A denied recipient sees administrator-actionable copy and no original share actions; no automatic grants added. |
| Permission revoke and restore | Pass live | Existing recipient role removed via admin API; reload returns typed 403 and hides Open/Clone. Re-grant plus Try again returns 200 and restores actions in the same session. |
| Configured recipient clone admission | Pass after timestamp repair | Browser POST returns 202, `status=queued`, progress 0, `result=null`; not enqueue-time success. |
| Reload during queued copy | Pass after timestamp repair | One clone POST followed by polling GETs; reload recovers operation. No browser page errors. |
| Real worker and terminal UI | Pass | Copy ready, 1/1 sources, 1/1 notes, 0/0 outputs. Vector search explicitly not configured. |
| Open copy target | **Fail, unresolved** | Navigation uses the correct clone UUID, but a fresh browser context loads an empty New Research workspace (`84e2b32e-4048-44e9-b3ef-22a8e63c2426`) instead. Route query is not consumed by ordinary workspace initialization; no clone-target request appeared before inspection. |
| Ordinary recipient connection state | Pass after authenticated-probe repair | JWT connection check now uses `/api/v1/users/me`, returns 200 and displays Connected without any `/health/live` request or operator grants. Invalid-session 401/403 behavior remains covered by regressions. |
| Recipient-owned copied content and note | Pass | Actual workspace/source/note GETs; source Media ID 3, note contains sentinel. |
| Source owner cannot access recipient target | Pass | Owner target GET returns 404. |
| Unrelated user cannot access target or operation | Pass | Outsider target and clone-operation GETs return 404 despite equal sharing.read grant. |
| Real workspace FTS retrieval | Pass after FTS repair | `/rag/search`, fts mode, returns copied document ID 3 with the 37 percent/42000 dollar facts. |
| Citation evidence | Pass for API evidence | Chunk citation points to recipient document ID 3 and includes source snippet. Academic citation currently contains internal `tldw-clone://` URL; original-source provenance presentation needs assessment. |
| Grounded model answer and citation navigation in UI | Blocked | Wrong target initialization still prevents the intended UI flow. False disconnected state is repaired. Do not infer grounded-answer success from retrieval or connectivity. |
| Same-key replay after lost response; simultaneous tabs | Not certified live | Prior UI fixture tests and backend tests are not substitutes. |
| Archive replay, revocation, retry | Not certified live | Integration coverage exists; actual live scenarios remain. |
| Interrupted/fatal work, hidden staging and eventual cleanup | Not certified live | SQLite Jobs acceptance and transactional regression tests cover selected paths, not the full live fault matrix. |
| PostgreSQL runtime parity | Blocked | Repository fixture reports Postgres not reachable. |

Latest successful operation: `ccd4242d-6290-4179-a12f-5bcd9e8418b6`.
Local artifacts: `browser-evidence.json`, `clone-result.json`,
`copy-verification.json`, `shared-progress.png`, `shared-complete.png`.
Only the latest successful run is retained at those filenames. Earlier failed
observations are recorded here and in Backlog notes; do not interpret overwritten
artifact names as historical evidence.

Permission evidence: `permission-browser-evidence.json`, `permission-denied.png`,
`permission-restored.png`. Target inspection: `target-loaded-evidence.json`,
`target-loaded.png`. The target screenshot is failure evidence, not a successful
copy-open acceptance screenshot. The first permission script's simulated
lifecycle transition did not trigger a refetch; the final run uses a normal reload.
Cached-data suppression is separately covered by the component regression.

JWT connection evidence: `jwt-connection-evidence.json`, `jwt-connected.png`.
This run uses the same ordinary recipient and real backend. It verifies Connected
and no operator-health request; it still renders the wrong empty workspace and
has no selected chat model. It is not evidence of grounded chat completion.

## Repairs

1. `shared_workspace_clone_operations._timestamp` previously forwarded SQLite
   `YYYY-MM-DD HH:MM:SS` timestamps, rejected by the strict client ISO schema.
   Normalize SQLite and PostgreSQL values to timezone-bearing UTC ISO and reject
   malformed timestamps. The regression failed before the change; 49 operation
   tests passed afterward.
2. `CloneSnapshotRepository.confirm_operation_owned_clone_media` activated copied
   Media without maintaining SQLite FTS. Actual RAG returned no results despite
   `text_search=ready`. Reuse `_update_fts_media` inside the publication transaction
   after ownership/hash verification. Regression proves pending invisibility,
   post-publication search and publication rollback on index failure. The fix
   covers newly published copies, not repair of already-published unindexed copies.
3. The Shared With Me list checked only rate limits while opening/cloning required
   `sharing.read`. Reuse the recipient read dependency, show typed denial in the
   existing error slot, and hide stale original-share actions after denial. Keep
   completed recipient-owned copy navigation available. Documentation now explains
   explicit role provisioning without broadening customized authorization.
4. JWT connection checks probed an operator-only health route and mistook missing
   `system.logs` for a disconnected server. Reuse `/api/v1/users/me` for multi-user
   sessions, following the existing cookie-session check. Keep single-user key
   behavior unchanged. Regression cases for 200/401/403 prove that valid ordinary
   sessions connect and invalid sessions fail closed without liveness fallback.

## Verification

- Six-file backend run: **174 passed, 21 skipped**. Seventeen skips were unavailable
  PostgreSQL Media DB cases and four were the default Jobs test gate.
- Explicit `RUN_JOBS=1` clone acceptance run: **4 passed, 4 skipped**; skips are the
  PostgreSQL variants. SQLite publication/archive replay, revocation, fatal
  publication and interrupted-finalization tests executed.
- Bandit over both changed production Python modules: no findings.
- Permission follow-up: **105 backend tests passed** across sharing endpoints,
  sharing integration and recipient endpoints; **27 UI/hook tests passed** across
  Shared With Me states, clone UI and sharing auth. Ruff and Bandit on the touched
  endpoint passed; scoped ESLint passed. Both primary regressions failed before
  their fixes (backend list bypass, UI generic denial with cached actions).
- Broader follow-up: **131 tests passed in 10 clone UI/service/hook files**;
  clone operations, Jobs acceptance and Media snapshot regression run with
  `RUN_JOBS=1`: **120 passed, 21 skipped**. PostgreSQL remains unavailable.
  ESLint reported its existing custom-pages-directory configuration warning;
  no file-level findings. All task-owned runtime processes were stopped afterward.
- Connection follow-up: all three new cases failed before the fix; **73 tests
  passed** across connection, persona, clone UI and clone hooks afterward. ESLint
  reported zero errors and 14 pre-existing `any` warnings plus the existing
  pages-directory configuration warning. No Python code changed in this follow-up.
- Independent static review: no actionable findings; PostgreSQL execution remains
  unverified. Existing warnings and pytest temporary-directory cleanup warnings
  are recorded in the local logs, not described as a warning-free run.

## Next Actions

Current sequence (2026-09-20): archive/restore is implemented and scoped-verified;
owned deletion has a scoped-verified backend foundation following design review.
See `../superpowers/plans/2026-09-20-owned-workspace-deletion.md` for atomicity,
writer fencing, tombstone recovery, sharing cleanup and UI acceptance gates.
The existing unmerged .49/.50 patch is also undergoing a new independent review.
The owned route remains disabled and the complete live clone matrix is not
certified. No PR exists for `codex/shared-workspace-clone-ux` at this checkpoint.
Fetched dev has advanced 446 commits; no rebase of uncommitted work occurred.

Earlier implementation checkpoint: the canonical owned metadata GET/PATCH context
and backend expected-user guards are ready for UI integration. Tests cover frozen
identity, version conflicts, immutable submitted settings and invalid receipts;
889 tests pass in the broader 35-file regression run, plus 37 domain tests.
The direct metadata HTTP verification below passed against the real backend.
Next: connect Header rename with retained draft/base version, scoped receipts and
explicit conflict recovery, then the remaining settings/source/artifact/streaming
boundaries. The owned route and full CDP acceptance remain pending.

Earlier implementation checkpoint: canonical owned QuickNotes now uses associated
notes and pinned versioned saves, preserves newer drafts and both conflict
revisions, and durably blocks duplicate retries after ambiguous creates. Scoped
Clear/Undo and stale legacy/owned callback guards are covered. Independent review
cleared the final nine-entry extension locale mirror fix; 859 tests passed in the
final 35-file run, including both localization contracts. The broader playground
locale parity test still has the pre-existing `composer_rolePlaySaveError`
omission, independently confirmed in HEAD. Scoped TypeScript and lint passed (five existing
QuickNotes warnings plus the pages-path configuration notice).
The direct-HTTP verification below is real backend evidence, not live editor or
hosted proxy acceptance. The owned route remains unwired. Next are the remaining
Header/source/artifact/sharing/transfer/streaming request and completion boundaries,
then route wiring and the full CDP matrix. TASK-12020.50 remains In Progress.

Earlier implementation checkpoint (owned-opening Stage 3 mount safeguards, still
In Progress): owned main-page opening skips legacy reconciliation/migration/prefill/
initialization. Main search reads only canonical associated notes; status failures
do not fall back to generic media readiness. Scoped composer drafts and audio
settings survive opening without normalization/cache writes. Same-ID account
changes invalidate saved-view state, callbacks and retries. Review fixes preserve
unsent drafts, block duplicate pending sends, and fence saved-view scope lifetimes.

- **747/747 tests passed across 31 files**; scoped TypeScript passed; ESLint zero
  errors with 18 existing warnings and the pages-path notice. The existing route
  test still emits its i18next notice. Whitespace check passed. No Python edits or
  Bandit applicability in this checkpoint.
- Independent review cleared the draft-loss, duplicate-send and saved-view findings
  after RED/GREEN regressions. The two previously baseline-reproduced StudioPane
  mock-arity failures below remain outside this focused green set.
- The owned editor route remains unwired. Next are canonical QuickNotes reads and
  versioned edits, request-pinned mutation/read dispatch, and the remaining
  Header/source/artifact/sharing/transfer/streaming callback boundaries. Then wire
  the route and resume the real backend/WebUI/CDP acceptance matrix.
- This is component/store verification, not new live evidence. No acceptance row
  is promoted and the task remains In Progress. See the owned-opening implementation
  plan for the precise checkpoint boundaries.

Earlier implementation checkpoint (owned-opening Stage 2 complete): actual Zustand
activation now captures the latest outgoing draft, installs a complete owned
bundle in one transition, and prevents owned content from entering legacy saved
lists/snapshots/chat caches. Account invalidation and late hydration are fenced;
failed draft writes remain recoverable and visible as reload risk across switches
and local rehydration. Ingestion status projections do not become pending edits.

- **312/312 scoped workspace tests passed in 19 files**, including 34 new store
  integration cases. Scoped TypeScript passed; ESLint reported no errors and 16
  existing unused-import warnings, plus the existing pages-path warning.
- Independent review findings were reproduced with tests and fixed. Final review
  reports no remaining Stage 2 blockers. No Python edits/Bandit applicability.
- The additional StudioPane suite remains **25 passed, 2 failed** on both current
  worktree and exact HEAD baseline. Both KittenTTS voice-catalog assertions expect
  one argument while production passes an additional `undefined`. Evidence is in
  `/tmp/tldw-owned-studio-{current,baseline}.json`; no unrelated audio edits made.
- Route mounting, account/auth event subscriptions, note API UI, explicit writes,
  and composer wiring remain Stage 3. No live backend/WebUI/CDP run was performed
  in this checkpoint and no acceptance row is promoted.

Implementation checkpoint (owned-opening Stage 2 foundation): scoped, versioned
draft persistence and pure activation preparation are implemented. Failed writes
retain detached in-memory recovery; scope mismatches, stale attempts, and dirty
note conflicts cannot silently activate. These helpers do not yet perform a
Zustand transition or change the route. Stage 2 remains In Progress.

- The previously reproduced split-storage failure is repaired by spying on
  JSDOM's `Storage.prototype.setItem` and restoring mocks, retaining all changed-key
  assertions. Instance assignments had been intercepted by the storage proxy.
- Last code verification: **278 passed in 18 files**, including **39** new
  state/draft cases, with Node native Web Storage disabled. Scoped TypeScript
  passed. ESLint: zero errors, one existing `any` warning in the split-storage
  test and the existing pages-path configuration warning.
- Independent foundation review found no actionable defects. Both suggested
  recovery coverage additions pass: newer failed-write recovery overrides older
  durable content, and failed deletion retains failed-write memory recovery.
- No Python changes in this checkpoint; Bandit is not applicable. No live
  backend/WebUI/CDP acceptance was performed and no acceptance row is promoted.
- Disk-full recovery: with user approval, removed this stopped task's ignored
  1.1 GiB Next.js build cache and three temporary dependency symlinks. Shared
  dependencies, other workspaces, and raw acceptance evidence were preserved.

Implementation checkpoint (owned-opening Stage 1): the complete read-only loader
is implemented in `store/workspace-api.ts`, with 48 loader cases plus preserved
mapper/optimistic-update coverage. Metadata and all required collections validate
before a bundle is returned; failure never becomes an empty collection. Cancellation
rejects promptly and late results cannot complete the load. Independent review's
assistant-default status-relation finding was fixed with RED/GREEN tests and cleared
on re-review. Route activation is not wired and no new live acceptance is claimed.

- Final focused verification: **121 passed** in five loader/mapping/domain files.
- Broader workspace verification: **238 passed, 1 failed** in 17 files using
  `NODE_OPTIONS=--no-experimental-webstorage` on Node v26.0.0. The failure is
  `workspace.split-storage.test.ts:155`, whose instance-level storage spy observes
  no writes. Reproduced against tracked TypeScript loaded from exact baseline
  HEAD through `/tmp/tldw-owned-loader-baseline.config.ts`: **66 passed, 1 failed**
  across the three storage files. The baseline code and tests were not edited.
- Without that Node option, five quota tests fail because native Web Storage does
  not use the JSDOM `Storage.prototype` hooks. The quota files pass unchanged with
  native Web Storage disabled. This does not make the full suite green: the separate
  instance-spy failure remains for the persistence-stage investigation.
- Scoped TypeScript check passed using `/tmp/tldw-owned-loader-tsconfig.json` with
  the project options and installed Node types. ESLint: zero errors, three existing
  `any` warnings in the optimistic-update helper, existing pages-path warning.
- No Python changes in this checkpoint; Bandit is not applicable to this slice.
  No backend/WebUI/Chrome process was started and no fixture test is presented as
  evidence of live target opening or grounded chat.

Design checkpoint: the five target-opening review findings are addressed in the
[revised spec](../superpowers/specs/2026-09-13-owned-workspace-opening-design.md)
and [test-first plan](../superpowers/plans/2026-09-13-owned-workspace-opening.md).
These are design corrections, not implemented or live-verified fixes. They require
read-only readiness, draft-preserving atomic activation, canonical workspace notes,
scope isolation after activation as well as during requests, and complete required
loads that cannot interpret failed reads as empty collections. No acceptance row
changes to passing based on this documentation update.

1. Fix clone-target route initialization so an explicit server UUID is hydrated,
   never replaced by an automatically created empty workspace. The local-only
   `switchWorkspace` cannot activate an unknown server copy. Load and authorize
   the target before mounting the research pane and its reconciliation writes;
   fail closed on inaccessible targets and preserve the previous local snapshot.
   Recheck fresh-context open, reload, route changes and denied targets in CDP.
2. Decide supported recovery for copies published before the FTS repair; avoid
   falsely treating those existing records as searchable.
3. Complete actual target-page, grounded chat, multi-tab/lost-response, archive,
   revocation and failure-cleanup scenarios. Use the real provider for generation.
4. Run PostgreSQL fixtures once a supported database is available, then update the
   matrix and task status. Do not close TASK-12020.50 before the remaining gates.

### Canonical Notes API Preparation

The owned notes transport and optional expected-user guards now have unit/API
coverage (782 frontend regression tests and 105 backend API/route tests passed).
This is preparation for QuickNotes integration, not a passing live acceptance row.
No backend/WebUI/CDP walkthrough was run for this checkpoint.

Add to the pending editor walkthrough: verify real hosted proxy forwarding of
`X-TLDW-Expected-User-ID`, switch cookie accounts between scope capture and save,
and confirm 412 with no notes write. Also test version conflicts, continued typing
while saving, create-response loss without automatic retry, and workspace/account
changes while loading or saving. The opening route stays disabled until its
remaining mutation paths and editor lifecycle are safe.

### Canonical Notes Direct-HTTP Verification

Passed a subsequent 13-request walkthrough against the actual FastAPI backend on
`127.0.0.1:18095`, using the existing isolated multi-user fixture under
`/tmp/tldw-clone-uat-jIXQx6`. Auth, per-user content, Jobs, and audit paths were
bound to that fixture; clone workers were disabled for this notes-only run. The
ordinary recipient authenticated afresh without modifying stored credentials.

| Check | Result |
| --- | --- |
| Authenticate and load canonical profile/notes | 200; one original note |
| Create associated note | 201; workspace ID and keywords match |
| Versioned update, including clearing keywords | 200; version increments and keywords become empty |
| Repeat stale update version | 409; saved content unchanged |
| Wrong expected-user assertion on list/create/update/delete | 412 on all four; scope-changed code |
| Existing client read without the optional assertion | 200; only the expected temporary note was added |
| Delete temporary note and compare notebook to initial state | 204/200; original list restored exactly |

Revision: `c70387f496d82fcee92926bf3715bf5cd240ba88` plus the uncommitted notes
guard. SHA-256 of `endpoints/workspaces.py`:
`f48d89d3034fa49df8a3dc5e3b4cc9f2c7a1e8919dfb597be3c8a53e640ceee7`.
SHA-256 of `API_Deps/auth_deps.py`:
`4d50114884d7dfc7bf1da488914573c6003c8ff737e5f5186d71a03d9be10bf8`.
The task-owned backend completed shutdown after the run. Redacted request evidence
is in `owned-notes-http-evidence.json` within the fixture; the probe and backend
log are `owned-notes-http.mjs` and `owned-notes-backend.log`.

This verifies the direct HTTP endpoint, not browser cookie switching, hosted proxy
forwarding, the QuickNotes component, or full clone acceptance. Those rows remain
pending until the corresponding real WebUI/CDP walkthrough.

### Canonical Metadata Direct-HTTP Verification

The real FastAPI backend ran on `127.0.0.1:18095` using only the existing isolated
multi-user fixture under `/tmp/tldw-clone-uat-jIXQx6`. The clone worker was disabled;
no WebUI, browser mock or ordinary user database was used for this checkpoint.

Fourteen requests passed: recipient login/profile; temporary workspace setup;
asserted GET and successful versioned PATCH; stale-version 409; asserted account
mismatch GET/PATCH 412 with no-store; unasserted legacy GET/PATCH compatibility;
missing-target PATCH 409 followed by GET 404; and temporary-workspace DELETE then
GET 404. Successful updates preserved unrelated policy and explicit null/empty
fields. Rejected requests left the confirmed name/version unchanged.

The first probe incorrectly expected missing-target PATCH 404. Inspection of
`CharactersRAGDB.update_workspace` confirmed the pre-existing ConflictError/409
contract. Its temporary workspace was cleaned up by the failure handler. The
corrected run explicitly verifies no creation with a subsequent GET 404; no
production status behavior was changed to satisfy the probe. Initial evidence is
retained separately as `owned-metadata-http-initial-evidence.json`.

Revision: `c70387f496d82fcee92926bf3715bf5cd240ba88` plus the uncommitted guards.
SHA-256 of `endpoints/workspaces.py`:
`df495e5791f3c2b63fea644d251a12e483d868d02c1dd51050ae61c15f6c05e3`.
Probe/evidence/log: `owned-metadata-http.mjs`,
`owned-metadata-http-evidence.json`, and `owned-metadata-backend.log` in the fixture.
The task-owned backend shut down cleanly after verification.

This does not certify Header rename, hosted cookie/proxy handling or the owned
route. Those require the pending UI integration and real CDP walkthrough.

### Owned Header Rename Acceptance Boundary

The Header rename integration adds unit/component verification for scoped draft
recovery, versioned PATCH, explicit conflict inspection, and late callback guards.
The combined regression run passes 1047 tests in 40 files, including 31 new
hook/owned-Header cases and 69 unchanged legacy-Header cases. Scoped TypeScript,
formatting and independent reviews pass; ESLint reports no errors and one existing
Header warning. The final log is `/tmp/tldw-owned-rename-combined-final.log`.
It does not yet change the live acceptance status above: the owned route remains
disabled while other mutation paths are being bound to canonical request contexts.

The eventual real backend/WebUI/CDP walkthrough must cover successful rename and
reload, another session's conflicting rename, failed-save recovery when the server
version is unchanged, explicit server-name acceptance, user-confirmed draft retry,
storage failure, account/target changes during requests, and cancellation before
versus after dispatch. Component tests and the earlier direct-HTTP run are separate
evidence, not substitutes for those end-to-end cases.

### Owned Default-Assistant Direct-HTTP Verification

The default-assistant checkpoint passed 16 requests against the actual FastAPI
application with real single-user API-key authentication and isolated SQLite
databases. The temporary server bound a loopback port and was stopped after the
run. Lifespan startup was disabled to omit background orchestration; authentication
and database dependencies were not mocked. This is not a full-startup, multi-user,
PostgreSQL or browser acceptance result.

| Check | Result |
| --- | --- |
| Unauthenticated read-only Persona catalog | 401 |
| Repeated `ensure_default=false` on an empty catalog | 200 with empty lists; no default created |
| Wrong expected user on the legacy, create-capable catalog | 412/no-store; subsequent read-only catalog remains empty |
| Legacy catalog without the new query/header | 200; existing default-bootstrap behavior retained |
| Set read-write default without confirmation | 422 |
| Set read-write default with explicit confirmation and current version | 200; version advances |
| Clear with stale version or wrong expected user | 409/412; canonical settings remain unchanged |
| Clear with current version | 200; default is null |
| Delete temporary workspace and verify absence | 204 then 404 |

Evidence: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/tldw-owned-assistant-http-8zjtfm9v/evidence.json`;
the same fixture contains `requests.json` and `backend.log`. Probe:
`/tmp/tldw-owned-assistant-live.py`; successful run log:
`/tmp/tldw-owned-assistant-http-run-2.log`. The first startup attempt failed to
import the bundled `tldw_profile_core`; adding its source directory to the probe's
`PYTHONPATH` fixed the environment without repository/dependency edits.

Independent focused backend verification passed 67 tests (9 existing warnings,
plus pytest temporary-directory cleanup warnings). Bandit reported zero findings
for the changed Persona endpoint. OpenAPI fingerprint verification passed with
2094 paths and 3203 schemas. Logs:
`/tmp/tldw-assistant-parent-{backend,openapi}.log` and
`/tmp/tldw-assistant-parent-bandit.json`.

The owned route remains disabled. Real WebUI/CDP acceptance must still cover
selection and clearing, failed catalog with successful metadata recovery,
read-write confirmation after reopening/conflict review, account changes while
requests are pending, and durable reload. No live acceptance row is promoted by
component tests or this direct-HTTP probe alone.

The component/service regression run passed 1117 tests across 42 files, including
33 assistant-hook and 10 owned-assistant Header cases alongside 69 legacy Header
tests. Independent foundation and modal reviews found no actionable issues.
Scoped UI/store TypeScript and focused formatting pass. ESLint has zero errors
and the existing Header unused-value warning/pages-path notice. Combined log:
`/tmp/tldw-owned-assistant-combined-final.log`. These are separate from the pending
real browser checks above.

### Owned Banner Text Direct-HTTP Verification

Thirteen requests passed against the real FastAPI application using API-key
authentication and isolated SQLite databases. The probe creates a temporary
workspace, sets original banner/color/audio metadata, saves new title/subtitle,
reloads them, rejects a stale reset (409) and wrong-account reset (412/no-store),
checks rejected writes did not change the version/content, resets title/subtitle,
reloads empty text with the original color/audio/name intact, and deletes the
workspace (204 followed by GET 404). Unauthenticated GET returns 401.

Evidence: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/tldw-owned-banner-http-6t59yyhm/evidence.json`;
the fixture also contains request evidence and the backend log. Probe:
`/tmp/tldw-owned-banner-live.py`; run log: `/tmp/tldw-owned-banner-http.log`.
The first attempt could not bind loopback inside the sandbox; the authorized
retry succeeded. No authentication or DB overrides were used. Lifespan startup
was off; this does not certify background startup, PostgreSQL, hosted cookies or
WebUI/CDP behavior. The task-owned backend is stopped.

Banner images remain outside the server metadata contract. The approved UI slice
disables owned image uploads rather than claiming browser-only image persistence
is a successful server save. Legacy image behavior remains separate. Browser
acceptance still needs save/reload, confirmed reset, conflicts, draft recovery,
scope switches and delayed confirmation/image-processing callbacks.

The implemented text-only modal passes 1165 regression tests in 44 files. A
post-review focused rerun passes 103 tests (25 banner-hook, 9 owned-banner Header,
69 legacy Header). Independent foundation/UI reviews are clear, including the
small legacy callback identity refinement. Scoped UI/store TypeScript and focused
formatting pass; ESLint has zero errors with the existing Header warning and
pages-path notice. Regression logs are `/tmp/tldw-owned-banner-combined-final.log`
and `/tmp/tldw-owned-banner-parent-postreview.log`. These results do not promote
the pending real WebUI/CDP rows. The owned route is still disabled.
## Owned Archive/Restore Verification

Approved bounded follow-up to owned Header metadata editing: archive through a
captured, account-pinned versioned PATCH; preserve drafts; restore through the
canonical Workspaces manager. No local snapshot Undo for owned workspaces, no
automatic write retry, and no route aliases or redirects.

Real backend evidence: `/tmp/tldw-owned-archive-http.log` and
`/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/tldw-owned-archive-http-xvosaef3/evidence.json`.
The isolated API-key/SQLite uvicorn probe passed **19 HTTP requests**, including
archive/restore, archived/active directory visibility, stale-version and wrong-user
rejections, context/list expected-user checks, and unchanged note/banner/audio
content. The server stopped in the probe's cleanup. Lifespan was disabled; this
does not certify full startup, JWT/hosted-cookie deployments, PostgreSQL or browser
interaction. One initial probe expectation was corrected from 200 to the existing
201 response for note creation before the successful complete run.

Foundation verification: 293 service/store/loader tests and 30 backend tests pass;
Bandit finds no endpoint issues. Review added seven failing cases for unrelated
pending-draft isolation and strict directory projections; 253 tests pass after
fixes. Directory details now expose only validated fields consumed by the manager,
not an unchecked full context. Final parent verification passes **1244 tests in
46 workspace files plus 84 tests in five directory files: 1328 total**. Scoped
TypeScript passes, including manager/root-panel and actual workspace-shortcut
tests. Same-account revalidation preserves drafts and selection; actual account
changes clear the retained view. Suspended root panels stop scheduled polling and
ignore retired async results. Archive storage recovery cannot be bypassed by
workspace shortcuts; Tab/Enter interaction remains available.

All scoped independent reviews and rereviews are clear. A new root lifetime lint
warning was fixed without suppression. ESLint retains only the prior Header
unused-value warning and shared pages-path notice. The full design-system checker
still reports existing repository findings; a comparison against HEAD confirms
zero introduced findings in the touched workspace screens (two existing Header
labels remain). No new baseline exception was added. OpenAPI fingerprint/type
generation and fingerprint comparison pass.

Final logs: `/tmp/tldw-owned-archive-regression-final.log`,
`/tmp/tldw-owned-archive-directory-postreview.log`,
`/tmp/tldw-owned-archive-tsc-postreview.log`,
`/tmp/tldw-owned-archive-eslint-final.log`,
`/tmp/tldw-owned-archive-design-system-delta.log`, and
`/tmp/tldw-owned-archive-openapi-types.log`.

The owned route is still disabled. Delete/duplicate and remaining mutation
boundaries must precede full WebUI/CDP acceptance. No full-functionality claim is
made by this checkpoint.

## Deletion Foundation And Unmerged Review (2026-09-20)

The deletion design now distinguishes one atomic local soft-delete transaction
from concurrent-writer fencing, cross-database sharing cleanup and the owned UI.
The local foundation adds an optional version precondition, expected-user
validation, owner-only minimal tombstone reads, and off-event-loop execution.
Worker-thread connection cleanup must finish with the operation, including after
request cancellation. Deletion-status responses must remain non-cacheable on
dependency errors as well as successful reads. Existing versionless DELETE
clients remain compatible; the future owned UI must send its observed version.

This does not implement secure erasure, child-writer fencing, sharing-cleanup
recovery or the deletion UI. Notes/artifacts/source associations remain retained
behind the tombstoned parent; library media, native files, Git roots, sandbox
volumes and independent copies are not erased. See
`../superpowers/plans/2026-09-20-owned-workspace-deletion.md` for stage status and
final verification evidence.

Final foundation verification: 186 backend tests passed; 16 PostgreSQL cases
skipped because the fixture was unavailable. The real isolated API-key/SQLite
backend passed 23 HTTP checks and stopped afterward. Production Bandit has zero
findings, OpenAPI fingerprint matches, and the final independent wrapper review
passes. These checks do not certify live PostgreSQL, concurrent writers,
JWT/cookie deployments, or WebUI/CDP deletion behavior.

The broader review of all earlier unmerged .49/.50 changes found six reproducible
open defects: prehydration persistence loss, unbound cookie-account bundle reads,
cross-window draft overwrite, stale clone scope after a cross-tab configuration
change, unrelated draft failures blocking note creation, and legacy JWT readiness
probing. The existing frontend regression run still passes 1328 tests in 51 files;
these results do not cover or negate the counterexamples. Full findings and
corrections: `workspace-unmerged-review-2026-09-20.md`.

This review supersedes earlier clean-review statements for merge readiness.
The owned route remains disabled and TASK-12020.50 remains In Progress. The dirty
worktree has not been rebased onto the fetched dev, which is 446 commits ahead;
newer connection-ownership machinery requires integration and fresh verification.

## P1 Review Remediation (2026-09-20)

R1-R4 from the unmerged-stream review are now corrected in the working tree:
prehydration persistence, principal-bound bundle reads, concurrent/mixed-version
draft preservation, and captured clone transport with account-safe recovery.
Review also fixed uncertain-key loss across auth errors, including two-tab 403
ordering and explicit CSRF refresh. All scoped rereviews are clear.

Final verification: 1510 frontend tests in 53 files; 141 selected backend tests;
23 real isolated API-key/SQLite HTTP assertions; seven actual Chromium/CDP
localStorage/Web Locks cases; scoped TypeScript, OpenAPI and production Bandit
checks passed. Existing-only lint warnings and unrelated design-system findings
remain. Detailed evidence and limits are in
`workspace-unmerged-review-2026-09-20.md`.

This is not full WebUI, JWT/cookie-switch, PostgreSQL or latest-dev certification.
R5/R6 were open at this checkpoint, as were deletion integration and draft
conflict-resolution UI; see the later P2 checkpoint below.
The owned route remains disabled and TASK-12020.50 remains In Progress. No commit,
push, rebase or CI cancellation was performed in this remediation pass.

## P2 Review Remediation (2026-09-20)

R5/R6 are corrected: current-target note durability is independent of unrelated
pending drafts, and session readiness uses the canonical authenticated identity
profile. Aggregate draft-loss warnings and current-target failure/conflict guards
are preserved. Bearer/cookie 401/403 responses remain auth failures.

Verification: 1521 frontend tests in 54 files passed; scoped TypeScript passes;
ESLint has zero errors and 14 existing connection-test warnings. Independent
scoped reviews found no actionable issues. Eighteen real HTTP requests passed
against isolated JWT and cookie deployments with legacy user endpoints disabled,
covering canonical success, legacy 410, auth denial, expired JWT, revoked sessions
and forbidden query parameters. Both backend processes stopped afterward.
Evidence, harness corrections and exact limits are recorded in
`workspace-unmerged-review-2026-09-20.md`.

These checks do not certify full WebUI/CDP account changes, application startup,
PostgreSQL or current-dev integration. Deletion completion, draft-conflict UI and
the overall live acceptance matrix remain open. TASK-12020.50 remains In Progress,
the owned route remains disabled, and all existing work remains uncommitted.

## Current-Dev Integration Follow-Up (2026-09-20)

The next checkpoint is recorded in `workspace-dev-integration-2026-09-20.md`.
Work continues in the separate `codex/shared-workspace-dev-integration` worktree
at fetched dev `d72b1d2850e`; the original dirty checkpoint remains preserved.
Upstream authenticated session readiness replaces the earlier R6 profile probe
only for global connectivity. Owned workspace identity authorization is unchanged.

Real PostgreSQL fixtures now exercise the integrated deletion and operation-scope
contracts, exposing two issues fixed in this pass: premature request-checkout
cleanup and missing message-deletion sync events. The final expanded backend run
passed 262 tests with no skips; previous PostgreSQL-unavailable skips are not
being treated as evidence of parity. Full startup, browser/CDP workflows, deletion
writer fencing/cleanup/UI, and the remaining acceptance matrix are still open.
