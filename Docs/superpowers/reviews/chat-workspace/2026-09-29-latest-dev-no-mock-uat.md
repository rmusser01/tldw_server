# Chat Workspace Latest-Dev Verification Checkpoint

Tracking: TASK-13398 and subtasks. **Full UAT is not accepted.**
Current dev's unrelated Buddy TASK-13396 forced a tracking-ID migration; original
records are archived, child suffixes preserved, and historical IDs below remain
original evidence attribution rather than references to the Buddy task.

## Candidate And Preservation

- Active branch: `codex/chat-workspace-a11y` in `.worktrees/chat-workspace-a11y`.
- Current integrated baseline: `03043d1c10cbdbfa53945c0641d90e4e37836754`.
  Latest integration from `955b1d9626` used autostash `2199ecb866`, applied
  without conflicts. All eight frozen retain-copies files remain byte-identical.
  The pre-integration tracked patch has SHA-256
  `45f55e27378e0229e78e256d8be2c84e1735c5583fa6204142df174c69b5aa7e`.
- Earlier integration to `607431154cf10129b5d9afa8f9b57d46636466fc`
  advanced from `60006a2fed` during repairs. That fast-forward preserved the entire
  tracked candidate diff byte-for-byte, SHA-256
  `32f07bd463dce9b43599fa67f1413e33a04cb98678504cc5a5a1a5566d0189eb`;
  no upstream path overlapped dirty/untracked candidate files. The index stayed
  empty and the retained stash was not changed.
- Preserved original tracked/untracked candidate: retained stash
  `18ab9c28fd63b438b83ef3ddca75c87f2f9e8d2c`; never dropped.
- Original index was empty. Apply-created staging was cleared without changing
  reconciled tracked content; no unmerged paths or conflict markers remain.
- SQLite insertion-order migration is 74, PostgreSQL 78. Latest upstream
  migrations and native history ownership contracts were retained.
- The earlier `2026-09-29-no-mock-uat.md` is historical, pre-latest-dev evidence,
  not acceptance evidence for this candidate. Its old task IDs collide with newer
  upstream tasks and must not be used to update those upstream tasks.

## Actual Services, No Response Doubles

Fresh isolated profile:
`/private/tmp/chat-workspace-real-uat-latest-dev-20260929`.

Official AuthNZ initialization exited zero; authenticated readiness returned 200
and healthy database status. No old UAT database was copied or schema-stamped.
Only the previously approved credential and actual model assets were reused.
FastAPI runs on loopback 8000, Next on 18089, actual Gemma via llama-server on
9099. The model was briefly stopped for a real outage check and restarted.
No mocked authentication, readiness, model or API responses were used here.

| Actual Check | Evidence / Result |
| --- | --- |
| New workspace and neutral chat | Authenticated official API endpoints; workspace `fc13e91b-8dd9-47a4-9851-880f730fb664`, conversation `7b35c2bc-e613-4c53-ac1f-b785bf8e0d60` |
| Provider offline | Real 502 after user commit; exactly one persisted user row |
| Provider restored / same-ID retries | Actual nonempty Gemma answers; one original user and two distinct assistant attempts |
| New identical-text turn | Separate user ID and assistant; final two user rows, five total rows |
| Changed-content / stale-turn retry | Both 409; canonical message rows unchanged |
| Next SSE, first run under load | First content 13.447 s; total 30.403 s; no Content-Encoding |
| Next SSE, second run | First content 3.040 s; total 14.966 s; no Content-Encoding |
| SSE headers | `Cache-Control: no-cache, no-transform`; incremental chunked response |
| Rejected-retry settings defect, original evidence | Real 409 increased settings_version 2 -> 3; failing evidence retained |
| Rejected-retry settings, after fix/restart | Changed-content and stale-turn 409 in both response modes; settings, version, conversation and rows unchanged |
| Separate image attachment, after fix/restart | Existing valid PNG appended via the actual DB abstraction; text-only retries return 409 in both modes, leaving rows/images/settings unchanged |
| Mixed durable / legacy retry metadata, original evidence | Real API 422; invalid mixed requests remain correctly rejected |
| Production wire projection, after fix/restart | Actual helper removes legacy flags; actual API/Gemma returns 200 for nonstream and streaming retry, one user/three rows, selected model persisted |
| Fresh document persistence | Actual processing-only endpoint followed by official media/add; new media ID 1, no copied DB or synthesized ingestion response |
| Captured-user workspace reads | Metadata/sources/artifacts/notes/preview: five actual 200 responses; ten stale/empty-header 412 responses with no-store |
| Bounded source preview | Actual 80-character preview of 393 characters, truncation true, one real chunk plus excerpt; source/workspace/media IDs match |
| Fresh embeddings | Actual MiniLM job completed through Jobs/Redis worker; Chroma count 1; source FTS/vector/citation readiness all true |
| Fresh hybrid retrieval and generation | Actual source-restricted retrieval and Gemma: 37 seconds/Mira Chen, two citations, no errors, no cache hit |
| Browser rendering / Stop | Not verified: earlier navigation/attachments and current existing-tab read timed out; human recovery requested |

Raw actual evidence is retained in the fresh profile: `durable-api-state.json`,
`durable-api-outage-results.json`, `durable-api-results.json`,
`durable-api-settings-conflict-results.json`, `durable-api-mixed-metadata-results.json`, `stream-timing-results.json` and
`stream-timing-next-results.json`. The conversation response hides neutral
settings; a subsequent GET of its official `/settings` endpoint confirmed the
rejected request replaced the selected model. Existing test records remain.

Fixed-behavior evidence: `durable-api-settings-conflict-fixed-results.json`,
`appended-image-real-api-results.json`, and
`production-projection-real-api-after-backend-restart-results.json`. API restarted
with stable backend production code as PID 65591. The wire check directly executes
the production `prepareChatCompletionRequest` helper with Bun, then sends its
output through the authenticated real API to Gemma. It is not a browser or full
factory UAT claim. Actual llama-server generation timing confirms provider calls.
The attachment diagnostic initially imported the primary checkout and failed at
constructor argument validation before a DB write. Its corrected run pins and
asserts the active worktree module path; both logs are retained.

HTTP 200 from `/chat-workspace` is not proof of successful browser rendering.
Earlier screenshots are not reused as proof for latest-dev behavior. Fresh
grounded RAG, embeddings, Browse, workspace Open, reload restoration and New Chat
have not yet passed browser UAT. The new embeddings and grounded retrieval checks
are actual API/worker/database/model evidence only.

Fresh UI-repair fixture: workspace `bad43a54-fd4d-4dc4-a116-6805d9159083`,
source `7c30412a-1ec2-4a7c-b1cc-d00dca85d34b`, native workspace note 1 and
completed fixture artifact `1768b1d0-b5d1-45d6-b350-22116f19ddd4`. The note/artifact
are explicit test inputs, not model-response doubles. Core policy remains
`general` and the source is selected. Raw results are
`workspace-content-document-ingestion.json`, `workspace-content-document-add.json`,
`workspace-content-real-api-results.json`, `workspace-content-real-embeddings.json`
and `grounded-rag-results.json` in the fresh profile. The first document endpoint
is processing-only; persistence was verified through media/add, not assumed.
The preview comparison excludes only its per-response `generated_at` timestamp;
all content fields match. An initial whitespace-only header was rejected by the
HTTP library before dispatch, so the actual check uses a legal empty header.

Current API PID 32940 and actual embeddings worker PID 20452 use fresh-profile Redis
stream names, leaving historical queues untouched. The historical worker was
already stopped. Redis is real, stub fallback stays disabled, and cached real
MiniLM weights are used. The API restart first hit sandbox permissions: denied
stop left the original process alive; a duplicate startup exited without binding.
The original owned PID was verified and stopped through the permitted escalation
before successful restarts. No unrelated processes were stopped.

## Post-Upgrade Checks

Only FastAPI was upgraded from 0.136.3 to 0.141.1. Existing Starlette 1.2.1 and
Pydantic 2.11.7 were retained. Installed editable-package metadata still declares
the old FastAPI pin; pip also reports four existing ML dependency-floor conflicts.
No full environment consistency is claimed, and no unrelated ML upgrades were
performed. The owned API and embeddings worker were verified and restarted.

- Route helper, router contract, auth ratchet, scoped-token and workspace-content
  regression batch: 68/68, no skips, pytest timeout increased to 600 seconds.
- Actual API-key owner matrix: five 200 reads, ten stale/empty-header 412s.
- Production hydration helper with four actual authenticated network reads:
  source, selection, artifact, canonical note and `general` policy match the real
  fixture. This is not route-gate/browser UAT.
- Actual opaque-cookie owner matrix: five 200 reads, ten 412 rejections, and 401
  after revoking this probe's own session. The first probe minted an HTTPS-only
  cookie against plain HTTP and received 401; the isolated loopback profile now
  sets `SESSION_COOKIE_SECURE=false`, with production defaults and CSRF unchanged.
  The first failed probe's session remains in this test fixture; no global session
  revocation or unrelated cleanup was performed.
- Actual Gemma nonstream and uncompressed stream: both 200, one durable user row,
  two assistant attempts. Changed-content retries return 409 in both modes and
  leave settings, conversation and message rows unchanged. New conversation:
  `695121de-3852-4324-94be-18e775a807a3`.
- Fresh production TypeScript passes with 8 GB heap/no incremental output.
  Bandit across all six touched Python production files has zero findings/errors.
- Final consolidated frontend regression run passes 411/411 in 23 files, zero
  failures/errors/skips. Five intended RED failures preceded the WebClip/undo
  repairs. Follow-up independent static review found no remaining actionable
  findings; the final parent production typecheck also passes.
- Timeouts: browser load/navigation 120 seconds (150-second tool call), actual
  HTTP/hydration helper 300 seconds, pytest 600 seconds, Vitest tests/hooks
  30 seconds. The browser's internal 21-30-second connection cap is unchanged.

Artifacts: `dev607431-route-auth-content.xml`,
`workspace-content-post-upgrade-real-api-results.json`,
`workspace-content-post-upgrade-real-cookie-results.json`,
`workspace-content-live-production-hydration.json`, `dev607431-chat-results.json`
and `bandit-dev607431-candidate.json`. Cookie API acceptance does not prove browser
cookie/account-switch behavior; JWT runtime acceptance remains separate.
Consolidated frontend evidence: `dev607431-frontend-final.xml`. The frozen reviewed
activation artifact is `activation-owner-undo-review.patch`, SHA-256
`899c629a7299706aae98e87e2f3adea567747d503b55227c815b5c58c7b5d244`.

## Regression And Security Checks, Not UAT

- Expanded canonical preview/Console/Page/Rail/transport regressions: 206/206,
  six files, no skips. An inherited hand-copied source-list fixture omitted a
  required field; it now spreads the existing canonical defaults, with assertions
  unchanged. Independent preview static review found no actionable findings.
  It did not run tests or verify browser focus/account switching. Four additional
  media-ID mismatch/ABA cases pass without production changes; invalid responses
  render the validation error, not content or snippets.
- Workspace content-read scope follow-up: ten intended stale/blank-header RED
  failures and ten controls; GREEN preview/sub-resource/CRUD suites 127/127, no
  skips. Exactly five GET routes reuse the existing expected-user dependency.
  Bandit finds zero issues/scan errors; compileall passes. Ruff's four production
  broad-catch findings and one test import-order finding exactly match HEAD.
  Independent frozen-patch static review found no actionable startup/read-guard
  issues; it did not execute tests. Real API-key checks do not prove cookie
  account-switch or browser behavior.
  The final pre-upgrade parent rerun also passes 127/127, including ten rejected
  cases asserting no workspace content read. The local mocker has no spy method;
  the failed harness was replaced with standard-library patch.object wrapping
  the real DB method. A wrong-path invocation collected zero and is not evidence.
  Test import ordering was subsequently normalized; production Ruff retains only
  the four unchanged broad catches. Final read-guard Bandit returns zero.
- Page preview-close wiring (TASK-13396.5): three intended RED failures/twenty
  controls; GREEN 23/23. Browse requires current hydrated workspace/source;
  close/reopen preserves chat mounting and staging, and obsolete close callbacks
  cannot clear a different source/workspace preview. Preview/store boundary
  doubles are regression-only, not UAT.
- Empty-store startup follow-up (TASK-13396.4): three intended RED failures,
  sixteen controls; GREEN Page 19/19, Research Workspace stage3 34/34 and shared
  events 2/2, 55 total. Production TypeScript exits zero; valid root ESLint across
  four files has zero errors/five inherited warnings. An extraction syntax typo
  was corrected before the successful rerun. Initial ignored-file lint output
  is not verification. This TypeScript-only repair adds no Python Bandit scope.
- Approved follow-up, fresh parent runs: loader 107/107 in five files,
  factory/model/transport 126/126 in eleven files, consolidated chat 285/285 in
  ten files. These overlap and must not be added into a unique test count.
- Backend follow-up: worker 119 durable/native-history and six real PostgreSQL
  cases pass; parent independently reran focused SQLite/API 11/11 and actual
  official-fixture PostgreSQL 6/6. No failures or skips in these follow-up runs.
  Twelve new backend cases had eight intended RED failures and four controls.
- New real factory-to-wire regression had eight intended RED failures and seven
  controls, then 15/15 GREEN. It uses actual factory/model/service code with
  boundary doubles, and therefore is regression evidence only.
- Loader follow-up RED covered pending replacement, qualification rejection and
  reset cleanup; final 107/107 GREEN includes existing multi-instance fencing.
- Fresh production TypeScript exits zero. Installed ESLint 9.39.2 checks six
  follow-up files: zero errors, 64 warnings; overlapping file counts are unchanged,
  transport's two warnings match HEAD, and the new wire test has zero warnings.
- Fresh Bandit across six changed Python production files: zero findings/scan
  errors (`bandit-four-repairs-production.json`). Ruff retains one HEAD import
  order finding in `chat.py`; no new findings. Worker compileall passed.
- Earlier latest-dev checkpoint checks, before these bounded follow-ups:
- Parent frontend: 409/409 tests in ten suites, including cancellation lookup
  rejection and late-resolution races, owner-bound history and loader fencing.
  After two test-only cast repairs, affected suites reran: 107/107.
- Production frontend: `bun run typecheck --pretty false`, exit zero.
- Parent lint with installed ESLint 9.39.2 and explicit repository config:
  five files actually linted, zero errors, 64 warnings matching dev counts.
  Initial ignored-file and incompatible cached-ESLint runs are not evidence.
- Transport worker: 733 owned tests and 166 related tests; baseline comparison
  found unchanged test-fixture TypeScript errors and ESLint error/warnings.
- Backend worker: 534 passed, 60 skipped before sandboxed PostgreSQL rerun.
  Dedicated durable PostgreSQL 5/5 used real official isolated AuthNZ databases
  and an actual NOSUPERUSER NOBYPASSRLS role.
- Parent native PostgreSQL rerun outside the network sandbox: 58 passed, one
  intentional SQLite connection-lifetime skip; JUnit and log at
  `/private/tmp/chat_backend_native_pg.xml` and `.log`.
- Backend JUnit artifacts confirmed durable 88/88 and setup 150/150. Scoped
  Bandit returned zero findings/errors; endpoint Bandit also returned zero.
  Fresh parent scan of all six changed Python production files returned zero
  findings/errors: fresh-profile `bandit-latest-production.json`.
- `git diff --check` exits zero. No full-repository test pass is claimed.

## Reviewed Repairs

1. Defer workspace provider/model defaults until durable admission succeeds;
   implemented after human approval. Actual conflict checks preserve settings
   and version; admitted explicit selections still persist.
2. Reject existing durable text anchors with separately appended image rows
   inside the existing admission transaction; implemented and verified on SQLite,
   real PostgreSQL, and the actual API.
3. Shared wire projection now suppresses legacy retry/regeneration flags when
   durable identity exists. Nonstream requests reuse it. Valid user correlation
   metadata and ordinary retry/native history guards remain intact. Actual
   projected requests succeed, but visible frontend Retry is still unverified.
4. Loader busy ownership now captures selection intent before qualification;
   old cleanup cannot clear a pending replacement before debounce. Independent
   review identified a reset leak; a RED test reproduced it and the null-reset
   cleanup fix passed. Final static review found no remaining actionable issues.

Independent final loader, transport and backend reviews inspected frozen artifacts
whose hashes still match the current tracked diffs. Reviews were static, not
independent test executions. No new commit, push, PR publication/merge, issue closure or epic
completion was performed. The broader candidate remains in progress.

## Open Work

1. The human continued the existing-flow startup, Browse preview and server-only
   activation repairs. Startup and preview bounded regressions pass; preview is
   Done for that scope only. Activation TASK-13396.6 also passed bounded regression
   acceptance after repairing the independently found canonical WebClip
   owner-verification bypass and missing undo provenance restoration. A follow-up
   static review found no remaining actionable findings. Parent's first
   17-file rerun passed 199/200; the lone split-storage spy failure was reproduced
   with jsdom storage and repaired by spying on its prototype/restoring the spy;
   both storage modes pass 8/8, with zero lint errors/one unchanged any warning.
   Canonical notes are explicitly view-only; scoped note mutations and unowned
   legacy snapshot migration remain follow-ups. Scoped conversation checkpoint
   persistence and explicit New Chat/restoration precedence remain a separate
   design decision. Preserve RAG/Persona and native H1 history ownership.
2. Browser-control blocker: no current render proof or screenshot was obtained.
   The documented load wait was increased to 120 seconds (150-second tool call),
   but setup aborted around 25 seconds on Emulation.setFocusEmulationEnabled;
   screenshot attachment also failed on that internal connection timeout.
   No documented override for that cap was found; further blind repeats stopped.
   A different recovery attempt created a clean tab 5: about:blank was readable,
   but real navigation hit a fixed Page.navigate timeout around 30 seconds.
   Next logged a 24.3-second compile and HTTP 200; this is not render proof.
   The subsequent extended load wait and captured-console read both failed on
   internal focus setup around 21 seconds. Both failed tabs are retained, and
   the human was asked whether to use Chrome or restart the in-app browser.
   Reading already-bound tab 2 still timed out after code changes. Human asked to
   close/reopen it and report its visible condition; no answer received yet.
   Read-only native Codex fallback was denied by tool safety policy; no bypass
   was attempted. The previously bound UAT tab was retained for recovery.

## Read-Only Epic Re-Audit

Baseline remains `607431154cf10129b5d9afa8f9b57d46636466fc`. GitHub tracker
#1239 remains open; #2033, #2034 and #2036 remain open. Closed #2031, #2032
and #2035 do not establish that their acceptance criteria passed. The browser
smoke suite intercepts API traffic and is regression evidence, not no-mock UAT.

Parent source checks and independent static audits identify these unresolved
items. No implementation or API writes were performed in this audit:

- **Scoped restoration:** Chat Workspace does not perform Research Workspace's
  session save/restore handoff. Its draft is mount-local. Existing
  `WorkspaceChatSession` records lack an owner and history-selection reference;
  `getWorkspaceChatSession` also falls back to a workspace-only legacy key.
  Copying the effects would not establish account ownership or explicit New Chat
  precedence. An existing-store, owner-fenced checkpoint/draft design was
  presented for separate approval. Cached rows must not become mutation authority.
- **Synthetic readiness (#2033):** Page readiness checks only `isConnected` and
  `CONNECTED`; both demo mode and offline bypass can publish those without a live
  health check. Distinguish synthetic connectivity honestly without globally
  changing caller-specific bypass policy. Neither mode counts as UAT readiness.
- **History and request state (#2033/#2034):** Panel runtime propagation omits
  history loading/load errors and nonstream busy/recovery state. Rails can show
  Ready while Send is disabled or server history failed to load.
- **Send contrast (#2034):** White text on `primaryStrong` yields 2.4298:1 in
  Primer dark and 2.6918:1 in Nord dark. Parent ran the production `contrastRatio`
  and `meetsTextContrast` helpers; both fail the existing 4.5:1 AA text threshold.
  This is a token calculation, not rendered contrast acceptance.
- **Preview announcements (#2034):** Shared source-preview loading, unavailable
  and failure transitions lack status/alert/live-region semantics.
- **Transcript headings (#2034):** The page supplies an H1 and message Markdown
  can render another H1. The existing browser heading assertion runs before send,
  so it does not cover populated transcripts.
- **Copy accuracy (#2036):** Guide/tour copies still describe no automatic
  workspace creation, non-activating manager Open, Browse as highlight-only and
  no model-picker exception. These contradict current initialization, canonical
  Open/preview and failed-turn model recovery paths.
- **Failed storage hydration:** `onRehydrateStorage` ignores the error callback
  when state is undefined. Malformed persisted JSON is a statically traced path
  to permanent loading. It has not been reproduced in actual UAT storage and is
  not established as the IAB timeout's cause. Any repair must preserve stored data
  and offer non-destructive recovery rather than initialize over it.

The inspector's global selected-model label is internally consistent even when a
retry uses an override. An active-attempt model display is a separate semantic
choice, not a confirmed wrong-selection defect; do not overwrite global selection.
The bounded status/accessibility/copy/hydration repair batch also awaits approval.
Canonical note CRUD and automatic legacy-owner adoption are not added to this
epic merely to remove previously documented implementation ceilings.

Browser inventory and dialog reads succeeded, but the direct screenshot API still
failed during focus setup despite the extended tool budget. No current render,
Stop, reload, Open, mobile geometry or focus proof was obtained. Browser-choice/
restart input remains pending; both UAT tabs are retained. Current listeners are
API 32940, Next 73244, Gemma 23875 and Redis 64586. The refused connections in
`web.log` predate the current API restart; current API polling returns 200.
This does not establish browser rendering. Full UAT and epic closure remain
unaccepted. This checkpoint changes documentation/tracking only; no new Bandit
scope or code verification claim is introduced.

Verification fixtures may have modified default worktree runtime DB/log files;
there were no pre-run snapshots, so no untouched-data claim is made. Nothing was
cleaned or reverted. Backend follow-up identified hard-coded MCP media/docs paths
that initial fixture lifespans also initialized in the worktree. Final worker
reruns use private MCP paths; first isolated SQLite fixture attempt failed on the
TMPDIR-derived allowlist, and the corrected attempt passed without weakening it.
Parent's initial focused SQLite rerun also lacked those MCP overrides. Future
owned API starts now use a copied MCP config with exactly media/docs DB paths
changed to this UAT profile; all module and guard settings are unchanged. This is
not a claim that every default module path or prior fixture side effect is audited.
The unused native latest-dev worktree completed but could not be archived because
it is protected by a pinned task/workspace; it remains untouched and retained.

## Chrome CDP Follow-Up

The human selected Chrome and explicitly prohibited plugin installation. A
separate real Chrome 154 profile uses loopback CDP 19239 and native persistent
storage. The installed Playwright client drives it over CDP; native CDP was also
used for read-only navigation diagnosis. No plugin, response interception,
authentication/readiness seeding or replacement storage is used for these checks.
Remote dev was freshly rechecked and remains at `607431154cf10129b5d9afa8f9b57d46636466fc`.

Actual findings and current disposition:

- The page renders. Authentication was entered through Settings with Remember
  disabled; native session storage retains it across document navigation. Core
  and RAG health return 200. The initial empty render is not chat acceptance.
- Workspaces list originally followed a slashless 307 to backend port 8000 and
  failed in Chrome. TASK-13396.13 fixes the shared list helper's canonical slash.
  Actual manager list/context requests now return 200 on port 18089 with no
  off-origin redirect. Independent review is clean; focused regressions 49/49.
  Domain lint retains one upstream `no-control-regex` error and 51 upstream
  warnings, exactly matching HEAD. No security or CORS policy was changed.
- Manager Open loaded the real `bad43a54...` workspace, Lumen source, artifact
  and native context through four canonical GETs returning 200. The product then
  automatically treated that canonical cache as legacy, migrated and removed
  its root/index, and wrote a native-workspace tombstone. Navigation reinitialized
  an empty `5eb0e654...` workspace. TASK-13396.14 tracks the confirmed local
  selection/context-loss fix. Canonical server-content deletion is not claimed;
  the pre-existing fixture tombstone remains retained.
- Sending from that empty local UUID returned real POST `/api/v1/chats/` 404
  "Workspace not found" before a model request. TASK-13396.15 now offers Open
  workspaces for local-only/mismatched provenance and reuses the existing complete
  account-qualified activation guard for canonical chat. Chrome rendered the
  recovery action and opened the actual manager. No model response or populated
  heading acceptance is claimed from that failed attempt.
- Through the actual creation dialog, Chrome created server workspace
  `ab1188e3-03fe-48f3-b92f-86343562c568`, "Chrome CDP grounded UAT", with an actual
  PUT 200. Its manager row is visible. Open, ingestion, grounded send and Stop
  remain pending the migration repair; this is not a prefilled/mock workspace.

Scoped review-driven repairs remain in progress. Parent Page recovery UI is
accessible and does not treat a resolved rehydrate promise as success. A raw ST
heading retained overriding aria-level 1; an intended semantic RED case now
passes after omitting that attribute during native heading replacement. Parent
Page/Markdown/hydration batch passed 233/233; Page/activation route batch 76/76;
copy/a11y batch 52/52; these overlap and are regression evidence, not UAT. Parent
production typecheck exits zero; six-file lint has zero errors/four inherited
warnings. Shared activation now supplies a Loading workspace status while pending.

Independent reviews identified pending normal-send scope lookup reporting Ready,
an outstanding persistence write able to overwrite bytes after failed hydration,
and inherited offload read failures being misclassified as success. Implementers
are repairing those validated gaps with focused RED/GREEN tests. Five wider quota
failures were attributed to unchanged upstream test mocks intercepting the wrong
storage prototype; no assertions are weakened to conceal them.

Actual screenshots retained in the fresh profile include settled render, Settings,
workspace redirect failure/fixed manager, local 404 and created workspace. Initial
loading/old-manager screenshots and one failed Research screenshot are not used
as successful activation evidence. Browser waits are 120 seconds, generation
completion waits 300 seconds, and screenshot waits explicitly 120 seconds.
Full no-mock UAT and epic closure remain unaccepted. The written checkpoint design
review remains pending; no owner-fenced checkpoint implementation is claimed.

### Native Workspace And Real Model Follow-Up

After the migration guard was present, actual Chrome Open of `ab1188e3...`
loaded metadata, sources, artifacts, notes, source views and context with 200
responses. The real Research body showed "Server context ready". No migration
writes or tombstone for that workspace were observed. A full document navigation
to Chat Workspace timed out waiting 120 seconds for the composer; subsequent
read-only inspection showed the correct canonical workspace, Ready composer and
native snapshot. This is eventual rendering evidence, not a load-time pass.

The first successful Chrome composer turn created real conversation
`9bbddabf-f8c2-4815-9012-d9517aa094f4` (POST chats 201), read messages/settings
(200), and called the real Gemma completion endpoint (200). The visible answer
was "2 + 2 is 4." under "CDP UAT". The accessibility tree contained one H1,
"Chat Workspace", with the workspace and response headings at H2. The screenshot
`/private/tmp/chat-workspace-real-uat-latest-dev-20260929/chrome-cdp-real-gemma-heading-success.png`
was captured and visually inspected. The earlier `chrome-cdp-real-model-heading.png`
shows the failed local-only 404, not this successful response. An automation wait
for the wrong Stop label delayed the successful-turn inspection; the actual
control is named "Stop generating" and is used for the live interruption check.

The migration guard's parent regression run passed 80/80. These are bounded
acceptances only: grounded browser chat, live Stop persistence, responsive
interaction and owner-qualified reload checkpoints still require their own
evidence. The isolated worktree and retained prior tombstone were not reset.

Live Stop remains unaccepted after three attempts. The first two genuine long
responses completed before the automation click; the second observed a visible
Stop control and real stream traffic but the click timed out. Their canonical
assistant rows contain 21,732 and 9,024 characters and are completed responses,
not interrupted turns. The third failed at pinned OpenAPI capability discovery
before any model call or saved user row. Its UI truthfully displays Send failed,
retains the draft and offers retry. Backend OpenAPI returned 200 in 53 ms; a
subsequent native Chrome public fetch measured 43 ms to headers and 54 ms JSON
parse for 2,104 paths. The precise cause of the original client timeout is not
established. Foreground CDP activation made later idle interactions responsive,
but is not proof of the original timeout cause. Do not call these Stop passes.

After reassessment, the requested longer timeouts are applied through real
Settings, not by changing production defaults or intercepting responses. The
inflated long-response conversation remains retained as failed-attempt evidence;
other acceptance flows continue independently rather than repeating the same
Stop script unchanged.

Parent fresh combined regression verification passed 419/419 across 30 files;
frontend production typecheck exited zero. The new IndexedDB substrate suite is
unit-only, with its already-installed 6.2.5 dependency explicitly declared in the
frontend test manifest/lockfile. No such substrate is used for browser UAT.

### Source And Recovery Browser Evidence

The real My Media dialog added "Lumen Project Field Memo - latest-dev UI repairs"
to the fresh canonical workspace (POST sources 201). Its native source ID is
`a8bcb45d-86ac-40aa-bd1c-0ef37e89edac`, referring to the already genuinely ingested
and embedded media 1. A later full document navigation rendered the composer in
5,781 ms and retained the canonical source. Browse made the actual preview GET
(200), displayed 393 captured characters and three snippets, and Escape returned
focus to the initiating Browse button. Staging showed "Context staged - not sent"
before an explicit Send with staged context.

That send created real conversation `c36bc315-89ab-456a-b106-1914abc4663f` (201),
called real RAG search (200) and Gemma completion (200). After lazy Markdown
settled, the visible answer correctly said 37 seconds and Mira Chen, with opaque
`doc id='0'`/`doc id='1'` references and no links. Factual grounded generation is
verified; citation navigation is not accepted and is under a read-only mapping
audit. The initial `chrome-cdp-real-grounded-answer.png` captures a loading
placeholder, not answer-render acceptance. The separately retained
`chrome-cdp-real-grounded-answer-settled.png` captures the settled answer.

A native-storage fault check used the isolated profile's actual workspace bytes,
not a replacement storage implementation or mocked hydration state. It retained
a mode-0600 backup at
`/private/tmp/chat-workspace-real-uat-latest-dev-20260929/chrome-native-workspace-before-corruption.json`
(SHA-256 `25eb301d9753d12e6f413f62f2955d927e8fb2e1561dca0cf3668742640f4de2`),
wrote malformed root bytes, and navigated the real page. The visible recovery
alert and Retry appeared; the malformed root and all split siblings were
unchanged and the composer count was zero. Restoring the original root bytes and
clicking the actual Retry recovered `ab1188e3...`, Ready state and no new tombstone.
Screenshots record both actual error and successful retry. This accepts native
parser/error/retry behavior, not native IndexedDB pending-write abort timing.

Fresh route/heading regressions additionally pass 52/52 across three actual test
files, and Research stage3 passes 36/36. Earlier commands with unmatched test-path
filters do not imply coverage of those files; the corrected explicit-path runs
provide that evidence. No tests were skipped or assertions weakened.

The separate pinned-capability preparation cancellation gap is source-validated:
the model factory awaits discovery without passing the owning turn's abort
signal. Its bounded signal-propagation repair has been presented for approval;
no implementation or timeout-root-cause claim is made from that observation.

### Bounded Review Revisions And Layout Evidence

Final independent review of the frozen migration and hydration deltas is not
clean. Three migration P1 gaps remain: a post-hash edit can be erased during
awaited cleanup, canonical/target transitions between removals do not revoke
the remaining deletions (with inaccurate partial-deletion reporting), and an
offload pointer arriving after preliminary discovery can evade exclusion.
The hydration P2 gap is a cached abnormal-closed IndexedDB connection that makes
repeated Retry fail until the cache is invalidated. These are incomplete repairs
of inherited candidate hazards, not established new upstream regressions.
TASK-13396.14 and TASK-13396.10 remain In Progress; their existing owners are
revising the bounded fixes with failing regressions and immutable review packets.
The previously recorded native JSON error/retry check remains valid but does not
exercise abnormal IndexedDB close or pending transaction commit timing.

Actual Chrome enabled-Send contrast is 4.8356:1 in both tested dark and light
default skins. Native Tab from the composer reaches Send with visible focus.
At 1440x900 and 390x844, document scroll width equals viewport width, and the
captured staging controls, source title and composer fit without overlap.
Both desktop and mobile screenshots were visually inspected. This is not an
all-skins or screen-reader speech acceptance. The temporary unsent draft and
staging were cleared, and the original theme restored after measurement.

The settled grounded-answer and native recovery error/retry screenshots have
also been visually inspected. Correct factual output still has opaque document
IDs, not navigable citations. A fresh CDP state read confirms the actual Chrome
tab remains Ready with canonical workspace `ab1188e3...`, native storage restored
and no tombstone for that workspace. No plugin was installed and no UAT response
or storage implementation was mocked. Full UAT and epic closure remain blocked
on the explicitly unaccepted flows and pending checkpoint design review.

Remote dev subsequently advanced to `955b1d9626a055ca44336a00d3d4c144949cb00f`.
Its sole file delta is an unrelated TASK-13392 historical Backlog record. A
fast-forward preserved the tracked candidate patch byte-for-byte (SHA-256
`a386320cf0c88a1757d16ce36c9c4ad7846683858996acbee50684c40933ba91`); no stash,
reset or revert was performed. Existing runtime evidence retains its original
production baseline, which this fast-forward did not change.

Fresh Page/route/store activation verification passes 71/71 across three files.
TASK-13396.15 is bounded Done with the independent review and real Chrome
activation/send evidence above. A first ESLint invocation ignored sibling files
and is not evidence; the corrected repository-root invocation with explicit
frontend config checks all four target files, has zero errors and three
inherited test warnings. Guide source/published bytes match. Other open tasks
and full UAT are not made complete by this bounded acceptance.

### Native Preview Network-Failure Check

Chrome CDP native network latency exposed the actual preview loading status
(`aria-live=polite`, `aria-atomic=true`), followed by the real 393-character
response. Native offline conditions produced the actual fetch failure, an
assertive atomic preview alert, and a Retry button outside that alert. No
request interception or response replacement was used.

Preview Retry is not accepted: the shared server-unreachable modal covered it,
and normal clicking timed out after 120 seconds with pointer interception.
The preview later reloaded without a proven Retry click; hot reload is possible,
but the cause is not established. The initial loading/error screenshots caught
modal animation and are retained as failed captures, not settled-render proof.
After closing the top preview with Escape and dismissing the shared dialog,
Chrome has no open dialogs and is online again. All native network conditions
were restored in cleanup. The existing nonblocking shared-error presentation
is under a read-only route/caller audit; no new error policy is implemented.

Fresh focused heading/preview/context regressions pass 69/69 in five files.
The actual enabled "Send with staged context" button has white foreground on
`rgb(62,106,224)`, measured contrast 4.8356:1 in the current default skin.
The temporary unsent draft and staging were cleared without sending a message.
TASK-13396.9 heading/contrast criteria are checked; recovery interaction and
full UAT remain open.

TASK-13396.17 now tracks the blocking preview/global-modal interaction. Reuse of
the existing nonblocking shared alert for this route has been presented for
approval and is not implemented. Historical TASK-12020.17 addressed stale
transient background-request popups, not this true-offline two-modal interaction.
A final actual Chrome state read confirms online, no dialogs, visible Ready
composer and no `__tldw_test_bypass` flag.

### Revised Storage And Citation Evidence

TASK-13396.14's bounded revision is frozen at helper SHA-256 `449ae954...` and
Research effect `2c837ea6...`, with isolated delta `732cc005...`. Parent rerun
passes 94/94 migration/inventory/effect tests, and frontend typecheck exits zero.
The actual Add source link (`/research-workspace?tab=sources`) loads Research
with Server context ready and the canonical source. Returning through a full
Chat document navigation retains workspace `ab1188e3...` and source `a8bcb45d...`.
Actual metadata, source, artifact, note, source-view and context reads return
200; there are no migration writes or new tombstones. The final source-retention
screenshot was visually inspected and both production hashes remain unchanged.
An earlier Open library run correctly reached `/media`, but waited for the wrong
Research status; that timeout is an automation expectation error, not a pass or
an application load failure. Independent final migration review remains pending.

TASK-13396.10 revision 3 is sealed: two-path delta `e0358eb7...`, store
`bb0e81f6...`, adapter test `129930f0...`. It invalidates only the owning cached
connection on close/open failure or known closed-transaction failure; old handlers
cannot invalidate a replacement. Parent explicit adapter/hydration tests pass
37/37; combined Chat Workspace/store regressions pass 438/438 in 30 files.
CDP Debugger inspection confirms actual Chrome loaded the revised invalidation
and conditional-removal/subscription source. This is runtime-code evidence, not
native abnormal-close or destructive legacy-race fault acceptance. Final bounded
review is pending; no broad storage-unchanged or transaction rollback claim is made.

A new actual staged-source turn creates a real chat (201), retrieves RAG (200)
and generates through Gemma (200). The protected native response artifact
`chrome-cdp-native-citation-rag-response.json` contains four documents, five
citations and four chunk citations. The settled visible answer is correctly
"The trial code is VIOLET-7429 (doc id='0').", but has zero citation controls and
zero links. Its screenshot was visually inspected. Source trace shows existing
retrieval normalization and the shared pipeline populate assistant `sources`,
while WorkspaceChatPanel omits that prop from the existing message renderer.
TASK-13396.18 tracks the minimal prop-forwarding repair, presented for approval
and not implemented. Literal generated references are not parsed into citations.
Full UAT, live Stop, preview Retry and owner-qualified checkpoints remain open.

Final hydration code additionally passes actual native repeated-failure recovery.
A mode-0600 backup (`chrome-native-workspace-before-repeated-recovery.json`,
SHA-256 `b4161426c43a08108c6a9e5903e318fe278dcc26cbcf4fc515f677a0c52ffa8b`)
precedes the malformed-root stimulus. Initial failure and an actual Retry both
leave malformed root/split sibling bytes intact and mount zero composers.
Restoring the original root and clicking Retry recovers the same canonical
workspace with no recovery alert or new tombstone. Both final screenshots were
visually inspected. This extends native parser/retry acceptance only, not native
abnormal IndexedDB close or transaction commit timing.

### Final Review Continuation

The migration re-review validates the three original P1 repairs, but retains a
P1 late-writer countercase: same-target root/snapshot bytes reappear after their
removal, receive a destructive tombstone, and are hidden/deleted by the actual
store consumer. A P2 compact notice also incorrectly describes failed partial
removal as failure before deletion. TASK-13396.14 remains In Progress with a
third bounded repair round; canonical reload evidence does not close these
findings. The exact review is `task-13396-14-frozen-revision-review-late-writer.md`
in the immutable review packet. No backend receipt redesign or cross-tab
atomicity claim is added. TASK-13396.10 revision 3 is being reviewed independently
without waiting for the migration repair.

A fresh connection through the existing Chrome CDP endpoint reads the actual
`http://127.0.0.1:18089/chat-workspace` tab: Ready composer, canonical workspace
`ab1188e3...`, online, no dialogs and no test-bypass flag. The first locator
expected a localhost tab and failed before interaction; correcting it to the
observed URL succeeds. That is an automation expectation error, not an app load
failure. No plugin was installed, and no mocked UAT response was introduced.

The fresh Ready screenshot `chrome-cdp-continuation-ready.png` was visually
inspected. Read-only native metadata shows the workspace IndexedDB database
exists, but no current local workspace chat/artifact offload references exist.
No records were seeded, replaced or deleted to manufacture a connection-recovery
scenario. Native abnormal-close/offloaded-content UAT therefore remains open;
the focused adapter tests do not substitute for it.

TASK-13396.10's separate immutable revision-3 review now finds no actionable
issues. Fresh exact source probes confirm original records survive recovery and
stale handlers/errors cannot invalidate a replacement connection. Parent rerun
passes 37/37 adapter/hydration cases, live hashes still match the sealed packet,
and whitespace checks pass. The bounded hydration repair is Done; native
abnormal-close/commit timing, checkpoints and full UAT are not accepted by this
ruling. No commit, PR, merge or GitHub epic closure was performed.

### Third-Round Migration Ceiling

The third bounded migration revision is frozen at
`/tmp/tldw-task-13396-14-round3.mtaxxk/HANDOFF.md`: helper `3b8136a0...`, effect
`656237f4...`, incremental patch `923c2606...`; sealed store `bb0e81f6...` is
unchanged. Synchronous absence checks and exact-byte marker retraction repair
the final-state controls; the partial-failure notice now uses actual removed IDs.
Parent fresh migration/effect tests pass 104/104, combined Chat Workspace/store
tests pass 445/445 in 30 files, frontend typecheck exits zero, and diff checks pass.
These test scopes overlap and must not be summed as distinct coverage.

P1 remains open. Parent reran the exact-source memory-only countercase script:
writes during awaited ack survive the helper but are hidden/deleted by the marker
consumer; a concurrently started consumer deletes a fresh snapshot before helper
detection and deletes the fresh root even after marker retraction. Mock receipts
are only the isolated regression substrate, not native UAT or durable import
evidence. Correct final marker state cannot restore already erased data. The
helper-only repair stopped after the third round rather than adding more rechecks.
The recommended policy is to retain writable legacy copies without automatic
destructive cleanup/markers/delete acknowledgment. That decision is awaiting
approval; coordinated writer/consumer quiescence is a broader alternative. No
policy change or old-tombstone recovery has been implemented. Final independent
bounded re-review confirms the retained P1, credits the guard/notice repair and
finds no additional actionable issue or new canonical-exclusion regression.
TASK-13396.14 remains In Progress. Precise concurrent-case chronology: fresh root
removal occurs after marker retraction but before the caller observes the blocked
result, not after that observed result. The separate immutable review is
`task-13396-14-round3-independent-review-open-p1.md`. This finite round is finished;
no fourth helper repair is underway.

Actual Chrome CDP final-round navigation is separately accepted: Add source
loads Research with Server context ready and the existing source; full Chat
document navigation retains canonical workspace `ab1188e3...` and the Lumen
source. Seven observed scoped GETs return 200, with no migration writes, no new
tombstone, online and no dialogs. The final source screenshot
`chrome-cdp-round3-canonical-retained.png` was visually inspected and production
hashes stayed unchanged. This proves canonical admission/retention, not native
legacy race safety. Full no-mock UAT and epic closure remain unaccepted.

Final remote-tip check still returns `955b1d9626a055ca44336a00d3d4c144949cb00f`.
No policy approval has been received, no policy mitigation has been implemented,
and there are no staged/unmerged changes or commits from this continuation.

### Approved Retain-Copies Continuation (2026-09-30)

The human approved the bounded retain-writable-local-copies policy. Automatic
content deletion, destructive migration marker publication and client-delete
acknowledgment are being removed. Existing marker bytes remain historical
metadata, not authority for store hydration/persistence to hide or delete fresh
content. The manager's separate ownership/reconciliation gate remains intact;
neither erased-data recovery nor legacy owner adoption is approved.

Fresh read-only Chrome CDP inspection renders the actual Chat Workspace Ready.
Native storage contains the canonical workspace and two empty legacy snapshots;
no records or API responses were manufactured. This is preparation, not acceptance
of the new policy. Fetched dev is `03043d1c10cbdbfa53945c0641d90e4e37836754`.
Its unrelated TASK-13396 collides with the candidate tracking root, so official
Backlog collision resolution precedes integration. Focused policy verification,
independent review and new native UAT are still in progress. Full UAT and the
epic remain open; other approval gates are unchanged.

The policy is now implemented and frozen in
`/tmp/tldw-retain-legacy-policy.bJO5KU/retain-writable-copies.patch`
(SHA-256 `b023cb46691f575a2e2efd007d89ddb1cc8b89e4862737423e96824cbe6ad995`).
It removes destructive runner callbacks/sinks and marker-driven store filtering
and deletion, preserving canonical admission, identity/ABA fences and ordinary
ownership/quota/stale-key behavior. Six intended policy RED cases pass GREEN.
Parent integrated verification passes 590/590 in 40 files, frontend TypeScript
exits zero, and scoped ESLint reports zero errors and 21 unchanged normalized
warnings. The isolated worker dependency-resolution failure does not reproduce
in the actual workspace. Fresh Bandit on the unchanged seven Python production
paths has zero findings/errors with 256 inherited suppressions.

Fresh no-mock Chrome CDP evidence is
`chrome-cdp-retain-policy-native.json` in the isolated runtime directory. Full
document reload and actual online Browse yield five scoped workspace GETs, all
200. Canonical workspace `ab1188e3...` and the Lumen source remain selected;
the two existing empty legacy snapshots and native historical bad43 marker have
identical before/after SHA-256 hashes. There are no migration writes, failed
requests or page errors. Actual Escape completes with zero dialogs and Browse
focus restored. Preview and stable Sources screenshots were visually inspected.
The first run waited for the wrong literal `VIOLET7429`; the actual displayed
source has `VIOLET-7429`. That automation timeout is retained, not attributed to
page loading; the corrected observed-text run passes.

This native check does not exercise an eligible legacy receipt or a fresh writer
under an old marker; regression doubles cover those source controls, not UAT.
No records were seeded to manufacture such a case. Already-erased bytes were not
restored, and metadata receipts are not claimed as durable content imports.
Independent source review and tracking semantic-copy verification are complete.
The exact eight-file review finds no actionable production or assertion-weakening
issues and binds live/frozen hashes; all fourteen protected recovery declarations,
its test and sealed packet are unchanged. See
`/private/tmp/tldw-retain-legacy-policy-bJO5KU-review/review.md` and its
`source-evidence.json`. TASK-13398.14 is bounded Done under the approved policy;
this is not merge permission or acceptance of the native limitations above.
Full UAT, the root task and GitHub epic stay open; pending approval gates are
unchanged. No commit, push, PR or GitHub mutation was performed.

### Supplementary Native Offload Checks (2026-09-30)

The actual Research chat composer and real Gemma backend created conversation
`24ce542c-2635-45db-88a2-a2a2b757c023` in canonical workspace `ab1188e3...`.
The first labelled message exceeded the existing 8 KiB offload threshold:
chat creation returned 201, scoped messages/settings and completion returned
200, and native storage produced the production offload pointer. No IndexedDB
records, authentication, model responses or readiness flags were seeded or
replaced. Evidence is `chrome-cdp-native-offload-chat.json` in the runtime
directory used above.

Two subsequent probes closed actual native database handles through Chrome
CDP, then sent real composer turns. Both completions returned 200. The final
native record contains six messages and all three expected assistant answers;
the first user message and historical marker remain unchanged. However, both
probe commands exited 1 on incomplete branch observations: the first did not
observe its expected inline fallback, and the second observed no application
`InvalidStateError` debugger pause. Their artifacts remain unchanged:
`chrome-cdp-native-closed-cache-recovery.json` and
`chrome-cdp-native-cache-path-recovery.json`. These functional observations do
not establish the specific cached-handle, abnormal-close or transaction-commit
recovery paths. No further fault probe was attempted and no acceptance oracle
was weakened. This supplementary Research flow does not accept primary Chat
Workspace checkpoint persistence or full UAT.

The final screenshot was visually inspected. It shows a real lorebook
diagnostics 404 and a transient Server context loading label, not a fully Ready
snapshot. The workspace-scoped diagnostics request returns 404 while scoped
chat reads and generation work. Source confirms
`export_lorebook_diagnostics` ignores the caller's scope query and defaults the
shared ownership verifier to global scope. TASK-13398.19 tracks the bounded
proposal to reuse existing scope resolution, preserving owner and wrong-scope
rejection. The approval question now includes this fix with TASK-13398.16-.18;
checkpoint TASK-13398.12 remains a separate written-spec gate. No implementation
approval is inferred from automatic goal continuation.

### Native Reload And Remaining Prop Omissions (2026-09-30)

Remote dev was rechecked and still equals integrated `03043d1c...`. The first
Research document reload timed out waiting 120 seconds for the final answer;
`chrome-cdp-native-offload-reload-failure.json` retains the failure and actual
request results. Next/API/model/Chrome listeners were confirmed live, so no
services were restarted. A subsequent read found the answer. Its six-row
retention assertion also failed: restoration had added the legitimate server
system row before the six conversation rows. Read-only role/hash diagnosis
confirms the original user message retains hash `a22ba697...` and all three
answers survive. This is not evidence that the first user was overwritten.
The failed checks and their assertions remain unchanged.

A second full reload used foreground Chrome and 300-second UI/navigation
bounds. It completed in 2,912 ms, restoring the answer and Server context ready.
All seven stored message rows, IDs and their exact serialized SHA-256
`d279e94fa07bd10e1dd71e9e23a12d1ee04925b3cecff2be7cde5a4c1a74962d`
match before/after; six rows are user/assistant turns. Conversation identity
and historical marker also match. Evidence is
`chrome-cdp-native-offload-foreground-reload.json`; its screenshot was visually
inspected. This accepts this existing Research session's ordinary reload
retention, not primary Chat Workspace draft/reference checkpoints, abnormal
database close, transaction commit timing or full UAT.

Native reload revealed two additional source-validated omissions:
TASK-13398.20 tracks ResearchChatPane not forwarding `msg.role` to the shared
message component. The component already supports System prompt presentation;
the omitted prop makes the restored system row appear as You. TASK-13398.21
tracks a sibling loader settings-reconciliation call omitting its existing
scope. Actual extra global GET/PUT settings requests return 404 for the workspace
chat, while matching scoped requests return 200. The proposal forwards those
existing props rather than remapping/deleting history or weakening owner/scope
checks. The diagnostics 404 and notification unread-count 401 are also retained
in the raw request evidence; there is no all-requests-success claim.
The late global-settings pair appears in the command's console output after the
JSON snapshot was written; that JSON alone does not establish those two requests.

The consolidated approval request covers TASK-13398.16-.21. TASK-13398.12 still
requires written-spec review. No application source was changed in this turn;
Bandit is not applicable to these tracking/report-only edits, and previous
production verification is not represented as a new run. Full UAT and epic
closure remain unaccepted. The required approval condition has recurred across
three consecutive goal turns; remaining implementation cannot proceed until
the human responds. No further repetitive probe is planned.
The goal status is now blocked, not complete. The root and epic remain open.

### Approved Repairs And Qualified Records (2026-09-30)

The human subsequently approved TASK-13398.16-.21 and the written checkpoint
specification for TASK-13398.12. This resolves the preceding approval gate;
historical failed probes and old goal status are retained, not current approval
requirements. Remote dev still matched integrated
03043d1c10cbdbfa53945c0641d90e4e37836754 at this continuation's start. The dirty
candidate and original stash were preserved. No commit, push, PR or GitHub
mutation is authorized in this continuation.

The bounded repairs propagate owning-turn cancellation into pinned capability
discovery, make the workspace backend error nonblocking, pass existing RAG
sources and restored role attribution to the shared renderer, and retain
workspace scope for diagnostics and restored settings reconciliation. Parent
fresh focused tests passed 255/255 in seven suites; diagnostics tests passed
22/22 with four warnings; frontend TypeScript exited 0. Touched Python production
Bandit reported zero findings/errors. Parent incremental ESLint reported zero
errors/new diagnostics, retaining 41 inherited warnings in its six-file scope.
Evidence: /private/tmp/chat-workspace-approved-six-20260930. These checks precede
the citation-control follow-up below and do not certify that later increment.

After preserving the exact private API environment, the owned API was restarted
to load the scoped endpoint repair, with a 120-second graceful-shutdown bound.
Real model, embeddings, Redis, Next and Chrome were reused; stub embeddings
remain disabled. A clean foreground Chrome reload observed scoped chat,
messages, diagnostics and settings GETs returning 200, System prompt attribution
and retained answers. No actual settings PUT was generated by that reload; the
GET/PUT forwarding contract has unit coverage, not new native PUT acceptance.
Artifacts: native-clean-scope-system.json/.png. An earlier reload crossing the
restart retained real 500s and later recovered through actual readiness Retry;
it is not an all-requests-success run.

Desktop preview acceptance used native CDP network failure on a frozen source
set: actual requests failed with ERR_INTERNET_DISCONNECTED, Retry preview stayed
hit-test reachable beside the shared inline alert, and actual Retry after native
network restoration returned canonical content with HTTP 200. The command exited
0 and screenshots were inspected. Earlier wrong-locator and interrupted
harnesses remain failures. Artifact: native-frozen-preview-recovery.json. Mobile
placement is not established by that desktop run.

A real scoped RAG search and Gemma completion returned 200 with the Lumen facts.
Expanding Citations displayed the canonical 393-character source and chunk ID.
No external URL was returned, so there is no external-navigation acceptance.
Independent review found the newly exposed Ask/Open Search buttons have no
workspace event consumers. TASK-13398.18 now suppresses only those unsupported
commands through the shared renderer's existing optional-hide-prop pattern;
evidence/details/safe URLs and legacy defaults stay intact. Follow-up tests and
native acceptance are pending. Artifacts: native-real-rag-citation-evidence.*
and /private/tmp/chat-workspace-approved-six-independent-review-20260930.md.

The first Stop probe did not identify the factory-owned discovery request:
Chrome's default initiator depth showed only performTldwRequest. Stop was never
clicked. A real completion occurred, but it does not accept cancellation.
native-stop-capability-failure.json retains that failed harness. The reassessed
probe will use native async-stack observability, without interception or mocked
responses. Independent review found no actionable findings in TASK-13398.17.

Checkpoint Stage 1 extends existing records with strict, versioned owner and H1
qualification, exact-key reads and independent clone/serialization. Unqualified
legacy copies are preserved and never become qualified authority. Focused
RED/GREEN passed 62/62; parent regression passed 216/216 in eight suites and
independent frozen-diff review found no actionable findings. Stage 2/3 are not
implemented or accepted: H1 selected-history admission rejects RAG and conflicts
with durable workspace admission for plain saved sends too. Removing guards or
mounting the provider is not a valid integration. The newly discovered protocol
decision must cover selected ancestry, citations, Retry identity and unknown
outcomes. Assessment: /private/tmp/chat-workspace-h1-rag-feasibility-review-20260930.md.
Full primary checkpoint reload/A/B draft UAT, full UAT and epic closure remain
unaccepted. The previous specification approval is not being requested again.

### Final Native Checks And Latest Dev Advancement (2026-09-30)

The reassessed Stop probe used native async-stack depth32 to identify the actual
getChatTurnIdentitySupport -> pageAssistModel -> runChatPipeline discovery GET.
An actual Stop generating click canceled that exact pending request with
net::ERR_ABORTED and canceled=true. No completion POST followed during the
recorded eight-second online observation; Send returned and the draft remained.
The eleven production file hashes matched before/after. Artifact:
native-stop-capability-accepted.json/.png; screenshot inspected. The earlier
failed probe remains unchanged and was not converted to acceptance.

The citation-control follow-up has independent review with no actionable
findings. Parent fresh checks passed 220/220 plus layout41/41; frontend TypeScript
exited0. Native Enter close/reopen of Citations, Space toggle of the actual
source summary and retained focus passed, showing canonical source details
without unsupported Ask/Open Search commands. Artifact:
native-final-citation-controls.json/.png. An earlier exact-name locator harness
was interrupted and retained: the accessible header name includes its expanded
image text. It required a corrected locator, not an application edit.

The final remote check found dev2256bc82afa154891c635df3ef955ed7a6bc61b3. Its five
commits changed four Playground tests, extension locale strings and two unrelated
task records, not repair production code. A fast-forward preserved all202 current
candidate file hashes and the unchanged61-entry stash list, without a new stash
or commit. Before-patch/manifest: pre-dev2256-preserved-candidate.patch and
pre-dev2256-manifest.json. No reset, cleanup or conflict-marker resolution was
performed. Earlier native acceptance remains explicitly attributed to03043,
with the exact production hashes preserved across the advancement.

Fresh2256 verification: focused repairs261/261 in8files; upstream changed tests
27/27 in4files; frontend TypeScript exit0; broader diagnostics53/53 with7warnings;
production Bandit zero findings/errors; git diff --check exit0. The changed
Playground fixture emits existing missing-method diagnostics while assertions
pass; it is not native service evidence. A wrong backend path selection collected
zero/exit4 before the corrected53-test run. Old pytest temporary-directory cleanup
warnings were retained; unrelated directories were not removed.

Actual mobile390x844 offline preview/retry ran on2256. Retry preview remained
hit-test reachable while real requests failed ERR_INTERNET_DISCONNECTED and the
shared backend alert was present. Restoring native network conditions and using
actual Retry returned canonical preview200 and393characters. Both screenshots
were inspected; no alert/Retry occlusion. Artifact:
native-mobile-preview-recovery.json, native-mobile-offline-preview.png and
native-mobile-recovered-preview.png. The first mobile run timed out before
any fault or request because its Browse locator omitted the source title; its
empty request trace/failure remain retained. No mocks or source changes were
used to correct that harness.

A fresh2256 real RAG search and Gemma turn both returned200 with correct facts
and three distinct late_chunk citations. The final-chunk-only assertion failed
because the trial-code chunk does not contain the interval/coordinator; the
other returned chunk does. native-latest-dev-real-rag-failure.json remains
unchanged. Native expansion of all three verified the separate evidence, not
an amended claim that the failed assertion passed.

Actual citation navigation exposed a separate issue: original uploaded filename
lumen-field-memo.md is returned in metadata.url and rendered as a relative web
link. Actual Open source opens origin/lumen-field-memo.md and a missing Next /404
route, not source content. Artifact: native-uploaded-citation-navigation.json/.png.
No fabricated static route, fake server or stored-provenance rewrite was added.
TASK-13398.18 navigation acceptance is reopened pending this new root-cause
repair. Scoped canonical Browse preview remains independently accepted. Full
checkpoint integration, full UAT and epic closure remain unaccepted.

TASK-13398.22 records the new filename navigation defect. Parent source tracing
confirms the shared renderer is the appropriate proposed boundary; the generic
URL guard intentionally permits internal relative paths and must not be globally
restricted. A finite additional independent assessment was stopped without a
completed report; it is not counted as review acceptance. The proposed source-card
absolute-URL eligibility design and staged plan are now written, with approval
pending. No application source was edited for TASK22. Direct actual GET to the
native link target confirms404. TASK16/17/19/20/21 are bounded Done; TASK18's source
display/control repair is verified, but navigation acceptance remains open.

Final read-only native observations on2256 show all three distinct source chunks,
all expected facts across them, zero unsupported workflow commands, an empty
composer and disabled Send, with Stop absent. The screenshot was inspected and
eleven production hashes rechecked unchanged. Artifacts:
native-final-latest-dev-workspace.json/.png. This is observation evidence, not a
passing version of the earlier final check: native-final-check-failure.json
retains its invalid expectation that Send be enabled for an empty composer, and
its subsequent Browse step was not reached. Earlier accepted canonical Browse
and mobile Retry evidence remain separate. No further unchanged fault probe is
planned. All finite parent commands are drained, no native fault remains active,
and the two new design questions concern source URL eligibility and H1/durable
checkpoint compatibility, not the already approved six repairs or written spec.

### Approved Source Navigation And Compatibility Checkpoint (2026-09-30)

The human approved TASK-13398.22 implementation and continued TASK-13398.12
compatibility design. Source navigation now uses the shared MessageSource
boundary with existing sanitization and base-free absolute HTTP(S) validation;
generic internal-link helpers, source provenance and canonical owned Browse are
unchanged. Corrected RED had seven actual anchor failures; GREEN36/36 in seven
physical files. Full frontend TypeScript exited0. Renderer lint retains exactly
three inherited errors/five warnings, with no new diagnostic; test lint clean.
Independent frozen source/test review found no actionable issue.

Real uploaded-document desktop/mobile acceptance preserves all four returned
excerpts, suppresses invented filename links and unsupported commands, and opens
the actual canonical393-character source preview200. Compound run failures remain
separate from source22-native-render-navigation-accepted.json. Fresh native URL
acceptance on f3 returns three actual media2 chunks and safe absolute links;
clicking Open source opens the real Example Domain page. Both desktop and popup
screenshots were inspected; reviewed source/test hashes unchanged. Artifacts:
source22-native-url-generation.json and source22-native-url-navigation.json.
The former accepts real request completion, not answer quality. Gemma abstained
despite all three chunks being present in the captured completion system prompt;
the separate backend RAG generated_answer correctly used the purpose paragraph.
No context-truncation or full grounded-answer success is inferred.

Remote dev advanced to f3f1b4fdbe3fe461b371ece30887c5fff8476d9d:27 commits,
21 RG/auth paths, no overlap with206 preserved dirty files. Preserved manifests
and all63 stash IDs were checked before/after. Owned API restarted with the
same environment/database; authenticated health200. An unauthenticated401
readiness harness exited1 and is retained, not called a server-start failure.
Fresh checks: source36/36, frontendTS0, backend343 passed/14 skipped/2 xfailed,
diagnostic Bandit0 findings/errors. Docker/Postgres unavailable and sandboxed
default Redis6379 are explicit skips; real UAT Redis16379 was not repurposed.

Actual web import fetched/stored media2 with HTTP207 analysis-unselected warning,
then failed vector indexing: existing MiniLM collection384 versus default Qwen
vector1024. Native UI truthfully showed text searchable/vector failed; actual
RAG recorded document_retrieval_failed and FTS fallback. TASK-13398.23 tracks
non-destructive private runtime alignment and real vector re-verification.
No mocks, fabricated routes or source metadata replacement were used.

TASK12 compatibility draft and independent review preserve server-owned H1
admission/settlement, logical durable UUID and protected recovery authority.
Three review corrections are incorporated: protected live result-state digest,
fresh operation identity per inference attempt, and disabled exact-payload Retry
after reload when frozen evidence is unavailable. Concrete wire/digest/capability
and assistant-context matrices remain gates; no provider mounting or guard
removal was performed. TASK18/22 navigation is bounded accepted; full primary
checkpoint reload, abnormal storage paths, full UAT and epic closure remain open.

### Vector Recovery And Runtime/A11y Acceptance (2026-09-30)

TASK23 is a private environment repair, not an application/model-migration patch.
Authenticated status confirmed original media1 MiniLM vectors, media2 missing
vectors and Qwen as the runtime default. The preserved private config now selects
MiniLM to match the existing384-dimensional collection; no collection/vector
deletion or forced regeneration occurred. Only the verified owned API restarted
48874->88122 with its exact prior environment/database. Authenticated health200
and the actual model catalog confirm MiniLMdefault. Real job26f2d795 completed;
media1 still has one vector, media2 now has one, both canonical sources vector-ready.

Post-recovery native Chrome hybrid RAG/completion200 returns four documents,
including stored MiniLM vector2, with errors[] and no FTS-on-error fallback.
All four excerpts are retained desktop/mobile; the vector result lacks URL and
does not invent one, while absolute links remain available on the other cards.
The original purpose question also now produces the correct answer and evidence.
Artifacts: source23-real-embedding-retry.json, source23-native-vector-acceptance.json,
source23-native-url-generation.json and source23-original-question-acceptance.json.
Screenshots inspected; earlier FTS/abstention records unchanged. This is bounded
functional acceptance, not a claim that the model will never ignore evidence.

TASK8/9 fresh regressions pass253/253 in16 files with no skips. Independent review
of frozen deltas/current integration found no actionable defects; source hashes
remain stable. Real plain Gemma completion200 emits a Markdown H1 which renders
as H2, leaving exactly one page H1 and no overriding aria-level. Actual enabled
Send foreground white/background rgb62,106,224 has contrast4.8355856:1, opacity1.
The preview uses polite/atomic loading and assertive/atomic error; real native
offline request fails ERR_INTERNET_DISCONNECTED, hit-test reachable Retry outside
the alert then returns scoped preview200/all393characters at390x844. The network
is restored, the probe's unsent draft/staging cleared, preview closed and Send
correctly disabled with an empty composer. No speech/screen-reader certification
is inferred from DOM live-region evidence. Artifacts: tasks89-f3-regressions.json,
tasks89-heading-native-url-generation.json, tasks89-native-acceptance.json and
tasks89-native-generated-heading-visible.json/.png, plus mobile preview PNGs.

The initial compound probe stopped before network faulting because Meta+A did
not select the textarea text in raw CDP. Its failure remains separately retained;
native SelectAll selected the observed range and Backspace cleared only the
probe text. The corrected run completed the actual fault/recovery. The first
heading PNG was off-scroll; the separate visible heading capture was inspected
instead. Earlier Stop acceptance remains native-stop-capability-accepted.json,
with the same protected source hashes; no new Stop acceptance is inferred from
the successful heading or empty draft alone. TASK8/9/18/22/23 are bounded accepted.
Completed parent-owned narrow plans are archived in the temp evidence directory
before removal; active checkpoint/root plans and other agents' plans are retained.
Final remote read still reports devf3f1b4fdbe3fe461b371ece30887c5fff8476d9d.
Full checkpoint integration/abnormal-storage/full UAT/epic remain open.

### Concrete Checkpoint Protocol Candidate (2026-09-30)

Authorized TASK12 design continuation now includes a concrete proposed v1 in
CHAT_WORKSPACE_H1_DURABLE_COMPATIBILITY_2026_09_30.md: strict source allowlist and
20-source/1,000-scalar excerpt/64-KiB result bounds, faithful evidence/source
ordering, finalized body-only digest policy, versioned pinned OpenAPI marker,
verified live protected recovery DTO and explicit supported/rejected context
matrix. Required unsupported/oversized evidence rejects before admission, rather
than being silently discarded. Plain result projection is explicitly empty;
accepted-reference inference has a fresh request-context digest/browser operation.
No exactly-once inference or uniqueness from pagination is promised.

Independent review found a P2: the candidate simultaneously said body-only and
included scope, but current scope is in query parameters. Parent verified both
transport/endpoint and corrected it: scope/auth/target are not synthesized into
the body digest; separate pinned leases and owner/scope checks remain mandatory.
Reviewer re-read the changed paragraphs and confirmed resolution/no new conflict.
The three earlier protected-state/operation-identity/missing-evidence corrections
remain intact. Application integration, backend byte-parity proof, source bounds/
support-matrix approval and native primary checkpoint acceptance are still gates.
No new API fields are implemented, no H1 provider mounted and no guard removed.
All sidecar agents are closed; TASK12/root remain In Progress.
