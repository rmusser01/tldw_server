# Migu Buddy UAT — 2026-09-05

Current verdict: **Chatbook physical dragging passes. Server Buddy setup, visuals, viewport containment, and browser transport are repaired; end-to-end conversational usability remains unaccepted.** The initial failures below are preserved as before-fix evidence. See the repair follow-up for implementation and verification.

## Runtime and method

Chatbook: session target was merged PR #2404 commit f8cb939e2bd3a111555acc8d87a4b4907ee2268e, with an isolated profile and native macOS Terminal. The runtime receipts did not record Git revision or dirty state, so exact tested-revision provenance is unverified. Foreground mouse gestures were explicitly authorized. Server: dev 2742468a19, separate worktree `codex/migu-server-buddy-uat`, full FastAPI app at 127.0.0.1:9101 and Next development WebUI at 127.0.0.1:18384 in advanced deployment mode. Locked Bun dependencies installed. Fresh synthetic single-user account and separate databases. Real REST/WebSocket requests; no browser route mocks, mocked replies, fake microphone, or mocked provider.

The previously running frontend at 18383 had a missing dependency and the backend at 9099 omitted Persona routes. Neither was changed or counted as a Buddy product failure. The fresh backend required the current in-repository profile-core/MCP packages on PYTHONPATH. No full test suite was run.

## Chatbook results

- Physical move: rendered region (41,31,28,15) → (69,25,28,15), with native mouse-down, multiple moves, and mouse-up observed by production handlers.
- Physical lower-right resize: rendered size 28×15 → 40×21; release delivered successfully.
- Fresh PID 46565 restored rendered geometry (69,25,40,21). Preferred size is stored separately from the current rendered fit. Native exit outcome remains unverified: the surviving exit receipt is not linked to a PID or return code.
- Separate PTY protocol probe: all 22 checks passed, including move, resize, capture release, boundary clamp, modal/navigation transitions and restart restore.
- Background per-PID dragging produced no application events; foreground gestures resolved that automation limitation without a code change.
- The long-running first harness detected normal config changed since the prior-day baseline; this interval does not support an unchanged-config claim. The fresh restart baseline remained unchanged. Runtime database paths were isolated.
- OpenAI realtime verification remains outside this completed physical check and is still open in Chatbook TASK-31585.

## Initial server acceptance matrix (before repairs)

| Journey | Result | Evidence |
|---|---|---|
| Load real Persona page and create Migu UAT | Pass | New profile e19c631e-e2ac-452d-9935-293af03cee4e created through UI |
| Complete voice-default setup | Fail | Repeated 409 optimistic-version conflicts; TASK-13182 |
| Open Buddy at default desktop position | Fail | Composer/navigation entirely below viewport; TASK-13175 |
| Drag Buddy and restore position | Pass | Native browser pointer gesture moved shell to (704,17); reload/reselect restored it |
| Select bundled Migu Marker Basic, copy draft and activate | Pass for backend publication | Active pack 30a938ed-3fbf-4232-a41f-8f6d372f036b; authenticated PNG GET 200, 9617 bytes |
| Render activated Migu in WebUI | Fail | Relative image URLs request frontend origin and repeatedly 404; TASK-13176 |
| Start and Stop live session from Buddy | Pass | Idle appears on Start; Stop becomes disabled after stop |
| Send real Buddy text from browser | Fail | Stream connection error, draft retained, no WebSocket created; TASK-13183 |
| Backend direct stream control | Pass for handshake/planning only | Authenticated WebSocket returns WS_CONNECTED notice and tool_plan for synthetic input; test session stopped with HTTP 200 |
| Conversational answer/provider quality | Not verified | Browser send blocked. Direct stream returns a RAG tool plan, not the requested conversational answer; no plan execution was approved |
| Voice | Unavailable, gating correct | Real session advertises capabilities.voice=false; shell offers no voice button. No microphone, speech, or provider UAT pass claimed |
| 390×844 web viewport | Current desktop-only policy observed | Buddy unmounts under useDesktop guard; builder remains visible. Not a mobile Buddy pass |

## Original defects and fix order

1. **TASK-13183 — Stream lifecycle after Strict Mode remount.** Start succeeds; Send repeatedly returns “Persona live stream failed to connect” without opening a WebSocket. `usePersonaLiveControl.tsx:183` cleanup sets mountedRef=false, and effect setup never restores it. `ensureStreamSocket` then rejects before socket creation. Next config enables Strict Mode. This is a source-supported mechanism; no production fix was applied during this UAT. Re-run actual browser Send after correction, then validate incoming plans/replies and their visible controls.
2. **TASK-13182 — Setup version handoff.** Create Migu → Save assistant defaults: database version 2 versus expected 1. Reload/reselect/retry repeats version 4 versus expected 3. Voice saving and setup advancement need to hand off the current version while retaining optimistic conflict protection.
3. **TASK-13176 — Visual transport.** Copy Migu Marker Basic as draft → activate → browser requests `/api/v1/persona/.../assets/.../content` on port 18384. Advanced mode has no same-origin API rewrite. The identical authenticated asset endpoint on 9101 returns image/png 200. Failed frames are retried repeatedly while animation runs. Resolve asset transport/auth consistently for both builder and shell, then retest animation and state changes.
4. **TASK-13175 — Expanded-shell containment.** At 1280×720, initial expanded dock (1104,609,220,687) and popover (1104,818,220,478) exceed the viewport. Normal click on Choose/Change Buddy times out as outside viewport. Dragging upward makes it usable. Host clamping watches position and window resize, but not expanded/content size changes. Retest expansion, errors, loaded visuals and reload without manually rescuing the position.

## Evidence

[Clipped controls](assets/migu-buddy-uat-2026-09-05/popover-clipped.png), [setup conflict](assets/migu-buddy-uat-2026-09-05/setup-conflict.yml), [failed browser send](assets/migu-buddy-uat-2026-09-05/ui-sent.txt), [stopped session](assets/migu-buddy-uat-2026-09-05/ui-stopped.txt), [backend stream frames](assets/migu-buddy-uat-2026-09-05/direct-stream-frames.json), [session capabilities](assets/migu-buddy-uat-2026-09-05/live-session.json).

The initial UAT created TASK-13182,13175,13176,13183 without production changes. Its browser and synthetic sessions were stopped before the repair pass. Chatbook follows existing ADR-074.


## Repair follow-up

The isolated `codex/migu-server-buddy-uat` branch repairs six concrete failures:

- **TASK-13182:** pass the saved profile version explicitly to the setup checkpoint; retain genuine optimistic conflicts and ignore stale save completions after persona selection changes.
- **TASK-13175:** clamp expanded/content-resized Buddy bounds before paint, constrain dock height, and scroll the compact controls.
- **TASK-13176:** fetch server-owned visual content through the existing authenticated binary transport, render disposable object URLs, retain failures until sources change, and abort/revoke on cleanup. Both sprite frames and generated candidate thumbnails share this path; external asset URLs do not receive credentials.
- **TASK-13183:** restore Strict Mode mount readiness and fence stale asynchronous session/stream work.
- **TASK-13184:** real Chromium then exposed the server's missing WebSocket subprotocol selection. After successful authentication, select only the offered bearer marker, never its credential. Invalid authentication remains rejected.
- **TASK-13179:** actual quickstart startup exposed a missing public `/health` rewrite. Next now forwards that path only in quickstart mode to the configured backend's public endpoint.

These are routine repairs to existing contracts; no new ADR is required. TASK-13180 records the separate stream-outcome/approval feedback gap and TASK-13181 records cookie-authentication coverage.

### Real browser evidence

Fresh synthetic persona `e0a442a5-3861-4529-a332-a5391626f51f` (Migu UAT Repaired) saved defaults and advanced to Starter commands, then Safety, without the previous 409. [Setup evidence](assets/migu-buddy-uat-2026-09-05/repaired-setup.txt).

Migu Marker Basic was copied and explicitly activated as pack `559d3eb8-9bde-4e0c-b01e-46440708e5a7`. Builder and Buddy frames decoded at 96×96 from blob URLs backed by authenticated PNG 200 responses on backend port 9101. [DOM/network evidence](assets/migu-buddy-uat-2026-09-05/repaired-desktop.json), [desktop screenshot](assets/migu-buddy-uat-2026-09-05/repaired-desktop.png).

At 1280×720 the expanded dock was `(938.55,229,325.45,416)`, bottom645. At1280×360 it was `(938.55,16,325.45,328)`, bottom344. The popover scrolled 112px and the bottom navigation link ended at328px, within360px. [Small viewport geometry](assets/migu-buddy-uat-2026-09-05/repaired-short-viewport.json), [scroll proof](assets/migu-buddy-uat-2026-09-05/repaired-scroll-proof.json), [screenshot](assets/migu-buddy-uat-2026-09-05/repaired-short-viewport.png). The small-height screenshot intentionally retains the intermediate handshake error; it demonstrates readable error and navigation containment before TASK-13184 was loaded.

After the handshake repair, Buddy Start→Send established `ws://127.0.0.1:9101/api/v1/persona/stream` and received a notice plus `tool_plan` from the real backend. [Browser stream evidence](assets/migu-buddy-uat-2026-09-05/repaired-live-browser.json), [post-send screenshot](assets/migu-buddy-uat-2026-09-05/repaired-send.png). No tool plan was approved or executed.

Quickstart now returns public health200/ok through port18385 and initializes a real cookie session with200. It still cannot open Persona: legacy user/DB dependencies reject cookie-only requests. Cookie metadata and a live request confirmed the browser holds and sends the session cookie; logs show session mint and users/me/profile200 followed by notifications401. This is a separate backend dependency coverage issue (TASK-13181), so TASK-13176 remains In Progress until quickstart image acceptance can run. Cookie WebSocket retesting also needs18385 explicitly added to the isolated backend allowed origins. [Readiness evidence](assets/migu-buddy-uat-2026-09-05/quickstart-readiness.json).

After the final stale-response fixes and a clean browser reload, defaults saving advanced again and Buddy sent another greeting, received notice+tool_plan, and cleared its draft. [Final stream evidence](assets/migu-buddy-uat-2026-09-05/final-live-browser.json).

### Verification

- Final targeted frontend regression run: **265 passed across 13 files**, covering setup, lifecycle, geometry, assets/editor/service, shell store, and quickstart configuration.
- Final targeted backend run: **54 passed** across Persona WebSocket authentication and live-control API suites.
- Production endpoint Bandit: **0 findings**. Test-inclusive Bandit differences are assertion checks; no credential material added.
- Scoped frontend ESLint: **0 errors,93 warnings** (existing hook/img guidance, plus native shared image rendering guidance); no ignored-file-only success used. Python Ruff findings unchanged from HEAD. New test/hook formatting and diff whitespace checks pass.
- No full test suite was run. The unrelated repository-wide typecheck failures below remain a validation limit.

### Remaining acceptance limits

- **TASK-13181:** canonical cookie-session authentication must reach legacy user/DB dependencies before quickstart Persona and image UAT can pass. Preserve permissions, CSRF, origin, revocation, and user isolation.
- **TASK-13180:** the compact shell does not display the incoming plan/reply or reconcile urgent approval state from its socket. A successful send therefore does not yet provide usable outcome feedback. The Buddy interaction PRD requires urgent-state feedback and routing to full Live for explicit approval; compact approval execution remains out of scope.
- No conversational model response or provider-quality acceptance is claimed: the real default response was a RAG tool plan.
- Voice capability was false; no microphone, STT/TTS, or live voice-provider pass is claimed.
- Desktop-only web Buddy policy remains; the mobile builder is not proof of mobile Buddy usability.
- Repository-wide TypeScript checking reports 80 errors across 6 unchanged files (Presentation Studio and skills certification tests). No diagnostic names a touched Buddy file. [Scope comparison](assets/migu-buddy-uat-2026-09-05/typecheck-scope.json). This is not a clean repository-wide typecheck claim.


### Final handoff

TASK-13182,13175,13183,13184,13179 are Done. TASK-13176 remains In Progress because its quickstart acceptance is blocked by TASK-13181. TASK-13180 and13181 are recorded as high-priority follow-ups.

Final real pointer drag moved the repaired dock from `(938.55,16)` to `(738.55,56)`; the asynchronous Stop completed and disabled its control. [Drag evidence](assets/migu-buddy-uat-2026-09-05/final-drag.json), [completed stop](assets/migu-buddy-uat-2026-09-05/final-stop.json). The initial immediate post-click stop flag is false because the API was still pending; the separate completion evidence is true.

The isolated UAT processes and browsers are stopped after verification. Synthetic runtime data is retained outside the repository. The repair pass was subsequently published as [server PR #2884](https://github.com/rmusser01/tldw_server/pull/2884) against dev63358431d7. All265 focused frontend and54 backend tests passed again after rebase; repository-wide typechecking still reports80 unrelated diagnostics. The setup task is renumbered to TASK-13182 during user-authorized review closeout; the older EPUB task retains TASK-13174. The requester supplied the human-written Change summary before the requested merge. [Chatbook PR #2418](https://github.com/rmusser01/tldw_chatbook/pull/2418) publishes the separate native UAT evidence against its dev branch.


### Qodo review follow-up

Protected frames now load on demand. Each renderer retains at most eight blobs and 16 MiB, except that one larger currently displayed frame is allowed by itself. Source changes and unmount revoke cached URLs; abandoned frame requests abort, and failed-source tombstones prevent retry loops for the current pack. Two regressions first reproduced eager loading (256 requests on mount, and unused frame downloads); nine authenticated-renderer tests now cover lazy loading, count/byte eviction, reuse, failure retention, cancellation, and source replacement.

A completed Start returns its successful backend create/resume result even if its original mount has gone away, while local state and sends remain fenced by generation. Sessions are persistent user-owned resources and can already have been resumed by another mount/client; they cannot safely be stopped on component cleanup without an exclusive ownership contract. Regressions cover both distinct and shared session responses.

Review verification after rebase onto dev dc0b7455f2: 271 focused frontend tests across 13 affected suites; 54 backend authentication/live-control tests. Repository-wide typechecking still reports the same 80 unrelated diagnostics, with none in touched Buddy files. Scoped ESLint: zero errors and two existing native-image warnings. The modified WebSocket test now includes parameter/return types and a behavior docstring. These review changes have automated regression evidence; the recorded real-browser UAT above predates them. Chatbook summary claims were narrowed to match the evidence review in PR #2418.

Task provenance: Buddy lifecycle was originally TASK-13177; rebase onto dc0b7455f2 introduced an independently allocated Docs Design recovery record at that ID. Buddy now uses TASK-13183, retaining its original record and notes; the task already merged into dev keeps TASK-13177.


### CI runtime follow-up

The shared-UI CI shard exposed editor test synchronization races absent from the initial Node26 frontend-config run. Replaying the failing file with Node20 and the CI deterministic config reproduced4 failures/60 passes. Tests now await manifest-initialized selections, custom-state options, and completed candidate-review feedback before the next action. The Node20 editor suite passes64/64; the original shard context plus both renderer suites passes93/93 across six files. Assertions remain intact, no timeout was increased, and production behavior is unchanged by this test-only follow-up. Scoped test-file ESLint has zero errors and47 existing any-type warnings. These checks supplement the earlier271 frontend/54 backend results; they are overlapping suites, not additional unique tests.

Final base update: rebase onto dev69c96ef715 preserved every code patch (range-diff equivalent). Buddy WebSocket task was originally13178 and now uses TASK-13184 because the new dev email-summarization task independently uses13178. No product scope changed.

Verification on dev69c96ef715 rebase:93 Node20 shared-UI tests across six files and54 backend tests pass again. All task IDs introduced by this PR are unique; diff whitespace checks pass.

## Cookie authentication and review handoff follow-up

The follow-up on `codex/migu-buddy-followups` closes TASK-13181, TASK-13192,
TASK-13176, and TASK-13180. The earlier failed acceptance results above remain
historical evidence; quickstart authentication, authenticated Migu images, and
visible plan review now pass against a real backend.

Legacy user dependencies now accept the canonical validated cookie principal.
Explicit authorization headers retain precedence, including invalid or blank
headers. Frontend readiness recognizes an active cookie bound to the exact
configured origin. Resource Governor ingress resolves that same authenticated
owner before applying owner-only quotas; it no longer permanently rejects
cookie traffic with 429 because it has only an IP identity. This preserves
owner quotas, CSRF, revocation, and cross-user isolation rather than relaxing
the policy. See [ADR-044](../ADR/044-cookie-session-governance-owner-preflight.md).

Buddy now shows pending, reply, error, and review feedback scoped to its current
persona/session/turn. Typed stream responses retain their client message ID;
late turns and obsolete connections cannot overwrite current feedback. The
full Live link identifies the same persisted session. Explicit Connect reads
its latest bounded pending plan, with every tool step unselected. Confirmation
rechecks persisted session ownership and active state, including plans rewritten
after a policy denial. An incomplete setup permits this explicit review detour
without marking setup complete. See
[ADR-045](../ADR/045-persona-live-pending-plan-handoff.md).

### Browser acceptance

- Cookie-only users/me, Persona profiles, notifications, ingestion capabilities,
  and config/docs-info returned 200. Both builder and dock images decoded at
  96×96 through authenticated blob loading. [Evidence](assets/migu-buddy-followups-2026-09-05/quickstart-builder.json),
  [screenshot](assets/migu-buddy-followups-2026-09-05/quickstart-builder.png).
- Two consecutive synthetic sends received distinct correlated plan envelopes.
  [Turn correlation](assets/migu-buddy-followups-2026-09-05/correlated-send.json).
- On the final production snapshot, synthetic text created session
  `5463d886-f01d-4943-91ce-52a8f3e6caa8` and plan
  `11938e31a16240d28e51d762e235c8d2`. Buddy visibly requested review. Explicit
  Connect returned that exact session and plan with the `rag_search` step
  unchecked; setup remained incomplete. The observed connection command was
  `voice_config`, with no approval or tool execution.
  [Send](assets/migu-buddy-followups-2026-09-05/final-send.json),
  [hydration](assets/migu-buddy-followups-2026-09-05/final-review.json),
  [screenshot](assets/migu-buddy-followups-2026-09-05/final-review.png).
- Cancel, Buddy Stop, and Disconnect removed the plan and feedback, disabled
  Stop, and restored Connect. [Terminal UI](assets/migu-buddy-followups-2026-09-05/final-stop.json).

Final backend run `backend-1788635007440317000` used wrapper 20585 and child
20594, which the wrapper reaped with exit 0. Its source base was
`86eb9e517cfdcd2af0987c3fcd01c71facaf9f30`; the tracked diff hash was independently
checked after exit and matched launch. Untracked paths are inventoried, not
included in that hash. [Launch identity](assets/migu-buddy-followups-2026-09-05/backend-current-identity.json),
[exit receipt](assets/migu-buddy-followups-2026-09-05/backend-current-exit.json),
[source comparison](assets/migu-buddy-followups-2026-09-05/source-verification.json).
The owned browser, quickstart frontend, and backend were stopped after UAT.
Synthetic runtime data remains outside the repository; evidence contains no
cookie values or provider credentials.

### Verification and limits

- Final combined frontend run: **214 tests passed across eight affected suites**
  under Node20 and the CI deterministic configuration.
- Persona manager, session detail, WebSocket, and dialogue-tree runtime:
  **147 targeted Python tests passed**. Cookie integration: **32 passed**;
  existing auth/resolver/JWT/API-key/WebSocket coverage: **78 passed**.
  Resource Governor: **15 new tests and 19 existing tests passed**. These scopes
  overlap with other recorded runs and are not a unique aggregate count.
- Scoped frontend lint has zero errors and no new changed-line findings.
  Fatal Python lint passes; full lint retains documented baseline findings.
  Bandit is clean for the three Persona production files and Governor change;
  the legacy auth file retains three pre-existing low-severity B106 findings.
- Repository-wide frontend typechecking still fails with **80 errors in six
  unchanged Presentation Studio/skills certification files**, and no touched
  file diagnostics. No full test sweep was run.
- Server provider replies and live microphone/STT/TTS remain unaccepted. The
  actual text outcome here was a tool plan. Buddy voice capability remains
  false; a connection's voice configuration notice is not audio evidence.
  OpenAI credentials and intentional microphone input remain prerequisites
  for the outstanding live voice UAT.
