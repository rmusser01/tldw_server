# Migu Buddy UAT — 2026-09-05

Current verdict: **Chatbook physical dragging passes. Server Buddy setup, visuals, viewport containment, and browser transport are repaired; end-to-end conversational usability remains unaccepted.** The initial failures below are preserved as before-fix evidence. See the repair follow-up for implementation and verification.

## Runtime and method

Chatbook: merged PR #2404 code, commit f8cb939e2bd3a111555acc8d87a4b4907ee2268e, isolated profile and native macOS Terminal. Foreground mouse gestures were explicitly authorized. Server: dev 2742468a19, separate worktree `codex/migu-server-buddy-uat`, full FastAPI app at 127.0.0.1:9101 and Next development WebUI at 127.0.0.1:18384 in advanced deployment mode. Locked Bun dependencies installed. Fresh synthetic single-user account and separate databases. Real REST/WebSocket requests; no browser route mocks, mocked replies, fake microphone, or mocked provider.

The previously running frontend at 18383 had a missing dependency and the backend at 9099 omitted Persona routes. Neither was changed or counted as a Buddy product failure. The fresh backend required the current in-repository profile-core/MCP packages on PYTHONPATH. No full test suite was run.

## Chatbook results

- Physical move: rendered region (41,31,28,15) → (69,25,28,15), with native mouse-down, multiple moves, and mouse-up observed by production handlers.
- Physical lower-right resize: rendered size 28×15 → 40×21; release delivered successfully.
- Graceful exit: no app exception. Fresh PID 46565 restored rendered geometry (69,25,40,21). Preferred size is stored separately from the current rendered fit.
- Separate PTY protocol probe: all 22 checks passed, including move, resize, capture release, boundary clamp, modal/navigation transitions and restart restore.
- Background per-PID dragging produced no application events; foreground gestures resolved that automation limitation without a code change.
- The long-running first harness detected normal config changed since the prior-day baseline; this interval does not support an unchanged-config claim. The fresh restart baseline remained unchanged. Runtime database paths were isolated.
- OpenAI realtime verification remains outside this completed physical check and is still open in Chatbook TASK-31585.

## Initial server acceptance matrix (before repairs)

| Journey | Result | Evidence |
|---|---|---|
| Load real Persona page and create Migu UAT | Pass | New profile e19c631e-e2ac-452d-9935-293af03cee4e created through UI |
| Complete voice-default setup | Fail | Repeated 409 optimistic-version conflicts; TASK-13174 |
| Open Buddy at default desktop position | Fail | Composer/navigation entirely below viewport; TASK-13175 |
| Drag Buddy and restore position | Pass | Native browser pointer gesture moved shell to (704,17); reload/reselect restored it |
| Select bundled Migu Marker Basic, copy draft and activate | Pass for backend publication | Active pack 30a938ed-3fbf-4232-a41f-8f6d372f036b; authenticated PNG GET 200, 9617 bytes |
| Render activated Migu in WebUI | Fail | Relative image URLs request frontend origin and repeatedly 404; TASK-13176 |
| Start and Stop live session from Buddy | Pass | Idle appears on Start; Stop becomes disabled after stop |
| Send real Buddy text from browser | Fail | Stream connection error, draft retained, no WebSocket created; TASK-13177 |
| Backend direct stream control | Pass for handshake/planning only | Authenticated WebSocket returns WS_CONNECTED notice and tool_plan for synthetic input; test session stopped with HTTP 200 |
| Conversational answer/provider quality | Not verified | Browser send blocked. Direct stream returns a RAG tool plan, not the requested conversational answer; no plan execution was approved |
| Voice | Unavailable, gating correct | Real session advertises capabilities.voice=false; shell offers no voice button. No microphone, speech, or provider UAT pass claimed |
| 390×844 web viewport | Current desktop-only policy observed | Buddy unmounts under useDesktop guard; builder remains visible. Not a mobile Buddy pass |

## Original defects and fix order

1. **TASK-13177 — Stream lifecycle after Strict Mode remount.** Start succeeds; Send repeatedly returns “Persona live stream failed to connect” without opening a WebSocket. `usePersonaLiveControl.tsx:183` cleanup sets mountedRef=false, and effect setup never restores it. `ensureStreamSocket` then rejects before socket creation. Next config enables Strict Mode. This is a source-supported mechanism; no production fix was applied during this UAT. Re-run actual browser Send after correction, then validate incoming plans/replies and their visible controls.
2. **TASK-13174 — Setup version handoff.** Create Migu → Save assistant defaults: database version 2 versus expected 1. Reload/reselect/retry repeats version 4 versus expected 3. Voice saving and setup advancement need to hand off the current version while retaining optimistic conflict protection.
3. **TASK-13176 — Visual transport.** Copy Migu Marker Basic as draft → activate → browser requests `/api/v1/persona/.../assets/.../content` on port 18384. Advanced mode has no same-origin API rewrite. The identical authenticated asset endpoint on 9101 returns image/png 200. Failed frames are retried repeatedly while animation runs. Resolve asset transport/auth consistently for both builder and shell, then retest animation and state changes.
4. **TASK-13175 — Expanded-shell containment.** At 1280×720, initial expanded dock (1104,609,220,687) and popover (1104,818,220,478) exceed the viewport. Normal click on Choose/Change Buddy times out as outside viewport. Dragging upward makes it usable. Host clamping watches position and window resize, but not expanded/content size changes. Retest expansion, errors, loaded visuals and reload without manually rescuing the position.

## Evidence

[Clipped controls](assets/migu-buddy-uat-2026-09-05/popover-clipped.png), [setup conflict](assets/migu-buddy-uat-2026-09-05/setup-conflict.yml), [failed browser send](assets/migu-buddy-uat-2026-09-05/ui-sent.txt), [stopped session](assets/migu-buddy-uat-2026-09-05/ui-stopped.txt), [backend stream frames](assets/migu-buddy-uat-2026-09-05/direct-stream-frames.json), [session capabilities](assets/migu-buddy-uat-2026-09-05/live-session.json).

The initial UAT created TASK-13174–13177 without production changes. Its browser and synthetic sessions were stopped before the repair pass. Chatbook follows existing ADR-074.


## Repair follow-up

The isolated `codex/migu-server-buddy-uat` branch repairs six concrete failures:

- **TASK-13174:** pass the saved profile version explicitly to the setup checkpoint; retain genuine optimistic conflicts and ignore stale save completions after persona selection changes.
- **TASK-13175:** clamp expanded/content-resized Buddy bounds before paint, constrain dock height, and scroll the compact controls.
- **TASK-13176:** fetch server-owned visual content through the existing authenticated binary transport, render disposable object URLs, retain failures until sources change, and abort/revoke on cleanup. Both sprite frames and generated candidate thumbnails share this path; external asset URLs do not receive credentials.
- **TASK-13177:** restore Strict Mode mount readiness and fence stale asynchronous session/stream work.
- **TASK-13178:** real Chromium then exposed the server's missing WebSocket subprotocol selection. After successful authentication, select only the offered bearer marker, never its credential. Invalid authentication remains rejected.
- **TASK-13179:** actual quickstart startup exposed a missing public `/health` rewrite. Next now forwards that path only in quickstart mode to the configured backend's public endpoint.

These are routine repairs to existing contracts; no new ADR is required. TASK-13180 records the separate stream-outcome/approval feedback gap and TASK-13181 records cookie-authentication coverage.

### Real browser evidence

Fresh synthetic persona `e0a442a5-3861-4529-a332-a5391626f51f` (Migu UAT Repaired) saved defaults and advanced to Starter commands, then Safety, without the previous 409. [Setup evidence](assets/migu-buddy-uat-2026-09-05/repaired-setup.txt).

Migu Marker Basic was copied and explicitly activated as pack `559d3eb8-9bde-4e0c-b01e-46440708e5a7`. Builder and Buddy frames decoded at 96×96 from blob URLs backed by authenticated PNG 200 responses on backend port 9101. [DOM/network evidence](assets/migu-buddy-uat-2026-09-05/repaired-desktop.json), [desktop screenshot](assets/migu-buddy-uat-2026-09-05/repaired-desktop.png).

At 1280×720 the expanded dock was `(938.55,229,325.45,416)`, bottom645. At1280×360 it was `(938.55,16,325.45,328)`, bottom344. The popover scrolled 112px and the bottom navigation link ended at328px, within360px. [Small viewport geometry](assets/migu-buddy-uat-2026-09-05/repaired-short-viewport.json), [scroll proof](assets/migu-buddy-uat-2026-09-05/repaired-scroll-proof.json), [screenshot](assets/migu-buddy-uat-2026-09-05/repaired-short-viewport.png). The small-height screenshot intentionally retains the intermediate handshake error; it demonstrates readable error and navigation containment before TASK-13178 was loaded.

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

TASK-13174,13175,13177,13178,13179 are Done. TASK-13176 remains In Progress because its quickstart acceptance is blocked by TASK-13181. TASK-13180 and13181 are recorded as high-priority follow-ups.

Final real pointer drag moved the repaired dock from `(938.55,16)` to `(738.55,56)`; the asynchronous Stop completed and disabled its control. [Drag evidence](assets/migu-buddy-uat-2026-09-05/final-drag.json), [completed stop](assets/migu-buddy-uat-2026-09-05/final-stop.json). The initial immediate post-click stop flag is false because the API was still pending; the separate completion evidence is true.

The isolated UAT processes and browsers are stopped after verification. Synthetic runtime data is retained outside the repository. The repair pass was subsequently published as [server PR #2884](https://github.com/rmusser01/tldw_server/pull/2884) against dev63358431d7. All265 focused frontend and54 backend tests passed again after rebase; repository-wide typechecking still reports80 unrelated diagnostics. The PR is a draft while the collision between the Buddy and older EPUB TASK-13174 awaits the repository-required exception for renumbering Buddy to TASK-13182. No merge is included. [Chatbook PR #2418](https://github.com/rmusser01/tldw_chatbook/pull/2418) publishes the separate native UAT evidence against its dev branch.
