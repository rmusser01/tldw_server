# Buddy v1 qualification — 2026-09-09

TASK-13227 is in progress. This report distinguishes browser interaction, automated contracts, and build evidence. No physical microphone or external model account was used.

## Source and environment

- Server/WebUI base: `1fc19c7c8384f38eca2becdf53fb8bbcd205be7a`, branch `codex/buddy-v1-live-qualification`. The initial qualification repaired optional token-limit parsing and readable saved Buddy errors. The subsequent usability repairs are recorded in the dated follow-up section below.
- Contract: [ADR-005](../../backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md).
- Disposable full server profile: `/private/tmp/buddy-v1-qualification-6_vfinnl`; authenticated single-user backend on loopback18181; WebUI on18180. Config and all observed open SQLite databases are under the disposable profile. User configuration and credentials were not loaded. Backend identity/diff hashes and logs are retained there.
- The WebUI uses the actual backend and built-in assets. A repository mock OpenAI provider on loopback18182 supplies controlled text for reply checks; it is not a real-model quality or voice test.
- Browser actions use the Codex in-app browser. Screenshots were captured and visually inspected inline in the task; no filesystem WebUI PNG is claimed. Cold development-route compilation caused transient loading states; those are not reported as settled product failures.

## Browser qualification

1. Fresh onboarding supports skipping provider/assistant setup. Console exposes **Buddy & Persona** and reports **Conversation Persona: None**.
2. Bundled Pixel Migu is discoverable by search, with an artwork preview. Before any conversation exists, attachment stays disabled and explains how to create one; workspace attachment is offered as an alternative.
3. Opening Research Workspace creates the fresh **New Research** workspace. The Buddy manager selects that workspace, offers **None** or **Research Assistant** as future-conversation defaults, and explains that existing conversations retain their identities.
4. Applying Pixel Migu with **None** creates independent artwork and a workspace attachment. The actual cyan sprite remains visible after navigation to Watchlists. Clicking it opens **pixel-migu — New Research** in place, with an explicit conversation selector and read-aloud checkbox. Workspace mode exposes no microphone input.
5. Expanding **Buddy options** exposes **Expressions**, **Reset position**, **Manage Buddy & Persona**, and **Detach Buddy**. Static persists after closing and reopening the modal. Dynamic subsequently advances the actual rendered image source. Keyboard repositioning moves the Buddy within the viewport.
6. Changing the workspace default to Research Assistant and cancelling preserves None on reopening. Applying the same selection saves the future-conversation default.
7. A new standalone Chat conversation retained Persona None. Attaching the Buddy to its exact ID, leaving Console for Watchlists, and opening the Buddy showed the saved transcript and a scoped reply field.
8. The real server accepted Buddy turn `7de5e300e402420eac350595de6a9ea2` at05:09:37UTC. While the local provider deliberately waited20seconds, the UI showed Working; the modal was closed and the page changed from Watchlists to Research Workspace. Reopening the Buddy showed the completed mock response. The server receipt records completion at05:09:57 and the same conversation `84d83245-a607-4883-ac51-2aff63cc0931`, attachment version3. Receipts are retained as `navigation-turns.json` and `navigation-attachment.json` in the profile directory. This verifies accepted Buddy work across navigation, not process restart durability.
9. Two new workspace conversations created through the Research Workspace send/retry flow inherited `assistant_kind=persona`, `assistant_id=research_assistant`, and `persona_memory_mode=read_only`. The actual scoped listing is retained as `workspace-conversations.json`. Normal-speed controlled streaming completed successfully after the artificial provider delay was removed.
10. Back on Watchlists, the workspace Buddy showed two unread results. Selecting conversation `c71cc7fc-7cd6-4463-b801-7acc6391f4a0` and sending a scoped reply added exactly two messages there (five to seven); sibling `5c26dc31-e2b9-4438-9d28-e772ca385622` stayed at five. The post-send scoped listing is retained as `workspace-conversations-after-reply.json`.
11. Marking the selected conversation's result read changed the visible badge from two to one and retained the other result row. `workspace-activity-after-ack.json` confirms the selected result acknowledged and the sibling unacknowledged. This verifies result targeting and acknowledgement; audible queue delivery was not exercised.

### Additional usability observations

- At 1090×990, the initial fixed Buddy position overlaps Console's Send/drag-control area. Moving it upward with Shift+ArrowUp restores unobstructed pointer access. [TASK-13228](../../backlog/tasks/task-13228%20-%20Keep-the-default-Buddy-placement-clear-of-primary-composer-controls.md) tracks a safe default placement that preserves intentional user positioning.
- Both workspace conversations had the same generated title. The selector routes to distinct IDs correctly, but its visible/accessibility labels and result rows do not distinguish them. [TASK-13229](../../backlog/tasks/task-13229%20-%20Distinguish-same-titled-conversations-in-the-Buddy-workspace-inbox.md) tracks distinguishing context.
- Newly generated workspace chats had no usable saved provider/model for a Buddy reply. Sending preserved the draft and reported "Choose a Chat provider and model before sending." Filling the collapsed optional override fields recovered successfully. [TASK-13230](../../backlog/tasks/task-13230%20-%20Make-Buddy-reply-model-recovery-clear-for-workspace-conversations.md) tracks clearer recovery and settings handoff.
- Artificially slow model output triggered the existing no-visible-output timeout. Standard Chat showed a readable error, whereas Buddy history initially exposed the saved `__tldw_error__` JSON envelope. A scoped presentation repair reuses the existing decoder; the underlying timeout is not evidence of a Buddy runtime failure.

The three linked usability issues remained open at the end of the initial qualification. The subsequent follow-up below records their implementation separately from these original observations.

## Reproduced defects and repairs

The shipped config contains blank optional `*_max_tokens` settings. Catalog construction called `int('')`, caught the exception at the whole-catalog boundary, and returned no providers. The live WebUI therefore displayed no models even though the custom provider was configured.

The repair treats blank/whitespace optional limits as unset. Valid integers remain exposed; malformed nonblank values retain the existing sanitized error contract. A new regression failed before the repair (one failure/two passes), then passed. Five focused/adjacent tests passed; Bandit and whitespace checks passed. The new test passes Ruff lint/format; whole-file source Ruff/format debt was confirmed unchanged from HEAD.

After restarting the real server with the **unchanged original blank configuration**, its catalog returns the configured `buddy-qualification` model. Refreshing WebUI model settings displays usable providers and permits selecting that model. The scratch configuration was not altered to hide the defect.

Buddy transcript rendering and queued read-aloud now share a private formatter using the existing decoder for **assistant messages only**. Both receive summary and hint without the internal envelope or diagnostic detail; speech preserves the conversation-name prefix. User-quoted envelopes and ordinary multiline messages remain verbatim. Independent review caught the initially unfixed speech sink; its activity-to-authorized-read-to-speech regression failed before the correction. The final component/decoder gate passed **9 tests**, and re-review found no remaining issues. Prettier and whitespace checks passed. ESLint returned zero errors and the same six warnings verified against base HEAD. WebUI inspection of the visual repair on Watchlists showed readable summary/hint alongside the successful scoped reply; the final shared transcript/speech formatter is covered by the focused tests. Speech assertions use a mocked TTS boundary, not physical playback.

The final catalog verification retained raw evidence: **5 tests passed, 4 warnings in 1.17 seconds**, covering the three metadata cases and two adjacent sanitized-error contracts. Bandit on the touched endpoint reported no findings. The earlier catalog RED and artwork run were summarized by their worker without standalone raw logs; this report does not claim those raw logs were retained.

## Backend and upgrade contracts

The independent Buddy, turn ledger and turn suites passed **51 tests, 1 skipped** in 113.78 seconds. The SQLite upgrade case creates an on-disk schema 65 database, inserts a conversation, closes it, reopens at schema 66 and checks preservation plus all six Buddy tables. The rollback case injects invalid migration DDL and verifies schema 65 remains without partial Buddy tables. This covers a synthetic predecessor database with one preserved conversation, not an archived production dump or a complete upgraded-profile WebUI journey.

The remaining cases cover owner isolation, immutable copied artwork, stale/foreign/deleted targets, exact receipts, authenticated Chat admission, accepted FIFO continuation across transport cancellation, idempotency and late-publication fencing. The provider is controlled. PostgreSQL was unavailable; the named migration case's explicit no-Docker rerun confirms the skip and supplies no PostgreSQL qualification. Commands and limits are retained in [backend evidence](artifacts/buddy-v1-13227/backend-qualification.md).

## Artwork incident

The historical TASK-13211 legacy Persona pack/session request loop has not been reproduced. The independent host's initial Buddy/attachment requests matched its five-second polling interval. Observed Buddy requests before the catalog repair returned200/201, with no429; actual artwork assets loaded successfully across workspace navigation and motion changes.

The source investigation found that repeated paired legacy pack and live-session requests require a legacy host remount or normalized Persona/surface changes. There is no 250 ms loader retry. The independent host landed after the historical incident and cannot be its original cause. Three focused existing suites passed **81 tests**. The retained [investigation](artifacts/buddy-v1-13227/artwork-investigation.md) describes the remaining integrated lifecycle evidence needed; no speculative artwork fix was made.

The final running-process request snapshot contains 2,723 Buddy requests: 2,720 HTTP 200, two accepted turns (202), and one missing-model rejection (422). No 429 appears in that snapshot. The [aggregate](artifacts/buddy-v1-13227/buddy-request-summary.json) does not establish a polling interval or reproduce the historical legacy-host incident; TASK-13211 remains In Progress.

## Packaged extension and remaining coverage

A clean archive of the pinned server commit was installed using the frozen Bun lock in `/private/tmp/tldw-buddy-chrome-prod-zy5ty355`. The baseline production Chrome build and token synchronization succeeded. The final build overlays exactly the corrected `BuddyInteraction.tsx` component; it is base plus patch, not byte-identical base source. The lock remains unchanged.

The final `bun run build:chrome:prod` completed in 43 seconds with token synchronization, 1,378 files, manifest-target checks and ZIP integrity verified. Load-unpacked directory: `/private/tmp/tldw-buddy-chrome-prod-zy5ty355/apps/extension/build/chrome-mv3`. ZIP: `tldw-chrome-production-1fc19c7-buddy-friendly-error-speech.zip`, SHA-256 `1375a51f13cb851ca3e12c277b0da2b7e6da19f510038382fe3c6d6c778048e8`. The UI patch SHA-256 is `5cb801894f1d0e4acedc22251d8a8888667fa7aa9e00560340166c75130968df`. The [build report](artifacts/buddy-v1-13227/chrome-build-report.md) records provenance; full build logs and file manifests remain in the build directory. Baseline and previous visual-only ZIP evidence is preserved separately.

Native Chrome automation reports **Computer Use permissions are not granted**. Terminal automation is explicitly disallowed by the computer-use tool. These prevent the packaged-extension installation walkthrough and physical native Chatbook verification; the build and headless checks do not replace them. No workaround was attempted.

Chatbook's separate report records **177 passing targeted checks**, fresh and synthetic 69→70 upgrade journeys, actual built-in Static/Dynamic frames, and conversation/workspace contracts. No Chatbook production change was needed.

Real microphone input, audible queue output, real-provider quality, native Chatbook interaction, packaged-extension installation, and the server's upgraded-profile UI journey remain unqualified. The fresh WebUI workspace reply and acknowledgement checks above are complete. TASK-13227 remains In Progress because the outstanding native/upgrade coverage prevents full acceptance. No full local test suite was run.

## Retained evidence

[Evidence manifest](artifacts/buddy-v1-13227/manifest.json) records hashes for the curated receipts, investigation reports, focused-test logs and build report beside this file. Repository log copies normalize trailing whitespace and excess blank lines at EOF; original and retained hashes are recorded separately, with raw logs kept in scratch storage. The full disposable profile and clean extension build remain in their scratch directories. These contain synthetic qualification content; user profiles were not used.

The owned WebUI, backend and mock provider processes were stopped after the live walkthrough. Their profiles/logs remain available, and the temporary Next build output was moved into the disposable profile. No user-run process was stopped.


## Usability follow-up — TASK-13228, TASK-13229, TASK-13230

PR [#2934](https://github.com/rmusser01/tldw_server/pull/2934) now includes all three repairs. Existing ADR-005 applies; no new database schema, provider credential store, or work-ownership boundary was introduced.

- Fresh/reset position moves to y96, below the top navigation. Saved placements remain unchanged. The live WebUI at 1090×991 reported Buddy bounds (942,96,128,172); actual DOM `elementFromPoint` checks found the composer input and Send center unobstructed after Home reset. Focused store/layout tests cover desktop/compact clamping and preservation of saved positions. Compact pointer interaction was not driven in the live browser. Floating artwork can still cover other page content; its existing drag/keyboard controls remain available.
- Same-titled loaded conversations use creation time before the title, with a stable identifier for missing or matching timestamps. Picker, result rows, transcript/reply labels, work status, and speech prefixes use the same derivation; saved titles and routing IDs stay unchanged. The API now includes creation timestamps. Independent review caught nullable timestamps rendering as Unix epoch; a failure-first regression now requires the stable-ID fallback for null and absent values.
- Ordinary neutral workspace Chat saves an explicitly selected provider/model on that owned conversation. A Buddy-scoped read projects the same effective settings used for acceptance. Missing settings open the required fields before Send; valid settings are shown, drafts persist during recovery, and temporary Buddy overrides do not replace conversation defaults. The new read checks the captured connection inside a transport-config factory and pins that configuration through WebUI/extension dispatch; a deferred connection-change regression proves no request reaches the changed account/server.

Live recovery was driven against copied synthetic qualification data in `/private/tmp/buddy-ux-followups-20260909`: the required fields appeared, the draft survived both selections, and turn `156c528764bb46ffaa23d5a7f4e7c1cf` completed in the selected conversation `c71cc7fc-7cd6-4463-b801-7acc6391f4a0`. The live selector/result labels were distinct. These visual checks preceded the final timestamp-null and pinned-transport review corrections; final source hashes and targeted regressions cover those corrections. The subsequent new-workspace browser handoff walkthrough was not completed after disposable connection reconfiguration caused credentials/redirect recovery problems. Canonical Chat-to-Buddy handoff is covered by the real HTTP/SQLite regression, not claimed as a completed browser journey.

Final focused gates: **94 UI tests passed**; **58 backend tests passed, one PostgreSQL availability skip**; **12 adjacent Persona compatibility tests passed** with the test-only mock-provider switch. Production Bandit reports zero findings/errors. ESLint reports zero errors on the root-owned files and existing warnings; the shared position test's existing lint debt is compared separately. Focused typechecking reports no diagnostics in the six changed entrypoints, with 52 dependency diagnostics outside them; this is not a project-wide typecheck pass. A re-review closed both findings. Full local suites were not run.

[Follow-up evidence](artifacts/buddy-ux-followups-13228-13230/manifest.json) retains raw/normalized hashes, source hashes, test logs, static checks, and the completed live turn receipt. Original qualification artifacts above remain historical evidence. Native Terminal, extension installation, upgraded-profile WebUI, and real audio remain open under TASK-13227/32108/13202.

The reviewed Chrome production build completed in 49.5 seconds using the unchanged frozen lock and six exact shared-UI overlays. Manifest targets and ZIP integrity passed (1,378 files). ZIP: `/private/tmp/tldw-buddy-chrome-prod-zy5ty355/tldw-chrome-production-buddy-ux-reviewed.zip`, SHA-256 `92a62668173f71804902357980cabb45e67b4d2503a97ff93f047d9b4ccf8ad5`. [Build inputs and receipt](artifacts/buddy-ux-followups-13228-13230/chrome-build.json) identify every overlay. Native installation remains unverified.

The follow-up backend, WebUI, and mock provider were stopped after verification. Their disposable profile and build output remain in scratch storage.
