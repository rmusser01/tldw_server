# I1 native editor history qualification

Executed 2026-10-06. **Web adoption blocked; Textual qualified independently.**

ADR required: no new ADR for this experiment.
Governing decisions: server `Docs/ADR/031-notes-capability-sync-domains.md` and
approved Chatbook `backlog/decisions/194-structured-note-language-and-local-editor-validation.md`.
The experiment does not change save ownership. A replacement editor needs revised approval.

## Real controlled Notes result

Production `NotesEditorPane` textarea, its `value={content}` and
`NotesManagerPage.handleEditorChange` were retained. A temporary button invoked
`execCommand('insertText', false, replacement)` synchronously, preserving focus
with `onMouseDown.preventDefault()`. The callback supplied to the candidate was a
no-op: the real React change handler alone maintained controlled state.
The button patch is retained as a test fixture, **not applied to product code**.

Each browser used a fresh API-created note; it was opened through the Notes list,
then deleted through the API using its latest expected-version header. Only the
owned note was deleted. No unavailable-server skips were accepted.

| State | Exact content |
| --- | --- |
| A, loaded | `{"a":1}` |
| B, normal typing | `{"a":1} ` |
| C, fixed format | `{\n  "a": 1\n}` |
| D, normal typing | `{\n  "a": 1\n} ` |

| Actual browser | Undo D | Undo C | Redo C | Undo earlier typing |
| --- | --- | --- | --- | --- |
| Chromium 145.0.7632.6 | C | B | C | A |
| Firefox 146.0.1 | C | B | C | A |
| WebKit 26.0 | **A (wrong; expected C)** | not reached | not reached | not reached |

The actual macOS shortcuts were Meta+z and Meta+Shift+z (`ControlOrMeta` in
Playwright); no script called undo or redo. WebKit coalesced prior typing,
formatting and subsequent typing. The retained fixture rerun reproduced the same
failure: **2 passed, 1 failed, 32.1s** (initial run 1.5m including cold compilation).

Runtime: Textual 8.2.8; Python Chatbook 3.12 and server 3.12.13; Bun 1.3.2;
Playwright 1.58.0; React/React DOM 18.3.1; Next 16.1.4; TypeScript 5.9.3;
Vitest 4.0.18. Firefox build1509 and WebKit build2248 were installed with
`bun x playwright install firefox webkit` into the standard Playwright cache.

## Retained failure harness (not production coverage)

- `apps/packages/ui/src/components/Notes/__tests__/fixtures/native-textarea-format-candidate.ts`
  is the **unqualified** candidate, deliberately outside production imports.
- `apps/packages/ui/src/components/Notes/__tests__/structured-editor-bridge.test.ts`
  has six passing guard tests. jsdom mocks only the unavailable native command;
  it does not claim native history qualification.
- `apps/tldw-frontend/e2e/fixtures/notes-native-format-qualification.patch` temporarily
  wires the real page to the candidate.
- `apps/tldw-frontend/e2e/notes-structured-editing.spec.ts` is opt-in with
  `TLDW_I1_QUALIFICATION=1`; otherwise explicitly skipped because there is no
  production formatter yet. A skipped run is **not** a qualification pass.
- `notes-firefox` and `notes-webkit` projects match only this file.

From server checkout root apply the patch with `git apply`, run the command below
from `apps/tldw-frontend`, then remove it with `git apply -R`. The recorded run
used these exact disposable endpoints/key:

```sh
TLDW_I1_QUALIFICATION=1 TLDW_WEB_URL=http://localhost:18080 \
TLDW_SERVER_URL=http://127.0.0.1:18081 \
TLDW_E2E_API_KEY=i1-native-history-disposable-key-2026 \
TLDW_WEB_AUTOSTART=false TLDW_LIVE_TIER_UAT=1 \
bun x playwright test e2e/notes-structured-editing.spec.ts \
  --project=chromium --project=notes-firefox --project=notes-webkit \
  --workers=1 --output=/private/tmp/i1-history/repro
```

Raw logs: `/private/tmp/i1-history/browser.log`, `browser-repro.log`.
Failure video: `/private/tmp/i1-history/repro/notes-structured-editing-f-36c90-n-the-controlled-Notes-page-notes-webkit/video.webm`.
Those temporary artifacts may expire; the exact trace above is durable evidence.

## Isolated live stack

Server checkout: `/private/tmp/tldw-server-structured-document-editing`.
Server interpreter: `/private/tmp/tldw-structured-server-venv/bin/python`.
Private source binding is unchanged; shared environments were not installed into
or rebound. First launch failed because `tldw_profile_core` was missing; the
second included the checkout's source on the process-local import path:

```sh
PYTHONPATH=/private/tmp/tldw-server-structured-document-editing/packages/tldw_profile_core/src \
TLDW_ENV_FILE=/private/tmp/i1-history/empty.env TLDW_ENV_FILE_EXCLUSIVE=1 \
MINIMAL_TEST_APP=1 AUTH_MODE=single_user \
SINGLE_USER_API_KEY=i1-native-history-disposable-key-2026 \
DATABASE_URL=sqlite:////private/tmp/i1-history/data/users.db \
USER_DB_BASE_DIR=/private/tmp/i1-history/data/users \
ALLOWED_ORIGINS=http://localhost:18080 DISABLE_BACKGROUND_TASKS=1 \
/private/tmp/tldw-structured-server-venv/bin/python -m uvicorn \
  tldw_Server_API.app.main:app --host 127.0.0.1 --port 18081
```

The empty env file was created deliberately. `MINIMAL_TEST_APP` still starts
some services (including embeddings); `DISABLE_BACKGROUND_TASKS` is not evidence
that every startup hook is disabled. The logs show startup completed and actual
Notes API requests. Optional Notes task/activity endpoints returned 404; core
Notes create/read/delete and the editor remained live. No response mocking.
Frontend from `apps/tldw-frontend`:

```sh
NEXT_PUBLIC_API_URL=http://127.0.0.1:18081 NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced \
bun run dev -- -p 18080
```

Owned sessions: backend exec96332, frontend exec68496. Logs: `backend.log`,
`frontend.log` under `/private/tmp/i1-history`. Disposable data and cache are
under that directory; Next's build output stays in the isolated checkout.
Cleanup: sent SIGTERM only to owned frontend PID6571 and backend PID6426 after
qualification. Shared servers were not stopped. Disposable evidence/data are
retained under /private/tmp/i1-history for review.

## Scope of checks and remaining gaps

Guard suite: stale current identity (all six key fields), newer capture replacing
old snapshot, widget text changed before host revision, read-only/disabled/conflict,
composition start/end, lost focus, detached/disposed editor, absent command,
no-op, selection direction, scroll assignment and command failure. Candidate
captures one frozen request, not a snapshot history stack. Linear non-whitespace
anchor mapping is O(n) and uses UTF-16 offsets in the DOM.

Live qualification proved **history and focused typing**, not a complete real-IME,
large-document scroll, conflict navigation or keyboard-toolbar accessibility
qualification. Those remain gates for any replacement. Composition guards used
synthetic events in unit tests, not an OS IME. No production browser bridge remains.
The six unit tests passed; targeted TypeScript compilation passed with no output.
Node emitted existing localStorage experimental and module.register deprecation
warnings; the test output is not claimed pristine. Bandit is not applicable to
the JS-only change scope and is absent from the private server interpreter.

## Smallest replacement proposal — awaiting revised approval

Qualify the already-installed Monaco stack **only for structured source modes
JSON/JSONL/YAML**. Keep Markdown editing unchanged. Reuse the lazy component,
editor/model handles and disposal seams demonstrated by
`src/components/Option/PresentationStudio/StandaloneHtmlSourceEditor.tsx`.
Do not reuse its HTML validation/save veto or setValue rollback policy for syntax
errors: structured syntax must remain saveable. Installed packages already include
`monaco-editor`0.55.1, `@monaco-editor/react`4.7.0, loader1.7.0.

Explicitly configure and prove **bundled/local assets and workers** in both Next
and extension builds; no silent default remote-CDN loading. No existing source
loader.config/MonacoEnvironment configuration was found. Preserve the programme's
future local engine/worker contracts and disable competing built-in document
validation/format providers. Dispose note models and listeners at authority/session
changes; no model/history crosses documents or users. Keep the same captured-source,
full-key, composition, focus, conflict and editability guards.

Apply formatting through a native Monaco edit transaction bracketed by undo stops,
with selection/view-state restoration. Repeat the actual mounted Notes
A/B/C/D → undo → undo → redo → earlier-typing trace on all three browser engines,
plus real IME/focus/readonly/navigation/selection/scroll tests. No custom snapshot
undo stack. If local loading fails, ordinary text editing may remain available,
but formatting must not silently fall back to this failed candidate.

A smaller **unproven** alternative is a synchronous native focus/selection boundary
around insertText to force WebKit to close its typing group. Current evidence only
proves that select-all + insertText + selection restoration while retaining focus
fails. It does not prove that every native boundary strategy is impossible. Such
an experiment must preserve focus/IME and pass the same live trace; do not infer
success from jsdom, state assignments, delays or a bare DOM editor. No such
correction or Monaco adoption has been implemented.

## Authorized single native-boundary follow-up — also failed

Executed 2026-10-06 after the controller requested one smaller native correction
before proposing widget replacement. The test-only candidate now retains this
exact variant for reproduction; the preceding retained-focus candidate is in
commit `f61e76ceb3`.

After the original source/key/editability/composition/focus guards, the candidate
synchronously calls `textarea.blur()` then `textarea.focus({preventScroll:true})`.
It then rechecks the captured request object, current text, composition, disposal,
attachment, focus and native readonly/disabled state before select-all/insertText.
After the native edit and ordinary callback, it repeats blur/focus, then restores
mapped selection/direction and scrolling. The actual toolbar still prevents its
pointer focus transfer. No timer, artificial keypress, programmatic undo or
additional snapshot stack was introduced.

**WebKit26.0 still fails the first actual Meta+z:**

```json
{"browser":"26.0","A":"{\"a\":1}","B":"{\"a\":1} ","C":"{\n  \"a\": 1\n}","D":"{\n  \"a\": 1\n} ","undoD":"{\"a\":1}"}
```

The run reached the formatted state and passed the editor-focused assertion;
failure was at the same Undo D → C assertion, not startup or admission.
Command: the earlier opt-in command, restricted to `--project=notes-webkit`, with
`--output=/private/tmp/i1-history/boundary`. Raw log:
`/private/tmp/i1-history/browser-boundary.log`; failure video is beneath that
boundary output directory. Six unit guard tests passed (1.78s); targeted tsc
passed without output. These cannot override the live history failure.

Per the one-experiment instruction, no further native variant or all-browser rerun
was attempted after failure. The temporary NotesEditorPane patch was removed;
Monaco remains unimplemented and unapproved. This supersedes the earlier statement
that a synchronous blur/focus boundary was untested: **this exact boundary on both
sides is now disproved by the controlled Notes trace**. Other native variations
are not claimed impossible, but further exploration is outside this experiment.

Same isolated data/env as above; restarted owned sessions backend14845,
frontend60234 (processes32968 and33547). Both were stopped after recording evidence.
Logs: backend-boundary.log, frontend-boundary.log, u-boundary.log under the same
private evidence directory. No shared server or user note was touched.
