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

## 2026-10-07 — Installed Monaco candidate qualification (I1 only)

The human selected the installed Monaco experiment (option 1). This section
supersedes only the earlier statement that Monaco was unimplemented. The controlled
textarea remains **unqualified**: its documented WebKit native-history failure is
unchanged. The already reviewed C Textual bridge is unchanged and was not retested.
Tasks 34570/13512 remain In Progress, with no acceptance criteria closed.

**Result:** the test-only Monaco candidate passes the real Next Notes page's
native keyboard history and representation controls described below in Chromium,
Firefox and WebKit. It earns further I3 integration consideration, not production
adoption. Actual OS IME input and extension Notes/history are not qualified.
Temporary Notes product wiring and generated public assets were removed before
commit; the reproduction patch, fixtures and evidence remain.

ADR required: existing ADR applies, with an experiment-boundary extension.
ADR paths: C `backlog/decisions/194-structured-note-language-and-local-editor-validation.md`
(as referenced by the task's existing plan) and S
`Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md`.
The latter was extended before the candidate's runtime/asset contract. No new
dependency, production runtime policy, storage normalization or save veto was added.

### Candidate and consumers

The source-only candidate lives under UI `Notes/__tests__/fixtures`, with an opt-in
real-page Playwright spec. Its temporary patch uses the existing NotesEditorPane
and NotesManagerPage owners and routes every model change into the actual
`handleEditorChange`. Markdown/WYSIWYG editing is unchanged. Authority, selected
note and language key the candidate host; cleanup retires pending results before
disposing the bridge, listener, editor and model. A real note switch checks a new
model URI, one remaining live model, disposed prior models, and refused late apply.

The bridge captures the existing frozen request type and requires all six key
fields across captured request/result/check/current state, exact current model
source, model identity, complete eligible check, full-document request, editable
current host and native editor, mounted DOM, text focus and no composition. It
rejects unchanged replacement without touching history. The host additionally
compares the current React draft against the model before applying. A real stale
version Save conflict removes editability and refuses a pending format while
retaining the draft. No HTML-editor validation/save gate was copied.

Formatting is exactly `pushUndoStop`, `executeEdits` with one full-range edit and
oriented end selections, `pushUndoStop`, then scroll restoration. There is no
second undo stack, setValue rollback, parser/re-serialization or timed undo trick.
Selection mapping is linear UTF-16 anchor mapping with unchanged literal whitespace
preservation, including backward selections and non-BMP offsets. Plaintext models
avoid installing a competing JSON/YAML formatter/validator. `editContext: false`
selects Monaco's textarea composition path; this is an installed public option.

The installed `@monaco-editor/react` loader is also consumed lazily by
PresentationStudio/StandaloneHtmlSourceEditor and Watchlists/TemplatesTab's
TemplateCodeEditor. The experiment configures its singleton before initialization
in fresh Notes contexts. **Prior-consumer loader initialization order is not
qualified**; I3 must resolve the shared initialization owner before adoption.
Same-note external replacement/reload ownership also remains an I3 integration
concern; this harness refuses source drift rather than inventing reset policy.

### Local assets and representation

The preparation script resolves the installed Monaco package, copies `min/vs` to
Next's actual `public/i1-monaco/vs` and WXT's configured shared UI
`src/public/i1-monaco/vs`, and creates static same-origin worker bootstraps for the
installed hashed workers. The candidate sets the loader path explicitly and
creates workers from same-origin URLs. No dependency installation, CDN, blob/eval
worker CSP allowance or external source-processing endpoint is required.

Real Next browser requests loaded `loader.js`, `editor/editor.main.js`, distribution
chunks/CSS and the editor worker. Chrome/WebKit recorded 13 asset requests;
Firefox recorded 12 because page request events omit its nested importScripts
request. Firefox's actual worker global contained the loaded `monaco` object;
worker resource entries were retained in the attachment. All observed candidate
assets were local; the spec found no CDN/jsdelivr/unpkg URLs and no lexeme-marker
request body outside the authorized Notes API. This is observed fixture traffic,
not a universal network noninterference proof. Source reaches the Notes API on
explicit Save through its existing handler.

Admitted profiles: exact homogeneous LF and CRLF, including newline-free input
(explicit LF). At document load the model EOL is set to the admitted source EOL,
and an exact `getValue(undefined, true)` round-trip is required before editable
mount. This load-only native EOL setting precedes user history. Application also
requires the model EOL and replacement profile to agree with the captured source.
Leading BOM, mixed LF/CRLF and lone CR are refused before editable mount. Their
original source is retained in a read-only preformatted textbox, with no editable
normalizing fallback. Browser tests compare exact original text and readonly state.

Exact lexeme fixture includes `a&b <tag>`, emoji, two literal spaces, escaped
`\\u0001`, `9007199254740993`, `1e400`, `-0`, a long decimal and YAML comments. It
survives the candidate transaction, actual Save, API GET and reload/reopen exactly.
The supplied replacement is deliberate fixture data, not an engine-formatting
claim. JSON and YAML source transport are exercised; engine JSONL semantics are
outside I1. The final viewport control measured top 900/left 120 before and after
formatting in all three engines (no arbitrary sleep).

### Native history and mounted controls

For every browser, LF A=`{"a":1}`, B=A plus a typed space,
C=`{\n  "a": 1\n}`, D=C plus a typed space. Actual native keys yielded:
D → Undo → C → Undo → B → Redo → C → Undo → B → Undo → A.
CRLF repeats the same chain with A=`{\r\n"a":1\r\n}\r\n` and
C=`{\r\n  "a": 1\r\n}\r\n`, preserving every CRLF throughout.
No direct editor undo API, React value assignment or synthetic history event is
used for these traces. Exact per-browser JSON traces are in
`/private/tmp/i1-monaco/browser-final-formatted.log` and Playwright attachments
under `/private/tmp/i1-monaco/final-formatted`.

Chromium 145.0.7632.6 and Firefox 146.0.1 use Desktop Windows user agents and native
Control+End/Control+z/Control+y bindings. WebKit 26.0 uses a Macintosh UA and
Meta+ArrowDown/Meta+z/Meta+Shift+z. The macOS host's navigator.platform is not the
Monaco platform authority. Mounted controls also cover all-key rejection, source
drift, no-op history, lost focus, native readonly, current editability, literal
whitespace selections in both directions, a caret between spaces, non-BMP caret,
selection restoration by native undo, real note navigation/disposal, optimistic
save conflict and Enter activation of the toolbar with focus restored.

Composition evidence is **mounted synthetic compositionstart/compositionend** on
the actual editor input, demonstrating pending format rejection during the
composition flag and admission after it ends. It is not actual OS IME text entry,
East Asian input-method acceptance or a universal composition-history claim.

### Reproduction and verification

Immutable candidate bases: C `dc9fcfb5d5e5b8444ec8fc92844baedcb2326382`,
S `abbd9c12b90020fb6230f8854a7b171017c2641e`. Commands below run from S unless a
cwd is shown. Only targeted tests were run, per the repository full-sweep rule.
Installed versions observed from W: Node 26.0.0, Bun 1.3.2, Vitest 4.0.18,
Playwright 1.58.0, Monaco 0.55.1, @monaco-editor/react 4.7.0, loader 1.7.0,
React 18.3.1, Next 16.1.4, TypeScript 5.9.3, WXT 0.20.13, Vite 5.4.21.
The handoff mentioned Node25; actual tested W processes used Node26. The retained
`--no-experimental-webstorage` Vitest flag is disclosed rather than hidden.

Reproduce only against an isolated live stack and synthetic data:

```sh
git apply apps/tldw-frontend/e2e/fixtures/notes-monaco-qualification.patch
node apps/tldw-frontend/e2e/fixtures/prepare-monaco-qualification.mjs
```

Backend cwd S, explicit empty exclusive env file and disposable SQLite directory:

```sh
PYTHONPATH="$PWD/packages/tldw_profile_core/src" \
TLDW_ENV_FILE=/private/tmp/i1-monaco/empty.env TLDW_ENV_FILE_EXCLUSIVE=1 \
MINIMAL_TEST_APP=1 AUTH_MODE=single_user \
SINGLE_USER_API_KEY=i1-monaco-disposable-key-2026 \
DATABASE_URL=sqlite:////private/tmp/i1-monaco/data/users.db \
USER_DB_BASE_DIR=/private/tmp/i1-monaco/data/users \
ALLOWED_ORIGINS=http://localhost:18080 DISABLE_BACKGROUND_TASKS=1 \
/private/tmp/tldw-structured-server-venv/bin/python -m uvicorn \
 tldw_Server_API.app.main:app --host 127.0.0.1 --port 18081
```

Frontend cwd W:

```sh
NEXT_PUBLIC_API_URL=http://127.0.0.1:18081 \
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced bun run dev -- -H 127.0.0.1 -p 18080
TLDW_I1_MONACO=1 TLDW_WEB_URL=http://localhost:18080 \
TLDW_SERVER_URL=http://127.0.0.1:18081 \
TLDW_E2E_API_KEY=i1-monaco-disposable-key-2026 \
TLDW_WEB_AUTOSTART=false TLDW_LIVE_TIER_UAT=1 \
bun x playwright test e2e/notes-monaco-qualification.spec.ts \
 --project=chromium --project=notes-firefox --project=notes-webkit \
 --workers=1 --output=/private/tmp/i1-monaco/final-formatted
```

Result: **36 passed (1.4m)** after source formatting (`browser-final-formatted.log`).
Then the sole viewport test was strengthened to require nonzero left=120 before
apply; the same command with `--grep 'exact lexemes'` and output directory
`/private/tmp/i1-monaco/viewport` gave **3 passed (14.9s)** (`browser-viewport.log`).
No product code changed after the 36-test run. Without the opt-in and applied patch,
tests intentionally skip; a skip is not qualification.

TDD: cwd W, command below initially failed because the candidate module did not
exist (`unit-red.log`); the browser command restricted to Chromium initially
failed because the real Notes page lacked the candidate (`browser-red.log`). These
are expected preimplementation failures. An earlier wrong-cwd setup failure was
not counted as RED. Later source-drift admission regression was added after the
bridge, so this is not a claim that every line was test-first.

```sh
NODE_OPTIONS=--no-experimental-webstorage ./node_modules/.bin/vitest run \
 ../packages/ui/src/components/Notes/__tests__/monaco-format-candidate.test.ts
NODE_OPTIONS=--no-experimental-webstorage ./node_modules/.bin/vitest run \
 --config vitest.extension.config.ts \
 ../packages/ui/src/components/Notes/__tests__/monaco-format-candidate.test.ts
```

Final results: **2 passed** in each web and extension resolver configuration,
733ms/444ms (`unit-final.log`, `unit-extension.log`). The admission fake asserts
no edit/undo stops on same-key source drift; it is not native-history evidence.

Actual extension cwd `apps/extension`:

```sh
NODE_OPTIONS=--max-old-space-size=12000 bun run build:chrome
bun run compile
```

WXT production Chrome MV3 build passed in 32.2s, 64.84 MB including local assets
(`extension-build-final.log`). Actual extension compiler passed after temporary
product wiring removal (`extension-compile-final.log`); its config checks extension
entry/config boundaries, not all UI. Whole W tsc and WXT bundle cover the shared UI.
From W, `node e2e/fixtures/check-monaco-extension.mjs` passed against the actual
built extension in a private Chromium profile (`extension-runtime.log`). The
options page loaded local distribution scripts and the nested installed
`editor.worker-Be8ye1pW.js`, preserved an exact CRLF model, and received real worker
readiness under the unmodified MV3 manifest/default self CSP. No CSP override was
added. The script reports Node's opaque URL origin as `null` and separately records
the real chrome-extension base. This is **asset/CSP smoke only**: extension Notes,
Firefox extension runtime, and Next standalone production packaging are unqualified.

After removing temporary product wiring and generated source public directories:

```sh
git apply -R apps/tldw-frontend/e2e/fixtures/notes-monaco-qualification.patch
node apps/tldw-frontend/e2e/fixtures/prepare-monaco-qualification.mjs --clean
# cwd W
NODE_OPTIONS=--max-old-space-size=12000 ./node_modules/.bin/tsc --noEmit --incremental false
```

Final whole W typecheck exited **0 with no output** (`tsc-final-unwired.log`).
An initial roughly 4GB/default heap attempt failed with an observed JavaScript OOM
(`tsc-wired.log`); the successful 12GB limit retry is resource-qualified evidence.
No claim attributes the OOM to inherited debt. Forward `git apply --check` passes;
`git diff --exit-code` on NotesEditorPane and NotesManagerPage confirms no retained
product wiring. The ignored WXT output is retained as local evidence.

Scoped ESLint: the three new U files (from `apps/packages`, using W's ESLint/config)
and new W scripts/spec plus playwright.config all report zero errors/warnings in
`eslint-u-final.json` / `eslint-w-final.json`. The U invocation prints the config's
pages-directory applicability warning. Six new files pass W Prettier `--check`
(`prettier-check.log`). The only modified existing code file, playwright.config,
was checked at immutable S base and current via ESLint stdin and Prettier stdin:
both ESLint exit0; existing formatting differences are identical, with no new
formatter delta (`static-comparison.json`). `git diff --check` passes. No Python
code changed, so Ruff/Bandit are inapplicable to this candidate scope. The scoped
security review found no new source disclosure endpoint, executable source or
credential logging; only synthetic note data and the disposable key appear in
fixtures/evidence. No blanket whole-repository lint/security claim is made.

Observed output is not pristine: browser Node module.register deprecation and
NO_COLOR/FORCE_COLOR warnings, WXT duplicate-autoimport warnings for
MediaNavigationFormat/estimateStorageCost and stale Browserslist data were present.
These were not compared to an immutable build baseline and are not labeled
inherited debt. Browser iterations exposed stale async readOnly capture, wrong
UA-native key selection, default EOL selection for newline-free input, the wrong
composition option name, a wrong Notes search locator, and disposed pending access;
each was corrected before the final run. Firefox's nested-worker request omission
was addressed with worker-global evidence, not a claimed loading failure. One
intermediate Chromium keyboard run failed; final two 36-test runs passed with the
simple direct focus implementation. No event-helper workaround is retained.

### Self-review and limits

Reviewed the bridge, mounted host, actual Notes patch/handler, source/network
fixture, cleanup and build paths. Removed an unused focus helper caught by ESLint;
kept direct native focus. Strengthened the viewport precondition so zero scroll
cannot count as horizontal preservation. Read-only refusal preserves exact source
without an editable fallback. Candidate checks have no production import after
patch removal, no dependency changes and no changes to Markdown or C code/tests.
Existing loader consumers, actual OS IME, extension Notes/history and same-note
external refresh integration remain explicit limits for I3/reviewer consideration.

Owned backend/frontend listeners were PID 99450/99492, in isolated sessions
41001/88503; both were terminated, and lsof confirmed no listeners on 18080/18081.
Private extension profile was closed/removed. Successful specs delete their
synthetic notes using current versions. Earlier failed-run leftovers, if any, are
confined to the disposable DB. Minimal-server configuration still starts some
services; DISABLE_BACKGROUND_TASKS is not asserted to suppress all background work.
No shared environment was rebound and no user notes/configuration were modified.

Final static command details (all paths relative to their stated cwd):

```sh
# cwd S/apps/packages
../tldw-frontend/node_modules/.bin/eslint \
 --config ../tldw-frontend/eslint.config.mjs \
 ui/src/components/Notes/__tests__/fixtures/monaco-format-candidate.ts \
 ui/src/components/Notes/__tests__/fixtures/MonacoNotesQualification.tsx \
 ui/src/components/Notes/__tests__/monaco-format-candidate.test.ts -f json
# cwd W
./node_modules/.bin/eslint e2e/notes-monaco-qualification.spec.ts \
 e2e/fixtures/prepare-monaco-qualification.mjs \
 e2e/fixtures/check-monaco-extension.mjs playwright.config.ts -f json
./node_modules/.bin/prettier --check \
 ../packages/ui/src/components/Notes/__tests__/fixtures/monaco-format-candidate.ts \
 ../packages/ui/src/components/Notes/__tests__/fixtures/MonacoNotesQualification.tsx \
 ../packages/ui/src/components/Notes/__tests__/monaco-format-candidate.test.ts \
 e2e/notes-monaco-qualification.spec.ts \
 e2e/fixtures/prepare-monaco-qualification.mjs e2e/fixtures/check-monaco-extension.mjs
```

The immutable-base comparison passes the exact `git show
abbd9c12b90020fb6230f8854a7b171017c2641e:apps/tldw-frontend/playwright.config.ts`
contents and current file contents separately to W ESLint
`--stdin --stdin-filename playwright.config.ts -f json` and Prettier
`--stdin-filepath playwright.config.ts`. Both ESLint JSON reports have no messages;
the identical pre-existing formatting deltas concern only the cookie URL line
break and webServer ternary indentation, outside this task's two testMatch changes.

Backlog metadata was updated using `/Users/macbook-dev/.local/bin/backlog-py task
edit 34570 --append-notes ...` in C and `task edit 13512 --append-notes ...` in S.
Successful logs are `backlog-c.log` and `backlog-s-authorized.log` in the evidence
directory. CLI warnings about other unreadable/duplicate task files are preserved;
no unrelated task repairs were attempted. The first sandboxed S CLI attempt failed
to create its application-support lock (`backlog-s.log`); the authorized retry
succeeded. The CLI's task-frontmatter/section-marker formatting changes are retained.
