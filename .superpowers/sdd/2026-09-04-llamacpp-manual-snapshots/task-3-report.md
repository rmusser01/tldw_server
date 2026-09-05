# Task 3 implementation report

Status: safe implementation ready for independent review. TASK-13163 remains In Progress. Live acceptance is blocked on operator-supplied llama-server/model assets and real Admin/Chatbook verification; no production build hash was added.

## Implementation

- Added six typed client methods matching slots, paginated catalog, save, restore, delete and receipt routes. Request bodies contain slot, signed request token, expected launch generation and explicit replace confirmation only. New manual saves/restores fetch fresh tokens; the controlled panel has no network imports.
- Added the approved Slot snapshots entry to each managed runtime. The Admin container owns reads/mutations. It aborts reads and fences late responses on profile/lifecycle changes, and brackets multi-request reads with generation/latest-receipt observations from GET slots (the runtime-list response may omit generation). Every mutation revalidates generation using a fresh slots response. No request is automatically resubmitted.
- Added enablement without implicit restart, visible restart/unsupported/busy/stopped guidance, idle-slot saves, sorted and paginated catalog, compatibility reason codes, copyable IDs, timestamps with local timezone, retention controls, inline restore/delete/Stop confirmations, keyboard Escape/cancel focus return, polite receipt announcements and responsive fields using shared themes/primitives/i18n.
- Active receipt polling runs only while the selected mounted Admin surface's document is visible. Reopening/reloading recovers latest receipt via GET slots. Transport ambiguity blocks mutation and offers explicit Stop recovery. A historical unknown receipt after observed stopped state does not disable saved-copy deletion.
- Added the operator guide with privacy/backups, quiescence, manual semantics, retention, recovery, routing limitations and an explicit statement that no production build is currently verified.
- Added an opt-in live harness. It only launches newly created disposable in-memory profiles and temporary private storage, requires explicit consent and operator-supplied non-symlink executable/model paths, never downloads assets or opens a production profile store, and uses the real runner/store/coordinator. Its candidate hash override exists only in the test service. It requires measured reused/processed token counters after save/stop/start/restore and an identical cold-process control; outputs contain hashes/options/synthetic-prompt token metrics, not private prompts or answers.

ADR required: yes. Existing ADR path: Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md. Reason: direct implementation of the approved supervisor-owned manual workflow, no new architectural boundary.

## RED/GREEN evidence

1. Initial panel command failed because the component did not exist. After implementation and initializing real test i18n, 13 tests passed. The first implementation run found missing interpolation in the test harness (2 failures); real i18next/production ICU initialization fixed the harness without weakening assertions.
2. Container tests initially produced 5 failures for the missing exported Admin container. Implemented receipt recovery, fresh-token mutation, generation rejection, aborted/late reads and uncertain-outcome recovery; the first combined run passed 18 tests.
3. Added visibility/unmount, late mutation, settings-no-lifecycle and final generation/stopped-unknown regressions. The final fence regression run failed 3 tests before correction, including publication across generations and deletion wrongly blocked after confirmed Stop. Final combined UI result: **50 passed** across the two new modules and existing AdminPage/RuntimePanel modules, 13.71 seconds.
4. Live metric validator tests initially failed 5 cases with the missing helper. Final targeted Python result: **5 passed, 1 skipped**, 7 existing bootstrap/dependency warnings, 1.87 seconds. The skipped test is the actual live-runtime test, not a substitute pass.

Commands (from apps/tldw-frontend unless noted):

```sh
npm run test:run -- ../packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsPanel.test.tsx
npm run test:run -- ../packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsAdmin.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsPanel.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
# Repository root, after activating /Users/macbook-dev/Documents/GitHub/tldw_server/.venv:
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshots_live.py -q
```

Logs: /private/tmp/snapshot-admin-red.log, /private/tmp/snapshot-final-fence-red.log, /private/tmp/snapshot-ui-final.log, /private/tmp/snapshot-live-red.log and /private/tmp/snapshot-live-final.log. No full test sweep was run. Node 26 emits the existing experimental localStorage warning. The RuntimePanel test now initializes i18n because that component acquired translated snapshot labels.

## Browser and visual evidence

Ran the **actual Next.js Admin page**, not a substitute static harness, on loopback port 18383 with Playwright and mocked API responses. No real model or server profile was mutated. The two 390×844 light/dark browser flows passed in 41.4 seconds. They cover explicit replace/delete target confirmations, destination keyboard focus, Escape return, exact mutation bodies, no mutation before confirmation, no replay after page reload, successful receipt recovery and absence of panel horizontal overflow.

```sh
TLDW_WEB_URL=http://localhost:18383 \
TLDW_WEB_CMD='npm run dev:webpack -- -p 18383 --hostname 127.0.0.1' \
node node_modules/@playwright/test/cli.js test \
  e2e/workflows/llamacpp-runtime-admin.spec.ts --project=chromium \
  --grep 'manual snapshot' --workers=1 --reporter=line
```

Initial sandbox server bind failed with EPERM; the approved loopback test invocation then succeeded. Visual inspection found the initial media-emulation fixtures both used the persisted dark application theme. Tests now seed the actual theme preference and assert the html theme class. Visual inspection also corrected the native destination select's surface token and arrow spacing. Final screenshots are real app renders with mocked data:

- Light: /private/tmp/tldw-server-llamacpp-manual-snapshots/apps/tldw-frontend/test-results/workflows-llamacpp-runtime-8cb83-d-API-light-narrow-viewport-chromium/snapshots-light-390.png
- Dark: /private/tmp/snapshot-stage3-dark/workflows-llamacpp-runtime-d3156-ed-API-dark-narrow-viewport-chromium/snapshots-dark-390.png

The light-only screenshot refresh passed in 27.0 seconds and the final dark-only run passed in 29.9 seconds. Both final screenshots were opened and visually inspected after the select spacing correction. The dark artifact is retained in a separate output directory so subsequent Playwright cleanup does not remove the light artifact. Browser logs: /private/tmp/snapshot-playwright.log, /private/tmp/snapshot-playwright-final-light.log, /private/tmp/snapshot-playwright-final-dark.log. Dev-server output also includes existing Browserslist/Node deprecation and color-environment warnings.

## Static and security checks

- Targeted TypeScript passed, exit 0, using /private/tmp/snapshot-targeted-tsconfig.json (extends the real frontend config, includes touched components/tests, their production dependency graph and actual Vitest setup). Command: `NODE_OPTIONS=--max-old-space-size=8192 node node_modules/typescript/bin/tsc --noEmit --pretty false -p /private/tmp/snapshot-targeted-tsconfig.json`. Log: /private/tmp/snapshot-targeted-tsc.log.
- A full frontend TypeScript diagnostic pass reported 80 errors in unrelated PresentationStudio/security/certification code. No changed-file diagnostic remained; this is not a claim that the whole frontend typecheck passes. Log: /private/tmp/snapshot-tsc.log. Did not modify those files.
- ESLint on changed UI/types/browser files with the repository frontend config and `--max-warnings=0` passed. Root invocation is needed because invoking ESLint from apps/tldw-frontend ignores sibling package files. Log: /private/tmp/snapshot-changed-eslint.log.
- Full touched client lint: 0 errors/532 warnings. Independently linted `git show HEAD:apps/packages/ui/src/services/tldw/TldwApiClient.ts` via stdin: exactly the same 0 errors/532 warnings. JSON: /private/tmp/snapshot-client-eslint-{baseline,current}.json. The root invocation emits the existing missing-pages configuration notice. Added no warning suppressions for these inherited client issues.
- Prettier checks passed on new/shared UI, touched tests/types, browser workflow and guide using existing shared semicolonless/double-quote conventions and frontend conventions for its workflow. The new client method region was formatted while preserving unrelated existing client formatting; no whole-client format claim is made.
- Ruff check and format check passed on the live harness; compileall passed.
- Bandit 1.9.4 was run from the project venv using the provided cached PYTHONPATH. Initial findings were the harness's 16 pytest `assert` uses (B101), not production boundaries. Reran this test-only file with `-s B101`: zero findings. JSON: /private/tmp/snapshot-stage3-bandit.json. No other rule was excluded.
- `git diff --check` passed. Preserved controller task edits and the untracked apps/packages/ui/node_modules symlink; none is included in the scoped code commit.

## Live evidence still required

No llama-server or GGUF paths were supplied; live inference was not run. No executable/model hashes or cache-reuse measurements are invented. No production allowlist entry was added. Real browser-to-managed-runtime flow, Chatbook original-message/tool/approval invariance, and Pause/Resume behavior against that runtime remain unverified acceptance work. The committed browser flow is explicitly mocked and cannot satisfy those requirements.

The metric fields were verified from public pinned source using approved curl: tools/server/server-common.cpp:67–71 maps `timings.cache_n` to `n_prompt_cached` and `timings.prompt_n` to `n_prompt_processed`. tools/server/server-context.cpp:2118 assigns top-level `tokens_cached` from final slot prompt size. The harness requires direct timings counters and fails closed if absent. Source-derived fixtures are not live build support.

Review attention: the live harness is unexercised with real assets and deliberately characterizes a candidate only; initial production capability stays unsupported. Native `/completion` explicitly routes to slot 0, while normal Chatbook requests may route elsewhere. Keep TASK-13163 open until the measured runtime proof and real client semantics checks are recorded.

## Round-one independent-review fixes

Reviewed against base `3d751231c3`. Corrected all three reported issues without changing the approved design or live-support boundary:

- Save/restore failures distinguish exact documented pre-admission status/detail pairs from transport, malformed-success, storage and unrecognized HTTP failures. The classifier uses the actual client's `error.status` and `error.details.detail` shape, not message parsing or a blanket HTTP classification. A definitive rejection with no new receipt permits Refresh and a new explicit action; ambiguous outcomes still require recovery and never resubmit automatically.
- Latest receipts remain profile-scoped historical display across launch changes, including when no runner generation exists. Only matching-current-generation active/unknown receipts control current mutations and polling. Historical receipts explicitly disclaim current slot state.
- Delete and Stop confirmations focus Cancel on opening; Restore still focuses its destination. Immediate Escape closes each confirmation, restores initiating-control focus and causes no mutation.

RED evidence: the initial regressions failed **8 tests**, with 23 passing (`/private/tmp/snapshot-round1-red.log`). An additional focused `-t unrecognized-rejection` regression failed **1 test** against an overly broad rejection classifier before tightening the exact status/detail allowlist (`/private/tmp/snapshot-round1-rejection-red.log`). Final GREEN: **36 passed**, 2 files, 5.99 seconds; targeted TypeScript exits 0 with no diagnostics.

Exact final commands from `apps/tldw-frontend`:

```sh
npm run test:run -- ../packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsAdmin.test.tsx ../packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsPanel.test.tsx > /private/tmp/snapshot-round1-green.log 2>&1
NODE_OPTIONS=--max-old-space-size=8192 node node_modules/typescript/bin/tsc --noEmit --pretty false -p /private/tmp/snapshot-targeted-tsconfig.json > /private/tmp/snapshot-round1-tsc.log 2>&1
node node_modules/prettier/bin/prettier.cjs --check e2e/workflows/llamacpp-runtime-admin.spec.ts
TLDW_WEB_URL=http://localhost:18383 TLDW_WEB_CMD='npm run dev:webpack -- -p 18383 --hostname 127.0.0.1' node node_modules/@playwright/test/cli.js test e2e/workflows/llamacpp-runtime-admin.spec.ts --project=chromium --grep 'manual snapshot' --workers=1 --reporter=line --output=/private/tmp/snapshot-stage3-round1-browser > /private/tmp/snapshot-round1-playwright.log 2>&1
```

The browser run passed **2 tests in 34.1 seconds**, exercising the actual Next.js Admin page with mocked APIs in light/dark 390px viewports, now including prior-launch receipt recovery and immediate Delete Escape/focus return. No real runtime/profile was mutated. Both screenshots were opened and inspected:

- /private/tmp/snapshot-stage3-round1-browser/workflows-llamacpp-runtime-8cb83-d-API-light-narrow-viewport-chromium/snapshots-light-390.png
- /private/tmp/snapshot-stage3-round1-browser/workflows-llamacpp-runtime-d3156-ed-API-dark-narrow-viewport-chromium/snapshots-dark-390.png

Exact final root static commands:

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx apps/packages/ui/src/components/Option/Admin/LlamacppSnapshotsPanel.tsx apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsAdmin.test.tsx apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsPanel.test.tsx apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts --max-warnings=0 > /private/tmp/snapshot-round1-eslint.log 2>&1
node apps/tldw-frontend/node_modules/prettier/bin/prettier.cjs --check --no-semi --trailing-comma none --single-quote false apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx apps/packages/ui/src/components/Option/Admin/LlamacppSnapshotsPanel.tsx apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsAdmin.test.tsx apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsPanel.test.tsx
git diff --check
```

All final static commands exited 0; Prettier reported all matched files use its code style. An initial combined format check incorrectly applied shared-package conventions to the frontend workflow and warned on that file; separate checks using each package's existing conventions pass without changing its style. ESLint retains only the existing root missing-pages configuration notice, not changed-code diagnostics. A first targeted TypeScript check caught an invalid test-only RTL `exact` option; removed it and reran successfully. No Python/security-boundary files changed in this round, so the recorded Ruff/Bandit harness checks were not rerun. No full sweep was run. Live blockers, empty production allowlist and TASK-13163 In Progress status are unchanged.
