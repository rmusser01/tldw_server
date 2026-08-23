# Task 17 implementation report

Task BASE: `e0f2bbb3f25713fec3070409c42a1809aa32340d`

## Status

Implementation and verification are complete enough for the controller's
explicit staged audit. This is not a merge-ready claim: Firefox and WebKit are
unavailable in this checkout, the broad backend matrix retains two reproducible
baseline failures, and the human requester must still write the required
`Change summary` in their own words. Nothing has been pushed.

The plan's original `docs(slides): document standalone HTML rollout
(TASK-12115)` subject is superseded by `fix(slides): harden standalone HTML
rollout (TASK-12115)` because the genuine acceptance RED required substantial
controller-approved runtime lifecycle and security hardening in addition to
documentation.

## Preflight

IMPECCABLE_PREFLIGHT: context=pass product=pass command_reference=pass shape=pass image_gate=skipped:no imagery belongs in integration, security, or rollout documentation mutation=complete

Read the binding product/design/spec/plan, repository instructions, and the
execution-plan, TDD, debugging, verification, code-review, and Impeccable skill
instructions. Backlog work was deliberately omitted by controller direction.

### Analogues inspected

1. `test_standalone_html_generation_jobs.py`: real Slides database, real
   JobManager acquisition, standalone worker dispatch, receipts, retry, and CAS
   behavior.
2. `test_standalone_html_api.py`: authenticated FastAPI composition, owner
   boundaries, negotiation, versions, save, export, and bounded errors.
3. `standalone_html_sources.py` and its tests: the prompt, chat, media, notes,
   and RAG adapter boundaries.
4. `standalone_html_generation_jobs_worker.py`, reconciler, validator pool, and
   startup wiring: the supported in-process lifecycle and source-free Jobs
   envelope.
5. `playwright.config.ts` and existing WebUI workflow suites: project selection,
   local WebUI lifecycle, responsive assertions, and browser fixtures.
6. Extension Presentation Studio E2E and launch helpers: pre-bootstrap bounded
   IPC/storage tripwires, built-extension target discovery, and fixed WebUI
   handoff.
7. Presentation Studio route, capability, principal, recovery, editor, outline,
   and conflict tests: metadata-first dispatch and same-scope/mismatch lifecycle
   authority.
8. Slides API/design/PRD/MCP/core documentation: legacy structured weak ETags
   and synchronous generation, standalone strong/async semantics, source
   isolation, and rollout language.

## TDD and RED evidence

The backend integration and both browser specifications were authored before
Playwright configuration or rollout-document changes. Their initial RED covered
the real owner database/Jobs/worker lifecycle, source adapters, exact transport
headers, cross-engine project selection, workflow history, and security sinks.
RED then exposed these real product gaps in the previously landed Task15 paths:

- Next could pull data-router-only exports through the shared router module;
  the fixed leaf prompt module now imports only shim-supported hooks.
- Page/workspace pagehide, pageshow, identity, and metadata epochs could race
  source-bearing detail or restore work; the parent publishes a source-free
  epoch fence before child effects and the child checks the captured epoch
  before request and adoption.
- Workspace SPA unmount omitted the synchronous latest-candidate recovery flush;
  pagehide and unmount now share the same scoped authority flush without a
  trusted-scope fallback.
- The mobile Code/Outline tabs lacked ArrowLeft/ArrowRight roving-tab behavior.
- Generation scope transitions could expose form/submitted source before
  authority settled, lose same-tab source when storage failed, or unmount the
  only quarantined copy after capability denial. The hook now keeps a bounded
  same-scope in-memory authority snapshot and the parent uses an opaque
  per-mounted-child retention token; mismatch/logout scrub before release.
- The outline worker lacked a fixed name, preventing closed worker identity
  auditing. It now uses exactly `StandaloneHtmlOutlineWorker`.

Test-harness changes were also RED-driven. The Blob URL recorder first missed
attribute/property/composite assignments; a source-free minimal-document probe
failed until all bounded Blob URL assignments were retained. The worker helper
then failed a focused negative characterization because a broad module-path
substring admitted a near match. It now admits only the captured named classic
same-origin Blob or exact Next chunk. A full Chromium run before the final fixes
was 11/13: the stale worker matcher missed the exact chunk, and the
same-principal form helper incorrectly released every held auth request before
asking for another. Diagnostics proved the latter was harness ordering, not
source loss: first Back cycle, `authRequestCount=7`, `authHeldCount=3`,
`authPendingCount=0`, exact final source already restored after settlement, and
the pageshow callback itself source-free. The corrected test holds the first
request, proves the source-free shell, and releases only afterward.

Three-attempt stops were honored. The security init-script blank-root problem
was reassessed after three equivalent app-bootstrap attempts; a single-feature
classifier proved that proxying `window.eval` changed direct-eval semantics and
caused `exports is not defined`. The proxy was removed rather than replaced.
The Blob calibration moved to a pure minimal document after the app-startup
seam reached its stop. The Page history race moved to an epoch-ref callback
architecture only after diagnostic request/initiator traces classified a new
post-restore child dispatch. No root cause received a fourth equivalent attempt.

## Backend evidence

- Focused real integration:
  `tldw_Server_API/tests/Slides/test_standalone_html_integration.py` passed
  `1/1`. It uses the real per-owner Slides database, real JobManager, real
  service/validator/worker, and HTTP router while mocking only source/provider
  adapters. It covers all five sources, owner isolation, provider counts and
  replay, Jobs envelopes, poll/list/search/detail/version/save/restore/export,
  database reopen, default-off readability, and legacy structured weak-ETag
  synchronous compatibility. Default search excludes standalone IDs; explicit
  opt-in includes a source-free summary.
- Broad backend matrix: `1600` collected, `1573 passed`, `25 skipped`, `2
  failed`. Both failures are deterministic pre-existing SQLite CAS tests:
  - `tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py::test_lost_terminal_cas_reloads_completed_winner`
  - `tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py::test_lost_retry_reset_cas_reloads_completed_or_terminal_winner[completed]`
  Focused alone and as a pair with seed `582673251` reproduce the worker line
  923 to `_retry` line 377 `StandaloneHtmlGenerationRetry("generation_store_unavailable")`
  stack. `git diff e0f2bbb3... --` for that worker and test is empty.
- OpenAPI/client contract audit passed: `317` client paths plus `49` fallback
  fields, with the existing `10` documented exceptions.

## Frontend, browser, and extension evidence

- Directly impacted Vitest matrix: `25/25` files and `631/631` tests passed.
  After the final dependency-list cleanup, the Workspace suite passed `110/110`.
- WebUI production compile passed with
  `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`: compiled in 46 seconds, generated
  150 static pages, and token synchronization passed.
- Playwright collection lists exactly `43` tests: Chromium selects the four
  workflow tests plus thirteen security tests; Firefox and WebKit each select
  only the thirteen security tests. Both security projects use `retries: 0`.
- Chromium workflow passed `4/4` in 35.2 seconds.
- Fresh Chromium security passed `13/13` in 1.4 minutes. This includes real
  localhost-to-127.0.0.1 CORS/preflight and headers, direct source responses,
  CSP header/meta comparison, accepted and corrupt source inertness, real
  Monaco gestures, closed Safe Outline, exact Blob/download confinement,
  worker timeout/replacement/termination, recovery, multi-user and same-user
  Back/Forward/pagehide, creation/submitted/workspace lifecycle, and direct
  version/restore/export routes.
- Firefox could not launch any of its 13 tests because
  `/Users/macbook-dev/Library/Caches/ms-playwright/firefox-1509/firefox/Nightly.app/Contents/MacOS/firefox`
  is missing. WebKit could not launch any of its 13 tests because
  `/Users/macbook-dev/Library/Caches/ms-playwright/webkit-2248/pw_run.sh` is
  missing. These are 13 launch failures per engine, not passes or skips; no
  browser was installed.
- `apps/extension bun run compile` passed (`tsc --noEmit`). The plan-exact,
  non-CI Task16 extension spec against the built artifact passed its pure audit
  and skipped three built-extension cases because no extension targets were
  discoverable: `1 passed, 3 skipped` in 31.1 seconds. A separate CI/headless
  run is retained as an inherited harness limitation: `1 failed, 3 skipped`
  because headless Chromium had no `window.chrome` and the Task16 fixture does
  not assign its manufactured fallback object back to the global. No Task16
  path was changed.

## Static, lint, Bandit, and hygiene evidence

- Focused lint over the changed runtime Hook/New/Page/Form/Workspace/router
  paths completed with zero lint diagnostics (the Next plugin prints only its
  informational missing-pages-directory message). Focused lint over Playwright
  config and both E2E specs completed with no output. An ad hoc broader
  test-file lint is not a clean project gate: it reports existing test-mock
  `any`/hook-name rules and the pre-existing C0-control regex; the only new
  Workspace exhaustive-dependency warning found by that audit was fixed and
  its focused suite rerun.
- The one permitted broad 8 GiB UI type audit was run during Task15, not rerun.
  It originally named three Task15 discriminant-narrowing diagnostics in
  `standalone-html-source.ts`, `StandaloneHtmlSourceEditor.tsx`, and
  `StandaloneHtmlWorkspace.tsx`; those were fixed afterward. The no-rerun
  limitation is intentional and is not described as a current broad type pass.
- Bandit JSON: `/tmp/bandit_standalone_html_presentations.json`; exit 1 from 72
  LOW test-only findings (70 B101 assertions and 2 B105 sentinel strings) in two
  `app/core/MCP_unified/tests` files. It scanned 95,436 LOC with zero production,
  medium, high, or scanner-error findings.
- `git diff --check` passed after the final report update. Added-production
  source/sink scans found only the fixed application-owned outline worker
  constructor; no added execution/DOM-HTML/navigation/provider-registration
  sink appeared. Documentation scans found no executable HTML sample, em dash,
  visual-effect copy, obsolete worker path, or standalone safety/preview claim.
  Protected artifacts remain excluded from every intended stage.

## Product and accessibility review

The implementation retains one clear page heading, shared primitives/tokens,
44px actions, visible focus, responsive no-overflow checks, keyboard-only tab
selection, labelled Speaker notes disclosure, bounded loading/error/retry
states, and explicit conflict/recovery actions. It adds no gradient, glass,
card-grid, decorative motion, preview, or execution affordance. Standalone
source is React text/Monaco model input only; Safe Outline is application-owned
closed text. Documentation consistently calls downloaded source executable and
untrusted and never calls it sanitized or safe.

## Authorized scope deviations

Controller-authorized deviations from the original thirteen Task17 paths are:

- `apps/packages/ui/src/entries/shared/route-leave-prompt.tsx` and
  `router-utils.tsx` for the Next-compatible prompt leaf split.
- `PresentationStudioPage.tsx` and its routed test for synchronous
  pagehide/pageshow kind-authority fencing.
- `StandaloneHtmlWorkspace.tsx` and its test for the parent epoch callback,
  scoped unmount flush, and keyboard tab behavior.
- `StandaloneHtmlGenerationForm.tsx` and its test for the source-free
  scope-resolution shell and Retry.
- `useStandaloneHtmlGeneration.ts` and its existing test for same-scope
  in-memory quarantine, phase/action authority, and overlapping Retry fencing.
- `PresentationStudioNew.tsx` and its integration test for the opaque
  per-mounted-child retention token.
- `standalone-html-outline-client.ts` and its outline test for the fixed worker
  name.

No global client, proxy, cache, storage, backend production, renderer, or MCP
implementation was added. No Backlog call/edit, repository dependency install,
or push was made. The modified `apps/packages/ui/node_modules/antd` entry and the two
untracked Watchlist templates pre-existed and remain protected and unstaged.

## Residual limitations

- Firefox and WebKit security behavior remains unexecuted until the pinned
  browser binaries are provided.
- Direct `eval` calls are not proxy-observed because doing so changes language
  semantics. The non-suppressing execution-sentinel setter, Function/DOM/worker/
  network probes, accepted-source browser tests, and static sink scan remain.
- Native `useBlocker` can inherit React Router's documented roughness under
  rapid repeated POP navigation; the application fences/cancels its own pending
  confirmation task and has real Hash/Memory/Next regressions.
- Real bfcache availability is browser-controlled. Tests assert every
  `pageshow.persisted` probe when the engine supplies one and annotate the
  engine limitation otherwise; they do not synthesize bfcache.
- The two inherited backend CAS failures and the Task16 headless fixture gap
  remain explicitly non-green.
