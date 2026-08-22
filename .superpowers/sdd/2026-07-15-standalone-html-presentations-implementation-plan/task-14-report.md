# Task 14 implementation report

## Scope and base

- Base: `e1fb3e3c6943dd767335415cb067cd8de98a3328`
- Branch: `codex/standalone-html-presentations`
- Backlog was not accessed, as required by the task brief.
- No dependencies were installed and no browser automation was run.
- The authorized client addition is limited to a metadata-returning generation status method. The existing receipt-only `getPresentationGeneration()` contract is unchanged.

## Analogue inspection

Before editing, I inspected these existing patterns:

1. `PresentationStudioPage`, its routes, and the presentation store for structured create/detail contracts and route guards.
2. The Task 13 presentation client and standalone contract tests for discriminated records, source-free metadata, negotiation, and offset pagination.
3. Existing source/data-table loading, empty, error, retry, and approximately 44px action patterns.
4. `useServerCapabilities` and server-online hooks for mount fetches, fail-closed states, and explicit retry behavior.
5. Shared `StatePanel`, `LoadingState`, `Badge`, `Button`, and `ActionGroup` primitives for product-state vocabulary, focus, and target sizing.
6. Playground configuration-scoped recovery for origin/account boundary handling.
7. Existing ingest job polling for long-running job state and retry patterns.

The generic form-draft hook was not reused because it uses `localStorage`, a seven-day lifetime, logging, and an unscoped namespace, which conflict with this task's security contract.

## TDD evidence

All four required Task 14 suites were added before production changes. The exact focused command collected all four files and produced the canonical RED:

- Test files: 4 failed
- Tests: 31 failed, 31 collected
- Cause: missing Task 14 modules and behavior
- Harness-only failures: none in the canonical escalated run
- Note: an initial sandbox-only `EPERM` prevented Vitest cache creation and collected zero tests; the exact command was rerun with the required filesystem permission before any production edit.

The first complete GREEN passed 4 files and 31 tests. The final expanded matrix passes 4 files and 34 tests after adding explicit duplicate-submit, definitive rejection, unavailable-principal fail-closed behavior, pre-admission reload replay, and admitted-job reload coverage.

## Implementation summary

- Added a source-free, offset-paginated index with ID deduplication, kind-specific metadata, read-only unknown kinds, and loading/empty/error/offline states.
- Added authoritative Slides capability discovery with explicit Retry and fail-closed disabled, validator-unavailable, malformed, auth/error, and offline states.
- Added the direct pasted-material standalone form with closed choices, immutable submitted-request display, locked submission, configured target metadata, local scalar/NUL/effective-limit validation, secure browser-field attributes, and no provider/model picker.
- Added principal plus canonical-origin scoped, schema-validated, UTF-8 byte-capped, 24-hour `sessionStorage` draft and replay records.
- Added cryptographically generated URL-safe idempotency keys, exact request/key replay for ambiguous outcomes, and new keys for corrected, different, or terminal retry attempts.
- Added real-state polling with bounded exponential fallback and bounded Retry-After, bounded server progress text, local-only Stop/Forget behavior, and terminal/auth/404/throttle/outage recovery.
- Added pagehide flush and synchronous memory clear, plus guarded pageshow/focus/visibility/config/account revalidation.
- Added the new creation-mode route while preserving the existing structured creation form.
- Changed structured detail dispatch to check source-free metadata before fetching detail, so Task 14 never fetches standalone HTML source.
- Added a narrow client status result carrying a validated receipt plus nullable, capped Retry-After metadata without changing Task 13's receipt-only method.

## Verification

- Final focused Task 14 command: 4 files, 34 tests passed.
- Adjacent Presentation Studio and route command: 11 files, 50 tests passed.
- Standalone presentation client contract: 1 file, 77 tests passed.
- Direct config, API-send, background-proxy, and online-state regressions: 4 files, 52 tests passed.
- OpenAPI guard: 317 client paths and 49 fallback fields verified. The guard reported its 10 pre-reviewed OSS exception paths and passed.
- One package typecheck with `NODE_OPTIONS=--max-old-space-size=8192`: completed with no diagnostics. Task 14 diagnostics: none. Inherited diagnostics: none.
- `git diff --check`: passed.
- Bandit: not applicable because the touched implementation contains no Python files.
- Prettier: the frontend Prettier configuration reports the package UI baseline as unformatted; an untouched package file (`Common/Button.tsx`) fails the same check. No bulk formatter rewrite was applied because it would expand the Task 14 diff.
- ESLint: the frontend configuration ignored all package UI paths as outside its configured base path; it reported no errors and 17 ignore warnings.

## Static security review

- No `dangerouslySetInnerHTML`, `DOMParser`, `srcdoc`, `innerHTML`, `insertAdjacentHTML`, Blob URL, iframe, popup, worker, or module execution path exists in the Task 14 implementation.
- The only `window.location` search hit reads `window.location.origin` to establish the canonical storage scope. It never carries form content or a replay key.
- Source and audience flow only through component/hook memory, the bounded scoped `sessionStorage` draft, the immutable request body, and ordinary React text nodes.
- Replay keys exist only in the scoped resume record and API request options. They are not rendered, logged, placed in URLs, or sent to analytics/global stores.
- Index and structured-detail dispatch remain source-free. Standalone detail is refused after metadata inspection and before a full-detail request.

## Visual and accessibility self-review

- Reviewed loading, empty, error, offline, disabled, submitting, queued/running, ambiguous, stopped, auth-lost, missing, throttled, outage, failed, cancelled, and missing-binding states against `PRODUCT.md` and `DESIGN.md`.
- Reused semantic surface/text/border/focus/state tokens and shared buttons, badges, loading, and state panels.
- No gradients, glass, decorative motion, card grid, side stripe, or bespoke visual system was added.
- Controls have visible labels and focus treatment; primary actions use the shared 44px large button size; status/error copy uses live/status or alert semantics.
- Responsive layouts collapse to one column and retain reachable actions without horizontal dependency.
- Source-bearing text fields disable spelling, autocorrection, autocapitalization, autocomplete, and supported password-manager capture, and omit `name` attributes.
- No custom imagery was required. Browser automation is intentionally deferred to Task 17.

## Protected artifacts

The following unrelated artifacts remain unstaged and were not modified by this task:

- `apps/packages/ui/node_modules/antd`
- `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`
- `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`

## Independent review fix round

This section supersedes the earlier verification counts where the independent review expanded the required matrix.

### Review verification and canonical RED

Each Critical, Important, Minor, and controller finding was checked against the implementation before changes. The coherent fix matrix was written before production edits and run with:

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/hooks/__tests__/useSlidesCapabilities.test.tsx \
  ../packages/ui/src/hooks/__tests__/useStandaloneHtmlGeneration.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioIndex.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGenerationForm.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGeneration.integration.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioNew.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioPage.test.tsx \
  ../packages/ui/src/services/__tests__/tldw-api-client.presentations-standalone.test.ts \
  ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx \
  __tests__/auth.logout.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

The canonical RED collected all 10 files and all 179 tests: 57 failed and 122 passed. Nine files failed for intended Task 14 behavior and the unchanged route-guard suite passed. There were no harness-only or unrelated baseline failures. The failures covered every required group:

- scope epoch fencing, synchronous auth/config/pagehide boundaries, stale POST/poll rejection, abort, and unmount scrubbing;
- capability request fencing, auth/offline/scope invalidation, no-store response enforcement, option gating, retry states, and revision display;
- real DOM snapshot commitment before POST;
- real client-envelope status/details/Retry-After handling and completed-without-binding normalization;
- quota-safe in-memory recovery, split draft/resume deletion, reload and Stop-to-New recovery, disabled-capability recovery, and transient scope outage retention;
- Unicode scalar, canonical UTF-8 request byte, effective slide, and terminal retry validation;
- source-free HTML/unknown detail dispatch, new-server metadata failure, and legacy structured-only fallback;
- invalid offset pagination and duplicate-only advancing pages;
- 409 configuration revision reconfirmation, bounded receipt text, logout boundary signaling, heading semantics, and downloaded-file wording.

### Fix implementation

- Added a captured canonical origin/principal scope epoch to every submit and poll result. Config, auth, pagehide, and unmount boundaries synchronously invalidate the epoch, detach scope, abort supported work, scrub source/key refs, and ignore late results before storage, UI, or handoff.
- Fenced authoritative capability requests with request generations and abort signals, subscribed to trusted config/auth and page lifecycle events, distinguished 401 from 403, kept the HTML option disabled until confirmation or trusted recovery, and enforced the exact `private, no-store` response policy at the response seam.
- Deferred POST to a committed-attempt effect so React renders the immutable snapshot and disabled controls before the client invocation.
- Preserved status, bounded details, and bounded Retry-After metadata for non-2xx client envelopes without changing Task 13's receipt-only method. The client accepts completed receipts without a binding so the hook can show the required safe state.
- Kept resume metadata in memory even when session storage fails, split draft and resume deletion, preserved drafts on pre-admission rejection and terminal edits, and added a source-free scoped recovery probe. Existing receipts remain recoverable when current generation is disabled or temporarily unavailable.
- Counted Unicode scalar values, enforced canonical request UTF-8 bytes, clamped slides to the content capability and fixed limit of 30, and routed terminal retries through current validation.
- Added source-free, kind-aware detail states. Full detail is requested only for metadata-proven structured records or when both metadata and exact Slides capabilities return 404, proving a legacy structured-only server.
- Rejected null, repeated, backward, noninteger, and nonfinite pagination offsets while allowing duplicate-only advancing pages.
- Added a synchronous frontend logout principal-boundary event, safe 409 revision-reconfirmation flow, bounded progress/error rendering, one top-level heading in embedded structured setup, and accurate downloaded-file execution wording.

### Fix-round verification

- First root-cause GREEN: detail dispatch and pagination passed 2 files and 23 tests.
- Scope/recovery/form/client-boundary GREEN: 3 files and 49 tests passed.
- Canonical amended focused matrix: 10 files and 179 tests passed.
- Adjacent Presentation Studio and route regressions: 13 files and 70 tests passed.
- Direct auth, config, API-send, request-core, background-proxy, connection-sync, and online-state regressions: 11 files and 72 tests passed.
- OpenAPI guard: 317 client paths and 49 fallback fields verified; the 10 pre-reviewed OSS exception paths were reported and allowed.
- The required package typecheck used `NODE_OPTIONS=--max-old-space-size=8192`. After corrections, it reports zero diagnostics in Task 14 production or test paths. It still exits nonzero on 47 inherited diagnostics grouped under Notes, Audio Studio, Research Workspace, Scheduled Tasks, Setup, Skills, Dexie, background entry code, MCP Hub, and voice cloning.
- `git diff --check` passed.
- The static no-execution scan found no `dangerouslySetInnerHTML`, `DOMParser`, `srcdoc`, HTML insertion, Blob URL, iframe, popup, worker, dynamic import, eval, or Function constructor in Task 14 production files.
- The source/key sink scan found replay keys only in the local hook record and client options. Source remains in component/hook memory, bounded scoped `sessionStorage`, the request body, and ordinary React text nodes. It is absent from logs, analytics, URL state, and global stores.
- Targeted ESLint reported zero errors; six package UI paths are ignored by the frontend base-path configuration. Targeted Prettier check reports the same package UI baseline formatting failure reproduced by untouched `Common/Button.tsx`, so no bulk formatting rewrite was made.
- Bandit remains not applicable because the fix round touches no Python.

### Fix-round visual and accessibility self-review

The review covered loading, offline, unavailable, disabled, validator-blocked, source-free recovery, idle, submitting, polling, ambiguous, stopped, auth-lost, missing, throttled, outage, rejected, configuration-changed, failed, cancelled, missing-binding, completed handoff, pagination error, standalone detail, unknown detail, and legacy structured states against `PRODUCT.md` and `DESIGN.md`.

The implementation keeps one top-level heading in the New route, visible labels and focus styles, shared 44px buttons and form targets, semantic `fieldset`/`legend`/`dl`/status/alert structure, reduced-motion-safe transitions, and single-column narrow-screen fallbacks. It reuses the existing restrained surface, border, text, focus, badge, loading, and state-panel system. No gradients, glass, card grid, decorative motion, em-dash copy, provider picker, or bespoke visual system was added. Browser automation remains intentionally deferred to Task 17.

## Authority-gap fix round 2

### Review verification and RED

The round-2 findings were verified against the committed round-1 implementation before production changes. The coherent six-file matrix was extended first and run with:

```bash
cd apps/tldw-frontend
bun run test:run -- \
  __tests__/auth.logout.test.ts \
  ../packages/ui/src/hooks/__tests__/useSlidesCapabilities.test.tsx \
  ../packages/ui/src/hooks/__tests__/useStandaloneHtmlGeneration.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGenerationForm.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGeneration.integration.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioNew.integration.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

The canonical RED collected all six files and all 72 tests: 14 intended tests failed and 58 passed. There were no unrelated baseline or harness failures. The failures covered logout ordering and no-refetch fencing, null and mismatched post-response scope confirmation, source-free draft presence and disabled-capability recovery, last-trusted namespace cleanup after verification outage, retained quota-failed form state during 409 authority refresh, and stale progress clearing.

After the first focused GREEN, one missing explicit recovery-controller boundary case was added. It produced a genuine one-test RED in a 40-test hook suite: logout did not yet delete the recovery probe's last trusted namespace while an older scope check was deferred. The minimal source-free last-trusted-scope fix made it GREEN. An intermediate assertion in the source-free probe test was also corrected after direct evidence showed it contradicted its own setup: the test removed the resume record to create draft-only state, then asserted that the removed resume record remained. The corrected assertion checks the draft record that the outage must preserve.

### Fix implementation

- Frontend logout now clears the local token and user before emitting the narrow `tldw:auth-principal-changed` logout event.
- Capability and recovery listeners treat logout as an invalidation-only boundary. They abort or fence older work, publish no old authority or recovery, and wait for a later trusted login, configuration, or lifecycle boundary before resolving again.
- Missing or mismatched post-response scope confirmation now becomes an explicit retryable error instead of remaining indefinitely in loading state.
- The recovery presence probe distinguishes resume and draft-only records. It enumerates scoped storage key names and reads only capped resume metadata; it never reads or parses source-bearing draft values before the trusted generation hook mounts.
- Both the recovery controller and generation hook retain a non-source-bearing last trusted scope across transient verification outages. A later confirmed switch or definitive logout removes the old scoped draft and resume keys synchronously.
- Draft-only hydration is exposed separately from receipt recovery, so preserved direct material and Forget remain available when current generation capability is disabled or unavailable.
- A 409 configuration revision refresh retains the mounted form and its in-memory source even when storage quota writes fail. Prior provider, model, and revision details are reference-only, all submission authority is disabled during refresh, and only a freshly confirmed revision enables a deliberate new submission.
- Receipt progress is replaced on every receipt, so omitted or empty progress cannot survive into a later or terminal state.

No module or global source cache, history state, URL state, token estimator, request `Cache-Control` header, provider picker, execution sink, new dependency, or cross-route quota-failure recovery was added.

### Round-2 verification

- First logout/capability GREEN: 2 files and 17 tests passed.
- First complete six-file GREEN: 6 files and 72 tests passed.
- Explicit deferred recovery-controller boundary: one intended RED in a 40-test suite, followed by 40 tests passed.
- Final six-file focused matrix after that boundary case: 6 files and 73 tests passed.
- Final amended canonical matrix, including the real New-to-form integration: 11 files and 189 tests passed.
- Adjacent Presentation Studio and route regressions: 14 files and 73 tests passed.
- Direct auth, configuration, connectivity, request, background proxy, connection sync, and standalone client regressions: 10 files and 165 tests passed.
- OpenAPI guard: 317 client paths and 49 fallback fields verified; the same 10 reviewed OSS exception paths were reported and allowed.
- Required package typecheck used `NODE_OPTIONS=--max-old-space-size=8192`. It reports zero diagnostics in Task 14 production or test paths. It exits nonzero on the same 47 inherited diagnostics documented in round 1, grouped under Notes, Audio Studio, Research Workspace, Scheduled Tasks, Setup, Skills, Dexie/background entry code, MCP Hub, and voice cloning.
- Targeted ESLint reports zero errors; the four package UI files are ignored by the frontend base-path configuration. Targeted Prettier reports the existing package/frontend formatting baseline, so no bulk formatter rewrite expanded the review scope.
- `git diff --check` passed before the final report update and is rerun after staging.
- Static execution-sink scan found no HTML parsing or insertion, `srcdoc`, Blob URL, iframe, popup, worker, dynamic import, eval, or Function constructor in round-2 production paths.
- Static source/key sink review found source and replay keys only in component/hook memory, bounded principal-and-origin-scoped `sessionStorage`, request bodies, ordinary React text nodes, and client request options. They remain absent from logs, analytics, URLs, history state, and global stores.

One exploratory broad-test command included `__tests__/hooks/useConfig.fetch-mode.test.ts` from the frontend working directory. That test constructs a repository-root-relative path and failed only because the selected working directory duplicated `apps/tldw-frontend` in the path; the other 10 suites and 165 tests passed. The corrected directly impacted command omitted that unrelated static-path harness and passed 10 files and 165 tests.

### Round-2 visual and accessibility self-review

The retained-refresh and preserved-draft states reuse the existing surface, border, text, degraded-state, button, and state-panel system. Stale target data is explicitly identified as non-authoritative, submission stays disabled until fresh confirmation, and Retry remains a shared large button action. The preserved-draft state uses a labelled section and an explicit Forget action without reading draft source in the controller.

The form keeps visible labels, semantic `form`, `fieldset`, `legend`, `dl`, status, and alert structure; shared large actions retain approximately 44px targets and visible focus. Existing responsive grids collapse to one column without hiding recovery actions. No gradients, glass, card grid, decorative motion, em-dash copy, or bespoke visual system was introduced. The flow remains non-executing and browser automation remains deferred to Task 17.
