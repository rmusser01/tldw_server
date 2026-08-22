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
