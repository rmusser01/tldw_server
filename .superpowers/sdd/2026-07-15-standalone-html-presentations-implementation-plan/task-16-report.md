# Task 16 implementation report

## Status

Implemented the extension-safe Presentation Studio resolver and WebUI handoff from base `b8197101345fe3d2762e8a50ed2c322d27bc8ca3`. The extension remains source-free: standalone HTML and unknown kinds stop at a validated metadata-only surface, while structured projects retain the existing editor only after an exact metadata kind decision.

No push was performed. Backlog was not called or edited. Dependencies were not installed. The protected `antd` artifact and both Watchlist templates were left untouched and unstaged.

## Preflight and analogues

Binding preflight:

`IMPECCABLE_PREFLIGHT: context=pass product=pass command_reference=pass shape=pass image_gate=skipped:no imagery belongs in a metadata-only handoff mutation=open`

The read-only preflight inspected these concrete repository patterns before tests or production edits:

1. The shared deferred extension route registry and extension-target filtering.
2. Exact route metadata lookup and per-target availability declarations.
3. Task 15's metadata-first `PresentationStudioPage` kind dispatch.
4. Task 13's source-free `getPresentationMetadata` normalization boundary.
5. The sidepanel's canonical `resolveSidepanelChatWebUiBaseUrl` helper.
6. Existing built-extension Playwright launch, route, request-interception, storage, and runtime-message patterns.

These analogues led to a two-stage direct route: validate exact source-free metadata first, then mount the existing structured wrapper only for `structured_slides`. The literal `new` is reserved in extension routing and redirects to `/start`, so the parameter route cannot capture it.

## Approved scope

Production:

- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/route-metadata.ts`
- `apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioIndex.tsx` (narrow RED-proven authority-fence addition)

Tests:

- `apps/packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx`
- `apps/packages/ui/src/routes/__tests__/option-presentation-studio-start.test.tsx` (approved compatibility adjustment)
- `apps/extension/tests/e2e/presentation-studio-start.spec.ts`

Task records:

- `.superpowers/sdd/2026-07-15-standalone-html-presentations-implementation-plan/task-16-brief.md`
- `.superpowers/sdd/2026-07-15-standalone-html-presentations-implementation-plan/task-16-global-constraints.md`
- `.superpowers/sdd/2026-07-15-standalone-html-presentations-implementation-plan/task-16-report.md`

No request-core, background proxy, presentation client, backend, OpenAPI, global cache, storage, or generic workspace file changed.

## TDD evidence

Canonical command, run from `apps/tldw-frontend`:

```text
bun run test:run -- ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx --maxWorkers=1 --no-file-parallelism
```

Complete test-only RED:

- 65 tests collected.
- 62 failed and 3 passed before any Task 16 production edit.
- Failures covered real route inventory/metadata, reserved `/new`, metadata-first direct dispatch, bounded metadata projection, source-bearing client tripwires, structured compatibility, WebUI target construction, stale/unmounted work, index retirement, and extension storage/runtime-message absence.
- Independent RED-phase extension compile completed successfully; the missing behavior was runtime policy, not a missing type export.

Focused correction rounds remained RED first:

- First GREEN attempt exposed three genuine failures: duplicate kind presentation and a terminal lone-surrogate/invalid-number validation edge. After the minimal fix, all 65 passed.
- Dot-segment regression: 65 passed and 2 failed because `.` and `..` reached metadata resolution. Rejecting dot-only route IDs produced 67/67.
- Callback-time authority regression: 67 passed and 1 failed because an authority event followed by an old ready-button click could capture the new epoch and open stale metadata. Synchronously retiring the trusted-ready record and checking route/metadata identity at handler entry and immediately before `window.open` produced 68/68.
- Extension-only smoke-policy regression: 67 passed and 1 failed because `/presentation-studio/start` inherited `smoke: "include"`; the test expected `manual`. Adding the explicit route policy restored 68/68 and keeps the WebUI Stage-2 inventory contract from claiming the extension-only route.
- Storage-audit regression: a real browser test expected two distinct runtime surfaces plus six Chrome/browser local, sync, and session write surfaces, but the original tripwire exposed only `{ runtimeWrapped: true, storageWrapped: true }`. After the test-only hardening, the focused browser test passed and canonical unit coverage proved all distinct runtime/storage mocks remain unused by the metadata-only product flow.
- Fragment-detector regression: after correcting the test reset to clear the detector's closed-over audit array in place, the committed detector returned no findings for storage `{ source: "<div/>" }` or runtime `{ source: "<!-- lead --><section>..." }`. The focused browser test failed with expected storage/runtime kinds versus `[]`; the bounded tag/comment scanner then made it pass while retaining benign `source: "prompt"` and international-prose controls.

The same-root-cause three-attempt limit was not reached.

## Implementation decisions

- `/presentation-studio` remains the source-free metadata index.
- `/presentation-studio/start` remains the existing structured quick-start and is explicitly `smoke: "manual"` because it is extension-only.
- WebUI alone registers `/presentation-studio/new`; extension runtime registers a source-free redirect to `/start` ahead of `/:projectId`.
- Extension direct links call exact-ID `getPresentationMetadata` first. Only `structured_slides` mounts the existing structured detail wrapper. `standalone_html` and unknown kinds render the same bounded metadata handoff.
- Raw metadata is immediately projected into a closed object. Strings reject blank required values, C0/C1 and bidi controls, lone surrogates, and scalar overflow. IDs/kinds/provenance cap at 256 scalars, title at 512, description at 2,048; counts must be finite nonnegative integers. Dot-only IDs are rejected.
- Config, auth-principal, Slides-scope, route, and unmount boundaries retire requests/results. The index applies the same source-free request-identity fence. Ready click authority is synchronously cleared on boundary events.
- Handoff URLs are built only on activation from the current `serverUrl`, `webUiUrl`, and `webuiUrl`, after at least one candidate independently parses as HTTP(S). The canonical helper preserves explicit subpaths and API `:8000` to WebUI `:8080` inference while removing credentials, query, and fragment. Only the validated route ID contributes to `presentation-studio/${encodeURIComponent(id)}`.
- Handoffs use `_blank` with `noopener,noreferrer`. Metadata/title/provenance never influences the destination.
- The E2E tripwires reject named detail/version/save/download/export/render requests and source-bearing storage/runtime-message payloads while allowing unrelated extension infrastructure. The storage/runtime audit discovers every available distinct Chrome/browser local, sync, and session `set` surface, verifies wrapper assignment, performs a benign calibrated write with cleanup, and applies a bounded iterative key-aware detector for snake/camel HTML fields, draft/version fields, and standalone markup. Its contextual markup predicate examines at most an 8 KiB prefix and scans at most 2 KiB of plausible opening, closing, or self-closing tag syntax with quote awareness; leading HTML comments are source-like. It uses no parser, URL constructor, or backtracking regular expression.

## Fresh GREEN and broader evidence

Post-smoke-policy-correction canonical route suite:

- 1 file passed.
- 68/68 tests passed.
- Duration 2.07 seconds after the fragment-detector correction.

Focused route-metadata coverage:

- 1/1 file passed.
- 10/10 tests passed.

Extension compile, run from `apps/extension`:

```text
bun run compile
```

- `tsc --noEmit -p tsconfig.compile.json` exited 0.

Directly impacted 12-file matrix:

- 12/12 files passed.
- 218/218 tests passed.
- Covered route metadata/path/start, design-system surface, index, Page and offline standalone lifecycle, canonical WebUI helper, presentation normalization/client boundaries, logout, and server capabilities.
- Duration 13.36 seconds.

Route governance characterization:

- `route-governance.sidepanel-availability.test.ts`: 5/5 passed.
- `route-governance.metadata-coverage.test.ts`: 2 passed and 4 inherited failures.
- The four failures are repository baselines unrelated to Task 16: 36 existing shared routes without metadata; 50 active smoke routes without metadata; skipped `/media/123/view` and `/settings/image-gen` entries lacking valid metadata/reasons; and `/settings/image-generation` declared included but absent from the page inventory.
- No Task 16 Presentation Studio path appears in those failure lists. New, start, and dynamic Presentation Studio routes use `smoke: "manual"`, consistent with WebUI-only, extension-only, or dynamic routes outside the active WebUI smoke inventory.

The feasible real Stage-2 static contract was also run directly:

```text
npx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line --grep "route metadata smoke policy is represented in the page inventory"
```

- It ran 1 test and failed only on the inherited `/settings/image-generation` page-inventory mismatch already recorded above.
- `/presentation-studio/start` was absent from `missingSmokeRoutes`, proving the Task 16 correction reaches the Stage-2 helper.

Vitest initially could not create its jsdom temp directory under the restricted sandbox (`EPERM`). The same exact commands were rerun with filesystem access and produced the counts above; the blocked attempts collected zero tests and are not presented as test evidence.

## Built-extension Playwright disposition

Planned command:

```text
cd apps/extension
bunx playwright test tests/e2e/presentation-studio-start.spec.ts --reporter=line
```

- The production extension build succeeded at `.output/chrome-mv3` (43.28 MB, with existing build warnings).
- The amended Task 16 spec collected 4 tests and reported **1 passed and 3 skipped** in 31.1 seconds against the freshly built artifact.
- The passing case is a real Chromium-page instrumentation audit covering distinct Chrome/browser local, sync, session, and runtime surfaces; benign metadata/international-prose controls; and forbidden full-document, fragment, comment-leading, camelCase HTML, draft, and version payloads.
- A fresh fragment-aware focused rerun passed that browser contract in 2.0 seconds, and Playwright compilation/collection listed all 4 cases.
- The three extension-launch cases remain skipped, not passed. An earlier forced bundled-Chromium retry also reported the same three skips.
- JSON diagnostics reported: `Extension launch unavailable in this environment (Error: Could not determine extension id from [no extension targets]).`
- The skip originates at `apps/extension/tests/e2e/utils/real-server.ts:249`.
- The unchanged calibration `tests/e2e/utils/extension-launch-health.spec.ts` also reported **1 skipped** after 30.471 seconds for the identical no-extension-target condition.

This is an environment-wide extension-launch limitation. The executable instrumentation contract passed, but the three launch-dependent flows remain an unresolved browser gate and are not described as passed.

## Static, diff, type, and product review

- `git diff --check`: exit 0.
- Execution/preview sink scan over Task 16 production files (`innerHTML`, `dangerouslySetInnerHTML`, iframe/srcdoc, eval/function construction, Blob/object URL): no matches.
- Source-bearing client/payload scan (`html_document`, `html_source`, detail/version/save/download/export/render methods): no matches.
- Storage, IPC, and logging scan (`localStorage`, `sessionStorage`, extension storage, `sendMessage`, `postMessage`, `console`): no production matches.
- Navigation scan found only application-owned index routes, the canonical helper, fixed encoded Presentation Studio route, and `_blank` `noopener,noreferrer` opens.
- Copy/style scan found no em dash, gradient, glass, card-grid, or decorative-motion additions in Task 16 production files.
- Extension compile is the Task 16 type gate and passed. No backend/OpenAPI files changed, so no OpenAPI regeneration or backend type gate was applicable.
- A shared-path ESLint attempt exited 0 but emitted six `File ignored because outside of base path` warnings, including with `--no-ignore`; it did not lint the requested shared files and is recorded as unavailable rather than clean lint evidence.
- Bandit is not applicable because Task 16 touches no Python source.

Product/accessibility self-review:

- The handoff uses `PageShell`, `StatePanel`, `Button`, `Badge`, and shared tokens rather than a new visual system.
- Ready states have one clear `h1`; loading/error/offline/auth/capability paths remain bounded and semantic; failures expose retry where recovery is possible.
- All untrusted metadata renders through React text nodes. No source preview or execution affordance is present.
- The primary action is keyboard reachable and retains shared visible-focus behavior. Provenance and kind are visible text, not color-only status.
- Copy is concise and literal, with no em dash, gradient, glass, card-grid, or decorative motion.

## Remaining concerns

1. The three real built-extension launch flows remain unexecuted because this environment exposes no extension targets; rerun them and the launch-health calibration in an extension-capable Chromium environment.
2. The four route-governance metadata failures, including the Stage-2 `/settings/image-generation` mismatch, are inherited repository baselines outside the approved scope.
3. The current shared ESLint configuration ignores these cross-package paths, so extension compile and the focused/broader Vitest suites provide the executable static/type evidence for this task.
