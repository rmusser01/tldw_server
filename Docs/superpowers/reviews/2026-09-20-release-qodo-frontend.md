# Qodo frontend findings — TASK-13263.1

Workspace: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`, branch `codex/release-main-0.1.43`. No commits, branch switches, publication, or shared task/plan edits.

## Dispositions

- **15 Fixed.** `StandaloneHtmlWorkspace` now uses the playground translation namespace for statuses, recovery/conflict headings, descriptions, confirmations, navigation/actions, guard states, and accessible labels. English keys are in `playground.json`. Async notices retain message keys and translate during render, so an existing conflict notice follows locale changes without restarting request effects. Regression exercises localized recovery, discard confirmation, conflict status and notice, then a locale change.
- **16 Fixed.** Direct streaming refresh explicitly recognizes request cancellation and propagates the established AbortError instead of parsing the original 401. JSON response parsing no longer swallows an aborted refresh-body read. Parameterized regressions cover abort during refresh fetch and body read and assert the original response is not read. Genuine refresh errors still follow the existing original-response behavior.
- **17 Fixed.** Shared `useTTS` uses `browser` from `wxt/browser` for speak/stop. Requires a real extension runtime ID because the web shim exposes a no-op `tts` object; native `speechSynthesis` remains the web fallback. Regression verifies extension speak/start/cancel through the wrapper with `chrome` absent. Existing browser-synthesis tests now also run with the Chromium platform flag enabled and shim runtime empty.
- **18 Already resolved.** `apps/tldw-frontend/pages/login.tsx` is a one-line re-export from `@web/routes/login`. Domain/UI code already moved out of the page. Keeping the route web-local is appropriate because it imports Next routing, Head, Link, and web RouteRedirect; moving these into shared `@/routes` would violate the documented shared package restriction on web-only imports. Both existing login suites pass; no edit needed.
- **20 Fixed.** Completed-ingest ID extraction now scans every result entry in order after top-level identifiers, checking all three aliases for a valid persisted identifier (nonempty string or finite positive number). Tests cover failed/skipped first items, invalid identifiers, later successful entries, and top-level precedence. Existing warning logic still requires the warning payload's own persisted ID and does not borrow IDs from mixed aggregates.

## Changed files

- apps/packages/ui/src/assets/locale/en/playground.json
- apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlWorkspace.tsx
- apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlWorkspace.test.tsx
- apps/packages/ui/src/hooks/useTTS.tsx
- apps/packages/ui/src/hooks/__tests__/useTTS.cancel.test.tsx
- apps/packages/ui/src/services/background-proxy.ts
- apps/packages/ui/src/services/__tests__/background-proxy.web-refresh.test.ts
- apps/packages/ui/src/services/tldw/ingest-job-results.ts
- apps/packages/ui/src/services/__tests__/ingest-job-results.test.ts

## Verification

From `apps/tldw-frontend`:

```sh
node_modules/.bin/vitest run ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlWorkspace.test.tsx ../packages/ui/src/services/__tests__/ingest-job-results.test.ts ../packages/ui/src/services/__tests__/background-proxy.web-refresh.test.ts ../packages/ui/src/hooks/__tests__/useTTS.cancel.test.tsx ../packages/ui/src/hooks/__tests__/useTTS.gateway-metadata.test.tsx __tests__/navigation/login-page.test.tsx __tests__/pages/login-page.test.tsx --reporter=dot
```

**7 files, 222 tests passed.** Log `/tmp/qodo-frontend-final-tests.log`. Test-first failures confirmed the missing ingest scans, swallowed refresh cancellation, bypassed browser wrapper, and untranslated recovery region (`/tmp/qodo-frontend-red.log`, `/tmp/qodo-frontend-i18n-red.log`). Vitest reports Node experimental localStorage warnings and expected error logging in auth failure tests.

```sh
NODE_OPTIONS=--max-old-space-size=8192 node_modules/.bin/tsc --noEmit --incremental false
```

**Exit 0, no diagnostics**, `/tmp/qodo-frontend-tsc.log`.

From repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlWorkspace.tsx apps/packages/ui/src/hooks/useTTS.tsx apps/packages/ui/src/services/background-proxy.ts apps/packages/ui/src/services/tldw/ingest-job-results.ts
```

**0 errors, 21 existing warnings**, `/tmp/qodo-frontend-production-eslint.log`. All-touched-file lint additionally reports the pre-existing `unstable_usePrompt` mock calling `ReactModule.useEffect`, which triggers hooks naming rules; verified this exact code is present in HEAD. Existing tests contain broad `any` warnings; no new explicit `any` remains in added tests. `git diff --check -- apps/packages/ui/src` passes.

Bandit was invoked from the root venv on the four touched production files, writing `/tmp/bandit_qodo_frontend.json`. **Not applicable coverage:** Bandit reports Python AST parse errors for these TypeScript files; do not treat its empty findings as a security scan pass. No Python code changed in this scope.
