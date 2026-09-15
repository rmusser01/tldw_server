# TASK13260.36 — failed Chat transport sanitation

Status: production/tests frozen for final independent review. No commits, browser/runtime changes, or native acceptance performed. Parent owns final combined TypeScript checkpoint and commit.

## Scope and behavior

Exact seven owned paths (one production module, five test files, task record) and SHA-256 hashes are in `/private/tmp/uat036-owned-paths.json`.

Only production change: `apps/packages/ui/src/services/background-proxy.ts`.

- Recognize failed canonical `POST /api/v1/chat/completions` requests, including optional trailing slash/query, in the existing failure handler shared by direct and extension transport.
- Reuse `sanitizeServerErrorMessage` for error text and opt into string sanitation in the existing recursive sensitive-field sanitizer. Error detail arrays/objects, ordinary codes, numeric values, status and retry metadata remain usable. Known sensitive fields retain the existing `[REDACTED]` policy.
- Sanitize before console request warnings, stored request-error diagnostics, rejected error construction and `returnResponse` delivery. Successful payloads and unrelated routes retain their existing behavior. Client/domain/auth production files are untouched.
- Preserve already-established cancellation classification before first-line sanitation. Explicit cancellations use `Aborted`, as the existing RAG error path does, so rejection remains `AbortError`/`REQUEST_ABORTED`, returned responses remain recognizable as cancellation, and no ordinary request error is logged.

## Root cause and stale assertions

The API client correctly treats a resolved `bgRequest` value as a successful completion and preserves it. The old `chat-mutations` assertion expected this successful payload to be scrubbed, contradicting existing success-preservation tests and potentially corrupting assistant code/examples. It now verifies complete preservation.

The actual error gap was in proxy failure normalization: generic messages and nested string values were passed to warnings, diagnostics, rejections and full response callers without the existing error-string sanitizer. Real synthetic 500/422 responses reproduce it through client → proxy → request core → fetch.

Two additional preexisting test mismatches were corrected without production expansion:

- The utility secret-redaction test required the literal label `Bearer [redacted-secret]`, although unchanged production emits `[redacted bearer token]` for Authorization headers. Removed only that marker-format assertion; all actual no-secret checks remain. The utility implementation is unchanged.
- The web-refresh scope negative fixture treated GET `/api/v1/chats/chat-123/messages` as prohibited, although committed Chat policy now allows it. Replayed exact HEAD proxy to reproduce the same failure, then changed only the negative fixture to `/messages/extra`, retaining no-dispatch assertions.

## RED → GREEN evidence

- `/private/tmp/uat036-red-confirmed.log`: five direct/extension failure or full-response controls RED; seven success/cancellation/other-route controls already pass. Failure assertions show raw private paths in message/details/returned data.
- `/private/tmp/uat036-quick-test-red.config.ts` replays exact original proxy source retained at `/private/tmp/uat036-before-proxy.ts`. Actual `usePromptInteractions` Quick Test hook → client → proxy → request core → fetch reproduces one failing notification leak control and one passing successful-output control. Log: `/private/tmp/uat036-quick-test-red.log`. Corrected source passes both; `/private/tmp/uat036-quick-test-green.log`.
- `/private/tmp/uat036-web-refresh-baseline.log`: exact original proxy reproduces the stale route-negative failure (seven other negatives pass).
- Independent review confirmed a first-line cancellation regression using real non-2xx 499 detail `Provider\nrequest aborted.`. Original source recognized it as cancellation; initial sanitation lost the suffix. Permanent direct/extension controls RED in `/private/tmp/uat036-cancellation-permanent-red.log`; corrected source preserves both rejection and returnResponse cancellation behavior. The reviewer’s **unchanged** `/private/tmp/uat036-independent-cancellation.config.ts` passes two controls in `/private/tmp/uat036-cancellation-original-green.log`.
- The original leak-characterization config `/private/tmp/uat034-sanitization-probe.config.ts` remains unchanged. It intentionally asserts the old leak; it is historical reproduction evidence, not a final GREEN expectation.

## Final verification

From `apps/packages/ui`:

```sh
npx vitest run src/services/__tests__/background-proxy.test.ts src/services/__tests__/background-proxy.monitoring-scope.test.ts src/services/__tests__/background-proxy.web-refresh.test.ts src/services/__tests__/tldw-api-client.chat-mutations.test.ts src/services/__tests__/tldw-api-client.chat-sanitization-regression.test.ts src/utils/__tests__/server-error-message.test.ts src/components/Option/Prompt/__tests__/usePromptInteractions.quick-test-errors.test.tsx --maxWorkers=1 --no-file-parallelism
```

**217 tests / 7 suites pass**, `/private/tmp/uat036-final-corrected-tests.log`.

Original reviewer probe:

```sh
npx vitest run --config /private/tmp/uat036-independent-cancellation.config.ts --maxWorkers=1 -t 'independent cancellation:'
```

**2 pass**, `/private/tmp/uat036-cancellation-original-green.log`.

Root ESLint command uses `apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs` against all six source/test paths in the manifest. **Every file covered; 0 errors, 100 unchanged baseline warnings, 0 added.** Current/baseline/comparison JSON: `/private/tmp/uat036-eslint-{current,baseline,comparison}.json`. Baseline uses exact HEAD file contents with `--stdin-filename` set to the real source path. The root pages-directory advisory is retained in stderr; no ignored-file result counted as verification. `git diff --check` passes for this scope.

Full TypeScript was deliberately not run concurrently with the active .33 production changes, per parent coordination. Known current baseline is 90 existing diagnostics, not a clean compiler; root owns the final combined stable-source comparison. Bandit is not applicable to this TypeScript-only scope.

## Evidence limits

No live backend disclosure was observed. Tests use synthetic fetch responses and valid synthetic manual device credential metadata. The extension test forwards the actual runtime payload to the real request core at a mocked extension-message boundary; it does not certify a launched extension service worker or popup lifetime. Worker source delegates these responses to the same request core without adding a separate raw-response log on this path. The Quick Test hook and notification sink are real; unrelated startup discovery and provider catalog queries are stubbed. No new dependencies or generic auth changes.
