# TASK13260.50 — UAT109/110 read-only diagnosis

Confirmed, on source matching cycle4 freeze7c9409fad2. No product/test, browser, runtime or global-document edits. Three private probes exercise actual handlers/transport and fail only the intended no-overlay/no-unhandled expectations. Full matrix remains running; no repair applied.

## Findings

### UAT109: discarded asynchronous stream cleanup

`background-proxy.ts` direct SSE transport creates an idle error then aborts its fetch controller (1684–1686). `reader.read()` rejects; catch selects the intended idleError and the RAG sanitizer converts it into a safe timeout. However finally calls `reader.cancel()` inside a synchronous try/catch without awaiting or handling its returned Promise (1836). Cancelling an already errored browser body returns a rejected Promise carrying the original AbortError. That second rejection is unowned and bypasses the primary generator error handling/sanitizer.

Private actual bgStream + HTTP200/native ReadableStream probe reproduces the exact native text: the primary operation is caught as `RAG search timed out. Try again.` while a separate unhandled rejection is `AbortError: BodyStreamBuffer was aborted`; fetch called exactly once. Native evidence already shows local actionable timeout UI beneath the AbortError overlay, consistent with two error paths rather than an uncaught QA search Promise.

### UAT109: handled QA failure also enters Next's console overlay channel

`KnowledgeQAProvider.tsx:2742` calls console.error with `Search failed:` and a sanitized log code after its catch has classified the failure. The actual handler probe resolves successfully, preserves the question, sets `isSearching=false`, exposes `Search timed out. Try the Fast preset or reduce sources.`, and makes no non-stream fallback call. It still emits the console.error. This is a second handled-error presentation path; fixing only reader cleanup would leave it.

### UAT110: caught Media failure is re-reported as a development runtime error

`AnalysisModal.tsx:463–479` already catches the failed completion, shows local failure feedback and resets generation in finally. It then calls `console.error('Generation error:', err)`. Actual handler probe verifies its returned Promise is fulfilled, prior analysis is not replaced (`onAnalysisGenerated`0, no versions request), modal remains open (`onClose`0), and Generate is usable again. The no-console-overlay assertion fails on that explicit console.error; it is not evidence of a missing outer catch.

Installed Next16.1.4 `next-devtools/userspace/app/errors/intercept-console-error.js:35–54` chooses an Error in args[1] when args[0] is a label, then calls handleConsoleError. Its `use-error-handler.js:44–59` also queues formatted string-only console errors. Thus both the Error-valued Media log and sanitized-code QA log can create development overlays even when their business promises are handled. Production logging behavior differs; this report does not claim production Next renders the dev overlay.

## Private proof and limits

- `/private/tmp/uat109-idle-cleanup-readonly.config.ts` + `.log`:1RED. Existing proxy test harness, real proxy generator and real native Response/ReadableStream, only storage/runtime/fetch boundary controlled. An abort event errors the HTTP200 body; process unhandledRejection observer records the leaked cleanup promise. No network access.
- `/private/tmp/uat109-qa-handler-readonly.config.ts` + `.log`:1RED. Actual KnowledgeQAProvider with existing authority fixture and mocked service error. Search promise fulfilled; local error/draft/fallback assertions pass; console-error channel assertion fails.
- `/private/tmp/uat110-analysis-catch-readonly.config.ts` + `.log`:1RED. Actual AnalysisModal handler with existing UI/transport boundary mocks. The test Button adapter records the actual returned promise; fulfilled/local error/no save/no close/button reset controls pass; console-error channel assertion fails.
- Transport's first attempt lacked the synthetic manual-credential scope metadata and failed before fetch (fetchCalls0). It is retained as `uat109-idle-cleanup-harness-credential-gap.{config.ts,log}` and explicitly is not product evidence. Final config includes the existing fixture's full synthetic credential metadata; final result fetchCalls1.
- These are independent narrow probes, not an end-to-end real-provider test. Native failures remain the primary browser evidence. No global error listener suppression is proposed.

Run each from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat109-idle-cleanup-readonly.config.ts --maxWorkers=1 --no-file-parallelism
./node_modules/.bin/vitest run --config /private/tmp/uat109-qa-handler-readonly.config.ts --maxWorkers=1 --no-file-parallelism
./node_modules/.bin/vitest run --config /private/tmp/uat110-analysis-catch-readonly.config.ts --maxWorkers=1 --no-file-parallelism
```

## Bounded repair after full matrix

1. Own the asynchronous `reader.cancel()` cleanup rejection locally without replacing the original timeout/caller-cancel/provider error. Keep abort listener removal/timer cleanup, no write replay or deadline changes.
2. Keep expected QA/Analysis failure feedback local; use a non-overlay diagnostic path with safe error codes/status if logging is needed. Preserve safe QA log-code/privacy contract, genuine error classification and current draft/old analysis.
3. Permanent tests in existing proxy, KnowledgeQAProvider streaming and AnalysisModal suites: actual errored ReadableStream on idle and explicit caller abort, normal completion/DONE, no leaked cleanup rejection, exactly one handled timeout, safe local feedback, failed-request retry, original source/analysis unchanged and no write on failure. Retain existing owner/stale-result/cancellation guards; do not relax server failures into success.

Scope indicated: `services/background-proxy.ts`, `components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`, `components/Media/AnalysisModal.tsx`, and their existing focused tests. Other Analysis persistence catches use console.error too, but the observed UAT110 is generation; broad logging cleanup is not required by this diagnosis. Any added provider recovery text should preserve unavailable versus missing-provider versus timeout/cancel distinctions. No task acceptance criterion is satisfied merely by this read-only diagnosis.
