# Task 0 report — Shared sidepanel mock-server helper extraction

Status: DONE_WITH_CONCERNS (deliverable complete and behavior-preserving; one pre-existing, unrelated smoke-test failure documented below)

## What was done

- **Created** `apps/extension/tests/e2e/utils/sidepanel-mock-server.ts`
  exporting `startSidepanelMockServer(opts)` / `stopSidepanelMockServer(server)`
  (+ `SidepanelMockServer`, `SidepanelMockServerOptions`,
  `SidepanelMockChatRequest` types).
- **Refactored** `apps/extension/tests/e2e/performance-streaming.spec.ts`
  (canned suite only; the live-backend suites are untouched) and
  `apps/extension/tests/e2e/sidepanel-chat-smoke.spec.ts` to consume the
  helper. No product code touched.

## Design: what was unified vs. kept per-spec

Diff of the two former implementations found the scaffolding byte-identical
(CORS headers on JSON responses and the OPTIONS 204 preflight,
`/api/v1/health`, `/api/v1/llm/models`, `/openapi.json`, the JSON 404
catch-all, the request-body reader, ephemeral-port listen on 127.0.0.1, and
the stop path: `close()` + `closeAllConnections()` + 1s unref'd fallback).
All of that moved into the helper verbatim.

The genuine differences are opt-driven, per the brief's "do NOT force-unify"
ruling:

- `modelId` / `modelName` — metadata payloads differed (`perf-stream-model` /
  "Perf Stream Mock Model" vs `mock-model` / "Mock Model").
- `handleChatCompletions(request)` — the one load-bearing behavioral split:
  - canned perf spec: always-SSE paced 100-chunk stream with
    `res.destroyed || res.writableEnded` guards and the `chunksServed`
    counter (kept as a spec-local closure; `CannedStreamServer` now
    `extends SidepanelMockServer` adding `chunksServed()`).
  - smoke spec: parses the request body's `stream` flag; `stream: false`
    gets the JSON "Mock reply from Playwright" reply, otherwise the
    immediate 2-chunk SSE response. Moved verbatim into the handler.
  The helper fully reads the body before invoking the handler (both originals
  read it themselves) and ignores the handler's return value, so the moved
  `return res.end()` / `return sendJson(...)` statements compile unchanged
  (handler return type is `void | Promise<unknown>`).

### One deliberate micro-unification (flagged)

The canned server had `server.on("error", () => {})` (ignore socket errors
after browser-context teardown); the smoke server did not. The helper applies
it unconditionally. This is not client-observable (no route/header/lifecycle
change; listen/close semantics identical) and only guards the same
teardown race for smoke that canned already guarded — smoke's SSE responses
end synchronously so the race never fired there in practice. Revertible by
an opt if a reviewer prefers strict parity.

## Requirement 3 ruling: `[SW_CONSOLE]` 404 noise — left as-is

The passing canned run surfaced the exact noise class:

```
[SW_CONSOLE] warning [tldw:request] GET /api/v1/notifications/unread-count 404 not found
[SW_CONSOLE] warning [tldw:request] GET /api/v1/config/docs-info 404 not found
[SW_CONSOLE] warning [tldw:request] GET /api/v1/llm/models/metadata?refresh_openrouter=true 404 not found
[SW_CONSOLE] error Failed to fetch models from tldw: Error: not found (GET /api/v1/llm/models/metadata?refresh_openrouter=true)  (+ 10-line stack)
```

- The brief's first suggested fix ("the helper's 404 handler not logging") is
  already the status quo: neither original nor the helper logs server-side;
  the noise is the extension SW's own console output relayed by
  `utils/extension.ts`'s `sw.on('console')` logger.
- The remaining fixes would change mock behavior, which this task's
  zero-behavior-change ruling forbids: serving 204s for
  `/api/v1/notifications/unread-count` / `/api/v1/config/docs-info` turns
  currently-failing probes into successes (downstream SW paths unverified),
  and matching `/api/v1/llm/models/metadata` by pathname (to absorb the
  `?refresh_openrouter=true` variant) would let the background `warmCache`
  succeed where it 404s today. Both need product-path validation — scope
  expansion per the brief ("If not trivial, leave it").
- Recommendation for W1+: either pathname-match the metadata route in the
  helper when a consumer opts in, or filter `tldw:request` 404 warnings in
  the extension.ts SW console relay.

## Verification (scoped runs, cached build)

- Typecheck: scoped `tsc --noEmit` over the three touched files — no errors
  in them (pre-existing ambient-`chrome` errors in other transitively-loaded
  utils are invocation-context noise; `tests/` is excluded from
  `tsconfig.compile.json`, which stays green by not regressing includes).
- `bunx playwright test tests/e2e/performance-streaming.spec.ts`:
  **1 passed** (canned 100-chunk stream — 100 chunks served/applied,
  all harness assertions green) + **5 skipped** — the live-backend suites
  skip env-gated exactly as before; their code is untouched.
- `bunx playwright test tests/e2e/sidepanel-chat-smoke.spec.ts`:
  **2 passed, 1 failed** — see concern below. Smoke is not env-gated locally
  (it only skips when extension launch fails); collection/gating unchanged.

## Concern: pre-existing smoke failure (proven not caused by this refactor)

`keeps packaged /chat handoffs route-only and rail-safe` fails on a product-UI
assertion: the `chat-open-full-screen` button `title` is
"Opens this selected history in the extension full page. Active streaming
stays in this panel." but the spec expects
"Opens /chat in a new tab. Use Continue in WebUI from the composer tools to
carry a draft or page context."

Proof it pre-exists: stashed all task-0 changes and re-ran that single test
on the pristine tree — fails identically (same assertion, same strings).
Restored the changes afterwards. Fixing it requires either spec-assertion or
product changes, both outside this task's scope (mock-server portion only,
no product code). Flagging for the dispatcher: the W1 branch inherited a
red smoke test from `dev` (post PR #3212 merge); the strict "both specs must
pass scoped" bar cannot be met without that separate fix.

## Other checks

- Stale references: no remaining uses of the removed spec-local symbols
  (`stopChatMockServer`, `readCannedRequestBody`,
  `stopCannedStreamServer`); `startChatMockServer` survives as a thin
  wrapper supplying smoke's options to the shared helper.
- Whitespace/EOF: touched files satisfy the pre-commit trailing-whitespace
  and end-of-file hooks (verified manually).
- Bandit: not applicable — touched files are TypeScript test infrastructure;
  Bandit scans Python (documented skip per DoD #4).
- Backlog: TASK-13521 set to In Progress with a Task 0 implementation note;
  committed with the work.

## Commit

Single commit on `codex/webui-perf-w1-chat-streaming` referencing TASK-13521
(see dispatch reply for SHA).
