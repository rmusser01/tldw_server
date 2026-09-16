# Independent review — final Cycle4 Task5 / UAT113 and UAT117

## Result

No actionable findings in the requested final working diff against `eb8782e5fd8bcf8d931c6a690b962b20c7483365`. The bounded 113/117 changes and delegated recovery transport correction are clear for coordinator-owned combined UI/compiler verification and native acceptance. The preceding Retry108 review is separately recorded at `/private/tmp/cycle4-task5-retry-independent-review.md`.

## Frozen scope

Reviewed the production changes in Playground route coordination, Playground Message recovery display, reasoning helper, chat pipeline, character chat action, error persistence helper/types, base and delegated persistCharacterCompletion methods, and the exact service-prompt path allowance. Inspected their corresponding permanent tests and adjacent session-scope/recovery implementations.

Authoritative freeze: `/private/tmp/cycle4-task5-code-freeze.json`, timestamp `2026-09-16T03:28:25.266Z`. All **24 file hashes matched** before the final verification and again afterward. Scoped `git diff --check` against the base passed. No repository changes, private probes, browser/runtime operations, inference, subagents, compiler runs, or commits performed.

## Assessment

### Character route consumption

The route is promoted only after current metadata identifies the selected character and the valid, resolved persisted session agrees with the active saved conversation ID. The existing session hook masks persisted IDs outside its current valid scope. Query and extension hash routes use the same narrowly changed parameter helper; unrelated parameters survive. The real coordinator/session tests cover remounting the saved transcript and explicit new-character navigation, including selecting the same character again. Delayed old-scope hydration cannot publish the abandoned target in the covered logout and A-to-B-to-A cases.

### Missing final answer

The classifier uses the existing reasoning parser and requires a recognized reasoning tag with no nonblank text part. Closed, unclosed, mixed supported tags, and empty tags are covered; ordinary prose and an actual final answer are positive controls. Structured reasoning chunks enter this classification through the existing stream normalizer. Normal pipeline and tracked-character execution preserve the trace but report a failed/interrupted completion rather than successful final prose. Restored reasoning-only rows expose recovery even when interruption metadata was not persisted. Active streams, tool/image output, and final-answer text are excluded from this fallback display.

The error persistence contract now carries the canonical assistant ID alongside the existing user ID across its branches. The real normal pipeline persistence test confirms one acknowledged pair, retained raw trace/history, recoverable display metadata, and no extra server-message insertion. Character tests cover acknowledged recovery IDs and existing fallback/degraded persistence outcomes.

### Captured recovery authority and transport

The character recovery path receives the existing turn snapshot, checks invalidation before recovery and after awaited persistence, passes captured requestScope/signal to bounded recovery transports, and passes the same snapshot into local error persistence. An invalidated response cannot publish recovered IDs or locally persist the old owner's response in the dedicated delayed-response test.

Both TldwApiClientBase and the delegated chat-rag implementation forward optional scope fields and AbortSignal outside the completion payload. The permanent transport test instantiates the actual composed TldwApiClient, so it exercises the delegated implementation that originally dropped these fields. Unscoped compatibility is preserved. The new allowlist entry is limited to POST /api/v1/chats/<id>/completions/persist, with the existing canonical-path validation and explicit negative controls for other verbs, suffixes, empty IDs, encoded slashes and traversal.

## Independent verification

From `apps/packages/ui`, using its existing Vitest configuration:

```sh
./node_modules/.bin/vitest run \
  src/libs/__tests__/reasoning-final-answer.test.ts \
  src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts \
  src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts \
  src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx \
  src/components/Common/Playground/__tests__/Message.error-recovery.integration.test.tsx \
  src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx \
  src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts \
  src/services/tldw/__tests__/service-prompt-scope-error.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

**171 tests passed in 8 suites**, exit 0, 15.58 seconds.

```sh
./node_modules/.bin/vitest run \
  src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx \
  -t 'reasoning-only|real failed-turn regeneration' \
  --maxWorkers=1 --no-file-parallelism
```

**2 passed / 46 intentionally skipped**, exit 0, 1.48 seconds. Total for this final review: **173 passing tests**.

Existing Node experimental localStorage warnings and intentional fallback/degraded-persistence console errors occurred; Vitest reported no failing tests or unhandled errors.

## Limits

This is a code and focused mocked-service integration review. Route tests inject the acknowledged saved target rather than issuing inference; real pipeline tests mock network/database adapters. No live server or full UAT acceptance is claimed. The review does not introduce a new session/lease abstraction or audit every pre-existing character lifecycle path. Full compiler, broader merged suites, lint-baseline comparison, and native behavior remain coordinator-owned. The earlier pre-correction `cycle4-task5-final-tests.log` is not treated as final green evidence; the independent runs above include the corrected delegated transport.
