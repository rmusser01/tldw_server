# Independent review — Cycle4 Task5 / Retry108 slice

## Result

No actionable findings in the requested Retry108 slice against `eb8782e5fd`. Clear for parent-owned combined verification and targeted live acceptance.

## Reviewed behavior

- `PlaygroundForm.tsx:3039` delegates Retry to the existing regeneration action rather than appending a new submission. Its event-handler binding and `regenerateLastMessage` acquisition were the only production Form changes reviewed.
- Traced the existing `createRegenerateLastMessage` integration through useMessageOption/useChatActions: it removes the failed assistant, rebuilds memory before the prior user turn, carries the original image/message type, and submits with `isRegenerate`, explicit memory/messages, and the failed assistant target.
- Checked error-history construction: saveMessageOnError appends the failed user plus display assistant to history. Thus the handler's latest history user identifies the failed request after earlier successful turns as well. The focused actual pipeline test exercises first-turn failure/retry; multi-turn reasoning here is from source inspection, not a new executable regression.
- `generate-history.ts:58` skips only assistant content accepted by the existing display-error decoder, before both custom and ordinary model projection. User quotations, malformed marker text, and ordinary/partial assistant prose retain the pre-existing projection behavior. The check does not broadly remove error-looking natural language.
- The Form regression mocks the message hook action to assert delegation; the separate real useChatActions/pipeline regression confirms the failed request is projected once and the display payload does not enter model context. PromptAssist/Home mocks isolate unrelated mounted behavior without replacing the Retry control.

## Independent verification

All commands ran from `apps/packages/ui`, using its existing Vitest config:

```sh
./node_modules/.bin/vitest run \
  src/utils/__tests__/generate-history.image-generation.test.ts \
  src/components/Option/Playground/__tests__/PlaygroundForm.pinned-fallback.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

**7 passed / 2 suites**, exit 0 (2.28 seconds).

```sh
./node_modules/.bin/vitest run \
  src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx \
  -t 'real failed-turn regeneration' --maxWorkers=1 --no-file-parallelism
```

**1 passed / 47 intentionally skipped**, exit 0 (1.49 seconds). Asserts one intended user in the projected retry, no display-error marker, one visible user, and recovered final answer.

```sh
./node_modules/.bin/vitest run \
  src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

**3 passed**, exit 0. Existing handler controls cover prior-user regeneration, invalid setter safety, and pre-submit overrides.

Scoped diff check against the review base passed. Retained RED log confirms the new projection and Retry-delegation tests failed for their intended assertions before the repair (2 failed / 5 passed).

## Limits

Bounded review: no approval implied for in-flight UAT113/117 or other shared-file changes. The reasoning-only integration addition in the same test file was excluded. Did not repeat the author's broad 19-suite run, lint, compiler, full UAT, or perform runtime/browser/inference operations. No repository edits or commits. Existing Node experimental localStorage warnings occurred; focused runs reported no failures. A live server's persistence/network behavior remains parent-owned acceptance.
