# Independent review: UAT108 Retry persistence

## Status

Review of the remaining Chat slice against `d94077c3543121f35f31fb9cd415c379600d2e8e` (UAT105 excluded). **Clear after correction: no actionable findings remain.** The P2 ordinary-successful-regeneration ACK regression below was reproduced, corrected and reverified. No product edits/browser/inference were performed by this reviewer.

## Resolved P2: gate existing-user acknowledgement to explicit failed-turn Retry

`apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts:230` resolves an existing user ID for every regeneration; subsequent ACK merges at394/854/1028 and helper calls in `hooks/chat-helper/index.ts:213,577` likewise do not distinguish failed Retry from successful answer regeneration. A normal regeneration with `retryFailedTurn=false` can receive a fresh canonical user ACK from existing backend repeated-turn behavior. It then repoints the original local user to that new ID; the strict mirror helper can reject its contradictory previous ID after in-memory state has changed.

Private actual `useChatActions`/pipeline probe reproduced original `server-user-1` changing to `server-user-2` although outbound retry intents were `[false,false]`. The unchanged probe copies the existing permanent harness, replaces the DB acknowledgement adapter with the production helper's contradictory-ID check, and supplies provider ACKs matching ordinary backend repeated-turn semantics. It does not claim native IndexedDB or live inference.

Probe: `/private/tmp/cycle4-uat108-successful-regeneration-probe.test.tsx`; config `/private/tmp/cycle4-uat108-successful-regeneration-probe.config.mts`; RED log `/private/tmp/cycle4-uat108-successful-regeneration-probe.log` (1 failed/48 unselected). From `apps/packages/ui`:

```
./node_modules/.bin/vitest run --config /private/tmp/cycle4-uat108-successful-regeneration-probe.config.mts -t 'private probe successful regeneration'
```

The config imports the standard UI Vitest config, changes include to the private file, and resolves that private file's bare package imports through the UI package. An initial resolution-only startup failure was corrected and is not product failure evidence.

## Independent verification already complete

- Backend: 76 passed across actual Chat endpoint/database integration and history/streaming unit suites. Log `/private/tmp/cycle4-uat108-independent-backend.log`. Six test warnings and existing pytest temporary-directory cleanup warnings retained. No new backend finding.
- Frontend pre-correction: 82 passed across five modified suites. Log `/private/tmp/cycle4-uat108-independent-ui.log`.
- Inspected meaningful API RED3failed/2passed, actual model invoke forwarding RED, and final valid-i18n actual pipeline baseline RED1failed/47unselected. Earlier fixture/config failures described in the author report are not counted as product RED.
- Verified explicit intent remains request metadata, does not reach provider kwargs/model context, and distinguishes a genuine Retry from ordinary equal-text submission. Real API tests persist the user before provider502 and return the same user ACK for explicit stream/nonstream Retry. Conflicting unanswered tail409 and no-persisted-user fallback are covered. Tail identity is checked outside the context-window selection; saved text/images are compared before reuse.
- Existing request owner/abort guards and scoped mirror transaction remain in place. Helper updates only serverMessageId on a matching history/user/content row; changed text is left unacknowledged and row metadata/images are not overwritten.

Backend command from repository root:

```
source .venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py
```

Frontend command from `apps/packages/ui`:

```
./node_modules/.bin/vitest run --config vitest.config.ts src/db/dexie/__tests__/helpers.user-acknowledgement.test.ts src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/services/tldw/__tests__/TldwChat.abort.test.ts
```

## Limits

No native runtime, browser, inference, full UAT or full compiler run in this review. Endpoint providers and browser storage seams are mocked in tests. Historical duplicate cleanup is outside this forward-only repair. Inspected author static results: Bandit0findings, ESLint0errors/35unchangedwarnings, Ruff15unchangedbaseline findings. Final hash and post-correction results are recorded below.

## Final correction and verification

The pipeline computes explicit failed-turn intent once, resolves the prior user only for that branch, and forwards the same optional flag into both success and error mirror helpers. Omitted/false defaults preserve ordinary regeneration behavior. The declaration-only SaveMessageBase addition is included in review. Permanent controls exercise ordinary successful regeneration plus direct helper omitted/false/true outcomes. The unchanged private probe now passes, preserving the original server-user-1 identity.

- Independent final frontend: **113 tests /10files passed**, log `/private/tmp/cycle4-uat108-independent-ui-final.log` (start21:40:37). This includes the five initial suites plus chatModePipeline.conversation-id, saveMessageOnError, ChatTldw.stream-metadata, ChatTldw.abort-signal and messageHandlers.regenerate.
- Unchanged private successful-regeneration probe: **1passed/48unselected**, log `/private/tmp/cycle4-uat108-successful-regeneration-probe-green.log`. Original failing log retained.
- Independent backend: **76passed/2files**, unchanged after frontend-only correction; no redundant rerun.
- Final owned manifest freeze **2026-09-16T04:41:26.702Z**: all **16 hashes match**, comprising8production/7tests/1officialtask. Hash audit `/private/tmp/cycle4-uat108-independent-hash-check.json`; author manifest `/private/tmp/cycle4-uat108-persistence-owned-manifest.json`. Two final forwarded-property whitespace alignments after tests were reviewed and have no behavior effect.
- Author final lint comparison covers12frontendpaths:0errors/35unchangedwarnings. Bandit0 and Ruff15unchanged remain the inspected static baseline. Full compiler and native fresh failure→Retry→reload acceptance remain parent-owned.

Final frontend command from apps/packages/ui:

```
./node_modules/.bin/vitest run --config vitest.config.ts src/db/dexie/__tests__/helpers.user-acknowledgement.test.ts src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/services/tldw/__tests__/TldwChat.abort.test.ts src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts src/models/__tests__/ChatTldw.stream-metadata.test.ts src/models/__tests__/ChatTldw.abort-signal.test.ts src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts
```
