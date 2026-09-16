# Cycle 4 Task 5 — Chat retry, saved character route, and missing final answer

Ready for independent review. Tasks TASK-13260.49 (UAT108), .53 (UAT113), .57 (UAT117) remain In Progress pending parent-owned combined checks/native acceptance.

Code/test freeze: 2026-09-16T03:28:25.266Z; baseline HEAD eb8782e5fd8bcf8d931c6a690b962b20c7483365.

## Behavior and causes

- **108:** the visible Retry action called ordinary submit, appending the failed question again; model-history serialization also retained decoded assistant display-error envelopes. Retry now invokes existing failed-turn regeneration. Only recognized assistant error envelopes are omitted by generateHistory. User quotations, malformed markers, actual partial prose and successful answers remain model context. The actual Form button and actual saved-normal pipeline/regeneration have separate behavioral regressions.
- **113:** the accepted character creation command remained in the query/hash after a saved conversation was established, so reload executed the creation reset again. The real route owner replaces it with the matching loaded canonical saved ID only after the scoped persisted session confirms that same active ID/character. The existing creation command still deliberately resets same/different character entries. Other URL data is retained. No asynchronous route-promotion callback, new authority watcher or session persistence redesign was added.
- **117:** nonempty reasoning was treated as a final answer. The existing reasoning parser now identifies settled reasoning with no answer text, including closed, unclosed, structured-stream and empty-tag cases. Normal/tracked completion enters the existing recoverable error path; actual Message renders recovery for persisted reasoning-only rows while leaving reasoning readable and excluding active streaming, real final text, tools and images. Confirmed user/assistant acknowledgements survive local error persistence; no saved ID is synthesized.
- The existing tracked turn already acquires a verified request lease, but its recovery path did not receive it. That captured snapshot now guards recovery before/after awaited persistence and scopes local error storage. Both ordinary addChatMessage fallback and emote completions/persist recovery carry the existing requestScope/signal. persistCharacterCompletion has backward-compatible optional scope forwarding in both the base client and actual delegated domain method, plus one exact POST policy allowance. Existing cache invalidation/degraded handling stays intact.

## Permanent RED evidence

| Boundary | Retained log | Result |
| --- | --- | --- |
| Recognized assistant envelope + actual Retry action | /private/tmp/cycle4-task5-retry-final-red.log | 2 failed / 5 passed |
| Normal/tracked reasoning-only, restored Message, error-save canonical ID | /private/tmp/cycle4-task5-reasoning-red.log | 9 failed / 39 passed |
| Real session/loader entry -> save -> reload query/hash | /private/tmp/cycle4-task5-route-red.log | 2 failed / 37 passed |
| Lease invalidated while tracked recovery awaited | /private/tmp/cycle4-task5-recovery-authority-red.log | 1 failed / 18 passed |
| Empty closed reasoning tag | /private/tmp/cycle4-task5-empty-reasoning-red.log | 1 failed / 8 passed |
| Scoped client forwarding + exact policy positive controls | /private/tmp/cycle4-task5-recovery-transport-red.log | 3 failed / 69 passed |

The Retry fixture first encountered stale unrelated post-merge Prompt Assist and Home milestone mocks; it now follows neighboring fixtures in isolating those unrelated surfaces. Its error banner and Retry callback are real. The transport fixture was corrected to retain existing scope_type=global query behavior before the authoritative RED above. The route fixture supplies actual global scope metadata. No failure coverage was disabled.

## Verification

Counts overlap; do not add them.

- Initial recovery GREEN: /private/tmp/cycle4-task5-recovery-green.log — 55 tests / 6 files.
- Route final controls: /private/tmp/cycle4-task5-route-final.log — 41 / 1, including query/hash save/reload, deliberate new same/different character, and delayed-scope hydration after boundary clear.
- Completion controls: /private/tmp/cycle4-task5-completion-controls.log — 76 / 3, including real ChatTldw -> normal pipeline -> local error helper acknowledgements, tracked recovery, parser edge cases.
- Last broad run: /private/tmp/cycle4-task5-final-tests.log — 314 passed / 1 failed across 22 files. The sole failure proved runtime delegation still omitted the captured scope after the base-method-only change. The bounded delegated method correction followed.
- Final unchanged transport assertions after that correction: /private/tmp/cycle4-task5-recovery-transport-green.log — **72 passed / 2 files**. Per parent instruction, no duplicate broad run; root will run the retained combined command after this final freeze.
- Exact combined command is retained at /private/tmp/cycle4-task5-verification-command.sh. It runs the actual coordinator, Form, normal/tracked handlers, error helpers, Message, parser, client and policy controls plus adjacent cancellation/provider/image/dynamic-UI/overlay/service-prompt cases.
- Scoped root ESLint script: node /private/tmp/cycle4-task5-static-check.mjs. It invokes apps/tldw-frontend/node_modules/.bin/eslint with explicit --config apps/tldw-frontend/eslint.config.mjs for all 24 paths. **0 errors, 979 existing warnings, 0 added, 1 removed** versus HEAD. Only embedded diagnostic line numbers are normalized for comparison. Raw baseline/current JSON and comparison retained under /private/tmp/cycle4-task5-lint-*.json; /private/tmp/cycle4-task5-static.log summarizes.
- git diff --check on owned paths: clean.
- No Python changes; Bandit is not applicable to this TypeScript-only scope.
- No full TypeScript run here by instruction; root owns the combined baseline comparison.

## Production files (12)

- apps/packages/ui/src/utils/generate-history.ts
- apps/packages/ui/src/libs/reasoning.ts
- apps/packages/ui/src/types/chat-modes.ts
- apps/packages/ui/src/hooks/chat-helper/index.ts
- apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts
- apps/packages/ui/src/hooks/chat/useChatActions.ts
- apps/packages/ui/src/components/Common/Playground/Message.tsx
- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/service-prompt-scope-error.ts
- apps/packages/ui/src/services/tldw/domains/chat-rag.ts

## Tests (12)

- apps/packages/ui/src/utils/__tests__/generate-history.image-generation.test.ts
- apps/packages/ui/src/libs/__tests__/reasoning-final-answer.test.ts
- apps/packages/ui/src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts
- apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx
- apps/packages/ui/src/components/Common/Playground/__tests__/Message.error-recovery.integration.test.tsx
- apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.pinned-fallback.test.tsx
- apps/packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts
- apps/packages/ui/src/services/tldw/__tests__/service-prompt-scope-error.test.ts

## Freeze and limits

- /private/tmp/cycle4-task5-production-freeze.json: 12 production hashes.
- /private/tmp/cycle4-task5-code-freeze.json: 24 production/test hashes.
- /private/tmp/cycle4-task5-owned-manifest.json: those 24 plus the three officially updated Backlog records; every source/test hash rechecked while writing this report.
- /private/tmp/cycle4-task5-owned.diff: reviewable tracked diff; the new parser test is listed/hashed separately in the manifest.
- Route regression uses real WebUI storage/session/loader with controlled router and acknowledged transport target. Delayed owner controls explicitly apply the existing boundary clear then settle its old scope read; they are not native login or full Auth-provider coverage. Existing real scope A->B->A stream/promotion tests remain in the combined command.
- Error persistence tests use existing local storage/transaction adapters; no new real browser/Dexie/native acceptance claim. Already-saved reasoning-only server rows are retained with canonical provenance; this patch does not rewrite backend completion status or delete content.
- No runtime/browser/inference, backend changes, commits, staging or shared design/tracker edits. Native Task5 replay and fresh full UAT remain pending.
