# UAT031 independent review

## Assessment

No blocking findings in the bounded failed-Retry repair. The code is ready for the parent's integration and exact native acceptance checks. This review does not close TASK13260.7 or claim live Notes/Flashcards/reload acceptance.

Base HEAD: 2d5ad06c86cf279fe0fcc6445009813d97ab1c1b, branch codex/fresh-install-uat-fixes. Review was read-only on repository source, tests, task records, tracker, index and branch. Temporary review loaders and evidence live only under /private/tmp. No runtime, browser, inference, staging or commit actions.

Reviewed full changed regression setup/cases and remaining integration suite, the production diff and surrounding Retry/Regenerate, character routing, request ownership, failed persistence, error decoder and branch behavior.

## Findings and rationale

- The production change only skips the pre-submit server branch when the final assistant message decodes as the existing structured error format. It does not match provider names or free-form prose. The decoder requires the exact prefix plus a complete JSON payload with string summary and hint. Malformed markers and ordinary/interrupted prose continue through existing branch behavior.
- The new regression exercises real useChatActions, real regenerate/branch factories, real complete-v2 SSE error parsing, and the real failed-history helper. Remote endpoints and IndexedDB writes are substituted at their boundaries. Two failed retries retain the exact original greeting/user receipts and the same conversation, with no extra canonical rows. That establishes the ID prerequisite for original-message actions; the action UI/endpoints still require native acceptance.
- The successful, partial and malformed-marker controls retain server branching. The implementation leaves existing successful-Regenerate branch/message-ID behavior unchanged; that broader behavior is not certified here.
- Character identity resolution and service-prompt ownership checks remain untouched. Fresh existing tests cover stale global character metadata, explicit character switching, latest-store identity, and scope invalidation preventing old-owner recovery writes. The guard adds no new asynchronous boundary and avoids the previous failed-Retry branch creation/mutation path.
- An independent temporary fixture runs the two failed retries concurrently via Promise.all. The same canonical conversation, greeting/user IDs, three-row visible list and three complete-v2 request checks pass. This is a deterministic overlap probe, not exhaustive browser scheduling or account-switch stress coverage.

## Fresh independent validation

1. Current implementation: **53 tests / 5 files passed** (character integration 27; character contract 9; failed-history helper 9; regenerate handler 7; branch handler 1). Evidence: `/private/tmp/uat031-independent-green.log`.
2. Baseline RED without editing repository files: temporary Vite loader reads only useChatActions.ts from `git show HEAD:<path>` and supplies that source to the current regression suite. **1 failed / 3 passed / 23 skipped**. Expected `conversation-1`, received `conversation-3` after two failed retries. All three content-Regenerate controls passed. Evidence: `/private/tmp/uat031-independent-red.log`; loader `/private/tmp/uat031-independent-head.config.mts`.
3. Existing error decoder suite: **12 tests passed**. Evidence: `/private/tmp/uat031-independent-decoder.log`.
4. Concurrent failed-Retry probe described above: **1 selected test passed / 26 skipped**. Evidence: `/private/tmp/uat031-independent-concurrent.log`; loader `/private/tmp/uat031-independent-concurrent.config.mts`.
5. Fresh ESLint and independent HEAD/current comparison: **0 errors; exactly 39 unchanged warnings** (production 17, test 22). Rule/message/severity sequences are identical ignoring shifted source locations. Evidence: `/private/tmp/uat031-independent-eslint.json`, `/private/tmp/uat031-independent-eslint-comparison.json`. Existing root-run config prints the pages-directory notice.
6. `git diff --check` passes. Author's full compiler log contains 90 TypeScript errors; this review did not independently re-run or baseline-diff the full compiler, and does not claim a clean build/typecheck.

The test runs emit the existing Node localStorage experimental warning and expected persistence-error output from the fallback/degraded fixtures.

Bandit cannot parse these TS/TSX files. Its zero-result output is not a TypeScript security pass. Source review found no new ownership bypass, request-scope change, unsafe free-text classification or sensitive-data logging in this diff.

Focused current implementation command (cwd apps/tldw-frontend):

```sh
node_modules/.bin/vitest run ../packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx ../packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts ../packages/ui/src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.branch.test.ts --maxWorkers=2
```

## Exact reviewed SHA-256

```text
de948f35cca0b8624edceb2128c75c51760028f3bf617edf7587b5ce5c2ffe6a  apps/packages/ui/src/hooks/chat/useChatActions.ts
5ec1e78d238dfe97fc235a451ba332f00bd7fe94d2ea7857dc6b4ba8ed59909b  apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
```

Retain this report, the five independent log/JSON outputs, the ESLint comparison JSON and both temporary loaders before temporary storage is reclaimed. Native acceptance remains pending.
