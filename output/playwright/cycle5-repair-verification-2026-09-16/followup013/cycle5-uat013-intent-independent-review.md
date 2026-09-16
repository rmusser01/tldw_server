# Independent final rereview: UAT013 accepted-source intent

**Approved within the bounded implementation scope. Prior P2 is closed; no new actionable findings.**

Reviewed against HEAD7c7f4093df and the prior candidate. All6 hashes match the author's final freeze2026-09-16T15:40:56.598Z in `/private/tmp/cycle5-repair-chat-013-manifest.json`; independent audit `/private/tmp/cycle5-uat013-intent-independent-hashes.json`.

## Closure and design assessment

The original same-value42-vs7 private fixture and Vite config were rerun **unchanged**. It now sends `include_media_ids:[42]` and includes Rowan evidence in the actual RAG model input. Result1PASS; `/private/tmp/cycle5-uat013-independent-private-fixed.log`. The previous failure is preserved in `/private/tmp/cycle5-uat013-independent-private-probe.log`.

The correction records accepted operation order instead of inferring intent from value changes. One sourceSelectionRevision lives in the existing playground-session store, outside PlaygroundSessionData and its explicit persisted partialize. clearSession leaves this revision monotonic while the existing restoreRevision cancels obsolete session work. No persisted schema/migration, extra store or event mechanism was added.

Form increments only after normalization/owner acceptance and valid numeric RAG ID, or for accepted ordinary full-content handoff. Invalid/foreign and stale owner storage results do not record intent. Existing legacy unowned acceptance is unchanged. The hook compares a mount revision only for its initial same persisted scope/history/server target; later explicit restores use invocation-time revision. It skips only mode/source replay after a newer accepted handoff, preserving transcript/metadata restoration and existing ownership cancellation. Same-value, handoff-before-restore, in-flight restore and source ABA are covered.

## Independent verification

**165 tests / 9 suites PASS**, no skips, exit0. `/private/tmp/cycle5-uat013-intent-independent-tests.log`.

From apps/packages/ui, installed local binary:

```sh
node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/store/__tests__/playground-session-store.test.ts src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts src/services/__tests__/media-chat-handoff.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx --maxWorkers=2 > /private/tmp/cycle5-uat013-intent-independent-tests.log 2>&1
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat013-independent-private.config.mts src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t 'independent:' --maxWorkers=1 > /private/tmp/cycle5-uat013-independent-private-fixed.log 2>&1
```

The second command intentionally selects only the unchanged private probe:1PASS/29 unselected. Fixture `/private/tmp/cycle5-uat013-independent-private-test.txt`; config injects it into the actual-boundary test module without repository writes. Permanent coverage independently verifies stored old history plus new source transport, same-value RAG and normal mode, source ABA, empty/error fail-closed generation, foreign/delayed ownership, late request invalidation, later restores, different persisted targets, ephemeral revision persistence/reset behavior and synchronous localStorage hydration.

## Limits

No source/test/task edits, browser/runtime/API/inference, staging or commits. Real Form/action/RAG serializer/session logic are exercised with controlled network/model/DB and unrelated UI fixtures; this is not native acceptance. Hydration evidence covers the currently configured synchronous localStorage adapter/migration only. Root owns whole compiler/native checks. No independent full lint rerun; author reports0errors/91unchanged baseline warnings. No Python Bandit scope. Existing Node/test warnings retained. Prior review/probe evidence remains unchanged.
