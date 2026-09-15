# UAT093 / TASK13260.33 repair freeze

Frozen at 2026-09-15T21:23:45.274Z. Five source/test paths are listed with hashes in /private/tmp/uat093-frozen-manifest.json. No commits or browser/runtime actions by this agent. Parent owns native acceptance and integration.

## Confirmed cause and correction

Canonical settingsServerChatId entry runs Playground's settings-return effect. That effect redundantly called cancelPendingRestore, incrementing the real session restoreRevision. The real persistence hook subscribes to that store; inline local-loader callback dependencies change on rerender, rerunning the effect. The mount-captured settings target stays present after URL cleanup. The read-only actual Playground/real local-loader/session subscription probe reached revision55 and Maximum update depth without mounting Prompt Assist. The native Portal/Drawer stack was the detection site, not the producer.

Remove that one redundant cancellation. initializePlayground still cancels the explicit settings target before readiness; generic restoration precedence is unchanged. Permanent normal/StrictMode entry regressions retain the actual subscription and loader instead of the former stabilizing mocks.

A required held-read teardown probe also proved the existing local loader wrote state after unmount. The helper now captures mounted lifetime, load generation and original restoreRevision before reads. It checks them before state publications and around model, prompt and files awaits. Explicit principal-change invalidation retires pending work; obsolete errors/title writes are suppressed. It adds no auth/network prerequisite, so local offline reading remains available.

The helper returns an accepted boolean. LocalChatList calls its selection callback only after accepted completion; Playground's existing settings/fallback/sidepanel continuations stop on exact false (legacy mocked void return compatibility retained). A real current failed read still displays its existing error; it does not report successful selection.

## Scope

- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Common/ChatSidebar/LocalChatList.tsx
- apps/packages/ui/src/hooks/useLoadLocalConversation.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx
- apps/packages/ui/src/hooks/__tests__/useLoadLocalConversation.test.tsx

No storage schema, ownership migration, provider, Prompt Assist, Drawer, auth transport or global document changes. .33 task was updated with the bounded follow-up.

## Evidence

- Native original: /private/tmp/uat093-saved-robot-crash-snapshot.txt and .playwright-cli/console-2026-09-15T21-00-42-248Z.log.
- Read-only original loop RED: /private/tmp/uat093-session-loop-red.log (revision55, nested-update exception).
- Permanent normal/StrictMode RED: /private/tmp/uat093-permanent-red.log (2 failed through actual React error boundary).
- Held-local original RED: /private/tmp/uat093-delayed-local-result.log (late publication after unmount).
- Helper RED: /private/tmp/uat093-local-helper-red.log (7 failed, current positive passed).
- Actual LocalChatList continuation RED: /private/tmp/uat093-caller-contract-red.log (unmount/failure2 failed,11 passed).
- Prior Playground continuation behavior reproduced with read-only transform: /private/tmp/uat093-continuation-red.log (1 failed); config /private/tmp/uat093-continuation-red.config.ts.
- Both original probes rerun unchanged and pass: /private/tmp/uat093-session-loop-final-green.log and /private/tmp/uat093-delayed-local-final-green.log. The retained delayed-probe transform now emits a duplicate getSessionFiles-key warning because the permanent fixture gained that export; this is only the preserved test transform, not repository code.
- Final broader run: /private/tmp/uat093-broader-green.log — 81 tests / 6 suites pass.
- ESLint: /private/tmp/uat093-eslint.json — 5 paths,0 errors,0 added normalized warnings (line references inside pre-existing hook warnings normalized).
- TypeScript: /private/tmp/uat093-typecheck.log and /private/tmp/uat093-typecheck-comparison.json — exact90 merged baseline signatures, no additions/removals; baseline /private/tmp/uat032-merged-typecheck.log. Compiler exit2 is established baseline. Parent .34 production writes were held for this run.
- git diff --check clean. Bandit N/A: TypeScript-only changes.

Run from apps/packages/ui:

```sh
npx vitest run src/hooks/__tests__/useLoadLocalConversation.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx --maxWorkers=1
```

## Coverage and limits

Current complete helper restoration, actual LocalChatList selection while offline, actual Timeline/settings local-return invocation, pending generic restoration precedence, StrictMode/rerender stability, unmount at four awaits, replacement load, existing restore cancellation, principal A-to-B-to-A invalidation, same-owner config notification, obsolete failure suppression, and caller accepted/failure behavior are covered. Storage IO and unrelated visual/network subtrees are controlled; native IndexedDB/browser acceptance is parent-owned.

Auth boundary trace: TldwAuth emits explicit logout principal notification; the connection store invalidates its authority; WebUI _app's requiresLogin branch unmounts private route content. New/server Chat selection uses clearSession/cancelPendingRestore. Direct configuration-target changes do not themselves advance playground restoreRevision; this correction does not certify ownership of legacy local history or introduce a new authority model. Parent approved preserving that existing local/offline contract rather than adding /auth/me to local reads. Native .15 Cedar results reported by parent are separate from this UAT093 fix; UAT093 remains pending independent review and targeted native retest.
