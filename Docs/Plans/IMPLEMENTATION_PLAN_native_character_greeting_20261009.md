# Native Character greeting persistence

Parent: TASK-13260.278.18.83.45; association recorded before edits.
ADR required: no. [ADR-049](../ADR/049-chat-history-selection-ownership.md) governs canonical owner-bound history; use existing server greeting seeding without changing architecture or backend authority.

## Stage 1: Reproduce
**Goal**: Prove mounted native callers omit greeting choice.
**Success Criteria**: Causal red in mounted history-selection and delegated transport controls.
**Tests**: Fresh default/alternate/disabled, canonical receipt, existing chat and uncertain ACK.
**Status**: Complete

## Stage 2: Repair and qualify
**Goal**: Seed exact chosen server greeting only for fresh owned chats before capture.
**Success Criteria**: Preserve settings/owner/lease/cancel boundaries; no invented greeting, duplicate seed or automatic creation retry.
**Tests**: Configured UI/Web suites, scoped lint and configured types; TS-only Bandit N/A.
**Status**: In Progress

## Stage 3: Independent/native acceptance
**Goal**: Peer review and root-owned browser validation.
**Success Criteria**: Reviewed frozen source and actual canonical greeting.
**Tests**: Root owns browser/build/runtime.
**Status**: In Progress

Scope: native helper, both callers, createChat-only client/domain changes; existing history-selection and request-scope tests. C08 hunks unchanged. No logs/proof/runtime/Git. Max3 failed attempts; Node20/default workers/600s outer cap.

## Qualification and limits

- Causal initial Node20 UI: 8 failed/184 deliberately filtered, 2.23s, natural exit1. Both mounted callers omitted seed intent and actual client omitted the query. No setup failure.
- First8 then expanded198/3 passed. Existing assertions preserved; prior fresh-creation fence fixtures explicitly disable greetings to retain original no-live-card assertions.
- Lifetime test first used nested act and was corrected to a real deferred read/load with separate rerender. Corrected RED1/positive1/94filtered: late draft choice blocked, accepted creation hydration wrongly cancelled. Fix retires draft-preference comparison only after valid create ACK; owner/model/tools/abort checks remain.
- Affected UI202/3 passed2.71s; Web202/3 passed2.40s, natural0 and zero skips. After type-only fixture cleanup/formatting, final18 passed UI2.34s/Web1.85s,184 deliberately filtered each. These follow-ups are not a second combined202 run. Retained unrelated cases emitted old i18n/act warnings; final18 stderr empty.
- Controls: default/alternate/disabled/empty, raw alternate index despite duplicate/empty entries, stale/unmappable choice, missing canonical greeting, unknown create ACK, late owner/preference, accepted hydration, base/delegated transport. Existing chats never seed. No client fabricated assistant append or automatic creation retry.
- Root-config ESLint eight paths:0errors/974 inheritedwarnings (helper0, callers23/24, client530, domain235, tests105/52/5). In-memory inverse baseline counts equal. Client inverse SHA matches frozen C08 d37a4eba5ea1576f698a8d24c55c1ef1ced1a9b0a9b5a057078b3c5ec3ee002d; outside createChat unchanged. Root pages-directory advisory remains.
- TS-only Bandit N/A, not security proof. Root owns final combined configured UI/Web types after concurrent freezes and independent/native acceptance; no author types or native claim. Stage2/3 In Progress.
- Node20.19.5/default workers/600s per command. No receipts/logs/manifests/runtime/browser/auth/Git actions. One pre-unit memory-capture read exceeded output size and one plan-update script had quoting syntax failure before any writes; neither affected source/test runs.

Commands: from apps/packages/ui run ../../tldw-frontend/node_modules/vitest/vitest.mjs run src/hooks/chat/__tests__/useChatActions.history-selection.test.tsx src/hooks/__tests__/useMessage.history-selection.test.tsx src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts --reporter=dot. Web uses node_modules/vitest/vitest.mjs from apps/tldw-frontend and ../packages/ui/src/... paths. Final focused follows with -t "native fresh greeting contract|preserves native greeting seed query". Scoped ESLint API uses repository cwd, overrideConfigFile apps/tldw-frontend/eslint.config.mjs, lintText and real filePath; no ignored-file qualification.
