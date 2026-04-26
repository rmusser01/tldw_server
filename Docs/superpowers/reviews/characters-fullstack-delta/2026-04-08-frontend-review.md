# Characters Frontend Delta Review

## Scope

- frontend Character workspace
- frontend Character chat integration
- frontend Character API client and shared utilities

## Baseline Artifacts Checked

- Docs/superpowers/specs/2026-03-27-chat-character-menu-and-system-prompt-editor-design.md
- Docs/superpowers/plans/2026-03-27-chat-character-menu-and-system-prompt-editor-implementation-plan.md
- Docs/Product/WebUI/PRD-Characters Playground UX Improvements.md

## Code Paths Reviewed

- apps/tldw-frontend/pages/characters.tsx and pages/settings/characters.tsx: route entry points and dynamic loading
- apps/packages/ui/src/routes/option-characters.tsx: route-level Character workspace composition
- apps/packages/ui/src/components/Option/Characters/Manager.tsx, CharactersWorkspace.tsx, CharacterEditorForm.tsx, CharacterDialogs.tsx, CharacterListContent.tsx, and CharacterGalleryCard.tsx: workspace state, editing, list rendering, dialogs, and gallery behavior
- apps/packages/ui/src/components/Option/Characters/import-state-model.ts, search-utils.ts, and tag-manager-utils.ts: Character workspace state helpers and pure logic
- apps/packages/ui/src/components/Option/Characters/hooks/useCharacterData.tsx, useCharacterCrud.tsx, useCharacterImportQueue.tsx, useCharacterVersionHistory.tsx, and useCharacterQuickChat.tsx: async data flow and workspace actions
- apps/packages/ui/src/components/Common/CharacterSelect.tsx and components/Sidepanel/Chat/CharacterSelect.tsx: shared Character selection behavior
- apps/packages/ui/src/hooks/useSelectedCharacter.ts, useCharacterGreeting.ts, hooks/chat/useCharacterChatMode.ts, hooks/chat/useServerChatLoader.ts, and hooks/chat/useSelectServerChat.ts: Character selection and chat-state synchronization
- apps/packages/ui/src/services/tldw/domains/characters.ts: frontend API client normalization, fallbacks, and cache behavior
- apps/packages/ui/src/utils/characters-route.ts, character-greetings.ts, character-mood.ts, default-character-preference.ts, selected-character-storage.ts, and character-export.ts: routing, greeting resolution, persistence helpers, and export shaping

## Tests Reviewed

- apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__/Manager.crossFeatureStage1.test.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__/CharacterGalleryCard.test.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__/import-state-model.test.ts
- apps/packages/ui/src/components/Option/Characters/__tests__/search-utils.test.ts
- apps/packages/ui/src/components/Option/Characters/__tests__/tag-manager-utils.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.characters-list-all.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.characters-delete.test.ts
- apps/packages/ui/src/hooks/__tests__/useCharacterGreeting.test.tsx
- apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts
- apps/packages/ui/src/utils/__tests__/character-greetings.test.ts
- apps/packages/ui/src/utils/__tests__/character-mood.test.ts
- apps/packages/ui/src/utils/__tests__/default-character-preference.test.ts
- apps/packages/ui/src/utils/__tests__/characters-route.test.ts
- apps/tldw-frontend/e2e/workflows/tier-2-features/characters.spec.ts
- apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts

## Validation Commands

- `cd apps/packages/ui && bun run test -- src/components/Option/Characters/__tests__/Manager.first-use.test.tsx src/components/Option/Characters/__tests__/Manager.crossFeatureStage1.test.tsx src/components/Option/Characters/__tests__/CharacterGalleryCard.test.tsx src/components/Option/Characters/__tests__/import-state-model.test.ts src/components/Option/Characters/__tests__/search-utils.test.ts src/services/__tests__/tldw-api-client.characters-list-all.test.ts src/services/__tests__/tldw-api-client.characters-delete.test.ts src/hooks/__tests__/useCharacterGreeting.test.tsx src/hooks/__tests__/useServerChatLoader.test.ts src/utils/__tests__/character-greetings.test.ts src/utils/__tests__/character-mood.test.ts src/utils/__tests__/default-character-preference.test.ts src/utils/__tests__/characters-route.test.ts --maxWorkers=1` — environment-limited
- No targeted UI tests executed. This checkout lacks a usable local frontend workspace install for the prescribed `apps/packages/ui` command path, so `bun run test` resolved to `vitest run` but could not execute it cleanly in this workspace and exited with `/bin/bash: vitest: command not found`.
- `cd apps/tldw-frontend && bun run e2e:pw -- e2e/workflows/tier-2-features/characters.spec.ts e2e/workflows/journeys/character-chat.spec.ts --reporter=line` — environment-limited
- No Character browser assertions executed. This workspace lacks a usable local frontend tool install for the prescribed `apps/tldw-frontend` command path, so `bun run e2e:pw` fell through to the non-repo binary `/Users/appledev/Documents/GitHub/tldw_server/3.13/bin/playwright`, which reported `unknown command 'test'`.

## Findings

### Probable risk: Character workspace handoffs store a truncated character selection payload

- Severity: Medium
- Type: correctness
- Novelty: net-new
- Baseline artifacts checked: Docs/superpowers/specs/2026-03-27-chat-character-menu-and-system-prompt-editor-design.md; Docs/superpowers/plans/2026-03-27-chat-character-menu-and-system-prompt-editor-implementation-plan.md; Docs/Product/WebUI/PRD-Characters Playground UX Improvements.md
- Why it matters: Before the selected character is replaced with a hydrated server record, cross-surface consumers only see the reduced handoff payload. In that pre-hydration window, alternate greetings and `extensions`-backed mood portraits are unavailable even though downstream chat UI reads those fields from the selected character state.
- Current evidence: `apps/packages/ui/src/components/Option/Characters/utils.ts:764-782` builds the workspace handoff payload with only `id`, `name`, `system_prompt`, `greeting`, and `avatar_url`; `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx:483-497` uses that payload for chat entry and `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterQuickChat.tsx:37-54` duplicates it for quick-chat promotion. Downstream, `apps/packages/ui/src/hooks/useCharacterGreeting.ts:331-388` reads greeting variants from `selectedCharacter` first and only later fetches and merges the full record, while `apps/packages/ui/src/components/Common/CharacterSelect.tsx:306-308` and `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx:353-358` derive mood images from `selectedCharacter?.extensions`. The audited baseline artifacts define Character UX goals, but none describe or constrain this selection-payload handoff, so the novelty claim is anchored to current code paths rather than implied recent churn.
- Validation status: environment-limited, confidence unchanged

### Probable risk: Quick-chat promotion does not seed server-chat assistant metadata before navigation

- Severity: Medium
- Type: correctness
- Novelty: net-new
- Baseline artifacts checked: Docs/superpowers/specs/2026-03-27-chat-character-menu-and-system-prompt-editor-design.md; Docs/superpowers/plans/2026-03-27-chat-character-menu-and-system-prompt-editor-implementation-plan.md; Docs/Product/WebUI/PRD-Characters Playground UX Improvements.md
- Why it matters: Promoting a quick-chat thread into the main chat route appears to leave a transient pre-hydration window where the server chat id is set but the assistant metadata has not yet been seeded. Any UI that gates character-specific behavior on `serverChatCharacterId` or assistant kind can therefore briefly evaluate against incomplete server-chat metadata before eager metadata loading fills it in.
- Current evidence: `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterQuickChat.tsx:265-305` sets `serverChatId`, state, topic, cluster/source/externalRef, history, and messages, but does not set `serverChatCharacterId`, `serverChatAssistantKind`, or `serverChatAssistantId`. `apps/packages/ui/src/components/Sidepanel/Chat/body.tsx:237-244` disables character identity whenever `serverChatId` exists and `serverChatCharacterId` is null. The recovery path is also visible statically: `apps/packages/ui/src/hooks/useMessage.tsx:281-305` eagerly reloads chat metadata when `serverChatId` exists and `serverChatMetaLoaded` is false, and `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts:728-737` applies the metadata backfill setters after resolving `getChat(...)` data. The audited baseline artifacts do not contain comparable guidance for quick-chat promotion metadata seeding, so the novelty label is anchored to this currently reviewed frontend path.
- Validation status: environment-limited, confidence unchanged

### Confirmed finding: The character chat E2E journey does not validate character selection or prompt propagation

- Severity: Medium
- Type: test gap
- Novelty: net-new
- Baseline artifacts checked: Docs/superpowers/specs/2026-03-27-chat-character-menu-and-system-prompt-editor-design.md; Docs/superpowers/plans/2026-03-27-chat-character-menu-and-system-prompt-editor-implementation-plan.md; Docs/Product/WebUI/PRD-Characters Playground UX Improvements.md
- Why it matters: The named journey test currently appears to cover the main create-character to chat flow, but it does not prove that the created character is selected in chat or that its system prompt reaches the completion request. That leaves the primary cross-feature integration path effectively unguarded despite a reassuring spec name.
- Current evidence: `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts:1-79` states that it selects the character and verifies system prompt inclusion, yet the body only creates a character, navigates to `/chat`, sends a generic message, and checks request status. No step selects the created character and no assertion inspects the request payload. The adjacent E2E coverage in `apps/tldw-frontend/e2e/workflows/tier-2-features/characters.spec.ts:1-133` covers page load, create, and delete only, so there is no directly relevant browser-level test in the reviewed suite that closes this create-character-to-chat selection and prompt-propagation gap. The audited baseline artifacts describe Character/chat UX goals, but none provide a comparable frontend baseline for this specific journey assertion.
- Validation status: environment-limited, confidence unchanged

## Exit Note

Frontend validation complete. All surviving confirmed findings now have either targeted test support or a clearly stated reason why browser-level confirmation was not feasible in this environment.
