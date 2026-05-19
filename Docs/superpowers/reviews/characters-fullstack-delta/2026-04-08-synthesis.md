# Characters Full-Stack Delta Synthesis

## Backend Findings

- Confirmed finding: manual character-memory extraction authorizes against `conversation.get("user_id")`, while Character conversations are stored and verified against `client_id`. That makes `/api/v1/characters/{character_id}/memories/extract` vulnerable to rejecting valid owned chats until the ownership field is aligned.
- Probable risk: streamed assistant persistence in [character_chat_sessions.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py) catches internal quota-check exceptions and continues persisting the assistant reply instead of failing closed. Adjacent targeted pytest slices did not reproduce the branch directly, so it remains a probable backend-only risk.

## Frontend Findings

- Probable risk: Character workspace handoffs store a truncated selected-character payload with only `id`, `name`, `system_prompt`, `greeting`, and `avatar_url`. Consumers such as greeting resolution and mood portrait rendering also read `selectedCharacter.extensions` and alternate greeting fields, so the pre-hydration window can degrade Character-specific UI state.
- Probable risk: quick-chat promotion sets `serverChatId` and navigates before seeding `serverChatCharacterId`, `serverChatAssistantKind`, or `serverChatAssistantId`. Frontend recovery depends on a later `getChat()` metadata load, so Character identity can be briefly disabled during the handoff.
- Confirmed finding: the named Character journey spec still does not select the created character in chat or assert that the Character system prompt reaches `/chat/completions`. The reviewed browser suite therefore leaves the primary Character-to-chat path unguarded even though the spec name implies coverage.
- Stage 5 frontend validation did not add runtime evidence. This checkout lacks a usable local frontend workspace install for the prescribed `apps/packages/ui` command path, so `bun run test` resolved to `vitest run` but could not execute it cleanly in this workspace. This workspace also lacks a usable local frontend tool install for the prescribed `apps/tldw-frontend` command path, so `bun run e2e:pw` fell through to `/Users/appledev/Documents/GitHub/tldw_server/3.13/bin/playwright`, which reported `unknown command 'test'`.

## Cross-Layer Contract Drift

- Assistant identity metadata is the highest-risk contract seam. The backend persists and returns `character_id`, `assistant_kind`, and `assistant_id` for chat sessions in [character_chat_sessions.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py), and the frontend loader in [useServerChatLoader.ts](/Users/appledev/Documents/GitHub/tldw_server/apps/packages/ui/src/hooks/chat/useServerChatLoader.ts) reconstructs Character identity from exactly those fields. Quick-chat promotion bypasses that contract temporarily by only setting `serverChatId`, so the UI falls into a local pre-hydration state that the backend never explicitly models.
- Selected-character payload shape is another drift seam. The backend `GET /api/v1/characters/{character_id}` contract returns the full character document, while frontend handoff helpers can seed a reduced local object first and rely on later hydration. Any UI path that reads the selected-character cache before hydration is therefore operating on a weaker contract than the backend actually serves.
- The Character list/query surface is intentionally defensive on the frontend. [characters.ts](/Users/appledev/Documents/GitHub/tldw_server/apps/packages/ui/src/services/tldw/domains/characters.ts) falls back between `/api/v1/characters`, `/api/v1/characters/`, and `/api/v1/characters/query` when it sees 404/405/422 or `path.character_id` parsing errors, while the backend currently exposes both `/` and `/query` plus `versions`, `revert`, `restore`, and `world-books` routes. That fallback reduces breakage but is also evidence that this contract has already been route-order-sensitive enough to require client-side recovery logic.
- Message speaker metadata looks aligned today but remains drift-sensitive. The backend persists `speaker_character_id` and `speaker_character_name` in streamed/persisted message metadata, and the frontend loader maps those fields back into chat messages for identity and mood rendering. The current review did not find a mismatch there, but there is also no targeted full-stack test covering it.

## Open Questions / Residual Risks

- It is still unverified whether the truncated selected-character handoff produces a visible user-facing regression in the main chat route before the full character fetch completes, or whether the hydration window is usually too short to notice.
- It is still unverified whether quick-chat promotion can race badly enough in production for Character identity gating in the sidepanel body to suppress Character-specific rendering during navigation.
- The manual memory-extraction ownership mismatch remains runtime-unconfirmed because the approved backend slices did not include `/api/v1/characters/{character_id}/memories/extract`.
- The streamed assistant persistence risk remains branch-unconfirmed because no targeted test injected a `count_messages_for_conversation()` or limiter failure into `/api/v1/chats/{chat_id}/completions/persist`.

## Coverage Gaps

- The prescribed frontend Vitest slice could not run in `apps/packages/ui` because this checkout lacks a usable local frontend workspace install for that command path, so none of the targeted Character unit/integration tests produced fresh Stage 5 evidence.
- The prescribed frontend Playwright slice could not run in `apps/tldw-frontend` because this workspace lacks a usable local frontend tool install for that command path and `bun run e2e:pw` fell through to `/Users/appledev/Documents/GitHub/tldw_server/3.13/bin/playwright`, so no browser-level evidence was added for the Character workspace or Character chat flows.
- The existing Character journey E2E still does not assert Character selection or system-prompt propagation into `/chat/completions`.
- No reviewed frontend or full-stack test proves that quick-chat promotion seeds assistant metadata before navigation.
- No reviewed frontend or full-stack test proves that selected-character handoffs preserve alternate greetings, `extensions`, or mood assets before hydration.
- No reviewed backend test covers the manual memory-extraction ownership branch or the streamed persist fail-open quota path.

## Improvement Opportunities

- Restore a usable local frontend workspace/tool install for the prescribed `bun run test` and `bun run e2e:pw` command paths in this checkout so the targeted Character validation commands execute against repo-local tooling.
- Add a narrow frontend regression test around Character handoff payload shape, either by preserving the richer selected-character object up front or by codifying that downstream consumers must tolerate pre-hydration truncation.
- Add a quick-chat promotion regression that asserts `serverChatCharacterId`, `serverChatAssistantKind`, and `serverChatAssistantId` are seeded before navigation when a server chat id is promoted into the main chat experience.
- Strengthen the Character journey E2E so it explicitly selects the created character and inspects the `/chat/completions` payload for Character prompt propagation.
- Add backend endpoint tests for `/api/v1/characters/{character_id}/memories/extract` ownership and `/api/v1/chats/{chat_id}/completions/persist` quota-check failures.

## Verification Summary

- Frontend targeted validation attempted:
- `cd apps/packages/ui && bun run test -- src/components/Option/Characters/__tests__/Manager.first-use.test.tsx src/components/Option/Characters/__tests__/Manager.crossFeatureStage1.test.tsx src/components/Option/Characters/__tests__/CharacterGalleryCard.test.tsx src/components/Option/Characters/__tests__/import-state-model.test.ts src/components/Option/Characters/__tests__/search-utils.test.ts src/services/__tests__/tldw-api-client.characters-list-all.test.ts src/services/__tests__/tldw-api-client.characters-delete.test.ts src/hooks/__tests__/useCharacterGreeting.test.tsx src/hooks/__tests__/useServerChatLoader.test.ts src/utils/__tests__/character-greetings.test.ts src/utils/__tests__/character-mood.test.ts src/utils/__tests__/default-character-preference.test.ts src/utils/__tests__/characters-route.test.ts --maxWorkers=1` failed before discovery because this checkout lacks a usable local frontend workspace install for that command path, so `bun run test` resolved to `vitest run` but could not execute it cleanly in this workspace.
- `cd apps/tldw-frontend && bun run e2e:pw -- e2e/workflows/tier-2-features/characters.spec.ts e2e/workflows/journeys/character-chat.spec.ts --reporter=line` failed at bootstrap because this workspace lacks a usable local frontend tool install for that command path, so `bun run e2e:pw` fell through to `/Users/appledev/Documents/GitHub/tldw_server/3.13/bin/playwright`, which reported `unknown command 'test'`.
- Contract comparison executed with `rg -n "/api/v1/characters|/characters/query|character_id|versions|revert|restore|world-books|speaker_character|assistant_kind" ...` across the prescribed backend and frontend files. That comparison confirmed that frontend assistant identity and route fallbacks are explicitly keyed to backend Character/chat response fields.
- Backend evidence was carried forward from the completed review in [2026-04-08-backend-review.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/superpowers/reviews/characters-fullstack-delta/2026-04-08-backend-review.md), including one confirmed backend correctness issue and one surviving probable backend risk.

## Prioritized Next Steps

1. Restore a usable local frontend workspace/tool install for the exact prescribed Vitest and Playwright Character command paths so those slices can run and either confirm or downgrade the two runtime-sensitive frontend risks.
2. Expand the Character journey E2E to assert actual Character selection and prompt propagation into `/chat/completions`; that closes the largest current full-stack coverage gap.
3. Add a focused regression for quick-chat promotion metadata seeding, because that is the clearest frontend/backend contract seam around `assistant_kind`, `assistant_id`, and `character_id`.
4. Add a focused regression for selected-character handoff fidelity or explicitly narrow the supported pre-hydration contract so greeting and mood consumers are not reading fields the handoff does not preserve.
5. Add backend endpoint tests for manual memory extraction ownership and streamed persistence quota failure so the surviving backend items move from static risk to directly exercised behavior.
