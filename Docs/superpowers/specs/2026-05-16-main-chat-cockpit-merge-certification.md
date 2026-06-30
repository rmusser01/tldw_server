# Main Chat Cockpit Merge Certification

Status: PR8 certification slice verified
Scope: Main WebUI `/chat` cockpit only. Browser extension sidepanel/sidebar surfaces are out of scope.
Backlog: TASK-403
Roadmap: `Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md`

## Merge Bar

The merge bar for the new `/chat` cockpit is a fully mature cockpit: first-class context, prompt, persona/character, model, session, MCP, mobile/focus, feedback, and recovery flows must be usable from the main chat page without regressing the existing chat composer and conversation flow.

## Real-Server Proof Command

Use the running local server and the configured `.env` API key. This command must not use mocked backend routes:

```bash
/bin/zsh -lc 'KEY=$(awk -F= '\''/^SINGLE_USER_API_KEY=/{print substr($0,index($0,"=")+1); exit}'\'' /Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/Config_Files/.env); KEY=${KEY%\"}; KEY=${KEY#\"}; export TLDW_E2E_API_KEY="$KEY"; export TLDW_E2E_SERVER_URL=http://127.0.0.1:8000; export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000; export TLDW_WEB_URL=http://localhost:8080; bun run e2e:pw -- e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line'
```

The Playwright spec includes a source guard that rejects `page.route(`, so the `/chat` proof cannot silently intercept or fulfill backend routes.

## Certification Checklist

| Status | Merge item | Evidence |
| --- | --- | --- |
| Verified | Real server and configured usable models drive `/chat` | `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` checks `/api/v1/health`, `/api/v1/llm/providers`, `/api/v1/llm/models/metadata`, configured provider filtering, and model selection from the running server. |
| Verified | Real prompt selection and clearing work from the left cockpit rail | `proves real prompt, model setting restore, and MCP state through the main cockpit rails` writes a disposable prompt into IndexedDB, selects it through the rail, verifies composition/context-source output, clears it, and restores focus to the prompt trigger. |
| Verified | Real character selection and clearing work from the runtime rail | `selects and clears a real disposable character through the runtime rail` creates a disposable character through `/api/v1/characters`, selects it from `/chat`, verifies runtime/composer/composition/context-source states, screenshots the selected state, then clears and deletes it. |
| Verified | Real persona selection and clearing work from the runtime rail | `selects and clears a real persona through the runtime rail` creates or falls back to a real active persona, selects it from `/chat`, verifies persona-specific runtime/composition/context-source states, screenshots the selected state, then clears and deletes it when disposable. |
| Verified | Assistant transition matrix is explicit | Unit coverage in `Playground.cockpit-controls.test.tsx` covers none, character, persona, legacy character mirror, selector tab routing, and clear behavior. Real-server proof covers none-to-character-to-none and none-to-persona-to-none transitions in `chat-cockpit.real-server.spec.ts`. |
| Verified | Provider:model settings persist and restore per selected scope | Real-server proof opens model settings from the runtime rail, verifies provider-qualified `settingsScope`, changes temperature, saves, verifies rail/composition preview update, reopens, verifies persisted value, restores the original/default value, and confirms focus return. |
| Verified | MCP populated and unavailable states are distinguished | `playground-cockpit-summaries.test.ts` covers unavailable, loading, degraded, empty, and available MCP summaries. `Playground.cockpit-controls.test.tsx` covers unavailable MCP without enabling tool choice. Real-server proof opens MCP settings and verifies either enabled/disabled/unavailable tool rows or clear unavailable/offline/empty states. |
| Verified | Conversation send still works from the cockpit | `uses the running server and keeps cockpit/focus controls working` and `proves model provider confidence through a real cockpit selection and conversation` send actual chat messages and verify the user turn plus assistant response or recoverable provider error. |
| Verified | Degraded health permits chat when unrelated to chat | `Playground.cockpit-controls.test.tsx` covers degraded warning-only readiness where streaming remains primary and the status strip says chat remains available; blocked chat-critical readiness remains unavailable. |
| Verified | Focus mode and rail collapse preserve composer/chat | `Playground.cockpit-shell.test.tsx`, `Playground.cockpit-a11y.test.tsx`, and real-server proof cover entering focus mode, showing cockpit panels again, independently hiding rails, and keeping chat/composer visible. |
| Verified | Mobile cockpit keeps context/runtime rails usable without losing drafts | `Playground.cockpit-maturity.test.tsx` covers controlled mobile panel behavior and draft preservation. Real-server proof captures mobile context, runtime, active draft, and focus screenshots while preserving the composer draft. |
| Verified | Keyboard and focus behavior are guarded | `Playground.cockpit-a11y.test.tsx` covers landmark labels, live status region, model catalog labels, rail toggle states, keyboard activation for rail/focus/mobile tab controls, and modal return-focus proof is covered in real-server prompt/model/MCP/mobile flows. |
| Verified | Visual/copy polish is protected | `Playground.cockpit-maturity.test.tsx`, `Playground.cockpit-controls.test.tsx`, `PlaygroundCompositionPreview.test.tsx`, and the real-server spec assert `Model route`, `Provider:model settings`, `MCP tools`, context source inventory, composition scope, and rail status language. |
| Verified | No sidepanel/sidebar/browser-extension scope drift | Touched files for PR8 are limited to main chat cockpit tests and docs: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`, this certification artifact, the PR8 plan, and TASK-403. |

## Screenshot Evidence

The real-server Playwright spec writes these QA artifacts into the Playwright test output directory:

- `chat-cockpit-desktop-initial.png`
- `chat-cockpit-desktop-focus.png`
- `chat-cockpit-desktop-conversation.png`
- `chat-cockpit-p0-rails-proof.png`
- `chat-cockpit-mobile-context.png`
- `chat-cockpit-mobile-runtime.png`
- `chat-cockpit-mobile-active-draft.png`
- `chat-cockpit-mobile-focus.png`
- `chat-cockpit-character-selected.png`
- `chat-cockpit-persona-selected.png`
- `chat-cockpit-model-provider-conversation.png`

## Final Verification Log

- Focused cockpit Vitest: `bun run test:run ...Playground.cockpit-a11y.test.tsx ...playground-cockpit-actions.test.ts` passed 10 files and 95 tests. The run printed existing unit-test stderr from an unconfigured tldw server fetch path, but exited 0.
- Real-server Playwright: the command above passed 9 tests in Chromium against `http://127.0.0.1:8000` and `http://localhost:8080`. It generated all screenshot artifacts listed above under `apps/tldw-frontend/test-results`.
- Targeted ESLint: `apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx` exited 0. It printed the existing Next pages-directory warning.
- Design-system verification: `bun run verify:design-system-state` exited 0 from `apps/packages/ui`; output reported 479 allowed baseline product-state exceptions.
- Whitespace check: `git diff --check` exited 0.
- Bandit: skipped because this PR8 slice only touches frontend TSX tests, Markdown docs, and Backlog metadata.
