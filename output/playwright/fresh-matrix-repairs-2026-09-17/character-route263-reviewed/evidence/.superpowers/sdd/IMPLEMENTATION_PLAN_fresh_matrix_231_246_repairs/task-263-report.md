# TASK13260.205 / UAT263 report

## Result

Character Retry still creates the intended server child conversation. Once the child and its copied prefix are accepted, the handoff now sends the existing owned-route event with both the prior route owner and the child ID. Playground replaces the stale saved URL directly with the child ID before the active chat store changes.

The route guard still validates the captured URL, parent chat, local history and restore revision. Existing clear actions omit `nextChatId`, so their route behavior is unchanged.

## Regression coverage

- The Character action integration retains complete, partial and malformed-assistant branch behavior, parent linkage and branch completion destination. It now asserts one accepted route event identifying both parent and child.
- The Playground coordinator integration begins at a saved Character URL, accepts `retry-branch`, allows a stale route cycle to run, verifies the parent URL cannot return, then verifies branch URL, active store, saved session and normal reload all resolve to the child transcript.
- Existing full suites retain encoded-error retry in the parent, local diagnostic retry behavior, normal saved Chat behavior, cancellation, saved-route navigation and account-transition controls.

The causal RED check failed before the change because no branch route event was dispatched. The first event-only implementation exposed a blank-route ordering issue: the temporary new-Character command cleared the accepted child. The final narrow extension carries the accepted child ID in the already-validated replacement event and writes that target directly.

## Verification

- Character actions: 47 passed.
- Playground coordinator: 61 passed.
- Adjacent branch/regeneration/ordinary saved Chat: 108 passed.
- `git diff --check`: passed.
- Scoped ESLint: 0 errors; 77 warnings outside changed lines.
- TypeScript differential: retained baseline 90, current 90, zero added or removed normalized diagnostics.
- Bandit: 0 findings. It reported six parser errors because every touched source is TypeScript/TSX, which Bandit cannot parse.

Evidence is in `.tmp/uat-repairs-231-246/character-route263/`, including `owned.patch`, source/static JSON, and `verification-summary.json`.

## Source SHA-256

- `components/Option/Playground/Playground.tsx`: `ee1349b4059581d49667e725767db308fe40ecac0f87c32002f82fa6cf60c3e9`
- `components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx`: `6fa7aa29e70104aa241113beb3a01a84f664ee3675e719ae5287e7ec547069db`
- `hooks/chat/useChatActions.ts`: `0b0524c04945282aba4f9add789fbba1d5f735a235fb92221a6d9fc0ae85db1b`
- `hooks/chat/__tests__/useChatActions.character.integration.test.tsx`: `9ab6e6f727f3ce573edf25496a3421db597fe99e9f31089502556d29ecb91a98`
- `hooks/handlers/messageHandlers.ts`: `ec6ddb62f75bf7c12eceae7d8db1d2db7c1cb79e3d0941662cb2a2d5d012d3d`
- `utils/character-chat-mode-intent.ts`: `bc4983feb9e494c000832a2f63bf84930ee5c99718eb70f9dbf2979f17b4dbbe`
