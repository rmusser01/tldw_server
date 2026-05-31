---
id: TASK-188
title: Implement client-managed conversation context workflow
status: Done
assignee:
  - Codex
created_date: '2026-05-09 20:05'
updated_date: '2026-05-09 21:38'
labels:
  - implementation
  - frontend
  - backend
  - character-chat
  - worldbooks
  - chat-dictionaries
  - conversation-context
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md
  - Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md
  - Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved client-managed Conversation Context implementation plan. The client composes effective context from server primitives; the server only provides composable processing, settings, prompt pieces, and validation primitives.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server primitive tests cover explicit worldbook processing without character context, ordered dictionary processing, conversationContext settings preservation, and invalid asset ID domain errors
- [x] #2 Client context composer assembles preview and send payload from server primitives without a monolithic backend context-preview endpoint
- [x] #3 Composer popover replaces or evolves the existing chat-composer character picker and keeps character selection as one slot
- [x] #4 Worldbooks and dictionaries remain conversation-scoped and usable without a selected character
- [x] #5 Targeted pytest, Vitest, Playwright/browser checks, Bandit touched-scope scan, and git diff checks are recorded or concrete skips documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Execute Task 1 from Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md: server primitive audit and hardening. 2. Follow TDD: write failing tests first, then minimal primitive changes. 3. Continue through client composer, hook/send integration, composer popover, asset selection, browser validation, and docs closeout in separate commits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 backend primitive hardening complete. Added integration coverage for explicit worldbook processing without a character, ordered dictionary_ids processing, conversationContext settings preservation, and invalid worldbook/dictionary ID domain errors. Implemented dictionary_ids on /api/v1/chat/dictionaries/process, explicit ID validation, ordered processing, and worldbook missing-ID validation. Verification: pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py -v (4 passed); pytest tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py -q (54 passed); bandit touched backend paths -> 0 findings; git diff --check -> clean.

Task 2 client composer core complete. Added typed conversation-context models, settings normalization for nested conversationContext plus legacy chat_dictionary_ids fallback/mirror, and a pure composer that calls dictionary processing before worldbook processing and returns shared previewSections/providerMessages from one composition object. Verification: bunx vitest run ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts (8 passed); bunx tsc --noEmit -p ../packages/ui/tsconfig.json --pretty false (passed); git diff --check (clean).

Task 3 hook/send integration slice complete. Added useConversationContextComposition to compose one shared preview/send object from client-managed settings, debounce background preview composition, compose immediately at send time, and preserve the authored user message while sending transformed model-only text/history overrides. Wired the Sidepanel chat form to read conversationContext settings and pass requestOverrides into the normal/persona/image-backed normal chat send paths; local chat settings now preserve conversationContext and legacy chat_dictionary_ids keys. Verification: bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts (12 passed); bunx tsc --noEmit -p ../packages/ui/tsconfig.json --pretty false (passed); git diff --check (clean). Known remaining scope: visible composer popover/asset selectors and character complete-v2 context pass-through remain for subsequent tasks; this slice intentionally keeps server composition primitive-only.

Task 4 composer popover slice complete. Replaced the direct ControlRow CharacterSelect call with ConversationContextPopover while keeping CharacterSelect as the character slot inside the broader context control. The popover shows readiness, character state, worldbook matched/configured counts, dictionary active/configured counts, source labels, and preview sections from the client composition without adding server-side effective-context assembly. Verification: bunx vitest run ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx --config vitest.config.ts (6 passed); bunx tsc --noEmit -p ../packages/ui/tsconfig.json --pretty false (passed); git diff --check (clean). Remaining scope: actual worldbook/dictionary selection controls and persistence from the popover are Task 5.

Task 5 minimal asset selection slice complete. ConversationContextPopover now loads worldbook and dictionary lists from existing tldwClient primitives, renders checkbox selectors, and saves only conversation-scoped worldBookIds/dictionaryIds through useConversationContextComposition.saveSelection. Blank non-character compositions can attach worldbooks and dictionaries; pre-persistence chats show disabled asset edits rather than forcing character attachment. The hook saveSelection path writes nested conversationContext plus the top-level chat_dictionary_ids compatibility mirror. Verification: bunx vitest run ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx ../packages/ui/src/hooks/chat/__tests__/useConversationContextComposition.test.tsx ../packages/ui/src/services/conversation-context/__tests__/conversationContextComposer.test.ts ../packages/ui/src/services/conversation-context/__tests__/conversationContextSettings.test.ts --config vitest.config.ts (22 passed); bunx tsc --noEmit -p ../packages/ui/tsconfig.json --pretty false (passed); git diff --check (clean). Bandit not run for this slice because only TypeScript/frontend files changed.

Task 6 browser validation and baseline cleanup complete. Added Playwright smoke coverage for the conversation context popover showing character, worldbook, and dictionary slots in blank sidepanel chat and verifying mobile trigger usability. Browser diagnosis found the popover trigger had nested Tooltip/Popover interaction; replaced the nested Tooltip with button title/aria text so click opens the context popover. Broader browser smoke initially exposed a pre-existing composer variant keyboard baseline failure caused by an Ant modal focus trap in the settings smoke plus missing explicit Space/Enter handling on radio cards; fixed the settings smoke setup and added product keyboard handling with unit coverage. Verification recorded: pytest conversation context primitives 4 passed; pytest chat dictionary endpoints 54 passed; Vitest focused suite 30 passed; TypeScript noEmit passed; Bandit touched backend paths 0 findings; git diff --check clean; Playwright smoke slice conversation-context-popover + composer-picker-keyboard + composer-mobile-viewport + playground-nextgen-composer 15 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the client-managed conversation context workflow across server primitives, client composition, chat send integration, and the composer popover. Worldbooks and dictionaries are conversation-scoped primitives that can be attached to blank/non-character conversations as well as character chats; the client rebuilds effective context from composable server pieces. Added browser smoke coverage for the conversation context popover and cleaned up the composer picker keyboard baseline discovered during validation. Known follow-up: character complete-v2 pass-through remains documented as out of this slice unless that send path needs the same request override integration.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
