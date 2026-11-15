---
id: TASK-15502
title: View and edit thinking blocks in saved WebUI chats
status: Done
labels:
- webui
- chat
references:
- Docs/Design/2026-11-14-chat-thinking-blocks.md
- https://github.com/rmusser01/tldw_server/pull/3475
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep model-provided thinking available in native chat history and let users edit an individual expanded block through the existing message edit flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every thinking block can be expanded and read in Chat, including after a native chat reload.
- [x] #2 An individual block edit survives reload and preserves the answer and neighboring blocks byte for byte.
- [x] #3 Streaming, Cancel and save failure preserve the existing message ownership and edit behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Reuse tagged assistant text and existing edit and history ownership under Docs/ADR/049-chat-history-selection-ownership.md. No new schema or endpoint.
Design: Docs/Design/2026-11-14-chat-thinking-blocks.md. Reproduced both gaps. Verified 813 frontend tests, including actual disclosure edits and two consecutive mounted native sends. The capture and completion opt-ins preserve compatibility with older peers. Existing ADR-049 governs ownership; this change adds no storage schema.
Verified 813 UI tests and the WebUI typecheck. Backend sweep passed 86 checks, then the corrected native fixture passed all 3 focused tests, including exact edit and reload, provider context, and message-size and old-client boundaries. Chromium saved and reloaded the untouched answer and both blocks in a seeded local chat and verified dark-mode rendering at desktop and mobile widths. Bandit reported zero findings excluding the pytest assert rule.
Renumbered before opening the PR because open PR #3474 already reserved TASK-15501.
Merge preflight reproduced OpenAPI fingerprint drift and existing function-size ceilings. Refresh the generated API contract and relocate the new history logic within the existing persistence helpers.
Also verified the merge preflight after regenerating the OpenAPI fingerprint and types, fitting the persistence changes within existing function ceilings, and moving the size-limit test to the core persistence helper. Existing OpenAPI path union is unchanged. Passed 17 function/backlog checks, 82 contract checks, 45 streaming and settlement tests before the fixture relocation, and all 3 focused native and size-limit tests afterward. Bandit reported zero findings in the touched production files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Edit thinking to expandable reasoning blocks and fixed disclosure contrast. Native saves retain negotiated reasoning while preserving answer content, scoped edit acknowledgement, and answer-only provider context.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
