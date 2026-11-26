---
id: TASK-15579
title: Restore chat sends in Document Workspace
status: In Progress
labels:
- webui
- chat
- document-workspace
references:
- Docs/superpowers/specs/2026-11-19-document-workspace-chat-design.md
- Docs/superpowers/plans/2026-11-19-document-workspace-chat.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document Workspace lacks the account-bound scene capture required by the shared chat submit path. Bind the workspace's private chat context to the selected document and verified account.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A question sent from Document Workspace reaches the shared chat dispatch with its verified document context.
- [x] #2 Changing documents, clearing chat, leaving the workspace, or changing accounts invalidates pending scene work.
- [ ] #3 Focused frontend regressions pass and the reviewed change is published to dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Reuses the private scene path and the owner-verification rule in ADR-049; there is no new persistence or public API contract.
The first regressions failed before the repair. The focused suite passed 96 checks, including the non-temporary Document Workspace send and reopened answer. Review addressed saved history, RAG, account scoping, interrupted turns, and separate desktop/tablet chat mounts. The owner waived the human-authored change-summary gate for this PR. Public Dev merge is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
