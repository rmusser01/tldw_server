---
id: TASK-13340
title: Remove 614 lines of dead chat orchestration that still takes commits
status: To Do
assignee: []
created_date: '2026-09-22 05:00'
labels:
  - dead-code
  - chat
dependencies: []
references:
  - 'tldw_Server_API/app/core/Chat/chat_orchestrator.py:869'
  - 'tldw_Server_API/app/core/Chat/chat_orchestrator.py:1379'
  - 'tldw_Server_API/app/core/Chat/Workflows.py:34'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_chat_sync_impl (869-1190, 322 lines) and achat (1379-1670, 292 lines) are near-twins implementing the entire multimodal turn-assembly policy twice: slash-command parsing, chat-dictionary pre/post-gen replacement, image-history handling, RAG prefix construction, custom-prompt placement, empty-message placeholder.

REACHABILITY: the only importer of chat_orchestrator.chat is core/Chat/Workflows.py:34, whose only importer is its own test, and which loads ./App_Function_Libraries/Workflows/Workflows.json - a path that does not exist in this repo. achat has no non-test importer outside chat_orchestrator.py. So this is 614 lines of dead production code that still takes commits. (chat_api_call in the same file IS live with 5 core callers, so this is a partial-file deletion.)

ONE REAL DIVERGENCE the duplication hides: achat wraps rag_text_prefix construction in try/except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS (which includes AttributeError and TypeError); _chat_sync_impl has the identical expression with NO try. Same input, two outcomes - the async path silently drops retrieval context and answers UNGROUNDED with no error and no log, while the sync path raises and returns a 500. Second divergence: _chat_sync_impl logs a masked API key under ALLOW_MASKED_KEY_LOG, achat has no such block.

Also blocks TASK-13336-adjacent work: the orchestrator CommandContext derives is_single_user_owner from the SERVER env rather than the caller, which is a latent auth footgun reachable only through this dead path.

Source: synthesis F39
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dead orchestration paths deleted, chat_api_call retained
- [ ] #2 Workflows.py resolved (deleted or repointed)
- [ ] #3 The divergent try/except is not carried into whatever remains
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
