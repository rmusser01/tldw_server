---
id: TASK-13610
title: Record LLM usage for the Messages API native path
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Anthropic-compatible Messages API (api/v1/endpoints/messages.py) sends requests for the providers in MESSAGES_NATIVE_PROVIDERS (anthropic, llama.cpp) straight to the provider over HTTP (_native_post_json, _prepare_native_stream_iterator). Those calls never reach a ChatProvider adapter, so the provider base class (ADR-067) records nothing for them and a check at the call (LLM usage accounting PR 2b) cannot see them. PR 2a gates the endpoint at the request boundary. Parent: TASK-13437.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A native Messages API call writes one llm_usage_log row for the user, streaming and not
- [ ] #2 The row's tokens come from the provider's usage block when it sends one
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
