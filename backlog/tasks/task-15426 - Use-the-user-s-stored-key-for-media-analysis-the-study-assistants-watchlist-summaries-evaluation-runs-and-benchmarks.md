---
id: TASK-15426
title: Use the user's stored key for media analysis, the study assistants, watchlist
  summaries, evaluation runs and benchmarks
status: To Do
labels:
- quotas
- backend
- byok
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-13956. These features call their provider with the server's key and never with a stored user key: media analysis (the /media endpoints and media ingest jobs, every call through Summarization_General_Lib._summarize_via_adapter with no provider_credentials), the flashcards and quizzes study assistants (study_assistant.generate_study_assistant_reply, perform_chat_api_call_async with no key), watchlist output summaries (outputs_service._llm_reply), evaluation runs (eval_runner, resolve_provider_api_key_from_config) and benchmarks (a raw api_key from the request or the server's). So their calls are the operator's spend and count against the per-user LLM limits, and their boundaries refuse an over-limit user even when the user stored a key for the provider. Product question for the owner: should these features use the user's stored key (moving their cost to that key)? If yes, resolve it through ProviderCredentialRuntime and pass the handle; then each boundary names its provider to enforce_llm_quota in one line, and the ingest job's start check (_admit_or_refuse_analysis) admits an over-limit user whose analysis provider runs on their own key without marking the job admitted. Cost differs by family (TASK-13956 plan review): the study assistants are the cheap first slice (build a runtime in the route, resolve, pass the handle; perform_chat_api_call_async already binds it); benchmarks and watchlist summaries are small to medium (analyze takes provider_credentials); evaluation runs are medium (a runtime that outlives the request); media analysis and the ingest job are about a day with tests (a handle through about ten ingestion libraries, or a context-scoped handle read in _summarize_via_adapter, plus a runtime per job).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether each listed feature uses the user's stored key
- [ ] #2 Each feature the owner approves resolves the user's key through ProviderCredentialRuntime and passes the handle on every LLM call, with a test
- [ ] #3 Its boundary names its provider to enforce_llm_quota, and the media ingest job's start check admits an over-limit user whose analysis provider runs on their own key without marking the job admitted, with a test
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
