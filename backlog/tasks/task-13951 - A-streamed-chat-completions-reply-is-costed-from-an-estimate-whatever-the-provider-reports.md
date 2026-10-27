---
id: TASK-13951
title: A streamed /chat/completions reply is costed from an estimate whatever the
  provider reports
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13936. chat_service records a streamed /chat/completions reply after the stream ends from an estimate (prompt estimate plus len(reply)/4, estimate_source stream_estimate, no usage_metadata; execute_streaming_call's save_callback in chat_service.py), and the call records itself, so the provider-level recorder that reads the provider's last usage chunk never runs. So a streamed reply is costed at the standard rate with no service tier (OpenAI Flex, Fast, Batch), no cache rates, no Gemini cache reads and no OpenRouter usage.cost, whatever the provider reports. TASK-13916 fixes the Anthropic adapter's stream usage; that alone does not change /chat/completions, which uses this recorder. Usage_Quotas.md documents the gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed /chat/completions reply whose provider sends a usage chunk is recorded with that usage (service tier, cache tokens, reported cost), with a test
- [ ] #2 A stream without a usage chunk is still recorded from the estimate
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
