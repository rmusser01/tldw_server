---
id: TASK-541
title: Add opt-in Chatterbox BF16 inference support
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-08 23:53'
labels:
  - tts
  - chatterbox
  - performance
  - config
dependencies: []
references:
  - 'https://github.com/devnen/Chatterbox-TTS-Server'
  - Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add opt-in BF16 handling for Chatterbox TTS generation, honoring chatterbox_use_bf16 / use_bf16 and TTS_BF16=on|auto while keeping default behavior off. Prepare T3 with bfloat16 when enabled and run generation under autocast when available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatterbox BF16 defaults to off and can be enabled with chatterbox_use_bf16, generic use_bf16, or TTS_BF16.
- [x] #2 Enabled BF16 prepares the Chatterbox T3 module with torch.bfloat16 and wraps TTS generation in torch.autocast when available.
- [x] #3 Generic use_bf16 in provider YAML is normalized to chatterbox_use_bf16.
- [x] #4 Provider YAML, setup runbook, and parity plan document the new option and default behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add failing adapter tests proving Chatterbox BF16 config prepares a model T3 module and runs generation under autocast, plus registry/config alias coverage for generic use_bf16. Implement defensive BF16 mode resolution with env support and best-effort runtime preparation, update provider YAML/docs/plan, then verify with focused tests, full Chatterbox adapter mock tests, adapter registry tests, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added BF16 mode normalization with off/on/auto values. The adapter keeps default precision unless explicitly enabled; auto currently requires CUDA BF16 support. Enabled TTS generation prepares model.t3 with torch.bfloat16 and uses a best-effort torch.autocast context. Voice conversion remains on the upstream default precision path. The Chatterbox provider alias map now carries generic use_bf16 into chatterbox_use_bf16.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed opt-in Chatterbox BF16 inference support with adapter tests, registry alias coverage, provider YAML/docs updates, and clean verification. Verification: python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v -> 39 passed; python -m pytest tldw_Server_API/tests/TTS/test_tts_adapters.py -v -> 37 passed, 3 skipped; python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/app/core/TTS/adapter_registry.py -f json -o /tmp/bandit_chatterbox_bf16_task541.json -> clean; git diff --check -> clean.
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
