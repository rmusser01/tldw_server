---
id: TASK-502
title: Allow custom OpenAI-compatible providers without default credentials
status: Done
labels:
- bug
- llm
- custom-openai-api
priority: High
modified_files:
- tldw_Server_API/app/core/LLM_Calls/provider_metadata.py
- tldw_Server_API/app/core/Evaluations/ms_g_eval.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py
- tldw_Server_API/tests/Chat_NEW/unit/test_provider_keys_map.py
- tldw_Server_API/tests/Config/test_config_providers_endpoints.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_ms_g_eval_validation.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_video_ingestion.py
- tldw_Server_API/tests/Chat/integration/test_chat_endpoint.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix custom-openai-api provider metadata so local/self-hosted OpenAI-compatible endpoints are not rejected before adapter invocation when no API key is configured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression coverage for custom OpenAI-compatible providers as keyless-by-default in provider metadata, config provider status, video confabulation checks, and chat completions credential gating.
2. Change shared provider metadata so custom-openai-api, custom-openai-api-2, and numbered custom-openai-api-N providers do not require keys by default.
3. Replace the video ingestion confabulation check's local custom-provider key rule with the shared provider_requires_api_key helper.
4. Verify focused tests, adapter regression tests, py_compile, and Bandit on touched production files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review follow-up: G-Eval validation also used its own provider key list, so custom OpenAI-compatible providers still failed no-key confabulation checks after video ingestion reached run_geval. G-Eval now uses provider_requires_api_key, provider metadata initializes all custom OpenAI-compatible provider names from the shared iterator, and video ingestion lazily imports the provider policy inside the optional confabulation branch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed custom OpenAI-compatible providers to be keyless by default in shared provider metadata, aligned video confabulation and G-Eval validation with the shared provider policy, and added regressions for metadata, config provider status, video confabulation, G-Eval validation, and chat completion credential gating. Verification: focused regression suite passed (21 passed), custom OpenAI adapter/provider map tests passed (7 passed), py_compile passed for touched production files, and Bandit reported 0 findings for touched production files.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
