---
id: TASK-128
title: Support numbered custom OpenAI endpoints through 99
status: Done
assignee:
  - Codex
created_date: '2026-05-08 23:31'
updated_date: '2026-05-08 23:45'
labels:
  - feature
  - config
  - llm-adapters
dependencies:
  - TASK-127
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the custom OpenAI-compatible provider support from the existing first and second endpoints to numbered endpoint slots through 99. Users should be able to configure additional providers such as custom-openai-api-3 through custom-openai-api-99 using numbered environment/config values while retaining existing custom-openai-api and custom-openai-api-2 behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Providers custom-openai-api-3 through custom-openai-api-99 are accepted by chat provider validation and the LLM adapter registry.
- [x] #2 Each numbered provider resolves to its own app_config section and endpoint env/config values without duplicating 97 hand-written adapter classes.
- [x] #3 Existing custom-openai-api and custom-openai-api-2 provider names, aliases, keys, and endpoint env behavior remain compatible.
- [x] #4 Numbered providers use the same OpenAI-compatible request field capabilities as the first two custom OpenAI providers.
- [x] #5 Documentation and .env examples explain how to configure additional numbered endpoints through 99.
- [x] #6 Regression tests cover at least a mid-range numbered endpoint and the upper bound custom-openai-api-99.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design: dynamically support provider IDs `custom-openai-api-3` through `custom-openai-api-99`, preserving existing `custom-openai-api` and `custom-openai-api-2` semantics. Numbered provider N maps to app_config section `custom_openai_api_N`, env endpoint aliases such as `CUSTOM_OPENAI_API_IP_N`/`CUSTOM_OPENAI_API_BASE_N`, and API key aliases such as `CUSTOM_OPENAI_API_KEY_N`/`CUSTOM_OPENAI{N}_API_KEY`.

Implementation plan:
1. Add failing regression tests for provider validation, adapter registry/provider materialization, config/env resolution for representative endpoint 37, and upper-bound endpoint 99.
2. Add shared custom OpenAI provider naming helpers so registry, config, adapter utilities, request validation, capability validation, BYOK mapping, and docs use the same provider-number rules.
3. Refactor the custom OpenAI adapter to create numbered adapter classes/factories dynamically instead of hand-writing classes through 99.
4. Extend `load_and_log_configs()` to populate configured numbered custom provider sections only when env/config values are present, with shared defaults for optional model/temperature/token settings.
5. Update docs and `.env.example` with numbered endpoint examples and compatibility notes.
6. Run focused tests, `git diff --check`, and Bandit on touched Python implementation files; then finalize the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented shared custom OpenAI provider numbering helpers and wired custom-openai-api-3 through custom-openai-api-99 into config loading, chat request validation, adapter registry/materialization, payload capability checks, provider metadata, BYOK config overrides, eval/media helper paths, tokenizer routing, and docs/examples.

Verification: initial focused red run failed 8 expected tests for missing numbered providers; after implementation, focused provider/config/schema tests passed 110/110. Secondary targeted tests for media eval key resolution, ms_g_eval validation, tokenizer resolver, and provider key metadata passed 19/19. git diff --check reported no whitespace errors.

Bandit: ran on touched Python implementation files and wrote /tmp/bandit_custom_openai_99.json. It reported three LOW findings on pre-existing lines outside this change: tokenizer_resolver.py:888 B110 try/except/pass, config.py:603 B105 'tiktoken', and config.py:663 B105 None; no findings in the new custom endpoint helper or dynamic adapter/config code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added dynamic numbered custom OpenAI-compatible provider support through custom-openai-api-99. The implementation centralizes provider names, aliases, env keys, config option names, and app_config section names, then reuses those helpers across config loading, adapter registration/materialization, chat schemas/validators, capability metadata, BYOK overrides, eval/media utilities, tokenizer routing, and docs. Existing custom-openai-api and custom-openai-api-2 aliases remain compatible while additional endpoints can be configured with numbered env vars such as CUSTOM_OPENAI_API_IP_37 / CUSTOM_OPENAI_API_KEY_37 / CUSTOM_OPENAI_API_MODEL_37 or config.txt fields such as custom_openai37_api_ip.
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
