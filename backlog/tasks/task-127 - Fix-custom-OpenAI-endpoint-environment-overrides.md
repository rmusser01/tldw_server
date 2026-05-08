---
id: TASK-127
title: Fix custom OpenAI endpoint environment overrides
status: Done
assignee:
  - Codex
created_date: '2026-05-08 23:22'
updated_date: '2026-05-08 23:26'
labels:
  - bug
  - config
  - llm-adapters
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1381'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Issue #1381 reports that setting CUSTOM_OPENAI_API_BASE in .env does not redirect the custom OpenAI-compatible adapter because config.txt custom_openai_api_ip remains effective. Investigation also found that load_and_log_configs reads custom_openai_api_ip/custom_openai2_api_ip directly from config.txt, while adapter fallbacks only look at CUSTOM_OPENAI_API_IP_1/_2 when app_config has no endpoint. Make endpoint resolution env-first, support documented and common alias names, and update docs so users can discover the supported variables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 load_and_log_configs resolves custom_openai_api.api_ip from process env or loaded .env before config.txt for the first custom OpenAI provider.
- [x] #2 load_and_log_configs resolves custom_openai_api_2.api_ip from process env or loaded .env before config.txt for the second custom OpenAI provider.
- [x] #3 CUSTOM_OPENAI_API_BASE is accepted as an alias for the first custom OpenAI endpoint and has equivalent behavior to the canonical endpoint env variable.
- [x] #4 Adapter fallback env names are aligned with the runtime config names without breaking existing CUSTOM_OPENAI_API_IP_1/_2 usage.
- [x] #5 Documentation lists the canonical custom OpenAI endpoint env variables and aliases clearly.
- [x] #6 Regression tests cover config precedence and adapter fallback behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression tests in tldw_Server_API/tests/Config/test_config_precedence_contract.py for custom_openai_api/custom_openai_api_2 endpoint env precedence and CUSTOM_OPENAI_API_BASE alias behavior.
2. Add focused adapter fallback tests in tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py covering canonical env names plus existing _1/_2 compatibility.
3. Update load_and_log_configs() to resolve custom OpenAI endpoint values env-first, preserving config.txt fallback.
4. Update CustomOpenAIAdapter fallback env tuples to include canonical and alias names while preserving existing CUSTOM_OPENAI_API_IP_1/_2.
5. Update environment-variable docs/templates so the canonical variables and aliases are discoverable.
6. Run focused tests and Bandit on touched Python files, then update Backlog acceptance criteria/final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented env-first custom OpenAI endpoint resolution in load_and_log_configs() using canonical variables CUSTOM_OPENAI_API_IP and CUSTOM_OPENAI2_API_IP, accepted CUSTOM_OPENAI_API_BASE and additional base-url aliases, and preserved CUSTOM_OPENAI_API_IP_1/_2 compatibility in config and adapter fallback paths.

Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Config/test_config_precedence_contract.py tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py -q` passed 11 tests with 6 existing warnings. Smoke: `CUSTOM_OPENAI_API_BASE=http://127.0.0.1:9000/v1 ... load_and_log_configs()["custom_openai_api"]["api_ip"]` printed `http://127.0.0.1:9000/v1`.

Bandit: `python -m bandit -r tldw_Server_API/app/core/config.py tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py -f json -o /tmp/bandit_custom_openai_env.json` exited 1 due to two pre-existing LOW B105 findings in config.py lines 581 and 641; no findings were reported for custom_openai_adapter.py or the new endpoint-resolution code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed custom OpenAI-compatible endpoint resolution so environment and .env values win over config.txt for both custom providers. The first provider now accepts CUSTOM_OPENAI_API_IP plus CUSTOM_OPENAI_API_BASE and related base-url aliases; the second provider accepts CUSTOM_OPENAI2_API_IP plus equivalent aliases. Adapter fallback env tuples were aligned with those names while retaining CUSTOM_OPENAI_API_IP_1 and CUSTOM_OPENAI_API_IP_2 for compatibility.

Updated Docs/Operations/Env_Vars.md, Config_Files/README.md, and Config_Files/.env.example so users can discover the canonical names and the compatibility aliases. Added regression tests for load_and_log_configs precedence, CUSTOM_OPENAI_API_BASE alias behavior, and adapter fallback behavior.

Verification: focused pytest file run passed 11 tests; actual config-loader smoke with CUSTOM_OPENAI_API_BASE returned the env endpoint. Bandit was run on touched Python implementation files and reported only pre-existing low B105 findings outside the changed logic.
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
