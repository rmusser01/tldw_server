---
id: TASK-13236
title: Resolve omitted chat models from provider configuration (issue 2938)
status: Done
assignee: []
created_date: '2026-09-10 03:35'
updated_date: '2026-09-10 04:18'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2938'
documentation:
  - Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the metrics placeholder unknown from becoming an upstream model. Honor configured provider models, preserve override and explicit model precedence, fail clearly when no model exists, and audit related callers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Omitted models resolve from provider configuration for local-llm and Ollama
- [x] #2 Explicit models and existing administrator defaults retain precedence
- [x] #3 Missing configuration never sends unknown upstream and returns a clear client error
- [x] #4 Regression tests include streaming and non-streaming behavior and related default-model paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed provider-default repair, metrics/execution separation, blank-model and shared-consumer audit, regressions, and independent review. Retained design and verification: Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Final resolver/default/payload suite: 146 passed. Endpoint omitted, empty, whitespace, configured-default, and explicit-model cases across both streaming modes: 20 passed. Exact LOCAL_LLM_MODEL and Local-API.ollama_model parsing is tested through real configuration helpers. Related suites passed: 209 Messages usage, 111 Messages endpoint/override/default/native-error, 30 character/Notes, and 7 macro tests. Full endpoint file before the whitespace follow-up: 203 passed and 1 existing skip (Streaming tests hang with TestClient); all affected cases were rerun afterward. Independent review found the whitespace bypass and cleared its correction in the resolver, payload builder, Messages, macro context, and tool-policy identity. Compilation, repository guards, and diff checks pass. Bandit on four production files has zero findings. Ruff has no new findings; three existing chat.py import-order findings were reproduced on base 751563a966. Existing whole-file Black drift was retained to avoid unrelated formatting changes. The full backend suite was not run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Execution now retains omitted or blank models as None while metrics may use unknown. Provider defaults consult the existing configuration snapshot after administrator, DEFAULT_MODEL environment, and Chat-Module precedence. Explicit models retain precedence; missing effective configuration returns HTTP 400 before dispatch. Added regressions for both streaming modes, exact configuration inputs, payload values, precedence, and shared Messages behavior. Audited macro and tool-policy contexts to prevent raw blank-model fallback.
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
