---
id: TASK-2350
title: Address PR 2344 permission runtime review feedback
status: Done
assignee: []
created_date: '2026-06-11'
updated_date: '2026-06-11'
labels:
  - mcp
  - profiles
  - permissions
  - runtime
  - review-feedback
dependencies:
  - TASK-2349
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address automated review feedback on PR #2344 (gateway runtime permission-rule enforcement). Qodo flagged request-driven CPU/memory amplification: permission rules were recompiled on every tools/call and all extracted subjects materialized without bounds, so oversized paths/urls/argv payloads drive O(subjects×rules) work before backend execution. Gemini requested a defensive None-guard for profile/policy_document before compiling rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compiled permission rules are cached per profile version so repeated tool calls against an unchanged profile compile once, and profile updates take effect on the next call.
- [x] #2 The cache is bounded so many distinct profiles cannot grow it without limit.
- [x] #3 Subject extraction enforces limits on subject count, subject value length, and argv token count, failing closed with a redacted explainable denial instead of truncating or skipping subjects.
- [x] #4 Profiles without a policy document (or absent profiles) skip permission-rule compilation without error.
- [x] #5 Rule-free profiles are unaffected by the extraction limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a bounded LRU cache (collections.OrderedDict, 64 entries) of compiled permission rules on ProfileAwareGatewayRuntime, keyed by (profile.id, profile.updated_at). updated_at is bumped by every gateway management mutation, so it acts as the profile version stamp; direct store writes that reuse a stale updated_at are out of scope (the management surface is the supported mutation path). Compile failures are not cached and keep raising GatewayPolicyDenied with reason_code=invalid_permission_rules. _compiled_permission_rules() returns () when profile or policy_document is None (Gemini guard). _enforce_permission_rules_for_tool_call() now receives pre-compiled rules instead of compiling per call.

Subject extraction is bounded during extraction (not after materializing): _MAX_PERMISSION_SUBJECTS=128, _MAX_SUBJECT_VALUE_LENGTH=4096, _MAX_COMMAND_ARGV_TOKENS=256. Exceeding any limit fails closed with GatewayPolicyDenied status=denied, reason_code=permission_subject_limits_exceeded, and redacted provenance (profile_id, tool_name, limit name only — no raw values). Fail-closed was chosen over truncation because silently skipping subjects past the cap could let an oversized payload push a denied subject out of evaluation. Limits only apply when the profile has permission rules, so rule-free profiles see no behavior change.

TDD evidence:
- Baseline before edits: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q` passed with 246 tests.
- Red tests: cache-reuse test failed with 2 compiles instead of 1; None-guard test failed with AttributeError (helper missing); oversized subject-count/value/argv tests failed because calls succeeded instead of returning policy denials.
- Green: same tests pass; full gateway file passes with 184 tests.
- Cache-key mutation check: sabotaging the cache key to (profile.id, None) makes the recompile-on-update test fail; restoring (profile.id, profile.updated_at) makes it pass.

Verification:
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q` passed with 254 tests.
- Ruff: `python -m ruff check mcp_unified/gateway/profile_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` passed.
- Compile smoke: `python -m compileall -q mcp_unified/gateway/profile_runtime.py` passed.
- Bandit: `python -m bandit -r mcp_unified/gateway/profile_runtime.py -q` reported no findings.
- Whitespace: `git diff --check` passed.

Deferred: pattern-level cache invalidation for direct profile_store writes that bypass the management surface; configurable extraction limits.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cached compiled MCP profile permission rules per (profile id, updated_at) with a bounded LRU on the gateway runtime, added an explicit None-guard for missing policy documents, and bounded permission-subject extraction (subject count, value length, argv tokens) failing closed with redacted permission_subject_limits_exceeded denials. Addresses Qodo and Gemini review feedback on PR #2344.
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
