---
id: TASK-13585
title: Record LLM usage for MCP tool calls and the in-process claims rebuild
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two roots still make chat LLM calls with no known user after the LLM usage accounting PR 1b. MCP: tool calls over /ws and HTTP reach perform_chat_api_call_async (core/MCP_unified quizzes module), and MCP authentication sets no content scope; the tool-execution entry has the user in its RequestContext. Claims rebuild: core/Claims_Extraction/claims_rebuild_service.py runs daemon threads whose ClaimsRebuildTask carries only media_id and db_path, so there is no owner to attribute to. Parent: TASK-13437.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An MCP tool call that makes an LLM call records one row for the calling user
- [x] #2 A claims rebuild task carries its owner and its LLM calls are recorded for that user
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: MCP test_tool_llm_usage_attribution.py with the outcome catalog test, 5 failed, 6 passed on dev (no rows; no refusal because no user, KeyError: 'isError'; a tool under chat tool auto-exec not recorded; catalog missing LLM_QUOTA_EXCEEDED). Claims, step by step: the worker test failed with (0, 0, [1]) == (1, 1, [1, 2]) (the worker thread died on a refusal), and the owner tests with the existing fakes 5 failed, 16 passed (submit has no owner_user_id; owners None in the override_db and startup fakes). GREEN: MCP 11 passed; claims 21 passed; every MCP suite 4380 passed, 16 skipped; tests naming records_itself or tool auto-exec 215 passed; Claims suites 1082 passed, 1 skipped.
MCP uses llm_call_context(user_id=..., records_itself=False) around module.execute_tool, not llm_call_root: the root clears prechecked, which would let §4b refuse a tool that an admitted /chat/completions request runs (tool auto-exec). records_itself is cleared because the chat pipeline records only its own chat call (owner decision D2, accepted 2026-10-18; ADR-075). A refusal becomes ExpectedToolFailure(LLM_QUOTA_EXCEEDED, BreakerAction.IGNORE), so the module breaker no longer counts it; this also covers /api/v1/tools/execute, where the content scope made refusals reach the breaker since PR 2b. There the refusal is HTTP 200 with ok: true and result.isError: true.
AC #1 covers MCP tool calls over the MCP HTTP and WebSocket transports, /api/v1/tools/execute, and tools run by /chat/completions tool auto-execution, each for the caller's numeric user id. A caller with no numeric user id (a service principal) is not recorded. Direct module.execute_tool callers outside the MCP runtime (workflow MCP and LLM steps, speech chat) are covered by their own roots.
ADR check: ADR required: yes. ADR path: Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md (Accepted by the owner, D3), extending ADR-067 (governing). Reason: clearing records_itself for MCP tools changes ADR-067's consequence that a tool calling an LLM inside the chat pipeline is recorded by nobody. The MCP user and the claims owner are new roots under ADR-067's mechanism and need no ADR.
Claims rebuild: the worker now treats LLMQuotaExceededError as one failed task; without that, the new owner would let a refusal end the only worker thread. It logs a refusal at warning and stores llm_quota_exceeded, not the quota details, as the health row's last failure. A refused rebuild keeps the media's claims (extraction runs before the soft delete). Test-quality ratchet: the claims-kept guard also asserts that the refusal reaches the worker unwrapped, and the claim-edit helper test checks the task the real ClaimsRebuildService.submit enqueues; the plan's collector-only versions were flagged tautology_suspect. Bandit: uvx bandit -ll, 0 findings. Flake seen once under load: test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property (Hypothesis FailedHealthCheck, slow input generation); green alone and in its file (142 passed).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
MCP tool calls now record and check their LLM calls for the user in the call's RequestContext, over the MCP transports and /api/v1/tools/execute; a tool that /chat/completions tool auto-execution runs is recorded too and, being admitted, never refused (ADR-075). A refusal is the tool error llm_quota_exceeded and does not trip the module circuit breaker. A caller with no numeric user id is not recorded. In-process claims rebuild tasks carry their owner from every caller (per-media and bulk rebuild, claim-edit follow-up, startup loop) and run under llm_call_root(owner); a refusal fails one task, logs a warning, and the worker continues. ADR-075, ADR-067, the spec, Usage_Quotas.md and the MCP README updated.
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
