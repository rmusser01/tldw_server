---
id: TASK-496
title: Run Hermes ACP backend live E2E certification
status: Done
labels:
- ACP
- certification
- Hermes
references:
- https://github.com/rmusser01/tldw_server/issues/1563
- https://github.com/rmusser01/tldw_server/issues/1532
documentation:
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/ACP_Certification_Checklist.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the next ACP certification slice for Hermes through the backend live-E2E path, record evidence, and make any small harness/docs fixes needed to keep claims accurate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Backend live-E2E helper exists in the certification manifest and refuses to run without explicit runtime env.
- [x] Hermes backend live E2E completes through health/setup-guide, session create, prompt, redacted support views, diagnostics, cancel, close, and runner verification.
- [x] Compatibility matrix and agent registry metadata reflect only evidence-backed Hermes support with caveats.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a backend REST live-E2E path to `Helper_Scripts/Testing-related/acp_certification_smoke.py`. The live run found two contract issues, so the runner now strips routing-only `agentType` before forwarding `session/new` downstream, and the backend preserves explicit empty MCP server lists through REST -> runner client -> JSON-RPC. Hermes is now marked `supported_with_caveats` / `live_e2e_tested` for the verified macOS host profile only.

PR review follow-up addressed Qodo and Gemini feedback: cleanup close failures now emit a warning with the session id, required env validation strips whitespace, non-local plaintext HTTP is rejected unless explicitly allowed, request timeout parsing is performed once per backend live-E2E run, `_fake_http` test helpers have complete type hints, and the Go runner documents why `json.RawMessage` is used for routing-param stripping.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hermes ACP backend live E2E passed on May 23, 2026 with `ACP_AGENT_PROFILE=hermes`: `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, and `diagnostics_total=0`. Verification also passed for focused pytest coverage, `tools/tldw-agent/scripts/verify-local-build.sh`, Bandit on touched Python files, and `git diff --check`. Remaining caveats: sandbox behavior, non-empty MCP injection, artifact-producing workflows, reviewer-loop behavior, and failure diagnostic payloads remain unverified.

PR review remediation verification: 39 focused pytest checks passed, `tools/tldw-agent/scripts/verify-local-build.sh` passed, Bandit produced zero findings for touched Python files, and `git diff --check` passed.
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
