---
id: TASK-499
title: Address PR 2038 ACP Goose/OpenCode review comments
status: Done
labels:
- ACP
- review
- Goose
- OpenCode
references:
- https://github.com/rmusser01/tldw_server/pull/2038
documentation:
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/Config_Files/acp_runner_home/.tldw-agent/config.yaml
modified_files:
- IMPLEMENTATION_PLAN_acp_goose_backend_live_e2e.md
- Docs/Plans/IMPLEMENTATION_PLAN_acp_pr2038_review_fixes.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/app/core/Agent_Client_Protocol/config.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_cwd.py
- tools/tldw-agent/internal/acp/runner.go
- tools/tldw-agent/internal/acp/runner_test.go
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the actionable Qodo/Gemini review feedback on PR #2038 after the OpenCode/Aider stack was merged into the Goose live-E2E PR. Focus on evidence metadata completeness, support-state doc consistency, new test return annotations, default runner agent selection, and host home propagation for absolute/override runner environments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Evidence docs include commit, runner, and host metadata for live-certified ACP agents covered by PR #2038.
- [x] ACP support-state docs consistently reflect Goose/OpenCode/Hermes support caveats and Aider's not-yet-certified status.
- [x] New/changed Python tests include return annotations requested by review.
- [x] Runner default agent selection skips configured agents without executable commands and preserves explicit agent requests.
- [x] Runner environment path resolution propagates `TLDW_ACP_HOST_HOME` for absolute host homes, including override/no-rewrite paths.
- [x] Focused Python/Go tests plus Bandit on touched Python code are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_acp_pr2038_review_fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added regression coverage for absolute runner `HOME`, explicit `ACP_RUNNER_ENV` override paths, and default agent selection when the configured default profile has an empty command.
- Moved host-home export ahead of `_resolve_runner_env_paths()` early returns so downstream registered-agent env expansion can use `TLDW_ACP_HOST_HOME` without rewriting the configured runner `HOME`.
- Added Go runner default selection that chooses the configured default only when it has a non-empty command, otherwise chooses the first runnable registered agent while preserving explicit agent type resolution.
- Updated ACP registry YAML, compatibility matrix, and entrypoint-strategy design docs with evidence commit/branch/host/runner metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the PR #2038 review feedback by exporting TLDW_ACP_HOST_HOME for absolute and override runner env paths, making omitted ACP agentType selection skip configured profiles without runnable commands, adding regression coverage for both behaviors, and adding return annotations to the reviewed Python tests. Updated ACP registry metadata and compatibility docs with branch, commit, host, and tldw-agent runner evidence for Goose/OpenCode/Aider/Hermes, and aligned the entrypoint-strategy design doc support states with the current live-E2E certification results.

Addressed the subsequent CodeRabbit comments by adding a top-level heading plus MD022 spacing to the Goose live-E2E plan and narrowing registered-agent env expansion so only `${NAME}` placeholders in the value side of `KEY=VALUE` entries are expanded; literal `$NAME`, keys, and no-equals entries are preserved.

Verification: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_cwd.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py -q` passed 47 tests; `go test ./internal/acp -count=1` passed; `tools/tldw-agent/scripts/verify-local-build.sh` passed; `python -m bandit -r ... -s B101 -f json` reported zero findings; `git diff --check` passed.

Known caveats remain the documented ACP support caveats: sandbox behavior, non-empty MCP injection, artifact-producing workflows, and reviewer-loop behavior require separate certification before those capabilities are claimed.
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
