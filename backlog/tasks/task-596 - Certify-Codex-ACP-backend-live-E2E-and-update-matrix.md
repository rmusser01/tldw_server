---
id: TASK-596
title: Certify Codex ACP backend live E2E and update matrix
status: Done
labels:
- ACP
- Codex
- agents
- certification
priority: high
documentation:
- Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
- Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
- Docs/Development/ACP_Certification_Checklist.md
- Docs/Development/ACP_Compatibility_Matrix.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the Codex external_acp_adapter profile through the backend live ACP certification lifecycle using the pinned codex-acp 0.15.0 adapter. Scope includes starting or reusing a real local backend, running the existing live-e2e helper with ACP_AGENT_PROFILE=codex, recording initialize/session/prompt/redacted views/diagnostics/cancel/close evidence, updating compatibility docs only to the level actually proven, and documenting any remaining Codex/MCP caveats without overclaiming support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real local backend is started or a current running backend is verified with health/API-key evidence.
- [x] #2 The Codex backend live-E2E helper is run with the pinned codex-acp 0.15.0 adapter on PATH and ACP_AGENT_PROFILE=codex.
- [x] #3 The evidence records whether health, setup-guide, sessions/new, sessions/prompt, redacted detail/events/artifacts, diagnostics, cancel, and close pass or fail.
- [x] #4 ACP compatibility documentation is updated only to the verified support level and includes concrete commit/branch/date/runtime/version evidence.
- [x] #5 Focused verification, Bandit where backend Python is touched, and git diff hygiene are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added the Codex external ACP adapter entry to the bundled runner home config so backend `sessions/new` resolves `agent_type=codex` instead of failing with `unknown agent type: codex`.
- Added registry regression coverage for the bundled runner Codex profile and seeded `agents.yaml` live-E2E metadata.
- Started a real backend on `127.0.0.1:18003` with `PATH=/private/tmp/tldw-codex-acp-0.15.0/node_modules/.bin:$PATH`, single-user API key auth, and the updated runner config.
- Ran `Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` with `ACP_AGENT_PROFILE=codex`; result: `PASS live_backend_acp_e2e` with session `019e89c4-bfc8-7b30-a9e0-38917be6d117`, `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, and `diagnostics_total=0`.
- The helper also ran `tools/tldw-agent/scripts/verify-local-build.sh`; Go runner packages passed.
- Live backend health after metadata update reports Codex `status=available`, `support_state=supported_with_caveats`, `verification_level=live_e2e_tested`, `probe_state=ready_to_probe`, and no blockers.
- Remaining caveats: macOS host runner only; sandbox behavior, non-empty MCP injection, artifact-producing workflows, and reviewer-loop behavior remain unverified; diagnostics endpoint was reachable but no failure diagnostic payload was produced in the passing run.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Codex ACP backend live E2E certification completed for the macOS host runner using Codex CLI 0.128.0 through pinned codex-acp 0.15.0. The seeded runner config now includes Codex, the stale live_certification_required blocker has been removed from the Codex registry row, active compatibility docs and user-guide snippets now report live_e2e_tested with caveats, and focused registry/setup-guide tests pass.
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
