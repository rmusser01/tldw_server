---
id: TASK-593
title: Implement Codex ACP adapter first slice
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-01 23:55'
labels:
  - ACP
  - Codex
  - agents
  - implementation
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
  - Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
priority: high
ordinal: 593
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved first-slice Codex ACP adapter implementation plan. Scope includes canonical external_acp_adapter backend/API strategy support, dynamic registry metadata persistence, seeded Codex codex-acp profile/docs, passive Go runner readiness and explicit launch rules, frontend structured readiness gating, certification helper compatibility, and verification. Live Codex ACP certification and Codex app-server support remain follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical external_acp_adapter strategy is implemented and legacy adapter_acp is accepted only as an import compatibility alias.
- [ ] #2 Backend/API/static fallback/DB registry metadata expose adapter readiness without stale OPENAI_API_KEY-only Codex semantics.
- [ ] #3 Go runner uses passive no-spawn readiness for inventory/initialize and explicit acp_command launch rules for ACP sessions.
- [ ] #4 Frontend ACP session creation gates on structured entrypoint readiness and does not let stale is_configured override blocked states.
- [ ] #5 Docs and seeded Codex profile describe pinned codex-acp 0.15.0 without overclaiming live certification.
- [ ] #6 Focused backend, Go, frontend, Bandit, and diff hygiene verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete and approved. Commit d7651601c3 implements canonical external_acp_adapter strategy normalization, legacy adapter_acp import aliasing in registry/schema/endpoint helpers, Codex static fallback external-adapter/delegated-auth semantics, endpoint test update for canonical forwarded strategy, and live_certification_required setup-step coverage. Verification: focused pytest command passed locally with 36 passed and 6 warnings using /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv; git diff --check passed; Bandit on touched backend files passed with no findings. Review: spec compliance passed after adding explicit ACPAgentEntrypointStatus legacy alias coverage; code quality issues were fixed before approval.

Task 2 complete and approved. Commit e86d935951 exposes ACP adapter readiness metadata, separates display-agent binary readiness from external adapter availability, blocks mutable npx @latest adapter invocations, adds credential/runtime/adapter metadata to registry and public entrypoint status payloads, maps new blocker codes into setup guide copy, and keeps delegated adapter credentials passive without OPENAI_API_KEY blocking. Verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py` passed with 47 passed and 6 warnings; `git diff --check` passed; Bandit on touched backend files passed with no findings. Review: spec compliance and code quality approved locally after delegated reviewers stalled due agent latency.

Task 3 complete and approved. Commit b3cb9dcc94 persists external ACP adapter metadata in the dynamic agent registry, adds schema version 15 migration/defaults for adapter package/version/version policy/install source/credential policy/runtime backend, preserves the fields through register/update/reload API paths, and keeps legacy adapter_acp normalized on DB write. Verification: focused pytest `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py` passed with 81 passed and 6 warnings; `git diff --check` passed. Bandit on touched production files exited 1 only for existing ACP_Sessions_DB.py baseline findings outside the Task 3 diff: B105 at line 205 and B608 at lines 1058, 1063, 1099, 1117, 1947, 2162, and 2247; endpoint/schema/registry files had zero findings. Review: delegated spec review approved; local code quality review approved with the baseline Bandit note recorded.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
