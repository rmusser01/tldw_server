---
id: TASK-12073
title: Fix post-PR-1982 main pre-commit failure
status: Done
priority: High
references:
- https://github.com/rmusser01/tldw_server/actions/runs/28421862168
modified_files:
- Docs/API-related/Watchlists_API.md
- Docs/Design/Evals.md
- Docs/Design/Sandbox.md
- Docs/Design/Security.md
- Docs/Design/tldw_web_design_system_inventory.md
- Docs/MCP/Unified/CodeGraph.md
- Docs/Published/API-related/Watchlists_API.md
- Docs/Reviews/WORKSPACE_PLAYGROUND_A11Y_CONTRAST_AUDIT_2026_02_18.md
- Docs/Reviews/assets/2026-05-09-character-chat-p1-smoke/puppeteer-p1-smoke.json
- Docs/Reviews/assets/2026-05-09-character-chat-reaudit/puppeteer-states.json
- Docs/User_Guides/Writing_Characters.md
- Docs/superpowers/specs/2026-05-26-research-workspace-mcp-hub-deep-link-design.md
- apps/mcp-unified/src/mcp_unified/py.typed
- apps/packages/ui/src/components/Common/CodeBlock.tsx
- apps/packages/ui/src/db/dexie/types.ts
- apps/packages/ui/src/hooks/keyboard/useShortcutConfig.ts
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
- backlog/milestones/Research-Workspace-UAT-Remediation.md
- tldw_Server_API/app/api/v1/schemas/audio_health.py
- tldw_Server_API/app/core/Evaluations/db_adapter.py
- tldw_Server_API/app/core/Sandbox/exceptions.py
- tldw_Server_API/cli/wizard/profile_verify.py
- tldw_Server_API/cli/wizard/profiles.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_helpers.py
- tldw_Server_API/tests/wizard/test_cli_verify_profiles.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the post-PR-1982 main pre-commit failure by applying the same EOF/whitespace/Black formatting changes CI reported, splitting synthetic PEM-like test markers so detect-private-key can pass without weakening redaction coverage, and updating the onboarding UAT fixture assertion to match the hosted mock auth contract. Verification: exact pre-commit range be2e7f8686e49d95e83827d3c2006ed37f29de58..06d0198d46d89465bc5e889fdfe32e973baf1274 passed; pytest for ACP hardening and wizard profile tests passed (26 passed); targeted frontend vitest passed (27 passed); Bandit on touched production Python files reported 0 findings.
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
