---
id: TASK-256
title: ACP commercial CLI live certification for issue 1564
status: In Progress
assignee: []
created_date: '2026-05-11 05:01'
updated_date: '2026-06-03 04:38'
labels:
  - ACP
  - compatibility
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1564'
  - 'https://github.com/rmusser01/tldw_server/pull/1562'
  - 'https://github.com/rmusser01/tldw_server/pull/2224'
  - 'https://github.com/rmusser01/tldw_server/issues/2244'
  - 'https://github.com/rmusser01/tldw_server/pull/2248'
documentation:
  - Docs/Development/ACP_Certification_Checklist.md
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/Agent_Client_Protocol.md
  - tldw_Server_API/Config_Files/agents.yaml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #1564 as the narrowed commercial ACP live-certification rollup. Codex CLI via the pinned Codex ACP adapter now has live backend E2E evidence from PR #2224 and remains supported_with_caveats for the verified macOS host profile. Claude Code is narrowed under child issue #2244 to an external ACP adapter candidate and must stay documented_unverified unless the pinned adapter is installed and live ACP evidence is produced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Codex CLI live evidence is recorded with explicit caveats.
- [ ] #2 Claude Code entrypoint/support decision is resolved in GitHub issue #2244 and reflected in issue #1564.
- [ ] #3 Docs/Development/ACP_Compatibility_Matrix.md and setup surfaces agree on support state and verification level after the narrowed Claude child issue lands.
- [ ] #4 Release-note wording avoids unsupported live-agent claims.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-03: PR #2224 merged Codex CLI via pinned codex-acp 0.15.0 as supported_with_caveats/live_e2e_tested for the verified macOS host profile. Narrowed GitHub issue #1564 without closing it and created #2244 for the remaining Claude Code entrypoint or adapter decision.
2026-06-03: TASK-256.1 classified Claude Code as an external ACP adapter candidate using pinned @agentclientprotocol/claude-agent-acp 0.40.0. Local Claude Code 2.1.142 has no native ACP command and local claude-agent-acp is missing, so the profile remains documented_unverified/documented_only with adapter_missing evidence.
2026-06-03: Opened draft PR #2248 for the Claude adapter decision and commented on #1564/#2244. Keep #1564 open until PR #2248 lands and issue state is reconciled.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Created `Docs/Development/ACP_Commercial_CLI_Certification_2026_05_11.md` with explicit blocker evidence for Claude Code and Codex CLI on macOS host stdio.
- Kept both registry rows at `documented_unverified` / `documented_only`; no live support claim was made.
- Validation: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_status_schema.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q` passed, 55 tests.
- Validation: `tools/tldw-agent/scripts/verify-local-build.sh` passed.
- Validation: `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --format json` emitted the expected manifest.
- Validation: `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` refused without required live env, exit 2.
- Validation: `git diff --check` passed.
- Bandit: skipped because this task touched only Markdown, YAML, and Backlog task metadata; no Python scope changed.
<!-- SECTION:NOTES:END -->
