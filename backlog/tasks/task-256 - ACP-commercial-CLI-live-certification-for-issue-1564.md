---
id: TASK-256
title: ACP commercial CLI live certification for issue 1564
status: Done
assignee: []
created_date: '2026-05-11 05:01'
updated_date: '2026-05-31 17:04'
labels:
  - ACP
  - compatibility
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1564'
  - 'https://github.com/rmusser01/tldw_server/pull/1562'
documentation:
  - Docs/Development/ACP_Certification_Checklist.md
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/Agent_Client_Protocol.md
  - tldw_Server_API/Config_Files/agents.yaml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #1564: run or explicitly block live ACP certification for the seeded Claude Code and OpenAI Codex CLI profiles. Work must use the existing ACP certification checklist, compatibility matrix, setup-guide/API status surfaces, and Agent Registry metadata. Do not claim live support unless the local evidence actually proves it; document missing credentials, binary limitations, or protocol blockers with the existing caveat taxonomy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Claude Code live ACP evidence is recorded or an explicit blocker is documented with host, binary/version, and caveat taxonomy.
- [x] #2 Codex CLI live ACP evidence is recorded or an explicit blocker is documented with host, binary/version, and caveat taxonomy.
- [x] #3 Docs/Development/ACP_Compatibility_Matrix.md and tldw_Server_API/Config_Files/agents.yaml agree on support state and verification level for Claude Code and Codex CLI.
- [x] #4 ACP setup/status surfaces remain consistent with the documented support state and do not expose unsupported live-agent claims.
- [x] #5 Focused tests or validation commands for touched ACP docs/config/API surfaces pass, and Bandit is run on any touched Python scope.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
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
- Closeout refresh on 2026-05-31: the task evidence, acceptance criteria, DoD, and final summary were already complete on `origin/dev`; this update marks the Backlog task Done. GitHub issue #1564 remains open as the live commercial-agent certification parent tracker for future recertification attempts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the #1564 commercial CLI certification attempt without overstating support. Codex CLI 0.128.0 and Claude Code 2.1.119 were present locally, but the configured registry commands did not complete an ACP `initialize` handshake in this environment, and required live credentials/env were unavailable. The compatibility matrix and Agent Registry notes now point to the blocker evidence and keep both profiles at `documented_unverified` / `documented_only`.
<!-- SECTION:FINAL_SUMMARY:END -->
