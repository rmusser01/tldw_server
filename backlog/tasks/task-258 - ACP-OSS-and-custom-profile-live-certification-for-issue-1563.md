---
id: TASK-258
title: ACP OSS and custom profile live certification for issue 1563
status: In Progress
assignee: []
created_date: '2026-05-11 05:17'
labels:
  - ACP
  - compatibility
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1563'
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
Track GitHub issue #1563: run or explicitly block live ACP certification for the OSS/custom registry profiles Aider, Goose, Continue, OpenCode, and custom ACP profiles. Use the existing ACP certification checklist, compatibility matrix, setup-guide/API status surfaces, and Agent Registry metadata. Do not upgrade any profile beyond documented_unverified unless real evidence proves the ACP stdio contract; document missing binaries, shell alias collisions, absent ACP entrypoints, credentials, or custom-profile incompleteness with the existing caveat taxonomy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Aider live ACP evidence is recorded or an explicit blocker is documented with host, binary/version, and caveat taxonomy.
- [x] #2 Goose live ACP evidence is recorded or an explicit blocker is documented with host, binary/version, and caveat taxonomy.
- [x] #3 Continue live ACP evidence is recorded or an explicit blocker is documented with host, binary/version, and caveat taxonomy.
- [x] #4 OpenCode live ACP evidence is recorded or an explicit blocker is documented with host, binary/version, and caveat taxonomy.
- [x] #5 Custom ACP profile requirements are documented without implying generic support for arbitrary commands.
- [x] #6 Docs/Development/ACP_Compatibility_Matrix.md and tldw_Server_API/Config_Files/agents.yaml agree on support state and verification level for the OSS/custom profiles.
- [x] #7 Focused tests or validation commands for touched ACP docs/config/API surfaces pass, and Bandit is run on any touched Python scope.
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
- Created `Docs/Development/ACP_OSS_Custom_Certification_2026_05_11.md` with explicit blocker evidence for Aider, Goose, Continue, OpenCode, and the default custom ACP template.
- Added missing compatibility-matrix rows for Aider, Goose, and Continue so the matrix agrees with `agents.yaml`.
- Kept all OSS/custom profiles at `documented_unverified` / `documented_only`; no live support claim was made.
- Validation: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_status_schema.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q` passed, 55 tests.
- Validation: `tools/tldw-agent/scripts/verify-local-build.sh` passed.
- Validation: `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --format json` emitted the expected manifest.
- Validation: `python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run` refused without required live env, exit 2.
- Validation: `git diff --check` passed.
- Bandit: skipped because this task touched only Markdown, YAML, and Backlog task metadata; no Python scope changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the #1563 OSS/custom certification attempt without overstating support. Aider, Goose, and OpenCode were not installed; Continue resolved to a shell builtin rather than a CLI; and the default custom profile has no concrete command to certify. The compatibility matrix and Agent Registry notes now keep all OSS/custom profiles at `documented_unverified` / `documented_only` and point to blocker evidence or concrete requirements for future certification.
<!-- SECTION:FINAL_SUMMARY:END -->
