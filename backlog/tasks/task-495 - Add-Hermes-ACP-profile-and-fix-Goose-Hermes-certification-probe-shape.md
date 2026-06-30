---
id: TASK-495
title: Add Hermes ACP profile and fix Goose/Hermes certification probe shape
status: Done
labels:
- acp
- agents
- certification
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/1563
- https://github.com/rmusser01/tldw_server/issues/1532
documentation:
- Docs/Development/ACP_Certification_Checklist.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/Agent_Client_Protocol.md
- tldw_Server_API/Config_Files/agents.yaml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track ACP-adjacent certification work for Goose and Hermes after Goose CLI install and Hermes provider reconfiguration. Scope: add Hermes as a native ACP candidate in the agent registry, update the ACP certification smoke helper to emit protocol-valid session/new and session/prompt payloads for ready-to-probe agents, add focused tests, and record live certification results without overstating Goose readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Hermes appears in the default agent registry as a native ACP profile.
- [x] Native ACP certification manifests emit protocol-valid initialize, session/new, and session/prompt frames for Goose/Hermes.
- [x] Goose and Hermes compatibility docs reflect current host-stdio probe evidence and remaining backend E2E gaps.
- [x] Focused tests and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Hermes profile using `hermes acp --accept-hooks`.
- Updated registry-backed native ACP probe shape to use numeric `protocolVersion`, absolute workspace `cwd`, required `mcpServers`, ACP content-list prompts, and runtime `sessionId` substitution.
- Local sandbox blocks Hermes home-directory log writes, so live downstream host-stdio probes were run with normal host permissions.
- Verification:
  - `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile hermes --run` -> exit 0.
  - `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile goose --run` -> exit 0.
  - `python -m pytest tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py -q` -> 75 passed, 5 warnings.
  - `python -m bandit -r Helper_Scripts/Testing-related/acp_certification_smoke.py -f json -o /tmp/bandit_acp_goose_hermes.json` -> 0 findings.
  - `git diff --check` -> clean.
- PR #2015 review pass:
  - Added commit and runner-version metadata to the Goose/Hermes compatibility-matrix evidence.
  - Removed local absolute paths from task verification notes.
  - Generalized the runtime placeholder error text and routed that new failure path through Loguru.
  - Added no-`sessionId` regression coverage and made the mocked protocol version numeric.
  - Review-fix verification:
    - `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile hermes --run` -> exit 0.
    - `python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile goose --run` -> exit 0.
    - `python -m pytest tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py -q` -> 76 passed, 5 warnings.
    - `python -m bandit -r Helper_Scripts/Testing-related/acp_certification_smoke.py -f json -o /tmp/bandit_acp_goose_hermes_review.json` -> 0 findings.
    - `git diff --check` -> clean.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Hermes ACP profile, corrected native ACP certification probe payload shape, recorded Goose and Hermes host-stdio probe results with commit/runner evidence metadata, and added focused regression coverage for the profile manifest, sessionId substitution, missing-sessionId failure path, and Hermes registry entry.
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
