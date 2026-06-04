---
id: TASK-2248
title: Backfill Security request-edge middleware ADR
status: Done
dependencies:
- TASK-2247
labels:
- docs
- process
- adr
- security
modified_files:
- Docs/ADR/019-security-request-edge-middleware.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- tldw_Server_API/app/core/Security/README.md
- backlog/tasks/task-2248 - Backfill-Security-request-edge-middleware-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill the first bounded Security ADR from TASK-2247 evidence. Scope the accepted decision to request-edge Security middleware only: normal startup installs setup access guard/CSP and security headers, RequestIDMiddleware and DrainGateMiddleware are always installed, CSP is path-sensitive, production defaults security headers on when ENABLE_SECURITY_HEADERS is absent, and caveats are explicit for test mode, security-header disablement, HSTS opt-in/HTTPS behavior, and relaxed Setup CSP/eval defaults. Do not include outbound egress or secret/serialization policy in this ADR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create the next accepted ADR under `Docs/ADR/` for Security request-edge middleware using the standard ADR template and TASK-2247 evidence.
- [x] #2 Keep accepted claims scoped to request-edge middleware startup wiring, request IDs, drain gate, setup guard/CSP, security headers, and documented caveats.
- [x] #3 Update `Docs/ADR/README.md`, the INV-029 inventory row, and relevant Security README backlink after ADR creation.
- [x] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use TASK-2247's confirmation audit as the evidence boundary. Create ADR-019 unless another ADR number has appeared on dev. Keep egress/SSRF and secrets/serialization out of scope except as alternatives/follow-up. Update the ADR index, inventory row, and `tldw_Server_API/app/core/Security/README.md` backlink. Run Markdown/link checks and targeted Security middleware tests before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/ADR/019-security-request-edge-middleware.md` as the accepted, bounded ADR for request-edge Security middleware using TASK-2247 evidence.
- Scoped ADR-019 to startup middleware wiring, setup access guard/CSP, request IDs, drain gate, security headers, path-scoped CSP, production-default security headers, and explicit caveats for test mode, header disablement, HSTS, and setup CSP/eval defaults.
- Left outbound egress/SSRF policy, SecretManager adoption, and serialization policy out of the accepted decision except as alternatives/follow-up.
- Updated `Docs/ADR/README.md`, INV-029 in `Docs/ADR/inventory/2026-06-03-decision-inventory.md`, and `tldw_Server_API/app/core/Security/README.md` to link the new request-edge ADR.
- Verification:
  - `git diff --check` exited 0.
  - `rg -n "ADR-019|019-security-request-edge|TASK-2248|INV-029|Architecture decision" Docs/ADR/019-security-request-edge-middleware.md Docs/ADR/README.md Docs/ADR/inventory/2026-06-03-decision-inventory.md tldw_Server_API/app/core/Security/README.md "backlog/tasks/task-2248 - Backfill-Security-request-edge-middleware-ADR.md"` exited 0 and found the ADR/index/inventory/backlink/task references.
  - `backlog task TASK-2248 --plain` exited 0.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Security/test_security_headers_middleware.py tldw_Server_API/tests/Security/test_request_id_middleware.py tldw_Server_API/tests/Security/test_setup_access_guard.py tldw_Server_API/tests/Security/test_setup_csp_eval_policy.py` passed with 27 passed, 6 warnings.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Services/test_drain_gate_middleware.py` passed with 25 passed, 6 warnings.
- Bandit: skipped because the touched implementation scope is Markdown documentation and Backlog.md task metadata only; no Python/code paths were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled ADR-019 for Security request-edge middleware, updated ADR/inventory/Security README links, and recorded verification. Remaining Security areas, including outbound egress/SSRF and secrets/serialization policy, stay outside this ADR for separate bounded review if needed.
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
