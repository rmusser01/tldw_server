---
id: TASK-13178
title: Negotiate safe Persona stream auth subprotocol for browser clients
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 15:29'
updated_date: '2026-09-05 16:41'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Buddy UAT reaches the Persona WebSocket after the Strict Mode fix, but Chromium rejects the handshake because the server accepts without selecting an offered subprotocol. Browser bearer-subprotocol authentication must complete without echoing credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authenticated browser-style bearer and API-key protocol offers receive only the offered safe auth marker in the handshake.
- [x] #2 Invalid authentication is still rejected and header/cookie clients without protocol offers retain existing behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce missing protocol selection at the real Persona stream handshake with focused authenticated endpoint tests.
2. Select only a recognized offered auth marker when accepting the existing WebSocket stream; preserve authentication and origin validation.
3. Run focused regression/auth tests, touched-code lint, and Bandit; coordinating agent performs browser retest.
ADR required: no
ADR path: N/A
Reason: protocol interoperability fix using the existing authenticated stream contract; no new authentication policy or boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Persona now selects an offered bearer auth marker after existing authentication succeeds, preserving offered casing and never reflecting the following credential. The existing WebSocketStream helper recognizes the already-accepted connection, so lifecycle metrics and payloads are unchanged. Unrecognized protocol offers and clients without offers preserve existing behavior; all invalid credential variants still close with 4401 before acceptance.

Changed tldw_Server_API/app/api/v1/endpoints/persona.py and tldw_Server_API/tests/Persona/test_persona_ws_auth.py. ADR required: no; existing authentication and streaming contract preserved.

Red: authenticated bearer and Bearer offers returned accepted_subprotocol=None (2 failed, 11 passed). Green: focused auth plus live-control API suites 54 passed (4 existing warnings). Production Bandit completed with zero findings. Combined endpoint/test scan differs from HEAD only by two additional test assertions (B101); existing test-only B105 synthetic JWT fixture unchanged. Ruff reports exactly the same three HEAD findings (two import-order findings and SIM114 at persona.py:2433); no new lint findings. git diff --check clean.

Coordinating agent notified to restart backend port 9101 for real Chromium UAT; no runtime changes, browser actions, full suite, or commit performed.

Real Chromium now selects the authenticated stream protocol, sends a benign greeting, and receives notice plus tool_plan. Final browser rerun after all frontend lifecycle repairs also passed; no plan approved or executed. Root verification54 targeted backend tests and production Bandit0 findings. Evidence in final-live-browser.json and UAT report.

Coordinated final validation: 265 focused frontend tests, 54 backend tests, production Bandit0 findings, scoped frontend ESLint0 errors (warnings documented), unchanged Python lint baseline, real browser evidence and limitations recorded in Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md. Repository-wide typechecking remains limited by80 diagnostics across6 unchanged unrelated files; no full suite run.

Qodo review: added pytest.MonkeyPatch, marker and credential parameter types, None return type, and safe-negotiation docstring to modified WebSocket test. Reverified54 backend tests passing. Production endpoint unchanged by this review repair.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Authenticated Persona streams now negotiate only the safe offered bearer marker so Chromium completes the handshake. Invalid credentials remain rejected; 54 focused tests pass. Browser UAT pending coordinator restart.
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
