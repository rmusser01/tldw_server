---
id: TASK-304.13
title: Implement Research Studio capability health contract
status: Done
assignee: []
created_date: '2026-05-13 01:46'
updated_date: '2026-05-13 03:40'
labels:
  - implementation
  - research-studio
  - webui
  - backend
  - verification
dependencies:
  - TASK-304.12
documentation:
  - >-
    Docs/superpowers/specs/2026-05-13-research-studio-capability-health-contract-design.md
  - >-
    Docs/superpowers/plans/2026-05-13-research-studio-capability-health-contract-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend exposes authenticated, permission-gated, rate-limited Research Studio capability endpoint with stable status/mode semantics.
- [x] #2 Frontend consumes the capability endpoint and gates chat, text artifacts, slides, audio summary, export/download, and sync/share at action boundaries.
- [x] #3 Tests cover backend derivation, endpoint auth/schema, frontend helper behavior, and UI allow/warn/block states.
- [x] #4 Docs and PR notes record authenticated CDP checks and local/manual real summary generation using existing saved LLM credentials.
- [x] #5 Bandit and focused frontend/backend tests pass or skips are explicitly documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Frontend capability gating implemented. Sequential Vitest evidence: research-studio-capabilities 6 passed; WorkspacePlayground.stage2.responsive 13 passed; ChatPane.stage1 12 passed; StudioPane.stage3 22 passed. Combined three-file Vitest run hit parallel per-test timeouts, then each modified file passed when run individually.

Verification update:
- Local saved config: single-user API key present (length recorded only) and 8 provider key variables present; provider list showed openai/gpt-4o-mini available.
- Live backend on 127.0.0.1:18002: authenticated /api/v1/health returned 200 status ok; authenticated /api/v1/research-studio/capabilities returned 200 in 25 ms with all seven capability IDs.
- Added config-only TTS capability collection after live endpoint verification exposed the setup health path initializing/downloading TTS provider assets. Regression test added.
- Opt-in CDP/Playwright live spec passed: TLDW_RESEARCH_STUDIO_LIVE_GENERATION=1 with local saved auth, openai provider, gpt-4o-mini model; 2 tests passed; generated real Summary artifact from a deterministic document source; outputCharacterCount=357; screenshot=/private/tmp/research-studio-live-summary-research-studio-summary-1778643186763-r8aqmx.png.
- Focused backend pytest: Research_Studio derivation+endpoint 12 passed; OpenAPI contract 5 passed.
- Focused frontend Vitest: research-studio-capabilities 6 passed; WorkspacePlayground.stage2 13 passed; StudioPane.stage3 22 passed; ChatPane.stage1 12 passed; frontend route/readiness tests 9 passed.
- Bandit touched backend scope: 0 findings in /tmp/bandit_research_studio_capabilities.json.
- git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Research Studio capability health follow-up for PR #1616. The backend now exposes a sanitized authenticated capability contract for Research Studio, including config-only TTS readiness so capability checks do not initialize or download provider assets. The WebUI consumes that contract to gate read-only source browsing, chat/generation, export/download, sync/share, slides, and audio summary at action boundaries while preserving degraded-health entry into Research Studio.

Added focused backend schema/derivation/endpoint coverage, frontend helper and UI gating coverage, and an opt-in manual Playwright/CDP workflow that uses the locally saved API key/provider credentials to seed a deterministic document and generate a real Summary artifact through openai/gpt-4o-mini. Verification recorded: Research Studio backend tests 12 passed, OpenAPI contract tests 5 passed, focused frontend Vitest suites passed, route/readiness tests passed, Bandit reported 0 findings on touched backend scope, git diff --check passed, and the manual live generation spec passed with a 357-character generated Summary artifact. Known deferred scope remains capability-specific policy hardening beyond the implemented current contract semantics; no current skips or blockers are left for this PR slice.
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
