---
id: TASK-13627
title: Run WebUI initial setup fixtures in their owning runtime
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI cookie-session and initial-connection tests use WebUI storage prototype semantics but live in the shared native SDK suite. Move them to the frontend test package and keep their authority, lock, readback and draft recovery assertions intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both relocated WebUI suites retain all39 tests and pass with actual WebUI storage aliases
- [x] #2 Native SDK storage/history coverage remains in the shared UI package; both client types and bounded review pass
- [x] #3 Normal exact-file hooks pass and relocation evidence is recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; existing WebUI/native test ownership convention. CI reports14initial-setup and17config regression cases under native shared UI; both unchanged suites pass39/39 in WebUI before relocation. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_webui_setup_tests_13627.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Original and relocated WebUI suites each39/39 passed. Full normalized bodies byte-identical except3import aliases and2owner comments; assertions intact. Required CI discovers new paths; native930coverage retained. Both normal client source type checks exit0 (no dedicated test type check). Bounded review clear; normal hooks next. Review Docs/Reviews/CHAT_WEBUI_SETUP_TEST_OWNERSHIP_13627_2026_10_13.md. Bandit N/A TS test-only.
Normal exact-file hooks pass. Both relocated suites39/39, normalized bodies unchanged, required CI discovery and bounded review clear, normal client source types pass. Bandit N/A TS test-only. Other PR CI/UX/H2 and final independent review/rebase/merge remain open.
Completed owned plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_webui_setup_tests_13627_completed.md; durable evidence remains in review document.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved WebUI-specific initial setup and cookie/session authority fixtures into their owning frontend runtime, changing only3import aliases and2owner comments. All39 cases and assertions retained, both relocated suites pass, normal source types and hooks pass. Native SDK coverage remains shared; bounded review clear.
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
