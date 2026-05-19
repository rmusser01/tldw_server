---
id: TASK-155
title: Write VN Play setup options design spec
status: Done
assignee: []
created_date: '2026-05-09 05:17'
updated_date: '2026-05-09 05:27'
labels:
  - vn-play
  - design
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1407'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and commit the design spec for GitHub issue #1407. The spec must define a backend setup-options endpoint for VN Play session setup, warn-but-allow asset pack selection, high-risk acknowledgement behavior, minimal character selector metadata, bounded pack/readiness pagination, frontend fallback/manual ID behavior, and implementation/testing guidance. This task only covers the design/spec artifact, not implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec document is created under Docs/superpowers/specs with decisions from issue #1407
- [x] #2 Spec defines setup-options API contract, warning severities, acknowledgement rules, pagination, frontend data flow, error states, and tests
- [x] #3 Spec is reviewed for obvious correctness issues and committed on the design branch
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-05-09-vn-play-setup-options-design.md. Local review tightened naming for readiness_warnings/readiness_errors, content-rating comparison severity, pack sorting, and no image_base64 character queries. Verification: git diff --check produced no output. Spec-review subagent was not dispatched because this Codex session only permits subagents when the user explicitly requests them; performed local self-review instead.

Reopened for design-review fixes requested after initial spec commit. Addressing bounded pack pagination, acknowledgement persistence, trust warning source, selected-character pagination, and page-scoped empty-state semantics.

Review fixes applied: setup spec now requires repository/service-level bounded pack pagination before readiness fanout, persisted WebUI acknowledgement metadata for accepted high-risk warnings, import-journal-derived trust provenance, selected_character preservation across pagination/search, scoped empty states, and consistent pagination metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote and tightened the VN Play setup-options design spec for issue #1407. The spec now defines the aggregate setup endpoint, bounded selector pagination before readiness fanout, minimal character and selected_character metadata, warning severities, high-risk acknowledgement persistence, import-journal trust provenance, scoped empty/error states, frontend fallback behavior, rollout compatibility, risks, and backend/frontend test expectations. Verification: git diff --check passed. Bandit is not applicable because this task only changes markdown/task metadata.
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
