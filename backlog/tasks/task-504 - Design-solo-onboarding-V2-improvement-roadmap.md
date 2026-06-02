---
id: TASK-504
title: Design solo onboarding V2 improvement roadmap
status: Done
labels:
- onboarding
- webui
- uat
- setup
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2214
documentation:
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
modified_files:
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the umbrella design for four solo onboarding improvements: repeatable UAT harness, guided diagnostics and recovery, first-value starter questions, and local model guided alternative V2. The harness should be built first and each later feature PR should extend the harness before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Umbrella design defines the four sequential PRs and their ordering, with UAT harness first.
- [x] #2 UAT harness design uses real backend, real WebUI, and repo `mock_openai_server` rather than Playwright provider route mocks.
- [x] #3 Harness plan includes Tier A scenario matrix, isolated runtime profile, artifacts, and pass/fail quality gates.
- [x] #4 Later diagnostics, starter-question, and local-model PR plans each specify harness-first scenarios and product acceptance criteria.
- [x] #5 Design records non-goals, state/data boundaries, implementation sequencing, and verification expectations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created the umbrella roadmap design at `Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md`.
- Design decisions captured from brainstorming:
  - One umbrella spec with four sequential PR plans.
  - PR1 builds a manual/dev UAT harness first.
  - Harness default fidelity is real backend + real WebUI + repo `mock_openai_server`.
  - Provider behavior should come from static mock-server config files per scenario, not Playwright route mocks.
  - First-source web URL path should ingest a local public fixture page.
  - First-source fixture should be a short structured research note.
  - Each run should use an isolated temp runtime profile.
  - Passing UAT requires functional assertions plus quality gates.
  - Later feature PRs are harness-first: add/update scenario, then implement product behavior.
- Local spec review completed after subagent dispatch was unavailable under current tool restrictions:
  - Confirmed the spec defines all four PRs and preserves UAT-first sequencing.
  - Confirmed the UAT harness uses real backend, real WebUI, and repo `mock_openai_server` provider behavior.
  - Confirmed static mock-server configs, local web fixture content, isolated runtime profiles, and artifact requirements are captured.
  - Clarified Tier A as the required manual run and future Tier B/CI promotion as separate work.
  - Clarified that hosted-provider validation must not pretend a live mock-server preflight occurred if the backend only performed local syntax validation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the umbrella Solo Onboarding V2 roadmap design. The spec defines four sequential PRs: repeatable onboarding UAT harness, guided diagnostics and recovery, first-value starter questions, and local model guided alternative V2. PR1 establishes a manual/dev harness using real backend, real WebUI, repo `mock_openai_server`, isolated temp runtime profiles, local first-source fixtures, screenshots, JSON summary, and backend/frontend/mock logs. PR2-PR4 are harness-first feature plans that must add or update UAT scenarios before product implementation.
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
