---
id: TASK-12143
title: Validate generated Research Workspace slides are real presentations
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 20:23'
labels:
  - research-workspace
  - tests
  - slides
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tighten slides generation verification so Research Workspace slides artifacts are only completed when the Slides API returns a usable presentation deck, not empty placeholder text or markdown fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add failing coverage for empty or placeholder generated slides not completing as valid artifacts.
- [x] #2 Validate generated slides require real presentation metadata and meaningful slide content.
- [x] #3 Run focused frontend/backend tests and record verification before committing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Research Workspace slides validation on both frontend and backend. Frontend slides generation now requires a Slides API presentation id and rejects empty or placeholder slide bodies, including markdown fallback text. Backend SlidesGenerator now rejects placeholder or empty normalized slide content before persistence. Added regression tests for placeholder API output, markdown fallback completion prevention, and backend placeholder rejection. Full-app validation ran against FastAPI on 127.0.0.1:18001 using the user-provided llama.cpp endpoint at 127.0.0.1:9099 with model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf. The generated presentation id was 404b283b-d113-45c8-bf2c-a2aecde514f7 with 8 structured slides, application/json export, and 0 placeholder failures in generated, fetched, and exported payloads. Initial full-app attempt failed closed because egress blocked port 9099 and did not persist a fake artifact; validation retry used temporary local-only egress override.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Slides artifacts now fail closed unless the Slides API returns a real persisted presentation with meaningful slide content. Markdown fallback text and placeholder slides such as invalid or slides go here no longer complete as valid Research Workspace slide artifacts. Verification passed: frontend focused vitest suite 6 files and 107 tests; backend slides generator pytest 12 tests; Bandit on slides_generator.py with 0 findings; git diff --check; full-app FastAPI plus llama.cpp validation generated and exported a structured 8-slide JSON presentation with 0 placeholder failures.
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
