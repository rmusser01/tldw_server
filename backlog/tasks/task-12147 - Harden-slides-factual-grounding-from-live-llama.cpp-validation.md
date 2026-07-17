---
id: TASK-12147
title: Harden slides factual grounding from live llama.cpp validation
status: Done
references:
- 'GitHub issue #2605'
- 'PR #2633'
- TASK-12146
modified_files:
- tldw_Server_API/app/api/v1/endpoints/slides.py
- tldw_Server_API/app/core/Slides/slides_generator.py
- tldw_Server_API/tests/Slides/test_slides_api.py
- tldw_Server_API/tests/Slides/test_slides_generator.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from live issue #2605 validation using the local llama.cpp server at 127.0.0.1:9099. Flashcards, quizzes, audio summaries, data tables, and mindmaps passed internal claim verification, but generated slides were real Marp markdown while still containing unsupported recommendations/status/methodology claims and were rejected by the verifier. Tighten the slide generation path and rerun focused and live validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Slide generation prompt forbids unsupported factual claims and avoids forced short-source conclusion slides.
- [x] Slide claim verification uses explicit visible slide claims without splitting abbreviations such as `Dr.`.
- [x] Title-slide cover text remains in the report text but is not treated as a factual claim.
- [x] Full live llama.cpp validation passes for flashcards, quizzes, audio summaries, data tables, mindmaps, and slides.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened slide grounding after live llama.cpp validation exposed slide-only failures. The slide generator prompt now explicitly requires source-supported facts, avoids unsupported recommendations/status/methodology/conclusions, permits shorter decks for short sources, and avoids forced conclusion slides. Slide verification now passes explicit visible content/speaker-note claim lines to the internal Claims gate, preserving abbreviations such as Dr. Mira Patel and excluding title-slide cover text from factual claim extraction. Verification: focused red/green prompt and slide-unit tests, full Slides API suite (80 passed), live llama.cpp validation for flashcards/quizzes/audio summary/data table/mindmap/slides with all verdicts grounded, git diff --check, and Bandit 0 findings on touched backend Python files.
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
