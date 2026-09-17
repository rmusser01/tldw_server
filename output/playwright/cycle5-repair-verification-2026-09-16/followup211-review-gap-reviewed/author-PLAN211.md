# TASK13260.149 — saved review gap display

Parent approved DESIGN211 option using existing date parser/ICU, including valid zero-day legacy review responses. Five production files: three consumers, date-display utility, English resource. No scheduler, request, state mutation, browser/native-card action.

## Stage 1 — permanent RED
Goal: retain actual mounted learning/relearning/legacy display failures; existing day-scale controls.
Tests: ReviewTab toast, Manage compact/expanded, EditDrawer metadata/reset content, pure display cases.
Status: Complete

## Stage 2 — minimal GREEN
Goal: shared pure formatter; stable saved timestamps for learning/relearning/zero-day legacy; known review days unchanged; explicit unavailable.
Tests: numeric ICU resource/fallback and mounted boundaries, same-instance refresh, rating/practice payload controls.
Status: Complete

## Stage 3 — validate/freeze
Goal: adjacent suites, lint/compiler baseline, Bandit limitations, unchanged-source baseline replay and independent handoff.
Status: Complete

Author implementation and verification complete. Independent review and parent-owned native acceptance remain pending. No commit or task/tracker mutation by author.
