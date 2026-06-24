# Flashcards Core Review Fixes Implementation Plan

## Stage 1: Regression Coverage
**Goal**: Pin the reviewed failure modes before changing implementation.
**Success Criteria**: Focused tests fail for oversized APKG import, oversized data URI export, empty export, skipped-row asset cleanup, and missing assistant context.
**Tests**: `tests/Flashcards/test_apkg_importer.py`, `tests/Flashcards/test_apkg_exporter.py`, `tests/Flashcards/test_study_assistant_service.py`.
**Status**: Complete

## Stage 2: APKG Import Hardening
**Goal**: Reject abusive APKG archives before unbounded decompression or all-note materialization, and avoid persisting assets for rows that are later skipped.
**Success Criteria**: Archive caps are enforced, notes are queried with a limit, mapped media is size-checked before reads, and failed/skipped row media does not call the endpoint asset importer.
**Tests**: Focused importer tests pass.
**Status**: Complete

## Stage 3: APKG Export Hardening
**Goal**: Make media-size limits authoritative for data URIs, return a controlled empty-export error, and remove dead exporter code.
**Success Criteria**: Oversized data URIs raise `ValueError`, empty rows raise `ValueError`, and existing APKG exporter tests keep passing.
**Tests**: Focused exporter tests pass.
**Status**: Complete

## Stage 4: Assistant Prompt Grounding
**Goal**: Send bounded card/question context and history to the LLM instead of only the front/anchor text.
**Success Criteria**: Prompt package includes front, back, notes/extra, citations, and recent history for flashcards; quiz context includes answer/explanation details.
**Tests**: Focused study assistant tests pass.
**Status**: Complete

## Stage 5: Verification And Closeout
**Goal**: Validate the touched scope and record results.
**Success Criteria**: Focused pytest, py_compile, and Bandit complete; Backlog task records touched files, verification, and summary.
**Tests**: Focused tests plus `python -m py_compile` and `python -m bandit -r` on touched files.
**Status**: Complete
