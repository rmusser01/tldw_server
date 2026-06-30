# Metrics Module Review Fixes Implementation Plan

Backlog task: TASK-2415

## Stage 1: Red Tests
**Goal**: Capture each review finding as a focused failing test.
**Success Criteria**: Tests fail against the current Metrics implementation for raw user labels, gauge eviction, incompatible duplicate registration, and negative counters.
**Tests**: Metrics unit tests for privacy labels, gauge latest series retention, duplicate type safety, and counter validation.
**Status**: Complete

## Stage 2: Registry Semantics
**Goal**: Fix core MetricsRegistry behavior for sensitive labels, gauges, duplicate definitions, and counters.
**Success Criteria**: Public metric labels hash raw user identifiers, latest gauge values are retained independently from the sample ring buffer, compatible registrations are idempotent, incompatible registrations are rejected, and counter decrements are ignored.
**Tests**: Focused Metrics unit tests from Stage 1 pass.
**Status**: Complete

## Stage 3: Legacy Logger Cleanup
**Goal**: Remove the unused Metrics logger_config helper and update module documentation references.
**Success Criteria**: No live references to logger_config remain in Metrics code or README.
**Tests**: Search verification plus focused Metrics tests.
**Status**: Complete

## Stage 4: Verification
**Goal**: Run targeted regression tests and security scan for touched code.
**Success Criteria**: Focused Metrics tests pass and Bandit reports no new findings in the touched Metrics scope.
**Tests**: `python -m pytest -q tldw_Server_API/tests/Metrics ...`; `python -m bandit -r <touched_paths> -f json`.
**Status**: Complete
