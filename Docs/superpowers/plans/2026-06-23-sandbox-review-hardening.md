# Sandbox Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the Sandbox review findings and fix every finding that reproduces against current code.

**Architecture:** Keep fixes local to the affected Sandbox components. Add narrow regression tests before production changes, favoring existing test files and helper patterns.

**Tech Stack:** Python, pytest, FastAPI-side Sandbox core, Docker/Lima/Firecracker runner helpers.

---

## Stage 1: Validate Findings

**Goal**: Confirm each review item against current code and existing tests.
**Success Criteria**: Each finding is marked valid or rejected with code-based rationale.
**Tests**: Read and run focused pytest targets before edits where existing tests already cover behavior.
**Status**: Complete

- [x] Check snapshot restore behavior for symlink workspace roots.
- [x] Check Docker hardening fallback behavior.
- [x] Check Docker allowlist egress failure handling.
- [x] Check Docker command logging for env leakage.
- [x] Check artifact storage symlink handling.
- [x] Check Lima status handling for non-zero commands.
- [x] Check Lima and Firecracker env-file quoting.
- [x] Check service runtime dispatch duplication and decide whether it is behavior-affecting.

## Stage 2: Add Regression Tests

**Goal**: Reproduce validated security/correctness findings with focused tests.
**Success Criteria**: New tests fail for the expected reason before production changes.
**Tests**: Targeted pytest invocations for each added test file/test case.
**Status**: Complete

- [x] Add a snapshot restore test that fails when the workspace root is a symlink.
- [x] Add Docker runner tests for security-option fallback and env redaction.
- [x] Add Docker egress setup tests that fail when rule application fails.
- [x] Add artifact store/path tests for symlink escape prevention.
- [x] Add Lima runner test proving non-zero commands still write status.
- [x] Add Lima/Firecracker env-file tests for shell-safe quoting and key validation.

## Stage 3: Implement Fixes

**Goal**: Patch the minimal production code needed to pass the regression tests.
**Success Criteria**: Targeted tests pass without weakening existing Sandbox behavior.
**Tests**: Same targeted pytest invocations from Stage 2.
**Status**: Complete

- [x] Reject symlink workspace roots before snapshot restore backup/clear.
- [x] Fail closed when configured Docker security options cannot be applied.
- [x] Fail closed when Docker allowlist egress rules cannot be installed.
- [x] Redact Docker env values from logs.
- [x] Guard artifact storage reads/writes against symlink escapes.
- [x] Ensure Lima entry scripts always write run status for non-zero commands.
- [x] Shell-quote Lima/Firecracker env values and reject invalid env keys.

## Stage 4: Verification and Task Finalization

**Goal**: Verify focused tests, Bandit touched scope, and task metadata.
**Success Criteria**: Test output and Bandit results are recorded in TASK-2420.
**Tests**: Focused pytest targets plus `python -m bandit -r` on touched Sandbox paths.
**Status**: Complete

- [x] Run focused Sandbox regression tests.
- [x] Run Bandit on touched Sandbox code.
- [x] Update TASK-9929 with validated/rejected findings, test results, and final summary.
