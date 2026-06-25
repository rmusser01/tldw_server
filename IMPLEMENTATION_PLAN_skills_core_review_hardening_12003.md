# Skills Core Review Hardening Implementation Plan

## Stage 1: Regression Coverage
**Goal**: Add focused tests for the validated Skills review findings before changing implementation.
**Success Criteria**: Tests fail against current code for unsafe file/registry mutation ordering, fork skills with no allowed tools, oversized zip import payloads, supporting-file-only versioning/I/O failure handling, and async context integration.
**Tests**: `tldw_Server_API/tests/Skills/unit/test_skills_service.py`, `tldw_Server_API/tests/Skills/unit/test_skill_executor.py`, `tldw_Server_API/tests/Skills/integration/test_skills_api.py`.
**Status**: Complete

## Stage 2: File And Registry Consistency
**Goal**: Make update/delete operations avoid publishing filesystem changes when optimistic registry writes conflict or fail.
**Success Criteria**: SKILL.md updates are staged and restored on registry conflicts, deletes mark the registry before removing files with safe rollback behavior, supporting-file changes bump skill versions, and supporting-file write/delete failures raise domain errors.
**Tests**: Focused Skills service tests pass.
**Status**: Complete

## Stage 3: Tool And Zip Safety
**Goal**: Deny fork skill tool access by default and reject abusive zip imports before unbounded reads.
**Success Criteria**: Fork skills without `allowed-tools` advertise no tools and do not execute tool calls; zip imports enforce entry count, SKILL.md size, supporting-file size, and aggregate size guards using ZipInfo before reads.
**Tests**: Focused executor and zip import tests pass.
**Status**: Complete

## Stage 4: Async Context Integration
**Goal**: Keep async chat request paths off synchronous Skills registry/filesystem scans.
**Success Criteria**: Async helpers use `get_context_payload_async()` for tool injection and system-message context, and chat endpoint callers await those helpers.
**Tests**: Focused Skills API/chat integration tests pass.
**Status**: Complete

## Stage 5: Verification And Closeout
**Goal**: Verify the touched scope and record task evidence.
**Success Criteria**: Focused pytest suites pass, compile check passes for touched Python files, Bandit scans touched production paths, and Backlog task records final files, verification, and summary.
**Tests**: Focused Skills tests plus `python -m compileall` and `python -m bandit -r` on touched production paths.
**Status**: Complete
