---
id: TASK-413
title: Implement Persona Visual provider archive retrieval adapter
status: Done
labels:
- persona
- persona-visual
- mcp
- backend
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1796
- https://github.com/rmusser01/tldw_server/pull/1792
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend-only Persona Visual provider archive retrieval/materialization slice tracked by GitHub issue #1796. Convert validated provider archive handoff descriptors from PR #1792 into local archive import-preview job input through a bounded retrieval adapter, without committing imports, activating packs, changing renderers, adding WebUI, or touching Buddy animation/VN behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider archive handoff descriptors can be materialized into a local archive input without committing or activating packs.
- [x] #2 The adapter fails closed for missing resource handles, unsupported media types, oversized payloads, checksum mismatch, unsafe paths/URIs, and retrieval failures.
- [x] #3 Existing import-preview job payload conventions remain the source of truth after materialization.
- [x] #4 Trace/log/output data remains bounded and does not expose raw provider payloads, local temp paths, secrets, or archive contents.
- [x] #5 Focused backend tests cover valid materialization and fail-closed cases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect merged provider handoff helper and existing Persona Visual import-preview job payload conventions.
2. Add focused RED tests for valid materialization and fail-closed retrieval/checksum/size/media/path cases.
3. Implement a bounded pure retrieval/materialization adapter around an injectable resource fetcher/writer abstraction.
4. Update docs/task notes with boundary and validation results.
5. Run focused pytest, py_compile, git diff --check, and Bandit on touched Python scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented `materialize_provider_archive_import_preview_handoff()` in
`provider_archive_retrieval.py`. The helper accepts a ready provider archive
handoff plus an injected resource reader, writes bounded byte chunks into local
import-preview staging, validates MCP resource handles/media type/checksum/size,
deletes partial files on failure, and returns the existing
`build_visual_pack_import_preview_payload()` shape without creating preview
rows, enqueueing Jobs, committing imports, activating packs, changing renderers,
or exposing raw provider payloads in diagnostics.

RED verification: provider archive retrieval pytest failed on missing module
before implementation.

GREEN verification: provider archive retrieval pytest passed; focused provider
envelope and visual jobs regressions passed with 50 total tests. py_compile
passed for the new module/test. git diff --check passed. Bandit wrote
`/tmp/bandit_persona_provider_archive_retrieval.json` with zero findings after
replacing assert statements with casts.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the backend-only Persona Visual provider archive retrieval/materialization adapter. Ready provider archive handoff descriptors can now be converted into local archive import-preview job payloads through an injected resource reader with fail-closed validation for blocked handoffs, unsafe resource handles, invalid media/checksum metadata, non-byte resources, oversized resources, checksum mismatch, retrieval failures, and staging write failures. Docs now record that endpoint/worker orchestration remains a separate follow-up slice.
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
