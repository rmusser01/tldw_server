---
id: TASK-15677
title: Validate VN asset references for runtime manifests and approval
status: Done
labels:
- vn-assets
- runtime
- integrity
references:
- https://github.com/rmusser01/tldw_server/issues/2024
- Docs/ADR/020-db-management-per-user-paths-and-content-backend.md
- Docs/Design/2026-11-26-vn-asset-runtime-integrity.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved-manifest integrity slice of GitHub #2024. Validate owner-local stored assets when approving candidates and reading runtime manifests or readiness. Preserve the v1 manifest shape and the portability backup workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Single and bulk approval reject unavailable or invalid assets with per-item diagnostics before mutating review state.
- [x] #2 The v1 manifest and readiness API detect broken approved file references and report actionable issues. Existing approved-only filtering and successful manifest shape remain compatible.
- [x] #3 Tests cover legacy metadata, missing and incorrect files, owner isolation, mixed bulk review, readiness, and draft backup compatibility.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base commit 57cef16ea5. ADR required: no. Use the existing v1 manifest and owner-local content path contract under ADR-020. GitHub issue: https://github.com/rmusser01/tldw_server/issues/2024
Runtime approval and manifest reads share the content resolver. The review transaction compares the checked generated-file ID and fallback MIME type before updating state, including during mixed bulk approval.

Readiness uses one captured set of slots and items for both coverage and file checks. A required slot must contribute a v1 manifest entry. Portability backup behavior is covered by the existing export tests.

All 265 tests in the VN asset API, manifest, readiness, repository, storage, and portability regression run passed. OpenAPI fingerprint and path checks, scoped Ruff, Bandit, pre-commit, and independent review passed. Bandit reported zero findings. The file checks cover owned registration, media metadata, path containment, and recorded byte count at the VN asset API boundary.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Single and bulk approval reject invalid or changed file references. The v1 manifest returns per-item diagnostics for unavailable approved assets, and readiness detects those issues and missing required manifest entries.
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
