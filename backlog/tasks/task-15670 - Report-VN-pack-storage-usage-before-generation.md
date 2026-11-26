---
id: TASK-15670
title: Report VN pack storage usage before generation
status: Done
labels:
- vn-assets
- storage
- webui
references:
- https://github.com/rmusser01/tldw_server/issues/2025
- Docs/ADR/064-usage-quotas-per-user-limits.md
- Docs/Design/2026-11-26-vn-pack-storage-preflight.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the unblocked storage usage and bounded estimate acceptance criterion in GitHub #2025. Reuse owner-scoped generated-file metadata and the existing quota service. Show the result in generation preflight and the workbench.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner sees current registered and approved storage for the selected pack, with no cross-owner attribution.
- [x] #2 Generation preflight reports an advisory size estimate where dimensions are known and identifies variants it cannot estimate.
- [x] #3 The workbench shows storage usage and any estimate-based quota warning, while generation remains governed by existing runtime quota enforcement.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base: remote dev ca6f793617. ADR required: no. ADR-064 governs the existing quota read and enforcement. This adds advisory data to the existing owner-scoped preflight.
The Generation monitor reports registered and approved file bytes for the selected pack and a default all-slot estimate from supported requested dimensions. The shared Storage quota check applies to complete estimates. Generation still enforces quota on the actual written bytes.

Owner and VN source attribution use the generated-file repository. The server keeps the synchronous pack and configuration reads on the existing threadpool boundary. The workbench rejects storage results from earlier account verification and refreshes after generation, review, and successful principal revalidation.

Verification passed for 35 focused Python tests, 295 WebUI tests, and the two Chromium smoke cases at 1440px and 390px with a mocked API. OpenAPI fingerprint and path union, WebUI typecheck, scoped ESLint and Ruff, and Bandit passed. Bandit reported zero findings.

Two workbench recovery cases timed out on both this branch and the unchanged dev version and were excluded from the focused WebUI run. Broader Python runs encountered blocked external DNS in the tokenizer fixture. Retention, orphan detection and attributable cleanup remain under GitHub #2025.
Rebased the unchanged storage patch onto the latest remote dev ac98eff0d5. Reran all 35 preflight tests and 8 focused Generation monitor and Workbench cases successfully.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The owner can see the selected pack’s registered and approved storage, its default generation estimate, and an advisory quota warning in the workbench.
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
