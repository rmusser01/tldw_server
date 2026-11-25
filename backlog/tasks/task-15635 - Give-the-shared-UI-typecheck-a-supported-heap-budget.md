---
id: TASK-15635
title: Give the shared UI typecheck a supported heap budget
status: In Progress
created_date: 2026-11-24 17:38
labels:
- frontend
- tooling
references:
- https://github.com/rmusser01/tldw_server/pull/3601
documentation:
- Docs/Development/Frontend.md
updated_date: 2026-11-24 17:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the full shared UI TypeScript check repeatable from its own package so it reaches compiler diagnostics without running out of Node heap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared UI exposes a documented Bun typecheck command that uses the heap budget established by the WebUI.
- [x] #2 The command checks the existing source and test project and reaches TypeScript diagnostics without an allocation failure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare the same shared UI compiler project on the upload PR and its exact base.
2. Configure the package command with the verified memory budget and document how to run it.
3. Run the actual entry point on current dev and record the result.
ADR required: no. The command uses the existing Bun, Node and TypeScript toolchain.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The full shared UI compiler uses about 7 GiB on the upload PR and its exact base. Node aborted at a 6 GiB heap. The package now uses the WebUI's 8 GiB allowance when run with Bun.
Verified the new command on dev 518bda3da3 using TypeScript 5.9.3 and Node 26. It checked 9,590 files and reached the same 775 compiler diagnostics as the exact pre-upload base and upload PR. No changed diagnostics were found across those runs. The WebUI typecheck passed on the upload PR.
ADR required: no. The change uses the existing toolchain and compiler project.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
