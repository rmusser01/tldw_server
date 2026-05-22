---
id: TASK-45.44.10.1
title: Migrate DocumentPickerModal alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- document-workspace
priority: medium
parent_task_id: TASK-45.44.10
references:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentPickerModal.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
- https://github.com/rmusser01/tldw_server/pull/1950
modified_files:
- apps/packages/ui/src/components/DocumentWorkspace/DocumentPickerModal.tsx
- apps/packages/ui/src/components/DocumentWorkspace/__tests__/DocumentPickerModal.design-system.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Document and Workspace product-state design-system migration by replacing DocumentPickerModal's remaining AntD Alert product-state surfaces with the shared design-system Alert primitive while preserving offline, upload-warning, and error behavior. Keep scope limited to this modal, focused tests, and removal of matching baseline entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] DocumentPickerModal no longer imports AntD Alert for product-state UI.
- [x] Offline server-required, upload storage warning, and error banners render through the shared design-system Alert primitive.
- [x] Upload-warning Open in Media action behavior is preserved.
- [x] The three matching DocumentPickerModal AntD Alert baseline entries are removed without introducing new unbaselined product-state findings.
- [x] Focused tests and the design-system product-state verifier pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused tests that prove DocumentPickerModal alert states render with `data-ds-component="Alert"` and preserve upload-warning action behavior.
2. Replace the three AntD Alert usages with the shared design-system Alert primitive, keeping AntD mechanics otherwise unchanged.
3. Remove the migrated DocumentPickerModal baseline rows.
4. Run focused component tests, product-state guard/unit verifier checks, baseline JSON parse, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red test evidence: the new focused DocumentPickerModal design-system test failed while the modal still rendered AntD Alert wrappers because each alert message lacked a `data-ds-component="Alert"` ancestor.
- Migrated only the offline server-required, upload-storage warning, and error banners to `components/ui/primitives/Alert`; kept AntD Modal, Tabs, Button, List, Empty, Spin, Tag, and Switch mechanics unchanged.
- Removed the three matching DocumentPickerModal AntD Alert rows from `design-system-product-state-baseline.json`.
- Full UI TypeScript check still fails on inherited repo-wide debt outside the touched files; no diagnostics referenced DocumentPickerModal or its new focused test.
- PR review follow-up: removed the unused AntD Alert mock entry from the focused design-system test so the mock surface matches the migrated component behavior.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated DocumentPickerModal's three product-state banners from AntD Alert to the shared design-system Alert primitive while preserving the upload-warning Open in Media action. Added focused regression coverage for offline, unsupported upload, and storage-warning states, removed the three migrated baseline entries, and removed the unused AntD Alert test mock after review. PR: https://github.com/rmusser01/tldw_server/pull/1950. Verification: red focused test failed on missing design-system Alert markers; green focused DocumentPickerModal test passed 3/3; product-state guard unit test passed 54/54; `bun run verify:design-system-state` passed with baseline exceptions reduced from 303 to 300 and Document/Workspace exceptions reduced to 9; review follow-up focused test passed 3/3; baseline JSON parse passed; `git diff --check` passed. Bandit skipped because this slice touches only TypeScript/TSX, JSON, and Backlog metadata.
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
