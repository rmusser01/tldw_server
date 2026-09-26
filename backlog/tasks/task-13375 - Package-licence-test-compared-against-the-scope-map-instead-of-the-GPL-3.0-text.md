---
id: TASK-13375
title: >-
  Package licence test compared against the scope map instead of the GPL-3.0
  text
status: Done
assignee: []
created_date: '2026-09-26 14:04'
updated_date: '2026-09-26 14:05'
labels:
  - licensing
  - mcp
  - packaging
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Renumbered from a colliding TASK-13356 (dev already has TASK-13356, the ADR inventory refresh), and corrected.

The original filing claimed apps/mcp-unified shipped GPL-3.0 text while the project was GPL-2.0. That premise was wrong. The root LICENSE is a multi-licence scope map: tldw_Server_API/** and all unlisted repository material -- which includes apps/mcp-unified -- are GPL-3.0-only, and both pyproject.toml files declare GPL-3.0-only. Shipping GPL-3.0 is correct.

The real defect was the test: test_mcp_unified_package_license_file_is_local_to_project asserted the package LICENSE was byte-identical to the root LICENSE, which stopped being a licence text when it became the scope map, so it could never pass. The package also carried a reformatted GPL-3.0 copy rather than the verbatim FSF text.

Fixed in PR #3007: the package LICENSE is now the verbatim LICENSES/GPL-3.0-only.txt, and the test compares against that file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The package ships the verbatim LICENSES/GPL-3.0-only.txt
- [ ] #2 The licence test compares against LICENSES/GPL-3.0-only.txt, not the root scope map
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
