---
id: TASK-313
title: Address PR 1624 Manifest V2 review comments
status: Done
assignee: []
created_date: '2026-05-13 05:36'
updated_date: '2026-05-13 05:37'
labels:
  - persona
  - buddy
  - visual-packs
  - docs
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1624'
  - 'https://github.com/rmusser01/tldw_server/issues/1623'
documentation:
  - Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR 1624: align capability field names with the existing visual-renderers API, prohibit embedded manifest data, clarify common versus renderer-specific asset roles, and define V2 activation state requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capability field names preserve the existing visual-renderers API shape and only add new fields additively.
- [x] #2 Manifest security rules explicitly reject embedded data such as base64 or Data URIs.
- [x] #3 Renderer asset roles distinguish literal common roles from renderer-specific roles that satisfy common categories.
- [x] #4 Activation validation defines minimum required V2 visual states.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR 1624 review feedback in Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md. Verified Qodo's capability-field mismatch against the existing API design spec, backend test, and WebUI type, then changed the design to preserve manifest_versions, buddy_runtime_supported, import_supported, and export_supported with additive V2 fields only. Added explicit embedded data rejection, clarified common role categories versus renderer-specific concrete roles, and defined the V2 activation state baseline. Verification: git diff --check passed; rg confirmed the conflicting field names are absent. Tests and Bandit skipped because this is docs-only plus Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR 1624 review comments by aligning the Manifest V2 capability contract with the existing visual-renderers API, adding embedded data prohibitions, clarifying asset role categories, and defining V2 activation state requirements.
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
