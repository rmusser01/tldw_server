---
id: TASK-2380
title: Define canonical Workspace container contract for issue 1988
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-18 00:44
labels: []
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1988
- https://github.com/rmusser01/tldw_server/issues/1984
- https://github.com/rmusser01/tldw_server/pull/2381
modified_files:
- Docs/Design/Workspace_Container_Contract_2026_06.md
- Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md
- Docs/superpowers/plans/2026-06-17-workspace-container-contract.md
- tldw_Server_API/app/core/Workspaces/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the docs-only canonical Workspace container contract for GitHub issue #1988, aligning tldw_server Workspace vocabulary with the tldw_chatbook operating-context model and assigning unresolved implementation questions to Phase 2 child issues.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Contract document exists and is linked from issue #1984 / PR #2381.
- [x] #2 Contract explicitly references the Chatbook workspace operating-context model with a stable GitHub URL.
- [x] #3 Contract states that workspace selection must not hide globally owned resources.
- [x] #4 Contract defines active-context eligibility separately from browse/search visibility.
- [x] #5 Contract names all Phase 2 resource types and their expected membership shape.
- [x] #6 Open questions are resolved or assigned to specific child issues.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the docs-only canonical Workspace container contract for GitHub issue #1988 in `Docs/Design/Workspace_Container_Contract_2026_06.md`, aligned it with the Chatbook operating-context PRD via a stable GitHub URL, linked it from the Workspace core README and existing canonical-model decision record, and opened PR #2381 against `dev`. Review follow-up: replaced the local Chatbook filesystem path, populated task acceptance criteria, and aligned the task DoD checklist with status `Done`. Verification: `git diff --check`; targeted `rg` acceptance check for Chatbook reference, global visibility, active-context eligibility, Phase 2 resource types, transfer policies, runtime binding follow-ups, and assigned child issues. Bandit was not run because no Python/backend code changed.
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
