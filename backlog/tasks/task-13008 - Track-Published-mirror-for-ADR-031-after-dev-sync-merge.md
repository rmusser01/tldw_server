---
id: TASK-13008
title: Track Published mirror for ADR 031 after dev sync merge
status: Done
created_date: 2026-08-09 00:23
labels:
- docs
- ci
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2774
- https://github.com/rmusser01/tldw_server/pull/2775
modified_files:
- Docs/Published/ADR/031-notes-capability-sync-domains.md
- Docs/Published/ADR/README.md
- backlog/tasks/task-13008 - Track-Published-mirror-for-ADR-031-after-dev-sync-merge.md
updated_date: 2026-08-09 00:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the onboarding-docs-gate failure inherited from dev merge PR #2775 by generating and tracking the canonical Docs/Published mirror for Docs/ADR/031-notes-capability-sync-domains.md. Keep the fix limited to generated documentation and verify the exact failing docs refresh contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs/Published/ADR/031-notes-capability-sync-domains.md is generated from the canonical Docs source and tracked.
- [x] #2 test_refresh_output_matches_tracked_published_files passes.
- [x] #3 The full Docs test suite passes or unrelated failures are documented.
- [x] #4 No unrelated generated documentation changes are committed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-08-09: Root cause confirmed from onboarding-docs-gate run 31285834592/job/93174357704 and reproduced locally under PYTEST_DISABLE_PLUGIN_AUTOLOAD=1: dev merge PR #2775 added canonical ADR 031 and its canonical ADR index entry without tracking the generated Docs/Published counterparts. Ran Helper_Scripts/refresh_docs_published.sh. The exact failing manifest test changed from 1 failed to 1 passed after staging the generated ADR mirror and index. Full local Docs suite: 189 passed, 1 failed only because mkdocs is not installed in the project virtual environment; CI installs mkdocs and had no other Docs failures. git diff checks passed. Bandit skipped because the fix contains generated Markdown and task metadata only. The two unrelated untracked Watchlists templates were not touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Generated and tracked Docs/Published/ADR/031-notes-capability-sync-domains.md plus the ADR-031 entry in Docs/Published/ADR/README.md, repairing the onboarding-docs-gate manifest failure inherited from dev PR #2775. No Embeddings production behavior changed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Verification results recorded
- [x] #3 Modified files recorded
- [x] #4 Final summary added
<!-- DOD:END -->
