---
id: TASK-510
title: Address PR 1982 review comments and merge conflicts
status: In Progress
labels:
  - pr-review
  - merge-conflicts
references:
  - https://github.com/rmusser01/tldw_server/pull/1982
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve PR #1982 review comments, inspect checks, and address merge conflicts for dev -> main. Conflict policy from requester: where main and dev overlap, keep dev content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1982 merge conflicts are resolved with dev taking precedence on overlapping hunks.
- [x] #2 Still-valid review comments are addressed or documented if skipped as invalid.
- [x] #3 Relevant local validation is run and recorded.
- [x] #4 Touched scope is checked with Bandit when applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-03: Review inventory found still-valid unresolved CodeRabbit items in `.github/workflows/sbom.yml`, `.github/workflows/vz-linux-host-gated.yml`, `.impeccable/design.json`, `DESIGN.md`, and `Dockerfiles/docker-compose.webui.yml`.
- 2026-06-03: Latest CodeRabbit outside-diff review also flagged `.github/workflows/ui-research-workspace-parity.yml` trigger filters and hidden artifact upload handling.
- 2026-06-03: GitHub Advanced Security review threads visible in the PR were either outdated or already resolved.
- 2026-06-03: `gh pr checks 1982 --repo rmusser01/tldw_server --json ...` showed pre-commit and CodeRabbit passing, CodeQL/onboarding pending or queued, and cubic skipped.
- 2026-06-03: Merged `origin/main` into the PR worktree with `-X ours`; resolved rename/delete and add/add conflicts according to the requester policy that `dev` wins overlapping hunks.
- 2026-06-03: Addressed review comments in `.github/workflows/sbom.yml`, `.github/workflows/vz-linux-host-gated.yml`, `.github/workflows/ui-research-workspace-parity.yml`, `.impeccable/design.json`, `DESIGN.md`, and `Dockerfiles/docker-compose.webui.yml`.
- 2026-06-03: Fixed MCP catalog connection validation ordering so private/reserved hosts are rejected before catalog lookup and catalog probes use the canonical catalog URL without request query/fragment secrets.
- 2026-06-03: Reproduced and fixed the failing `onboarding-docs-gate` docs discoverability tests by restoring user-guide index entries for benchmark creation, OpenWebUI import/hydration, and Flashcards Study Guide.
- 2026-06-03: Updated PR title/body to address metadata warnings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1982 merge conflicts and review feedback. `dev` was kept for overlapping conflict hunks, non-conflicting `main` changes were retained, review-targeted workflow/docs/design/compose fixes were applied, MCP catalog connection validation was corrected, and validation passed locally.
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
