---
id: TASK-13348
title: >-
  Confirm the pypi environment requires reviewers for auto-published MCP Unified
  releases
status: Done
assignee: []
created_date: '2026-09-23 00:40'
updated_date: '2026-09-28 06:28'
labels:
  - ci
  - security
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP Unified publish workflow auto-publishes to PyPI on a push to main that bumps the version:

.github/workflows/mcp-unified-publish.yml, job publish-pypi:
  if: (workflow_dispatch && target == pypi && confirm_publish == 'MCP_UNIFIED_PUBLISH') || (push && detect-version-change.outputs.publish_candidate == 'true')

The push path carries NO confirm_publish token. This was deliberate -- e4231f6d82 'Auto publish MCP Unified version bumps' -- and test_mcp_unified_publish_workflow_is_manual_and_gated now encodes it (TASK-13343), asserting the push is confined to main and to the three version-bearing paths, that pull_request cannot trigger the workflow at all, and that TestPyPI is unreachable from push.

The remaining human gate on that path is the GitHub 'pypi' environment's protection rules, which are configured in repository settings and cannot be verified from the tree. Confirm required reviewers are set on it; if they are not, a single merged version bump publishes to PyPI unattended.

Source: found while draining the quarantine in TASK-13343.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The pypi GitHub environment's protection rules are confirmed, and the finding recorded either way
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Checked 2026-09-27 via the GitHub API (gh api repos/rmusser01/tldw_server/environments/pypi): protection_rules = [] (no required reviewers, no wait timer), deployment_branch_policy = null (any branch may deploy), can_admins_bypass = true. testpypi is the same. So the risk this task described is real: publish-pypi (mcp-unified-publish.yml, environment pypi, OIDC id-token: write, pypa/gh-action-pypi-publish) runs unattended on any push to main that bumps the MCP Unified version; the only gate is the workflow's own if: (main + the three version-bearing paths). Remediation is a repository-settings change, not a tree change, and is the owner's call because e4231f6d82 made auto-publish deliberate: either (a) add a required reviewer to the pypi environment (restores a human gate, costs one click per release), or (b) at minimum restrict pypi deployments to the main branch (keeps auto-publish, closes other branches). Recommended: (b) now; (a) if unattended publishing is not intended. Bandit: N/A (no code change).

Correction (Qodo review on #3037): the recommendation above was wrong to call (b) the immediate remediation. A main-only deployment branch policy only stops deployments from other branches; the workflow already restricts the auto-publish path to pushes on main, so automatic releases from main would remain unattended. The only setting that adds a human gate is (a) required reviewers on the pypi environment. Recommendation: add required reviewers unless unattended publishing of MCP Unified version bumps is intended; a main-only branch policy is optional defence-in-depth on top, not a substitute.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed: the pypi environment has no protection rules and no branch policy, so a version bump merged to main publishes unattended. The fix that adds a human gate is required reviewers on the environment (repository settings, owner's call); a main-only branch policy alone does not.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
