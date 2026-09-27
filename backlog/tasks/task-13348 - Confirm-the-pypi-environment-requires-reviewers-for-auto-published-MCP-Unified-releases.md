---
id: TASK-13348
title: >-
  Confirm the pypi environment requires reviewers for auto-published MCP Unified
  releases
status: To Do
assignee: []
created_date: '2026-09-23 00:40'
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
- [ ] #1 The pypi GitHub environment's protection rules are confirmed, and the finding recorded either way
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
