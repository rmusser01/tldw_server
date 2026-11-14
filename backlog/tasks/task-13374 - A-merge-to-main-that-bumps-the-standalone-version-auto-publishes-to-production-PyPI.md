---
id: TASK-13374
title: >-
  A merge to main that bumps the standalone version auto-publishes to production
  PyPI
status: Done
assignee: []
created_date: '2026-09-26 14:04'
updated_date: '2026-09-28 19:29'
labels:
  - security
  - ci
  - packaging
  - supply-chain
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Renumbered from a colliding TASK-13357 (dev already has TASK-13357, the audio preset ADR backfill). Content unchanged.

.github/workflows/mcp-unified-publish.yml gates its live PyPI upload behind a typed human confirmation on the manual path, and the push path bypasses that gate entirely. Verified on dev 2026-09-23:

1. on: push: branches: [main], with paths including apps/mcp-unified/pyproject.toml (:4-10).
2. publish-pypi.if (:260) is (workflow_dispatch AND target == 'pypi' AND confirm_publish == 'MCP_UNIFIED_PUBLISH') OR (push AND publish_candidate == 'true'). The push branch requires neither the target selection nor the typed confirmation.
3. environment: pypi (:267) would be the compensating control, but GET /repos/rmusser01/tldw_server/environments/pypi returns protection_rules: [] and deployment_branch_policy: null -- no reviewers, no wait timer, no branch policy.
4. permissions: id-token: write with trusted publishing, so no stored credential is needed.
5. publish-testpypi (:223) has no push branch, so the push path skips TestPyPI and goes straight to production PyPI.

Effect: merging any PR to main that changes the version in apps/mcp-unified/pyproject.toml publishes that version to real PyPI with no confirmation step and without staging to TestPyPI. Requires push access to main, so not an external-attacker path; the defect is that a deliberate two-step control on a live publish is defeated by an ordinary merge.

test_runtime_package_boundary.py::test_mcp_unified_publish_workflow_is_manual_and_gated asserts set(triggers) == {"workflow_dispatch"} and exists to prevent exactly this. It is red.

Owner decision on intent: if a version-bump push is meant to publish, the typed confirmation is not a control and the test should say so. If it is meant only to check readiness -- as the workflow's name "MCP Unified Publish Readiness" suggests -- the push path must not reach publish-pypi.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The push path either cannot reach publish-pypi, or reaches it only behind an equivalent explicit control
- [x] #2 The pypi environment has required reviewers, or the task records why it does not need them
- [x] #3 No path reaches production PyPI without first staging to TestPyPI, or that is recorded as intended
- [x] #4 test_mcp_unified_publish_workflow_is_manual_and_gated passes against the intended design
- [x] #5 The three sibling red assertions in the same file are triaged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
OWNER DECISION 2026-09-28: version bumps merged to main SHOULD auto-publish to PyPI. So: AC1: the push path reaching publish-pypi is intended; the explicit control is the reviewed PR that bumps apps/mcp-unified's version and is merged to main by the owner, and the typed confirmation stays on the manual path only. AC2: the pypi environment needs no required reviewers for the same reason. AC3: the push path skipping TestPyPI is intended. AC4: test_mcp_unified_publish_workflow_is_manual_and_gated already encodes this design on dev (triggers {workflow_dispatch, push}, push narrowed to main plus the version-carrying paths, no pull_request reachability, manual path still typed-confirm-gated) and passes. AC5: test_runtime_package_boundary.py is fully green on dev (44 passed; 6 skipped: offline package smokes that need preinstalled build requirements).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
