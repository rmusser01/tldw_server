---
id: TASK-13357
title: >-
  A merge to main that bumps the standalone version auto-publishes to production
  PyPI
status: To Do
assignee: []
created_date: '2026-09-23 15:03'
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
`.github/workflows/mcp-unified-publish.yml` gates its live PyPI upload behind a typed human confirmation on the manual path, and that gate is bypassed entirely by the push path.

Verified end to end on dev (91e8bbf), 2026-09-23:

1. `on: push: branches: [main]`, `paths:` includes `apps/mcp-unified/pyproject.toml` (`:4-10`).
2. `publish-pypi.if` (`:260`) is:
   ```
   (github.event_name == 'workflow_dispatch' && inputs.target == 'pypi' && inputs.confirm_publish == 'MCP_UNIFIED_PUBLISH')
   || (github.event_name == 'push' && needs.detect-version-change.outputs.publish_candidate == 'true')
   ```
   The push branch requires **neither** `target == 'pypi'` **nor** the `confirm_publish` typed string that the manual branch requires.
3. `environment: name: pypi` (`:267`) would be the compensating control. `GET /repos/rmusser01/tldw_server/environments/pypi` returns `"protection_rules": []` and `"deployment_branch_policy": null` -- **no required reviewers, no wait timer, no branch policy**.
4. `permissions: id-token: write` with trusted publishing, so the push path needs no stored credential to upload.
5. `publish-testpypi` (`:223`) has **no** push branch -- it is `workflow_dispatch`-only. So the push path skips TestPyPI and goes straight to production PyPI.

**Effect:** merging any PR to `main` that changes the version in `apps/mcp-unified/pyproject.toml` publishes that version to real PyPI, with no confirmation step and without ever having been staged to TestPyPI. This does require push access to `main`, so it is not an external-attacker path -- the defect is that a deliberate two-step control on a live publish is defeated by an ordinary merge, so a routine version bump ships unintentionally.

`test_runtime_package_boundary.py::test_mcp_unified_publish_workflow_is_manual_and_gated` asserts `set(triggers) == {"workflow_dispatch"}` and exists to prevent exactly this. It is red and has been unseen because `app/core/MCP_unified/tests` runs in no CI job -- TASK-13291. This is that task's strongest concrete instance.

Three sibling assertions in the same file are also red and likely the same change: `test_mcp_unified_rc_workflow_uses_private_permissions`, `test_root_pypi_package_workflow_is_tldw_server_only`, `test_mcp_unified_publish_workflow_uses_trusted_publishing_for_pypi`. Triage them together.

**Owner decision on the intent:** if a version-bump push is meant to *publish*, the typed confirmation is not a control and the test should say so. If it is meant only to check readiness -- which the workflow's own name, "MCP Unified Publish Readiness", suggests -- then the push path must not reach `publish-pypi`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The push path either cannot reach publish-pypi, or reaches it only behind an equivalent explicit control
- [ ] #2 The pypi environment has required reviewers, or the task records why it does not need them
- [ ] #3 No path reaches production PyPI without first staging to TestPyPI, or that is recorded as intended
- [ ] #4 test_mcp_unified_publish_workflow_is_manual_and_gated passes against the intended design
- [ ] #5 The three sibling red assertions in the same file are triaged
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
