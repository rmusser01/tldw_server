---
id: TASK-13343
title: 'Drain the platform-mcp-inapp CI quarantine (10 files, 21 red entries)'
status: Done
assignee: []
created_date: '2026-09-22 06:13'
updated_date: '2026-09-23 00:40'
labels:
  - tests
  - ci
  - mcp
dependencies: []
references:
  - .github/workflows/ci.yml
  - tldw_Server_API/app/core/MCP_unified/tests/test_refresh_token.py
  - tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13291 wired app/core/MCP_unified/tests into CI as the platform-mcp-inapp shard with 10 files quarantined behind --ignore because they were already red. The shard is green at 2948 passed; this task drains the list.

TWO GROUPS.

A. Distribution-building (13 of the 21 entries) - these build sdists/wheels and shell out, and test_runtime_package_boundary.py:29 sets pytestmark = pytest.mark.unit while doing so. They likely do not belong in a unit shard at all; decide between a dedicated packaging job and a non-unit marker:
  test_runtime_package_boundary.py (10 entries)
  test_gateway_protocol_artifact_consumer.py (3 entries)

B. Genuine drift accumulated while the tree was outside CI (8 entries, 8 files). Sampled causes: TypeError: refresh_token() missing 1 required positional argument: request (endpoint signature changed, test not updated); KeyError: rows (result shape changed); gateway status payload gained fields the test does not expect. Each needs a judgement about whether the test or the product is correct:
  test_server_batch_and_formatting.py, test_refresh_token.py, test_gateway_tool_discovery.py,
  test_gateway_fastapi_package.py, test_gateway_admin_auth.py, test_flashcards_module.py,
  test_external_federation_integration.py, test_codegraph_module.py

SEPARATE, DO NOT FOLD IN: test_rag_module.py and test_gateway_protocol_stdio.py are NOT quarantined. They pass in isolation and in the new shard, but fail when run in-process with tests/MCP_unified - cross-test state pollution, the class the 2026-07-04 audit named as its top defect category. That is why the tree got its own shard. Worth its own investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Group A relocated to an appropriate job or marker, not silently ignored
- [x] #2 Group B triaged: each test either fixed or its product defect filed
- [x] #3 Quarantine list in ci.yml is empty and the --ignore block removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE in 7934c39e40 and 64c253c0c3. Shard is green: 3343 passed, 14 skipped, 0 failed, 0 errors (was 13 failed + 8 errors).

AC3: all 50 --ignore lines removed from ci.yml (the matrix is duplicated five times). The stale "SHRINKING quarantine" comment is replaced with an instruction not to add new entries.

AC1 (Group A, 13 entries): the eight artifact tests errored on `python -m build` with BackendUnavailable. Cause is environmental, not a packaging regression: the build runs --no-isolation, so the backend apps/mcp-unified declares (setuptools>=79.0.1) must be importable, and setuptools is not a tldw-server runtime dependency. A require_build_backend() guard now skips them loudly -- matching the guard the same file already applied at line 198 for the offline smoke test. They are additionally marked `packaging` (registered in pyproject) so a unit shard can deselect with -m "not packaging" and a dedicated job can select with -m packaging. Only the five tests that request standalone_distributions are marked in test_runtime_package_boundary.py; the rest assert on files already in the tree. Wiring a separate CI job is left as the owner's call.

Three Group A entries were NOT build-related and were real assertions that had drifted:
- _workflow_run_blocks indexed job["steps"], which KeyErrors on a job that calls a reusable workflow. Both mcp-unified-rc.yml and pypi-package.yml now have an `admission` job of that shape.
- The PyPI publish job gained a "Guard against duplicate PyPI version" step. Accommodated, while still asserting the UPLOAD step is a pinned action with no shell of its own -- the property that keeps a credential out of any workflow command.
- package_license_file_is_local asserted byte-equality with the root LICENSE, which became a multi-license SCOPE MAP at da0ec87d7d. It was therefore demanding the package ship a scope map instead of a license -- the opposite of the test's own name. Now asserts the package carries the GPL-3.0 text it declares, against LICENSES/GPL-3.0-only.txt.

AC2 (Group B, 8 entries, each judged test-versus-product):
- refresh_token: product right. The endpoint gained `request: Request` and a demo-auth gate (MCP_ENABLE_DEMO_AUTH, debug/test mode, a 16-char secret, loopback/private peer). Test updated AND extended to cover the gate, which nothing asserted.
- server_batch_and_formatting: product right. tool_observability.attach_execution_eval_metadata adds an `eval` block to dict results, which the rest of the suite already asserts on. Exact dict equality replaced by field assertions plus a check on the telemetry.
- gateway_admin_auth: product right. Status payload grew nine blocks. The test is about admin auth not gating status, so it asserts that, the identity fields, and that admin_auth never discloses the key on an unauthenticated route.
- gateway_tool_discovery: test wrong. Hardcoded REPO_ROOT/"mcp_unified" path went stale when the package moved under apps/mcp-unified/src/. Derived from the imported module instead, so it cannot drift again.
- gateway_fastapi_package: test wrong. Routes moved behind an _IncludedRouter with paths relative to the mount, so app.routes no longer carries "/mcp/status". GET /mcp/status still answers 200; the lookup now searches nested routers.
- flashcards: product right, and the security property is intact. The module short-circuits on an empty result and never calls the exporter, so the captured["rows"] probe became unobservable. The outcome is now asserted directly: no file is produced for another workspace's deck.
- codegraph: NOT drift, and NOT a product defect. The .ts half needs the optional `codegraph` extra; without tree_sitter_typescript the language registry marks TypeScript as having no symbol extraction and drops the file before indexing, so files_indexed is 1 not 2. Sibling tests in the same file already skip on exactly this -- this one simply lacked the guard.
- external_federation: test wrong. It injected a custom external_server_loader but not the matching external_credential_broker, so the call fell through to the DB-backed broker service, which raised "Unknown external server: docs" for a server existing only in the injected registry. In production loader and broker read the same registry; injecting the other half restores the symmetry.

ONE FINDING WORTH THE OWNER'S ATTENTION (not a defect, but a governance change the quarantine hid): test_mcp_unified_publish_workflow_is_manual_and_gated asserted workflow_dispatch was the ONLY trigger of .github/workflows/mcp-unified-publish.yml. Commit e4231f6d82 ("Auto publish MCP Unified version bumps") deliberately added a push trigger, and publish-pypi now runs on `github.event_name == 'push' && needs.detect-version-change.outputs.publish_candidate == 'true'` -- i.e. a push to main that bumps the version publishes to PyPI without the confirm_publish token. Because this test was quarantined at the time, nothing flagged the contradiction. The commit title makes the intent explicit, so the test was updated to encode that design rather than fight it, while still asserting every safety property: no pull_request trigger; the push confined to main and to the three version-bearing paths; TestPyPI unreachable from push; and the PyPI path conditional on a detected version change inside the `pypi` environment. Whether that environment has required reviewers is configured in repo settings and cannot be verified from the tree -- worth confirming, since it is the only remaining human gate on that path.

NOT FOLDED IN, as the task instructed: test_rag_module.py and test_gateway_protocol_stdio.py were never quarantined. They pass in isolation and in this shard but fail in-process with tests/MCP_unified -- cross-test state pollution, which is why the tree has its own shard. Still worth its own investigation.

Verification: full tree 3343 passed / 14 skipped / 0 failed / 0 errors. -m "not packaging" 3317 passed / 34 deselected. -m packaging 26 passed / 8 skipped in 5s. ci.yml parses as YAML after the edit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quarantine fully drained: 21 red entries resolved, 10 files back in CI, shard green at 3343 passed. Group B was eight distinct test-versus-product judgements, two of which turned out to be missing optional dependencies rather than drift. Group A's build tests now skip loudly and carry a packaging marker. One governance change surfaced: the MCP Unified publish workflow gained an auto-publish-on-version-bump push trigger while its guard test was quarantined.
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
