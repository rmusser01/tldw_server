---
id: TASK-328
title: Implement VN script authoring graph API
status: Done
assignee: []
created_date: '2026-05-14 01:45'
updated_date: '2026-05-14 03:08'
labels:
  - vn
  - scripts
  - api
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/pull/1641'
  - 'https://github.com/rmusser01/tldw_server/pull/1656'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend-only VN script authoring graph API from Docs/superpowers/specs/2026-05-14-vn-script-authoring-graph-design.md and Docs/superpowers/plans/2026-05-14-vn-script-authoring-graph-api-implementation-plan.md. Scope includes a pure computed graph builder, service methods, VN Scripts schemas/endpoints, capability flag, API docs, focused backend tests, compile verification, Bandit, and PR preparation. No WebUI changes, no model calls, no runtime session mutation, and no graph persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pure graph builder returns deterministic outline and graph layers with encoded stable IDs, bracket JSON paths, content hashes, graph semantics version, limits, truncation behavior, conservative terminal states, and graph diagnostics.
- [x] #2 Service methods support stored draft graph, supplied draft graph preview, and published-version graph without persisting drafts or diagnostics; published-version graphs use pinned version context where available.
- [x] #3 VN Scripts API exposes draft graph, draft graph-preview, and version graph endpoints with Pydantic schemas, existing VN error envelopes, and capability discovery through features.script_authoring_graph.
- [x] #4 API documentation covers endpoints, source modes, response shape, diagnostics, limits, hashing, custom frontend flow, and non-goals.
- [x] #5 Focused VN script/platform tests, compile checks, Bandit on touched Python scope, and git diff checks are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Subagent-driven execution on branch `codex/vn-script-authoring-graph-api` in worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vn-script-authoring-graph-design`.

Implementation sequence from `Docs/superpowers/plans/2026-05-14-vn-script-authoring-graph-api-implementation-plan.md`:

1. Pure authoring graph builder: create `authoring_graph.py`, add/adjust validator reachability helper, add `test_vn_script_authoring_graph.py` builder coverage, commit graph builder slice.
2. Service graph methods: add `get_draft_graph`, `preview_draft_graph`, `get_version_graph`, supplied draft guards, non-mutating validation behavior, version snapshot context tests, commit service slice.
3. API schemas and endpoints: add graph request/response schemas, draft graph/preview/version graph routes, error mapping, endpoint tests, commit API slice.
4. Capabilities and docs: add `features.script_authoring_graph`, update capability schema if needed, update `Docs/API/VN.md` and tests, commit docs/capability slice.
5. Final verification: focused VN_Scripts tests, VN capabilities tests, compileall, Bandit on touched Python scope, `git diff --check`, update Backlog task, prepare PR.

Constraints: backend-only; no WebUI changes; no model calls; no runtime session mutation; no graph persistence; graph preview and stored graph must not persist diagnostics; edge semantics remain statically knowable only; use encoded stable IDs and bracket JSON paths; preserve existing VN error envelope style.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 pure graph builder completed and accepted after review. Commit 85068f2da adds authoring_graph.py and focused graph tests. Verification: focused graph pytest passed with 16 tests; spec compliance review approved; code quality review initially found omitted-target, duplicate edge ID, and mutable diagnostics issues, all fixed with regression tests; re-review approved. The worktree lacks its own .venv, so verification used /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python.

Task 2 service graph methods completed and accepted after review. Commit d0d05dfea4 adds get_draft_graph, preview_draft_graph, and get_version_graph plus service tests. Code quality review found a published-version snapshot drift bug where validation could recompute from live character metadata; fixed by using stored version validation for version graph responses, with regression coverage. Verification: focused graph pytest passed with 22 tests; publish snapshot tests passed; spec and quality re-reviews approved.

Task 3 API schemas/endpoints completed and accepted after review. Commit b02d27df89 adds graph request/response schemas, draft graph, draft graph-preview, and version graph endpoints, plus API tests. Verification: VN script API tests passed with 36 tests; spec compliance review approved; code quality review approved with no findings.

Task 4 capabilities/docs completed. Commits 22e0224ed and 89a221561 add route-gated features.script_authoring_graph capability discovery, API docs for graph endpoints/source modes/response shape/diagnostics/limits/hashing/custom frontend flow/non-goals, and a regression test that keeps the graph feature disabled when only partial scripts routes are registered.

PR review fixes added on 2026-05-14: graph draft endpoints now resolve AuthNZ profile rows and accessible audio refs before validation; graph reachability is derived from emitted bounded edges so truncated responses stay internally consistent; capabilities now require exact graph route path+method pairs; supplied_draft_invalid_shape has explicit 400 mapping; helper docstrings and wrapped long lines added. Regression coverage added for custom AuthNZ graph validation context, method-aware capability gating, emitted-edge reachability under edge truncation, and static fallthrough warning diagnostics.

Final verification after review fixes: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Scripts -q -> 123 passed, 5 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py -q -> 4 passed, 8 warnings; compileall on touched VN modules passed; Bandit on touched backend VN modules reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL:BEGIN -->
Opened PR #1656 for the backend-only VN script authoring graph API and addressed live PR review feedback. The API now computes deterministic authoring graphs for stored drafts, supplied draft previews, and published versions; keeps graph responses non-mutating; validates draft graph responses with the same resolved profile/audio context as existing VN Scripts validation; uses emitted bounded edges for reachability; advertises features.script_authoring_graph only when the exact graph routes and methods are registered; and documents graph behavior for custom frontend clients.
<!-- SECTION:FINAL:END -->
