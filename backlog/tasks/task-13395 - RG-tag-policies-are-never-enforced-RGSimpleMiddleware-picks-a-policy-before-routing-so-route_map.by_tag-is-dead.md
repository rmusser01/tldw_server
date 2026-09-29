---
id: TASK-13395
title: >-
  RG tag policies are never enforced: RGSimpleMiddleware picks a policy before
  routing, so route_map.by_tag is dead
status: To Do
assignee: []
created_date: '2026-09-29 15:45'
labels:
  - resource-governance
  - security
  - backend
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by Qodo on #3053 (FastAPI 0.141 bump); pre-existing, identical on FastAPI 0.136.

RGSimpleMiddleware.__call__ derives the policy (_derive_policy_id) before delegating to the app, so request.scope has no 'route' yet. The by_tag branch (middleware_simple.py, 'Fallback to tag-based routing (may not be available early in ASGI pipeline)') therefore never matches. After by_path, only the chat/audio path heuristics apply.

Consequences:
- Every route whose only mapping is by_tag runs ungoverned. resource_governor_policies.yaml has about 40 by_tag entries (admin, users, jobs, outputs, reading, kanban, flashcards, quizzes, sandbox, connectors, ocr, media, research, skills, writing, ...), and many have no by_path equivalent.
- The coverage audit (coverage_audit._route_is_mapped) and the startup route-map audit (startup_resource_governor._audit_route_map_coverage) both count a by_tag match as protected. The admin coverage report and the startup warning therefore overstate enforcement.

Options:
(a) Enforce tags before routing. Build a (path regex, methods) -> tags index from iter_served_routes(app.routes) when the route map is compiled (starlette.routing.compile_path(served.path)), and match request path and method against it after by_path. This turns on governance for every tag-only route at once, under whatever limits the mapped policies set (mostly core.default). Some routes would start returning 429 or hitting budgets.
(b) Make the audits honest: count only by_path and the heuristics as protected, and drop or annotate by_tag. There is no enforcement change.

Recommendation: (a), because the policy file states the intent that these routes be governed. Ship it with a pre-merge listing of newly governed routes and their policies, and (b)'s audit change until (a) lands. This is an owner decision because (a) changes production rate limiting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Owner decision recorded: (a) enforce tag policies pre-routing, or (b) audits count only enforceable mappings
- [ ] #2 Whichever is chosen, the coverage audit and startup audit agree with what RGSimpleMiddleware actually enforces
- [ ] #3 A request-level test: an included route with a tag-only policy, no path policy and no heuristic match is governed under (a), or reported unprotected under (b)
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
