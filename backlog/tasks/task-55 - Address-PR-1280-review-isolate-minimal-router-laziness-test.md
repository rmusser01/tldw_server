---
id: TASK-55
title: 'Address PR #1280 review: isolate minimal router laziness test'
status: Done
assignee:
  - codex
created_date: '2026-05-05 02:25'
updated_date: '2026-05-05 02:31'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1280'
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the Qodo PR #1280 finding that the focused minimal Llama.cpp/messages laziness test still allows unrelated eager endpoint imports from iter_minimal_optional_router_specs. Scope is test-only isolation unless verification exposes a production issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused PR #1280 laziness test installs lightweight fakes for unrelated eager-imported minimal optional endpoint modules before calling iter_minimal_optional_router_specs
- [x] #2 The test continues to verify Llama.cpp/messages module import and router/public_router attr lookup stay deferred until resolution
- [x] #3 Focused test, full router group tests, main router/OpenAPI contracts, Bandit touched source scope or documented test-only skip, and diff hygiene are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce focused test and identify unrelated real endpoint imports still happening during spec construction. 2. Add lightweight fakes for the unrelated eager-imported minimal optional endpoints before the targeted Llama.cpp/messages assertion. 3. Re-run focused/full router groups, main router contracts, OpenAPI contracts, Bandit or document test-only skip, and diff hygiene. 4. Commit, push, and update PR #1280.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #1280 Qodo reliability finding by moving unrelated eager endpoint isolation into test_iter_minimal_optional_router_specs_defers_llamacpp_messages_attr_lookup and using a test-local __import__ shim that returns fake routers for unrelated endpoint imports while leaving the target Llama.cpp/messages importlib path tracked for laziness.

Verification passed: focused Llama.cpp/messages laziness test; full router groups; main router contract; OpenAPI contracts; git diff --check.

Bandit skipped for this review fix because only test code and Backlog task metadata changed; no production source scope was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1280 review feedback by isolating the minimal optional Llama.cpp/messages laziness regression test from unrelated eager endpoint imports. The test now dynamically fakes unrelated endpoint router imports and explicitly verifies the early vector/embedding/media embedding imports are intercepted while the target lazy import and attr lookup assertions remain intact.
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
