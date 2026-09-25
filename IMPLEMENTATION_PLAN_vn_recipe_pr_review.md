## Stage 1: Slot outcome ordering
**Goal**: Preserve failed-slot provenance and prevent older jobs from replacing newer slot state.
**Success Criteria**: Mixed variant outcomes retain Retry; old jobs cannot overwrite a newer batch's slot status.
**Tests**: Worker and repository regression tests for sibling completion order and cross-batch completion order.
**Status**: Complete

## Stage 2: Retry availability and fanout completion
**Goal**: Make historical legacy Retry availability accurate and finish fully processed fanout replays.
**Success Criteria**: The monitor disables Retry for an older recipe-less source; resumed fanout becomes completed when all children finished.
**Tests**: Service, frontend component, and fanout regression tests; OpenAPI drift check.
**Status**: Complete

## Stage 3: Review hygiene
**Goal**: Resolve valid performance, documentation, formatting, and test-classification comments without expanding production surface for test-only concerns.
**Success Criteria**: Blocking synchronous generation routes run off the event loop; new helpers are documented and formatted; tests are categorized; review threads have technical responses.
**Tests**: Focused endpoint tests, formatter, linter, and scoped suites.
**Status**: Complete

## Stage 4: Final verification
**Goal**: Recheck local and remote gates, update TASK-13356, and merge only when review and CI permit.
**Success Criteria**: No unresolved actionable review finding, green relevant checks, clean branch, and PR policy satisfied.
**Tests**: VN backend suite, frontend VN tests and typecheck, OpenAPI drift, Ruff, Bandit, and PR checks.
**Status**: In Progress
