# Knowledge integration with latest Media UX development

Associated task: TASK-13453. Preserve the approved Knowledge remediation and the Media UX work merged through PR #3194 on dev `587cd8e9fe3b42eba451b83c1ba690607035eace`. No new feature scope.

## Stage 1: Reconcile the shared components
**Goal**: Rebase onto the new dev and resolve any shared ingestion, review, scope, and locale conflicts.
**Success Criteria**: Both feature sets remain represented; no unresolved conflict markers or unexpected product deletions.
**Tests**: Inspect every conflict and overlapping product diff against both parent implementations.
**Status**: In Progress

## Stage 2: Verify integrated behavior
**Goal**: Run the combined affected Knowledge and upstream Media tests, official client type checks, API coverage, and contract check.
**Success Criteria**: Integrated selections, ingestion recovery, canonical source handoffs, review actions, and account fences pass the existing CI contract.
**Tests**: Combined affected Vitest manifest with one worker and 15-second timeout; WebUI typecheck; extension compile; canonical clipper API tests; canonical OpenAPI drift check; Bandit on touched production Python.
**Status**: Not Started

## Stage 3: Review and prepare delivery
**Goal**: Independently review the overlapping ingestion and review/scope integration and prepare one verified current-base push.
**Success Criteria**: Findings resolved; hooks and whitespace checks pass; evidence and Backlog notes current; original dirty checkout preserved.
**Tests**: Independent scoped reviews, explicit-file repository hooks, final diff review, base/head verification.
**Status**: Not Started

Remove this owned plan after the local integration stages are complete; retain the final verification record and task summary. Remote required gates and the authorized merge follow the verified push.
