# UAT430: restore skipped backend CI execution

Task: TASK-13260.278.17.2. PR2979; full/native UAT remains paused.

## Stage 1: Establish the cause
**Goal**: Explain skipped backend shards and Jobs PostgreSQL despite successful prerequisites.
**Success Criteria**: Retain hosted job states and reproduce missing status guards and false-success summary shell behavior.
**Tests**: Existing workflow contracts plus execution of real summary commands.
**Status**: Complete

Normal pull_request admission is intentionally skipped. GitHub propagates skips through dependency chains unless a status function overrides the implicit success check. CI35764582157 and Jobs35764582170 exhibit this; the earlier draft-status attribution is incorrect. Two causal regressions fail. Reference: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idneeds

## Stage 2: Repair conditions and summaries
**Goal**: Preserve admission, path filtering and failure/cancellation boundaries while allowing intended jobs.
**Success Criteria**: Explicit cancellation status function plus direct prerequisite success on all eight affected jobs; selected summaries accept only successful shards.
**Tests**: Workflow contracts, actual summary shell outcomes, actionlint, scoped lint/Bandit, independent review.
**Status**: Complete

## Stage 3: Confirm hosted execution
**Goal**: Verify required shards and PostgreSQL are scheduled and execute on the published head.
**Success Criteria**: No unexplained skips or false-success summaries; retain failures and address their causes.
**Tests**: First-attempt CI job states and results; full/native UAT remains paused.
**Status**: In Progress
