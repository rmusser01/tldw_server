# UAT413: report exhausted output without claiming provider outage

Task13260.278.11; bounded repair under the approved UAT sweep. Original uncaptured512-token failure remains unproven.

## Stage 1: Capture and causal regression
**Goal**: Capture actual upstream and API behavior, then reproduce classification in the real adapter caller.
**Success Criteria**: Hidden-only length exhaustion fails with a specific safe error; stop/malformed cases retain existing behavior.
**Tests**: Real captured128/512 comparison, failing adapter cases and valid-answer controls.
**Status**: Complete

## Stage 2: Minimal safe classification
**Goal**: Reuse bounded provider errors for output exhaustion while retaining non-replay and no-success-side-effect contracts.
**Success Criteria**: Focused and neighboring tests, lint, Bandit and source review pass.
**Tests**: Chat fallback/semantic/error suites; scoped Ruff and Bandit.
**Status**: Complete

## Stage 3: Native controlled acceptance
**Goal**: Verify exhausted request and successful recovery on SQLite and official PostgreSQL using committed source and exact provider capture.
**Success Criteria**: Useful error, retained user turn, no empty assistant saved, successful bounded recovery; restore owned runtime config.
**Tests**: Native error/recovery receipts and updated tracker/task.
**Status**: In Progress
