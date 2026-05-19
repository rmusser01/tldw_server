## Stage 1: Audit Existing UX E2E Coverage

**Goal**: Identify the highest-value weak UX e2e specs across the WebUI and extension.
**Success Criteria**: At least one extension suite and one WebUI suite are selected based on concrete issues like brittle waits, weak selectors, or shallow assertions.
**Tests**: Read current Playwright configs, representative UX specs, and shared UI selectors/helpers.
**Status**: Complete

## Stage 2: Strengthen Extension UX Coverage

**Goal**: Refresh the selected extension UX spec so it uses stable waits/selectors and asserts meaningful UX outcomes.
**Success Criteria**: The updated extension spec avoids fixed sleeps where practical, reduces `networkidle` dependence, and checks user-visible state transitions instead of mere text presence.
**Tests**: Focused Playwright run for the touched extension spec.
**Status**: Complete

## Stage 3: Strengthen WebUI UX Coverage

**Goal**: Improve one WebUI UX smoke/spec flow with stronger interaction and accessibility expectations.
**Success Criteria**: The updated WebUI spec verifies a real UX behavior boundary such as keyboard dismissal, focus recovery, or explicit control state.
**Tests**: Focused Playwright run for the touched WebUI spec.
**Status**: Complete

## Stage 4: Verify and Summarize Residual Gaps

**Goal**: Run targeted verification and record the remaining UX e2e debt surfaced during the audit.
**Success Criteria**: Touched specs are re-run, Bandit is run on the touched scope per repo guidance, and remaining weak areas are called out explicitly.
**Tests**: Focused Playwright runs, any touched unit/guard tests, and Bandit on touched paths.
**Status**: Complete
