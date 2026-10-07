# Retained WebUI opening source — TASK-13522

Approved Stage4 continuation of the chat conversation UX remediation design.
ADR required: no; existing ADR049/050 govern ownership/lifecycle. Reuse the actual Form captureHandoffGuard, draftReady and callback-ready pattern. Internal source data stays outside the serialized transfer overrides.

## Stage 1: Reproduce source retirement
**Goal**: Demonstrate input/H1/rendered-source and hydration failures through existing real route/production callback boundaries.
**Success Criteria**: Held config/document waits and stale rendered actions fail before the repair; header and completion lifetime are covered.
**Tests**: sidepanel-chat.ownership, form.webui-document-handoff, ControlRow.role-play-handoff, HeaderSimple.webui-handoff.
**Status**: Complete

## Stage 2: Carry one rendered source
**Goal**: Capture draft/setup once in the actual hydrated Form and retain its predicate through preparation/opening.
**Success Criteria**: Rendered document selection and actual owner/input/H1 remain current; header uses the actual registered Form action; retired completion cannot focus/report. No internal source fields serialize.
**Tests**: Failure-first source cases plus existing successful source, local transfer and qualified expansion regressions.
**Status**: Complete

## Stage 3: Verify and record
**Goal**: Commit the bounded source repair.
**Success Criteria**: Appropriate shared regressions, both clients, no-added lint, two independent reviews, canonical task/docs/commit.
**Tests**: Existing test/client tooling. Bandit N/A TS/TSX/docs. Native popup/storage/package/geometry and destination settings/document qualification stay parent scope.
**Status**: In Progress

Failure-first evidence: source/input/H1 (12 failures), hydration (2), retired feedback/focus (4), and independent review regressions (3). Registration loop reproduced by real route fixture; repaired with ref plus readiness boolean. Active controller now retires in layout cleanup. Same-entry and shared-entry supersession retain only the latest current completion.

Registration repair reassessment: storing action identity in state loops; ref plus readiness state still loops when every callback cleanup publishes null/new action. Reuse a stable ready ref, publish callback changes without identity-change cleanup, and unregister only on stable registration callback cleanup. Parent changes readiness state only for actual readiness transitions. No selector/schema/protocol changes.

Seven-suite bounded verification passes253 tests. Remote dev advanced to3ca1ff055be3c75a7fa844a02b5d8ba925baa2be with a substantial integrated chat/notes/backend UX series; capture this bounded repair in a verified checkpoint, then merge and reassess actual latest-dev root contracts before finalization. Added raw URL JSON check for internal source exclusion.
