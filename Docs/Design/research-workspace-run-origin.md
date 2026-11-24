# Research Workspace run origin

- ADR required — yes
- ADR path — [ADR-102](../ADR/102-research-workspace-run-origin.md)
- Reason — A dedicated API and a durable receipt bind a canonical Run to one authenticated Workspace launch.

## Admission

`GET /api/v1/workspaces/{workspace_id}/research-runs` advertises
`research-workspace-runs.v1` for an accessible Workspace.
`POST` to the same path accepts a bounded launch identity, question, selected
source IDs and versions, and optional initiating artifact or chat message
references. The route uses the authenticated principal and checks the
expected-user precondition. A first request checks the Workspace, selected
association revisions, and the initiating owner.

## Persistence and return

ResearchSessionsDB commits the queued Run, the first phase dispatch
descriptor, and the launch receipt before submitting the phase to Jobs.
The existing idempotent dispatch path handles retries. The signed-in owner
gets the workspace origin on the normal Run detail route and reads the
bundle from the existing owner endpoint. A client confirms a Report import
as a separate action. Completion leaves Chat delivery state unset.

## Implementation plan

1. Add API and storage tests for first admission, replay, conflicts, owner
   fences, and missing origin on ordinary runs.
2. Add the Workspace admission endpoint and the transactional receipt. Reuse
   native phase dispatch and project the origin on Run detail.
3. Verify a lost response, first-phase recovery, deletion, and unchanged
   legacy Runs and Chat paths. Refresh the generated OpenAPI contract.
