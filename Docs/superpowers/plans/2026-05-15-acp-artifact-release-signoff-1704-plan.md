## Stage 1: Backend Contract Coverage
**Goal**: Prove ACP completion artifacts promote into canonical workspace artifacts with stable provenance, lineage, review, redaction, versioning, and export-reference metadata.
**Success Criteria**: Focused tests cover accepted creation, accepted update/versioning, rejected/needs-revision skip paths, malformed candidate rejection, and exportable metadata invariants.
**Tests**: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q`
**Status**: Complete

## Stage 2: UI Release Regression Coverage
**Goal**: Keep artifact detail rendering and API hydration aligned with the release contract for ACP-produced artifacts.
**Success Criteria**: UI tests cover artifact detail, session drill-through, redacted/support-safe behavior, review state display and controls, export references, and backend field mapping.
**Tests**: `bunx vitest run src/components/Option/WorkspacePlayground/StudioPane/__tests__/TraceableArtifactDetail.test.tsx src/store/__tests__/workspace-api-first.test.ts`
**Status**: Complete

## Stage 3: Readiness Evidence And Deferrals
**Goal**: Make the release signoff auditable from docs and GitHub issue links.
**Success Criteria**: Readiness evidence names landed implementation surfaces, verification commands, issue coverage, and explicit deferrals for non-golden-path artifact types/export channels.
**Tests**: Docs review plus `git diff --check`.
**Status**: Complete

## Stage 4: Verification And PR Closeout
**Goal**: Run focused verification, update Backlog task state, and package the branch for review.
**Success Criteria**: Focused backend/UI tests pass or any environment-only blocker is documented, Bandit runs on touched Python scope, Backlog task has final summary, and PR references #1704/#1532.
**Tests**: Focused pytest, focused Vitest from `apps/packages/ui`, Bandit on touched Python paths, `git diff --check`.
**Status**: Complete
