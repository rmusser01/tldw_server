# ACP Artifact Release Verification

Date: 2026-05-15
Primary issue: [#1704](https://github.com/rmusser01/tldw_server/issues/1704)
Parent tracker: [#1532](https://github.com/rmusser01/tldw_server/issues/1532)

## Scope

This document records release-grade verification for the first ACP traceable
artifact golden path: an accepted, source-grounded ACP deliverable promoted
into a canonical Workspace artifact and exported as an accepted artifact
version.

The verified implementation stack is:

| Issue | Surface | Verified posture |
| --- | --- | --- |
| [#1703](https://github.com/rmusser01/tldw_server/issues/1703) | Workspace artifact storage/API | Traceable fields, version rows, redaction posture, source lineage, producer metadata, review metadata, schema version, and export refs persist through DB/API normalization. |
| [#1706](https://github.com/rmusser01/tldw_server/issues/1706) | ACP completion promotion | Accepted ACP completion artifacts promote into workspace work-product artifacts; rejected and needs-revision outputs remain execution evidence. |
| [#1707](https://github.com/rmusser01/tldw_server/issues/1707) | Workspace UI detail | Artifact detail renders review state, ACP provenance, authenticated drill-through links, versioning, redaction posture, source lineage, export refs, and review-state controls. |
| [#1705](https://github.com/rmusser01/tldw_server/issues/1705) | Accepted export identity | Markdown, HTML, and JSON exports are generated only from accepted versions and preserve artifact/workspace/version identity plus traceability metadata. |
| [#1704](https://github.com/rmusser01/tldw_server/issues/1704) | Release signoff | Focused backend and UI tests now exercise the combined contract and document remaining caveats. |

## Golden-Path Fixture

The release fixture is a source-grounded workspace brief with:

- `artifact_type="workspace_brief"` and `review_state="accepted"`
- `producer_metadata.producer_type="acp"` with ACP task, run, session, review
  run, and canonical workspace IDs
- `source_lineage.sources[]` containing the cited workspace source ID, type,
  display label, and citation spans
- `review_metadata.decision="accepted"` plus reviewer-loop attempt metadata
- version metadata with completion summary and revision reason
- support-safe redaction posture and schema version
- version-specific export refs

The canonical backend fixture lives in
`tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py`.
The UI fixture shape is covered by
`apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/__tests__/TraceableArtifactDetail.test.tsx`
and the server-to-client mapping coverage in
`apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts`.

## Verification Evidence

Commands were run from the isolated worktree
`.worktrees/acp-artifact-release-signoff-1704`.

| Gate | Command | Result |
| --- | --- | --- |
| New ACP artifact promotion contract | `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py -q` | `6 passed`, 5 warnings in 31.20s. |
| Existing backend artifact/API regression set | `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q` | `104 passed`, 5 warnings in 653.88s. |
| UI artifact detail and hydration regression set | `cd apps/packages/ui && bun run test src/components/Option/WorkspacePlayground/StudioPane/__tests__/TraceableArtifactDetail.test.tsx src/store/__tests__/workspace-api-first.test.ts` | 2 files passed, 27 tests passed. |
| Security scan for touched Python test path | `source .venv/bin/activate && python -m bandit -r tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py -s B101 -f json -o /tmp/bandit_acp_artifact_signoff_1704.json` | `results=[]`, `errors=[]`; pytest assertion rule `B101` skipped for test code. |
| Formatting | `git diff --check` | Clean. |

The UI worktree needed `bun install` from `apps/` before the focused Vitest
command because package-local symlinks point at `apps/node_modules`.

## Covered Invariants

- Accepted ACP deliverables create canonical workspace artifacts with stable
  owner/workspace placement and ACP producer metadata.
- Repeated accepted ACP deliverables update the same artifact, create a new
  version row, preserve `root_artifact_id`, and advance
  `previous_version_id`.
- Rejected or needs-revision ACP outputs are not promoted as accepted workspace
  work products.
- Malformed artifact candidates fail closed and do not partially create
  workspace artifacts.
- Only `accepted` artifact versions export through the current
  Markdown/HTML/JSON export contract.
- UI detail suppresses provenance and source-lineage details when redaction
  posture is restricted, while preserving safe state labels.
- UI hydration maps backend snake_case trace fields to the shared WebUI
  camelCase artifact model.

## Remaining Deferrals

The following items remain outside the #1704 release signoff and should not be
claimed as complete by this evidence:

- Non-golden-path ACP artifact types beyond source-grounded briefs, reports,
  specs, action plans, and tables that already satisfy the promotion contract.
- Automatic promotion of meeting, evaluation, persona, raw session, prompt,
  transcript, or tool-payload artifacts.
- Rich export channels such as DOCX, PDF, slides, XLSX, Chatbooks, external
  file-artifact materialization, and export retention cleanup.
- Live downstream-agent certification for named third-party tools, still
  tracked by [#1563](https://github.com/rmusser01/tldw_server/issues/1563) and
  [#1564](https://github.com/rmusser01/tldw_server/issues/1564).
- Full live-backend browser E2E against a seeded release host and installed
  ACP-compatible downstream agent.

Release notes should phrase the artifact support as: accepted structured ACP
deliverables can be promoted into traceable workspace work-product artifacts
with source lineage, review state, versioning, redaction posture, and
Markdown/HTML/JSON export identity. They should not claim all ACP session
artifacts or all generated outputs are canonical workspace work products.
