## Stage 1: Contract Tests
**Goal**: Add focused API tests for Research Workspace source status and capability responses.
**Success Criteria**: Tests assert `/api/v1/workspaces/{workspace_id}/sources/status` reports source lifecycle/readiness and `/api/v1/workspaces/{workspace_id}/capabilities` fails closed when no queryable sources exist.
**Tests**: `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`
**Status**: Complete

## Stage 2: Status Projection
**Goal**: Implement a small read-computed source status projector backed by workspace sources, optional Media DB details, and optional media-ingest Jobs.
**Success Criteria**: Missing media, pending extraction/chunking/indexing, queryable, partially queryable, and failed states are deterministically mapped without a new table.
**Tests**: Focused unit/API tests from Stage 1.
**Status**: Complete

## Stage 3: Workspace Capability Endpoint
**Goal**: Expose a conservative workspace capability summary for first-run and power-user UI gates.
**Success Criteria**: Response includes workspace kind, access level, source summary, provider/MCP/ACP/sandbox state, and allowed actions with reason codes.
**Tests**: Focused API tests plus existing workspace endpoint tests.
**Status**: Complete

## Stage 4: Verification
**Goal**: Validate touched backend scope and record results in Backlog.
**Success Criteria**: Focused pytest passes; Bandit runs on touched backend files; any known skips are documented.
**Tests**: `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py`; Bandit touched-scope scan.
**Status**: Complete

Verification completed:
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py -q` -> 3 passed
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q` -> 39 passed
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/core/Workspaces -f json -o /tmp/bandit_research_workspace_status.json` -> no findings
- Live uvicorn HTTP validation against `/api/v1/workspaces/{workspace_id}/sources/status` and `/api/v1/workspaces/{workspace_id}/capabilities` succeeded with lifespan disabled to isolate the route table from unrelated startup services.
