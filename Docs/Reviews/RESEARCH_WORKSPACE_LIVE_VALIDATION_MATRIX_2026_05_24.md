# Research Workspace Live Validation Matrix - 2026-05-24

## Scope

Validation target: Research Workspace replacement route and connected workspace API flows on a live local backend and WebUI.

Environment:
- Backend: `http://127.0.0.1:8000`
- WebUI: `http://127.0.0.1:3000`
- Route under test: `/research-workspace`
- Removed route under test: `/workspace-playground`
- Browser method: Playwright browser automation. Computer Control was not used.
- Screenshot: `Docs/Reviews/research-workspace-live-validation-2026-05-24.png`

Artifacts:
- API sweep result: `/tmp/research_workspace_live_api_validation.json`
- Extension stale-build smoke result: `/tmp/research_workspace_extension_stale_build_smoke.json`
- Backend log: `/tmp/tldw_backend_live_validation.log`
- WebUI screenshot: `Docs/Reviews/research-workspace-live-validation-2026-05-24.png`

## Summary

| Area | Result | Notes |
| --- | --- | --- |
| Backend availability | Pass | `/health`, `/docs`, and authenticated `/api/v1/workspaces/` returned `200`. |
| WebUI route replacement | Pass | `/research-workspace` returned `200`; `/workspace-playground` returned `404` with no redirect header. |
| WebUI browser smoke | Pass | Research Workspace shell, Sources, Chat, Studio, Add Sources modal, My Media selection, source add, status labels, and source preview worked. |
| Browser console after fresh reload | Pass | After local workspace storage reset and reload, Playwright reported 0 errors and 0 warnings. |
| Workspace API sweep | Pass | 38/38 live API checks passed. |
| Source status projection | Pass | Existing media and missing media statuses returned explicit states; no unsupported `workspace_source_ingest` job type was projected. |
| Migration protocol | Pass | Create, idempotent retry, conflict rejection, chunk receipt, chunk idempotency, chunk conflict, finalize, and delete acknowledgement path were exercised. |
| MCP status and Shared Workspaces | Pass | `/api/v1/mcp/status` returned healthy; `/api/v1/mcp/hub/shared-workspaces` returned `200` with an empty list. |
| Research Workspace capability gates | Pass | `migration`, `sharing`, `mcp`, `acp`, `sandbox`, and `provider` service keys were present in `/capabilities`. |
| ACP workspace route | Gap | `/api/v1/agent-orchestration/workspaces` is absent from live OpenAPI and returned `404`; ACP has session/agent/run APIs but no live workspace CRUD route in this config. |
| Sandbox route | Gap | No live OpenAPI paths containing `sandbox` were exposed. |
| Extension current-build E2E | Blocked | Standard `test:e2e:workspace-parity:real` hung in WXT production build and was terminated. |
| Extension stale-build smoke | Fail, non-conclusive | Existing `build/chrome-mv3` launched and connected to backend, but did not render the current Research Workspace contract. Build timestamp was 2026-04-18, so this is stale-build evidence only. |
| Grounded answer generation | Not verified | UI RAG mode stayed disabled because sources were processing/partially queryable and no model/provider selection was completed in this pass. |

## Matrix

| ID | Scenario | Surface | Method | Result | Evidence | Follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| RW-LIVE-001 | Backend starts and responds | Backend | `curl /health`, `/docs`, authenticated `/api/v1/workspaces/` | Pass | All returned `200`. | None. |
| RW-LIVE-002 | Current Research Workspace route exists | WebUI | `curl http://127.0.0.1:3000/research-workspace` | Pass | Returned `200`. | None. |
| RW-LIVE-003 | Old workspace-playground URL is fully removed | WebUI | `curl -I /workspace-playground` | Pass | Returned `404 Not Found`; no `Location` header. | None. |
| RW-LIVE-004 | No extra workspace trust banner | WebUI | Playwright snapshot and screenshot | Pass | Header, Sources, Chat, Studio, and footer are present; no separate workspace trust bar. | None. |
| RW-LIVE-005 | First-run-ish shell after local storage reset | WebUI | Cleared Research Workspace local storage keys and reloaded `/research-workspace` | Pass | Page rendered `New Research`, Sources, Chat, Studio, skip links, and footer status. | Full first-run onboarding copy still needs product review outside this smoke. |
| RW-LIVE-006 | Add Sources modal opens | WebUI | Playwright click `Add Sources` | Pass | Modal displayed `Upload`, `My Media`, `URL`, `Paste`, `Search Server`. | None. |
| RW-LIVE-007 | My Media list works without Ant Design List warning | WebUI | Playwright modal snapshot and console check | Pass | Native `role=list` / `role=listitem` rendered; console check showed 0 warnings after reload. | None. |
| RW-LIVE-008 | Existing media can be added as a source | WebUI + API proxy | Playwright selected a My Media PDF and clicked `Add 1 selected` | Pass | Source count moved from 1 to 2; proxied `POST /api/v1/workspaces/{id}/sources` returned `201`. | None. |
| RW-LIVE-009 | Source status is visible after add | WebUI + API proxy | Playwright source panel snapshot | Pass | Added source displayed `Processing` with "Text search is available while vector indexing continues." | None. |
| RW-LIVE-010 | Source preview uses friendly status label | WebUI | Playwright opened `Preview & annotate` | Pass | Modal displayed `pdf • Processing`, matching source-row wording. | None. |
| RW-LIVE-011 | Browser console is clean after fresh reload and source workflow | WebUI | `browser_console_messages(level=warning)` after reload and flow | Pass | 0 errors, 0 warnings. | Continue checking this after Ant Design upgrades. |
| RW-LIVE-012 | Workspace CRUD works | API | Live API sweep | Pass | List, upsert, get, patch, delete cleanup passed. | None. |
| RW-LIVE-013 | Source CRUD and bulk operations work | API | Live API sweep | Pass | Add, duplicate add, update, list, selection, reorder, delete passed. | None. |
| RW-LIVE-014 | Missing media status is explicit | API | Live API sweep with `media_id=999999999` | Pass | Status projection returned `missing_media`. | None. |
| RW-LIVE-015 | Unsupported workspace-source job type is not projected | API | Live API sweep and WebUI network inspection | Pass | Status payload had no `workspace_source_ingest` job; WebUI polling returned `200`. | None. |
| RW-LIVE-016 | Workspace capabilities include workspace-model services | API | `GET /api/v1/workspaces/{id}/capabilities` | Pass | Services included `migration`, `sharing`, `mcp`, `acp`, `sandbox`, `provider`. | Capability states remain conservative and not configured. |
| RW-LIVE-017 | Artifact CRUD works | API | Live API sweep | Pass | Create, list, update, delete passed. | None. |
| RW-LIVE-018 | Note CRUD works | API | Live API sweep | Pass | Create, list, update, delete passed. | None. |
| RW-LIVE-019 | Migration session is durable and idempotent | API | Live API sweep | Pass | Create returned `201`; duplicate returned success; conflicting manifest returned `409`. | None. |
| RW-LIVE-020 | Migration chunk receipts are durable and idempotent | API | Live API sweep | Pass | First chunk accepted, duplicate accepted, conflicting chunk returned `409`. | None. |
| RW-LIVE-021 | Migration finalize and delete acknowledgement path responds | API | Live API sweep | Pass | Finalize returned success; delete acknowledgement returned an accepted status for this run. | Add a dedicated E2E for pre-finalize rejection. |
| RW-LIVE-022 | MCP server status is available | API | `GET /api/v1/mcp/status` | Pass | Returned `healthy`, 12/12 modules healthy. | None. |
| RW-LIVE-023 | MCP Hub Shared Workspaces canonical management surface is present | API | `GET /api/v1/mcp/hub/shared-workspaces` | Pass | Returned `200` with `[]`. | Add create/update/delete live coverage once a safe fixture root is defined. |
| RW-LIVE-024 | ACP workspace CRUD route is available | API | OpenAPI inspection and direct curl | Gap | `/api/v1/agent-orchestration/workspaces` absent from OpenAPI and returned `404`; `/api/v1/acp/*` session/agent APIs exist. | Decide whether ACP workspace CRUD should be exposed in this route group or represented through MCP Hub workspace bindings. |
| RW-LIVE-025 | Sandbox management route is available | API | OpenAPI inspection | Gap | No live OpenAPI path contains `sandbox`. | Add first-class sandbox workspace management API or update capabilities copy to point to existing config surface. |
| RW-LIVE-026 | Extension current Research Workspace real-backend E2E runs | Extension | `CI=1 ... bun run test:e2e:workspace-parity:real` | Blocked | WXT production build hung after `Building chrome-mv3 for production`; terminated with `SIGTERM`. | Fix/build-cache WXT hang or add a no-build current-output path for live validation. |
| RW-LIVE-027 | Existing packaged extension can connect to backend | Extension | Direct packaged-extension smoke using `build/chrome-mv3` | Partial | Existing build launched, connection store mounted, connected to backend. | Build timestamp is 2026-04-18; not current enough for release validation. |
| RW-LIVE-028 | Existing packaged extension renders current Research Workspace contract | Extension | Direct packaged-extension smoke | Fail, non-conclusive | Failed waiting for `workspace-workspaces-button`; stale build did not satisfy current contract. | Re-run after current extension build succeeds. |
| RW-LIVE-029 | Grounded chat with citations works | WebUI + backend | UI inspection | Not verified | RAG mode disabled in the live browser pass; no model/provider selection completed. | Add seeded queryable source + deterministic mock provider path. |
| RW-LIVE-030 | Export/import resume across browser storage | WebUI + API | Migration API and local storage smoke | Partial | Migration API passed; full UI import/export handoff was not exercised. | Add browser-driven migration/import wizard coverage. |

## Fixes Already Made During Validation

- Removed unsupported `workspace_source_ingest` job creation from `POST /workspaces/{workspace_id}/sources`; status now reads supported `media_ingest_item` jobs only.
- Replaced deprecated Ant Design `Modal destroyOnClose` with `destroyOnHidden`.
- Replaced deprecated Ant Design `List` usage in Add Sources with native list/listitem markup and keyboard selection.
- Aligned source preview status copy with source-row status labels.

## Remaining Risks

- Extension validation is the largest unresolved risk. The current source cannot be proven through the standard real-backend extension E2E path until the WXT production build hang is fixed.
- ACP and sandbox are represented in Research Workspace capability gates, but the live API surface does not expose workspace CRUD for ACP and exposes no sandbox route. This may be intentional configuration, but the UI/capability copy should not imply those surfaces are ready unless they are reachable.
- Grounded chat/citation behavior still needs a seeded queryable source and deterministic provider harness. Source collection and status work, but answer grounding was not proven here.
- The WebUI polls `sources/status` and `capabilities` frequently. All observed requests returned `200`, but the volume is high enough that later performance work should review polling cadence.

## Commands Run

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspaces.py -f json -o /tmp/bandit_research_workspace_workspaces_endpoint.json
CI=1 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=... bun run test:e2e:workspace-parity:real
```

Results:
- Backend tests: 44 passed, 6 warnings.
- UI tests: 59 passed.
- Bandit: 0 findings.
- Live API sweep: 38 passed, 0 failed.
- Extension current-build E2E: blocked by WXT production build hang.
