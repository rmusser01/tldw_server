# Agent Orchestration

Agent_Orchestration coordinates ACP-backed workspaces, projects, tasks, runs, review gates, and promoted artifacts. It defines the task state machine, validates dependency and review transitions, stores orchestration data in per-user SQLite databases, and connects completed ACP work products to workspace artifacts when they are explicitly accepted.

## Start Here

- `models.py` defines workspaces, projects, tasks, runs, task statuses, run statuses, and valid state transitions.
- `orchestration_service.py` contains the service-level task and run orchestration rules.
- `completion_signals.py` parses structured ACP completion and review markers.
- `artifact_promotion.py` promotes accepted ACP work products into workspace artifacts.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`, declared under `/agent-orchestration`.
- Related persistence: `tldw_Server_API/app/core/DB_Management/Orchestration_DB.py`.
- Related tests: `tldw_Server_API/tests/Agent_Orchestration/`.

## Responsibilities

- Model the task lifecycle: `todo`, `inprogress`, `review`, `triage`, and `complete`.
- Enforce dependency gating so a task cannot start until its dependency is complete.
- Track ACP workspaces, projects, tasks, and agent runs per user.
- Start ACP sessions for task execution and reviewer flows.
- Parse completion and review signals emitted by ACP agents.
- Promote only accepted structured work products into workspace artifacts with source lineage.
- Support workspace health checks, discovery, MCP server metadata, and canonical workspace bridging through the API.

## Module Map

- `models.py`: dataclasses, enums, and state-transition helpers.
- `orchestration_service.py`: orchestration operations, dependency checks, review handling, and run coordination.
- `completion_signals.py`: parsing and validation for structured ACP completion and review markers.
- `artifact_promotion.py`: conversion of accepted ACP work products into durable workspace artifacts.
- `__init__.py`: package marker.

## How It Connects

- `agent_orchestration.py` exposes workspace, project, task, run, review, discovery, MCP server, and canonical bridge routes.
- The endpoint uses AuthNZ permission guards and user-aware dependencies before reaching orchestration behavior.
- `Orchestration_DB.py` provides the durable per-user SQLite store for workspaces, projects, tasks, runs, and canonical workspace links.
- ACP execution is delegated through `Agent_Client_Protocol.runner_client.get_runner_client`.
- Workspace artifact promotion uses `ChaChaNotes_DB` so accepted work can appear in the workspace artifact system while raw ACP artifacts remain evidence.
- Workspace root validation is tied to `[ACP-WORKSPACE]` configuration and `ACP_WORKSPACE_ALLOWED_BASE_PATHS`.

## Architecture Notes

### Core Flow

- `models.py` is the state-machine source of truth. The service and DB layer should reject transitions that do not match the `todo -> inprogress -> review/triage/complete` rules.
- Task start checks dependency readiness before creating an ACP run. A dependent task is ready only when its dependency is complete or the dependency has been removed.
- ACP completion markers are parsed in `completion_signals.py`; reviewer markers are parsed separately so completion and review decisions cannot be conflated.
- Artifact promotion is a post-review step. `artifact_promotion.py` only promotes accepted ACP work products into ChaChaNotes workspace artifacts, preserving run, review, and source lineage.

### State And Data

- `DB_Management/Orchestration_DB.py` owns projects, tasks, runs, reviews, ACP workspaces, and workspace MCP servers in a per-user SQLite store.
- Workspace root paths and canonical workspace links are persisted, while workspace creation is constrained by the allowlist resolved in `agent_orchestration.py`.
- `ACPWorkspace.env_vars` are stored as plaintext JSON. Use them for runner environment convenience, not high-sensitivity secret storage.

### Security And Operations

- Workspace root validation is part of the endpoint boundary. Do not bypass `_validate_workspace_root` when adding alternate workspace creation or discovery paths.
- Review gates are state transitions, not comments. A rejected review moves the task back toward work or triage based on attempt count, and an accepted review is what unlocks artifact promotion.
- Completion-marker and review-marker formats are automation contracts. Prompt changes should be tested against parser and endpoint behavior.

### Extension Checklist

- New task or run state: update `models.py`, `orchestration_service.py`, `DB_Management/Orchestration_DB.py`, API schemas, and service/API tests.
- New completion or review marker: update `completion_signals.py`, endpoint review handling, parser tests, and orchestration API tests.
- New promotable artifact type: update `artifact_promotion.py`, workspace artifact expectations, and `tests/Agent_Orchestration/test_artifact_promotion_contract.py`.

## Extension Points

- Add a task transition by starting in `models.py`, then update `orchestration_service.py`, API schemas, and tests.
- Add new completion or review payload fields in `completion_signals.py` before changing endpoint behavior.
- Add a promotable artifact type in `artifact_promotion.py` and verify the corresponding workspace artifact storage contract.
- Extend workspace discovery or health behavior in `agent_orchestration.py` together with `Orchestration_DB.py`.
- Change run creation or reviewer behavior by inspecting `orchestration_service.py` and ACP runner interactions first.

## Testing

- Direct tests live under `tldw_Server_API/tests/Agent_Orchestration/`.
- Use `test_orchestration_service.py` for service-level task and run rules.
- Use `test_orchestration_api.py` for route behavior.
- Use `test_orchestration_db.py`, `test_workspace_db.py`, and workspace discovery or health tests for persistence changes.
- Use `test_artifact_promotion.py` and `test_artifact_promotion_contract.py` for promoted work-product behavior.

## Gotchas

- `ACPWorkspace.env_vars` are stored as plaintext JSON in the per-user SQLite database; do not put high-sensitivity secrets there.
- Completion and review signals depend on exact structured marker formats, so prompt or parser changes can break automation.
- Workspace roots are constrained by allowed base path configuration.
- Artifact promotion is intentionally gated by acceptance and source lineage; raw ACP artifacts are not automatically promoted.
