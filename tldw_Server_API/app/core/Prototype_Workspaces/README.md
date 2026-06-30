# Prototype Workspaces

Prototype_Workspaces coordinates collaborative prototype workspace sessions, preview handles, external collaborator access, snapshot saves, and promotion flows. It keeps orchestration in core services while persistence lives in the AuthNZ prototype workspace repository and asynchronous runtime actions run through Jobs.

## Start Here

- `models.py` defines actor types, job types, preview scopes, runtime statuses, and shared records.
- `service.py` creates workspaces and branch sessions, saves snapshots, boots previews, renews preview grants, and promotes candidate snapshots.
- `access.py` creates and verifies private-link external collaborator tokens and resume cookies.
- `preview_broker.py` issues opaque preview handles and short-lived signed preview grants.
- `jobs.py` creates stable Jobs entries for runtime operations.
- `jobs_worker.py` dispatches prototype Jobs through `PrototypeWorkspaceService`.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`.
- Related tests: `tldw_Server_API/tests/PrototypeWorkspaces/`.

## Responsibilities

- Seed new owner workspaces with a canonical snapshot.
- Create or reuse owner and external collaborator branch sessions.
- Enforce archived, revoked, expired, and actor-scope checks before session or snapshot operations.
- Save session snapshots and maintain last-saved session state.
- Issue and renew preview handles with signed grants.
- Create, review, reject, or promote external collaborator promotion requests.
- Enqueue and handle branch bootstrap, preview boot, snapshot save, and publish validation jobs.
- Mint and validate private-link external collaborator session tokens and resume cookies.

## Module Map

- `models.py`: enums, dataclasses, actor-key helpers, and preview scope helpers.
- `service.py`: workspace, session, snapshot, preview, and promotion orchestration.
- `access.py`: private-link external collaborator token and resume-cookie exchange.
- `preview_broker.py`: preview handle cache, persistence sync, signed grant creation, and renewal.
- `jobs.py`: Jobs domain constants, idempotency keys, and enqueue helpers.
- `jobs_worker.py`: WorkerSDK dispatcher for prototype runtime job types.

## How It Connects

- `prototype_workspaces.py` exposes owner workspace creation/detail, owner and external branch session creation, promotion request creation/review, and preview renewal routes.
- `AuthNZ/repos/prototype_workspaces_repo.py` persists workspaces, sessions, snapshots, shared actors, promotion requests, and preview handle records.
- Jobs use the `prototype_workspaces` domain and the default queue unless overridden by prototype worker environment variables.
- Sharing and private-link flows are adjacent to this module through endpoint code and AuthNZ repository methods.
- Operational, API, contract, threat-model, and user-facing docs live under `Docs/API-related/`, `Docs/Operations/`, `Docs/Security/`, and `Docs/User_Guides/` with Prototype Workspaces names.

## Extension Points

- Add a runtime job by extending `PrototypeJobType`, `PROTOTYPE_JOB_TYPES`, `PrototypeWorkspaceJobs`, and `handle_prototype_job`.
- Change promotion rules in `service.py`, especially stale-candidate detection and publisher validation.
- Add preview metadata or grant behavior in `preview_broker.py`.
- Change external collaborator policy in `access.py` and endpoint private-link flows.
- Add API fields in `prototype_workspace_schemas.py` and update endpoint contract tests.

## Testing

- Direct coverage lives under `tldw_Server_API/tests/PrototypeWorkspaces/`.
- Tests cover repository behavior, endpoints, authorization, link exchange, preview broker behavior, runtime Jobs, promotion service, documentation contract checks, and release readiness smoke coverage.

## Gotchas

- Preview grants require a stable signing secret from `PROTOTYPE_PREVIEW_SIGNING_SECRET`, `JWT_SECRET_KEY`, or `SINGLE_USER_API_KEY`.
- External collaborator tokens and resume cookies also require a stable signing secret.
- `jobs_worker.py` requires an injected `PrototypeWorkspaceService`; the module intentionally does not provide a standalone worker bootstrap.
- Promotion can return `stale` instead of promoting when the candidate is based on an old canonical baseline.
