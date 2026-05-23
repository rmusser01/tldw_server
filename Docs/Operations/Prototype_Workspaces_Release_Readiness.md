# Prototype Workspaces Release Readiness

## Purpose

This document records Risk Gate 8 release-readiness evidence for prototype workspace collaboration. It is intentionally evidence-focused: the goal is to prove the owner-to-collaborator-to-promotion path works with CI-friendly stubs and to identify any remaining production risks before closing #1461.

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

Risk Gate 8: https://github.com/rmusser01/tldw_server/issues/1461

## Backend Verification Matrix

| Area | Evidence | Status |
| --- | --- | --- |
| Owner workspace API | `POST /api/v1/prototype-workspaces`; `GET /api/v1/prototype-workspaces/{id}` | Covered by focused prototype endpoint tests and Gate 8 smoke path. |
| Public share exchange | `POST /api/v1/sharing/public/{token}/prototype-session` with prototype share tokens | Covered by focused link-exchange tests and Gate 8 smoke path. |
| Collaborator branch session | `POST /api/v1/prototype-sessions` with collaborator `session_token` | Covered by focused endpoint tests and Gate 8 smoke path. |
| Candidate snapshot persistence | `PrototypeWorkspaceService.save_session_snapshot` with runtime/preview stub data | Covered by focused service tests and Gate 8 smoke path. |
| Promotion request submission | `POST /api/v1/prototype-promotions` with collaborator session token | Covered by focused endpoint tests and Gate 8 smoke path. |
| Owner review failure | Failing publish validator returns `status = "failed"` and preserves canonical state | Covered by focused promotion tests and Gate 8 smoke path. |
| Owner review success | Passing publish validator returns `status = "promoted"` and advances canonical state | Covered by focused promotion tests and Gate 8 smoke path. |
| Non-enumerating security path | Revoked and expired prototype links return `invalid_or_unavailable_link` / `link_unavailable` | Covered by focused link-exchange tests and Gate 8 negative smoke path. |
| Preview broker | Preview grants stay brokered through opaque handles | Covered by focused preview broker tests and owner review smoke success. |
| Runtime jobs | Jobs are queued with stable job types and idempotency keys | Covered by focused runtime jobs tests; external runtime host is stubbed for Gate 8 smoke. |

## Frontend Verification Matrix

| Area | Evidence | Status |
| --- | --- | --- |
| Public share handoff | `PublicShare` routes prototype links to `/prototype-workspaces?share_token=...` | Covered by focused PublicShare tests and Gate 8 frontend verification. |
| Collaborator route state | New share/session-token entries do not reuse stale owner workspace or stale mutation state | Covered by `PrototypeWorkspaceSessionView` tests and Gate 8 frontend verification. |
| Contract fixture states | Frozen Risk Gate 4 states remain available to frontend tests | Covered by `contract-states.json` and docs-contract tests. |
| Owner review UI | Pending, terminal, validation, conflict, and failure states render distinctly | Covered by `PrototypeWorkspaceOwnerView` tests and Gate 8 frontend verification. |
| Owner action gating | Review actions are disabled when state/backend semantics would reject them | Covered by `PrototypeWorkspaceOwnerView` tests and Gate 8 frontend verification. |
| Browser-observed UX | Final manual browser pass for owner and collaborator flows | Remaining Gate 8 evidence item. |

## Gate 8 CI Smoke Path

The first Gate 8 smoke test must exercise this path without external runtime services:

1. Owner creates a prototype workspace.
2. Owner creates a private `prototype_workspace` share link.
3. Collaborator exchanges the link for a scoped shared actor and session token.
4. Collaborator creates an isolated branch session.
5. Runtime/preview stub saves a candidate snapshot.
6. Collaborator submits a promotion request.
7. Owner approval with a failing publish validator returns `status = "failed"` and does not advance canonical state.
8. Owner approval with a passing publish validator returns `status = "promoted"` and advances canonical state.

The negative smoke path must verify revoked and expired prototype share links fail without confirming whether the token, workspace, or actor exists.

## Verification Log

2026-05-22:

- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_release_readiness_smoke.py -q`
  - RED: failed before the in-memory app/fixture harness was completed.
  - GREEN: `2 passed, 5 warnings`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_release_readiness_smoke.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py -q`
  - GREEN: `5 passed, 5 warnings`.
- `git diff --check`
  - GREEN: no output.
- `bun install --frozen-lockfile` from `apps/`
  - Completed; hydrated ignored workspace dependencies from the existing lockfile.
- `./node_modules/.bin/vitest run src/components/Option/__tests__/PublicShare.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceSessionView.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx --maxWorkers=1 --no-file-parallelism` from `apps/packages/ui/`
  - GREEN: `5 passed (5)`, `30 passed (30)`.
- Review follow-up: `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/tests/PrototypeWorkspaces/test_release_readiness_smoke.py -s B101 -f json -o /tmp/bandit_prototype_gate8_review.json`
  - GREEN: `results: []`; `B101` skipped because the touched backend Python path is a pytest file where `assert` is expected.

## Remaining Risks For Gate 8

- Browser-observed UX evidence still needs to be captured against the local frontend route.
- Final security review should confirm private-link/session-token flows against the latest merged code.
- If production deployment requires operator dashboards instead of documented log/status-field workflows, that should be filed as a follow-up rather than folded into this smoke-coverage slice.
