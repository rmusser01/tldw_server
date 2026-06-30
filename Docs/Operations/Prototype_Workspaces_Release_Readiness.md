# Prototype Workspaces Release Readiness

## Purpose

This document records Risk Gate 8 release-readiness evidence for prototype workspace collaboration. It is intentionally evidence-focused: the goal is to prove the owner-to-collaborator-to-promotion path works with CI-friendly stubs and to identify any remaining production risks before closing #1461.

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

Risk Gate 8: https://github.com/rmusser01/tldw_server/issues/1461

Final signoff follow-up: https://github.com/rmusser01/tldw_server/issues/1977

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
| Browser-observed UX | Final browser pass for owner and collaborator flows against the real WebUI route with API-shaped Playwright stubs | Complete; see 2026-05-23 signoff evidence. |

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

## 2026-05-23 Final Browser Signoff Evidence

Browser evidence was captured against the local Next WebUI at `http://127.0.0.1:18027` using the real `/share/{token}` and `/prototype-workspaces` pages with Playwright network stubs for the prototype APIs. Screenshots were written to `/private/tmp/prototype-final-signoff-1977/`.

During the first probe, `/share/proto-token` rendered the prototype public handoff, but `/prototype-workspaces` resolved to the WebUI 404 because the shared route had no Next page shim. This was fixed with `apps/tldw-frontend/pages/prototype-workspaces.tsx` and guarded by `apps/tldw-frontend/__tests__/navigation/prototype-workspaces-route.test.tsx`.

The full browser pass then observed:

- Owner empty/create route: `owner-empty.png`, `owner-created.png`.
- Owner private-link creation: `owner-share-link.png` with `/share/proto-token`.
- Owner promotion review: `owner-review-pending.png`.
- Owner validation failure handling: `owner-validation-failed.png`.
- Owner promotion success handling: `owner-promoted.png`.
- Public share handoff: `collaborator-public-share.png` to `/prototype-workspaces?share_token=proto-token`.
- Collaborator link exchange and session creation: `collaborator-link-handoff.png`, `collaborator-link-exchanged.png`, `collaborator-session-started.png`.
- Collaborator promotion submission: `collaborator-promotion-submitted.png`.

The collaborator pass also verified that the UI remains on the collaborator session surface after token-bearing route cleanup. This is guarded by `PrototypeWorkspacePage.test.tsx` so `/prototype-workspaces?workspace=...` does not fall back to the owner view when the collaborator session is already stored.

## 2026-05-23 Private-Link And Session-Token Signoff

| Invariant | Evidence | Signoff |
| --- | --- | --- |
| Prototype token ownership | `create_token` calls `_get_owned_prototype_workspace()` for `prototype_workspace` resources before `ShareTokenService.generate_token()`. | Pass. Non-owners cannot mint prototype links for another owner's workspace. |
| Revocation | `ShareTokenService.validate_token()` rejects revoked tokens; public exchange returns `link_unavailable`; repo/session/service paths reject revoked shared actors, sessions, and preview handles. | Pass. Covered by link-exchange, endpoint, service, and preview broker tests. |
| Expiration | Share-token expiry, session-token `exp`, shared actor expiry, active-session lookup, and preview grant expiry are all checked before use. | Pass. Expired links and actors fail without confirming resource existence. |
| Resume-cookie flags | `public_prototype_session_exchange()` sets `prototype_shared_actor` with `HttpOnly`, `SameSite=Lax`, and `secure=_request_is_secure(request)`, where `_request_is_secure()` honors `X-Forwarded-Proto: https` before falling back to request scheme. | Pass. Secure-cookie behavior matches proxy-aware HTTPS detection. |
| Non-enumerating errors | Public prototype exchange maps missing, revoked, expired, exhausted, missing-workspace, and owner-mismatch cases to stable `invalid_or_unavailable_link` / `link_unavailable` responses. | Pass. Covered by focused link-exchange tests and the Gate 8 negative smoke path. |
| Preview grants | `PrototypePreviewBroker` issues opaque handles, signs short-lived grants, persists active scope handles, renews only still-authorized handles, and invalidates grants when sessions/shared actors are revoked or expired. | Pass. Covered by preview broker and endpoint renewal tests. |
| Promotion authority | Promotion submission validates the session token workspace, branch session, shared actor, snapshot ownership, and active actor/session state; review is limited to workspace owner or designated promoters. | Pass. Covered by endpoint and promotion service tests. |

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

2026-05-23:

- `bun install --frozen-lockfile` from `apps/`
  - GREEN: workspace dependencies hydrated from the existing lockfile.
- Initial Playwright route probe via `/private/tmp/prototype-signoff-probe.mjs`
  - RED: `/prototype-workspaces` returned the WebUI 404 while `/share/proto-token` rendered the prototype handoff.
- `bunx vitest run __tests__/navigation/prototype-workspaces-route.test.tsx --maxWorkers=1 --no-file-parallelism` from `apps/tldw-frontend/`
  - RED before adding the page shim; GREEN after adding `pages/prototype-workspaces.tsx`: `1 passed`.
- `bunx vitest run src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx --maxWorkers=1 --no-file-parallelism` from `apps/packages/ui/`
  - RED before preserving stored collaborator context after token cleanup; GREEN after the page-state fix: `6 passed`.
- `bunx vitest run src/components/Option/__tests__/PublicShare.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceSessionView.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx --maxWorkers=1 --no-file-parallelism` from `apps/packages/ui/`
  - GREEN: `5 passed (5)`, `31 passed (31)`.
- Full Playwright signoff flow via `/private/tmp/prototype-signoff-flow.mjs`
  - GREEN: owner create/share/review/failure/success and collaborator public-share/link-exchange/session/promotion states observed; screenshots written to `/private/tmp/prototype-final-signoff-1977/`.
- `../../.venv/bin/python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_release_readiness_smoke.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py -q`
  - GREEN: `5 passed, 5 warnings`.
- `git diff --check`
  - GREEN: no output.
- `../../.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/core/Sharing/share_token_service.py tldw_Server_API/app/core/Prototype_Workspaces/access.py tldw_Server_API/app/core/Prototype_Workspaces/preview_broker.py -f json -o /tmp/bandit_prototype_security_review_1977.json`
  - GREEN: `results: []`.

## Remaining Risks For Gate 8

- No production-blocking browser UX or private-link/session-token security issues remain from #1977.
- Operator dashboards beyond documented log/status-field workflows remain a possible future enhancement, not a Gate 8 blocker.
