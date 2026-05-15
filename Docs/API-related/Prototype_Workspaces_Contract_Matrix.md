# Prototype Workspaces Contract Matrix

## Purpose

This document is the frontend/backend contract artifact for prototype workspace collaboration productionization. Risk Gate 1 creates the draft. Risk Gate 4 freezes it.

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

Risk Gate 1 security model: ../Security/Prototype_Workspaces_Threat_Model.md

## Status

- Draft owner: Backend/Core
- Frontend reviewer: Frontend/Product
- Current gate: Risk Gate 1 draft
- Frozen by: Risk Gate 4

## Error And State Matrix

| State | Backend condition | HTTP status | Stable error category | Frontend state bucket | Retryable | User-facing handling | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_link | Token cannot be verified without confirming existence | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| expired_link | Token/link is expired | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| revoked_link | Token/link or shared actor is revoked | 404 for public link exchange; 403 for active session-token paths | invalid_or_unavailable_link or inactive_session | Link unavailable or session inactive | No | Show generic unavailable link state before exchange; show collaborator session expired after exchange | Draft |
| exhausted_link | Link has no remaining collaborator uses and same-browser resume is unavailable | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| password_required | Protected link exchange has no password and no valid same-browser resume | 403 | password_required | Password required | Yes | Prompt for password | Draft |
| bad_password | Protected link exchange password does not verify | 403 | invalid_password | Password rejected | Yes | Keep password prompt visible with inline error | Draft |
| archived_workspace | Workspace is archived | 403 currently; Risk Gate 4 decides whether public archived links become 404 | workspace_unavailable | Workspace unavailable | No | Show unavailable workspace state | Draft |
| missing_workspace | Token points at a workspace that no longer exists | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| owner_mismatch | Token owner does not own the prototype workspace | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| inactive_session | Session token is expired or maps to revoked/expired shared actor or session | 403 | inactive_session | Session inactive | No | Ask collaborator to request a fresh link/session | Draft |
| bootstrap_failed | Branch session bootstrap failed | 409 or 500-class mapped safe response | bootstrap_failed | Setup failed | Yes, if backend marks retryable | Offer retry when allowed | Draft |
| preview_unavailable | Preview handle missing, revoked, or unhealthy | 409 or 503 | preview_unavailable | Preview unavailable | Yes, if backend marks retryable | Show preview retry/status state | Draft |
| stale_promotion | Candidate is stale versus canonical snapshot | 409 | stale_promotion | Promotion stale | No | Ask user to resubmit from current branch | Draft |
| promotion_conflict | Promotion validation detects conflict | 409 | promotion_conflict | Promotion conflict | No | Show conflict/review state | Draft |
| promotion_validation_failed | Validation failed without promoting | 409 | promotion_validation_failed | Promotion failed | Yes, if backend marks retryable | Show validation failure details | Draft |

## Token And Session Security Dispositions

| Requirement | Disposition | Gate | Notes |
| --- | --- | --- | --- |
| Token storage/hash rules | Document existing behavior | Risk Gate 1 | Share tokens are verified by hash and list responses strip token/password hashes. |
| TTLs | Enforce now | Risk Gate 1 | Share token expiry, shared actor expiry, prototype session expiry, and collaborator session-token `exp` are checked in active paths. |
| Replay handling | Document existing behavior; defer stronger replay controls | Risk Gates 1 and 4 | First-time max-use links use claim/release. Resume cookies rotate binding secrets. Session tokens are valid until expiry but recheck active actor state. Risk Gate 4 decides whether nonce tracking is required. |
| Cookie flags | Document existing behavior | Risk Gate 1 | Resume cookies are `HttpOnly`, `SameSite=Lax`, seven-day max-age, and `Secure` when HTTPS is detected by scheme or forwarded proto. |
| Referrer leakage controls | Defer | Risk Gate 5 | Frontend should remove token-bearing route state after exchange and avoid storing passwords/tokens in durable client state. |
| Password-protected link behavior | Enforce now | Risk Gate 1 | First exchange requires password. Same-browser resume may skip password only with a valid signed resume cookie for the same active actor and link. |
| Signing secret rotation | Defer | Risk Gate 7 | Stable signing secrets are required now; rotation and operator checks belong to operational readiness. |
| Revocation propagation | Enforce now for auth-sensitive paths; defer cleanup/visibility | Risk Gates 1, 2, 3, and 7 | Branch session creation, snapshot save, preview grants, and promotion submission reject revoked actors. Retention and observability are later gates. |

## Frontend Fixture Notes

- Fixture schema owner: Frontend/Product.
- Mock state owner: Frontend/Product.
- Risk Gate 1 seed fixture: `apps/tldw-frontend/e2e/fixtures/prototype-workspaces/contract-states.json`.
- Contract feedback deadline: before Risk Gate 4 contract freeze.
- Required fixture states: `invalid_link`, `expired_link`, `revoked_link`, `exhausted_link`, `password_required`, `bad_password`, `archived_workspace`, `inactive_session`, `preview_unavailable`, `stale_promotion`, `promotion_conflict`, `promotion_validation_failed`, and `bootstrap_failed`.
- Route-state audit checklist:
  - Token-only collaborator entry must call the workspace detail hook with `null` and must not fall back to a stale owner workspace id.
  - Password handoff from `/share/:token` must remain transient and must not be written to persistent workspace state.
  - Collaborator entry must clear or overwrite stale session/share-token state when the URL changes.
  - Owner view must not render for `session_token` or `share_token` entries before workspace detail is loaded.
- Open frontend questions for Risk Gate 4:
  - Should public archived prototype links render the same user-facing bucket as invalid/exhausted links?
  - Does the UI need distinct copy for inactive session token vs revoked private link?
  - Which retryable backend fields are required to make bootstrap and preview retry buttons deterministic?

## Gate 4 Freeze Checklist

- [ ] All stable error categories are final.
- [ ] HTTP statuses are final or explicitly documented as non-enumerating policy choices.
- [ ] Retryability is final.
- [ ] Frontend state buckets are final.
- [ ] Backend/Core reviewer recorded.
- [ ] Frontend/Product reviewer recorded.
