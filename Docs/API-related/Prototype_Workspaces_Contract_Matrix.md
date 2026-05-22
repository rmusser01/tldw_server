# Prototype Workspaces Contract Matrix

## Purpose

This document is the frontend/backend contract artifact for prototype workspace collaboration productionization. Risk Gate 1 creates the draft. Risk Gate 4 freezes it.

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

Risk Gate 1 security model: ../Security/Prototype_Workspaces_Threat_Model.md

## Status

- Draft owner: Backend/Core
- Frontend reviewer: Frontend/Product
- Current gate: Risk Gate 4 contract freeze
- Frozen by: Risk Gate 4
- Fixture: `apps/tldw-frontend/e2e/fixtures/prototype-workspaces/contract-states.json` version 2
- Operational docs: `Docs/Operations/Prototype_Workspaces_Runbook.md`
- User guide: `Docs/User_Guides/Prototype_Workspaces.md`

## Structured Error Detail

Prototype-specific HTTP errors use FastAPI's normal outer envelope with a stable object in `detail`:

```json
{
  "detail": {
    "category": "inactive_session",
    "message": "Prototype session token is no longer active",
    "frontend_state": "session_inactive",
    "retryable": false
  }
}
```

`category`, `frontend_state`, and `retryable` are contract fields. `message` is safe user-facing fallback copy but frontend states should not branch on it.

## Error And State Matrix

| State | Backend condition | HTTP status | Stable error category | Frontend state bucket | Retryable | User-facing handling | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_link | Token cannot be verified without confirming existence | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Frozen |
| expired_link | Token/link is expired | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Frozen |
| revoked_link | Token/link or shared actor is revoked | 404 for public link exchange; 403 for active session-token paths | invalid_or_unavailable_link or inactive_session | Link unavailable or session inactive | No | Show generic unavailable link state before exchange; show collaborator session expired after exchange | Frozen |
| exhausted_link | Link has no remaining collaborator uses and same-browser resume is unavailable | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Frozen |
| password_required | Protected link exchange has no password and no valid same-browser resume | 403 | password_required | Password required | Yes | Prompt for password | Frozen |
| bad_password | Protected link exchange password does not verify | 403 | invalid_password | Password rejected | Yes | Keep password prompt visible with inline error | Frozen |
| archived_workspace | Workspace is archived | 403 | workspace_unavailable | Workspace unavailable | No | Show unavailable workspace state | Frozen |
| missing_workspace | Token points at a workspace that no longer exists | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Frozen |
| owner_mismatch | Token owner does not own the prototype workspace | 404 | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Frozen |
| inactive_session | Session token is expired or maps to revoked/expired shared actor or session | 403 | inactive_session | Session inactive | No | Ask collaborator to request a fresh link/session | Frozen |
| bootstrap_failed | Branch session bootstrap failed | 409 | bootstrap_failed | Setup failed | Yes, if backend marks retryable | Offer retry when allowed | Frozen |
| preview_unavailable | Preview handle is missing or revoked; renewal conflicts with current preview state | 404 for missing/revoked handles; 409 for renewal conflicts | preview_unavailable | Preview unavailable | No for missing/revoked handles; Yes for renewal conflicts | Show preview unavailable state and retry only when `retryable` is true | Frozen |
| stale_promotion | Candidate is stale versus canonical snapshot | 200 promotion-review response with `status = "stale"`; future HTTP errors use 409 | stale_promotion | Promotion stale | No | Ask user to resubmit from current branch | Frozen |
| promotion_conflict | Promotion validation detects conflict | 409 | promotion_conflict | Promotion conflict | No | Show conflict/review state | Frozen |
| promotion_validation_failed | Validation failed without promoting | 200 promotion-review/job response with `status = "failed"`; future HTTP errors use 409 | promotion_validation_failed | Promotion failed | Yes, if backend marks retryable | Show validation failure details | Frozen |

## Runtime Job Result Matrix

| Job type | Success status | Terminal failure fields | Retryable failure fields | Idempotency basis | Frontend/operator note |
| --- | --- | --- | --- | --- | --- |
| `branch_session_bootstrap` | `status = "ok"`, `job_type`, `retryable = false`, `session_id`, `created` | `failure_code = "invalid_job_payload"`, `permission_denied`, or `runtime_terminal` | `failure_code = "runtime_retryable"`, `retryable = true` | workspace, actor, baseline snapshot, request nonce | Safe to retry from Jobs when retryable; terminal actor/workspace state needs a fresh session/link. |
| `preview_boot` | `status = "ok"`, `job_type`, `retryable = false`, `preview_handle`, `preview_url` | `failure_code = "invalid_job_payload"` or `runtime_terminal` | `failure_code = "runtime_retryable"`, `retryable = true` | preview scope, snapshot, runtime profile version, target fingerprint | Same target retries renew the active handle; changed targets replace the handle for the scope. |
| `snapshot_save` | `status = "ok"`, `job_type`, `retryable = false`, `snapshot_id` | `failure_code = "invalid_job_payload"` or `runtime_terminal` | `failure_code = "runtime_retryable"`, `retryable = true` | session and save request id; explicit snapshot id is reused on handler retry | UI can keep showing one saved revision for duplicate completion/retry paths. |
| `publish_validate_and_promote` | `status = "promoted"`, `job_type`, `retryable = false`, canonical/candidate ids, optional `preview_handle` | `status = "failed"` or `"stale"`, `failure_code`, `retryable = false`, canonical/candidate ids | Reserved for future validators that explicitly return `retryable = true`; default validation failures are terminal | workspace, candidate snapshot, review baseline/canonical snapshot | Failed validation never advances canonical or last-known-good pointers. |

## Operational Support Fields

Risk Gate 7 does not change the frozen error/state contract. It records the fields operators and Frontend/Product reviewers should use when diagnosing support cases and preparing Gate 8 release evidence.

| Surface | Fields | Support use |
| --- | --- | --- |
| Workspace detail | `canonical_preview_status`, `publish_validation_status` | Check whether the shared preview and last publish validation are healthy before inspecting branch state. |
| Prototype session | `runtime_status`, `preview_status`, `last_saved_snapshot_id`, `expires_at`, `revoked_at` | Determine whether a branch is booting, active, failed, expired, revoked, or missing a saved candidate. |
| Prototype snapshot | `preview_health` | Separate saved candidate existence from preview boot health. |
| Promotion request | `status`, `reviewed_by_user_id`, `review_notes` | Explain pending, rejected, approved, promoted, failed, stale, and conflict outcomes to owners and collaborators. |
| Job response/result | `job_id`, `job_type`, `status`, `idempotency_key`, `retryable` | Correlate UI actions with Jobs worker outcomes and decide whether retry is safe. |
| Structured error detail | `category`, `frontend_state`, `retryable` | Map user-visible states to backend conditions without parsing messages. |
| Audit/support breadcrumb | workspace id, session id, shared actor id, share-link id, promotion request id, `job_id` | Correlate logs and audit records without recording raw tokens, passwords, grants, or cookies. |

Gate 8 handoff:

- Backend/Core must preserve these field names or open a contract follow-up before changing them.
- Frontend/Product must provide release evidence for password-protected entry, single-use and exhausted links, resume cookie continuation, revoked links, archived workspaces, promotion conflicts, and validation failures.
- Operators should use the runbook for signing secret, quota, preview health, and promotion triage until a dedicated admin dashboard exists.

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
- Risk Gate 4 frozen fixture: `apps/tldw-frontend/e2e/fixtures/prototype-workspaces/contract-states.json`.
- Contract feedback deadline: complete for Risk Gate 4. Later gates should open follow-up issues instead of changing this contract inline.
- Required fixture states: `invalid_link`, `expired_link`, `revoked_link`, `exhausted_link`, `password_required`, `bad_password`, `archived_workspace`, `inactive_session`, `preview_unavailable`, `stale_promotion`, `promotion_conflict`, `promotion_validation_failed`, and `bootstrap_failed`.
- Route-state audit checklist:
  - Token-only collaborator entry must call the workspace detail hook with `null` and must not fall back to a stale owner workspace id.
  - Password handoff from `/share/:token` must remain transient and must not be written to persistent workspace state.
  - Collaborator entry must clear or overwrite stale session/share-token state when the URL changes.
  - Owner view must not render for `session_token` or `share_token` entries before workspace detail is loaded.
- Frozen Risk Gate 4 decisions:
  - Public archived prototype links use `workspace_unavailable` with HTTP 403 because the caller has a valid private token but the workspace is no longer available.
  - Invalid, expired, revoked, exhausted, missing, and owner-mismatched public links remain non-enumerating as `invalid_or_unavailable_link`.
  - Active collaborator session-token paths use `inactive_session` for expired/revoked actor or session state.
  - Retry buttons must branch on the `retryable` field, not on HTTP status or message text.

## Gate 4 Freeze Checklist

- [x] All stable error categories are final.
- [x] HTTP statuses are final or explicitly documented as non-enumerating policy choices.
- [x] Retryability is final.
- [x] Frontend state buckets are final.
- [x] Backend/Core reviewer recorded: Risk Gate 4 implementation owner.
- [ ] Frontend/Product reviewer recorded.
