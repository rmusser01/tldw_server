# Prototype Workspaces Operations Runbook

## Purpose

This runbook is for operators and support engineers running the Prototype Workspaces collaboration surface. It covers runtime setup, link/session diagnosis, preview health, promotion review failures, and the support fields available before a full admin dashboard exists.

Canonical contracts:

- API overview: `Docs/API-related/Prototype_Workspaces_API.md`
- Frontend/backend state matrix: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
- Security model: `Docs/Security/Prototype_Workspaces_Threat_Model.md`

## Setup Checklist

Before enabling public prototype collaboration:

- Apply AuthNZ migrations `086` and `087` so prototype workspace tables and the `prototype_workspace` share-token resource type exist.
- Confirm the API process has a stable signing secret. `PrototypeAccessService` requires `JWT_SECRET_KEY` or `SINGLE_USER_API_KEY`; preview grants use an explicit preview secret or the same stable server secret fallback.
- Run Jobs workers for domain `prototype_workspaces` on queue `default`.
- Confirm runtime policy names used by workspaces are configured server-side, especially `runtime_policy.owner_profile` and `runtime_policy.external_collaborator_profile`.
- Configure share-token use counts, session expiry, preview-grant TTLs, and cleanup retention before exposing links outside trusted testers.
- Keep frontend routes `/prototype-workspaces` and `/share/:token` behind the intended rollout flag until Gate 8 release evidence is complete.

## Signing Secret Posture

Prototype collaborator session tokens, resume cookies, and preview grants are signed. The signing secret must be stable across API restarts or active collaborators will lose access and preview renewals will fail.

Operational rules:

- Do not use an ephemeral generated secret in deployed environments.
- Treat a signing secret rotation as a planned session invalidation event unless a dedicated rotation flow exists.
- Before rotation, drain or pause new public link exchange, notify owners that active collaborator sessions may need new links, and let Jobs workers finish in-flight prototype jobs where practical.
- After rotation, revoke or expire old share links whose collaborators cannot resume cleanly, then ask owners to create fresh private links.
- Never log raw share tokens, session tokens, preview grants, passwords, or cookie values while diagnosing signing failures.

Current gap for Gate 8: automated secret-age checks and dual-secret token verification are not implemented. Operators must validate the configured secret manually during rollout.

## Runtime And Jobs Behavior

Prototype runtime work uses the shared Jobs module rather than a prototype-specific queue.

Job types:

- `branch_session_bootstrap`: creates or reuses an owner or collaborator branch session and initializes runtime state.
- `preview_boot`: starts or renews a preview target and returns a brokered preview handle.
- `snapshot_save`: persists a session-owned candidate snapshot.
- `publish_validate_and_promote`: validates a candidate and advances canonical state only on success.

Job responses and result payloads should expose:

- `job_id`
- `job_type`
- `status`
- `idempotency_key`
- `retryable`

Use `idempotency_key` to identify duplicate requests and worker retries. Retry only when the job result or structured error sets `retryable = true`. Terminal runtime failures such as archived workspaces, revoked actors, expired sessions, missing snapshots, and permission failures require a fresh link, fresh session, or owner action.

## Status Fields For Support

Use these fields before escalating to database inspection.

| Area | Fields | How to use them |
| --- | --- | --- |
| Workspace | `canonical_preview_status`, `publish_validation_status` | Determine whether the shared canonical preview is healthy and whether the last promotion validation succeeded, failed, or became stale. |
| Session | `runtime_status`, `preview_status`, `last_saved_snapshot_id`, `expires_at`, `revoked_at` | Identify active, bootstrapping, failed, expired, or revoked owner/collaborator branches. |
| Snapshot | `preview_health` | Distinguish a saved candidate that exists from a candidate whose preview cannot boot. |
| Promotion request | `status`, `reviewed_by_user_id`, `review_notes` | Confirm whether the owner rejected, approved, promoted, failed, or left a request pending. The `promotion_requests` table is the durable support source. |
| Job response/result | `job_id`, `job_type`, `status`, `idempotency_key`, `retryable` | Correlate UI operations with worker outcomes and decide whether a retry is safe. |
| Structured error | `category`, `frontend_state`, `retryable` | Match user-visible failures to the frozen Risk Gate 4 matrix without parsing message text. |

Audit breadcrumbs are available through the existing audit/logging paths and should be correlated with these status fields. The expected audit subjects are workspace creation, share-link exchange, shared-actor resume/revocation, branch bootstrap, preview grant renewal, snapshot save, promotion submission, and promotion review. Gate 7 documents the taxonomy and support workflow; Gate 8 should attach release evidence for the exact dashboards, logs, or queries operators will use in production.

## Link And Session Triage

Use the structured error `category`, `frontend_state`, and `retryable` fields first.

| Symptom | Likely state | Operator response |
| --- | --- | --- |
| Collaborator sees unavailable link before entering | Invalid, expired, revoked, exhausted, missing, or owner-mismatched link | Keep response non-enumerating. Ask the owner to confirm the link is current and issue a new link if needed. |
| Collaborator sees password prompt | `password_required`, retryable | This is expected for password-protected links without a valid resume cookie. |
| Collaborator sees password rejected | `invalid_password`, retryable | Ask the owner to confirm the password out of band. Do not log the submitted password. |
| Single-use link worked once, then fails in another browser | `exhausted_link` | Expected for max-use links. Same-browser resume may still work if the resume cookie is valid. |
| Same browser resumes without password | Valid resume cookie | Expected when browser-session resume is allowed and the actor/link/session are still active. |
| Revoked link fails after owner action | `invalid_or_unavailable_link` before exchange or `inactive_session` after exchange | Expected. Ask the owner to create a new link if collaboration should continue. |
| Archived workspace link fails | `workspace_unavailable` | Expected. Owner must unarchive or create a new workspace. |
| Active collaborator session suddenly fails | `inactive_session` | Check `expires_at`, `revoked_at`, actor revocation, and signing secret stability. A fresh link/session is usually required. |

## Preview Health Triage

Preview access is intentionally brokered. Clients should never receive raw runtime target URLs.

When preview renewal fails:

1. Check the frontend/backend state matrix for `preview_unavailable`.
2. Inspect the session `preview_status` and snapshot `preview_health`.
3. Check the related `preview_boot` job by `job_id` or `idempotency_key`.
4. Retry only when the result marks `retryable = true`.
5. If the handle is missing, revoked, expired, or bound to a stale target, start a fresh preview flow from the current workspace/session state.

If the canonical preview is unhealthy but collaborator branch previews work, inspect workspace `canonical_preview_status` and the latest promoted snapshot. A failed collaborator preview does not change the canonical preview unless a promotion was approved and validation succeeded.

## Promotion Triage

Promotion is owner-reviewed and validation-gated.

Common support cases:

- `stale`: the canonical snapshot changed after the collaborator started from an older baseline. The collaborator should create a new branch from the current canonical snapshot.
- `promotion_conflict`: validation found a conflict that cannot be auto-promoted. The owner should review the candidate and ask for changes or reject the request.
- `promotion_validation_failed`: validation failed without advancing canonical or last-known-good pointers. Retry only when the response marks `retryable = true`.
- rejected request: review `status`, `reviewed_by_user_id`, and `review_notes` to explain owner intent without inspecting runtime internals.

Never manually update canonical snapshot pointers to bypass validation. If an approved promotion does not advance canonical state, correlate the `publish_validate_and_promote` job and workspace `publish_validation_status`.

## Quotas, Rate Limits, And Cleanup

Current quota and rate-limit controls are intentionally conservative:

- Public share exchange can return 429 when the configured rate limit is exceeded.
- Share links can have max-use limits; exhausted links remain non-enumerating to collaborators.
- Session expiry and preview-grant TTLs limit stale collaborator access.
- Cleanup retention can revoke or delete inactive prototype collaboration rows according to the persistence contract.

Current gaps for Gate 8:

- There is no dedicated quota dashboard for prototype workspace collaboration.
- Operators need deployment-specific log or database queries to summarize exhausted links, near-expiry sessions, and repeated preview failures.
- Rate-limit tuning should be validated against the expected number of external stakeholders per workspace before wider rollout.

## Incident Checklist

1. Identify whether the reporter is the owner, designated promoter, or external collaborator.
2. Capture workspace id, session id, promotion request id, job_id, and approximate timestamp. Do not capture tokens, passwords, preview grants, or cookies.
3. Map the user-visible state to the contract matrix using `category`, `frontend_state`, and `retryable`.
4. Check workspace, session, snapshot, promotion request, and job status fields listed above.
5. Retry only retryable worker failures; otherwise ask the owner or collaborator to create a fresh link/session/branch as appropriate.
6. If many users are affected after a deploy or restart, verify signing secret stability and Jobs worker availability first.
7. Record the resolution and any missing observability in the Gate 8 handoff.

## Gate 8 Handoff

Backend/Core deliverables already documented here:

- stable setup prerequisites
- signing secret posture
- Jobs behavior and retry rules
- support status fields
- failure-state triage
- audit breadcrumb taxonomy

Frontend/Product deliverables to close in Gate 8:

- release evidence that user-facing states match the frozen contract fixture
- owner/collaborator walkthrough screenshots or test recordings for password-protected, single-use, resumed, revoked, archived, exhausted, and promotion-conflict flows
- final support handoff that names the exact logs, dashboards, or admin surfaces used by operators
- product decision on whether quota/preview status belongs in the initial UI or operator-only docs
