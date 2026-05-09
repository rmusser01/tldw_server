# Prototype Workspaces Contract Matrix

## Purpose

This document is the frontend/backend contract artifact for prototype workspace collaboration productionization. Risk Gate 1 creates the draft. Risk Gate 4 freezes it.

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

## Status

- Draft owner: Backend/Core
- Frontend reviewer: Frontend/Product
- Current gate: Risk Gate 1 draft
- Frozen by: Risk Gate 4

## Error And State Matrix

| State | Backend condition | HTTP status | Stable error category | Frontend state bucket | Retryable | User-facing handling | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| invalid_link | Token cannot be verified without confirming existence | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| expired_link | Token/link is expired | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| revoked_link | Token/link or shared actor is revoked | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| exhausted_link | Link has no remaining collaborator uses | 404 or configured non-enumerating status | invalid_or_unavailable_link | Link unavailable | No | Show generic unavailable link state | Draft |
| archived_workspace | Workspace is archived | 404 or configured non-enumerating status | workspace_unavailable | Workspace unavailable | No | Show unavailable workspace state | Draft |
| bootstrap_failed | Branch session bootstrap failed | 409 or 500-class mapped safe response | bootstrap_failed | Setup failed | Yes, if backend marks retryable | Offer retry when allowed | Draft |
| preview_unavailable | Preview handle missing, revoked, or unhealthy | 409 or 503 | preview_unavailable | Preview unavailable | Yes, if backend marks retryable | Show preview retry/status state | Draft |
| stale_promotion | Candidate is stale versus canonical snapshot | 409 | stale_promotion | Promotion stale | No | Ask user to resubmit from current branch | Draft |
| promotion_conflict | Promotion validation detects conflict | 409 | promotion_conflict | Promotion conflict | No | Show conflict/review state | Draft |
| promotion_validation_failed | Validation failed without promoting | 409 | promotion_validation_failed | Promotion failed | Yes, if backend marks retryable | Show validation failure details | Draft |

## Token And Session Security Dispositions

| Requirement | Disposition | Gate | Notes |
| --- | --- | --- | --- |
| Token storage/hash rules | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| TTLs | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Replay handling | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Cookie flags | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Referrer leakage controls | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Password-protected link behavior | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Signing secret rotation | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |
| Revocation propagation | TBD: enforce now / document existing behavior / defer | Risk Gate 1 | |

## Frontend Fixture Notes

- Fixture schema owner:
- Mock state owner:
- Contract feedback deadline:
- Open frontend questions:

## Gate 4 Freeze Checklist

- [ ] All stable error categories are final.
- [ ] HTTP statuses are final or explicitly documented as non-enumerating policy choices.
- [ ] Retryability is final.
- [ ] Frontend state buckets are final.
- [ ] Backend/Core reviewer recorded.
- [ ] Frontend/Product reviewer recorded.
