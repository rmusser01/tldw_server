# Prototype Workspaces Threat Model

Status: Risk Gate 1 draft

Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

Risk Gate 1 issue: https://github.com/rmusser01/tldw_server/issues/1453

## Scope

This document defines the first productionization security contract for prototype workspace collaboration. It covers private-link exchange, external shared actors, collaborator session tokens, resume cookies, preview grants, promotion submission, and promotion review.

Risk Gate 1 is not the final OpenAPI freeze. Risk Gate 4 owns the contract freeze after Backend/Core and Frontend/Product validate these states against implementation and UI needs.

## Actor Identities

| Actor | Identity source | Allowed actions | Must not be allowed |
| --- | --- | --- | --- |
| Owner | AuthNZ user id matching `prototype_workspaces.owner_user_id` | Create workspace, open owner branch session, create prototype share token, view owner detail, renew owner previews, approve/reject promotions | Act on another owner's prototype workspace |
| Internal collaborator | AuthNZ user id stored on a prototype session | Work in a branch session when future internal collaboration is enabled | Use external shared-actor tokens or bypass owner/promoter review |
| Designated promoter | AuthNZ user id listed in `designated_promoter_ids` | Review promotion requests for that workspace | Create owner sessions or mint share links unless also owner |
| External shared actor | `prototype_shared_actors.id` created from a prototype share token | Exchange private link, resume same browser-bound identity when allowed, create collaborator branch session, save snapshots, submit promotion requests for its own branch | Become an AuthNZ user, use another share link, use another workspace, review/promote directly, renew owner-only previews |

## Trust Boundaries

- Authenticated owner APIs trust AuthNZ identity and must re-check workspace ownership server-side.
- Public share endpoints receive bearer-like URL tokens and must avoid revealing whether a token, workspace, or exhausted link exists except where password verification already implies a valid protected link.
- Collaborator session tokens are signed capability tokens scoped to one `Prototype Shared Actor`, one prototype workspace, one share-link id, and one runtime policy profile.
- Resume cookies are browser-binding hints, not standalone authorization. They must be signed, bound to a stored shared-actor binding secret, and rotated after successful resume.
- Preview grants are short-lived signed grants behind opaque preview handles. Raw runtime URLs are not a client contract.
- Promotion review is authenticated owner/designated-promoter authority. Promotion submission is external actor authority and must be bound to the actor's own active session and candidate snapshot.

## Authorization Invariants

### Token Creation

- Prototype share tokens may only be created by the workspace owner.
- The token's `resource_type` must be `prototype_workspace`.
- The token's `resource_id` must point to a prototype workspace whose `owner_user_id` matches the token owner.
- Token hashes and password hashes must never be returned in list or preview responses.

### Public Link Exchange

- Token validation must check token hash, revocation, expiry, and resource type.
- Password-protected links require password verification for first-time exchange.
- Exhausted links may allow same-browser resume only when a valid resume cookie maps to the same active shared actor and same share link.
- A first-time exchange must claim a token use before creating a new shared actor and release the claim when provisioning fails before the actor/session is usable.
- The created shared actor must be scoped to the prototype workspace and share-link id.
- Archived or missing workspaces must not create a shared actor.

### Collaborator Session Creation

- A collaborator branch session requires a valid signed collaborator session token.
- The token must decode to `actor_type = external_collaborator`.
- The shared actor referenced by the token must exist, be unrevoked, unexpired, and belong to the token workspace and share-link id.
- The branch session base snapshot must belong to the same prototype workspace.
- Reuse must only return an active session for the same actor key, base snapshot, workspace, and share-link scope.

### Snapshot Save

- Revoked or expired sessions cannot save snapshots.
- Revoked or expired shared actors cannot save snapshots through an otherwise active session.
- A shared actor may only author snapshots for its own branch session.

### Preview Grants

- Preview grants must be issued through the preview broker, not by exposing raw runtime URLs.
- Preview grant issuance and renewal must reject revoked or expired sessions and shared actors.
- Revoked preview handles must not renew or validate.
- Owner preview renewal must require the authenticated owner of the preview handle's workspace.

### Promotion Submission

- A promotion request requires a valid signed collaborator session token.
- The token workspace must match the request workspace.
- The requested prototype session must belong to that workspace.
- The token shared actor must match the session shared actor.
- The candidate snapshot must belong to the requested workspace, the requested branch session, and the token shared actor.
- The shared actor and session must still be active at submission time. A revoked or expired shared actor cannot use an older still-signed session token.

### Promotion Review

- Only the owner or a designated promoter may review promotion requests.
- Promotion approval must validate the candidate against the current canonical baseline.
- Stale candidates must not advance canonical pointers.
- Failed publish validation must not update canonical or last-known-good snapshots.
- Rejection records reviewer id and notes but never changes canonical state.

## Non-Enumerating Error Policy

| Public state | Required behavior | Current disposition |
| --- | --- | --- |
| Invalid token | Return generic unavailable response, target 404 | Enforce now |
| Expired token | Return generic unavailable response, target 404 | Enforce now |
| Revoked token | Return generic unavailable response, target 404 | Enforce now |
| Exhausted token with no valid same-browser resume | Return generic unavailable response, target 404 | Enforce now |
| Missing prototype workspace | Return generic unavailable response, target 404 | Enforce now |
| Token owner/workspace owner mismatch | Return generic unavailable response, target 404 | Enforce now |
| Archived prototype workspace | Return unavailable workspace response without creating actor | Document existing 403; Risk Gate 4 should decide whether to make public archived links 404 |
| Bad or missing password for protected link | Return 403 password failure without revealing internals | Document existing behavior |
| Revoked or expired session token paths | Return 403 inactive session | Enforce now |

Authenticated owner APIs may return direct owner-facing 403/404 states because the caller is already authenticated. Public-link states should prefer non-enumeration.

## Token And Session Requirements

| Requirement | Disposition | Gate | Notes |
| --- | --- | --- | --- |
| Token storage/hash rules | Document existing behavior | Risk Gate 1 | Share tokens are looked up by prefix and verified by hash. List responses strip `token_hash` and `password_hash`. |
| TTLs | Enforce now | Risk Gate 1 | Share token `expires_at`, shared actor `expires_at`, prototype session `expires_at`, and collaborator session token `exp` are enforced in current paths. |
| Replay handling | Document existing behavior; defer stronger replay controls | Risk Gates 1 and 4 | Token use claims protect first-time max-use links. Resume cookies rotate binding secrets. Session tokens remain valid until expiry and are rechecked against active actor state. Risk Gate 4 should decide whether session-token nonce tracking is required. |
| Cookie flags | Document existing behavior | Risk Gate 1 | Resume cookie is `HttpOnly`, `SameSite=Lax`, seven-day `max_age`, and `Secure` when request scheme or forwarded proto is HTTPS. |
| Referrer leakage controls | Defer to Risk Gate 5 | Risk Gate 5 | Frontend currently uses `/share/:token` and `/prototype-workspaces?share_token=...`. Risk Gate 5 should remove token-bearing URLs from persistent route state after exchange. |
| Password-protected link behavior | Enforce now | Risk Gate 1 | First exchange requires password. Same-browser resume may skip password when the signed resume cookie maps to the same active actor and link. |
| Signing secret rotation | Defer to Risk Gate 7 | Risk Gate 7 | Stable signing secrets are required. Rotation policy and startup/operator checks belong to operational documentation. |
| Revocation propagation | Enforce now for auth-sensitive paths; defer cleanup | Risk Gates 1, 2, 3, and 7 | Branch sessions, snapshot save, preview grants, and promotion submission must reject revoked actors. Cleanup of stale rows and operator visibility is later-gate work. |

## Audit Event Taxonomy

| Event | Producer | Required fields |
| --- | --- | --- |
| `token.created` | Share-token creation | resource type/id, owner id, token id, access level, expiry, max uses |
| `token.revoked` | Share-token revocation | owner id, token id, resource type/id |
| `token.password_verified` | Public password verification | token id, resource type/id, owner id, IP, user agent |
| `token.password_failed` | Public password failure | token id when known, resource type/id when known, IP, user agent |
| `token.prototype_session_exchanged` | Prototype public exchange | token id, workspace id, owner id, shared actor id, resume flag, claimed-use flag |
| `prototype.shared_actor.created` | Access service or public exchange | workspace id, token id, shared actor id, runtime profile |
| `prototype.shared_actor.resumed` | Access service or public exchange | workspace id, token id, shared actor id, rotated binding indicator |
| `prototype.preview_grant.issued` | Preview broker | workspace id, preview handle, scope, session id when present, actor key |
| `prototype.preview_grant.renewed` | Preview broker/API | workspace id, preview handle, requester identity |
| `prototype.preview_grant.revoked` | Preview broker | workspace id, preview handle, reason |
| `prototype.promotion.requested` | Promotion submission | workspace id, session id, candidate snapshot id, shared actor id |
| `prototype.promotion.reviewed` | Promotion review | workspace id, promotion id, reviewer id, decision |
| `prototype.authorization_denied` | API/service guards | state bucket, route/action, workspace id when safe, actor type when safe |

Risk Gate 1 does not require every event to exist in code. Missing events are explicit carry-forward work for Risk Gate 7 unless they block an invariant test.

## Rate And Quota Policy

| Operation | Risk Gate 1 policy |
| --- | --- |
| Public link preview/exchange/password verification | Use public IP rate limit; keep error states generic. |
| First-time collaborator session creation | Enforce token `max_uses` with atomic claim/release behavior. |
| Same-browser resume | Does not consume another token use, but must rotate the resume binding secret. |
| Collaborator branch bootstrap | Defer job-level quota to Risk Gate 3. |
| Preview grant renewal | Defer per-handle/user quotas to Risk Gate 3 or 7; keep auth checks enforced now. |
| Promotion request creation | Defer per-actor request quotas to Risk Gate 7; enforce ownership/session/snapshot invariants now. |

## Frontend/Product Preparation

Risk Gate 1 prepares Frontend/Product to build without inventing backend semantics:

- Use `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md` as the only draft source for error categories, frontend state buckets, and retryability.
- Create fixtures for: unavailable link, password required, exhausted link, archived workspace, inactive session token, preview unavailable, stale promotion, promotion conflict, validation failed, bootstrap failed.
- Route-state audit must verify token-only collaborator entry never falls back to stale owner workspace state.
- Route-state audit must verify password handoff does not persist beyond exchange and does not get logged or stored in workspace state.
- Feedback for missing fields, ambiguous categories, or user-facing buckets should be recorded for Risk Gate 4 before contract freeze.

## Deferred Items

- Final public archived-link status choice: Risk Gate 4.
- Session-token nonce replay store: Risk Gate 4 if required by contract freeze, otherwise Risk Gate 7.
- Referrer/token URL cleanup in frontend route state: Risk Gate 5.
- Runtime job quota/retry/cancellation details: Risk Gate 3.
- Audit dashboards and operator visibility: Risk Gate 7.
- Cleanup/retention for archived workspaces, expired sessions, stale promotions, and revoked actors: Risk Gate 2 and Risk Gate 7.

## Verification Evidence

Risk Gate 1 verification should include:

- `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_service_authorization.py -q`
- `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py -q`
- `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q`
- Bandit on touched backend endpoint code when backend code changes occur.
