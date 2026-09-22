# AuthNZ module review ledger (2026-09-21)

Module under audit: `tldw_Server_API/app/core/AuthNZ/` — 121 Python files, 74,539 LOC, 39 repo classes
under `repos/`.

Part of the repo-wide core-module duplication + correctness audit. Slug: `authnz`.

## Four rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with
  a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

| Stage | File | Subject |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-architecture-inventory.md` | Module survey, churn, backend-discriminator sprawl, layering, migration ledger asymmetry |
| 2 | `2026-09-21-stage2-repos-data-boundary.md` | `repos/` private-helper duplication, dual-backend SQL divergence, JSON/datetime coercion (cluster C7) |
| 3 | `2026-09-21-stage3-crypto-identity-and-tests.md` | Key derivation, API-key verification, base64 token padding (cluster C1), five-way test-tree split, synthesis |

Stages 4-6 of the standard RAG arc are deliberately collapsed into the three above. AuthNZ has no
distinct "composition" layer worth its own stage, and the API/schema boundary for this module was
already covered by the adjacent ledger (below), so re-staging it would pad rather than inform.

## Sidecar inventories

- `2026-09-21-stage1-source-inventory.txt` — every `.py` file by LOC
- `2026-09-21-stage1-churn-baseline.txt` — 12-month commit counts per file
- `2026-09-21-stage2-repo-helper-inventory.txt` — all backend-discriminator and private row/JSON/datetime helper definitions

## Relationship to the adjacent `auth-dependencies` ledger

`Docs/superpowers/reviews/auth-dependencies/` (2026-04-25, four files) covers the **endpoint-side auth
dependency surface** — `app/api/v1/API_Deps/auth_deps.py` and the 210 endpoint modules that consume it.
It is a different layer from this one. What this ledger **inherits and does not re-report**:

- The dependency-pattern inventory (`get_auth_principal` 553 refs, `rbac_rate_limit` 362, `require_permissions` 198, …)
  and the route-family mix table — `2026-04-25-auth-dependency-inventory.md`.
- The risk scan: 99 modules on legacy user dependencies, 24 with raw user-dict signals, 57 with manual
  admin checks, 120 with ordering-sensitive signals — `2026-04-25-auth-risk-scan.md`.
- The special-route triage (setup-local, webhook/OAuth callback, provider-compatible, public health,
  test-support) and the four admin-check buckets — `2026-04-25-auth-special-route-and-admin-triage.md`.
- The standard helper contract (`CurrentPrincipal`, `AdminPrincipal`, `TokenScopeGuard`, …), the
  request-state invariants, and the ruling that `require_token_scope` must NOT be folded into an
  `AuthPrincipal`-returning dependency — `2026-04-25-helper-contract-spec.md`.

Consequence for this audit: **no finding here concerns endpoint auth dependency spelling, admin-check
placement in routes, or the `AuthPrincipal` alias surface.** Those are that ledger's scope and its
migration order stands. This ledger reviews what sits behind those dependencies: key derivation,
credential verification, session/token storage, and the `repos/` persistence layer.

## Binding ADRs checked before asserting

- ADR-017 scoped org/team RBAC core semantics — no finding here contradicts it.
- ADR-019 security request-edge middleware — out of scope for this module (Security module owns it).
- ADR-027 AES-GCM JSON envelope helpers — **checked and AuthNZ complies**; see stage 3.
- ADR-044 cookie-session governance owner preflight — no finding here contradicts it.

## Constraints on every recommendation in this ledger

- Read-only audit. No source file was modified. No Backlog task was created; tasks are proposed only.
- `tldw_Server_API/app/api/v1/**` is owner-only per CONTRIBUTING.md. Findings are labelled `owner-only`
  where a fix touches that tree.
- Non-trivial refactors named here need the design-first treatment
  (`Docs/Design/YYYY-MM-DD-<slug>-design.md` + ADR + Backlog task + `IMPLEMENTATION_PLAN_<slug>.md`);
  each stage says which findings clear that bar and which do not.
