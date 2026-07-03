# ADR-018: Resource Governance Endpoint Policy and Route Map

**Status:** Accepted
**Date:** 2026-06-04
**Backfilled from:** `tldw_Server_API/app/core/Resource_Governance/README.md`
**Decision owner:** TASK-2233 confirmation and TASK-2234 backfill scope
**Related task:** TASK-2234
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-resource-governance-confirmation-audit.md`

## Decision

New API endpoints must use claim-first authorization, explicitly decide Resource Governor applicability for latency/cost-sensitive or user-facing ingress, and own policy-store plus `route_map` coverage when governed; DB policy-store mode merges file `route_map` entries into DB policy snapshots with file precedence, and request ingress denies route-map entries that resolve to missing request policies.

## Context

The project has multiple endpoint families that can be expensive, latency-sensitive, or operationally sensitive. Contributors need a durable rule for how endpoint authorization and Resource Governance coverage are chosen so new routes do not rely on ad hoc mode checks, undocumented rate-limit heuristics, or policy IDs that only work in one policy-store mode.

The confirmed current behavior is:

- New endpoints use `get_auth_principal` plus `RequirePermission(...)`, `RequireRole(...)`, or `require_service_principal()` rather than granting/bypassing access through `AUTH_MODE` or single-user/multi-user mode checks.
- Latency/cost-sensitive or user-facing endpoints should decide whether Resource Governor policy applies.
- Governed ingress routes need a policy-store entry and `route_map` coverage keyed by path or tag.
- Middleware route-map resolution is path first, then tag.
- Resource Governor middleware enforces the request category at ingress; token, stream, job, and minute-budget categories require endpoint-level reserve/commit plumbing.
- In DB policy-store mode, DB policies are the policy source of truth, while the file `route_map` is merged into the snapshot and takes precedence over DB route-map entries on conflict.
- When request ingress resolves to a missing request policy, missing request limits default to deny.

This ADR is deliberately bounded by the TASK-2233 audit. It does not claim every existing endpoint is already Resource Governor-covered or claim global fail-closed behavior for Redis outages, tokens, streams, jobs, or other non-request categories.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Allow new endpoints to branch on `AUTH_MODE` or mode helpers for authorization | Reintroduces mode-specific authorization shortcuts and conflicts with the claim-first AuthNZ guardrails. |
| Leave Resource Governor applicability as an implicit per-endpoint implementation detail | Makes expensive or user-facing routes easy to add without policy ownership, route-map coverage, or tests. |
| Require DB policy-store mode to own both policies and route maps exclusively | Loses the repo-owned file route map that documents primary ingress coverage and makes path/tag mapping harder to review with code changes. |
| Let missing route-map policies fail open for request ingress | Allows a typo or missing DB seed to silently remove request protection from a governed route. |
| Treat middleware as full Resource Governor enforcement for all categories | Current middleware only reserves request units. Other categories need endpoint-level plumbing and should not be accepted as middleware behavior. |
| Claim all existing endpoints satisfy the new-endpoint governance rule | The confirmation audit verified the rule, representative coverage, and guardrails, not universal coverage for all historical routes. |

## Consequences

New endpoints should first establish claim-first auth dependencies. Admin/control surfaces should use claim-derived roles or permissions, not new mode-specific bypasses.

Endpoint owners must make a Resource Governor applicability decision for latency/cost-sensitive or user-facing ingress. If governed, the route needs an explicit policy and a `route_map` entry. Path mappings take precedence over tag mappings, so path entries should be used for primary ingress routes where precise coverage matters.

DB-backed Resource Governor deployments must seed every request policy referenced by the merged route map. YAML `policies:` are not runtime policies in DB mode, but YAML `route_map` entries still merge into the snapshot and override DB route-map conflicts.

Missing request policies deny request ingress by default. This is a safety property for request-rate enforcement, not a blanket claim about all Resource Governor categories. Token limits with missing config can be unbounded unless another durable cap denies later, concurrency categories are not middleware-enforced, and Redis backend outage behavior remains configurable through fail-mode settings.

Future broadening of Resource Governor coverage, Redis fail-mode posture, or endpoint-level token/stream/job plumbing should be tracked as implementation follow-up and, if it becomes a durable architecture rule, recorded in a new or superseding ADR.

## Follow-up

- Use this ADR as the covering record for INV-028.
- Consider adding a focused regression test for a route-map entry that resolves to a missing DB request policy and returns the expected request-ingress denial.
- Keep `Docs/ADR/inventory/2026-06-04-resource-governance-confirmation-audit.md` as the evidence record and scope boundary for this backfill.
