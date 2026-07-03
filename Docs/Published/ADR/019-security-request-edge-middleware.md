# ADR-019: Security Request-Edge Middleware

**Status:** Accepted
**Date:** 2026-06-04
**Backfilled from:** `tldw_Server_API/app/core/Security/README.md`
**Decision owner:** TASK-2247 confirmation and TASK-2248 backfill scope
**Related task:** TASK-2248
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md`

## Decision

Request-edge Security middleware is owned by the Security module: normal startup installs setup access guard, setup CSP, and security headers; request ID and drain-gate middleware are always installed; CSP behavior is path-scoped; and production keeps security headers enabled by default unless explicitly disabled.

## Context

The project has several request-edge controls that need consistent startup wiring and path-sensitive behavior: request correlation, shutdown draining, setup UI access, setup-specific CSP, documentation/API CSP differences, and HTTP hardening headers. Keeping these concerns in one Security module boundary gives endpoint and feature owners a clear place to add tests and prevents each router from inventing its own request-edge policy.

The TASK-2247 confirmation audit verified the current behavior that bounds this ADR:

- `app/main.py` imports and wires Security middlewares during startup.
- Test startup still installs setup CSP and setup access guard, while explicitly skipping nonessential security headers in test mode.
- Normal startup installs setup CSP and setup access guard, computes `ENABLE_SECURITY_HEADERS`, defaults production security headers on when the variable is absent, and installs `SecurityHeadersMiddleware` when enabled.
- `DrainGateMiddleware` and `RequestIDMiddleware` are always installed.
- `SecurityHeadersMiddleware` sets response hardening headers, removes `Server`, applies path-scoped CSP behavior for setup/docs/API surfaces, and only enables HSTS when `SECURITY_ENABLE_HSTS` is true and the request is HTTPS or `X-Forwarded-Proto: https`.
- `RequestIDMiddleware` sanitizes or generates request and session identifiers, stores them on request state, propagates response headers, and mirrors request IDs into tracing baggage.
- `SetupAccessGuardMiddleware` gates `/setup`, permits loopback access, and applies denylist, allowlist, and explicit remote setup settings.
- `SetupCSPMiddleware` applies only to `/setup`, permits inline scripts, permits `eval` by default, and removes `eval` when `TLDW_SETUP_NO_EVAL` is set.

This ADR deliberately covers request-edge middleware only. It does not accept a project-wide outbound egress/SSRF decision, a universal secret-management adoption claim, or a serialization policy decision.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Scatter request-edge behavior across feature routers | Makes setup access, CSP, request IDs, drain handling, and HTTP hardening dependent on local router choices instead of one startup-owned Security boundary. |
| Treat setup, docs, and API CSP as one global policy | Current behavior is intentionally path-sensitive, and setup has relaxed script needs that should not leak into normal API or docs responses. |
| Force HSTS for every response independent of deployment protocol | HSTS can break deployments behind proxies or local HTTP setups when not coordinated with HTTPS ingress, so the middleware keeps HSTS opt-in and HTTPS-aware. |
| Install request IDs and drain gate only in production | Test and development requests still need correlation IDs and deterministic drain behavior, so these middlewares should stay always installed. |
| Backfill one broad Security ADR for request edge, egress, secrets, and serialization | TASK-2247 found different caveats for those areas. Combining them would overclaim adoption and make future review harder. |

## Consequences

New request-edge middleware behavior belongs in `tldw_Server_API/app/core/Security/` and startup wiring in `app/main.py`, with path-specific tests under `tldw_Server_API/tests/Security/`.

Production deployments keep Security headers enabled by default unless `ENABLE_SECURITY_HEADERS` explicitly disables them. Test mode and explicit disablement remain recognized caveats and must be called out when verifying behavior.

HSTS remains opt-in and HTTPS-aware. Setup CSP remains intentionally more relaxed than normal API/docs CSP, including the default `eval` allowance unless `TLDW_SETUP_NO_EVAL` disables it.

Outbound egress/SSRF policy, secret-management adoption, and serialization safety remain separate Security decisions. They should be confirmed and backfilled in separate bounded ADRs if they become priority work.

## Follow-up

- Use this ADR as the covering record for the request-edge middleware portion of INV-029.
- Keep `Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md` as the evidence record and scope boundary for this backfill.
- Consider separate ADRs for outbound egress/SSRF policy and for secret/serialization adoption after focused confirmation work.
