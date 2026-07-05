# ADR-026: Security Outbound Egress and SSRF Policy

**Status:** Accepted
**Date:** 2026-06-07
**Backfilled from:** `tldw_Server_API/app/core/Security/README.md`
**Decision owner:** TASK-2247 confirmation and TASK-2311 outbound egress backfill scope
**Related task:** TASK-2311 (`backlog/tasks/task-2311 - Backfill-Security-outbound-egress-policy-ADR.md`)
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md`

## Decision

Outbound integrations that handle untrusted or user-configurable URLs must route those URLs through the Security module egress policy helpers, which enforce scheme, host, port, allow/deny, environment profile, tenant, DNS, and private/reserved-address checks before network calls.

## Context

The Security module owns outbound network policy helpers in `egress.py` and endpoint-friendly URL assertions in `url_validation.py`. The source README already directs Web scraping, Watchlists, WebSearch, Workflows, Text2SQL, and third-party providers to use those helpers before outbound work. Without one shared egress boundary, individual feature modules can drift into inconsistent SSRF handling, local allowlists that skip private-IP checks, or different production defaults.

TASK-2247 confirmed the current implementation that bounds this ADR:

- `evaluate_url_policy()` accepts only HTTP and HTTPS schemes.
- URLs must include a hostname and a valid port. Allowed ports default to 80 and 443 unless `WORKFLOWS_EGRESS_ALLOWED_PORTS` changes that policy.
- Global and workflow allow/deny lists are merged from `EGRESS_ALLOWLIST`, `EGRESS_DENYLIST`, `WORKFLOWS_EGRESS_ALLOWLIST`, and `WORKFLOWS_EGRESS_DENYLIST`.
- Denylist entries win over allowlist entries.
- Production-like environments default to a strict profile that requires an allowlist; non-production defaults to permissive public-host behavior unless allowlists are configured.
- Private and reserved resolved addresses are blocked by default through `WORKFLOWS_EGRESS_BLOCK_PRIVATE`.
- Tenant egress and webhook helpers call the same policy with tenant-specific allow/deny environment variables.
- `assert_url_safe()` wraps the central policy for endpoint code and raises a 400 response when a URL is blocked.

This ADR is intentionally bounded to the shared egress policy and future outbound integration rule. It does not claim every historical network call in the repository already uses these helpers. The protection is effective for paths that route URL decisions through the Security module boundary.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Let each outbound feature define local URL validation | Local validators drift over time and can omit scheme, port, DNS, private-address, or environment-profile checks that the central policy already handles. |
| Treat allowlists as the only SSRF control | Allowlists do not replace DNS and private/reserved-address checks, especially when hostnames can resolve to local or reserved ranges. |
| Use a strict allowlist profile in every environment | Development and local integrations often need public-host testing without central allowlist setup. The accepted policy keeps production strict by default while preserving permissive non-production behavior. |
| Allow outbound callers to bypass egress helpers when they perform their own checks | Bypasses make it hard to audit SSRF behavior and can create incompatible policy semantics across modules. New outbound integrations should use the shared helpers. |
| Combine egress, request-edge middleware, secrets, and serialization into one Security ADR | TASK-2247 found different enforcement boundaries and caveats. ADR-019 already covers request-edge middleware, and secrets/serialization need a separate adoption audit before any accepted ADR. |

## Consequences

New outbound integrations that accept user-configurable URLs should validate through `tldw_Server_API.app.core.Security.egress` or `tldw_Server_API.app.core.Security.url_validation` before constructing clients, sessions, downloads, callbacks, webhooks, search requests, or scraper fetches.

Policy changes for outbound network access belong in `egress.py`, with tests covering allowed and denied URLs, global/workflow/tenant controls, strict versus permissive profiles, port handling, DNS behavior, and private/reserved-address blocking.

Feature modules should not add independent allowlists that bypass the central private-IP, scheme, port, DNS, or denylist checks. If a module needs a narrower rule, it should compose with the central policy rather than replacing it.

Existing outbound call sites that do not yet use the Security helpers are not proven compliant by this ADR. They should be audited or migrated under separate tasks when touched.

Request-edge middleware remains covered by ADR-019. Secret management, crypto helpers, and restricted serialization remain outside this ADR until a focused adoption audit confirms the repository-wide behavior worth recording.

## Follow-up

- Use this ADR as the covering record for the outbound egress/SSRF portion of INV-029.
- Keep `Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md` as the evidence and caveat record for Security split backfills.
- Consider a separate secrets/serialization ADR only after an adoption audit confirms which consumers use `SecretManager`, encrypted JSON helpers, and restricted pickle compatibility paths.
