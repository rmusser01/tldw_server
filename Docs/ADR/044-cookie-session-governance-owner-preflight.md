# ADR-044: Cookie-session governance owner preflight

**Status:** Accepted  
**Date:** 2026-09-05  
**Task:** TASK-13192  
**Extends:** ADR-018 and ADR-019

## Decision

For governed HTTP requests whose policy includes `user` or `api_key` and includes none of `global`, `ip`, or `entity`, and carrying the configured single-user session cookie and no explicit Authorization or X-API-KEY header, Resource Governor ingress resolves the canonical AuthNZ principal before deriving its entity. The resolver validates the session and caches its owner and AuthContext in shared request state. Endpoint authentication reuses that context. Authentication errors retain their status, detail and headers.

Explicit headers, including empty headers, preserve canonical precedence. Requests without the session cookie, ungoverned requests, and policies supporting anonymous admission retain existing behavior. Missing or empty policy scopes retain the governor’s global/entity default; missing request limits still deny. This preserves stale-cookie health checks and idempotent logout, whose policies admit anonymous requests. Policy limits and scopes are unchanged; multiple sessions for one owner consume the same user bucket. No cookie token becomes a quota identifier or authenticated claim.

## Context

Quickstart Persona UAT on 2026-09-05 repeatedly returned 429 on the first profile request with retry_after 1. Ingress ran before endpoint authentication, derived an IP for cookie-only requests, and found no matching scope in character_chat.default's user/api_key policy. Waiting could not restore headroom because no bucket existed for that scope.

## Alternatives

- Hashing each cookie into an API-key bucket changes quota ownership and allows session rotation to reset the owner's quota.
- Adding IP scope or relaxing the policy masks the missing authenticated identity and changes existing limits.
- Reimplementing session validation in governance duplicates AuthNZ ownership and risks drifting revocation semantics.
- Authenticating every ingress request changes unrelated header and anonymous behavior unnecessarily.

## Consequences

Governed cookie requests perform canonical authentication before quota admission. Validation results are reused within the request, avoiding duplicate session lookups. This requires AuthNZ availability for those requests and does not fail open on resolver errors. Existing CSRF middleware remains outside governance in main.py registration order; the preflight does not remove endpoint authorization, CSRF, Origin checks, or other request-edge controls.
