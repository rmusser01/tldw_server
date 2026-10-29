# ADR-084: Scoped egress for the server-configured Searx endpoint

**Status:** Accepted
**Date:** 2026-10-29
**Backfilled from:** not backfilled
**Decision owner:** UAT requester authorization and S04 implementation review
**Related task:** TASK-13260.278.18.83.45
**Related spec/plan:** Docs/Plans/IMPLEMENTATION_PLAN_uat_s04_research_sources_20261028.md

## Decision

Use the existing exact configured-endpoint scope and checked HTTP transport for the Searx origin read from current server-owned configuration, without relaxing ordinary URL policy.

Only `Search-Engines.search_engine_searx_api` grants authority. The adapter reads a current server configuration snapshot for each operation. A request-provided `searx_url` may use that authority only when its canonical HTTP(S) scheme, hostname and effective port match the configured origin. A different-origin override retains ordinary outbound policy. Blank, invalid or credential-bearing configuration grants no scope.

The shared policy precheck and actual checked request carry the same scope. The central transport retains denylist precedence, forbidden-address rejection, pinned DNS and revalidation of every retry and redirect. Redirects cannot change the scoped origin. Public-capture context overrides provider scope and still requires public credential-free destinations. Search result URLs do not inherit the search provider's authority.

## Context

Search Server uses Searx and supports an operator-configured instance. The existing adapter applied arbitrary-URL policy to that trusted endpoint, rejecting a local deployment. A public fallback can also become unavailable or rate-limited. Opening private networks or nonstandard ports globally would weaken unrelated capture and integrations.

ADR-026 governs ordinary egress and ADR-030 established the exact configured-origin primitive for local LLMs. This decision adopts that primitive for Searx; it does not change either accepted record or grant other search providers implicit local authority.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Disable private-address protection or open ports globally | This widens unrelated outbound operations and public capture. |
| Trust every request-provided Searx URL | A request is not server-owned endpoint configuration. |
| Maintain a separate Searx HTTP client or redirect policy | Existing checked transport already owns DNS, redirects and retries. |
| Depend on one public Searx instance | Public rate limits do not provide reliable operator-owned search availability. |

## Consequences

Operators can configure one local or LAN Searx origin. The existing endpoint and response contracts remain unchanged. Public endpoint overrides continue to work under ordinary policy, and imported result pages remain subject to their own capture checks. Other configured search providers require a separate explicit assessment before receiving scoped authority.

The ADR number is local to this unpublished branch and must be checked against the destination before publication.
