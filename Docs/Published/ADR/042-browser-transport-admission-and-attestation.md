# ADR-042: Browser Transport Admission and Attestation

**Status:** Accepted
**Date:** 2026-08-31
**Backfilled from:** `Docs/superpowers/plans/2026-08-27-browser-transport-safety-gate.md`
**Decision owner:** TASK-13139.2 implementation and PR #2839 review
**Related task:** TASK-13139.2
**Related spec/plan:** `Docs/superpowers/plans/2026-08-27-browser-transport-safety-gate.md`

## Decision

Browser retrieval is admitted only when single-user `compat` mode uses a request-routed, URL-guarded Playwright adapter, or when a configured governed transport supplies complete in-process evidence that it routes every request, pins DNS, and verifies the connected peer; all other profiles and invalid configuration fail closed before browser resources are reserved or started.

## Context

ADR-026 requires untrusted outbound URLs to pass the central egress policy. That policy validates names and resolved addresses before dispatch, but it cannot prove which address Chromium later connects to. Playwright request interception remains necessary for navigations, redirects, frames, subresources, HTTP requests, and WebSockets, yet interception alone does not pin Chromium's DNS result or attest the connected peer.

The browser boundary therefore needs a separate admission contract in addition to ADR-026:

- `disabled` always denies browser retrieval.
- `auto` and `url_guarded` allow only the exact `single_user` plus `compat` profile, and capability metadata reports `dns_peer_attested=false`.
- Strict or multi-user profiles require `attested_proxy` plus complete in-process evidence that the governed mechanism routes every request, pins DNS, and verifies the peer.
- Selecting `attested_proxy` is configuration, not evidence; missing or incomplete attestation remains denied.
- Malformed configuration, configuration-load errors, and decision-provider failures produce bounded config-invalid denial before browser capacity, launch, context creation, or navigation.
- Every admitted URL-guarded browser path uses the shared guarded adapter, blocks service workers, and applies fresh central egress decisions to browser HTTP and WebSocket dispatches.

TASK-13139.2 and its review remediation add the admission contract and denial metadata only. They do not provide a production attestor, governed proxy, resolver, persistent cookie store, credentialed browser, or authenticated browser session. Authenticated browser work remains dependent on TASK-13100.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Append browser admission to ADR-026 | Browser peer attestation is a separate durable decision from the central URL egress policy, and the ADR workflow requires one decision per ADR. |
| Permit URL interception in strict or multi-user profiles | URL checks cannot bind Chromium to the validated DNS answer or verify the connected peer, leaving DNS-rebinding risk unresolved. |
| Treat `attested_proxy` configuration as proof | A string setting cannot establish runtime routing, DNS pinning, or peer verification. |
| Disable browser retrieval in every deployment | This removes useful local single-user compatibility even when every browser request is routed through the existing URL guard; the accepted tradeoff keeps that bounded mode explicit and unattested. |

## Consequences

Production browser call sites must consult the shared transport decision and use a guarded adapter before Playwright work. A raw caller-supplied Playwright context is not an admissible shortcut; the legacy article helper delegates to `GuardedArticleBrowser` so redirects, subresources, WebSockets, and service-worker behavior remain within the governed boundary.

Credentialless HTTP extraction remains available when browser escalation is denied. Public denials expose only the fixed transport capability fields and safe reason codes. Resolver and provider errors may be logged only with bounded operation and exception-type metadata, never raw values or exception messages.

ADR-026 remains the governing central outbound egress decision and is not superseded. This ADR composes with it by defining when a browser transport may act after URL-policy checks.

## Follow-up

- Keep direct article, preflight, enhanced, and legacy browser adapters covered by admission-before-launch and route-interception tests.
- Require a separate reviewed implementation and attestation contract before enabling governed proxy transport in production.
- Keep credentialed or persistent browser sessions out of scope until TASK-13100 defines their authority and isolation model.
