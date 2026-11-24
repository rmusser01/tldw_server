# ADR-097: Reviewed outbound destinations and privacy modes

**Status:** Proposed
**Date:** 2026-11-24
**Backfilled from:** not backfilled
**Decision owner:** Project maintainer and design review
**Related task:** TASK-15633
**Related spec/plan:** [Outbound destination inventory and privacy regression checks](../Design/2026-11-24-outbound-destination-inventory-design.md)

## Decision

Require reviewed declarations for built-in outbound destinations and downloaded resources, and enforce the same outbound regression checks locally and in required CI for covered setup and runtime scenarios. Offer deployment-wide strict privacy modes once the selected mode's destination rules govern or contain each enabled outbound boundary in the advertised profiles.

## Context

App-controlled requests can originate in the API, workers, installers, native libraries, the WebUI server or browser, and the packaged extension. Existing checked HTTP and SSRF policy governs the call paths that use it. Startup success alone gives limited evidence because some network failures can be caught by the app, and a host can serve multiple unrelated resources. A checked inventory and observed traffic reports answer different questions.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Static origin scan alone | It misses computed destinations, same-origin asset changes, redirects, and traffic initiated by dependencies. |
| Shared HTTP wrapper as the whole boundary | Browser requests, custom transports, and child or native code have additional network boundaries. |
| Mandatory proxy or host firewall for every installation | It adds deployment requirements and still needs explicit coverage and attribution for the app's declared capabilities. Process-level isolation remains available for verification and supported strict profiles. |

## Consequences

- Use the existing required CI contexts and a locally runnable entry point. Keep this compatible with the proposed CI tiers in [ADR-096](096-tiered-development-and-release-ci.md).
- Keep product declarations separate from instance-local approvals. Strict admission composes with [ADR-026](026-security-outbound-egress-and-ssrf-policy.md) and preserves the scoped local endpoint rules in [ADR-030](030-configured-local-llm-egress-policy.md) and [ADR-090](090-configured-searx-egress-policy.md).
- Clearly report which scenarios and transports produced evidence. Require coverage or fail-closed handling for every enabled boundary before offering deployment-wide strict modes. Use audit as the compatibility default. Loopback-only offline mode excludes sibling network services.
- Keep observed contact history local, bounded, access-controlled, and separate from external telemetry. Track policy decisions separately from observed contacts.
