# WP1 installer trust and readiness follow-up

This clarifies implementation of the already approved complete-app design
sections5/10/11 for TASK-13343. Whole-WP1 review at19f2174383 found four
load-bearing gaps and a cleanup-reporting defect. It does not expand the
release feature set or authorize publication.

## Authoritative persistent inputs

Every managed Compose interpolation value must come from verified persistent
configuration. Remove conflicting inherited interpolation values from child
Compose calls, or reject them before any Compose action, on Unix and PowerShell.
Preserve Docker connection settings and explicit first-install input knobs.
Never evaluate config.env as shell code or print secrets. Stop/status use the
same rule. Tests inject conflicting images, public port, keys, hop and cookie
names and verify the actual child environment/Compose resolution.

## Complete paired-bundle inventory

At the paired-bundle verification boundary, require the signed compose.yaml,
README.md and all six platform host helper files with the correct selected
platform records, before init or promotion. Verify their bytes as today. Generic
manifest parsing stays usable by later native packages. Both control and
promotion consumers reject omitted entries, missing files, wrong-platform-only
records and tampering. No source allowlist or frontend-publication gate weakens.

## Authenticated installer readiness

Use the existing pinned control image and Python standard library, joining
only the verified instance's private Compose network for this bounded probe.
Other control verification/initialization stays network-isolated. Check running
role identities, signed image references/release labels and private/published
ports using Docker inspection passed privately to control; do not mount Docker
sockets into application or control containers. Require the actual gateway
listener with the persisted public Host/Origin envelope for backend readiness,
WebUI assets, a fresh session bootstrap and cookie-only profile access. Never
call private backend/Next directly for the authenticated exchange, nor add
master-key injection to ordinary routes.

Temporary readiness cookies stay in the disposable process. Revoke exactly its
session in cleanup and verify refusal after revocation before reporting/opening
the URL. Bound requests, overall duration and body size. If probing or session
cleanup fails, readiness fails and the helper attempts only its owned stack
cleanup. A failed cleanup is reported separately with retained state/recovery
location; never claim resources stopped without checking the result. Raw
inspection JSON, credentials, cookies, bodies and request errors are private.
Host binding also remains covered by actual outside-checkout lifecycle/browser
qualification; this probe introduces no host Python/Node/curl requirement.

## First-install port selection

Before committing a new installation's origin, confirm the requested/default
port is bindable using Docker-owned disposable preflight resources. An occupied
default offers an available alternative; an occupied explicit choice fails
before config creation so an alternate explicit retry works. Always remove only
owned preflight resources and fail closed on cleanup failure. Existing working
state retains its exact origin/credentials/data; no silent rotation, deletion or
adoption/termination of an unrelated process.

## Qualification boundary

Unit/fake-child tests prove contracts but do not replace fresh signed candidate
qualification. Run focused Python/helper/real-socket tests, scoped formatting,
lint and Bandit, then one scoped final rereview. Rebuild exact clean final local
and both native Linux container architectures, verify signed bytes independently,
and require full lifecycle/two-instance browser checklists and owned cleanup.
Windows helper syntax is separate from Windows Docker runtime; mark runtime
unqualified until tested on a supported Windows host. G12 stays false and the
frontend freeze remains in force. No PR, merge or public artifact is authorized.
