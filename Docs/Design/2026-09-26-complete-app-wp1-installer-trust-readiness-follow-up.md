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

## Actual backend route correction (Task19)

The exact bd0de933 candidate proved that /internal/ready correctly rejects the
private gateway peer. Keep that endpoint and its loopback Docker healthcheck
unchanged. Use the existing permission-protected /api/v1/health/ready projection
through the gateway with the fresh probe cookie after bootstrap/profile; require
HTTP200 and JSON status ready, then retain exact-session revocation and replay
refusal. Actual cached artifacts prove anonymous401, cookie200/ready and revoked
cookie401. Section5 requires all readiness components before opening the browser
and imposes no ordering that would require weakening the internal route.
The operator payload is consumed privately under the existing time/body limits;
its sanitized details are not logged or turned into a new public endpoint.


## Actual managed setup CSRF correction (Task20)

The exact2840100 candidate passes all13 signed lifecycle checks, but the initial
wizard state mutation fails403. The noAuth flag suppresses explicit key/bearer
authentication for local setup; it must not suppress CSRF protection for unsafe
requests that already use the browser cookie session. Attach the existing
runtime-instance token only within the existing same-origin cookie transport
boundary. Continue stripping stale explicit authentication and CSRF headers,
and never send page cookies/tokens to an external or mismatched origin. Safe
methods and missing instance-token handling remain unchanged. Preserve backend
CSRF/hop/remote policy and strict qualification refusal latches. Actual unchanged
cached proof of missing-token403 and same-body expected-token200 establishes the
cause; scoped tests/review and fresh exact artifact qualification establish the
fix. This does not claim full provider setup or authorize publication.


## Actual forbidden-upgrade response correction (Task22)

Both exact13d9 native architectures pass signed lifecycle and normal wizard,
then fail the hostile-Origin cookie WebSocket refusal because gateway socket
reset supplies no HTTP status. Keep exact authorization and strict probe unchanged.
Send a small constant403 handshake refusal without cookie, reflected input or
proxy dispatch for unauthorized upgrades, then end the socket with an absolute
1000ms destruction backstop cleared on close/unreferenced. Other invalid-path,
phase and managed-route refusals retain existing behavior. Real-gateway negative
and successful-upgrade regressions plus exact local/native artifacts are required.
This does not imply an accepted hostile connection or authorize publication.
