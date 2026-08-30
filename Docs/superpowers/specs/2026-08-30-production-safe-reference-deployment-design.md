# Production-Safe Reference Deployment and Health Surface Design

**Status:** Approved
**Date:** 2026-08-30
**Backlog:** TASK-13013.6
**Parent:** TASK-13013

## Purpose

Ship one production reference deployment that fails closed before startup,
keeps stateful services off the host network, terminates TLS at a reverse
proxy, and exposes only a minimal public liveness response. Detailed readiness,
provider, database, and metrics data remain available only to trusted
operators or private-network monitoring services.

This design replaces unsafe production examples with an enforceable reference
contract. It does not turn local-development Compose files into production
profiles and does not depend on a new authentication system.

## Current Problems

The current deployment surface has several unsafe or broken defaults:

- `docker-compose.multi-user-postgres.yml` defaults `tldw_production` to
  `false` and publishes the application on the host loopback interface.
- The Caddy and Nginx overlays proxy every application path, including detailed
  readiness and metrics endpoints.
- The proxy overlays mount `Dockerfiles/Samples/...`, but the checked-in sample
  files live under `Helper_Scripts/Samples/...`.
- Public `/health` returns resource-governor policy metadata rather than a
  minimal liveness result.
- Detailed API-v1 health and metrics routes are mostly unauthenticated.
- Production documentation recommends TLS, backups, and rollback preparation,
  but no preflight enforces those requirements.

## Goals

1. Provide a dedicated, self-contained production Compose profile.
2. Publish only the TLS reverse proxy; keep the app, PostgreSQL, and Redis on a
   private Compose network.
3. Require explicit, non-placeholder production secrets and configuration.
4. Make public `/health` a dependency-free minimal liveness contract.
5. Keep machine readiness and Prometheus scraping private-network-only.
6. Require admin authentication for detailed API-v1 operational diagnostics.
7. Validate topology, TLS, proxy trust, secrets, backup preparation, and
   rollback preparation with a fail-closed preflight.
8. Document a directly executable backup, restore, upgrade, and rollback flow.

## Non-Goals

- Replacing Docker Compose with Kubernetes or another orchestrator.
- Publishing, signing, or generating SBOM/provenance for container images;
  those remain in TASK-13013.7.
- Adding a new service-token or monitoring-token system.
- Exposing remote unauthenticated monitoring.
- Rewriting every existing local or CI Compose profile.
- Proving that an operator has performed a restore test; preflight validates
  the required backup destination and rollback inputs, while the runbook owns
  the restore exercise.

## Chosen Architecture

The implementation adds a dedicated production profile rather than layering
security-sensitive overrides onto the existing local profile. This avoids
Compose merge ambiguity around published ports and avoids breaking local
development workflows.

```text
Internet
   |
   | 80/443 only
   v
TLS reverse proxy
   |
   | private Compose network
   v
tldw app :8000 ---- PostgreSQL :5432
   |
   +------------- Redis :6379
   |
   +------------- Prometheus/private health consumers
```

Only the reverse proxy has host `ports`. The application uses `expose` for
private service discovery. PostgreSQL and Redis have neither `ports` nor public
proxy routes.

The production profile sets production mode and multi-user authentication
explicitly. It never relies on a development fallback. Required values use
Compose's fail-fast `${NAME:?message}` form where practical, and the preflight
performs the stronger semantic checks Compose cannot express.

## Operational Route Contract

| Route | Intended consumer | Public proxy | Application authorization | Response contract |
| --- | --- | --- | --- | --- |
| `GET/HEAD /health` | External load balancer and uptime check | Allowed | None | Minimal status only |
| `GET/HEAD /ready` | Compose/orchestrator on private network | Blocked | None | Detailed readiness |
| `GET/HEAD /health/ready` | Private readiness compatibility client | Blocked | None | Detailed readiness |
| `GET /metrics` | Prometheus on private network | Blocked | None | Prometheus text |
| `/api/v1/health*` | Human operator/API client | Allowed | Admin required | Detailed diagnostics |
| `/api/v1/metrics*` | Human operator/API client | Allowed | Admin required | Detailed metrics/control |

The public `/health` body is exactly a stable minimal status object such as
`{"status":"ok"}`. It does not include timestamps, database type, pool sizes,
provider status, policy metadata, build identifiers, environment names, paths,
or secrets. It does not probe dependencies, so an unhealthy dependency cannot
turn liveness into a restart loop.

The reverse proxy rejects the private root paths before the catch-all proxy
rule. A not-found response is preferred over an authorization challenge so the
public edge does not advertise the private monitoring surface.

Detailed API-v1 endpoints reuse the existing claim-first `RequireRole("admin")`
dependency. Private machine endpoints rely on network isolation, not a new
credential lifecycle. This matches the approved private-network-only
monitoring decision.

## Production Configuration Contract

The reference profile requires explicit configuration for:

- public domain and ACME/TLS contact;
- HTTPS-only allowed origins;
- JWT signing material suitable for multi-user mode;
- PostgreSQL credentials without known placeholder/default values;
- authenticated Redis configuration without a default credential;
- initial administrator bootstrap or an explicit existing-installation mode;
- trusted proxy IP/CIDR configuration that matches the private proxy hop;
- application image and rollback image references;
- a host backup destination.

Secrets remain outside version control in an operator-owned environment file
with restrictive permissions. The repository provides an example containing
names and instructions only, never deployable values.

The application and rollback image inputs must be explicit. The reference
documentation recommends digest-pinned images; the preflight rejects empty,
bare, or `latest` references and records digest enforcement as a supply-chain
hardening point for TASK-13013.7.

## Fail-Closed Preflight

The repository provides one preflight command that runs before `docker compose
up`. It exits nonzero and prints actionable, non-secret diagnostics when any
required invariant fails.

The preflight validates:

1. The selected file is the production reference profile.
2. Production mode and multi-user authentication are explicit.
3. Required secrets are present, meet minimum strength/length rules, and are
   not recognized placeholders or shared defaults.
4. Redis authentication is configured consistently for the Redis service and
   application URL.
5. Allowed origins are explicit HTTPS origins and match the configured domain.
6. TLS domain/contact inputs are not sample values.
7. Trusted proxy configuration is explicit and does not trust every address.
8. Only the reverse proxy publishes host ports; app, PostgreSQL, and Redis do
   not.
9. The proxy configuration blocks `/ready`, `/health/ready`, and `/metrics`.
10. The backup destination exists, is a directory, and is writable by the
    invoking operator.
11. The rollback image is explicit, differs from the target image, and is not a
    known mutable/default reference.

Preflight must never generate or rewrite secrets, modify system configuration,
start containers, pull images, or install dependencies. It is a validator only.

## Backup and Rollback Contract

The production guide defines four gates:

1. Run preflight successfully.
2. Create a PostgreSQL logical backup in the configured host backup directory.
3. Verify the backup is nonempty and perform a documented staging restore test
   before the first production rollout and after material schema changes.
4. Record the previous application image as the rollback image before upgrade.

Rollback never starts an older binary against a database that has crossed an
incompatible migration boundary. The operator stops writes, restores the
matching pre-upgrade backup when required, selects the recorded rollback image,
runs preflight again, and then restarts the profile.

Redis is treated as disposable cache/coordination state unless a feature
runbook explicitly says otherwise. PostgreSQL and the application data volume
are the durable backup boundary.

## Error Handling and Logging

- Preflight reports variable names and invariant failures, never secret values.
- Multiple independent preflight errors are reported in one run to reduce
  operator iteration time.
- Unexpected parser or filesystem errors fail closed with a concise message.
- Public liveness remains deterministic and contains no exception detail.
- Detailed authenticated readiness continues to sanitize backend exceptions.
- Proxy denials do not forward requests to the application.

## Verification Strategy

Implementation follows test-driven development and adds tests before behavior:

- public liveness returns only the minimal contract for GET and HEAD;
- root readiness and metrics remain registered for private consumers;
- detailed API-v1 health and metrics reject anonymous/non-admin callers and
  permit administrators;
- production Compose topology publishes only proxy ports;
- PostgreSQL and Redis have no host port mappings;
- proxy configuration blocks all private root operational paths;
- every required preflight condition has a failing regression test;
- a complete valid fixture passes preflight without reading real secrets;
- documentation commands and referenced repository paths exist.

Static Compose/proxy contract tests do not start Docker or access the network.
A final manual verification may run `docker compose config` using an already
installed Docker CLI, but the test suite and preflight do not install or start
system software.

## Delivery Boundary

TASK-13013.6 is one reviewable PR containing the production profile, proxy
configuration, preflight validator, health/auth changes, tests, and focused
operator documentation. Existing local/CI profiles remain intact except for
documentation that clearly redirects production users to the new reference
profile.

Container signing, SBOM publication, provenance, and full digest policy are
explicit follow-up work in TASK-13013.7 rather than hidden expansion of this
slice.
