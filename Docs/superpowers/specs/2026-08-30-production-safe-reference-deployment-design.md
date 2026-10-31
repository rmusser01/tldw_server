# Production-Safe Reference Deployment and Health Surface Design

**Status:** Approved
**Date:** 2026-08-30
**Backlog:** TASK-13013.6
**Parent:** TASK-13013

## Purpose

Ship one production reference deployment that fails closed before application
startup, keeps stateful services off the host network, terminates TLS at a
reverse proxy, and exposes only a minimal public liveness response. Detailed
readiness, provider, database, security, and metrics data remain available
through the existing claim-first authentication system.

This design replaces unsafe production guidance with an enforceable reference
contract. It does not turn local-development Compose files into production
profiles and does not introduce a parallel operator-authentication system.

## Current Problems

The current deployment and control-plane surfaces have several unsafe or
ambiguous behaviors:

- local-oriented Compose profiles can default production mode off and publish
  the application on a host interface;
- the existing proxy overlays inherit from other Compose files, so clearing
  published ports depends on Compose merge behavior;
- those overlays proxy every application path, including readiness, setup,
  and metrics endpoints;
- proxy volume paths point at `Dockerfiles/Samples/...`, while the checked-in
  samples live under `Helper_Scripts/Samples/...`;
- public `/health` includes resource-governor metadata rather than a constant
  minimal liveness response;
- detailed health, readiness, and metrics routes are inconsistently protected;
- Docker and Kubernetes examples probe detailed `/ready` anonymously;
- production documentation recommends TLS, backups, and rollback preparation,
  but no canonical deployment command verifies those requirements.

## Goals

1. Provide a dedicated, self-contained production Compose profile.
2. Publish only the TLS reverse proxy; keep the app, PostgreSQL, and Redis on
   private Compose networks.
3. Require explicit, non-placeholder production secrets and immutable current
   and rollback application image references.
4. Make public `/health` a dependency-free, minimal liveness contract.
5. Provide a private, minimal readiness probe for local container or exec
   probes without exposing dependency details.
6. Require the existing `system.logs` permission for detailed health and
   metrics; administrator principals continue to pass through the normal
   admin bypass.
7. Validate topology, TLS, proxy trust, secrets, backup preparation, backup
   artifacts, and rollback preparation through fail-closed deployment gates.
8. Document directly executable backup, upgrade, restore, and rollback flows.
9. Preserve local development profiles except where their probe paths or
   production labeling must change to match the secured route contract.

## Non-Goals

- Replacing Docker Compose with Kubernetes or another orchestrator.
- Normalizing all application-visible client identity in global middleware;
  that remains TASK-13144.
- Publishing, signing, or verifying SBOMs and provenance for every image;
  those remain TASK-13013.7.
- Adding a new service token, monitoring token, or proxy authentication system.
- Exposing anonymous remote readiness or metrics.
- Rewriting every existing local or CI Compose profile.
- Claiming that archive inspection proves a full restore or application/schema
  rollback. A disposable restore drill remains an explicit operator exercise.

## Considered Approaches

### Application authentication only

This is portable, but it does not protect private aliases or stateful services
from an incorrect proxy or Compose topology.

### Proxy and network controls only

This keeps monitoring simple, but direct access to the app bypasses the
boundary and turns a deployment mistake into a data-exposure issue.

### Hybrid defense in depth

This is the selected approach. The app always authenticates detailed
operational surfaces, the proxy blocks private and legacy aliases, stateful
services have no host ports, and preflight validates the final topology.

## Reference Topology

The production profile is standalone. It does not extend or merge the current
development profiles.

```text
Internet
   |
   | host ports 80/443 only
   v
Caddy TLS reverse proxy
   |
   | private edge network
   v
tldw app :8000
   |
   | internal backend network
   +---------------- PostgreSQL :5432
   +---------------- Redis :6379
```

The topology has two networks:

- `edge`: Caddy and the app only;
- `backend`: the app, PostgreSQL, and Redis, declared `internal: true`.

Only Caddy has `ports`. The app may use `expose` for service discovery.
PostgreSQL and Redis have neither `ports` nor public proxy routes.

The edge and backend CIDRs are explicit, private, configurable, and
non-overlapping. Preflight rejects public, default-route, malformed, or
overlapping CIDRs and verifies that only the declared services join each
network. This avoids trusting an arbitrary Docker bridge while allowing an
operator to choose subnets that do not conflict with the host environment.

The production profile sets multi-user authentication and production mode
explicitly. It never relies on a development fallback. Required values use
Compose's fail-fast `${NAME:?message}` form where practical, and semantic
preflight performs the stronger checks Compose cannot express.

## Operational Route Contract

| Route | Consumer | Public proxy | App guard | Response |
| --- | --- | --- | --- | --- |
| `GET/HEAD /health` | load balancer or uptime check | allowed | public | exactly minimal liveness |
| `GET/HEAD /internal/ready` | container-local or exec probe | blocked | loopback only | ready/not-ready only |
| `/ready`, `/health/ready` | legacy detailed aliases | blocked | `system.logs` | detailed readiness |
| `/api/v1/healthz`, `/api/v1/readyz` | legacy API aliases | blocked | `system.logs` | operational detail |
| `/api/v1/health*` | operator | allowed | `system.logs` | detailed diagnostics |
| `GET /metrics` | Prometheus/operator | allowed | `system.logs` | Prometheus text |
| `/api/v1/metrics*` | operator | allowed | `system.logs` | detailed metrics/control |

The public `/health` body is exactly a stable object such as
`{"status":"ok"}`. It contains no timestamp, database type, provider state,
queue depth, policy metadata, build identifier, environment name, path, or
secret. It does not probe dependencies, so a dependency outage cannot turn a
liveness check into a restart loop.

`/internal/ready` uses the same underlying readiness calculation but returns
only `{"status":"ready"}` or `{"status":"not_ready"}` with HTTP 200 or 503.
It is excluded from OpenAPI, permits only loopback peers, and is denied by the
public proxy before the catch-all route. Production Compose overrides the
image health check to call this path locally. Kubernetes examples use an exec
probe so the request originates inside the container rather than from the
kubelet network.

All detailed control-plane routes are protected regardless of environment.
The implementation uses the existing claim-first
`RequirePermission(SYSTEM_LOGS)` dependency. Administrator principals pass
through the established admin bypass, while Prometheus may use a constrained
virtual API key whose principal has `system.logs`. Mutating endpoints such as
metrics reset retain their stricter admin guard in addition to the router-level
diagnostics permission.

Anonymous detailed requests fail through the normal authentication path.
Requests for proxy-blocked aliases receive an edge 404 without reaching the
application. This keeps proxy blocking as defense in depth rather than the
primary authorization boundary.

## Shared Readiness Calculation

The implementation owns one internal readiness snapshot builder. It gathers
database, workflow-schema, provider, engine, OpenTelemetry, and resource
governor state once and returns a typed internal result.

Consumers project that result differently:

- public liveness does not call it;
- internal readiness returns only the boolean state;
- authenticated operator routes return the sanitized detail;
- legacy authenticated aliases reuse the same projection.

This removes the current drift between `readyz()`, `api_readiness()`, and
`readiness_check()` while preserving sanitized failure behavior.

## Proxy and Client-Identity Contract

Caddy terminates TLS and explicitly overwrites, rather than blindly preserving,
the incoming `X-Forwarded-For`, `X-Real-IP`, and `X-Forwarded-Proto` headers.
The app trusts forwarding data only from the configured edge CIDR.

Until TASK-13144 provides one global physical-peer contract, the deployment
must align every applicable existing trust setting:

- Uvicorn `FORWARDED_ALLOW_IPS`;
- AuthNZ `AUTH_TRUST_X_FORWARDED_FOR` and `AUTH_TRUSTED_PROXY_IPS`;
- Setup `TLDW_TRUSTED_PROXIES`;
- Resource Governor `RG_CLIENT_IP_HEADER` and `RG_TRUSTED_PROXIES`;
- MCP `MCP_TRUST_X_FORWARDED` and `MCP_TRUSTED_PROXY_IPS` when MCP is enabled.

Preflight rejects wildcard trust and inconsistent networks. The reference
configuration does not broaden trust to the backend network.

The proxy evaluates explicit deny matchers before its catch-all application
route. It blocks internal readiness, legacy detailed aliases, and setup routes
after bootstrap. It permits canonical authenticated API-v1 diagnostics and
`/metrics`, allowing existing application authentication to make the final
access decision.

## Production Configuration and Secret Contract

The profile requires explicit values for:

- public domain and ACME contact;
- HTTPS-only allowed origins matching the public domain;
- JWT signing and session-encryption material for multi-user mode;
- PostgreSQL user, database, password, and application connection URL;
- Redis password and authenticated application connection URL;
- administrator bootstrap or an explicit existing-installation mode;
- edge and backend CIDRs and aligned trusted-proxy settings;
- immutable target and rollback application image references;
- an absolute host backup destination;
- completed setup with remote setup disabled.

The repository provides a names-only example environment file. It contains no
deployable credential or usable production default. The real raw environment
file remains outside version control, must have restrictive permissions, and
is never copied into documentation or logs.

Preflight parses the raw file without shell evaluation. It checks required
secret length and known-placeholder deny lists, verifies secrets that must be
distinct, safely compares PostgreSQL and Redis connection credentials, and
reports variable names without printing values. Password URL encoding is
handled through URL parsing rather than string splitting.

The current and rollback app images must use a documented immutable `sha-*`
tag or digest and must differ. Third-party services use exact version tags in
this task; digest and provenance enforcement for the full image set remains
TASK-13013.7.

## Fail-Closed Deployment Gates

The canonical production deployment command coordinates two layers of checks.

### Static and semantic preflight

This validator is deterministic and does not start containers, pull images,
generate secrets, or modify configuration. It aggregates independent errors
and exits nonzero when any invariant fails.

It validates:

1. the selected file is the standalone production profile;
2. production and multi-user modes are explicit;
3. required secrets are present, strong enough, distinct where required, and
   not recognized defaults or placeholders;
4. PostgreSQL and Redis credentials match their application URLs;
5. TLS domain, ACME contact, allowed origins, and setup state are production
   safe;
6. proxy trust settings are explicit, private, aligned, and non-wildcard;
7. only Caddy publishes host ports and only ports 80/443 are published;
8. the app, PostgreSQL, and Redis are not host-published;
9. network membership and `internal` backend isolation match the contract;
10. proxy route order blocks private aliases and completed setup routes;
11. the backup destination is absolute, separate from live data, present, and
    writable;
12. current and rollback images are explicit, immutable, and different.

The production Compose profile also retains required-variable expansion and a
one-shot semantic preflight dependency so bypassing the wrapper does not
silently restore credential defaults.

### Operational deployment verification

The host-side deploy command performs checks that a static validator cannot
prove and stops before the target app is started when any step fails:

1. render and inspect `docker compose config`;
2. pull and smoke-test the target and rollback application images;
3. create a PostgreSQL custom-format dump and validate it with
   `pg_restore --list`;
4. create a Redis RDB snapshot and validate it with `redis-check-rdb`;
5. archive the application data volume while the app is stopped and verify
   that the archive can be listed and read;
6. write a non-secret manifest recording timestamps, image references, and
   artifact checksums;
7. start the target profile only after every required gate succeeds.

No container receives the Docker socket. Docker inspection, pull, and image
smoke operations remain host-side. Backup helpers receive only the mounts and
credentials needed for their specific operation.

## Backup, Upgrade, and Rollback Contract

The durable recovery boundary includes:

- PostgreSQL logical data;
- the application data volume containing per-user databases and managed
  content;
- Redis security/coordination state for deployments that use it;
- operator-managed configuration and secrets, backed up separately through
  the operator's secret-management process.

Before an upgrade, the canonical command creates and verifies fresh artifacts
and records the currently running immutable image as the rollback image.

Archive inspection proves that the backup files are structurally readable; it
does not prove an end-to-end restore. The production runbook requires a
disposable restore drill before the first rollout and after material storage
or migration changes.

Rollback never starts an older binary blindly against a database that may have
crossed an incompatible migration boundary. The operator stops writes,
restores the matching pre-upgrade artifacts when compatibility is uncertain,
selects the recorded rollback image, reruns preflight, and starts the prior
profile. Extended recovery timing and soak evidence remain TASK-13013.9.

## Error Handling and Logging

- Preflight reports variable names and invariant failures, never secret values.
- Multiple independent validation errors are returned in one run.
- Unexpected parser, subprocess, or filesystem errors fail closed.
- Public liveness is deterministic and contains no exception detail.
- Authenticated readiness continues to sanitize backend exceptions.
- Health and metrics responses use no-store cache controls.
- Proxy denials are handled at the edge and are not forwarded to the app.
- Operational verification records checksums and status, not credentials or
  connection URLs.

## Compatibility and Migration

Securing detailed routes is an intentional compatibility change:

- Docker health checks move from `/ready` to local `/internal/ready`;
- Kubernetes HTTP readiness examples move to exec probes against that local
  path;
- Prometheus must send a scoped `X-API-KEY` or other existing credential whose
  principal has `system.logs`;
- human operators may continue to use administrator JWTs or API keys;
- production documentation points only to the standalone reference profile;
- legacy proxy examples are corrected or clearly labeled non-production;
- canonical production assets are scanned for `change-me`, `changeme`, known
  test passwords, sample domains, and other deployable placeholders.

Feature-level user-facing capability endpoints are not reclassified wholesale
by this task. The protected boundary covers canonical control-plane health,
readiness, provider/database detail, security health, and metrics surfaces.

## Verification Strategy

Implementation follows test-driven development. Coverage includes:

- exact public liveness body and headers for GET and HEAD;
- internal readiness loopback allow, remote deny, and detail minimization;
- forwarding-header spoof cases at the internal probe boundary;
- anonymous 401, insufficient-permission 403, `system.logs` success, and admin
  success for detailed health and metrics;
- continued admin-only enforcement for metrics reset;
- one shared readiness snapshot feeding internal and operator projections;
- standalone Compose topology with only Caddy host ports;
- private, non-overlapping edge/backend networks and exact membership;
- PostgreSQL and Redis authentication without deployable defaults;
- proxy header overwrite and deny-route ordering;
- every static preflight rejection condition plus one valid fixture;
- secret-redaction tests for parser and subprocess failures;
- mocked operational verification for image, PostgreSQL, Redis, app-data, and
  manifest failure modes;
- Docker/Kubernetes probe and documentation contract tests;
- `docker compose config` validation when Docker is already available;
- focused Bandit on touched Python and shell scopes.

Static tests do not require network access or running containers. Live image,
backup, TLS, and restore verification are documented operator or trusted-CI
gates and are recorded honestly when the execution environment cannot provide
them.

## Delivery Boundary

TASK-13013.6 is one reviewable PR containing:

- the standalone production profile and Caddy configuration;
- environment contract and fail-closed validators;
- canonical deployment/backup/rollback helpers;
- health, readiness, metrics, and authorization changes;
- Docker and Kubernetes probe migrations;
- focused tests and production operator documentation;
- corrections or deprecation notices for unsafe production examples.

Global client-identity middleware remains TASK-13144. Full image digest,
attestation, SBOM, and dependency provenance enforcement remains
TASK-13013.7. Long-duration capacity, restore-time, and soak certification
remains TASK-13013.9.
