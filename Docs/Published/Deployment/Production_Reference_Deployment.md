# Production Reference Deployment

This is the production-safe reference profile for a single Docker host. It is
separate from the quickstart and legacy overlay examples. The profile is
fail-closed: only Caddy publishes host ports, PostgreSQL and Redis remain on an
internal network, application images are immutable inputs, and startup depends
on an offline preflight.

The public surface is intentionally small. `GET /health` returns exactly
`{"status":"ok"}`. Caddy returns `404` for `/internal/ready`, `/ready`,
`/health/ready`, `/api/v1/healthz`, `/api/v1/readyz`, `/setup`, `/setup/*`,
`/api/v1/setup`, and `/api/v1/setup/*`. Detailed health, readiness, provider,
database, and metrics routes require an authenticated operator principal.

## Files and boundaries

- `Dockerfiles/docker-compose.production.yml` is the standalone topology.
- `Dockerfiles/Monitoring/docker-compose.production.yml` is its standalone,
  private-network monitoring companion.
- `Dockerfiles/Production/Caddyfile` terminates TLS and overwrites forwarding
  headers before proxying to `app:8000`.
- `Dockerfiles/production.env.example` is a names-only template, not a usable
  environment file.
- `Helper_Scripts/Deployment/production_preflight.py` performs offline checks.
- `Helper_Scripts/Deployment/production_deploy.py` owns verified deployment and
  restore-backed rollback ordering.
- `TLDW_BACKUP_DIR` is an absolute, operator-owned directory separate from live
  volumes. Back up the secret/config file separately; it is deliberately absent
  from non-secret deployment manifests.

Do not add host port mappings for the app, PostgreSQL, or Redis. The reference
topology is the boundary: only Caddy publishes `80` and `443`.

## 1. Create the raw environment securely

Choose absolute paths outside the repository. Copy the names-only template,
restrict it before filling any values, and populate it through an editor or
secret-management tool that does not echo secrets into terminal history.

```bash
export PRODUCTION_ENV_FILE=/srv/tldw/secrets/production.env
export TLDW_BACKUP_DIR=/srv/tldw/backups
install -d -m 700 "$(dirname "$PRODUCTION_ENV_FILE")" "$TLDW_BACKUP_DIR"
install -m 600 Dockerfiles/production.env.example "$PRODUCTION_ENV_FILE"
chmod 600 "$PRODUCTION_ENV_FILE"
```

Fill every required value. Important invariants include:

- `TLDW_APP_IMAGE` and `TLDW_ROLLBACK_IMAGE` are different immutable digests or
  `sha-<commit>` tags. `CADDY_IMAGE`, `POSTGRES_IMAGE`, and `REDIS_IMAGE` use
  exact full versions or digests.
- `TLDW_EDGE_SUBNET` and `TLDW_BACKEND_SUBNET` are non-overlapping private IPv4
  CIDRs. Every trusted-proxy input derives from the edge CIDR.
- `TLDW_PUBLIC_DOMAIN`, `TLDW_ACME_EMAIL`, and `ALLOWED_ORIGINS` describe the
  real HTTPS origin.
- New installations provide strong `ADMIN_*` bootstrap values and set
  `TLDW_SETUP_COMPLETED=true`. Existing initialized installations attest that
  state with `TLDW_EXISTING_INSTALLATION=true` and do not retain an admin
  bootstrap password.
- PostgreSQL and Redis URLs encode the same external credentials supplied to
  their services. Do not reuse secrets across fields.
- Remote setup remains disabled. The public proxy denies setup routes.

Make a protected, operator-managed backup of the secret/config file. Do not put
it inside `TLDW_BACKUP_DIR`, because manifests cover PostgreSQL, Redis, and
app-data artifacts but never secrets.

```bash
install -d -m 700 /srv/tldw/secret-backups
install -m 600 "$PRODUCTION_ENV_FILE" /srv/tldw/secret-backups/production.env
```

## 2. Run the fail-closed preflight

Run this before the first deployment and before every upgrade. It validates raw
env permissions and parsing, secret strength, URL credential alignment, TLS and
origin inputs, network separation, immutable images, setup state, writable
backup storage, static Compose topology, proxy denials, and trusted-proxy
alignment. It performs no Docker, network, or provider calls.

```bash
make production-preflight \
  PRODUCTION_ENV_FILE="$PRODUCTION_ENV_FILE"
```

Stop on any error. The containerized preflight uses environment-only inputs and
read-only mounts; environment-only skips apply only to the host file-mode and
host-writability checks already enforced by the authoritative Make target.

## 3. Prove recovery before rollout

Before the first rollout or a material migration, complete a disposable restore
drill on an isolated host or project. Do not call a backup valid merely because
it exists.

The deployment command creates and verifies all three recovery classes:

- a PostgreSQL custom-format dump checked with `pg_restore --list`;
- a Redis RDB checked with `redis-check-rdb`;
- an app-data tar inspected locally for unsafe paths, links, devices, and a
  readable nonempty regular member.

For the disposable restore drill, copy one complete manifest directory to the
isolated environment, point an equivalent pinned Compose configuration at it,
run the documented rollback command, and verify authenticated application data.
Record the drill result outside the manifest directory.

Archive inspection has limits: it proves the tar is readable and rejects unsafe
member shapes, but it does not prove application-level database consistency or
semantic correctness. A real disposable restore drill remains mandatory.

## 4. Deploy in the verified order

```bash
make production-deploy \
  PRODUCTION_ENV_FILE="$PRODUCTION_ENV_FILE"
```

The command renders Compose only in memory, revalidates the resolved model,
pulls and network-isolates smoke checks for both current and rollback images,
starts only healthy PostgreSQL and Redis, stops app/Caddy writes, creates and
verifies all backup artifacts, writes a checksummed non-secret manifest, and
only then starts the target profile. A failed gate prevents every later command.

Keep the emitted absolute manifest path with the change record. Each upgrade
requires a fresh verified manifest; do not reuse an artifact set after state has
advanced.

## 5. Verify public and operator surfaces

From outside the host, verify the exact public liveness response and denial of
private, legacy, and setup paths:

```bash
curl --fail --silent "https://$TLDW_PUBLIC_DOMAIN/health"
curl --silent --output /dev/null --write-out '%{http_code}\n' \
  "https://$TLDW_PUBLIC_DOMAIN/internal/ready"
curl --silent --output /dev/null --write-out '%{http_code}\n' \
  "https://$TLDW_PUBLIC_DOMAIN/ready"
curl --silent --output /dev/null --write-out '%{http_code}\n' \
  "https://$TLDW_PUBLIC_DOMAIN/setup"
```

The first command must print only `{"status":"ok"}`; each denied path must
return `404`. Container and Kubernetes readiness probes use loopback
`/internal/ready`, so detailed readiness is never anonymously remote.

Use a trusted operator path and an existing principal with `system.logs` for
detailed diagnostics and metrics. The app has no host-published port. The
following commands execute the request inside the app container rather than
weakening that boundary or traversing the public Caddy denial rules. Export
`TLDW_OPERATOR_TOKEN` without placing it in shell history, then run. The
in-container request sends it using the standard `Authorization: Bearer`
header.

```bash
for path in /api/v1/health/detailed /api/v1/metrics/text; do
  docker compose --env-file "$PRODUCTION_ENV_FILE" \
    -f Dockerfiles/docker-compose.production.yml \
    exec -T -e TLDW_OPERATOR_TOKEN app python -c \
    'import os, sys, urllib.request
request = urllib.request.Request(
    "http://localhost:8000" + sys.argv[1],
    headers={"Authorization": "Bearer " + os.environ["TLDW_OPERATOR_TOKEN"]},
)
print(urllib.request.urlopen(request, timeout=5).read().decode())' "$path"
done
```

Provider and database detail is operator information. Never make those routes,
the legacy readiness aliases, or metrics anonymous at the public proxy.

## 6. Authenticate Prometheus

Create a dedicated existing AuthNZ principal/role whose only diagnostic
permission is `system.logs`, then create an API key for that principal. Do not
introduce a shared anonymous metrics token and do not reuse an administrator's
general-purpose credential.

Write the returned key once to an owner-only file without printing it:

```bash
export TLDW_METRICS_API_KEY_FILE=/srv/tldw/secrets/prometheus-api-key
install -m 600 /dev/null "$TLDW_METRICS_API_KEY_FILE"
printf '%s' "$TLDW_METRICS_API_KEY" > "$TLDW_METRICS_API_KEY_FILE"
unset TLDW_METRICS_API_KEY
chmod 600 "$TLDW_METRICS_API_KEY_FILE"
```

Pin the monitoring images and create Grafana credentials without echoing the
password. Then start the standalone production monitoring companion after the
main production stack is healthy:

```bash
export PROMETHEUS_IMAGE=prom/prometheus:v2.55.1
export ALERTMANAGER_IMAGE=prom/alertmanager:v0.30.1
export GRAFANA_IMAGE=grafana/grafana:11.5.2
export GRAFANA_ADMIN_USER=tldw-operator
printf 'Grafana admin password: ' >&2
IFS= read -r -s GRAFANA_ADMIN_PASSWORD
printf '\n' >&2
export GRAFANA_ADMIN_PASSWORD
docker compose -f Dockerfiles/Monitoring/docker-compose.production.yml \
  up -d --wait
```

`Dockerfiles/Monitoring/docker-compose.production.yml` mounts the API-key file
read only at `/run/secrets/tldw_metrics_api_key`. Only Prometheus joins the
existing `tldw-production_edge` network to scrape `app:8000`; Alertmanager and
Grafana remain on the companion's separate monitoring network. Prometheus uses
the key as a Bearer credential for `/api/v1/metrics/text`. All three services
publish only on host loopback; use an authenticated SSH tunnel for remote
operator access. The legacy `docker-compose.monitoring.yml` is a non-production
customization overlay and is not compatible with the standalone production
boundary.

Rotate the API key and file together, then reload Prometheus. Stop monitoring
without removing its Grafana data by omitting `-v`:

```bash
docker compose -f Dockerfiles/Monitoring/docker-compose.production.yml down
```

## 7. Upgrade and restore-backed rollback

For an upgrade, change only immutable image references in the protected raw env
file, retain the prior image as `TLDW_ROLLBACK_IMAGE`, rerun preflight, and run a
fresh production deploy. Verify public liveness, public denials, authenticated
health, providers, database state, and metrics before declaring the change
healthy.

If migration compatibility or data integrity is uncertain, stop writes and use
the matching pre-upgrade manifest:

```bash
make production-rollback \
  PRODUCTION_ENV_FILE="$PRODUCTION_ENV_FILE" \
  PRODUCTION_MANIFEST=/srv/tldw/backups/deployment-TIMESTAMP/manifest.json
```

Rollback is never binary-only. It verifies the manifest, Compose checksum,
image pairing, artifact sizes, and checksums; stops app/Caddy; restores the
PostgreSQL dump, Redis RDB, and app-data volume; then starts the prior image and
writes a separate rollback completion manifest. Missing, altered, or failed
restore artifacts prevent startup.

## 8. Backup retention and current limits

Retain complete manifest directories atomically. A manifest is useful only with
all three matching artifacts and the independently protected secret/config
backup. Replicate them to operator-managed storage, test restores on a schedule,
and document retention and deletion policy.

Known follow-up boundaries are explicit:

- `TASK-13013.7` owns broader image provenance, digest, attestation, and SBOM
  enforcement.
- `TASK-13013.9` owns long-duration capacity, restore-time, and soak evidence.
- `TASK-13144` owns the global client-identity middleware and physical-peer
  normalization across trusted proxy consumers.

Those follow-ups do not relax this profile's fail-closed preflight, private
stateful services, authenticated operator surfaces, or restore-backed rollback.
