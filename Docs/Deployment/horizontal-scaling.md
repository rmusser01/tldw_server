# Horizontal Scaling Guide

This document describes how to run multiple tldw_server instances behind a load balancer, sharing rate-limiting and governance state through Redis.

## Prerequisites

| Component | Purpose |
|-----------|---------|
| **Redis 7+** | Shared state for the Resource Governor (rate limits, concurrency leases) |
| **Load balancer** | Distributes traffic across instances (nginx, Caddy, Traefik, cloud ALB, etc.) |
| **Shared filesystem or object store** | Required if instances share SQLite databases; alternatively use PostgreSQL for AuthNZ |

## Configuration

### Environment variables

Set these on every application instance:

```bash
# Required for shared governance state
REDIS_URL=redis://redis-host:6379/0

# Supported high-scale backend for cross-process OpenAI credential mutations.
# The db backend remains valid with direct/session-pooled PostgreSQL at modest
# concurrency; Redis is mandatory with PgBouncer transaction pooling.
OPENAI_OAUTH_REFRESH_LOCK_BACKEND=redis

# AuthNZ — use PostgreSQL for multi-node (SQLite does not support concurrent writers)
DATABASE_URL=postgresql+asyncpg://user:pass@pg-host:5432/tldw_auth

# Optional: tune governor fail mode when Redis becomes unreachable at runtime
# Options: "allow" (default, open-fail) or "deny" (closed-fail)
RG_REDIS_FAIL_MODE=allow
```

### Governor backend selection

The governor factory (`governor_factory.py`) selects the backend automatically:

1. If `REDIS_URL` is set **and** Redis responds to a `PING`, the `RedisResourceGovernor` is used.
2. Otherwise, the `MemoryResourceGovernor` is used (suitable for single-node only).

You can also call the factory explicitly in application code:

```python
from tldw_Server_API.app.core.Resource_Governance.governor_factory import create_governor

governor = create_governor()  # auto-detects from REDIS_URL
```

### OpenAI credential mutation locks

`OPENAI_OAUTH_REFRESH_LOCK_BACKEND` retains its historical name but governs
all whole-row OpenAI credential mutations. The default, `db`, uses a native
file lock with SQLite or a dedicated PostgreSQL advisory-lock pool that starts
with zero connections and is capped at four sessions per application process.
Those sessions are isolated from the main AuthNZ pool. Direct PostgreSQL
connections and PgBouncer session pooling provide correct cross-process
serialization with this backend at modest mutation concurrency.

Use `redis` on every replica for the supported high-scale multi-process profile
or high credential-mutation concurrency. Redis selection is fail-closed:
`REDIS_URL` must be nonempty and reachable; the lock never silently falls back
to process memory or the database.

PostgreSQL advisory locks are session-scoped. A direct PostgreSQL connection
or PgBouncer session pooling preserves that session; PgBouncer transaction
pooling does not. Deployments using PgBouncer transaction pooling must select
the Redis lock backend.

### Credential-runtime rollout and rollback

Deploy the unified provider-credential runtime as a coordinated API-and-worker
cutover, not as a mixed-version rolling update. Before starting the new
version:

1. Stop accepting new background work and drain or stop every old API, Prompt
   Studio worker, TTS worker, and other provider-calling worker.
2. Wait for leased jobs and in-flight provider calls to finish, then verify no
   old process remains.
3. Apply the normal database migrations and start the new API and worker
   versions together. Do not send traffic until the initial provider-override
   refresh has succeeded.

Old processes do not participate in the new cross-process credential locks.
They also cannot safely consume the new secret-free job payloads, which carry
trusted credential scope instead of serialized provider secrets. Running old
and new workers together can therefore race credential mutation or fail queued
provider work.

Rollback has the same drain requirement. Stop all new-version processes before
starting the old version, and inspect jobs created after the cutover before
releasing old workers. Jobs using the new secret-free payload contract must be
completed by the new worker version, cancelled and resubmitted through the old
API, or handled by a forward fix. Do not blindly restart old workers against
that queue.

## What is shared via Redis

| Data | Redis key pattern | Notes |
|------|-------------------|-------|
| Sliding-window request counts | `rg:win:{policy}:{category}:{scope}:{entity}` | ZSET with timestamps |
| Token counters | `rg:win:{policy}:tokens:{scope}:{entity}` | Fixed-window INCRBY with TTL |
| Concurrency leases | `rg:lease:{policy}:{category}:{scope}:{entity}` | ZSET with expiry scores |
| Reservation handles | `rg:handle:{handle_id}` | JSON blob with TTL |
| Idempotency records | `rg:op:{op_id}` | JSON blob with TTL |
| OpenAI credential mutation locks | `tldw:openai-oauth-refresh:{digest}` | Ownership-token lease when `OPENAI_OAUTH_REFRESH_LOCK_BACKEND=redis` |

Resource Governor keys use the configurable `rg:` namespace. OpenAI lock keys
use the fixed `tldw:` namespace. Both Redis data sets use automatic TTLs so
stale state is cleaned up.

## What remains per-instance

| Component | Reason |
|-----------|--------|
| In-memory caches (RAG semantic cache, LRU caches) | No distributed cache layer yet |
| Event broadcaster (SSE/WebSocket) | Events are dispatched locally; no Redis pub/sub bridge |
| Background task queues | FastAPI `BackgroundTasks` are process-local |
| SQLite databases (Media DB, ChaChaNotes) | File-level locking; see limitations below |
| Provider/RAG worker and task caps | `CHAT_SYNC_ADAPTER_MAX_WORKERS` and `CHAT_STREAM_*` are enforced independently by each process |

Worker/task caps therefore multiply across application processes and replicas.
For example, a limit of `32` across two processes on three replicas permits up
to `192` workers in aggregate. Configure the same values on every replica and
size the aggregate against provider, memory, thread, and file-descriptor limits.

The process-local safety controls are:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHAT_SYNC_ADAPTER_MAX_WORKERS` | `32` | Credential-bearing synchronous provider adapter calls; saturation fails closed before dispatch |
| `CHAT_STREAM_DAEMON_MAX_WORKERS` | `32` | Synchronous Chat, Audio, Character, and provider stream work |
| `CHAT_STREAM_CLEANUP_DAEMON_MAX_WORKERS` | `4` | Capacity reserved for synchronous late-work cleanup |
| `CHAT_STREAM_ASYNC_MAX_TASKS` | `256` | Asynchronous provider stream work |
| `CHAT_STREAM_ASYNC_CLEANUP_MAX_TASKS` | `32` | Capacity reserved for asynchronous late-work cleanup |

Each value must be an integer from `1` through `256`. Invalid or out-of-range
values use the listed default, and changes require an application restart.

## Limitations

1. **No distributed event bus.** Server-sent events and WebSocket notifications are per-instance. Clients connected to instance A will not see events triggered on instance B.

2. **Per-instance caches.** The RAG semantic cache and other in-memory caches are not synchronized across instances. Cache warm-up happens independently on each node, and cache invalidation is local only.

3. **SQLite databases.** SQLite does not support concurrent writers from multiple processes on a network filesystem. For multi-node deployments:
   - Migrate AuthNZ to PostgreSQL (`DATABASE_URL=postgresql+asyncpg://...`).
   - Media DB and ChaChaNotes remain SQLite and are per-user; if instances share the same filesystem, only one writer should access a given user database at a time.

4. **Background tasks.** Long-running ingestion or transcription jobs run in-process. There is no distributed task queue (e.g., Celery) yet.

## Docker Compose example

```yaml
version: "3.9"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: tldw
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: tldw_auth
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tldw"]
      interval: 10s
      timeout: 3s
      retries: 5

  app:
    build:
      context: .
      dockerfile: Dockerfiles/Dockerfile
    deploy:
      replicas: 3
    environment:
      REDIS_URL: redis://redis:6379/0
      OPENAI_OAUTH_REFRESH_LOCK_BACKEND: redis
      DATABASE_URL: postgresql+asyncpg://tldw:changeme@postgres:5432/tldw_auth
      AUTH_MODE: multi_user
      RG_REDIS_FAIL_MODE: allow
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    volumes:
      - shared_data:/app/Databases

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app

volumes:
  redis_data:
  pg_data:
  shared_data:
```

## Load balancer configuration

### General guidelines

- Use **least-connections** or **round-robin** balancing for stateless REST endpoints.
- Enable **sticky sessions** (IP hash or cookie-based) if clients rely on WebSocket connections or SSE streams, since the event broadcaster is per-instance.
- Set appropriate health check paths: `GET /api/v1/config/quickstart` or a dedicated `/health` endpoint.
- Forward the original client IP via `X-Forwarded-For`. Forwarded identity is opt-in and is trusted only when the physical peer is a valid IP in the subsystem's trusted-proxy host/CIDR list.

### nginx example

```nginx
upstream tldw_backend {
    least_conn;
    server app:8000;
    # With Docker Compose deploy.replicas, Docker DNS resolves
    # "app" to all replica IPs automatically.
}

server {
    listen 80;

    location / {
        proxy_pass http://tldw_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long-running requests (transcription, ingestion)
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

### Trusted proxy configuration

Set these environment variables on each app instance when forwarding is enabled. If both AuthNZ and Resource Governor use forwarded identity, their trusted-proxy sets must be equivalent and their headers compatible so login lockouts and request governance derive the same client identity:

```bash
# Resource Governor: header and trusted proxy CIDRs
RG_CLIENT_IP_HEADER=X-Forwarded-For
RG_TRUSTED_PROXIES=172.16.0.0/12,10.0.0.0/8

# AuthNZ: explicit opt-in and equivalent trusted proxy CIDRs
AUTH_TRUST_X_FORWARDED_FOR=true
AUTH_TRUSTED_PROXY_IPS=172.16.0.0/12,10.0.0.0/8
```

- `X-Forwarded-For` is parsed as a complete chain from the trusted edge inward (right-to-left); malformed chains fall back to the physical peer. Other `RG_CLIENT_IP_HEADER` values must contain one plain IP literal.
- Invalid physical peers resolve to the safe `unknown` sentinel. Leaving the AuthNZ opt-in or either RG setting unset uses the physical peer.
- Rollout no longer consults legacy raw-IP password-login buckets; account-wide lockout and Resource Governor protections remain active.

## Monitoring

When running multiple instances, aggregate metrics across all nodes:

- Each instance exposes Prometheus metrics at `/metrics` (if enabled).
- Resource Governor metrics (`rg_decisions_total`, `rg_denials_total`, `rg_concurrency_active`) include a `backend` label (`redis` vs `memory`) to confirm all nodes use the shared backend.
- Monitor Redis memory usage and connection count to ensure the governor data fits comfortably in RAM.

## Scaling checklist

- [ ] Redis is deployed and reachable from all app instances
- [ ] `REDIS_URL` is set on every instance
- [ ] `OPENAI_OAUTH_REFRESH_LOCK_BACKEND=redis` is set on every instance
- [ ] AuthNZ database migrated to PostgreSQL
- [ ] Load balancer configured with health checks
- [ ] `RG_CLIENT_IP_HEADER` and `RG_TRUSTED_PROXIES` set for correct IP resolution
- [ ] Sticky sessions enabled for WebSocket/SSE endpoints (if used)
- [ ] Process-local stream/evaluation capacity caps are set consistently on every replica and their aggregate has been sized safely
- [ ] Prometheus scraping configured for all instances
- [ ] Tested failover: the Resource Governor follows its configured Redis fail mode, while OpenAI credential mutations fail closed until Redis recovers
