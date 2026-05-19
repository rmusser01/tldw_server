# Admin UI Deployment Guide

## 1. Docker Build

The Dockerfile at `Dockerfiles/Dockerfile.admin-ui` uses a two-stage build:

### Builder Stage (`oven/bun:1.3.2-debian`)

- Installs dependencies with `bun install --frozen-lockfile`
- Runs `bun run build` to produce the Next.js standalone output
- Build-time args are baked into the client JavaScript bundle

### Runtime Stage (`node:20-bookworm-slim`)

- Copies only the standalone output (no source code or `node_modules`)
- Runs as non-root user `adminui` (UID 10003)
- Exposes port 3001
- Entry point: `node server.js`

### Build Args

These are embedded into the client bundle at build time and cannot be changed at runtime:

| Arg | Default | Description |
|-----|---------|-------------|
| `NEXT_PUBLIC_API_URL` | (empty) | Backend API base URL. In Docker Compose this defaults to `http://app:8000`. |
| `NEXT_PUBLIC_API_VERSION` | `v1` | API version path segment. |
| `NEXT_PUBLIC_DEFAULT_AUTH_MODE` | `password` | Login mode shown by default (`password` or `apikey`). |
| `NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN` | `false` | Whether to show API key login option. |
| `NEXT_PUBLIC_BILLING_ENABLED` | `false` | Enable billing/plans UI. |

### Building

```bash
docker build \
  -f Dockerfiles/Dockerfile.admin-ui \
  --build-arg NEXT_PUBLIC_API_URL=http://app:8000 \
  -t tldw-admin-ui:prod .
```

## 2. Docker Compose

Use `docker-compose.admin-ui.yml` as an overlay on top of the main `docker-compose.yml`:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.yml \
  -f Dockerfiles/docker-compose.admin-ui.yml \
  up -d --build
```

The compose file:
- Builds from `Dockerfiles/Dockerfile.admin-ui` with context at repo root
- Maps `${ADMIN_UI_PORT:-3001}` to container port 3001
- Passes runtime environment variables (JWT secrets, auth mode)
- Depends on the `app` service with `condition: service_healthy`

### Service Dependencies

The admin UI requires the backend (`app`) to be healthy before starting. The compose `depends_on` with `service_healthy` condition enforces this. If the backend is unreachable at runtime, the proxy returns 502 and the readiness probe reports `not_ready`.

## 3. Environment Variables

### Build-Time (NEXT_PUBLIC_*)

These are compiled into the JavaScript bundle. Changing them requires a rebuild.

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | Backend URL visible to the Next.js server (not the browser). |
| `NEXT_PUBLIC_API_VERSION` | No | `v1` | API version prefix. |
| `NEXT_PUBLIC_DEFAULT_AUTH_MODE` | No | `password` | `password` or `apikey`. |
| `NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN` | No | `false` | Show API key login tab. |
| `NEXT_PUBLIC_BILLING_ENABLED` | No | `false` | Enable billing pages. |

### Runtime

These are read by the Node.js server at startup and by middleware on each request.

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `NODE_ENV` | No | `production` | Set by the Dockerfile. |
| `PORT` | No | `3001` | HTTP listen port. |
| `HOSTNAME` | No | `0.0.0.0` | Bind address. |
| `JWT_SECRET_KEY` | Yes (multi-user) | (none) | Must match the backend's JWT signing key. Used by middleware for local token verification. |
| `JWT_ALGORITHM` | No | `HS256` | `HS256`, `HS384`, or `HS512`. |
| `JWT_SECONDARY_SECRET` | No | (none) | For key rotation: middleware accepts tokens signed with either secret. |
| `AUTH_MODE` | No | `multi_user` | `single_user` or `multi_user`. |
| `NEXT_PUBLIC_SENTRY_DSN` | No | (none) | Sentry error reporting DSN. |
| `NEXT_TELEMETRY_DISABLED` | No | `1` | Disables Next.js telemetry (set in Dockerfile). |

### Validation

The `lib/env.ts` module validates `NEXT_PUBLIC_*` variables using Zod at runtime. If `NEXT_PUBLIC_API_URL` is missing or not a valid URL, the application throws on first access.

## 4. Health Probes

### Liveness: GET /api/health

Returns `200` with a JSON timestamp. Verifies the Next.js process is running. Does not probe the backend.

### Readiness: GET /api/health/ready

Returns `200 { status: "ready" }` when the backend is reachable. Returns `503 { status: "not_ready" }` when the backend health check fails or times out (2-second timeout).

### Docker HEALTHCHECK

The Dockerfile configures:
```
HEALTHCHECK --interval=30s --timeout=5s --retries=5
  CMD node -e "fetch('http://localhost:3001/api/health/ready').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
```

### Kubernetes Example

```yaml
livenessProbe:
  httpGet:
    path: /api/health
    port: 3001
  initialDelaySeconds: 5
  periodSeconds: 15
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /api/health/ready
    port: 3001
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3
```

## 5. Security Headers

Configured in `next.config.js` via the `headers()` function. Applied to all routes (`/:path*`).

| Header | Value | Notes |
|--------|-------|-------|
| Content-Security-Policy | `default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'` | In development, `connect-src` adds `ws:` and `script-src` adds `'unsafe-eval'` for HMR/source maps. |
| X-Frame-Options | `DENY` | Prevents embedding in iframes. |
| X-Content-Type-Options | `nosniff` | Prevents MIME sniffing. |
| Strict-Transport-Security | `max-age=63072000; includeSubDomains; preload` | HSTS with 2-year max-age. |
| Referrer-Policy | `strict-origin-when-cross-origin` | |
| Permissions-Policy | `camera=(), microphone=(), geolocation=()` | Disables browser APIs not needed. |
| X-DNS-Prefetch-Control | `on` | Allows DNS prefetching for performance. |

The `poweredByHeader` option is set to `false` to suppress the `X-Powered-By: Next.js` header.

## 6. Monitoring

### Sentry

If `NEXT_PUBLIC_SENTRY_DSN` is set, `@sentry/nextjs` is activated via `next.config.js`. Error boundaries and unhandled exceptions are reported automatically. Sentry is configured with `silent: true` and `disableLogger: true` to avoid noisy build output.

### Structured Logging

The `lib/logger.ts` module provides structured JSON logging with fields:
- `component` -- which subsystem produced the log (e.g., `proxy`, `auth`)
- `path` -- the request path
- `method` -- HTTP method
- `error` -- error message
- `timeout` -- whether a timeout occurred

### Correlation IDs

Every request gets an `X-Request-Id` header:
- The middleware generates one via `lib/correlation-id.ts` if not already present on the incoming request.
- The proxy forwards it to the backend.
- It is included in error responses and logs for end-to-end tracing.

### Bundle Analysis

Set `ANALYZE=true` to enable `@next/bundle-analyzer` during builds:

```bash
ANALYZE=true bun run build
```
