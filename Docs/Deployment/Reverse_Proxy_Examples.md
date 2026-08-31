# Reverse Proxy Examples (Nginx & Traefik)

This guide shows example configurations for running tldw_server behind a reverse proxy with TLS and WebSocket support.

These are custom, non-production examples. The reviewed production boundary is
the [production reference deployment](Production_Reference_Deployment.md) with
`Dockerfiles/Production/Caddyfile`; do not treat a legacy overlay or copied
snippet as a substitute for that standalone reference.

Checked-in starting points:

- Caddy: `Helper_Scripts/Samples/Caddy/Caddyfile.compose`
- Nginx: `Helper_Scripts/Samples/Nginx/nginx.conf`

Important endpoints needing WebSocket upgrade:
- `/api/v1/audio/stream/transcribe`
- `/api/v1/mcp/*`

General guidance
- Terminate TLS at the proxy and forward HTTP to the app (default: `http://app:8000`).
- Configure timeouts high enough for long-running streaming or inference requests.
- Set `tldw_production=true` in the app environment, and restrict CORS (see `ALLOWED_ORIGINS`).

## Nginx

Example server block:

```nginx
server {
    listen 443 ssl;
    server_name your.domain.com;

    ssl_certificate     /etc/letsencrypt/live/your.domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain.com/privkey.pem;

    # Increase limits for uploads and long requests
    client_max_body_size 200m;
    proxy_read_timeout   3600;
    proxy_send_timeout   3600;
    proxy_connect_timeout 60s;

    # Common headers
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    # Propagate request IDs & tracing headers if provided by upstream clients
    proxy_set_header X-Request-ID $http_x_request_id;
    proxy_set_header traceparent $http_traceparent;
    proxy_set_header tracestate $http_tracestate;

    # WebSocket upgrade rules
    map $http_upgrade $connection_upgrade {
        default upgrade;
        ''      close;
    }

    # API and WebUI
    location / {
        proxy_pass http://app:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
    }
}
```

Docker Compose snippet (reverse proxy + app):

```yaml
services:
  app:
    image: tldw-server:prod
    environment:
      - tldw_production=true
      - ALLOWED_ORIGINS=https://your.domain.com
    expose:
      - "8000"

  nginx:
    image: nginx:stable
    volumes:
      # Use the provided sample and adjust paths/domains
      - ./Helper_Scripts/Samples/Nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    ports:
      - "443:443"
    depends_on:
      - app
```

## Traefik

Static configuration (Docker provider recommended). Dynamic config example:

```yaml
http:
  routers:
    tldw:
      rule: Host(`your.domain.com`)
      entryPoints:
        - websecure
      service: tldw
      tls: {}

  services:
    tldw:
      loadBalancer:
        servers:
          - url: "http://app:8000"

  middlewares:
    long-timeouts:
      headers:
        customResponseHeaders:
          X-Accel-Buffering: "no"
```

Docker labels with Traefik (Compose):

```yaml
services:
  app:
    image: tldw-server:prod
    environment:
      - tldw_production=true
      - ALLOWED_ORIGINS=https://your.domain.com
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.tldw.rule=Host(`your.domain.com`)"
      - "traefik.http.routers.tldw.entrypoints=websecure"
      - "traefik.http.routers.tldw.tls=true"
      - "traefik.http.services.tldw.loadbalancer.server.port=8000"
      # WebSocket upgrade handled automatically with Traefik on same router
```

Notes
- Ensure Traefik is configured with the Docker provider and certificate resolver for TLS (e.g., LetsEncrypt).
- Adjust timeouts using Traefik middleware if needed for long-lived streaming.

Sample dynamic config file
- See `Helper_Scripts/Samples/Traefik/traefik-dynamic.yml` for a ready-to-copy
  non-production dynamic configuration. Mount it into Traefik, for example:

```yaml
services:
  traefik:
    image: traefik:v3.0
    command:
      - --providers.docker=true
      - --providers.file.directory=/etc/traefik/dynamic
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.letsencrypt.acme.tlschallenge=true
      - --certificatesresolvers.letsencrypt.acme.email=admin@your.domain.com
      - --certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./Helper_Scripts/Samples/Traefik:/etc/traefik/dynamic:ro
      - ./letsencrypt:/letsencrypt
    ports:
      - "80:80"
      - "443:443"
```

## CORS

In production, restrict CORS to trusted origins. You can set via environment (comma-separated list or JSON array):

```bash
export ALLOWED_ORIGINS=https://your.domain.com,https://admin.your.domain.com
# or JSON array
export ALLOWED_ORIGINS='["https://your.domain.com", "https://admin.your.domain.com"]'
```

This overrides the default origins configured in `tldw_Server_API/app/core/config.py`.

## Security reminders
- Run the app as non-root (Dockerfile.prod already does this).
- Don’t log secrets in production; the app masks the single-user API key when `tldw_production=true`.
- Keep dependencies patched; consider container image scanning (e.g., Trivy).

## Content Security Policy (CSP)

For the WebUI, use CSP to reduce XSS risk.

- Strict CSP (no inline scripts):
  ```nginx
  add_header Content-Security-Policy "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'" always;
  ```

- If inline scripts are present, prefer nonces and inject them into HTML (advanced), or as a last resort allow inline scripts:
  ```nginx
  add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'" always;
  ```

Adjust CSP policies to your WebUI’s needs; if you add third-party fonts or images, update the sources accordingly.
