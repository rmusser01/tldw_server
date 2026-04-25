# Docker Compose & Images

This folder contains the base Compose stack for tldw_server, optional overlays, and worker/infra stacks. All commands assume you run from the repo root.

## Base Stack

- Public single-user file: `Dockerfiles/docker-compose.single-user.yml`
- Public multi-user file: `Dockerfiles/docker-compose.multi-user-postgres.yml`
- Shared WebUI overlay: `Dockerfiles/docker-compose.webui.yml`
- Legacy base file: `Dockerfiles/docker-compose.yml`
- Services:
  - Single-user: `app` (FastAPI), `redis`
  - Multi-user: `app` (FastAPI), `postgres`, `redis`
- Start (single-user + WebUI):
  - `make setup-docker-single`
  - `make start-docker-single`
  - `make verify-docker-single`
  - First start in `single_user` mode auto-generates a secure `SINGLE_USER_API_KEY` if missing/placeholder and runs `AuthNZ.initialize --non-interactive` only when `/app/Databases/.authnz_initialized_single_user` is absent in the attached Docker volume.
  - If that marker exists in the volume, AuthNZ initialization is skipped on container restart; it runs again only if the volume is replaced or the marker is removed.
- Start (multi-user, Postgres users DB):
  - `ADMIN_USERNAME=admin ADMIN_PASSWORD='replace-with-a-long-password' make setup-docker-multi`
  - `make start-docker-multi`
  - `make verify-docker-multi`
- Logs and status:
  - `docker compose -f Dockerfiles/docker-compose.single-user.yml ps`
  - `docker compose -f Dockerfiles/docker-compose.single-user.yml logs -f app`

## Persistence and Backups

- Default quickstart persistence uses Docker named volumes, not repo-local folders.
- `app-data` backs `/app/Databases`, which includes the default SQLite AuthNZ DB, per-user databases, first-run marker files, vector stores, and filesystem-backed uploads such as `Databases/user_files`.
- No nested named volume is mounted under /app/Databases/user_databases.
- `postgres_data` and `redis_data` back the bundled Postgres and Redis services.
- Startup configuration is also persisted in `tldw_Server_API/Config_Files/.env`; keep that file with your volume backups.
- `docker compose down` keeps named volumes. `docker compose down -v` deletes them and will remove the persisted databases, user files, and vector stores.
- If you want host-visible storage for easier inspection or external backups, use `Dockerfiles/docker-compose.host-storage.yml` instead of the default compose file:
  - `docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.host-storage.yml up -d --build`
  - Optional WebUI: `docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.host-storage.yml -f Dockerfiles/docker-compose.webui.yml up -d --build`
- The host-storage variant writes under `docker-data/` in the repo root and is optional; the default named-volume quickstart remains the recommended first path.

## Overlays & Profiles

- Production overrides: `Dockerfiles/docker-compose.override.yml`
  - `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.override.yml up -d --build`
  - Sets production flags, disables API key echo, and tightens defaults.

- Reverse proxy (Caddy): `Dockerfiles/docker-compose.proxy.yml`
  - `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.proxy.yml up -d --build`
  - Exposes 80/443 via Caddy; unpublish app port on host.

- Reverse proxy (Nginx): `Dockerfiles/docker-compose.proxy-nginx.yml`
  - `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.proxy-nginx.yml up -d --build`
  - Mount `Samples/Nginx/nginx.conf` and your certs.

- Postgres (basic standalone): `Dockerfiles/docker-compose.postgres.yml`
  - Start a standalone Postgres you can point `DATABASE_URL` to.
  - Example:
    - `export DATABASE_URL=postgresql://tldw_user:TestPassword123!@localhost:5432/tldw_users`
    - `docker compose -f Dockerfiles/docker-compose.postgres.yml up -d`

- Postgres + pgvector + pgbouncer (dev): `Dockerfiles/docker-compose.pg.yml`
  - `docker compose -f Dockerfiles/docker-compose.pg.yml up -d`

- Dev overlay (unified streaming pilot): `Dockerfiles/docker-compose.dev.yml`
  - `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.dev.yml up -d --build`
  - Sets `STREAMS_UNIFIED=1` (keep off in production until validated).

- Host-visible storage variant: `Dockerfiles/docker-compose.host-storage.yml`
  - `docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.host-storage.yml up -d --build`
  - Use this instead of the default base compose file when you want bind mounts under `docker-data/`.

- WebUI overlay: `Dockerfiles/docker-compose.webui.yml`
  - `docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml up -d --build`
  - Adds `webui` (Next.js standalone) on `http://localhost:8080`.

- Embeddings workers + monitoring: `Dockerfiles/docker-compose.embeddings.yml`
  - Base workers only: `docker compose -f Dockerfiles/docker-compose.embeddings.yml up -d`
  - With monitoring profile (Prometheus + Grafana):
    - `docker compose -f Dockerfiles/docker-compose.embeddings.yml --profile monitoring up -d`
  - With debug profile (Redis Commander):
    - `docker compose -f Dockerfiles/docker-compose.embeddings.yml --profile debug up -d`
  - Scale workers: `docker compose -f Dockerfiles/docker-compose.embeddings.yml up -d --scale chunking-workers=3 --scale embedding-workers=2 --scale storage-workers=1 --scale content-workers=1`

## Images

- App image: `Dockerfiles/Dockerfile.prod` (built by base compose)
  - Uses a multi-stage build so compiler/dev packages stay in the builder stage.
- WebUI image: `Dockerfiles/Dockerfile.webui` (used by WebUI overlay)
- Worker image: `Dockerfiles/Dockerfile.worker` (used by embeddings compose)

## Published Images

For the full CI/CD pipeline details (workflow triggers, tagging conventions, attestation, and how to add new images), see [Docs/Development/Container_Image_Lifecycle.md](../Docs/Development/Container_Image_Lifecycle.md).

- The release workflow publishes release artifacts separately from the rolling `main` snapshot workflow.
- `publish-ghcr-main` publishes `main` and `sha-<shortsha>` snapshots for the API, WebUI, and Admin UI images:
  - API: `ghcr.io/<owner>/<repo>:main`
  - WebUI: `ghcr.io/<owner>/<repo>-webui:main`
  - Admin UI: `ghcr.io/<owner>/<repo>-admin-ui:main`
- `sha-<shortsha>` is the cross-image-consistent tag for pinning all three images to the same revision.
- The API image is direct-run friendly.
- The WebUI and Admin UI images are compose-first in v1 unless the operator supplies compatible runtime wiring.
- `BYOK_ENCRYPTION_KEY` is only auto-generated for fresh auth databases. When reusing an existing auth DB or volume with encrypted provider secrets, keep the prior key (or rotate via `BYOK_SECONDARY_ENCRYPTION_KEY`) instead of starting with a blank placeholder.

## Notes

- Run compose commands from repo root so relative paths resolve correctly.
- For production, pair the app with a reverse proxy and set strong secrets in `.env`.
- GPU for embeddings workers: ensure the host has NVIDIA runtime configured and adjust `CUDA_VISIBLE_DEVICES` as needed in the embeddings compose.
- To avoid publishing the app port on host when using a proxy overlay, do not also map `8000:8000` in `app`.

## Troubleshooting

- Health checks: `app` responds on `/ready`; `postgres`/`redis` include health checks.
- If the app fails waiting for DB, verify `DATABASE_URL` and Postgres readiness.
- `single_user` quickstart bootstraps a strong `SINGLE_USER_API_KEY` (when missing/placeholder) and performs one-time AuthNZ initialization based on the marker file `/app/Databases/.authnz_initialized_single_user` stored in the attached volume.
- Initialization is skipped when that marker already exists, and will re-run only after volume replacement or marker removal (force re-init by deleting the marker, or by reinitializing the auth DB and clearing the marker).
- For `multi_user`, run AuthNZ initialization manually to create your admin account.
- View full logs: `docker compose ... logs -f`
