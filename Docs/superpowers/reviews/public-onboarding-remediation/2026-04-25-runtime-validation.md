# Public Onboarding Remediation Runtime Validation

Date: 2026-04-25

## Environment

- Host OS: macOS 15.7.3 (24G419)
- Docker: Docker version 28.1.1, build 4eba377
- Python: 3.14.3 in `.venv`
- Branch: `dev`
- Baseline commit before this validation pass: `09660ec56`

## Docker Single-User + WebUI

- Commands: clean compose volumes with `COMPOSE_PROJECT_NAME=tldw_ftux_single`, then `make setup-docker-single`, `DOCKER_BUILD=true make start-docker-single`, and `make verify-docker-single`.
- Result: passed. The verifier reached `/health`, `/ready`, `/docs`, `/api/v1/config/quickstart`, `/api/v1/auth/me`, WebUI `http://127.0.0.1:8080`, and first-value ingest/search/detail.
- Notes: the initial Docker image build is large because the backend image installs the project ML/media dependency set. This is not a functional blocker, but users should expect the first build to take time.

## Docker Multi-User + Postgres

- Commands: clean compose volumes with `COMPOSE_PROJECT_NAME=tldw_ftux_multi`, then `ADMIN_USERNAME=tldw-admin ADMIN_PASSWORD='CorrectHorseBatteryStaple1!' ADMIN_EMAIL=tldw-admin@example.com make setup-docker-multi`, `make start-docker-multi`, and `make verify-docker-multi`.
- Result: passed. Postgres, Redis, and app containers reached healthy state; verifier completed login, `/api/v1/auth/me`, and first-value ingest/search/detail.
- Fixed during validation: the docs originally used `ADMIN_USERNAME=admin`, but `admin` is intentionally reserved. Public examples now use `tldw-admin`, and profile setup rejects reserved admin usernames before container startup.

## Local Single-User

- Commands: `TLDW_ENV_FILE=/tmp/tldw_local_single.env make install-local`, `TLDW_ENV_FILE=/tmp/tldw_local_single.env make setup-local-single`, `TLDW_ENV_FILE=/tmp/tldw_local_single.env make start-local-single`, then `TLDW_ENV_FILE=/tmp/tldw_local_single.env make verify-local-single`.
- Result: passed after the local env-file fix. The verifier reached `/health`, `/ready`, `/docs`, `/api/v1/config/quickstart`, `/api/v1/auth/me`, and first-value ingest/search/detail.
- Fixed during validation: local setup and verification honored `TLDW_ENV_FILE`, but server startup loaded the canonical repo `.env`. App config, AuthNZ settings, and `start-local-single` now honor the selected env file.
- Fixed during validation: `/api/v1/config/quickstart` logged `load_comprehensive_config is not defined` and fell back to `/docs`. The endpoint now calls `config_mod.load_comprehensive_config()` and has a regression test for configured quickstart redirects.
- Notes: local validation required elevated command execution in this agent environment because the sandbox cannot bind `127.0.0.1:8000`. That is not a user-facing project issue.

## Remaining Follow-Ups

- No blocking onboarding follow-ups remain for the three public profiles.
- Non-blocking observations: first Docker build is heavy; `setup-wizard-tools` may refresh its lightweight venv dependencies; the existing local workspace emitted a stale `Databases/jobs.db` migration warning, but clean Docker volume validation did not reproduce it as a profile blocker.
