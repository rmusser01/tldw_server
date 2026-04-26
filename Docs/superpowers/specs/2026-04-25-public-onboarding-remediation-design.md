# Public Onboarding Remediation Design

Date: 2026-04-25
Status: Revised for user review
Owner: Codex brainstorming session

## Summary

Fix the public first-time setup experience identified by the public onboarding readiness review. The remediation keeps the three public setup profiles as peers, but changes the public contract so every profile follows the same lifecycle:

1. prepare
2. start
3. verify
4. first chat/search value
5. audio-path clarity

The existing `tldw-setup` CLI wizard is the shared setup and verification layer. The remediation should extend that CLI where necessary instead of creating a second setup wizard or duplicating complex setup behavior in Makefile shell blocks.

## Problem

The review found that all three public profiles currently fail before a straightforward first-value moment on macOS:

- Docker single-user + WebUI builds but leaves a broken stack with no reachable API or WebUI.
- Docker multi-user + Postgres builds but fails before first admin creation or login.
- Local single-user fails before a reachable API and labels an install command that also starts runtime behavior.

The review also found documentation and command-contract traps:

- Docker multi-user docs omit required `SESSION_ENCRYPTION_KEY`.
- Docker multi-user docs say to run `AuthNZ.initialize` manually, while runtime already attempts non-interactive auth initialization and can exit before the manual step exists.
- Local `quickstart-install` is not install-only.
- Docker `/setup` guidance conflicts on restart versus rebuild behavior.
- Audio docs use single-user `X-API-KEY` examples but do not show the multi-user bearer-token path.

These are public readiness issues, not polish issues. A new user should not need to inspect Docker logs, entrypoint scripts, Makefile targets, or internal AuthNZ state to complete first setup.

## Goals

- Preserve all three public setup profiles as peers:
  - Docker single-user + WebUI
  - Docker multi-user + Postgres
  - Local single-user
- Allow public command and profile-flow changes when they improve first-time UX.
- Promote `tldw-setup` into the shared prepare/verify contract for first-time setup.
- Make each profile reach the same bar: first auth success, first chat readiness, first ingest/search success, and aligned audio guidance.
- Split install, prepare, start, and verify behavior so command names match what they do.
- Replace silent or late failures with clear diagnostics and recovery instructions.
- Preserve equal support for macOS, Linux, and Windows/WSL public setup paths.
- Add tests that lock down command boundaries, Docker compose contract, CLI wizard behavior, and onboarding docs.

## Non-Goals

- Build a new setup wizard separate from `tldw-setup` and `/setup`.
- Redesign the product WebUI onboarding flow beyond what is needed to make the documented profiles reachable.
- Guarantee every optional provider, local model, or accelerated audio backend works in this remediation.
- Convert all storage backends or content databases to Postgres.
- Remove legacy quickstart aliases if keeping them as compatibility wrappers is cheap and low-risk.

## Target User Standard

The standard is not "a developer can eventually debug it." The fixed state should let a careful first-time self-hoster:

- choose any public profile
- run the documented commands in order
- see an understandable success or failure at each step
- authenticate using the profile's auth mode
- verify the server and docs are reachable
- run or clearly skip first chat because provider credentials are missing
- ingest and search a repo-local sample item
- understand how audio verification works for their auth mode

## Cross-Platform Contract

All three public profiles should be documented and supported for macOS, Linux, and Windows/WSL.

Expected behavior:

- Docker profile commands work from a normal POSIX shell on macOS/Linux and from WSL on Windows.
- Windows documentation states WSL as the supported terminal path when a command depends on POSIX shell behavior.
- Local profile docs call out platform-specific prerequisites where they differ, especially Python, FFmpeg, and shell activation commands.
- Make targets remain the canonical repo checkout commands, but docs should include the direct `tldw-setup` command underneath when that improves Windows/WSL troubleshooting.
- Verification should not rely on macOS-only behavior.

## Public Profile Contract

Each profile should present the same shape.

### Prepare

Prepare validates prerequisites and writes the minimum required environment. It should not start a long-running server.

Expected behavior:

- single-user profiles generate or preserve `SINGLE_USER_API_KEY`
- multi-user profile validates `DATABASE_URL`
- multi-user profile requires or generates all required secrets:
  - `JWT_SECRET_KEY`
  - `SESSION_ENCRYPTION_KEY`
  - `MCP_JWT_SECRET`
  - `MCP_API_KEY_SALT`
  - `BYOK_ENCRYPTION_KEY`, when required by the existing AuthNZ/BYOK rules
- local profile creates `.venv` and installs dependencies only when the user runs the install command
- Docker profiles prepare `.env` through a deterministic profile-aware path
- prepare emits a concise next command

### Start

Start launches runtime services only. It should not hide setup mutations that belong in prepare.

Expected behavior:

- Docker single-user starts API + WebUI
- Docker multi-user starts API + Postgres, and optionally WebUI through the documented overlay
- Local single-user starts the API with plain `uvicorn`, without `--reload` in the public first-run path
- if startup fails, the command exits nonzero and points to the first useful diagnostic

### Verify

Verify waits for readiness and validates the profile-specific auth path.

Expected behavior:

- health/ready endpoint succeeds
- docs endpoint is reachable
- quickstart redirect is reachable
- single-user verifies an authenticated request with `X-API-KEY`
- multi-user verifies admin creation/login and a bearer-token authenticated request
- diagnostics distinguish "server not running", "database unavailable", "auth bootstrap failed", "provider missing", and "port in use"
- the verifier uses one documented endpoint contract per profile instead of mixing `/health`, `/ready`, and `/api/v1/healthz` assumptions in different places

### First Value

First value should be explicit rather than implied.

Expected behavior:

- first ingest/search uses a deterministic repo-local sample file
- first chat readiness checks provider configuration
- if provider credentials are absent, the result is `provider_missing` with exact env examples, not a generic chat failure
- the docs show the shortest successful path for each auth mode

### Audio

Audio setup stays in scope but is reported separately from core readiness.

Expected behavior:

- docs show `X-API-KEY` examples for single-user
- docs show `Authorization: Bearer` examples for multi-user
- `/setup`, CLI guidance, and CPU/GPU audio docs agree on Docker restart versus rebuild behavior
- Docker docs clearly distinguish stock CPU/default paths from GPU or host-customized audio paths

## Existing CLI Wizard Integration

The existing console script is:

```text
tldw-setup = tldw_Server_API.cli.wizard.cli:main
```

The remediation should use this as the shared operator-facing setup tool.

Existing commands already provide a useful base:

- `tldw-setup doctor`
- `tldw-setup init`
- `tldw-setup auth`
- `tldw-setup db`
- `tldw-setup providers`
- `tldw-setup verify`

Design changes:

- Remove or replace scaffold language from user-facing output once behavior is real.
- Add profile-aware modes if needed, for example `--profile docker-single-webui`, `--profile docker-multi-postgres`, and `--profile local-single`.
- Make `doctor`, `init`, `auth`, `db`, and `verify` agree on env file location for repo-local setup (`tldw_Server_API/Config_Files/.env`) versus package-local `.env`.
- For public repo-checkout onboarding, `tldw_Server_API/Config_Files/.env` is the default env file. A cwd `.env` is allowed only when explicitly supplied with `--env-file` or when running in a package-local/advanced layout.
- Extend `verify` so it can validate running Docker profiles without trying to spawn an unrelated ephemeral local server.
- Extend `verify` so first auth, first ingest/search, and provider/chat readiness are explicit checks.
- Keep JSON output for tests and automation.

## Command Contract

Public docs may move to clearer command names. Existing quickstart names can remain as compatibility wrappers if they do not preserve confusing behavior.

Recommended Makefile direction:

- `make setup-docker-single`
  - prepare Docker single-user + WebUI
- `make start-docker-single`
  - start Docker single-user + WebUI
- `make verify-docker-single`
  - verify health, WebUI reachability, single-user auth, ingest/search, provider/chat readiness
- `make setup-docker-multi`
  - prepare Docker multi-user + Postgres
- `make start-docker-multi`
  - start Docker multi-user + Postgres
- `make verify-docker-multi`
  - verify health, first admin/login, bearer auth, ingest/search, provider/chat readiness
- `make install-local`
  - create `.venv` and install dependencies only
- `make setup-local-single`
  - prepare local single-user env/auth/db
- `make start-local-single`
  - start local API
- `make verify-local-single`
  - verify health, docs, single-user auth, ingest/search, provider/chat readiness

Compatibility aliases:

- `make quickstart` may call the Docker single-user profile sequence.
- `make quickstart-docker-webui` may call the Docker single-user profile sequence.
- `make quickstart-docker` may call the Docker single-user API-only subset if still supported.
- `make quickstart-install` must become install-only or fail early with replacement guidance. It must not start runtime services.
- No compatibility alias may print success before the matching profile verifier passes.
- If an old alias no longer maps cleanly to a supported profile, it should fail early with a short message pointing to the replacement commands.
- Default output should not print full secrets. If an API key or generated password is needed after setup, provide an explicit reveal command or masked output with recovery instructions.

## Docker Remediation

### Docker single-user + WebUI

Fixes required:

- Ensure app data and per-user database paths are writable by `appuser`.
- Resolve the named-volume overlap between `/app/Databases` and `/app/Databases/user_databases`.
- Make the default stack not depend on a broken Postgres container for single-user SQLite auth.
- Add startup wait/verify behavior so the command does not print success before readiness.
- Keep same-origin WebUI proxy behavior as the default browser path.

### Docker multi-user + Postgres

Fixes required:

- Fix the PostgreSQL 18 volume layout by mounting named storage at the path expected by the image, or pin to a Postgres version whose data directory matches the existing mount contract.
- Ensure `DATABASE_URL=postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users` resolves from the app container and points to the bundled service.
- Include `SESSION_ENCRYPTION_KEY` in the profile setup path.
- Use one public first-admin creation mechanism: env-driven `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and optional `ADMIN_EMAIL`, verified by `tldw-setup verify`.
- Keep manual admin creation only as an advanced or recovery path. It must not be required for the documented first-run sequence.
- Make auth bootstrap idempotent and make failure messages point to DB readiness, secrets, or admin bootstrap specifically.

## Local Remediation

Fixes required:

- Make install-only behavior real.
- Use plain `uvicorn` for the public first-run local path, not `--reload`.
- Keep `--reload` as a developer convenience, documented separately.
- Make local env creation deterministic and compatible with AuthNZ single-user invariants.
- Add a verification path that does not invent or hide server state.
- Ensure setup/docs explain how to stop the local server.

## Documentation Remediation

Update the public onboarding surface:

- `README.md`
- `Docs/Getting_Started/README.md`
- `Docs/Getting_Started/Profile_Docker_Single_User.md`
- `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- `Docs/Getting_Started/Profile_Local_Single_User.md`
- `Docs/Deployment/setup-wizard-guide.md`
- `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- `Dockerfiles/README.md`
- `Docs/Getting_Started/QUICKSTART.md`, if it remains a public entry point
- `Docs/Website/index.html`, if it mirrors public quickstart commands

Docs should show all three public profiles as peers, with the same lifecycle headings and the same success bar.
Docs should also include Windows/WSL notes anywhere a command sequence assumes POSIX shell behavior.

## Implementation Staging

Keep the umbrella remediation as one plan, but stage implementation so each profile can be fixed and verified independently.

Recommended stages:

1. CLI and command contract: profile flags, env-file resolution, verifier modes, compatibility alias behavior, and tests.
2. Docker single-user + WebUI: compose path, volume writability, startup readiness, and profile verification.
3. Docker multi-user + Postgres: Postgres volume path, required secrets, env-driven first admin, and bearer-token verification.
4. Local single-user: install-only behavior, deterministic setup, plain `uvicorn` startup, and local verifier.
5. Documentation and runtime validation: peer profile docs, audio auth examples, cross-platform notes, and clean-state validation logs.

## Testing Strategy

Add or update tests before implementation where practical.

Unit and doc tests:

- Makefile command-boundary tests:
  - install target is install-only
  - local start target does not use `--reload`
  - public profile targets call the expected setup/start/verify helpers
  - compatibility aliases either route through the new verifier-backed profile flow or fail with replacement guidance
  - default setup output does not print full generated secrets
- CLI wizard tests:
  - profile-aware prepare writes required env keys
  - multi-user requires or generates `SESSION_ENCRYPTION_KEY`
  - Docker verify mode does not spawn an ephemeral local server
  - verify reports provider-missing distinctly from endpoint failure
  - repo-checkout onboarding defaults to `tldw_Server_API/Config_Files/.env`
  - explicit `--env-file` overrides default env discovery
- Docker contract tests:
  - default single-user compose does not require Postgres health
  - Postgres volume path matches the selected Postgres image
  - app/user database volumes do not overlap in a way that makes `user_databases` unwritable
  - multi-user compose exposes or documents the selected WebUI overlay behavior without making optional WebUI availability block API-first verification
- Documentation tests:
  - all three profile docs use the same lifecycle headings
  - audio docs include both `X-API-KEY` and bearer-token examples
  - multi-user docs include `SESSION_ENCRYPTION_KEY`
  - setup/restart/rebuild guidance is consistent across Docker profile and audio docs
  - Windows/WSL notes are present for POSIX-dependent command sequences

Runtime validation:

- Execute Docker single-user + WebUI from clean volumes.
- Execute Docker multi-user + Postgres from clean volumes.
- Execute local single-user from a clean venv/env.
- For each profile, verify health/docs/auth, sample ingest/search, provider-missing or first chat, and audio guidance endpoint shape.
- Capture the validated command transcript or log summary so future regressions can be compared against the public flow.

## Risks

- Docker image builds are heavy and slow, especially with ML dependencies.
- Multi-user admin creation may expose deeper AuthNZ assumptions beyond onboarding docs.
- Provider/chat success depends on credentials; tests should support a deterministic provider-missing result and optional live-provider validation.
- Fixing Docker volumes may require a migration note for existing named volumes.
- The CLI wizard currently mixes repo-local and cwd-local env behavior; making it the public contract may require careful path handling.
- Windows support is easiest to overclaim. The public contract should explicitly support WSL for shell-heavy flows unless native PowerShell commands are also validated.

## Design Decisions For Planning

- Use the Make target names listed in the command contract section.
- Keep compatibility aliases where they do not preserve misleading behavior.
- Make Docker single-user use a profile-specific compose path that does not start or depend on Postgres.
- Keep `postgres:18-bookworm` only if the compose volume is moved to `/var/lib/postgresql`; otherwise pin to a version compatible with `/var/lib/postgresql/data`. The preferred implementation is to keep the modern image and fix the mount path.
- Use env-driven first-admin bootstrap for the public Docker multi-user path. Manual admin creation remains recovery/advanced documentation only.
- Add profile options to existing `tldw-setup` commands instead of adding a new top-level `profile` subcommand. Example: `tldw-setup init --profile docker-single-webui`.
- Put first-value checks behind `tldw-setup verify --first-value` so the default verifier remains usable for quick health checks while public profile docs can use the stricter onboarding verifier.
- Make `tldw-setup` resolve the repo root and default to `tldw_Server_API/Config_Files/.env` for repo checkout onboarding. A separate `--env-file` option can support package-local or advanced layouts.
- Normalize the documented readiness endpoints used by `tldw-setup verify` so implementation, tests, and docs do not drift across endpoint names.
- Do not print full generated secrets in default setup output. Use masked output or an explicit reveal command for intentional secret display.
- Treat Docker multi-user as API-first with bundled Postgres. WebUI may be documented as an overlay, but the multi-user verifier should not fail solely because an optional WebUI overlay was not selected.

## Acceptance Criteria

- All three public profile docs present peer setup paths with the same lifecycle.
- Each profile has documented prepare, start, verify, first-value, and audio guidance.
- `quickstart-install` no longer starts runtime behavior. It either performs install-only work or exits early with replacement guidance.
- No public compatibility alias reports success before its profile verifier passes.
- Default setup output does not expose full generated secrets.
- Docker single-user reaches API and WebUI readiness from clean state.
- Docker multi-user reaches Postgres readiness, first admin/login, and bearer-auth readiness from clean state.
- Local single-user reaches API readiness from clean venv/env using the documented command sequence.
- First ingest/search is verified for each profile.
- Provider-missing state is clear when no live LLM credentials are supplied.
- Audio docs include both single-user and multi-user auth examples.
- Public docs include Windows/WSL guidance for commands that assume POSIX shell behavior.
- Tests cover command boundaries, CLI wizard profile behavior, Docker compose contract, and documentation consistency.
