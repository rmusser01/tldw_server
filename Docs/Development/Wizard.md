# tldw_setup Wizard — Developer Guide

This document describes the setup wizard CLI skeleton, usage patterns, and troubleshooting tips.

## Overview

- Console entrypoint: `tldw-setup [command] [options]`
- Module fallback (repo/local): `python -m tldw_Server_API.cli.wizard.cli [command] [options]`
- Current state: scaffold implementation with safe, idempotent file operations and JSON output mode for automation.

## Commands

- `init` — full guided setup (scaffold)
  - Options: `--default`, `--install-dir PATH`, `--non-interactive`, `--debug`, `--dry-run`, `--json`, `--yes/--no-input`, `--no-format`, `--no-color`, `--quiet`
  - Behavior (scaffold): detects environment, plans `.env` creation and `.gitignore` updates. In non-dry-run mode creates `.env` (0600) with basic defaults and ensures `.gitignore` entries. When `AUTH_MODE=multi_user` and a valid Postgres `DATABASE_URL` is present, `--yes` auto-runs the AuthNZ initializer; otherwise the wizard prompts in interactive shells.

- `auth` — configure authentication mode
  - Options: `--mode [single_user|multi_user]`, `--json`, `--dry-run`, `--yes/--no-input`
  - Behavior: updates `.env` with `AUTH_MODE`, generates `SINGLE_USER_API_KEY` when needed, validates `DATABASE_URL` for multi-user, and prompts to run the AuthNZ initializer when appropriate. Creates a timestamped backup on first modification.

- `db` — initialize/validate databases
  - Options: `--json`, `--dry-run`
  - Behavior: creates per-user SQLite files for Media/ChaChaNotes/Evaluations, ensures a shared evaluations DB under `Databases/`, and validates Postgres connectivity when `DATABASE_URL` uses a Postgres scheme.

- `providers` — collect/store provider keys
  - Options: `--json`, `--dry-run`, `--check-provider`, `--write-config`
  - Behavior: reads provider keys from environment variables, writes them into `.env` (masked in output), and optionally updates `config.txt` when `--write-config` is set. Provider checks remain offline/mock and only run when `--check-provider` or `TLDW_CHECK_PROVIDER=1` is set.

- `mcp [add|remove]` — manage MCP client configs
  - Options: `--json`, `--dry-run`, `--client`, `--config-path`, `--server-url`, `--api-key`, `--api-key-env`, `--verify`, `--yes/--no-input`
  - Behavior: updates per-client JSON settings for Cursor/Claude/VS Code/Zed with a `mcpServers` entry, creating timestamped backups and providing unified diffs in dry-run mode. Removal prompts for confirmation unless `--yes` is provided. Override detection with `--config-path` (single client) and set `TLDW_MCP_URL` or `--server-url` to customize the target endpoint. Without a credential the config is written with a placeholder and reported as configured but not ready.

Examples:

```bash
python -m tldw_Server_API.cli.wizard mcp add --client cursor --api-key "$SINGLE_USER_API_KEY" --verify
python -m tldw_Server_API.cli.wizard mcp add --client cursor --api-key-env SINGLE_USER_API_KEY --dry-run
```

- `verify` — run verification checks (scaffold)
  - Options: `--json`, `--check-provider`, `--dry-run`
  - Behavior: detects `ffmpeg`/CUDA, probes `/api/v1/health`, `/api/v1/healthz`, and `/api/v1/mcp/status`, and can spin up an ephemeral server on a free port (or `TLDW_SERVER_PORT`) when no server is running.

- `format` — format changed files with Black/Ruff when available
  - Options: `--json`, `--dry-run`
  - Behavior: if in a Git repo, formats changed/untracked files. Skips gracefully if tools missing.

- `doctor` — detect issues and propose fixes (scaffold)
  - Options: `--json`, `--dry-run`, `--yes/--no-input`
  - Behavior: checks `.env` and `.gitignore` basics, validates `DATABASE_URL` in multi-user mode, detects port conflicts, and flags missing `ffmpeg`. Uses `--yes` to apply recommended fixes.

## Non-Interactive Usage

When running with `--non-interactive` or `--yes`, the wizard consumes environment variables instead of prompting. Common variables:

- `AUTH_MODE`, `SINGLE_USER_API_KEY`, `DATABASE_URL`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (and other providers)
- `TLDW_CHECK_PROVIDER=1` to enable provider verification
- `TLDW_SERVER_PORT` to control the port used by ephemeral server verification

Example (CI):

```bash
AUTH_MODE=single_user \
OPENAI_API_KEY=sk-... \
tldw-setup init --non-interactive --yes --json --no-format
```

## Output Modes

- Human-readable (default) — concise status, actions, and notes.
- JSON (`--json`) — machine-readable, stable schema per command (`command`, `status`, `facts`, `actions`, `paths`, etc.).

## JSON Output Schema (Stable Envelope)

All wizard commands emit a common JSON envelope when `--json` is provided:

```json
{
  "command": "init|auth|db|providers|mcp|verify|format|doctor",
  "status": "ok|error",
  "actions": [],
  "facts": {},
  "notes": [],
  "paths": {},
  "dry_run": false
}
```

Command-specific additions:
- `init`/`auth`: include `mode`, `paths.env`, and `actions` entries like `set_env`, `validate_database_url`, `authnz_initializer`.
- `db`: includes `validate_database_url`, `authnz_db`, `postgres_check`, and `sqlite_files` actions.
- `verify`: includes `facts.server_mode`, `actions.server`, and `actions.endpoints` for each probed path.
- `providers`: includes `actions.providers`, `set_env`, optional `config_txt`, and optional `provider_checks`.
- `mcp`: includes `actions.mcp_client` entries with `path`, `status`, `readiness`, `credential_source`, and optional `diff`, `backup`, or `verification`.
- `doctor`: includes `actions.env`, `set_env`, `gitignore`, `ffmpeg`, and optional `validate_database_url` or `port` actions.

Error responses keep the same envelope and set `status=error` plus an error-relevant action (for example `validate_database_url`).

## Files Managed

- `.env`: created with mode 0600; idempotent merge updates with de-duplication and timestamped backups.
- `.gitignore`: ensures `.env`, `.env.local`, and `wizard.log` entries.
- `wizard.log`: optional local troubleshooting log (future step); must be redacted and gitignored.

## Troubleshooting

- `ffmpeg` not detected
  - Install via package manager: `brew install ffmpeg` (macOS), `apt-get install -y ffmpeg` (Debian/Ubuntu), `choco install ffmpeg` or `scoop install ffmpeg` (Windows).

- CUDA check noisy on Apple Silicon or CPU-only systems
  - Pass `--non-interactive` without GPU flags; the wizard only performs passive detection in scaffold.

- Formatting didn’t run
  - Ensure you’re in a Git repo and Black/Ruff are installed (`pip install black ruff`). Use `--no-format` to skip explicitly.

- Multi-user AuthNZ initializer
  - The full implementation will offer to run: `python -m tldw_Server_API.app.core.AuthNZ.initialize`.

## Roadmap Notes

- CI/DX polish: coverage targets and README quickstart link.
