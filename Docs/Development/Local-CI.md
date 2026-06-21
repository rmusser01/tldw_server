# Local CI — run the gating checks before you push

`Helper_Scripts/ci/run_local_ci.py` reproduces the **blocking** lanes of
`.github/workflows/ci.yml` on your machine so you can get the same signal locally
and avoid waiting on (or burning minutes in) the remote GitHub runners. It is
pure standard library and works the same on Linux, macOS, and Windows Server.

## One-time setup (uv venv — never the system Python)

The project targets Python 3.11–3.13. Create a uv-managed venv and install dev deps:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
# align lint tools with the versions CI pins:
uv pip install --python .venv/bin/python "ruff==0.15.10" "mypy==1.20.1"
```

> The runner **auto-detects `.venv` and re-executes itself under it**, so even if
> you launch it with a system `python3`, the actual checks run against the venv
> interpreter. Set `TLDW_CI_NO_REEXEC=1` to opt out.

## Usage

| Command | What it runs |
|---|---|
| `make ci-local` | **Fast tier**: compileall + ruff(changed) + guards + pytest on changed test files |
| `make ci-local-full` | **Full tier**: compileall + ruff(all) + guards + whole suite under `-n auto` |
| `make ci-local-lane LANE=tldw_Server_API/tests/Security` | compileall + guards + pytest on one area |

Or call the script directly (Windows, or when you want flags):

```bash
.venv/bin/python Helper_Scripts/ci/run_local_ci.py --fast
.venv/bin/python Helper_Scripts/ci/run_local_ci.py --full --jobs 8
.venv/bin/python Helper_Scripts/ci/run_local_ci.py --lane tldw_Server_API/tests/RAG
.venv/bin/python Helper_Scripts/ci/run_local_ci.py --full --pytest-args "-k embeddings"
# Windows:
.venv\Scripts\python.exe Helper_Scripts\ci\run_local_ci.py --fast
```

Useful flags: `--base <ref>` (diff base for change detection; defaults to the
merge-base with `origin/dev`/`origin/main`), `--jobs auto|N|0` (xdist workers),
`--mypy` (run mypy too, non-blocking), `--no-pytest`, `--list-changed`.

## How phases map to CI jobs

| Local phase | CI job | Blocking? |
|---|---|---|
| `compileall (syntax-check)` | `syntax-check` (compileall over `app/`) | **Yes** |
| `repo guards` | pre-commit local hooks (http-client patch / legacy body / syntax) | **Yes** |
| `pytest` | `full-suite-*` shard jobs | **Yes** |
| `ruff (non-blocking)` | `lint` job (ruff) | No (baseline backlog) |
| `mypy (non-blocking)` | `lint` job (mypy) | No (baseline backlog) |

ruff/mypy are reported for visibility but never fail the local run, matching the
remote `lint` job's `continue-on-error`. The phases that actually gate a PR
(compileall, guards, pytest) do fail the local run.

## Postgres / Redis dependent tests

The runner honors the same environment the suite already uses:

- `TEST_DATABASE_URL=postgresql://…` — point at a running Postgres.
- `TLDW_TEST_NO_DOCKER=1` — skip the autostart (for pure-SQLite lanes).
- Spin up a throwaway Postgres: `bash Helper_Scripts/Testing-related/start_postgres_for_tests.sh`.

## Pre-push hook (optional)

Run the fast tier automatically on `git push`:

```bash
uv pip install --python .venv/bin/python pre-commit   # if not already present
pre-commit install --hook-type pre-push
```

The `local-ci-fast` hook in `.pre-commit-config.yaml` runs `--fast` and (via the
auto re-exec) uses `.venv`. Bypass a single push with `git push --no-verify`.

## Cross-OS strategy (your linux / macOS / windows boxes)

Run `make ci-local-full` (or the script) **natively on each box** to validate the
same suite per-OS without GitHub. This is the recommended way to cover
macOS/Windows-specific behavior locally. For unattended, scheduled cross-OS runs
on your hardware, see [Self-Hosted-Runners.md](./Self-Hosted-Runners.md).
