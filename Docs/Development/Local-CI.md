# Local CI — run the gating checks before you push

`Helper_Scripts/ci/run_local_ci.py` reproduces the **blocking** lanes of
`.github/workflows/ci.yml` on your machine so you can get the same signal locally
and avoid waiting on (or burning minutes in) the remote GitHub runners. It avoids
shell-specific behavior and works the same on Linux, macOS, and Windows Server.

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
`--pytest-args "<args>"` (parsed with platform shell quoting rules), `--mypy`
(run mypy too, non-blocking), `--no-pytest`, `--list-changed`.

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

## Outbound privacy checks

Run from the repository root with the project virtual environment active and a
Docker engine that runs Linux containers. On macOS, use a Linux container engine
or a Linux VM with Docker. Build preparation needs access to the configured
dependency and image registries. The smoke runs with loopback networking and a
private filesystem inside each prebuilt container.

Preparation reads the committed `tldw_Server_API/Config_Files/config.txt`
defaults. A local edit to this file stops preparation before Docker can cache
it. Test deliberate changes to the defaults from a commit that contains no
credentials.

```bash
outbound_report_dir="$(mktemp -d)"
python -m Helper_Scripts.ci.outbound_isolation prepare all --report-dir "$outbound_report_dir"
python -m Helper_Scripts.ci.outbound_check all --report-dir "$outbound_report_dir"
```

Use `backend` or `browser` in both commands to run a slice. Reprepare that
slice after changing source. `backend` verifies the reviewed inventory, fresh
ASGI startup through shutdown, a first-run setup edit, a local note, and opted-in
fixture dispatch through the Hugging Face SDK and OpenAI adapter. Installer
package and Git checks verify selected command arguments. `browser` checks the
shipped setup wizard, standalone production WebUI, and packaged Chrome worker and
content script.

In GitHub Actions each slice runs on a fresh Linux runner after license
admission. `backend-required` uses the normal backend flag. `frontend-required`
uses the browser privacy flag for changes to the API, app defaults, browser
build inputs, or check code. Each required job checks the result of its privacy
job. The preparation step records elapsed time and the lowest sampled free
Docker disk space in the run summary.

The native trace detected ONNX Runtime's startup telemetry. The app sets
`ORT_DISABLE_TELEMETRY=1` before importing the SDK unless the operator supplies
an explicit value. The destination is documented in the reviewed inventory.
The WebUI run also exercises the backend's model catalog against four named
loopback provider fixtures. Custom OpenAI endpoints that require credentials
skip DNS resolution while those credentials are missing.

Each run writes a fresh `outbound-run-*/report.json` with the terminal status,
coverage and redacted violation codes for each selected scenario. The CI run
summary includes the same decision codes and allowlisted test or source file
identifiers where available. Preparation, containment preflight and timeouts have
distinct codes. Missing tests, skips, incomplete evidence, and unexpected observed
attempts fail the run. Reviewed exceptions are checked against the exact Python
definitions at their source paths.
The reports cover the exercised flows, instrumented Python DNS and connect calls,
Playwright request events, and the browser process tree's traced Linux `connect`
and `sendto` calls. The extension profiles use Chromium's private startup
network log to detect WebSocket attempts before DNS. A separate isolated worker
launch calibrates this log with a denied hostname. Each application scenario
requires a complete native trace and evidence of a successful fixture connection.
Chromium's known IPv6 route query is classified separately only when a UDP
`connect` returns `ENETUNREACH`. The container still has no
external route. Package-index selection beyond explicit installer arguments,
real CDN and provider behavior, other native socket calls, unexercised features,
and other platforms and browsers are outside this coverage.

The pure checks also run without Docker:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -o addopts='' -q \
  tldw_Server_API/tests/CI/test_outbound_inventory.py \
  tldw_Server_API/tests/CI/test_outbound_backend_smoke.py \
  tldw_Server_API/tests/CI/test_outbound_check.py
```

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
