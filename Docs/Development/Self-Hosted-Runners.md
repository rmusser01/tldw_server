# Self-hosted runners (linux / macOS / Windows)

Run the **OS full-suite coverage on your own boxes** — free cross-OS signal with no
GitHub macOS/Windows minutes. This pairs with PR #2258, which already moves the
OS full runs to `release` / push-to-`main` / `workflow_dispatch` (off PRs).

> ⚠️ **Security — read first. `rmusser01/tldw_server` is a public repo.**
> A self-hosted runner executes whatever code the job checks out. If a job that
> runs on a self-hosted runner is ever triggered by an untrusted **fork pull
> request**, that PR gets arbitrary code execution on your machine and network.
> **Therefore: only ever target self-hosted runners at trusted events** —
> `push` to `main`, `release: published`, and `workflow_dispatch`. Keep all
> `pull_request` validation on GitHub-hosted runners.
> PR #2258's OS full shards are already guarded with
> `if: github.event_name != 'pull_request'`, so they are the *only* jobs that
> should opt into self-hosted execution.

## 1. Register a runner on each box

Repo → **Settings → Actions → Runners → New self-hosted runner**. Pick the OS, then
run the shown `config`/`run` commands on the box. Use clear labels:

| Box | Suggested labels |
|---|---|
| Linux | `self-hosted, linux, x64` |
| macOS (Apple Silicon) | `self-hosted, macOS, arm64` |
| Windows Server | `self-hosted, windows, x64` |

Prefer a **runner group** restricted to this repo, and consider
`--ephemeral` runners (one job per registration) so state never leaks between runs.
In **Settings → Actions → General**, set "Require approval for all outside
collaborators" so first-time contributors' workflows never auto-run.

## 2. Toggle `runs-on` without exposing PRs

Add a repo **variable** `OS_RUNNER_MODE` (Settings → Actions → Variables). Leave it
unset/`hosted` normally; set it to `self-hosted` when you want the OS shards to land
on your boxes. The OS full-suite jobs (already non-PR) select the runner by mode:

```yaml
  full-suite-macos-312:
    if: github.event_name != 'pull_request'   # never untrusted PRs
    runs-on: >-
      ${{ vars.OS_RUNNER_MODE == 'self-hosted'
          && fromJSON('["self-hosted","macOS","arm64"]')
          || 'macos-14' }}
```

```yaml
  full-suite-windows-312:
    if: github.event_name != 'pull_request'
    runs-on: >-
      ${{ vars.OS_RUNNER_MODE == 'self-hosted'
          && fromJSON('["self-hosted","windows","x64"]')
          || 'windows-latest' }}
```

`runs-on` accepts either a string (hosted label) or an array (self-hosted label
set); the ternary yields one or the other. Flipping the variable needs no code change.

## 3. Provision each box

The jobs assume these are present (GitHub-hosted images bundle them; your boxes need
them installed once):

- **ffmpeg** on PATH (audio/video tests).
- **Docker** — for the `services:` Postgres/Redis containers. On macOS/Windows
  self-hosted runners, service containers are **not** supported the same way as on
  Linux; instead run Postgres/Redis on the host and export
  `TEST_DATABASE_URL=postgresql://…` (and `REDIS_URL`) in the runner environment, or
  set `TLDW_TEST_NO_DOCKER=1` for SQLite-only lanes. See
  `Helper_Scripts/Testing-related/start_postgres_for_tests.sh`.
- **uv** + Python 3.12 and 3.13 (`uv python install 3.12 3.13`).
- **Node 20 + Bun 1.3.2** if the box also runs frontend lanes.
- Git, and enough disk for model/HF caches (set `HF_HOME` to a persistent path to
  avoid re-downloads between runs).

## 4. Validate

1. Set `OS_RUNNER_MODE=self-hosted`.
2. Trigger the workflow via **Run workflow** (`workflow_dispatch`) on `main`.
3. Confirm the macOS/Windows full-suite jobs pick up on your boxes (runner name shows
   in the job log) and that a normal fork PR still uses GitHub-hosted runners.

## Alternative: skip GitHub entirely

If you only want local validation (not GitHub orchestration), just run the suite
natively on each box with [Local-CI.md](./Local-CI.md):
`make ci-local-full`. Self-hosted runners are the right tool when you want the
*scheduled / release* OS coverage to run unattended on your hardware with the normal
required-check integration.
