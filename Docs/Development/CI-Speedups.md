# CI full-suite: correctness + speed (sharding follow-ups)

This branch shards the full suite by hand-listed `matrix.shard[].paths`. Two
follow-ups below are **already wired** on this branch; a third (pytest-split) is
documented for adoption.

## B1 — Shard coverage guard (WIRED — job `shard-coverage`)

A test file in no shard is never collected, so the suite goes green while skipping
it. `Helper_Scripts/ci/check_shard_coverage.py` fails when a **newly** unshared
test appears.

- Permanent cross-workflow exclusions: `Helper_Scripts/ci/shard_coverage_ignore.txt`.
- Known backlog at introduction (771 files): `Helper_Scripts/ci/shard_coverage_baseline.txt`
  — the guard stays green on these; **shrink it over time** by assigning files to
  shards, then `python Helper_Scripts/ci/check_shard_coverage.py --write-baseline`.

Run locally: `python Helper_Scripts/ci/check_shard_coverage.py`.

## B3 — Path-filter gating (WIRED — job `changes`)

The `changes` job (reusing `.github/actions/detect-required-gate-changes`) gates the
PR-running full-suite jobs:
`if: github.event_name != 'pull_request' || backend_changed == 'true'`. So
docs/frontend-only PRs skip the backend suite; `main`/`release` never skip. The
`Full Suite (...)` summary checks treat a `skipped` shard as a pass, so branch
protection stays green.

## B2 — Replace the manual partition with `pytest-split` (TO ADOPT)

`pytest-split` (added to `[dev]`) splits the whole tests root into N timing-balanced
groups — **one** path root instead of 984 hand-maintained entries duplicated 5×, and
a file can't fall through the cracks (which makes B1's baseline shrink to zero).

**1. Durations file** (nightly job; commit it):
```bash
python -m pytest tldw_Server_API/tests --store-durations --durations-path .test_durations
```

**2. Reusable workflow** `.github/workflows/_full-suite.yml`:
```yaml
on:
  workflow_call:
    inputs:
      os:             { required: true, type: string }
      python-version: { required: true, type: string }
jobs:
  shard:
    runs-on: ${{ inputs.os }}
    timeout-minutes: 35
    strategy:
      fail-fast: false
      matrix:
        group: [1, 2, 3, 4, 5, 6, 7, 8]   # keep --splits == list length
    services:                              # carry over postgres/redis from the manual shards
      postgres: { image: mirror.gcr.io/library/postgres:18-bookworm, ... }
      redis:    { image: mirror.gcr.io/library/redis:8-alpine, ... }
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - uses: ./.github/actions/setup-python-deps
        with: { python-version: ${{ inputs.python-version }} }
      - run: |
          python -m pytest tldw_Server_API/tests \
            --splits 8 --group ${{ matrix.group }} --durations-path .test_durations \
            -n auto -q --disable-warnings -p no:cacheprovider
```

**3. Caller + preserved required-check name** (replaces a manual `*-shards` block;
keep the existing skip-tolerant summary so the check name and B3 gating still work):
```yaml
  full-suite-linux-312:
    needs: [lint, syntax-check, changes]
    if: ${{ github.event_name != 'pull_request' || needs.changes.outputs.backend_changed == 'true' }}
    uses: ./.github/workflows/_full-suite.yml
    with: { os: ubuntu-latest, python-version: '3.12' }
```

Each runner now provisions Postgres/Redis (any group may draw DB tests) — the normal
trade for a single complete partition. `-n auto` adds within-runner parallelism.

> Adoption is a **destructive cutover** of a 6.5k-line workflow whose balance depends
> on `.test_durations` and per-runner services; validate it with a real CI run
> (start with one variant, e.g. linux-3.12) before removing the others. Not done here
> for that reason.

See also: `Docs/Development/Local-CI.md`, `Docs/Development/Self-Hosted-Runners.md`
(local + cross-OS strategy).
