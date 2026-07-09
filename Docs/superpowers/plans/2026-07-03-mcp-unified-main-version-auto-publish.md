# MCP Unified Main Version Auto Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `mcp-unified` to PyPI automatically when a merged `main` commit changes the standalone package version.

**Architecture:** Extend the existing MCP Unified publish workflow instead of adding a second workflow. A version-detection job compares the previous and current `apps/mcp-unified/pyproject.toml` versions, verifies `mcp_unified.__version__` is in sync, checks that the target version is not already on PyPI, then lets the existing RC and trusted-publishing path run for valid `main` version bumps.

**Tech Stack:** GitHub Actions, Python 3.12, `tomllib`, stdlib `urllib`, existing `make mcp-unified-rc` and `make mcp-unified-publish-dry-run` targets, PyPI trusted publishing.

---

### Task 1: Add guarded main-push release detection

**Files:**
- Modify: `.github/workflows/mcp-unified-publish.yml`

- [ ] **Step 1: Add push trigger**

Add a `push` trigger for `main` scoped to release-relevant files:

```yaml
  push:
    branches:
      - main
    paths:
      - apps/mcp-unified/pyproject.toml
      - apps/mcp-unified/src/mcp_unified/__init__.py
      - .github/workflows/mcp-unified-publish.yml
```

- [ ] **Step 2: Add `detect-version-change` job**

Create a job that checks out with `fetch-depth: 0`, reads the current package version, reads the prior version from `${{ github.event.before }}` for push events, and writes outputs:

```text
version_changed=true|false
old_version=<old or empty>
new_version=<current>
publish_candidate=true|false
```

The job must:
- skip publishing for manual dispatch unless manual inputs request it
- fail if `apps/mcp-unified/src/mcp_unified/__init__.py` does not contain the same `__version__`
- fail early if the current version already exists on PyPI
- treat deleted/missing previous files as `version_changed=true` only for push events

- [ ] **Step 3: Run YAML/lint sanity checks**

Run:

```bash
python - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/mcp-unified-publish.yml').read_text())
PY
```

Expected: command exits successfully.

### Task 2: Wire existing RC and PyPI publish jobs to the detector

**Files:**
- Modify: `.github/workflows/mcp-unified-publish.yml`

- [ ] **Step 1: Gate `publish-plan`**

Make `publish-plan` depend on `detect-version-change` and run when either:
- the workflow is manually dispatched, or
- a `main` push produced `publish_candidate=true`

- [ ] **Step 2: Gate `publish-pypi`**

Keep the existing manual PyPI condition and add the automatic condition:

```text
push to main AND publish_candidate=true
```

Keep `environment: pypi` and `id-token: write`.

- [ ] **Step 3: Add final duplicate guard before upload**

Before `pypa/gh-action-pypi-publish`, add a Python stdlib check against:

```text
https://pypi.org/pypi/mcp-unified/<version>/json
```

Expected:
- `404` means safe to publish
- `200` fails with a clear duplicate-version message
- other/network errors fail closed

### Task 3: Update release docs and task notes

**Files:**
- Modify: `apps/mcp-unified/README.md`
- Modify: `apps/mcp-unified/src/mcp_unified/README.md`
- Modify: `backlog/tasks/task-12118 - Prepare-MCP-Unified-0.2.0-package-release.md`

- [ ] **Step 1: Update README release text**

Replace the manual-only PyPI language with:

```text
Merging a package version bump to main triggers the guarded PyPI publishing workflow.
Manual TestPyPI and PyPI workflow dispatch remain available for release rehearsals and operator-driven publishes.
```

- [ ] **Step 2: Update Backlog task notes**

Record that the release PR now includes guarded main-version auto-publish behavior.

### Task 4: Verify and commit

**Files:**
- `.github/workflows/mcp-unified-publish.yml`
- `apps/mcp-unified/README.md`
- `apps/mcp-unified/src/mcp_unified/README.md`
- `backlog/tasks/task-12118 - Prepare-MCP-Unified-0.2.0-package-release.md`
- Docs/superpowers/plans/2026-07-03-mcp-unified-main-version-auto-publish.md

- [ ] **Step 1: Run package validation**

Run:

```bash
make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-publish-dry-run
```

Expected: `RC status: ok`.

- [ ] **Step 2: Run targeted workflow syntax validation**

Run the YAML parse check from Task 1.

Expected: command exits successfully.

- [ ] **Step 3: Run Bandit on touched Python if any**

If only YAML/Markdown changes are made, record Bandit as skipped because no Python source changed.

- [ ] **Step 4: Commit**

Run:

```bash
git add .github/workflows/mcp-unified-publish.yml apps/mcp-unified/README.md apps/mcp-unified/src/mcp_unified/README.md "backlog/tasks/task-12118 - Prepare-MCP-Unified-0.2.0-package-release.md" docs/superpowers/plans/2026-07-03-mcp-unified-main-version-auto-publish.md
git commit -m "Auto publish MCP Unified version bumps"
```
