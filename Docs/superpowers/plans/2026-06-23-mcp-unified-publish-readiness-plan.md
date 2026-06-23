# MCP Unified Publish Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the standalone `mcp-unified` package for an opt-in TestPyPI/publish dry run without allowing normal PR CI to publish artifacts.

**Architecture:** Extend the existing internal RC harness instead of adding a separate release script. Keep publishing disabled by default through explicit CLI flags, workflow dispatch inputs, and package metadata that still reports `internal-experimental` / `not-published`.

**Tech Stack:** Python 3.10+, setuptools/build/twine, pytest, GitHub Actions, Make, existing `Helper_Scripts/mcp_unified_rc.py` evidence model.

---

## File Structure

- Modify: `apps/mcp-unified/pyproject.toml`
  - Add publish-ready metadata: authors, maintainers, keywords, classifiers, URLs, and license-file declaration.
- Create: `apps/mcp-unified/LICENSE`
  - Package-local license copy so built standalone artifacts include the license from the nested project root.
- Modify: `apps/mcp-unified/src/mcp_unified/package_metadata.py`
  - Add metadata constants that mirror the new pyproject publish-readiness fields.
- Modify: `Helper_Scripts/mcp_unified_rc.py`
  - Add guarded publish-plan/TestPyPI dry-run command construction and evidence recording without live upload by default.
- Modify: `Makefile`
  - Add a dry-run publishing target that delegates to the RC harness.
- Create: `.github/workflows/mcp-unified-publish.yml`
  - Manual-dispatch-only workflow for dry-run/TestPyPI/PyPI targets with environment gates and no PR trigger.
- Modify: `apps/mcp-unified/README.md`
- Modify: `apps/mcp-unified/USER_GUIDE.md`
- Modify: `apps/mcp-unified/src/mcp_unified/README.md`
- Modify: `apps/mcp-unified/src/mcp_unified/USER_GUIDE.md`
  - Document internal RC, publish dry run, TestPyPI prerequisites, and real publish guardrails.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
  - Add metadata, license, workflow, and Make target assertions.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py`
  - Add RC harness publish-plan command/redaction tests.

## Task 1: Publish-Ready Package Metadata

**Files:**
- Modify: `apps/mcp-unified/pyproject.toml`
- Create: `apps/mcp-unified/LICENSE`
- Modify: `apps/mcp-unified/src/mcp_unified/package_metadata.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing metadata tests**

Add tests asserting:
- pyproject has `authors`, `maintainers`, `keywords`, `classifiers`, `[project.urls]`, and `license-files = ["LICENSE"]`;
- classifiers include Python 3.10-3.13, FastAPI, OS Independent, and internal/alpha development status;
- project URLs point at the repository, issues, docs/user guide path, and source package path;
- package metadata summary mirrors those publish-readiness fields;
- `apps/mcp-unified/LICENSE` exists and matches the root license text.

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_publish_metadata_is_ready_but_internal \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_package_license_file_is_local_to_project \
  -v
```

Expected: fail because metadata/license fields are missing.

- [x] **Step 2: Add minimal metadata and license file**

Update `apps/mcp-unified/pyproject.toml` with:
- `authors = [{ name = "Robert Musser", email = "contact@tldwproject.com" }]`
- matching `maintainers`
- concise package keywords
- PyPI classifiers
- `license-files = ["LICENSE"]`
- `[project.urls]` entries for repository, issues, docs, user guide, and source path.

Copy the root `LICENSE` content into `apps/mcp-unified/LICENSE`.

- [x] **Step 3: Mirror metadata constants**

Add package metadata constants for authors, maintainers, keywords, classifiers, URLs, and license files, then include them in `package_metadata_summary()`.

- [x] **Step 4: Verify metadata tests pass**

Run the focused pytest command from Step 1.

## Task 2: Guarded Publish/TestPyPI Dry-Run Tooling

**Files:**
- Modify: `Helper_Scripts/mcp_unified_rc.py`
- Modify: `Makefile`
- Create: `.github/workflows/mcp-unified-publish.yml`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing guardrail tests**

Add tests asserting:
- RC parser exposes `publish-plan`;
- default publish plan is dry-run only and never uploads;
- `--target testpypi` builds a redacted `twine upload --repository-url https://test.pypi.org/legacy/ ...` plan;
- `--execute` is rejected unless `MCP_UNIFIED_ALLOW_PUBLISH=1`;
- Make target calls `publish-plan --target testpypi --dry-run`;
- workflow is `workflow_dispatch` only, has no `pull_request` or `push` trigger, uses pinned actions, has `contents: read`, and keeps upload jobs behind explicit confirmation plus environment-scoped token secrets.

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py::test_rc_publish_plan_is_dry_run_by_default \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_publish_workflow_is_manual_and_gated \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_make_targets_do_not_call_root_pypi_check \
  -v
```

Expected: fail because the command, workflow, and target do not exist.

- [x] **Step 2: Implement publish-plan command**

Add `publish-plan` to `Helper_Scripts/mcp_unified_rc.py`. It should:
- record existing artifacts and require a built wheel/sdist;
- accept `--target testpypi|pypi`, `--dry-run`, and `--execute`;
- default to dry-run;
- reject `--execute` unless `MCP_UNIFIED_ALLOW_PUBLISH=1`;
- construct but not log secrets;
- record the target, repository URL, artifact filenames, command shape, and execution mode in RC evidence.

- [x] **Step 3: Add Make and workflow guardrails**

Add:

```make
mcp-unified-publish-dry-run:
	$(MCP_UNIFIED_RC) build
	$(MCP_UNIFIED_RC) publish-plan --target testpypi --dry-run
```

Create `.github/workflows/mcp-unified-publish.yml` as manual-dispatch-only with targets `dry-run`, `testpypi`, and `pypi`. The dry-run job should run RC + `publish-plan --dry-run`. The upload jobs should be environment-gated and require explicit dispatch input, but this implementation may keep actual upload steps as guarded command-plan placeholders until credentials/trusted publishing are explicitly configured.

- [x] **Step 4: Verify guardrail tests pass**

Run the focused pytest command from Step 1.

## Task 3: Publish-Readiness Documentation

**Files:**
- Modify: `apps/mcp-unified/README.md`
- Modify: `apps/mcp-unified/USER_GUIDE.md`
- Modify: `apps/mcp-unified/src/mcp_unified/README.md`
- Modify: `apps/mcp-unified/src/mcp_unified/USER_GUIDE.md`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing docs parity/readiness test**

Extend package docs tests to assert project docs and package-resource docs mention:
- internal RC;
- `make mcp-unified-rc`;
- `make mcp-unified-publish-dry-run`;
- TestPyPI is opt-in and requires maintainer-owned credentials/trusted publishing;
- public PyPI remains disabled while `PUBLISHING_STATUS` is `not-published`.

Expected: fail until docs are updated and package-resource copies match.

- [x] **Step 2: Update docs and package resource copies**

Add a short “Publishing Readiness” section to README and USER_GUIDE. Keep the language explicit that `pip install mcp-unified` is not yet public availability guidance.

- [x] **Step 3: Verify docs tests pass**

Run the focused docs test.

## Task 4: Final Validation And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-2400 - Prepare-MCP-Unified-standalone-publish-readiness-and-TestPyPI-gate.md`

- [x] **Step 1: Run focused tests**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -v
```

- [x] **Step 2: Run static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  Helper_Scripts/mcp_unified_rc.py \
  apps/mcp-unified/src/mcp_unified/package_metadata.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall \
  Helper_Scripts/mcp_unified_rc.py \
  apps/mcp-unified/src/mcp_unified/package_metadata.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py

/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  Helper_Scripts/mcp_unified_rc.py \
  apps/mcp-unified/src/mcp_unified/package_metadata.py \
  -f json -o /tmp/bandit_mcp_unified_publish_readiness.json
```

- [x] **Step 3: Run RC and dry-run publish plan**

```bash
make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-rc
make PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python mcp-unified-publish-dry-run
```

- [ ] **Step 4: Finalize task and commit**

Update `TASK-2400` with validation evidence, then commit the branch.
