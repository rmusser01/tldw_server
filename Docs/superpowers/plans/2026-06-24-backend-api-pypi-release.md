# Backend API PyPI Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the root `tldw-server` PyPI release path as a backend/API-only distribution while keeping the Next.js WebUI on separate container/release-artifact channels.

**Architecture:** Keep setuptools discovery constrained to Python backend packages and add an artifact-content guard that validates built wheel/sdist contents. Reuse the existing manual Trusted Publishing workflow, but align its action pins and PR trigger paths with the hardened MCP publishing style.

**Tech Stack:** Python packaging with setuptools/build/twine, GitHub Actions, Makefile, repository documentation.

---

### Task 1: Add Backend-Only Artifact Validation

**Files:**
- Create: `Helper_Scripts/Packaging/check_pypi_artifacts.py`
- Modify: `Makefile`

- [x] **Step 1: Create an artifact checker**

Implement a Python script that inspects `dist/*.whl` and `dist/*.tar.gz`, fails if frontend paths such as `apps/tldw-frontend`, `.next`, `node_modules`, `package.json`, or `bun.lock` appear, and confirms expected backend package roots are present.

- [x] **Step 2: Wire the checker into `make pypi-check`**

Add a `pypi-check-contents` target and call it after `twine check dist/*` so the existing local and CI commands validate package scope.

- [x] **Step 3: Run validation**

Run: `make pypi-check`

Expected: build succeeds, `twine check` passes, and the content guard reports valid backend/API-only distributions.

### Task 2: Harden PyPI CI And Publish Workflows

**Files:**
- Modify: `.github/workflows/pypi-package.yml`
- Modify: `.github/workflows/publish-pypi.yml`

- [x] **Step 1: Align action pins**

Replace remaining mutable action tags with the pinned SHAs already used by the MCP publish workflow where matching actions are present.

- [x] **Step 2: Expand PR trigger paths**

Ensure `pypi-package.yml` runs when `Makefile`, `Helper_Scripts/Packaging/**`, or `.github/workflows/publish-pypi.yml` changes.

- [x] **Step 3: Preserve manual publish controls**

Keep `publish-pypi.yml` manual-dispatch-only and keep the `testpypi` default target.

### Task 3: Document Backend/API-Only Release Boundary

**Files:**
- Modify: `Docs/Development/PyPI_Publishing.md`
- Modify: `Docs/Development/Packaging_and_Distribution_Strategy.md`

- [x] **Step 1: Update PyPI guide scope**

State that `tldw-server` on PyPI is backend/API/CLI only, does not bundle the WebUI, and should be paired with WebUI Docker/release artifacts when a UI is needed.

- [x] **Step 2: Update distribution strategy**

Make the recommended default unambiguous: PyPI for backend/API, GHCR or release tarball for the WebUI.

- [x] **Step 3: Add release operator checks**

Document local build/check, isolated wheel smoke install, TestPyPI smoke, and final PyPI publish order.

### Task 4: Final Verification

**Files:**
- Update: `backlog/tasks/task-12014 - Harden-backend-only-PyPI-release-setup.md`

- [x] **Step 1: Run packaging checks**

Run: `make pypi-check`

Expected: distributions build and validate.

- [x] **Step 2: Run isolated smoke install**

Run a temporary virtualenv install from the built wheel and import `tldw_Server_API`.

Expected: import succeeds.

- [x] **Step 3: Run Bandit on touched Python script**

Run: `python -m bandit -r Helper_Scripts/Packaging -f json -o /tmp/bandit_backend_api_pypi.json`

Expected: no new security findings.

- [x] **Step 4: Record results**

Update `TASK-12014` with touched files, verification results, and known skips or blockers.
