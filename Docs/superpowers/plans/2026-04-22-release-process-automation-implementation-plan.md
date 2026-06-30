# Release Process Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-owned local release command that cuts releases from `main`, gates on the documented required checks for the pre-bump `origin/main` commit, updates release metadata/docs consistently, and publishes a GitHub Release that triggers the existing Docker release workflow.

**Architecture:** Keep the rollout narrow and explicit. Introduce a focused Python helper under `Helper_Scripts/` for release orchestration, wire Makefile entrypoints to that helper, narrow the PyPI workflow trigger so GitHub Release publication does not auto-publish PyPI, and update the maintainer docs to describe the real release contract. Validate behavior with utility-script tests, workflow contract tests, and docs contract tests instead of relying on manual inspection.

**Tech Stack:** Python 3, Makefile, GitHub Actions YAML, pytest, Markdown docs

---

### Task 1: Lock The Release Boundary And Workflow Contracts

**Files:**
- Modify: `.github/workflows/publish-pypi.yml`
- Modify: `Docs/Development/PyPI_Publishing.md`
- Modify: `Docs/Development/Container_Image_Lifecycle.md`
- Test: `tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py`
- Create: `tldw_Server_API/tests/CI/test_release_workflow_contracts.py`

- [ ] **Step 1: Write the failing workflow-contract tests**

Add a new CI contract test file that asserts:

- `publish-pypi.yml` no longer has a `release` trigger
- `publish-pypi.yml` still supports `workflow_dispatch`
- `publish-docker.yml` remains release-driven
- the documented release image set stays `app`, `worker`, `audio-worker`

Also extend `tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py` with a failing assertion that `publish-pypi.yml` is manual-dispatch-only.

- [ ] **Step 2: Run the workflow-contract tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py -v`
Expected: FAIL because `publish-pypi.yml` still triggers on `release.published` and the new release workflow contract test file does not exist yet.

- [ ] **Step 3: Narrow the PyPI publish trigger**

Edit `.github/workflows/publish-pypi.yml` so:

- `workflow_dispatch` remains
- `release.published` is removed
- the existing `publish-pypi` job still works for manual `target == pypi`

Keep the existing build job and trusted publishing configuration unchanged.

- [ ] **Step 4: Update the maintainer publishing docs**

Revise:

- `Docs/Development/PyPI_Publishing.md`
- `Docs/Development/Container_Image_Lifecycle.md`

So they state clearly:

- GitHub Release publication drives Docker release publishing
- PyPI publishing is manual-dispatch-only in this rollout
- `main` pushes continue to republish rolling GHCR snapshots

- [ ] **Step 5: Re-run the workflow-contract tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py -v`
Expected: PASS

- [ ] **Step 6: Commit the workflow-boundary change**

```bash
git add .github/workflows/publish-pypi.yml \
  Docs/Development/PyPI_Publishing.md \
  Docs/Development/Container_Image_Lifecycle.md \
  tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py
git commit -m "build: narrow PyPI release trigger"
```

### Task 2: Add The Release Helper Core With Check-Gate Parsing

**Files:**
- Create: `Helper_Scripts/release.py`
- Test: `tldw_Server_API/tests/Utils/test_release_helper.py`
- Read for reference: `Docs/Development/CI_REQUIRED_GATES.md`

- [ ] **Step 1: Write the failing helper tests**

Create `tldw_Server_API/tests/Utils/test_release_helper.py` with focused unit tests for:

- parsing the current version from `pyproject.toml`
- computing patch and minor bumps
- extracting stable required gate names from `Docs/Development/CI_REQUIRED_GATES.md`
- rejecting unsupported release branches
- detecting exact duplicate bullets within the same changelog subsection after simple normalization
- classifying resumable states:
  - local release commit only
  - local tag only
  - remote tag without GitHub Release
  - existing GitHub Release

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_release_helper.py -v`
Expected: FAIL because `Helper_Scripts/release.py` does not exist yet.

- [ ] **Step 3: Write the minimal helper skeleton**

Implement `Helper_Scripts/release.py` with pure functions first:

- `read_current_version(...)`
- `bump_version(...)`
- `extract_required_check_names(...)`
- `normalize_bullet_text(...)`
- `find_exact_duplicate_bullets(...)`
- `classify_release_state(...)`

Keep shell/`gh` execution out of these functions so the logic stays easy to test.

- [ ] **Step 4: Re-run the helper tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_release_helper.py -v`
Expected: PASS

- [ ] **Step 5: Commit the helper core**

```bash
git add Helper_Scripts/release.py tldw_Server_API/tests/Utils/test_release_helper.py
git commit -m "feat: add release helper core"
```

### Task 3: Add Changelog Promotion And Docs Metadata Updates

**Files:**
- Modify: `Helper_Scripts/release.py`
- Modify: `README.md`
- Modify: `Docs/mkdocs.yml`
- Modify: `Docs/Published/RELEASE_NOTES.md`
- Test: `tldw_Server_API/tests/Utils/test_release_helper.py`
- Create: `tldw_Server_API/tests/Docs/test_release_docs_contract.py`

- [ ] **Step 1: Add failing tests for changelog promotion and docs metadata**

Extend `tldw_Server_API/tests/Utils/test_release_helper.py` with failing tests for:

- promoting `CHANGELOG.md` `Unreleased` into a dated `X.Y.Z` section
- leaving the `Unreleased` headings reset but empty
- reporting cross-section near-duplicates as warnings instead of hard failures

Create `tldw_Server_API/tests/Docs/test_release_docs_contract.py` with failing assertions that:

- `README.md` release line matches the package version source under the release helper’s update path
- `Docs/mkdocs.yml` version/copyright strings can be updated coherently
- `Docs/Published/RELEASE_NOTES.md` points to the authoritative release-process doc path once that doc exists

- [ ] **Step 2: Run the changelog/docs tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_release_helper.py tldw_Server_API/tests/Docs/test_release_docs_contract.py -v`
Expected: FAIL because changelog promotion and release-doc update helpers are not implemented yet.

- [ ] **Step 3: Implement changelog and docs metadata helpers**

Extend `Helper_Scripts/release.py` with functions to:

- parse and promote the `Unreleased` section
- reject exact duplicate bullets within one subsection
- update `README.md`’s release line text
- update version-bearing metadata in `Docs/mkdocs.yml`
- update the release-notes entry point text in `Docs/Published/RELEASE_NOTES.md`

Do not attempt `Docs/site` rewriting here. Keep this step focused on deterministic source-file updates; generated docs are built separately and remain out of release commits.

- [ ] **Step 4: Re-run the changelog/docs tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_release_helper.py tldw_Server_API/tests/Docs/test_release_docs_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit the changelog/docs metadata layer**

```bash
git add Helper_Scripts/release.py README.md Docs/mkdocs.yml Docs/Published/RELEASE_NOTES.md \
  tldw_Server_API/tests/Utils/test_release_helper.py \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py
git commit -m "feat: add release metadata promotion"
```

### Task 4: Keep Generated Docs Site Out Of Release Commits

**Files:**
- Modify: `Helper_Scripts/release.py`
- Modify: `.gitignore`
- Test: `tldw_Server_API/tests/Docs/test_release_docs_contract.py`
- Read for reference: `Docs/mkdocs.yml`, `.github/workflows/mkdocs.yml`

- [ ] **Step 1: Add a failing generated-docs policy test**

Extend `tldw_Server_API/tests/Docs/test_release_docs_contract.py` with a focused assertion that `Docs/site/` is ignored generated output, has no tracked files, and is not managed by the release helper.

Use a narrow repo-policy assertion rather than checking every generated page.

- [ ] **Step 2: Run the docs contract test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Docs/test_release_docs_contract.py -v`
Expected: FAIL if `Docs/site` is staged, tracked, or still treated as release-managed output.

- [ ] **Step 3: Implement generated-doc exclusion**

Update the release path so that:

- `Docs/site/` remains ignored in `.gitignore`
- no `Docs/site` files are staged or committed
- `Helper_Scripts/release.py` updates source docs metadata only
- generated docs can be built separately by docs publishing workflows

Do not redesign the docs pipeline.

- [ ] **Step 4: Re-run the docs contract test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Docs/test_release_docs_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit the deterministic docs handling**

```bash
git add Helper_Scripts/release.py .gitignore \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py
git commit -m "docs: keep generated docs site out of releases"
```

### Task 5: Add Orchestration, Push Abort Semantics, And Make Targets

**Files:**
- Modify: `Helper_Scripts/release.py`
- Modify: `Makefile`
- Test: `tldw_Server_API/tests/Utils/test_release_helper.py`
- Create: `tldw_Server_API/tests/Utils/test_makefile_release_targets.py`

- [ ] **Step 1: Add failing orchestration and Makefile tests**

Extend `tldw_Server_API/tests/Utils/test_release_helper.py` with orchestration-level tests for:

- requiring branch `main`
- requiring a clean worktree
- hard-aborting on non-fast-forward push failure when `main` has moved
- resuming from “remote tag exists, GitHub Release missing”
- treating the release commit as not requiring a second green CI cycle

Create `tldw_Server_API/tests/Utils/test_makefile_release_targets.py` with failing assertions that:

- `release-patch` target exists
- `release-minor` target exists
- `release` target exists
- the targets delegate to `Helper_Scripts/release.py`

- [ ] **Step 2: Run the orchestration/Makefile tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_release_helper.py tldw_Server_API/tests/Utils/test_makefile_release_targets.py -v`
Expected: FAIL because orchestration functions and Make targets are incomplete.

- [ ] **Step 3: Implement the orchestration layer**

Finish `Helper_Scripts/release.py` so it:

- fetches `origin/main`
- validates green required checks for the documented gate names on the pre-bump `origin/main` SHA
- prepares metadata edits
- creates the release commit and tag
- pushes `main` and the tag
- hard-aborts on non-fast-forward push failure without rebasing or retrying
- creates/publishes the GitHub Release
- resumes cleanly from partially completed states

Keep network and shell calls behind small wrapper functions so they can be stubbed in tests.

- [ ] **Step 4: Add the Make targets**

Update `Makefile` help text and targets to add:

- `release-patch`
- `release-minor`
- `release`
- optional `release-dry-run` or `DRY_RUN=1` wiring

Use the existing Makefile style: thin wrapper targets that delegate to Python helpers.

- [ ] **Step 5: Re-run the orchestration/Makefile tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_release_helper.py tldw_Server_API/tests/Utils/test_makefile_release_targets.py -v`
Expected: PASS

- [ ] **Step 6: Commit the release command wiring**

```bash
git add Helper_Scripts/release.py Makefile \
  tldw_Server_API/tests/Utils/test_release_helper.py \
  tldw_Server_API/tests/Utils/test_makefile_release_targets.py
git commit -m "feat: add local release command"
```

### Task 6: Add The Maintainer Release Process Document

**Files:**
- Create: `Docs/Development/Release_Process.md`
- Modify: `Docs/Published/RELEASE_NOTES.md`
- Modify: `Docs/Release_Checklist.md`
- Test: `tldw_Server_API/tests/Docs/test_release_docs_contract.py`

- [ ] **Step 1: Add a failing docs contract assertion**

Extend `tldw_Server_API/tests/Docs/test_release_docs_contract.py` so it fails until:

- `Docs/Development/Release_Process.md` exists
- it names the authoritative local release commands
- it distinguishes release artifacts from `main` snapshot artifacts
- it states that pushing the release commit republishes `main` snapshots
- it points to `Docs/Release_Checklist.md` as the broad readiness checklist

- [ ] **Step 2: Run the docs contract test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Docs/test_release_docs_contract.py -v`
Expected: FAIL because the release-process doc does not exist yet.

- [ ] **Step 3: Write the maintainer release-process doc**

Create `Docs/Development/Release_Process.md` as the short authoritative operator path. It should cover:

- what the local release command does
- why `main` is the only supported source branch
- the required-check gate source (`Docs/Development/CI_REQUIRED_GATES.md`)
- snapshot republish as a first-class side effect
- which images are formal release artifacts vs `main` snapshots
- retry/recovery behavior
- the narrow PyPI boundary for this rollout

Update the release-notes entry point and checklist cross-links if needed so the new document is the authoritative process reference.

- [ ] **Step 4: Re-run the docs contract test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Docs/test_release_docs_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit the release-process docs**

```bash
git add Docs/Development/Release_Process.md Docs/Published/RELEASE_NOTES.md Docs/Release_Checklist.md \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py
git commit -m "docs: add authoritative release process"
```

### Task 7: Verify The Full Release-Automation Change Set

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-22-release-process-automation-implementation-plan.md`

- [ ] **Step 1: Run the focused Python test suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/Utils/test_release_helper.py \
  tldw_Server_API/tests/Utils/test_makefile_release_targets.py \
  tldw_Server_API/tests/Docs/test_release_docs_contract.py -v
```

Expected: PASS

- [ ] **Step 2: Run docs build verification**

Run: `source .venv/bin/activate && python -m mkdocs build --strict -f Docs/mkdocs.yml`
Expected: PASS

- [ ] **Step 3: Run Bandit on the touched scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  Helper_Scripts/release.py \
  tldw_Server_API/tests/CI \
  tldw_Server_API/tests/Utils \
  tldw_Server_API/tests/Docs \
  Docs/Development \
  Docs/Published \
  .github/workflows \
  -f json -o /tmp/bandit_release_process_automation.json
```

Expected: command completes without new actionable findings in the touched executable paths. Markdown AST parse errors are acceptable if Bandit reports them only for documentation files.

- [ ] **Step 4: Run a dry-run release invocation**

Run one of:

```bash
source .venv/bin/activate && python Helper_Scripts/release.py --dry-run --bump patch
```

or

```bash
make release-patch DRY_RUN=1
```

Expected:

- no files are committed or pushed
- the helper prints the pre-bump `origin/main` SHA it validated
- the helper prints the required gate names it expects
- the helper prints the planned metadata edits, tag name, GitHub Release title/body, and snapshot republish warning

- [ ] **Step 5: Mark plan progress**

Update this plan so completed steps are checked off before handoff.
