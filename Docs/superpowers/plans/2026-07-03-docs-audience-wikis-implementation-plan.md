# Docs Audience Wikis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add audience-focused User Wiki and Developer Wiki entry points to the existing public MkDocs site.

**Architecture:** Keep one MkDocs site and one published docs pipeline. Add `Docs/Wiki` as source landing pages, sync that folder to `Docs/Published/Wiki`, and reorganize navigation around audience tabs without moving existing guide files.

**Tech Stack:** Markdown, MkDocs Material, Bash refresh script, pytest docs contract tests.

---

### Task 1: Write Docs Contract Test

**Files:**
- Create: `tldw_Server_API/tests/Docs/test_docs_audience_wikis.py`

- [ ] **Step 1: Add a failing test**

Create a pytest module that asserts the `Docs/Wiki` source pages, `Docs/Published/Wiki` generated pages, MkDocs top-level nav entries, and README links exist.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Docs/test_docs_audience_wikis.py`

Expected: fail because the new wiki pages do not exist yet.

### Task 2: Add Wiki Source Pages and Publish Sync

**Files:**
- Create: `Docs/Wiki/index.md`
- Create: `Docs/Wiki/User_Wiki.md`
- Create: `Docs/Wiki/Developer_Wiki.md`
- Modify: `Helper_Scripts/refresh_docs_published.sh`

- [ ] **Step 1: Add concise source landing pages**

Create the chooser, user wiki, and developer wiki pages with links to existing stable docs.

- [ ] **Step 2: Sync Wiki into Published**

Update `Helper_Scripts/refresh_docs_published.sh` to copy `Docs/Wiki` to `Docs/Published/Wiki` and include the wiki links in the generated `Docs/Published/index.md`.

- [ ] **Step 3: Refresh generated docs**

Run: `bash Helper_Scripts/refresh_docs_published.sh`

Expected: `Docs/Published/Wiki/index.md`, `Docs/Published/Wiki/User_Wiki.md`, and `Docs/Published/Wiki/Developer_Wiki.md` exist.

### Task 3: Reorganize Navigation and Guidance

**Files:**
- Modify: `Docs/mkdocs.yml`
- Modify: `README.md`
- Modify: `Docs/Code_Documentation/Docs_Site_Guide.md`

- [ ] **Step 1: Make the nav audience-first**

Replace the current broad top-level docs tabs with `Home`, `User Wiki`, `Developer Wiki`, and shared release/status links.

- [ ] **Step 2: Update authoring guidance**

Document that source pages remain in existing folders, generated pages under `Docs/Published` are not edited manually, and audience chooser pages live under `Docs/Wiki`.

- [ ] **Step 3: Update README entry points**

Point users to `Docs/Wiki/User_Wiki.md` and contributors to `Docs/Wiki/Developer_Wiki.md`.

### Task 4: Verify and Commit

**Files:**
- Modify: `backlog/tasks/task-12119 - Split-published-docs-navigation-into-user-and-developer-wiki-entry-points.md`

- [ ] **Step 1: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Docs/test_docs_audience_wikis.py`

Expected: pass.

- [ ] **Step 2: Run docs checks**

Run: `python Helper_Scripts/docs/check_public_private_boundary.py`

Expected: pass.

Run: `mkdocs build -f Docs/mkdocs.yml`

Expected: pass with existing baseline warnings only.

- [ ] **Step 3: Update Backlog task and commit**

Record verification results in `TASK-12119`, stage the docs/test/task changes, and commit.
