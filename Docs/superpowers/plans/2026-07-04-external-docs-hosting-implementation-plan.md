# External Docs Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point users to `https://tldwproject.com/server/docs/` as the canonical public docs URL, keep GitHub Pages as a mirror, and document how the MkDocs site is built and administered.

**Architecture:** Keep the existing MkDocs pipeline: source Markdown under `Docs/`, generated curated input under `Docs/Published/`, and static output from `mkdocs build -f Docs/mkdocs.yml`. Make `tldwproject.com/server/docs/` canonical in public links and MkDocs metadata; document manual external deploy first, with optional site-side clone/pull/build automation later.

**Tech Stack:** Markdown, static HTML, MkDocs Material, GitHub Actions Pages mirror.

---

### Task 1: Update Public Docs Links And Canonical URL

**Files:**
- Modify: `README.md`
- Modify: `Docs/Website/index.html`
- Modify: `Docs/mkdocs.yml`

- [ ] **Step 1: Update MkDocs canonical URL**

Change `Docs/mkdocs.yml`:

```yaml
site_url: https://tldwproject.com/server/docs/
```

- [ ] **Step 2: Add README public docs link**

In `README.md`, add a visible link near the existing docs resources:

```markdown
- [Public documentation site](https://tldwproject.com/server/docs/) - canonical docs hosted on tldwproject.com
```

- [ ] **Step 3: Add website docs link**

In `Docs/Website/index.html`, add `Docs` to the header and footer nav, pointing at:

```html
<a href="https://tldwproject.com/server/docs/">Docs</a>
```

- [ ] **Step 4: Text-check the URL appears where expected**

Run:

```bash
rg -n "https://tldwproject.com/server/docs/" README.md Docs/Website/index.html Docs/mkdocs.yml
```

Expected: matches in all three files.

### Task 2: Update Docs Site Admin Guide

**Files:**
- Modify: `Docs/Code_Documentation/Docs_Site_Guide.md`

- [ ] **Step 1: Update overview hosting model**

Document:

- canonical public URL: `https://tldwproject.com/server/docs/`
- GitHub Pages remains a mirror, not the canonical public URL
- source docs are edited under `Docs/`, not `Docs/Published/`

- [ ] **Step 2: Add external deployment instructions**

Document the two supported admin paths:

```markdown
Manual external deploy:
1. Run `bash Helper_Scripts/refresh_docs_published.sh`
2. Run `mkdocs build -f Docs/mkdocs.yml`
3. Copy the built static site to the external host at `/server/docs/`

Optional site-side automation:
1. Clone or pull this repository on the external host
2. Detect a new version or commit
3. Run the refresh and build commands
4. Replace the served `/server/docs/` files atomically if the build succeeds
```

- [ ] **Step 3: Document mirror caveat**

Add that GitHub Pages mirror builds from the same MkDocs source and intentionally uses external canonical URLs from `site_url`.

### Task 3: Verify Docs Build And Link Coverage

**Files:**
- Read/check only after Tasks 1-2.

- [ ] **Step 1: Refresh curated docs**

Run:

```bash
bash Helper_Scripts/refresh_docs_published.sh
```

Expected: exit 0.

- [ ] **Step 2: Build MkDocs**

Run:

```bash
mkdocs build -f Docs/mkdocs.yml
```

Expected: exit 0. Existing baseline warnings are acceptable if the command exits 0.

- [ ] **Step 3: Check no stale docs URLs remain in canonical slots**

Run:

```bash
rg -n "tldwproject.org/server/docs" README.md Docs/Website/index.html Docs/mkdocs.yml Docs/Code_Documentation/Docs_Site_Guide.md
rg -n "^site_url: https://tldwproject.com/server/docs/$" Docs/mkdocs.yml
```

Expected: first command has no matches; second command has one match. GitHub Pages mirror references are allowed when clearly labeled as mirror-only.

- [ ] **Step 4: Bandit scope note**

No Bandit command is needed for this task because the touched implementation files are Markdown, HTML links, YAML metadata, and Backlog records only.

### Task 4: Update Backlog And Commit

**Files:**
- Modify: `backlog/tasks/task-12128 - Document-external-docs-hosting-links.md`

- [ ] **Step 1: Record verification and final summary**

Update `TASK-12128` with:

- files changed
- verification command results
- Bandit skip reason
- final summary

- [ ] **Step 2: Commit implementation**

Run:

```bash
git add README.md Docs/Website/index.html Docs/mkdocs.yml Docs/Code_Documentation/Docs_Site_Guide.md "backlog/tasks/task-12128 - Document-external-docs-hosting-links.md"
git commit -m "docs: link external docs hosting"
```

Expected: commit succeeds. Do not stage unrelated existing worktree changes.
