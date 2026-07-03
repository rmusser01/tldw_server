# Documentation Site Guide

This document explains how the tldw_Server documentation site is organized, built, and deployed - and how to safely update it.

## Overview

- Site generator: MkDocs with the Material theme
- Published scope: Only a curated subset of the `Docs/` tree
- Source of truth: Update files under `Docs/`, not `Docs/Published/`
- Build root: `Docs/Published/` (MkDocs `docs_dir`)
- Deployment: GitHub Pages via GitHub Actions

The public docs site is for OSS, self-host, and developer documentation. Hosted/commercial docs are excluded from the published site and should live in the private repo instead of this public docs pipeline.

The published site is audience-first:

- `User Wiki`: install, run, configure, and use tldw_server.
- `Developer Wiki`: contribute to, test, package, and understand the codebase.

These wiki pages are MkDocs landing pages in this repository, not the separate GitHub Wiki feature.

## What Gets Published

Only these folders are included on the public site:

- `Docs/Wiki`
- `Docs/API-related`
- `Docs/ADR`
- `Docs/Code_Documentation`
- `Docs/Deployment` (excluding its nested `Monitoring`)
- `Docs/Deployment/Monitoring` (published as top-level `Monitoring`)
- `Docs/Evals`
- `Docs/User_Guides`

These root-level files are also included when present:

- `Docs/Architecture.md`

The curated content is synced into `Docs/Published/`. Do not manually edit files in `Docs/Published/` - they can be overwritten by the refresh script or CI.

Hosted/commercial docs are excluded from this curated set even when they live under similarly named source areas. If a page exists mainly to run, sell, support, or differentiate the hosted SaaS service, keep it in the private repo rather than adding it to the public docs tree.

## Refreshing Curated Docs

- Script: `Helper_Scripts/refresh_docs_published.sh`
- What it does:
  - Copies the approved folders from `Docs/` to `Docs/Published/`
  - Copies the audience wiki landing pages from `Docs/Wiki`
  - Copies architecture decision records from `Docs/ADR`
  - Copies `Docs/Architecture.md` when present
  - Promotes `Docs/Deployment/Monitoring` to top-level `Docs/Published/Monitoring`
  - Removes the nested `Monitoring` under `Deployment` to avoid duplication
  - Preserves each section's `index.md` landing page
  - Copies `Docs/Logo.png` into `Docs/Published/assets/` as `logo.png` and `favicon.png`

If you are working on hosted SaaS runbooks, billing/customer-surface guides, or other commercial operating material, use the private repo instead of expanding this public refresh flow.

Run locally:

```
bash Helper_Scripts/refresh_docs_published.sh
```

CI also runs this script automatically before building the site.

## Local Preview and Build

Install dependencies (once):

```
pip install mkdocs mkdocs-material mkdocs-git-revision-date-localized-plugin
```

Serve locally (auto-reloads on file changes):

```
mkdocs serve
```

Build static site (outputs to `site/`):

```
mkdocs build
```

## "Last updated" Dates

- The site shows per-page "Last updated" timestamps via the `git-revision-date-localized` plugin, using relative time ("time ago").
- Accurate dates require git history. Locally, ensure your repo has the relevant commits. In CI, we fetch full history (`fetch-depth: 0`).
- Configuration lives in `mkdocs.yml` under `plugins.git-revision-date-localized` (with `type: timeago`).
- If a page has no history (e.g., new file), the plugin falls back to the current build time.

## Theme and Assets

- Theme: Material for MkDocs (configured in `mkdocs.yml`)
- Features: tabs, instant navigation, top nav, tracking, section indexes, copy buttons on code
- Palettes: light/dark based on system preference
- Logo/Favicon: `Docs/Logo.png` is copied to `Docs/Published/assets/` as `logo.png` and `favicon.png`

To change the logo: replace `Docs/Logo.png` and run the refresh script.

## Navigation

- The sidebar and ordering are defined explicitly in `mkdocs.yml` under `nav:`
- When adding a new page you want visible in the sidebar, add a new entry under the appropriate section in `mkdocs.yml`
- The nav uses paths relative to `Docs/Published/`
- Keep the top-level navigation audience-first: `Home`, `User Wiki`, `Developer Wiki`, and shared reference links.
- User-facing workflow docs belong under the `User Wiki` nav tree.
- Contributor, implementation, architecture, and docs-maintenance material belongs under the `Developer Wiki` nav tree.

Example nav entry (under Code section):

```
- Code:
    - Documentation Site Guide: Code_Documentation/Docs_Site_Guide.md
```

Tip: keep titles short and parallel (e.g., "Guide", "Reference", "Checklist").

## Adding or Updating Docs

1. Edit or add Markdown files under the appropriate source folder in `Docs/`
2. Run `bash Helper_Scripts/refresh_docs_published.sh` to re-sync curated content
3. If the new page should appear in the sidebar, update `mkdocs.yml` `nav:` accordingly
4. Preview with `mkdocs serve`
5. Commit and push; CI will refresh, build, and deploy the site

Notes:
- Put audience chooser pages in `Docs/Wiki/`
- Put user-facing workflow guides in `Docs/User_Guides/` or `Docs/Getting_Started/`
- Put contributor-facing implementation guides in `Docs/Code_Documentation/`
- Put public architecture decision records in `Docs/ADR/`
- Keep file names stable after they’re published to avoid broken links
- Use relative links within the allowed folders; avoid linking to WIP docs outside the curated set
- Prefer images stored under `Docs/assets/` or section subfolders; the refresh script copies section contents
- Avoid linking to hosted/private docs from public pages; those references belong in the private repo

## Deployment

- Workflow file: `.github/workflows/mkdocs.yml`
- Triggers: pushes to `main` and `PG-Backend`, and manual runs
- Steps: checkout → install → refresh curated docs → build → deploy to GitHub Pages
- Repository Settings → Pages: set Source = GitHub Actions

If a deploy fails:
- Check the workflow logs for build errors (usually missing files/links)
- Re-run the refresh script locally and build with `mkdocs build` to reproduce

## Advanced (Optional)

- Strict builds: enable `--strict` in the workflow once all links remain within the curated set
- Extra plugins: consider `mkdocs-material[imaging]` for image optimizations
- Custom features: adjust `theme.features` and `markdown_extensions` in `mkdocs.yml`

## Summary

- Author docs in `Docs/` (not `Docs/Published/`)
- Use the refresh script to curate and sync
- Keep `mkdocs.yml` `nav:` updated for sidebar visibility and order
- CI builds and deploys automatically to GitHub Pages
- Hosted/commercial docs are excluded from the public docs site and belong in the private repo
