# Documentation Site Guide

This document explains how the tldw_Server documentation site is organized, built, and deployed - and how to safely update it.

## Overview

- Site generator: MkDocs with the Material theme
- Published scope: Only a curated subset of the `Docs/` tree
- Source of truth: Update files under `Docs/`, not `Docs/Published/`
- Build root: `Docs/Published/` (MkDocs `docs_dir`)
- Canonical public URL: `https://tldwproject.com/server/docs/`
- Production deployment: external tldwproject.com host at `/server/docs/`
- Mirror deployment: GitHub Pages via GitHub Actions

The public docs site is for OSS, self-host, and developer documentation. Hosted/commercial docs are excluded from the published site and should live in the private repo instead of this public docs pipeline. GitHub Pages is a mirror of the same MkDocs output; `tldwproject.com/server/docs/` is the canonical URL and the MkDocs `site_url`.

The published site is audience-first:

- `User Wiki`: install, run, configure, and use tldw_server.
- `Developer Wiki`: contribute to, test, package, and understand the codebase.

These wiki pages are MkDocs landing pages in this repository, not the separate GitHub Wiki feature.

## What Gets Published

The refresh script publishes this explicit directory manifest:

- `Docs/Wiki`
- `Docs/API-related`
- `Docs/ADR`
- `Docs/Code_Documentation`
- `Docs/Deployment` (excluding its nested `Monitoring`)
- `Docs/Deployment/Monitoring` (published as top-level `Monitoring`)
- `Docs/Evaluations`, with `Docs/Evals` as the current compatibility fallback
- `Docs/Getting_Started`
- `Docs/User_Guides`

It also publishes these individual files and assets:

- `Docs/Site/index.md` as `Docs/Published/index.md`
- `Docs/Site/RELEASE_NOTES.md` as `Docs/Published/RELEASE_NOTES.md`
- `Docs/Architecture.md`
- `Docs/Operations/Env_Vars.md` as `Docs/Published/Env_Vars.md`
- `Docs/Overview/Feature_Status.md`
- `Docs/Logo.png` as both `Docs/Published/assets/logo.png` and
  `Docs/Published/assets/favicon.png`

Files under `Docs/`, including the special `Docs/Site` sources above, are
canonical. The corresponding `Docs/Published/` files are deterministic,
checked-in generated output. Never edit `Docs/Published/` directly; update the
canonical source and refresh it.

Hosted/commercial docs are excluded from this curated set even when they live under similarly named source areas. If a page exists mainly to run, sell, support, or differentiate the hosted SaaS service, keep it in the private repo rather than adding it to the public docs tree.

## Refreshing Curated Docs

- Script: `Helper_Scripts/refresh_docs_published.sh`
- What it does:
  - Builds the complete manifest above in a fresh staging directory
  - Promotes `Docs/Deployment/Monitoring` to top-level
    `Docs/Published/Monitoring`
  - Removes nested `README.md` files when a sibling `index.md` is canonical
  - Atomically swaps the complete staged tree into `Docs/Published/`
  - Removes stale generated files by replacing, rather than overlaying, the
    previous tree
  - Restores the prior complete tree if the swap fails

If you are working on hosted SaaS runbooks, billing/customer-surface guides, or other commercial operating material, use the private repo instead of expanding this public refresh flow.

Run locally:

```
bash Helper_Scripts/refresh_docs_published.sh
```

CI also runs this script automatically before building the site.

After changing any canonical published source, run the refresh and stage the
resulting `Docs/Published/` changes with the source change. A second refresh
must produce no diff.

## Local Preview and Build

Install dependencies (once):

```
pip install mkdocs mkdocs-material mkdocs-git-revision-date-localized-plugin
```

Serve locally (auto-reloads on file changes):

```
mkdocs serve -f Docs/mkdocs.yml
```

Build static site with the required zero-warning check (outputs to
`Docs/_site/`):

```
mkdocs build --strict -f Docs/mkdocs.yml
```

Any warning is a failed build and must be fixed in the canonical source,
navigation, or publication manifest before merging.

## "Last updated" Dates

- The site shows per-page "Last updated" timestamps via the `git-revision-date-localized` plugin, using relative time ("time ago").
- Accurate dates require git history. Locally, ensure your repo has the relevant commits. In CI, we fetch full history (`fetch-depth: 0`).
- Configuration lives in `Docs/mkdocs.yml` under `plugins.git-revision-date-localized` (with `type: timeago`).
- If a page has no history (e.g., new file), the plugin falls back to the current build time.

## Theme and Assets

- Theme: Material for MkDocs (configured in `mkdocs.yml`)
- Features: tabs, instant navigation, top nav, tracking, section indexes, copy buttons on code
- Palettes: light/dark based on system preference
- Logo/Favicon: `Docs/Logo.png` is copied to `Docs/Published/assets/` as `logo.png` and `favicon.png`

To change the logo: replace `Docs/Logo.png` and run the refresh script.

## Navigation

- The sidebar and ordering are defined explicitly in `mkdocs.yml` under `nav:`
- When adding a new page you want visible in the sidebar, add a new entry under the appropriate section in `Docs/mkdocs.yml`
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
2. Run `bash Helper_Scripts/refresh_docs_published.sh` to regenerate the
   complete curated tree
3. If the new page should appear in the sidebar, update `Docs/mkdocs.yml`
   `nav:` accordingly
4. Run `mkdocs build --strict -f Docs/mkdocs.yml`
5. Run the refresh again and confirm it produces no diff
6. Stage the canonical and generated changes together, then commit and push;
   CI repeats the refresh and strict build before deployment

Notes:
- Put audience chooser pages in `Docs/Wiki/`
- Put user-facing workflow guides in `Docs/User_Guides/` or `Docs/Getting_Started/`
- Put contributor-facing implementation guides in `Docs/Code_Documentation/`
- Put public architecture decision records in `Docs/ADR/`
- Keep file names stable after they’re published to avoid broken links
- Use relative links within the allowed folders; avoid linking to WIP docs outside the curated set
- Prefer images stored under `Docs/assets/` or section subfolders; the refresh script copies section contents
- Avoid linking to hosted/private docs from public pages; those references belong in the private repo

## External Deployment

The production docs live on the external tldwproject.com host at `/server/docs/`.

Manual external deploy:

1. Run `bash Helper_Scripts/refresh_docs_published.sh`
2. Run `mkdocs build --strict -f Docs/mkdocs.yml`
3. Copy the generated static site output to the external host under `/server/docs/`

Optional site-side automation:

1. Clone or pull this repository on the external host
2. Detect a new version or commit
3. Run the refresh and build commands
4. Replace the served `/server/docs/` files atomically if the build succeeds

If an external deploy fails:
- Confirm `Docs/mkdocs.yml` has `site_url: https://tldwproject.com/server/docs/`
- Check the external host logs for copy/job failures
- Check GitHub workflow logs for mirror build errors (usually missing files/links)
- Re-run the refresh script locally and build with `mkdocs build --strict -f Docs/mkdocs.yml` to reproduce

## GitHub Pages Mirror

- Workflow file: `.github/workflows/mkdocs.yml`
- Triggers: pushes to `dev`, `main`, and `PG-Backend`, and manual runs;
  `dev` builds are validated but are not deployed
- Steps: checkout -> install -> refresh curated docs -> boundary check -> strict
  zero-warning build -> deploy to GitHub Pages
- Repository Settings -> Pages: set Source = GitHub Actions

The GitHub Pages site is a mirror. It intentionally builds from the same MkDocs source while using canonical links from `site_url`, so generated canonical URLs point to `https://tldwproject.com/server/docs/`.

## Advanced (Optional)

- Extra plugins: consider `mkdocs-material[imaging]` for image optimizations
- Custom features: adjust `theme.features` and `markdown_extensions` in `mkdocs.yml`

## Summary

- Author docs in their canonical `Docs/` source (never directly in
  `Docs/Published/`)
- Use the refresh script to atomically regenerate and stage the complete
  checked-in curated tree
- Require a zero-warning `mkdocs build --strict -f Docs/mkdocs.yml`
- Keep `mkdocs.yml` `nav:` updated for sidebar visibility and order
- Deploy the built site to `https://tldwproject.com/server/docs/`
- CI builds and deploys a GitHub Pages mirror
- Hosted/commercial docs are excluded from the public docs site and belong in the private repo
