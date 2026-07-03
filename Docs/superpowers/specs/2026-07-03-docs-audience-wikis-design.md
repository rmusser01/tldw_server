# Docs Audience Wikis Design

## Goal

Split the existing public MkDocs site into two clear audience entry points: a user-facing wiki and a developer/contributor-facing wiki.

## Scope

This keeps one MkDocs site and one GitHub Pages deployment. It does not move existing guide files or introduce a second documentation build. Existing source paths such as `Docs/User_Guides/index.md`, `Docs/API-related/API_README.md`, and `Docs/Code_Documentation/index.md` remain valid.

## Design

Add shared source landing pages under `Docs/Wiki/`:

- `Docs/Wiki/index.md`: audience chooser for the docs site.
- `Docs/Wiki/User_Wiki.md`: user-focused route map for setup, WebUI, extension, local providers, character chat, knowledge workflows, audio, admin, and operations.
- `Docs/Wiki/Developer_Wiki.md`: contributor-focused route map for development setup, architecture, code guides, API references, testing, docs process, ADRs, and release work.

Publish `Docs/Wiki` through `Helper_Scripts/refresh_docs_published.sh` so generated pages land in `Docs/Published/Wiki/`. Do not manually edit generated published pages.

Rework `Docs/mkdocs.yml` so the first tabs are:

- `Home`
- `User Wiki`
- `Developer Wiki`
- `Release Notes`

User-facing guides and practical API usage are grouped under `User Wiki`. Contributor-oriented API/code/architecture references are grouped under `Developer Wiki`. Shared references may be linked from both audiences.

Update `README.md` and `Docs/Code_Documentation/Docs_Site_Guide.md` to document the audience split and source-of-truth rules.

## Verification

Add a focused docs contract test that checks:

- source wiki pages exist;
- refreshed published wiki pages exist;
- `Docs/mkdocs.yml` exposes top-level User Wiki and Developer Wiki entries;
- `README.md` links to both audience entry points.

Run the docs refresh script, the focused docs test, docs hygiene checks, and `mkdocs build -f Docs/mkdocs.yml`.
