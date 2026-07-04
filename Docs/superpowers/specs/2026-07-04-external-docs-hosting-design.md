# External Docs Hosting Design

Backlog task: `TASK-12128`

## Goal

Make `https://tldwproject.com/server/docs/` the public documentation URL for tldw Server, and make the repo explain how those docs are built, updated, and administered.

## Context

The repo already has a MkDocs Material site configured in `Docs/mkdocs.yml`. Source docs live under `Docs/`; the curated published input is generated under `Docs/Published/` by `Helper_Scripts/refresh_docs_published.sh`; the current workflow builds that static site.

The project website source is `Docs/Website/index.html`. The README already has a large documentation section, but it does not clearly point users to the external public docs URL.

## Decision

Use `https://tldwproject.com/server/docs/` as the canonical public docs URL.

The implementation should keep the existing MkDocs build pipeline and change only the publishing assumptions and links:

- README links should point users to `https://tldwproject.com/server/docs/`.
- `Docs/Website/index.html` should include a visible docs link to `https://tldwproject.com/server/docs/`.
- `Docs/mkdocs.yml` should set `site_url` to `https://tldwproject.com/server/docs/` so generated canonical links match the external host.
- `Docs/Code_Documentation/Docs_Site_Guide.md` should describe external hosting at `/server/docs/` as the production target.
- The GitHub Pages workflow should remain enabled as a mirror of the same docs, not as the canonical public URL.

## Data Flow

1. Authors edit Markdown under `Docs/`.
2. The refresh script syncs approved public docs into `Docs/Published/`.
3. MkDocs builds static HTML from `Docs/mkdocs.yml`.
4. The external website host serves that built output at `/server/docs/`.
5. GitHub Pages serves the same built docs as a mirror.
6. README and website visitors use `https://tldwproject.com/server/docs/` as the primary link.

## Admin Notes

The docs guide should document the operational owner actions without assuming a specific external provider:

- run `bash Helper_Scripts/refresh_docs_published.sh`
- run `mkdocs build -f Docs/mkdocs.yml`
- publish the generated static site output to the external host under `/server/docs/`
- start with manual copy/deploy of the built static site if no host automation exists yet
- optional later automation: a site-side clone/pull/build job that detects a new repo version, runs the refresh/build steps, and updates `/server/docs/`
- keep the GitHub Pages workflow as a mirror build/deploy for the same docs content
- keep `Docs/Published/` generated rather than hand-edited
- keep private or hosted-only docs out of the public curated docs pipeline

## Error Handling

No application error handling is needed. The docs should call out the practical failure cases:

- missing page in `Docs/Published/`
- broken MkDocs links
- stale `site_url`
- external host not serving the built output at `/server/docs/`
- GitHub Pages mirror drifting from the same MkDocs build source
- automated site clone job failing to pull or build after a new version is detected

## Testing

Use the smallest verification that proves the docs path still builds:

- `bash Helper_Scripts/refresh_docs_published.sh`
- `mkdocs build -f Docs/mkdocs.yml`
- text checks for the canonical external URL in README, website, MkDocs config, and the docs guide

Bandit is not applicable because this is documentation and static HTML metadata only.

## Scope Control

Skipped: building a new docs application or adding a provider-specific deployment script. Start with documented manual deployment; add a repo-owned or site-owned automation only when the external host details are known.
