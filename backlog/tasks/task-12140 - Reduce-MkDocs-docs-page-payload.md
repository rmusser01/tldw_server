---
id: TASK-12140
title: Reduce MkDocs docs page payload
status: Done
assignee: []
created_date: '2026-07-04 14:47'
updated_date: '2026-07-04 15:17'
labels:
  - docs
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shrink the published docs payload by replacing oversized docs branding assets, disabling local MkDocs search, and documenting/deploying cache headers for the external tldwproject.com docs path.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs logo/favicon assets are reduced from multi-megabyte 1024px PNGs to small web-sized assets.
- [x] #2 MkDocs local search is disabled and search_index.json is no longer built.
- [x] #3 External Apache docs route sends cache headers for static docs assets.
- [x] #4 Verification records built asset sizes, absence of search index, and live/cache header checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented payload fixes: removed the MkDocs search plugin from Docs/mkdocs.yml; replaced Docs/Logo.png and Docs/Published/assets/logo.png with 128x128 PNGs (~39 KB each); replaced Docs/Published/assets/favicon.png with a 32x32 PNG (~3 KB). Apache cache headers were installed directly in /etc/apache2/sites-enabled/tldwproject-le-ssl.conf because AllowOverride is None; backup: /etc/apache2/sites-enabled/tldwproject-le-ssl.conf.bak-20260704T145642Z. Header verification before redeploy: /server/docs/ returns Cache-Control public max-age=300; /server/docs/assets/logo.png returns public max-age=86400.

Live deploy verification: rebuilt docs with search disabled; deployed /tmp/tldw-server-docs-site-payload-20260704.tar.gz (SHA-256 f46dd9821fe795bb2fae28574250161663aa813e801a3a4d6c6f7ba6daaab8be) to /var/www/tldwproject/public/server/docs. Previous live docs were backed up at /var/www/tldwproject/public/server/docs.bak-20260704T150345Z. Live checks against 134.209.75.12: /server/docs/ returns 200, Content-Length 56296, Cache-Control public max-age=300; /assets/logo.png returns 39679 bytes and max-age=86400; /assets/favicon.png returns 3179 bytes and max-age=86400; /search/search_index.json returns 404; downloaded live index contains no md-search/search_index markers. Bandit skipped: touched repo files are docs config, static image assets, and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the MkDocs docs per-page payload by removing local search, replacing oversized docs logo/favicon assets with small web-sized PNGs, and deploying Apache cache headers plus a rebuilt lean docs site to tldwproject.com/server/docs.
<!-- SECTION:FINAL_SUMMARY:END -->
