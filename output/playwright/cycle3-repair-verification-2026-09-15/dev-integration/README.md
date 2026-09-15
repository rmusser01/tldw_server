# Verified dev integration checkpoint

Parents: reviewed UAT repair a0c48e84c4 and freshly fetched origin/dev 2e1a5e58d3. The merge was staged without conflicts and validated before its commit. This is integration verification, not fresh UAT. The branch did not start from latest dev; the separate ancestry audit retains the missed 32 commits.

## Validation

- Backend: 207 passed, four warnings, across auth login/refresh/RBAC, privilege catalog, VN preflight/API and image model resolution/listing.
- Frontend: 174 passed in 14 suites covering navigation, app auth/layout, titles, ICU, VN actions, refresh and the repaired Chat mirror.
- Portable VZ workflow controls: initial 98 passed; two process-cleanup checks failed because sandbox policy blocked ps. The same two tests pass with approved process-query access, eight other cases deselected in that focused retry. No VM or host drill ran; native Swift/Go and Linux-specific runtime acceptance are outside this web integration check.
- Full TypeScript after refreshing generated types: compiler exit2, exactly the existing90 normalized diagnostics, none added or removed. This is not a passing typecheck.
- ESLint:20 incoming TS/TSX files, zero errors,24 warnings identical to origin/dev. Existing root-run pages-directory configuration advisory remains; no new lint signatures. Ruff on the composed auth dependency reports zero findings.
- Bandit:11 API/image/VN production files plus the incoming portable helper script, zero findings and scan errors.
- API drift check: stale snapshot reproduced before and after the merge. The combined export adds only the expected VN preflight path and two schemas relative to the UAT checkout; removes/changes no previous paths or schemas. The existing exporter and openapi-typescript regenerated the snapshot/types. The final drift check passes at2097paths/3208schemas.

The shared editable environment points to another worktree; command-local PYTHONPATH matches the isolated UAT launcher and resolves this checkout's local packages. No shared installation was changed. Generated OpenAPI/type artifacts remain ignored; the small fingerprint is the tracked output.

Five unrelated untracked source/task files retain their pre-merge hashes. Incoming duplicate Backlog logical IDs13249 and13259 are preserved and documented; no ambiguous task edit was made. Pytest emitted unrelated temporary-directory cleanup warnings after successful checks; no manual cleanup was performed.

Copied text/logs have only trailing blank lines normalized. Schema logs retain explicitly filtered exporter status lines; full original logs stay at the recorded private paths. This bundle excludes runtime credentials. Final independent integration review is clear and retained in independent-review.md. Merge267c00cab1 now includes dev2e1a5e58d3 (zero dev-only commits). Both isolated API/frontend pairs restarted from existing profiles and returned200 at20:48:28UTC; see runtime-restart-health.json. This is startup/HTTP verification, not authenticated workflow acceptance.
