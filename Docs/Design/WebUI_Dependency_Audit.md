# WebUI Dependency Audit

Date: 2026-05-07
Status: Draft audit for issue #1346

## References

- GitHub issue: https://github.com/rmusser01/tldw_server/issues/1346
- Design spec: ../superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
- Backlog task: TASK-101

## Scope

This audit covers direct package declarations and usage signals for:

- `apps/tldw-frontend/package.json`
- `apps/packages/ui/package.json`
- `apps/bun.lock`
- `apps/extension/package.json` as an impact-check surface for shared UI candidates

This audit does not remove packages or rewrite runtime code.

## Methodology

1. Read direct dependency declarations from the WebUI, shared UI, and extension manifests.
2. Scan source, test, script, and config files for import/config usage.
3. Classify direct dependencies with the approved decision values.
4. Rank follow-up work into quick cleanup, replacement, deferred design, and keep groups.
5. Record verification commands and known skips.

## Decision Legend

| Decision | Meaning |
| --- | --- |
| `keep` | Current dependency is justified. |
| `remove-now` | Candidate for a narrow package-removal PR. |
| `replace-later` | Replacement is plausible but needs its own PR. |
| `defer-design` | Needs a separate design before replacement. |
| `investigate-lockfile` | Needs lockfile or ownership confirmation before action. |

## Dependency Inventory

| Package | Declared locations | Import count | Representative sites | Consumer surface | Category | Decision | Risk | Expected impact | Follow-up slice |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |

## Ranked Follow-Up Queue

### Quick Cleanup Candidates

### Replacement Candidates

### Deferred Design Candidates

### Explicit Keeps

## Verification

## Known Skips And Blockers
