---
id: TASK-13176
title: Resolve Persona visual asset URLs against the configured backend
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 14:55'
updated_date: '2026-09-05 17:04'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Migu UAT in advanced deployment with separate frontend/backend origins: bundled Migu draft creation and activation succeeded, but images requested relative /api/v1 paths from the frontend and returned repeated 404s. Authenticated backend GET returned PNG 200 for the same asset.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Activated Buddy images and builder previews load using the configured server transport in advanced and quickstart deployments.
- [x] #2 Protected assets preserve authentication and failed image loads do not generate an unbounded request loop.
- [x] #3 Buddy loads protected frames on demand with bounded retained blob memory, cancellation, and source-change cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use existing authenticated binary transport for server-owned Persona asset paths and disposable object URLs, share the loader with candidate thumbnails, retain failed loads until source changes, and release URLs/cancel requests on unmount. Qodo review: load only the displayed frame; retain a small per-renderer cache bounded by count and bytes. Add regressions for unused assets, eviction, and cancellation. CI follow-up: reproduce VisualPackEditor failures with Node20 and the package-owned deterministic config; correct proven readiness/scheduling or contract issues without weakening assertions; rerun targeted editor and renderer suites. Advanced browser UAT already recorded; quickstart acceptance remains blocked by TASK-13181. ADR required: no. ADR path: N/A. Reason: routine repair of existing transport, component lifetime and test synchronization contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented authenticated binary asset loading and disposable object URLs in usePersonaVisualAssetUrls, SpriteFrameRenderer, and generated candidate thumbnails. Failures remain bounded across animation changes, requests abort and URLs revoke on cleanup/source change, and external asset origins never receive credentials. Red3/3 auth regressions failed; final renderer tests include source replacement and external-origin credential exclusion. Real advanced-mode Migu builder and Buddy decode96x96 frames from backend PNG200. Quickstart now passes readiness after13179, but its cookie-auth legacy user dependency blocks Persona; AC1 remains open pending that separate authentication repair. Bandit N/A TypeScript. Scoped ESLint0errors; standard img warning and preexisting editor dependency warnings remain. ADR not required: existing transport and lifetime contract repaired.

Published in draft server PR https://github.com/rmusser01/tldw_server/pull/2884 against dev63358431d7. Rebased-tree targeted frontend265/backend54 tests pass; repository-wide typecheck remains80 unrelated errors. Draft also documents separate setup-task ID13174 collision awaiting manual-renumber exception.

Qodo review repaired eager pack loading: only displayed assets are fetched, cache retains at most8 blobs/16MiB (single larger current frame permitted), old URLs revoke on eviction, and requests abort on frame/source/unmount changes. Red2 regressions reproduced256/unneeded downloads;9 auth renderer tests now pass including count/byte eviction, reuse and cancellation. Combined review scope106 frontend tests and54 backend tests pass; scoped ESLint0errors/2native-image warnings. Quickstart AC1 remains open under TASK-13181. Setup task collision resolved as TASK-13182; PR2884 is ready with requester-authored summary.

CI follow-up in progress: diagnose VisualPackEditor package-owned Vitest failures under Node20 and the deterministic workflow config. Initial focused UI-package run on local Node26 passes64 tests; compare runtime/config and asynchronous readiness before choosing a correction.

CI frontend shard4 follow-up: package-owned VisualPackEditor tests raced manifest initialization and review completion under Node20. A pack status/selector or builder container can exist before its manifest-derived controls populate; observing a POST does not mean AntD loading buttons are ready for the next click. Tests now await actual idle mappings/custom-state options, settled management attention, and accepted/rejected confirmations. Original manifest content, saved edits, queued payload, and review request assertions remain intact; no timeout increase and no production change.

Reproduction: local Node26 passed64 tests; using PATH=/Users/macbook-dev/.nvm/versions/node/v20.19.5/bin:$PATH with CI=true NODE_OPTIONS=--max-old-space-size=4096 SKIP_WXT_PREPARE=1 and bun x vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --config .tldw-vitest-ratchet.config.ts --maxWorkers=1 --testTimeout=15000 --passWithNoTests from apps/packages/ui reproduced4 failed/60 passed. The temporary config was copied exactly from Helper_Scripts/ci/vitest_deterministic.config.ts. After test synchronization, the same command passed64/64. Original four-file shard context plus both SpriteFrameRenderer suites passed93/93 across6 files under the same Node20/config/environment. Scoped ESLint0errors/47 existing no-explicit-any warnings; git diff --check clean. Temporary CI config removed after verification. Bandit not applicable to TSX-test-only changes. Task remains In Progress because quickstart image AC1 still depends on TASK-13181; no full suite, commit or push performed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Advanced-mode Migu images and bounded authenticated loading repaired. Quickstart image acceptance remains pending its cookie-authentication dependency failure; no end-to-end quickstart pass claimed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
