---
id: TASK-2351
title: Implement Audio Studio migration and compatibility
status: Done
labels:
- audio
- migration
priority: high
documentation:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- Docs/superpowers/specs/2026-06-23-audio-studio-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement render/export services, Audiobook/Dexie migration APIs and UI, /audiobook-studio compatibility behavior, documentation, E2E coverage, and final focused verification for the Audio Studio MVP.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Render and export jobs produce distinct Audio Studio artifacts with provenance and revision validation.
- [x] #2 Legacy Audiobook/Dexie migration preview and commit work without deleting local data before successful commit.
- [x] #3 Docs, E2E, audiobook compatibility tests, focused backend/frontend verification, and Bandit results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 3, Stage 4.5, and Stage 5 in Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md after backend jobs/providers and frontend route are available. Use explicit file staging from the plan and record all verification results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added backend render/export artifact recording with job-scoped artifact IDs, target revision validation, stale-job skip handling, generation artifact file persistence, provider error secret redaction, and workflow filtering for project listing.
- Kept legacy Audiobook migration metadata-only in MVP by preserving legacy audio upload references without creating renderable artifacts for blobs that are not yet imported server-side.
- Migrated the compatibility redirect to the singular backend migration contract, preserving local Dexie data until all commit responses succeed.
- Added first-class Narration, Podcast, Briefing, and Music route/workflow coverage. Podcast and Briefing now persist draft sections before queueing speech generation.
- Hardened Audio Studio tabs with ARIA tab IDs, panel linkage, and keyboard navigation.
- Updated docs/config to describe environment-only Audio Studio provider configuration, strict allowlist/secret handling, synchronous migration commit behavior, and the MVP limitation that local Dexie blobs are not copied into server render/export artifacts yet.
- Post-review fixes addressed reviewer findings around renderability, artifact ID collisions, stale jobs, migration contract mismatch, inert config documentation, secret leakage, API docs drift, E2E migration coverage, Podcast/Briefing persistence, route inventory mapping, and tab accessibility.
- Verification on 2026-06-23:
  - `.venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio tldw_Server_API/tests/Audiobooks -q` -> 220 passed, 11 warnings.
  - `bunx vitest run ../packages/ui/src/db/dexie/__tests__/audiobook-migration.test.ts ../packages/ui/src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx ../packages/ui/src/services/__tests__/audio-studio.test.ts __tests__/pages/audio-studio-route.test.tsx` -> 5 files passed, 33 tests.
  - `bunx playwright test e2e/workflows/tier-2-features/audio-studio.spec.ts e2e/workflows/tier-2-features/audiobook-studio.spec.ts --reporter=line` -> 5 passed.
  - Shared UI ESLint for touched Audio Studio/Dexie/service files -> 0 errors, 0 warnings; existing Next pages-directory notice only.
  - Frontend E2E ESLint for touched page objects/spec mapping -> 0 errors, 0 warnings.
  - `.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py tldw_Server_API/app/core/Audio_Studio tldw_Server_API/app/core/DB_Management/Collections_DB.py -f json -o /tmp/bandit_audio_studio.json` -> 0 findings.
  - `git diff --check` on touched task paths -> clean.
  - `rg -n "AUDIO_STUDIO_ACE_STEP_API_KEY=.*[A-Za-z0-9_-]{16,}" Docs tldw_Server_API/Config_Files` -> no populated secret examples.
- Known baseline: `bunx tsc --noEmit` still fails in unrelated files (`TaskActivityNotice`, Evaluations embeddings config, WritingPlayground, persona visuals, VN play API, and calendar route import). No Audio Studio paths appeared in the typecheck errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Audio Studio migration/compatibility slice with durable generated artifacts, collision-safe render/export outputs, a safer legacy Audiobook migration path, canonical `/audio-studio` plus legacy `/audiobook-studio` browser coverage, first-class Narration/Podcast/Briefing/Music UI workflows, and updated provider/security documentation. Focused backend, frontend, browser, lint, Bandit, whitespace, and secret-scan verification is recorded above; the only remaining typecheck failure is the pre-existing unrelated frontend baseline.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
