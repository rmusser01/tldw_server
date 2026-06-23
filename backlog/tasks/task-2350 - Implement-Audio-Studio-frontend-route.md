---
id: TASK-2350
title: Implement Audio Studio frontend route
status: Done
labels:
- audio
- webui
priority: high
documentation:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- Docs/superpowers/specs/2026-06-23-audio-studio-design.md
modified_files:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- apps/packages/ui/src/services/audio-studio.ts
- apps/packages/ui/src/services/__tests__/audio-studio.test.ts
- apps/packages/ui/src/store/audio-studio.tsx
- apps/packages/ui/src/store/__tests__/audio-studio.test.tsx
- apps/packages/ui/src/hooks/useAudioStudioProjects.ts
- apps/packages/ui/src/hooks/useAudioStudioGeneration.tsx
- apps/packages/ui/src/hooks/useAudioStudioMigration.ts
- apps/packages/ui/src/hooks/__tests__/useAudioStudioProjects.test.tsx
- apps/packages/ui/src/components/Option/AudioStudio/AudioStudioPage.tsx
- apps/packages/ui/src/components/Option/AudioStudio/WorkflowSwitcher.tsx
- apps/packages/ui/src/components/Option/AudioStudio/ProjectSidebar.tsx
- apps/packages/ui/src/components/Option/AudioStudio/ProjectHeader.tsx
- apps/packages/ui/src/components/Option/AudioStudio/NarrationWorkflow.tsx
- apps/packages/ui/src/components/Option/AudioStudio/PodcastWorkflow.tsx
- apps/packages/ui/src/components/Option/AudioStudio/BriefingWorkflow.tsx
- apps/packages/ui/src/components/Option/AudioStudio/MusicWorkflow.tsx
- apps/packages/ui/src/components/Option/AudioStudio/GenerationPanel.tsx
- apps/packages/ui/src/components/Option/AudioStudio/generationPayload.ts
- apps/packages/ui/src/components/Option/AudioStudio/RenderExportPanel.tsx
- apps/packages/ui/src/components/Option/AudioStudio/MigrationBanner.tsx
- apps/packages/ui/src/components/Option/AudioStudio/CompatibilityRedirect.tsx
- apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx
- apps/packages/ui/src/routes/option-audio-studio.tsx
- apps/packages/ui/src/routes/option-audiobook-studio.tsx
- apps/packages/ui/src/routes/route-registry.tsx
- apps/packages/ui/src/routes/route-metadata.ts
- apps/packages/ui/src/routes/route-paths.ts
- apps/packages/ui/src/routes/app-route.tsx
- apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts
- apps/packages/ui/src/components/Layouts/header-shortcut-items.ts
- apps/packages/ui/src/components/Layouts/ModeSelector.tsx
- apps/packages/ui/src/assets/locale/en/option.json
- apps/packages/ui/src/public/_locales/en/option.json
- apps/tldw-frontend/pages/audio-studio.tsx
- apps/tldw-frontend/__tests__/pages/audio-studio-route.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the shared UI and Next.js route for /audio-studio, with Narration, Podcast, Briefing, and Music as first-class workflows and Narration reusing the existing Audiobook Studio experience as its base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared Audio Studio service, store, hooks, and components are implemented and tested.
- [x] #2 /audio-studio is registered in shared routes, route metadata, navigation, localization, and Next.js pages.
- [x] #3 Narration, Podcast, Briefing, and Music workflows are visibly first-class; /audiobook-studio remains a compatibility route.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 4 tasks 4.1 through 4.4 in Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md after backend API shape is available. Use targeted Vitest route/component/service tests and preserve existing Audiobook Studio compatibility.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage 4.1-4.4 only. Added typed Audio Studio service calls over shared `bgRequest` helpers; added Zustand store with first-class Narration, Podcast, Briefing, and Music workflow definitions and dirty-project revision conflict tracking; added React Query hooks for project load/create plus typed generation/render/export/migration endpoint wrappers.

Added dense Audio Studio route components with Narration reusing existing Audiobook `TextEditor`, `ChapterList`, `GenerationPanel`, and `OutputPanel` controls. Added Podcast, Briefing, and Music workflow panels, render/export placeholder panel for TASK-2351 scope, and compatibility redirect from `/audiobook-studio` to `/audio-studio?workflow=narration`.

Registered `/audio-studio` in shared route registry, route metadata, route constants, route bootstrap namespaces, header shortcut target/label, mode selector label, English locale files, and Next.js page shim.

Verification:
- `bunx vitest run ../packages/ui/src/services/__tests__/audio-studio.test.ts` - 5 passed.
- `bunx vitest run ../packages/ui/src/store/__tests__/audio-studio.test.tsx ../packages/ui/src/hooks/__tests__/useAudioStudioProjects.test.tsx` - 5 passed.
- `bunx vitest run ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx` - 4 passed.
- `bunx vitest run ../packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts __tests__/pages/audio-studio-route.test.tsx` - 10 passed.
- `git diff --check` on TASK-2350 paths passed.

Bandit: not applicable; frontend-only TypeScript/React route work, no Python/backend files touched.

Known deferrals by scope: Dexie migration UI and render/export services remain for TASK-2351.

Spec review follow-up: wired previously inert Audio Studio generation controls. MusicWorkflow now keeps prompt, lyrics, style, provider, and duration as controlled state and submits a backend-shaped music generation request through `useCreateAudioStudioGeneration(projectId)`. The shared GenerationPanel now queues music jobs against a track target and speech jobs against the first section target when the active project has a usable revision, with disabled states when the required target is unavailable. Updated generation service typing to match the backend `AudioStudioGenerationCreate` contract (`kind`, provider, target resource, target revision, idempotency key, and options).

Follow-up verification:
- `bunx vitest run ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx` - 8 passed.
- `bunx vitest run ../packages/ui/src/services/__tests__/audio-studio.test.ts ../packages/ui/src/store/__tests__/audio-studio.test.tsx ../packages/ui/src/hooks/__tests__/useAudioStudioProjects.test.tsx ../packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts __tests__/pages/audio-studio-route.test.tsx` - 20 passed.
- `git diff --check` on TASK-2350 follow-up paths passed.
- Bandit remains not applicable; frontend-only TypeScript/React changes, no Python/backend code touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audio Studio frontend route is implemented for Stage 4.1-4.4. /audio-studio is canonical and supports workflow query selection; /audiobook-studio remains a legacy compatibility route to Narration. Narration reuses the existing Audiobook Studio controls, while Podcast, Briefing, and Music are visible first-class workflows. Dexie migration UI and render/export implementation remain deferred to TASK-2351 by scope.
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
