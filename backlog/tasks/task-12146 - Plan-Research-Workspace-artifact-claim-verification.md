---
id: TASK-12146
title: Plan Research Workspace artifact claim verification
status: In Progress
labels:
- research-workspace
- claims
- planning
references:
- 'GitHub issue #2605'
- 'PR #2633'
documentation:
- Docs/superpowers/specs/2026-07-04-research-workspace-artifact-claim-verification-design.md
- Docs/superpowers/plans/2026-07-04-research-workspace-artifact-claim-verification-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-07-04-research-workspace-artifact-claim-verification-design.md
- Docs/superpowers/plans/2026-07-04-research-workspace-artifact-claim-verification-implementation-plan.md
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/TraceableArtifactDetail.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/TraceableArtifactDetail.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
- apps/packages/ui/src/services/flashcards.ts
- apps/packages/ui/src/services/quizzes.ts
- apps/packages/ui/src/services/researchWorkspaceArtifacts.ts
- apps/packages/ui/src/services/tldw/domains/presentations.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/types/workspace.ts
- tldw_Server_API/app/api/v1/endpoints/flashcards.py
- tldw_Server_API/app/api/v1/endpoints/quizzes.py
- tldw_Server_API/app/api/v1/endpoints/research_workspace.py
- tldw_Server_API/app/api/v1/endpoints/slides.py
- tldw_Server_API/app/api/v1/schemas/flashcards.py
- tldw_Server_API/app/api/v1/schemas/quizzes.py
- tldw_Server_API/app/api/v1/schemas/research_workspace_artifacts.py
- tldw_Server_API/app/api/v1/schemas/slides_schemas.py
- tldw_Server_API/app/core/Claims_Extraction/artifact_verification.py
- tldw_Server_API/app/core/Research_Workspace/artifact_generation.py
- tldw_Server_API/app/services/quiz_generator.py
- tldw_Server_API/tests/Claims/test_artifact_verification.py
- tldw_Server_API/tests/Claims/test_artifact_verification_properties.py
- tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py
- tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py
- tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py
- tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_service.py
- tldw_Server_API/tests/Slides/test_slides_api.py
- tldw_Server_API/tests/Slides/test_slides_endpoint_sanitization.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the internal claims-based factuality gate for Research Workspace generated artifacts before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec and implementation plan are written. Next implementation should proceed stage-by-stage: internal Claims artifact verification helper, provider/model propagation, backend-generated artifact gates, data-table persistence gate, backend migration for remaining Research Workspace generators, frontend display/review state, and final verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the internal Claims artifact verification helper with generation-bound provider/model defaults plus explicit claims-verifier provider/model overrides. Wired the flashcards generate endpoint to verify generated card content internally, return `claim_verification` on clean responses, and return a structured 422 on failed verification. Added Research Workspace Studio controls for claims verifier provider/model, a visible notice when an override is active, flashcard request forwarding, and artifact provenance/data storage for returned verification reports.

Verification so far:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_artifact_verification.py tldw_Server_API/tests/Claims/test_artifact_verification_properties.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_generate_flashcards_endpoint_returns_generated_cards tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_generate_flashcards_endpoint_uses_default_provider_when_omitted tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_generate_flashcards_endpoint_returns_422_when_claim_verification_fails tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_generate_flashcards_endpoint_uses_claims_verification_provider_override -q` passed.
- `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx -t "exposes accessible names for studio option controls|uses structured flashcard generation with one scoped deck and bulk saves"` passed.
- `bun run typecheck` passed.
Extended the implementation from the initial flashcards path to the full Research Workspace target set. Added an internal ClaimsEngine artifact verifier with generation-model defaults, optional verifier provider/model overrides, unit/claim/text caps, and non-grounded rejection. Wired flashcards, quizzes, slides, and the new Research Workspace artifact draft service for audio summaries, data tables, and mindmaps so verification runs inside backend generation instead of calling public Claims HTTP endpoints. Added Studio controls and artifact metadata display so non-default claims verifier use is visible to users.

Fresh verification:
- `python -m pytest tldw_Server_API/tests/Claims/test_artifact_verification.py tldw_Server_API/tests/Claims/test_artifact_verification_properties.py -q` -> 15 passed.
- `python -m pytest tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_service.py tldw_Server_API/tests/Slides/test_slides_api.py tldw_Server_API/tests/Slides/test_slides_endpoint_sanitization.py -q` -> 85 passed.
- `python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py -q` -> 170 passed.
- `bun run typecheck` -> passed.
- `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/TraceableArtifactDetail.test.tsx --reporter=dot` -> 112 passed.
- `git diff --check` -> passed.
- `python -m bandit -r <touched backend source paths> -f json -o /tmp/bandit_research_workspace_claims.json` -> 0 results.

PR: #2633 against `dev`. Note: user-owned Change summary is still required by the AI-generated PR merge gate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Research Workspace artifact factuality gate across the target generated assets. Flashcards, quizzes, slides, audio summaries, data tables, and mindmaps now route through backend generation paths that invoke the internal ClaimsEngine artifact verifier before a completed artifact is returned or persisted. The verifier defaults to the generation provider/model, supports explicit claims-verification provider/model overrides, records verifier metadata, applies unit/claim/text caps, and rejects non-grounded outputs as structured generation errors.

The full app Studio pane now exposes claims verifier provider/model controls, shows an override notice when the verifier differs from the generation model, forwards the verifier configuration to backend generation, and displays verifier metadata on completed artifacts. Verification evidence is recorded in the implementation notes. Remaining merge-gate note: PR #2633 still requires the human-owned Change summary required by the AI-generated PR policy.
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
