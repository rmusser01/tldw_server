# WebUI Study Safety Specialized Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make study, safety, review, data, chunking, kanban, and VN routes clearly classified, route-owned, and recoverable without redesigning unrelated product surfaces.

**Architecture:** Add a route-job contract for the Task 11B route family, then implement route identity, readiness, mode framing, alias behavior, and labs classification in separate sub-slices. Keep each route on its current product component and use existing route wrappers, connection gates, capability probes, tabs, stores, and route-specific tests before introducing any new shared helper.

**Tech Stack:** React, Next.js pages, shared `apps/packages/ui` route shells, extension route wrappers, TanStack Query, existing study and evaluation hooks, existing VN workbench components, Vitest, React Testing Library, Playwright.

---

## Source Documents

- Backlog task: `TASK-418.9`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- UX remediation spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Dependency plans:
  - `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`

## Audit Findings Addressed

- `F2 support`: Route purpose, mode, and primary workflow are unclear on overlapping study, review, and labs routes.
- `F9 support`: Missing, unsupported, unavailable, unauthorized, degraded, and partial states are inconsistent across advanced routes.
- `F15 support`: Advanced controls need discoverability without removing power-user density.
- `F18 support`: Hosted, beta, labs, specialized, and internal surfaces need explicit visibility policy.
- `F19`: Deprecated UI cleanup is in scope only when it blocks this route-family UX work.

## Scope Split

Task 11B must not become one implementation PR. Use this plan as an umbrella and split implementation into reviewable sub-slices:

1. Route contract and classification.
2. Evaluations readiness.
3. Flashcards and quiz study modes.
4. Moderation, content review, and claims alias.
5. Data tables and chunking advanced tools.
6. Kanban and VN specialized routes.
7. Final browser and smoke verification.

Each sub-slice needs its own Backlog child task before product code changes.

## Route Inventory And Ownership

| Route | Primary user goal | Current ownership | Main workflows | UX contract |
| --- | --- | --- | --- | --- |
| `/evaluations` | Define eval recipes, inspect runs, manage datasets, review synthetic items, and view history | Shared and extension `option-evaluations.tsx`, `EvaluationsPlaygroundPage` re-exporting `EvaluationsPage` | Recipes, Review, Evaluations, Runs, Datasets, Webhooks, History | Treat as an advanced evaluation workbench. Add route boundary parity and worker/capability readiness without hiding tabs. |
| `/flashcards` | Study, manage, generate, import, export, template, and schedule spaced-repetition cards | Shared and extension `option-flashcards.tsx`, `FlashcardsWorkspace`, `FlashcardsManager` | Study, Manage, Import / Export, Templates, Scheduler | Treat as a study workspace with first-use setup and expert study controls. Preserve sidepanel handoff. |
| `/quiz` | Take, generate, create, manage, and review quiz results | Shared and extension `option-quiz.tsx`, `QuizWorkspace`, `QuizPlayground` | Take Quiz, Generate, Create, Manage, Results | Treat as a study assessment workspace. Keep demo mode and degraded setup states clear. |
| `/moderation-playground` | Configure and test content safety controls | Shared and extension `option-moderation-playground.tsx`, `ModerationPlaygroundShell` | Policy and Settings, Blocklist Studio, User Overrides, Test Sandbox, Advanced | Treat as a safety administration route, not generic entertainment. Keep admin and offline states visible. |
| `/content-review` | Review and commit drafts created before saving media | Shared and extension `option-content-review.tsx`, `ContentReviewPage` | Batch selection, edit, diff, metadata, AI fix, commit, mark reviewed, skip, discard | Treat as a review queue. Keep empty state tied to Quick Ingest review mode. |
| `/claims-review` | Reach the review queue from a legacy or claims-specific route | `apps/tldw-frontend/pages/claims-review.tsx` redirects to `/content-review` | Alias route only | Keep alias intentional. Test redirect behavior and route metadata; do not create a second claims queue unless a separate product decision exists. |
| `/data-tables` | Generate, save, preview, edit, and export structured tables | Shared and extension `option-data-tables.tsx`, `DataTablesPage` | My Tables, Create Table, source selection, generated preview, save, export | Treat as a beta structured-data workbench with backend readiness and generated-output recovery. |
| `/chunking-playground` | Tune and compare chunking strategies | Shared and extension `option-chunking-playground.tsx`, `ChunkingPlayground` | Single, Compare, Templates, Capabilities, PDF or text chunking, save template | Treat as an advanced RAG tuning surface. Keep capabilities visible and classify out of default novice flows. |
| `/kanban` | Manage local project boards and cards | Shared and extension `option-kanban-playground.tsx`, `KanbanPlayground` | Board gallery, create board, import, export, cards, due dates, labels, archive | Decide whether this is production planning or labs. Route label and visibility must match that decision. |
| `/vn-assets` | Prepare visual-novel asset packs | Next page dynamic import of `VNAssetsWorkbench` | Pack setup, matrix, readiness, generation monitor, review board, portability | Treat as specialized VN tooling with readiness and generation status. It is not part of default research navigation unless WP1 classifies it there. |
| `/vn-play` | Run VN play sessions and inspect runtime state | Next page dynamic import of `VNPlayWorkspace` | New Freeform, New Story, sessions, scene, dialogue, choices, checkpoints, branches, retry | Treat as specialized runtime tooling with session recovery. It is not a generic chat route. |

## Frontend-Only Versus Backend-Gated Work

### Frontend-Only Work

Use frontend-only changes when the route can already derive state from:

- Existing route registries and Next pages.
- `WorkspaceConnectionGate`.
- `useConnectionUxState`.
- `useServerCapabilities`.
- Existing evaluation hooks and tab queries.
- Existing flashcard and quiz connection-state workspaces.
- Existing moderation settings, blocklist, overrides, and test hooks.
- Existing content review local draft state.
- Existing data table store, prefill, and generation state.
- Existing chunking capability and template queries.
- Existing kanban local board state.
- Existing VN asset readiness, generation, review, and portability state.
- Existing VN play session, checkpoint, branch, retry, and recovery state.

Frontend-only changes include:

- Route labels, headings, and page landmarks.
- Empty, loading, unavailable, unsupported, unauthorized, not-configured, degraded, partial, and error states.
- Route boundaries and accessible loading states.
- Mode and tab labels when the route already owns that mode.
- Local alias tests.
- Route metadata tests.
- Browser QA and Playwright assertions.

### Backend-Gated Work

Create a separate backend contract task before implementation if a route needs state that is not exposed by current frontend inputs.

Backend-gated examples:

- New evaluation worker health endpoints.
- New quiz generation algorithms.
- New flashcard scheduler APIs.
- New moderation permission APIs.
- New content-review queue APIs beyond current local draft or ingest state.
- New data-table generation endpoints.
- New chunking strategy APIs.
- New kanban persistence APIs.
- VN runtime redesign or new generation orchestration APIs.

Do not add backend API changes inside a Task 11B implementation PR unless the Backlog task explicitly broadens scope and this plan is updated first.

## Non-Goals

- Do not redesign every advanced route visually.
- Do not create a new study system.
- Do not add quiz algorithms.
- Do not add flashcard scheduling models.
- Do not add moderation backend policy features.
- Do not build a new content review queue.
- Do not create a second claims-review workflow.
- Do not replace the VN runtime.
- Do not remove advanced controls that returning users rely on.
- Do not hide labs routes without preserving explicit route metadata and smoke coverage.
- Do not change backend APIs in this planning slice.

## File Structure

### New Files

- `apps/packages/ui/src/routes/study-safety-specialized-route-jobs.ts`
  - Owns route labels, product classification, primary jobs, modes, route ownership, capability inputs, and alias policy for Task 11B.
- `apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts`
  - Verifies route coverage, finding coverage, route classification, alias policy, and implementation ownership.
- `apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx`
  - Verifies route wrappers use `OptionLayout` or the intended Next-page owner, route boundaries, labels, and canonical components.
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/kanban-visibility.spec.ts`
  - Add only if existing `kanban.spec.ts` does not cover labs or production classification.

### Modified Route Files

- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`
- `apps/packages/ui/src/routes/option-evaluations.tsx`
- `apps/tldw-frontend/extension/routes/option-evaluations.tsx`
- `apps/packages/ui/src/routes/option-flashcards.tsx`
- `apps/tldw-frontend/extension/routes/option-flashcards.tsx`
- `apps/packages/ui/src/routes/option-quiz.tsx`
- `apps/tldw-frontend/extension/routes/option-quiz.tsx`
- `apps/packages/ui/src/routes/option-moderation-playground.tsx`
- `apps/tldw-frontend/extension/routes/option-moderation-playground.tsx`
- `apps/packages/ui/src/routes/option-content-review.tsx`
- `apps/tldw-frontend/extension/routes/option-content-review.tsx`
- `apps/tldw-frontend/pages/claims-review.tsx`
- `apps/packages/ui/src/routes/option-data-tables.tsx`
- `apps/tldw-frontend/extension/routes/option-data-tables.tsx`
- `apps/packages/ui/src/routes/option-chunking-playground.tsx`
- `apps/tldw-frontend/extension/routes/option-chunking-playground.tsx`
- `apps/packages/ui/src/routes/option-kanban-playground.tsx`
- `apps/tldw-frontend/extension/routes/option-kanban-playground.tsx`
- `apps/tldw-frontend/pages/vn-assets.tsx`
- `apps/tldw-frontend/pages/vn-play.tsx`

### Modified Component Files

- `apps/packages/ui/src/components/Option/Evaluations/EvaluationsPage.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/SyntheticReviewTab.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/EvaluationsTab.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/RunsTab.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/DatasetsTab.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/WebhooksTab.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/HistoryTab.tsx`
- `apps/packages/ui/src/components/Flashcards/FlashcardsWorkspace.tsx`
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx`
- `apps/packages/ui/src/components/Quiz/QuizWorkspace.tsx`
- `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/CreateTab.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/PolicySettingsPanel.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/BlocklistStudioPanel.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/UserOverridesPanel.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/TestSandboxPanel.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/AdvancedPanel.tsx`
- `apps/packages/ui/src/components/ContentReview/ContentReviewPage.tsx`
- `apps/packages/ui/src/components/Option/DataTables/DataTablesPage.tsx`
- `apps/packages/ui/src/components/Option/DataTables/CreateTableWizard.tsx`
- `apps/packages/ui/src/components/Option/DataTables/DataTablesList.tsx`
- `apps/packages/ui/src/components/Option/ChunkingPlayground/index.tsx`
- `apps/packages/ui/src/components/Option/ChunkingPlayground/ChunkingCapabilitiesPanel.tsx`
- `apps/packages/ui/src/components/Option/KanbanPlayground/index.tsx`
- `apps/tldw-frontend/components/vn-assets/VNAssetsWorkbench.tsx`
- `apps/tldw-frontend/components/vn-assets/ReadinessPanel.tsx`
- `apps/tldw-frontend/components/vn-assets/GenerationMonitor.tsx`
- `apps/tldw-frontend/components/vn-assets/ReviewBoard.tsx`
- `apps/tldw-frontend/components/vn-assets/PortabilityPanel.tsx`
- `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- `apps/tldw-frontend/components/vn-play/SessionList.tsx`
- `apps/tldw-frontend/components/vn-play/SceneStage.tsx`
- `apps/tldw-frontend/components/vn-play/DialoguePanel.tsx`
- `apps/tldw-frontend/components/vn-play/SceneInspector.tsx`

### Existing Tests To Extend

- `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`
- `apps/packages/ui/src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/EvaluationsTab.empty-state.test.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RunsTab.benchmark-option.test.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/DatasetsTab.pagination.test.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/HistoryTab.filters.test.tsx`
- `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx`
- `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx`
- `apps/packages/ui/src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx`
- `apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx`
- `apps/packages/ui/src/components/Quiz/__tests__/quiz-ftux.test.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/__tests__/CreateTab.save-progress.test.tsx`
- `apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.filters-retake.test.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.connection.test.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.test.tsx`
- `apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlayground.progressive-disclosure.test.tsx`
- `apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesPage.golden-path.test.tsx`
- `apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesList.a11y.test.tsx`
- `apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.golden-path.test.tsx`
- `apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.responsive-layout.test.tsx`
- `apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/KanbanPlayground.empty-state.test.tsx`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts`
- `apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts`
- `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`

## Route Job Contract

Create a route metadata file that keeps this mixed route family explicit:

```ts
export type SpecializedRouteConcept =
  | "evaluation"
  | "study_flashcards"
  | "study_quiz"
  | "safety"
  | "review_queue"
  | "legacy_alias"
  | "structured_data"
  | "rag_tuning"
  | "planning_board"
  | "vn_assets"
  | "vn_runtime"

export type SpecializedRouteClassification =
  | "advanced_self_hosted"
  | "study_workspace"
  | "operator_safety"
  | "review_workflow"
  | "beta_tool"
  | "labs_tool"
  | "legacy_alias"

export type SpecializedRouteJob = {
  route: string
  concept: SpecializedRouteConcept
  classification: SpecializedRouteClassification
  label: string
  primaryJob: string
  primaryActionLabel: string
  routeOwner: "shared_route" | "extension_route" | "next_page" | "next_alias"
  canonicalComponent: string
  modes: string[]
  findings: Array<"F2 support" | "F9 support" | "F15 support" | "F18 support" | "F19">
  visibilityDecision: "default_nav" | "advanced_nav" | "labs_nav" | "alias_only"
}
```

Initial inventory:

```ts
export const STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS: SpecializedRouteJob[] = [
  {
    route: "/evaluations",
    concept: "evaluation",
    classification: "advanced_self_hosted",
    label: "Evaluations",
    primaryJob: "Define and inspect evaluation recipes, runs, datasets, webhooks, and history.",
    primaryActionLabel: "Create evaluation",
    routeOwner: "shared_route",
    canonicalComponent: "EvaluationsPage",
    modes: ["Recipes", "Review", "Evaluations", "Runs", "Datasets", "Webhooks", "History"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "advanced_nav"
  },
  {
    route: "/flashcards",
    concept: "study_flashcards",
    classification: "study_workspace",
    label: "Flashcards",
    primaryJob: "Study, manage, import, export, template, and schedule flashcards.",
    primaryActionLabel: "Start studying",
    routeOwner: "shared_route",
    canonicalComponent: "FlashcardsWorkspace",
    modes: ["Study", "Manage", "Import / Export", "Templates", "Scheduler"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "default_nav"
  },
  {
    route: "/quiz",
    concept: "study_quiz",
    classification: "study_workspace",
    label: "Quiz",
    primaryJob: "Take, generate, create, manage, and review quiz results.",
    primaryActionLabel: "Take quiz",
    routeOwner: "shared_route",
    canonicalComponent: "QuizWorkspace",
    modes: ["Take Quiz", "Generate", "Create", "Manage", "Results"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "default_nav"
  },
  {
    route: "/moderation-playground",
    concept: "safety",
    classification: "operator_safety",
    label: "Moderation Playground",
    primaryJob: "Configure and test content safety policy.",
    primaryActionLabel: "Test policy",
    routeOwner: "shared_route",
    canonicalComponent: "ModerationPlayground",
    modes: ["Policy and Settings", "Blocklist Studio", "User Overrides", "Test Sandbox", "Advanced"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "advanced_nav"
  },
  {
    route: "/content-review",
    concept: "review_queue",
    classification: "review_workflow",
    label: "Content Review",
    primaryJob: "Review and commit drafts created before saving content.",
    primaryActionLabel: "Open draft",
    routeOwner: "shared_route",
    canonicalComponent: "ContentReviewPage",
    modes: ["Batch", "Drafts", "Edit", "Diff", "Metadata", "Actions"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "advanced_nav"
  },
  {
    route: "/claims-review",
    concept: "legacy_alias",
    classification: "legacy_alias",
    label: "Claims Review",
    primaryJob: "Redirect to the canonical content review queue.",
    primaryActionLabel: "Open Content Review",
    routeOwner: "next_alias",
    canonicalComponent: "RouteRedirect:/content-review",
    modes: [],
    findings: ["F2 support", "F18 support"],
    visibilityDecision: "alias_only"
  },
  {
    route: "/data-tables",
    concept: "structured_data",
    classification: "beta_tool",
    label: "Data Tables",
    primaryJob: "Generate, save, preview, edit, and export structured tables.",
    primaryActionLabel: "Create table",
    routeOwner: "shared_route",
    canonicalComponent: "DataTablesPage",
    modes: ["My Tables", "Create Table"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "advanced_nav"
  },
  {
    route: "/chunking-playground",
    concept: "rag_tuning",
    classification: "advanced_self_hosted",
    label: "Chunking Playground",
    primaryJob: "Tune and compare chunking strategies.",
    primaryActionLabel: "Run chunking",
    routeOwner: "shared_route",
    canonicalComponent: "ChunkingPlayground",
    modes: ["Single", "Compare", "Templates", "Capabilities"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "advanced_nav"
  },
  {
    route: "/kanban",
    concept: "planning_board",
    classification: "labs_tool",
    label: "Kanban",
    primaryJob: "Manage boards, cards, labels, due dates, imports, and exports.",
    primaryActionLabel: "Create board",
    routeOwner: "shared_route",
    canonicalComponent: "KanbanPlayground",
    modes: ["Boards", "Cards", "Import", "Export", "Archive"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "labs_nav"
  },
  {
    route: "/vn-assets",
    concept: "vn_assets",
    classification: "labs_tool",
    label: "VN Assets",
    primaryJob: "Prepare VN asset packs and review generated variants.",
    primaryActionLabel: "Create pack",
    routeOwner: "next_page",
    canonicalComponent: "VNAssetsWorkbench",
    modes: ["Setup", "Matrix", "Generate", "Review", "Portability"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "labs_nav"
  },
  {
    route: "/vn-play",
    concept: "vn_runtime",
    classification: "labs_tool",
    label: "VN Play",
    primaryJob: "Run VN play sessions and inspect runtime state.",
    primaryActionLabel: "New session",
    routeOwner: "next_page",
    canonicalComponent: "VNPlayWorkspace",
    modes: ["Sessions", "Scene", "Dialogue", "Choices", "Inspector", "Checkpoints"],
    findings: ["F2 support", "F9 support", "F15 support", "F18 support", "F19"],
    visibilityDecision: "labs_nav"
  }
]
```

## Route State Vocabulary

Use the WP2 shared states. Do not invent route-specific state names for equivalent conditions.

| State | Task 11B meaning | Required UI behavior |
| --- | --- | --- |
| `loading` | Queries for evals, decks, quizzes, drafts, capabilities, boards, or VN sessions are in flight | Preserve route landmark and mode controls. Use role status or existing loading affordance. |
| `ready` | The route can perform its primary action | Enable primary action and show current route mode or selected object state. |
| `not_configured` | Server, worker, provider, API key, source object, deck, quiz, board, pack, or session is missing | Disable unsafe action, preserve input, and point to setup or creation action. |
| `unsupported` | Current deployment or server version does not expose the feature | Use shared unsupported state and keep raw endpoints behind diagnostics. |
| `unavailable` | Expected endpoint or connection is unreachable | Show retry, setup, and diagnostics disclosure. |
| `unauthorized` | Route requires admin or a permission not held by the current user | Block mutating controls and identify required permission in user language. |
| `degraded` | Route can still work with reduced worker, provider, template, or capability support | Keep available workflows enabled and label unavailable ones. |
| `partial` | Some imported/generated/evaluated/reviewed items succeeded and others failed | Preserve completed work, mark failed items, and offer item-level retry. |
| `error` | User action failed | Preserve user input and selection, explain what failed, and offer retry or recovery. |

## Implementation Tasks

### Task 1: Lock Study, Safety, And Specialized Route Contract

**Files:**
- Create: `apps/packages/ui/src/routes/study-safety-specialized-route-jobs.ts`
- Create: `apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts`
- Create: `apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`

- [ ] **Step 1: Write the failing route-job coverage test**

Create `apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS } from "../study-safety-specialized-route-jobs"

const routes = [
  "/evaluations",
  "/flashcards",
  "/quiz",
  "/moderation-playground",
  "/content-review",
  "/claims-review",
  "/data-tables",
  "/chunking-playground",
  "/kanban",
  "/vn-assets",
  "/vn-play"
] as const

const findings = ["F2 support", "F9 support", "F15 support", "F18 support", "F19"] as const

describe("study, safety, and specialized route jobs", () => {
  it("covers every Task 11B route once", () => {
    expect(STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(
      Array.from(routes).sort()
    )
  })

  it("keeps labels, jobs, and classifications usable", () => {
    for (const job of STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS) {
      expect(job.label).not.toHaveLength(0)
      expect(job.primaryJob).not.toHaveLength(0)
      expect(job.primaryActionLabel).not.toHaveLength(0)
      expect(job.classification).not.toHaveLength(0)
      expect(job.canonicalComponent).not.toHaveLength(0)
    }
  })

  it("maps all Task 11B audit findings", () => {
    const covered = new Set(STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS.flatMap((job) => job.findings))
    for (const finding of findings) {
      expect(covered.has(finding)).toBe(true)
    }
  })
})
```

- [ ] **Step 2: Run the route-job test to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts
```

Expected: FAIL because `study-safety-specialized-route-jobs.ts` does not exist.

- [ ] **Step 3: Add route-job metadata**

Create `apps/packages/ui/src/routes/study-safety-specialized-route-jobs.ts` with the contract above. Keep it pure data with no React imports.

- [ ] **Step 4: Add route-boundary ownership tests**

Create `study-safety-specialized-route-boundaries.test.tsx` with mocks for the owned page components. Assert:

- `/evaluations` uses `EvaluationsPlaygroundPage` and gains a route boundary in Task 2.
- `/flashcards` uses `FlashcardsWorkspace`.
- `/quiz` uses `QuizWorkspace`.
- `/moderation-playground` uses `ModerationPlayground`.
- `/content-review` uses `ContentReviewPage`.
- `/data-tables` uses `DataTablesPage`.
- `/chunking-playground` uses `ChunkingPlayground`.
- `/kanban` uses `KanbanPlayground`.
- Labels match the route-job metadata.

- [ ] **Step 5: Verify route registry and sidepanel flashcards coverage**

Extend `route-registry.sidepanel-flashcards.test.ts` or add route-registry assertions so:

- `/flashcards` remains registered in sidepanel route registry.
- `sidepanel-flashcards.tsx` still opens `/options.html#/flashcards`.
- Task 11B labs routes do not appear in default sidepanel navigation unless WP1 metadata says they should.

- [ ] **Step 6: Run route contract tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit route contract**

```bash
git add apps/packages/ui/src/routes/study-safety-specialized-route-jobs.ts apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
git commit -m "test: lock study safety specialized route contract"
```

### Task 2: Make `/evaluations` Ready-Or-Recoverable

**Files:**
- Modify: `apps/packages/ui/src/routes/option-evaluations.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-evaluations.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/EvaluationsPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/SyntheticReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/EvaluationsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/RunsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/DatasetsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/WebhooksTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/HistoryTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/EvaluationsTab.empty-state.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RunsTab.benchmark-option.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/DatasetsTab.pagination.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/HistoryTab.filters.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/evaluations-recipes-guided.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/evaluations-synthetic-review.spec.ts`

- [ ] **Step 1: Add failing route-boundary coverage**

Extend `study-safety-specialized-route-boundaries.test.tsx`:

```ts
it("wraps the evaluations route in a route boundary", () => {
  render(<OptionEvaluations />)
  expect(screen.getByTestId("route-boundary")).toHaveAttribute("data-route-id", "evaluations")
  expect(screen.getByTestId("route-boundary")).toHaveAttribute("data-route-label", "Evaluations")
  expect(screen.getByTestId("evaluations-page")).toBeVisible()
})
```

- [ ] **Step 2: Run the boundary test to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx
```

Expected: FAIL while `option-evaluations.tsx` lacks a route boundary.

- [ ] **Step 3: Add shared and extension route boundaries**

Wrap shared and extension `OptionEvaluations` in:

```tsx
<RouteErrorBoundary routeId="evaluations" routeLabel="Evaluations">
  <OptionLayout>
    <EvaluationsPlaygroundPage />
  </OptionLayout>
</RouteErrorBoundary>
```

- [ ] **Step 4: Add evaluations readiness tests**

Extend existing evaluations tests to cover:

- `WorkspaceConnectionGate` setup state before tab content.
- Beta identity without hiding the primary recipe flow.
- Recipes, Review, Evaluations, Runs, Datasets, Webhooks, and History tabs remain discoverable.
- Empty evaluations state points to recipes or creation.
- Worker unavailable or endpoint unavailable state uses WP2 vocabulary and diagnostics disclosure.
- Run, dataset, webhook, and history failures preserve filters or form input.

- [ ] **Step 5: Implement minimal evaluations adjustments**

Use `WorkspaceConnectionGate`, existing tab components, and existing query error states. Add shared state components only where raw endpoint or vague error text is primary UI.

- [ ] **Step 6: Run evaluations unit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/EvaluationsTab.empty-state.test.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RunsTab.benchmark-option.test.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/DatasetsTab.pagination.test.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/HistoryTab.filters.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Run evaluations browser tests**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/smoke/evaluations-recipes-guided.spec.ts apps/tldw-frontend/e2e/smoke/evaluations-synthetic-review.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 8: Commit evaluations readiness**

```bash
git add apps/packages/ui/src/routes/option-evaluations.tsx apps/tldw-frontend/extension/routes/option-evaluations.tsx apps/packages/ui/src/components/Option/Evaluations apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/smoke/evaluations-recipes-guided.spec.ts apps/tldw-frontend/e2e/smoke/evaluations-synthetic-review.spec.ts
git commit -m "feat: clarify evaluations route readiness"
```

### Task 3: Make Flashcards And Quiz Study Modes Clear

**Files:**
- Modify: `apps/packages/ui/src/routes/option-flashcards.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-flashcards.tsx`
- Modify: `apps/packages/ui/src/routes/option-quiz.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-quiz.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/QuizWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/CreateTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts`

- [ ] **Step 1: Add study mode assertions**

Extend flashcards and quiz tests to assert:

- Flashcards tabs expose Study, Manage, Import / Export, Templates, and Scheduler when eligible.
- Quiz tabs expose Take Quiz, Generate, Create, Manage, and Results.
- First-time empty state puts the user on a productive start route without hiding expert tabs.
- Demo mode is labeled as local demo state.
- Unsupported or unconfigured states keep route headings and recovery actions visible.

- [ ] **Step 2: Run study tests before implementation**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx apps/packages/ui/src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx apps/packages/ui/src/components/Quiz/__tests__/quiz-ftux.test.tsx
```

Expected: FAIL only where the added assertions expose missing route identity or state handling.

- [ ] **Step 3: Implement minimal study adjustments**

Use existing connection gates, demo previews, tab state, sidepanel handoff, review queues, generate intents, save-progress behavior, retry queues, and results tabs. Avoid new study abstractions unless duplication prevents a clear state fix.

- [ ] **Step 4: Run expanded flashcards tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run expanded quiz tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx apps/packages/ui/src/components/Quiz/__tests__/quiz-ftux.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/CreateTab.save-progress.test.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.filters-retake.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Run study E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 7: Commit study route clarity**

```bash
git add apps/packages/ui/src/components/Flashcards apps/packages/ui/src/components/Quiz apps/packages/ui/src/routes/option-flashcards.tsx apps/packages/ui/src/routes/option-quiz.tsx apps/tldw-frontend/extension/routes/option-flashcards.tsx apps/tldw-frontend/extension/routes/option-quiz.tsx apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts
git commit -m "feat: clarify study route modes"
```

### Task 4: Clarify Safety, Content Review, And Claims Alias

**Files:**
- Modify: `apps/packages/ui/src/routes/option-moderation-playground.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-moderation-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-content-review.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-content-review.tsx`
- Modify: `apps/tldw-frontend/pages/claims-review.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/PolicySettingsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/BlocklistStudioPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/UserOverridesPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/TestSandboxPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModerationPlayground/AdvancedPanel.tsx`
- Modify: `apps/packages/ui/src/components/ContentReview/ContentReviewPage.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts`

- [ ] **Step 1: Add safety and review route tests**

Extend tests to assert:

- Moderation route has one semantic route heading.
- Moderation tabs expose Policy and Settings, Blocklist Studio, User Overrides, Test Sandbox, and Advanced.
- Admin permission errors block unsafe controls and identify the permission class.
- Offline and auth states use the shared recovery vocabulary.
- Content Review empty state points to Quick Ingest review mode.
- Content Review exposes batch, draft, diff, metadata, and action sections.
- `/claims-review` redirects to `/content-review`.

- [ ] **Step 2: Run safety and review tests before implementation**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.connection.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlayground.progressive-disclosure.test.tsx
```

Expected: FAIL only where new assertions expose unclear state or landmarks.

- [ ] **Step 3: Implement minimal safety and review adjustments**

Use existing moderation context, policy, blocklist, override, test, and advanced panels. Use existing content-review batch, draft, editor, metadata, and commit controls. Do not add a new claims-review page.

- [ ] **Step 4: Run safety and review component tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.connection.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlayground.progressive-disclosure.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run safety and review E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit safety and review clarity**

```bash
git add apps/packages/ui/src/components/Option/ModerationPlayground apps/packages/ui/src/components/ContentReview/ContentReviewPage.tsx apps/packages/ui/src/routes/option-moderation-playground.tsx apps/packages/ui/src/routes/option-content-review.tsx apps/tldw-frontend/extension/routes/option-moderation-playground.tsx apps/tldw-frontend/extension/routes/option-content-review.tsx apps/tldw-frontend/pages/claims-review.tsx apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts
git commit -m "feat: clarify safety and review routes"
```

### Task 5: Clarify Data Tables And Chunking As Advanced Tools

**Files:**
- Modify: `apps/packages/ui/src/routes/option-data-tables.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-data-tables.tsx`
- Modify: `apps/packages/ui/src/routes/option-chunking-playground.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-chunking-playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/DataTables/DataTablesPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/DataTables/CreateTableWizard.tsx`
- Modify: `apps/packages/ui/src/components/Option/DataTables/DataTablesList.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChunkingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChunkingPlayground/ChunkingCapabilitiesPanel.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts`

- [ ] **Step 1: Add advanced-tool route tests**

Extend tests to assert:

- Data Tables route identifies itself as Data Tables Studio but route metadata label remains Data Tables.
- Data Tables tabs expose My Tables and Create Table.
- Backend setup, beta state, generation failure, save failure, and export states preserve generated table data.
- Chunking route exposes Single, Compare, Templates, and Capabilities.
- Chunking capability load failure does not block sample text or local input where a route path still works.
- Chunking compare results, templates, and applied options remain discoverable.

- [ ] **Step 2: Run advanced-tool tests before implementation**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesPage.golden-path.test.tsx apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesList.a11y.test.tsx apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.golden-path.test.tsx apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.responsive-layout.test.tsx
```

Expected: FAIL only where new assertions expose missing state or unclear route framing.

- [ ] **Step 3: Implement minimal advanced-tool adjustments**

Use existing `WorkspaceConnectionGate`, `DismissibleBetaAlert`, Data Tables tabs, chunking capability panel, templates panel, single and compare views. Avoid creating a new advanced-tool wrapper unless WP1 or WP2 already introduced one.

- [ ] **Step 4: Run advanced-tool unit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesPage.golden-path.test.tsx apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesList.a11y.test.tsx apps/packages/ui/src/components/Option/DataTables/__tests__/ExportMenu.a11y.test.tsx apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.golden-path.test.tsx apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.responsive-layout.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run advanced-tool E2E**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit advanced-tool clarity**

```bash
git add apps/packages/ui/src/components/Option/DataTables apps/packages/ui/src/components/Option/ChunkingPlayground apps/packages/ui/src/routes/option-data-tables.tsx apps/packages/ui/src/routes/option-chunking-playground.tsx apps/tldw-frontend/extension/routes/option-data-tables.tsx apps/tldw-frontend/extension/routes/option-chunking-playground.tsx apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts
git commit -m "feat: clarify data and chunking advanced routes"
```

### Task 6: Classify Kanban And VN Routes

**Files:**
- Modify: `apps/packages/ui/src/routes/option-kanban-playground.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-kanban-playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/KanbanPlayground/index.tsx`
- Modify: `apps/tldw-frontend/pages/vn-assets.tsx`
- Modify: `apps/tldw-frontend/pages/vn-play.tsx`
- Modify: `apps/tldw-frontend/components/vn-assets/VNAssetsWorkbench.tsx`
- Modify: `apps/tldw-frontend/components/vn-assets/ReadinessPanel.tsx`
- Modify: `apps/tldw-frontend/components/vn-assets/GenerationMonitor.tsx`
- Modify: `apps/tldw-frontend/components/vn-assets/ReviewBoard.tsx`
- Modify: `apps/tldw-frontend/components/vn-assets/PortabilityPanel.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/SessionList.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/SceneStage.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/DialoguePanel.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/SceneInspector.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`

- [ ] **Step 1: Add labs classification tests**

Extend route-job and browser tests to assert:

- `/kanban` has an explicit labs or production classification. If it remains labs, default navigation and smoke inventory treat it as labs.
- `/vn-assets` has a route heading, readiness state, pack state, generation state, review state, and portability state.
- `/vn-play` has a route heading, session state, scene state, dialogue state, runtime inspector state, checkpoint state, and retry recovery state.
- Raw VN errors do not become the primary route identity.

- [ ] **Step 2: Run labs tests before implementation**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/KanbanPlayground.empty-state.test.tsx apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/CardDetailPanel.due-date.test.tsx
```

Expected: PASS or FAIL based on added assertions. Do not edit VN runtime code before adding browser assertions because VN pages do not currently have package-local component tests.

- [ ] **Step 3: Implement minimal kanban and VN adjustments**

Use existing kanban board gallery, import, export, card detail, label, due-date, archive, VN asset readiness, generation monitor, review board, portability panel, VN play session list, scene stage, dialogue panel, choice panel, and inspector. Avoid runtime redesign.

- [ ] **Step 4: Run kanban unit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/KanbanPlayground.empty-state.test.tsx apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/CardDetailPanel.due-date.test.tsx apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/kanbanDateTime.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run kanban and VN browser tests**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit labs route classification**

```bash
git add apps/packages/ui/src/routes/option-kanban-playground.tsx apps/tldw-frontend/extension/routes/option-kanban-playground.tsx apps/packages/ui/src/components/Option/KanbanPlayground apps/tldw-frontend/pages/vn-assets.tsx apps/tldw-frontend/pages/vn-play.tsx apps/tldw-frontend/components/vn-assets apps/tldw-frontend/components/vn-play apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts
git commit -m "feat: classify kanban and vn routes"
```

### Task 7: Verify Task 11B Across Browser, Tests, And Responsive States

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`

- [ ] **Step 1: Run route contract tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.connection.test.tsx apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesPage.golden-path.test.tsx apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.golden-path.test.tsx apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/KanbanPlayground.empty-state.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run required parent-plan Playwright coverage**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Run expanded Task 11B Playwright coverage**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Perform browser QA**

With the WebUI running, inspect these routes in desktop and 390px mobile viewports:

- `/evaluations`: route purpose, beta state, tab identity, recipe start, worker or endpoint recovery.
- `/flashcards`: setup state, demo state, Study, Manage, Import / Export, Templates, Scheduler, sidepanel handoff.
- `/quiz`: setup state, demo state, Take Quiz, Generate, Create, Manage, Results, unsaved create guard.
- `/moderation-playground`: safety identity, admin permission, offline state, tabs, unsaved changes, test sandbox.
- `/content-review`: empty state, batch list, selected draft, editor, diff, metadata, actions.
- `/claims-review`: alias resolves to `/content-review`.
- `/data-tables`: beta state, My Tables, Create Table, generation, preview, save, export, failure recovery.
- `/chunking-playground`: Single, Compare, Templates, Capabilities, results, capability load state.
- `/kanban`: classification, board empty state, create/import/export, card detail, archive.
- `/vn-assets`: labs classification, readiness, pack setup, matrix, generation, review, portability.
- `/vn-play`: labs classification, session list, new session, scene, dialogue, choices, inspector, retry.

Capture before and after observations in the Backlog task and PR description.

- [ ] **Step 6: Run final repository hygiene checks for touched scope**

Run:

```bash
git diff --check
```

Expected: PASS.

Run:

```bash
bunx tsc --noEmit
```

Expected: PASS, or document pre-existing TypeScript failures with exact file and error evidence.

- [ ] **Step 7: Commit verification updates**

```bash
git add apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts
git commit -m "test: verify study safety specialized routes"
```

## Acceptance Criteria

- `/evaluations`, `/flashcards`, `/quiz`, `/moderation-playground`, `/content-review`, `/claims-review`, `/data-tables`, `/chunking-playground`, `/kanban`, `/vn-assets`, and `/vn-play` are represented in `study-safety-specialized-route-jobs.ts`.
- Every Task 11B route has a label, classification, primary job, primary action, route owner, canonical component, and visibility decision.
- Evaluations exposes Recipes, Review, Evaluations, Runs, Datasets, Webhooks, and History with clear readiness and recovery states.
- Flashcards exposes Study, Manage, Import / Export, Templates, and Scheduler where applicable.
- Quiz exposes Take Quiz, Generate, Create, Manage, and Results.
- Moderation identifies itself as a safety/admin route and handles offline, auth, permission, unsaved, and test states.
- Content Review identifies itself as a review queue and `/claims-review` remains an intentional alias to `/content-review`.
- Data Tables and Chunking Playground are classified as advanced or beta tools with visible readiness and output recovery.
- Kanban has an explicit labs or production classification.
- VN Assets and VN Play have labs classification, route identity, readiness, session or generation status, and recovery states.
- Shared WebUI and extension route wrappers have explicit route identity tests or documented intentional differences.
- Desktop and 390px browser QA confirms no route has overlapping controls, hidden primary actions, or inaccessible recovery paths.
- No backend API change is included without a separate Backlog task and updated plan.

## Verification Commands

Run these before considering Task 11B complete:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
```

```bash
bunx vitest run apps/packages/ui/src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx apps/packages/ui/src/components/Option/ModerationPlayground/__tests__/ModerationPlaygroundShell.connection.test.tsx apps/packages/ui/src/components/Option/DataTables/__tests__/DataTablesPage.golden-path.test.tsx apps/packages/ui/src/components/Option/ChunkingPlayground/__tests__/ChunkingPlayground.golden-path.test.tsx apps/packages/ui/src/components/Option/KanbanPlayground/__tests__/KanbanPlayground.empty-state.test.tsx
```

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts --reporter=line
```

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/tier-2-features/evaluations.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/content-review.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/data-tables.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/kanban.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/moderation-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/chunking-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/claims-review.spec.ts apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts apps/tldw-frontend/e2e/smoke/vn-play.spec.ts --reporter=line
```

```bash
git diff --check
```

```bash
bunx tsc --noEmit
```

## Review Notes For Implementers

- Start with route metadata and tests before route wrapper changes.
- Split implementation by sub-slice. Do not combine study, safety, data, kanban, and VN changes in one PR.
- Preserve power-user workflows: eval recipes, flashcard scheduler, quiz create and results, moderation advanced controls, content review diff and metadata, chunking compare, kanban import/export, VN inspector, and VN checkpoint recovery.
- Keep labs and beta labels factual and route-level. Do not bury classification in paragraphs.
- Keep raw endpoint and stack details behind diagnostics disclosure.
- Prefer disabled-control reasons, inline retry, and selection-preserving errors over broad explanatory copy.
- Treat `F19` as a blocker cleanup trigger only. Replace deprecated Ant Design usage only when it blocks a touched UX fix or produces test noise in touched routes.
