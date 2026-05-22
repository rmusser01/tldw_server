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

export type SpecializedRouteOwner =
  | "shared_route"
  | "shared_alias"
  | "extension_route"
  | "next_page"
  | "next_alias"

export type SpecializedRouteFinding =
  | "F2 support"
  | "F9 support"
  | "F15 support"
  | "F18 support"
  | "F19 support"

export type SpecializedRouteVisibilityDecision =
  | "default_nav"
  | "advanced_nav"
  | "labs_nav"
  | "alias_only"

export type SpecializedRouteJob = {
  route:
    | "/evaluations"
    | "/flashcards"
    | "/quiz"
    | "/moderation-playground"
    | "/content-review"
    | "/claims-review"
    | "/data-tables"
    | "/chunking-playground"
    | "/kanban"
    | "/vn-assets"
    | "/vn-play"
  concept: SpecializedRouteConcept
  classification: SpecializedRouteClassification
  label: string
  primaryJob: string
  primaryActionLabel: string
  routeOwner: SpecializedRouteOwner
  canonicalComponent: string
  modes: string[]
  findings: SpecializedRouteFinding[]
  visibilityDecision: SpecializedRouteVisibilityDecision
}

export const STUDY_SAFETY_SPECIALIZED_ROUTE_FINDINGS = [
  "F2 support",
  "F9 support",
  "F15 support",
  "F18 support",
  "F19 support"
] as const satisfies SpecializedRouteFinding[]

export const STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS: SpecializedRouteJob[] = [
  {
    route: "/evaluations",
    concept: "evaluation",
    classification: "advanced_self_hosted",
    label: "Evaluations",
    primaryJob:
      "Define and inspect evaluation recipes, runs, datasets, webhooks, and history.",
    primaryActionLabel: "Create evaluation",
    routeOwner: "shared_route",
    canonicalComponent: "EvaluationsPlaygroundPage",
    modes: [
      "Recipes",
      "Review",
      "Evaluations",
      "Runs",
      "Datasets",
      "Webhooks",
      "History"
    ],
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
    visibilityDecision: "advanced_nav"
  },
  {
    route: "/flashcards",
    concept: "study_flashcards",
    classification: "study_workspace",
    label: "Flashcards",
    primaryJob:
      "Study, manage, import, export, template, and schedule flashcards.",
    primaryActionLabel: "Start studying",
    routeOwner: "shared_route",
    canonicalComponent: "FlashcardsWorkspace",
    modes: ["Study", "Manage", "Import / Export", "Templates", "Scheduler"],
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
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
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
    visibilityDecision: "default_nav"
  },
  {
    route: "/moderation-playground",
    concept: "legacy_alias",
    classification: "legacy_alias",
    label: "Moderation Playground",
    primaryJob: "Redirect to the canonical moderation rules workflow.",
    primaryActionLabel: "Open Moderation Rules",
    routeOwner: "shared_alias",
    canonicalComponent: "Navigate:/moderation/rules",
    modes: [],
    findings: ["F2 support", "F18 support", "F19 support"],
    visibilityDecision: "alias_only"
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
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
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
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
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
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
    visibilityDecision: "labs_nav"
  },
  {
    route: "/kanban",
    concept: "planning_board",
    classification: "advanced_self_hosted",
    label: "Kanban",
    primaryJob: "Manage boards, cards, labels, due dates, imports, and exports.",
    primaryActionLabel: "Create board",
    routeOwner: "shared_route",
    canonicalComponent: "KanbanPlayground",
    modes: ["Boards", "Cards", "Import", "Export", "Archive"],
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
    visibilityDecision: "advanced_nav"
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
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
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
    modes: [
      "Sessions",
      "Scene",
      "Dialogue",
      "Choices",
      "Inspector",
      "Checkpoints"
    ],
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ],
    visibilityDecision: "labs_nav"
  }
]

export const getStudySafetySpecializedRouteJob = (
  route: SpecializedRouteJob["route"]
): SpecializedRouteJob | undefined =>
  STUDY_SAFETY_SPECIALIZED_ROUTE_JOBS.find((job) => job.route === route)
