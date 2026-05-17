import type {
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualStarterComplexityTier,
  PersonaVisualStarterPackSummary
} from "@/types/persona-visuals"

export type BuddyBuilderSource =
  | "bundled"
  | "codex_import"
  | "native_import"
  | "library"
  | "duplicate"
  | "blank"

export type BuddyBuilderStep =
  | "source"
  | "draft"
  | "review"
  | "configure"
  | "activate"

export const BUDDY_BUILDER_STEPS: BuddyBuilderStep[] = [
  "source",
  "draft",
  "review",
  "configure",
  "activate"
]

export const BASIC_BUDDY_STARTER_IDS = [
  "search-lens-basic",
  "index-card-basic",
  "archive-cube-basic",
  "paperclip-basic",
  "terminal-tile-basic",
  "migu-marker-basic"
] as const

const BASIC_BUDDY_STARTER_ID_SET = new Set<string>(BASIC_BUDDY_STARTER_IDS)
const BASIC_BUDDY_STARTER_INDEX = new Map<string, number>(
  BASIC_BUDDY_STARTER_IDS.map((id, index) => [id, index])
)

export type BuddyBuilderState = {
  source: BuddyBuilderSource | null
  selectedStarterId: string | null
  selectedImportFile: File | null
  importPreview:
    | PersonaVisualImportPreviewStartResponse
    | PersonaVisualImportPreviewResponse
    | Record<string, unknown>
    | null
  selectedDraftPackId: string | null
  activationReady: boolean
}

export const resetBuddyBuilderForSource = (
  state: BuddyBuilderState,
  source: BuddyBuilderSource
): BuddyBuilderState => ({
  ...state,
  source,
  selectedStarterId: null,
  selectedImportFile: null,
  importPreview: null,
  selectedDraftPackId: null,
  activationReady: false
})

export type BuddyStarterCatalogItem = PersonaVisualStarterPackSummary & {
  recommended: boolean
}

export type BuddyStarterCatalogGroups = Record<
  PersonaVisualStarterComplexityTier,
  BuddyStarterCatalogItem[]
>

const makeCatalogItem = (
  pack: PersonaVisualStarterPackSummary
): BuddyStarterCatalogItem => ({
  ...pack,
  recommended:
    pack.complexity_tier === "basic" &&
    pack.production_status === "art_ready" &&
    BASIC_BUDDY_STARTER_ID_SET.has(pack.id)
})

const getCatalogSortIndex = (pack: BuddyStarterCatalogItem): number =>
  BASIC_BUDDY_STARTER_INDEX.get(pack.id) ?? Number.MAX_SAFE_INTEGER

export const groupBuddyStarterPacksByTier = (
  packs: PersonaVisualStarterPackSummary[]
): BuddyStarterCatalogGroups => {
  const groups: BuddyStarterCatalogGroups = {
    basic: [],
    intermediate: [],
    intricate: []
  }

  for (const pack of packs) {
    groups[pack.complexity_tier].push(makeCatalogItem(pack))
  }

  for (const tier of Object.keys(groups) as PersonaVisualStarterComplexityTier[]) {
    groups[tier].sort((left, right) => {
      const leftIndex = getCatalogSortIndex(left)
      const rightIndex = getCatalogSortIndex(right)
      if (leftIndex !== rightIndex) return leftIndex - rightIndex
      return left.title.localeCompare(right.title)
    })
  }

  return groups
}
