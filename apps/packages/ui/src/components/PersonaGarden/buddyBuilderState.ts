import type {
  PersonaVisualBuiltinStateId,
  PersonaVisualCustomStateId,
  PersonaVisualImportBundleAssetSummary,
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualManifest,
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

export const BUDDY_REQUIRED_STATES: PersonaVisualBuiltinStateId[] = [
  "idle",
  "listening",
  "thinking",
  "speaking",
  "error"
]

export const BUDDY_MOVEMENT_STATES = ["moving_left", "moving_right"] as const

export type BuddyMovementStateId = (typeof BUDDY_MOVEMENT_STATES)[number]

export type BuddyDraftReadinessSummary = {
  sourceLabel: string
  atlasSummary: Array<{
    assetId?: string
    width: number | null
    height: number | null
  }>
  requiredStates: Array<{ id: PersonaVisualBuiltinStateId; resolved: boolean }>
  movementStates: Array<{ id: BuddyMovementStateId; resolved: boolean }>
  customStates: Array<{
    id: string
    label: string
    kind: string
    fallback?: string
  }>
  blockers: string[]
  warnings: string[]
  canActivate: boolean
}

export type SummarizeBuddyDraftReadinessInput = {
  manifest?: PersonaVisualManifest | null
  importPreview?: PersonaVisualImportPreviewResponse | null
  activationBlockers?: string[]
}

const getImportPreviewSourceLabel = (
  importPreview?: PersonaVisualImportPreviewResponse | null
): string => {
  const schemaVersion = String(importPreview?.schema_version || "").toLowerCase()
  if (schemaVersion.startsWith("codex.pet")) return "Codex/Petdex pet"
  if (schemaVersion.startsWith("persona_visual_pack")) return "Persona Visual pack"
  if (importPreview) return "Imported visual pack"
  return "Draft visual pack"
}

const isAtlasAsset = (asset: PersonaVisualImportBundleAssetSummary): boolean =>
  asset.asset_group === "animation_atlas" || asset.asset_role === "sprite_sheet"

const getAtlasSummary = (
  importPreview?: PersonaVisualImportPreviewResponse | null
): BuddyDraftReadinessSummary["atlasSummary"] =>
  (importPreview?.bundle_summary.assets || [])
    .filter(isAtlasAsset)
    .map((asset) => ({
      assetId: asset.source_asset_id || undefined,
      width: typeof asset.width === "number" ? asset.width : null,
      height: typeof asset.height === "number" ? asset.height : null
    }))

const hasManifestState = (
  manifest: PersonaVisualManifest | null | undefined,
  stateId: string
): boolean =>
  Boolean(
    manifest?.states?.[stateId as keyof PersonaVisualManifest["states"]]
      ?.animation_id
  )

const getStateCatalog = (
  manifest: PersonaVisualManifest | null | undefined
): Record<PersonaVisualCustomStateId, NonNullable<PersonaVisualManifest["state_catalog"]>[PersonaVisualCustomStateId]> =>
  manifest?.state_catalog || {}

const summarizeRequiredStates = (
  manifest: PersonaVisualManifest | null | undefined
): BuddyDraftReadinessSummary["requiredStates"] =>
  BUDDY_REQUIRED_STATES.map((id) => ({
    id,
    resolved: hasManifestState(manifest, id)
  }))

const summarizeMovementStates = (
  manifest: PersonaVisualManifest | null | undefined
): BuddyDraftReadinessSummary["movementStates"] => {
  const stateCatalog = getStateCatalog(manifest)
  return BUDDY_MOVEMENT_STATES.filter(
    (id) => hasManifestState(manifest, id) || Boolean(stateCatalog[id as PersonaVisualCustomStateId])
  ).map((id) => ({
    id,
    resolved: hasManifestState(manifest, id)
  }))
}

const summarizeCustomStates = (
  manifest: PersonaVisualManifest | null | undefined
): BuddyDraftReadinessSummary["customStates"] => {
  const stateCatalog = getStateCatalog(manifest)
  return Object.entries(stateCatalog)
    .filter(([id]) => !(BUDDY_MOVEMENT_STATES as readonly string[]).includes(id))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, entry]) => ({
      id,
      label: entry.label || id,
      kind: entry.kind || "custom",
      fallback: entry.fallback
    }))
}

const getPreviewMessages = (values: unknown[] | undefined): string[] =>
  (values || [])
    .map((value) => {
      if (typeof value === "string") return value
      if (value && typeof value === "object" && "message" in value) {
        const message = (value as { message?: unknown }).message
        return typeof message === "string" ? message : JSON.stringify(value)
      }
      return JSON.stringify(value)
    })
    .filter((value) => value && value !== "undefined")

export const summarizeBuddyDraftReadiness = ({
  manifest,
  importPreview = null,
  activationBlockers = []
}: SummarizeBuddyDraftReadinessInput): BuddyDraftReadinessSummary => {
  const requiredStates = summarizeRequiredStates(manifest)
  const missingStateBlockers = requiredStates
    .filter((state) => !state.resolved)
    .map((state) => `Missing required state: ${state.id}`)
  const previewBlockers = getPreviewMessages(
    importPreview?.conflicts?.filter((conflict) => conflict.severity === "blocker")
  )
  const warnings = [
    ...getPreviewMessages(importPreview?.validation_warnings),
    ...getPreviewMessages(importPreview?.target_warnings),
    ...getPreviewMessages(
      importPreview?.conflicts?.filter((conflict) => conflict.severity !== "blocker")
    )
  ]
  const blockers = [
    ...missingStateBlockers,
    ...previewBlockers,
    ...activationBlockers
  ]

  return {
    sourceLabel: getImportPreviewSourceLabel(importPreview),
    atlasSummary: getAtlasSummary(importPreview),
    requiredStates,
    movementStates: summarizeMovementStates(manifest),
    customStates: summarizeCustomStates(manifest),
    blockers,
    warnings,
    canActivate: blockers.length === 0
  }
}
