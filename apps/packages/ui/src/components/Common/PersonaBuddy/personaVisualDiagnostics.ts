import type {
  PersonaVisualAnimation,
  PersonaVisualAsset,
  PersonaVisualFrame,
  PersonaVisualPack,
  PersonaVisualStateId
} from "@/types/persona-visuals"

export type PersonaVisualDiagnosticCode =
  | "load_failed"
  | "no_active_pack"
  | "unsupported_renderer"
  | "missing_manifest"
  | "missing_assets"
  | "missing_animation"
  | "missing_asset"
  | "unsupported_region"

export type PersonaVisualDiagnosticSeverity = "info" | "warning" | "error"

export type PersonaVisualDiagnostic = {
  code: PersonaVisualDiagnosticCode
  severity: PersonaVisualDiagnosticSeverity
  title: string
  message: string
  actionLabel?: string
}

export type PersonaVisualDiagnosticsInput = {
  pack?: PersonaVisualPack | null
  visualState?: PersonaVisualStateId
  loadStatus?: "idle" | "loading" | "loaded" | "error"
  loadError?: unknown
  renderError?: PersonaVisualDiagnosticCode | null
  includeNoActivePack?: boolean
}

type ResolvedAnimation = {
  animation: PersonaVisualAnimation
  animationId: string
}

const SUPPORTED_RUNTIME_RENDERERS = new Set(["sprite_frames"])

const createDiagnostic = (
  code: PersonaVisualDiagnosticCode,
  severity: PersonaVisualDiagnosticSeverity,
  title: string,
  message: string,
  actionLabel = "Open Visuals"
): PersonaVisualDiagnostic => ({
  code,
  severity,
  title,
  message,
  actionLabel
})

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.trim()) return error.message.trim()
  if (typeof error === "string" && error.trim()) return error.trim()
  return "The active visual pack could not be loaded."
}

const getAssetsById = (
  pack: PersonaVisualPack | null | undefined
): Record<string, PersonaVisualAsset> => {
  if (!pack) return {}
  if (pack.assets_by_id && Object.keys(pack.assets_by_id).length > 0) {
    return pack.assets_by_id
  }
  const assets: Record<string, PersonaVisualAsset> = {}
  for (const asset of pack.assets || []) {
    if (asset?.id) assets[asset.id] = asset
  }
  return assets
}

const normalizeFrames = (
  animation: PersonaVisualAnimation | null | undefined
): PersonaVisualFrame[] => {
  if (!animation) return []
  if (Array.isArray(animation.frames) && animation.frames.length > 0) {
    return animation.frames.filter((frame) => Boolean(frame?.asset_id))
  }
  return (animation.asset_ids || [])
    .filter((assetId) => Boolean(String(assetId || "").trim()))
    .map((assetId) => ({ asset_id: String(assetId) }))
}

const resolveAnimationForState = (
  pack: PersonaVisualPack,
  visualState: PersonaVisualStateId
): ResolvedAnimation | null => {
  const manifest = pack.manifest
  const stateOrder = [
    visualState,
    ...(manifest.fallbacks?.[visualState] || []),
    ...(visualState === "idle" ? [] : ["idle" as PersonaVisualStateId])
  ]

  for (const state of stateOrder) {
    const animationId = manifest.states?.[state]?.animation_id
    if (!animationId) continue
    const animation = manifest.animations?.[animationId]
    if (animation) return { animation, animationId }
  }
  return null
}

export const resolvePersonaVisualDiagnostics = ({
  pack = null,
  visualState = "idle",
  loadStatus = "idle",
  loadError = null,
  renderError = null,
  includeNoActivePack = false
}: PersonaVisualDiagnosticsInput = {}): PersonaVisualDiagnostic[] => {
  if (loadStatus === "error") {
    return [
      createDiagnostic(
        "load_failed",
        "warning",
        "Visual pack did not load",
        getErrorMessage(loadError)
      )
    ]
  }

  if (!pack) {
    return includeNoActivePack
      ? [
          createDiagnostic(
            "no_active_pack",
            "info",
            "No active visual pack",
            "This persona is using the text Buddy fallback until a visual pack is activated."
          )
        ]
      : []
  }

  if (!SUPPORTED_RUNTIME_RENDERERS.has(pack.renderer_type)) {
    return [
      createDiagnostic(
        "unsupported_renderer",
        "warning",
        "Visual renderer is not supported here",
        `The Buddy runtime cannot render ${pack.renderer_type} packs yet.`
      )
    ]
  }

  if (!pack.manifest) {
    return [
      createDiagnostic(
        "missing_manifest",
        "error",
        "Visual pack manifest is missing",
        "The active visual pack needs a manifest before Buddy can render it."
      )
    ]
  }

  const assetsById = getAssetsById(pack)
  if (Object.keys(assetsById).length === 0) {
    return [
      createDiagnostic(
        "missing_assets",
        "error",
        "Visual pack has no assets",
        "The active visual pack references no uploaded assets."
      )
    ]
  }

  const resolved = resolveAnimationForState(pack, visualState)
  const frames = normalizeFrames(resolved?.animation)
  if (!resolved || frames.length === 0 || renderError === "missing_animation") {
    return [
      createDiagnostic(
        "missing_animation",
        "error",
        "Visual animation is missing",
        `No renderable animation was found for the ${visualState} state.`
      )
    ]
  }

  const missingFrame = frames.find((frame) => !assetsById[frame.asset_id])
  if (missingFrame || renderError === "missing_asset") {
    const assetId = missingFrame?.asset_id
    return [
      createDiagnostic(
        "missing_asset",
        "error",
        "Visual asset is missing",
        assetId
          ? `The ${resolved.animationId} animation references missing asset ${assetId}.`
          : `The ${resolved.animationId} animation references an asset Buddy cannot load.`
      )
    ]
  }

  if (renderError === "unsupported_region") {
    return [
      createDiagnostic(
        "unsupported_region",
        "warning",
        "Visual frame region is unsupported",
        "Buddy fell back because this frame uses an unsupported sprite region."
      )
    ]
  }

  return []
}

export const getPrimaryPersonaVisualDiagnostic = (
  input: PersonaVisualDiagnosticsInput = {}
): PersonaVisualDiagnostic | null => resolvePersonaVisualDiagnostics(input)[0] ?? null
