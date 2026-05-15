import type {
  PersonaVisualAnimation,
  PersonaVisualPack,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import { getAssetsById, normalizeFrames } from "./personaVisualAssets"
import { getPersonaVisualRenderer } from "./personaVisualRenderers"

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

export const getPersonaVisualDiagnosticToneClassName = (
  severity: PersonaVisualDiagnosticSeverity
): string => {
  if (severity === "error") {
    return "border-danger/30 bg-danger/10 text-danger"
  }
  if (severity === "warning") {
    return "border-warn/30 bg-warn/10 text-warn"
  }
  return "border-primary/30 bg-primary/10 text-primary"
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

  if (!getPersonaVisualRenderer(pack.renderer_type)) {
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
