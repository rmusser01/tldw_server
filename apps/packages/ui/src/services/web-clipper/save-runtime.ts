import type {
  WebClipperOutcomeState,
  WebClipperSaveResponse
} from "./types"

export type WebClipperBannerSeverity = "success" | "warning" | "error"

export interface WebClipperBanner {
  severity: WebClipperBannerSeverity
  title: string
  message: string
  warnings: string[]
}

export interface WebClipperSaveRuntime {
  clip_id: string
  note_id: string
  status: WebClipperOutcomeState
  workspace_placement_saved: boolean
  workspace_placement_count: number
  warnings: string[]
  banner: WebClipperBanner
}

const cloneWarnings = (warnings: string[] | undefined): string[] =>
  Array.isArray(warnings) ? [...warnings] : []

type WebClipperBannerSource = {
  status: WebClipperOutcomeState
  warnings?: string[]
}

export const mapWebClipperOutcomeToBanner = (
  outcome: WebClipperBannerSource
): WebClipperBanner => {
  const warnings = cloneWarnings(outcome.warnings)
  switch (outcome.status) {
    case "saved":
      return {
        severity: "success",
        title: "Clip saved",
        message: "The clip was saved successfully.",
        warnings
      }
    case "saved_with_warnings":
      return {
        severity: "warning",
        title: "Clip saved with warnings",
        message: "The clip was saved, but follow-up work reported warnings.",
        warnings
      }
    case "partially_saved":
      return {
        severity: "warning",
        title: "Clip partially saved",
        message: "Some clip stages completed, but at least one destination failed.",
        warnings
      }
    case "failed":
    default:
      return {
        severity: "error",
        title: "Clip save failed",
        message: "The clip could not be saved.",
        warnings
      }
  }
}

export const buildWebClipSaveRuntime = (
  response: WebClipperSaveResponse
): WebClipperSaveRuntime => ({
  clip_id: response.clip_id,
  note_id: response.note_id,
  status: response.status,
  workspace_placement_saved: Boolean(response.workspace_placement_saved),
  workspace_placement_count: Number(response.workspace_placement_count || 0),
  warnings: cloneWarnings(response.warnings),
  banner: mapWebClipperOutcomeToBanner(response)
})
