import { getRagSourceLabel } from "@/services/rag/sourceMetadata"
import type { RagSource } from "@/services/rag/unified-rag"
import type { KnowledgeAnswerTrustState, KnowledgeSourceStatus } from "./types"
import { getKnowledgeAnswerTrustLabel } from "./trustState"

export type AnswerTrustLabel = "Strong" | "Partial" | "Weak"

type BuildAnswerTrustSummaryInput = {
  selectedSources: RagSource[]
  resultCount: number
  citationCount: number
  webFallbackEnabled: boolean
  webFallbackTriggered: boolean
  generationProvider: string | null | undefined
  generationModel: string | null | undefined
  sourceHealthCaveatCount: number
  sourceStatus?: Record<string, KnowledgeSourceStatus>
  trustState?: KnowledgeAnswerTrustState | null | undefined
  trustLabel?: AnswerTrustLabel | null | undefined
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return count === 1 ? singular : plural
}

function formatGenerationModel(
  provider: string | null | undefined,
  model: string | null | undefined
): string {
  const normalizedProvider = provider?.trim() || null
  const normalizedModel = model?.trim() || null

  if (normalizedProvider && normalizedModel) {
    return `${normalizedProvider} / ${normalizedModel}`
  }
  return normalizedProvider ?? normalizedModel ?? "Server default"
}

export function formatSourceList(sources: RagSource[]): string {
  const labels = sources.map(getRagSourceLabel)
  if (labels.length <= 1) return labels[0] ?? "selected sources"
  if (labels.length === 2) return `${labels[0]} and ${labels[1]}`
  return `${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`
}

export function buildAnswerTrustSummary({
  selectedSources,
  resultCount,
  citationCount,
  webFallbackEnabled,
  webFallbackTriggered,
  generationProvider,
  generationModel,
  sourceHealthCaveatCount,
  sourceStatus,
  trustState,
  trustLabel,
}: BuildAnswerTrustSummaryInput): string[] {
  const failedSources = selectedSources.filter((source) =>
    ["error", "unavailable"].includes(sourceStatus?.[source]?.status ?? "")
  )
  const unavailableSources = failedSources.filter((source) => !sourceStatus?.[source]?.count)
  const partialSources = failedSources.filter((source) => Boolean(sourceStatus?.[source]?.count))
  const searchedSources = selectedSources.filter((source) => !unavailableSources.includes(source))
  const searchSummary = searchedSources.length > 0
    ? `Searched ${formatSourceList(searchedSources)}.`
    : "No selected sources were searched."
  const lines = [
    `${searchSummary} ${resultCount} ${pluralize(
      resultCount,
      "source"
    )} returned, ${citationCount} cited.`,
  ]

  lines.push(
    webFallbackEnabled
      ? `Web fallback enabled, ${webFallbackTriggered ? "used" : "not used"}.`
      : "Web fallback disabled."
  )
  lines.push(`AI model: ${formatGenerationModel(generationProvider, generationModel)}.`)

  if (unavailableSources.length > 0) {
    lines.push(`Could not search ${formatSourceList(unavailableSources)}.`)
  }
  if (partialSources.length > 0) {
    lines.push(`Some searches failed for ${formatSourceList(partialSources)}.`)
  }
  if (sourceHealthCaveatCount > 0) {
    lines.push(
      `${sourceHealthCaveatCount} selected ${pluralize(
        sourceHealthCaveatCount,
        "source"
      )} ${sourceHealthCaveatCount === 1 ? "needs" : "need"} attention.`
    )
  } else if (failedSources.length === 0) {
    lines.push("Selected sources look ready.")
  }

  if (trustState) {
    lines.push(`Trust: ${getKnowledgeAnswerTrustLabel(trustState)}.`)
  } else if (trustLabel) {
    lines.push(`Trust: ${trustLabel}.`)
  }

  return lines
}
