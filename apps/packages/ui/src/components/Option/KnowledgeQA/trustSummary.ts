import { getRagSourceLabel } from "@/services/rag/sourceMetadata"
import type { RagSource } from "@/services/rag/unified-rag"

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
  trustLabel: AnswerTrustLabel | null | undefined
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
  trustLabel,
}: BuildAnswerTrustSummaryInput): string[] {
  const lines = [
    `Searched ${formatSourceList(selectedSources)}. ${resultCount} ${pluralize(
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

  if (sourceHealthCaveatCount > 0) {
    lines.push(
      `${sourceHealthCaveatCount} selected ${pluralize(
        sourceHealthCaveatCount,
        "source"
      )} ${sourceHealthCaveatCount === 1 ? "needs" : "need"} attention.`
    )
  } else {
    lines.push("Selected sources look ready.")
  }

  if (trustLabel) {
    lines.push(`Trust: ${trustLabel}.`)
  }

  return lines
}
