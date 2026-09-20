import { getRagSourceLabel, getRagSourceTranslationKey } from "@/services/rag/sourceMetadata"
import type { RagSource } from "@/services/rag/unified-rag"
import type { KnowledgeAnswerTrustState, KnowledgeSourceStatus } from "./types"
import { getKnowledgeAnswerTrustLabel } from "./trustState"

export type AnswerTrustLabel = "Strong" | "Partial" | "Weak"
type SummaryTranslator = (key: string, options: Record<string, unknown>) => string

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

export function formatSourceList(sources: RagSource[], t: SummaryTranslator): string {
  const labels = sources.map((source) => t(getRagSourceTranslationKey(source), {
    defaultValue: getRagSourceLabel(source),
  }))
  if (labels.length <= 1) {
    return labels[0] ?? t("trustSummary.selectedSources", { defaultValue: "selected sources" })
  }
  if (labels.length === 2) {
    return t("trustSummary.sourcePair", {
      defaultValue: "{{first}} and {{last}}", first: labels[0], last: labels[1],
    })
  }
  return t("trustSummary.sourceList", {
    defaultValue: "{{first}}, and {{last}}",
    first: labels.slice(0, -1).join(t("trustSummary.sourceSeparator", { defaultValue: ", " })),
    last: labels[labels.length - 1],
  })
}

export function buildSourceFailureSummary(
  sources: RagSource[],
  sourceStatus: Record<string, KnowledgeSourceStatus> | undefined,
  t: SummaryTranslator
): string[] {
  const failedSources = sources.filter((source) =>
    ["error", "unavailable"].includes(sourceStatus?.[source]?.status ?? "")
  )
  const unavailableSources = failedSources.filter((source) => !sourceStatus?.[source]?.count)
  const partialSources = failedSources.filter((source) => Boolean(sourceStatus?.[source]?.count))
  const lines: string[] = []
  if (unavailableSources.length > 0) {
    lines.push(t("trustSummary.unavailable", {
      defaultValue: "Could not search {{sources}}.",
      sources: formatSourceList(unavailableSources, t),
    }))
  }
  if (partialSources.length > 0) {
    lines.push(t("trustSummary.partialFailure", {
      defaultValue: "Some searches failed for {{sources}}.",
      sources: formatSourceList(partialSources, t),
    }))
  }
  return lines
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
}: BuildAnswerTrustSummaryInput, t: SummaryTranslator): string[] {
  const failedSources = selectedSources.filter((source) =>
    ["error", "unavailable"].includes(sourceStatus?.[source]?.status ?? "")
  )
  const unavailableSources = failedSources.filter((source) => !sourceStatus?.[source]?.count)
  const searchedSources = selectedSources.filter((source) => !unavailableSources.includes(source))
  const searchSummary = searchedSources.length > 0
    ? t("trustSummary.searched", {
        defaultValue: "Searched {{sources}}. {count, plural, one {# source returned} other {# sources returned}}, {{citationCount}} cited.",
        sources: formatSourceList(searchedSources, t), count: resultCount, citationCount,
      })
    : t("trustSummary.unsearched", {
        defaultValue: "No selected sources were searched. {count, plural, one {# source returned} other {# sources returned}}, {{citationCount}} cited.",
        count: resultCount, citationCount,
      })
  const lines = [searchSummary]

  lines.push(
    webFallbackEnabled
      ? `Web fallback enabled, ${webFallbackTriggered ? "used" : "not used"}.`
      : "Web fallback disabled."
  )
  lines.push(`AI model: ${formatGenerationModel(generationProvider, generationModel)}.`)

  lines.push(...buildSourceFailureSummary(selectedSources, sourceStatus, t))
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
