// Retrieval diagnostics extraction for the ResearchWorkspace ChatPane.
//
// Stage 3 (WP-04, TASK-13521): buildRetrievalDiagnostics moved out of
// ChatPane/index.tsx into this pure module so the transcript can memoize it
// per message id (see getRowRetrievalDiagnostics in ChatPane) and tests can
// spy on the computation. The small coercion helpers it shares with the rest
// of the ChatPane live here and are re-exported for index.tsx; behavior is
// byte-for-byte the pre-extraction logic.

export type RetrievalDiagnostics = {
  chunkCount: number | null
  sourceCount: number | null
  sourceLabels: string[]
  averageRelevanceScore: number | null
  faithfulnessScore: number | null
  promptTokens: number | null
  completionTokens: number | null
  totalTokens: number | null
  costUsd: number | null
  confidenceLevel: "high" | "medium" | "low" | null
}

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

export const toPositiveInt = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10)
    if (Number.isInteger(parsed) && parsed > 0) {
      return parsed
    }
  }
  return null
}

export const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

export const normalizeText = (value: unknown): string =>
  typeof value === "string" ? value.trim().toLowerCase() : ""

export const extractStringCandidate = (value: unknown): string => {
  if (typeof value !== "string") return ""
  return value.trim()
}

export const extractCitationMediaId = (citation: unknown): number | null => {
  if (!isRecord(citation)) return null
  const metadata = isRecord(citation.metadata) ? citation.metadata : null
  const includeMediaIds = Array.isArray(metadata?.include_media_ids)
    ? metadata?.include_media_ids
    : []
  const mediaCandidates: unknown[] = [
    citation.mediaId,
    citation.media_id,
    citation.mediaID,
    metadata?.mediaId,
    metadata?.media_id,
    metadata?.mediaID,
    ...includeMediaIds
  ]

  for (const candidate of mediaCandidates) {
    const mediaId = toPositiveInt(candidate)
    if (mediaId !== null) {
      return mediaId
    }
  }
  return null
}

const extractSourceScore = (source: unknown): number | null => {
  if (!isRecord(source)) return null
  const metadata = isRecord(source.metadata) ? source.metadata : null
  const scoreCandidates: unknown[] = [
    source.score,
    source.relevance,
    source.relevance_score,
    source.similarity,
    source.rerank_score,
    metadata?.score,
    metadata?.relevance,
    metadata?.relevance_score,
    metadata?.similarity,
    metadata?.rerank_score
  ]

  for (const candidate of scoreCandidates) {
    const parsed = toFiniteNumber(candidate)
    if (parsed !== null) {
      return parsed
    }
  }
  return null
}

const extractSourceLabel = (source: unknown): string | null => {
  if (!isRecord(source)) return null
  const metadata = isRecord(source.metadata) ? source.metadata : null
  const candidates: unknown[] = [
    source.name,
    source.title,
    source.source,
    source.url,
    metadata?.title,
    metadata?.source,
    metadata?.url
  ]

  for (const candidate of candidates) {
    const normalized = extractStringCandidate(candidate)
    if (normalized.length > 0) {
      return normalized
    }
  }

  return null
}

export const normalizeConfidenceScore = (score: number | null): number | null => {
  if (score === null || !Number.isFinite(score)) {
    return null
  }
  if (score > 1 && score <= 100) {
    return score / 100
  }
  if (score < 0) {
    return 0
  }
  if (score > 1) {
    return 1
  }
  return score
}

const getConfidenceLevel = (
  score: number | null
): RetrievalDiagnostics["confidenceLevel"] => {
  const normalized = normalizeConfidenceScore(score)
  if (normalized === null) {
    return null
  }
  if (normalized >= 0.8) {
    return "high"
  }
  if (normalized >= 0.55) {
    return "medium"
  }
  return "low"
}

const getNumericField = (
  generationInfo: unknown,
  paths: Array<string[]>
): number | null => {
  if (!isRecord(generationInfo)) return null

  for (const path of paths) {
    let current: unknown = generationInfo
    for (const key of path) {
      if (!isRecord(current)) {
        current = undefined
        break
      }
      current = current[key]
    }
    const parsed = toFiniteNumber(current)
    if (parsed !== null) {
      return parsed
    }
  }

  return null
}

export const buildRetrievalDiagnostics = (
  messageSources: unknown[] | undefined,
  generationInfo: unknown
): RetrievalDiagnostics | null => {
  const sources = Array.isArray(messageSources) ? messageSources : []

  const chunkCountFromGeneration = getNumericField(generationInfo, [
    ["retrieval", "chunks_retrieved"],
    ["retrieval", "chunk_count"],
    ["retrieval", "chunks"],
    ["chunks_retrieved"],
    ["chunk_count"],
    ["retrieved_chunks"],
    ["context_chunks"]
  ])

  const sourceCountFromGeneration = getNumericField(generationInfo, [
    ["retrieval", "source_count"],
    ["retrieval", "sources_used"],
    ["source_count"],
    ["sources_used"]
  ])

  const avgScoreFromGeneration = getNumericField(generationInfo, [
    ["retrieval", "avg_relevance_score"],
    ["retrieval", "average_relevance_score"],
    ["avg_relevance_score"],
    ["average_relevance_score"],
    ["relevance_score"]
  ])
  const faithfulnessScore = getNumericField(generationInfo, [
    ["faithfulness", "score"],
    ["faithfulness", "faithfulness_score"],
    ["faithfulness", "overall_score"],
    ["faithfulness_score"],
    ["claim_verification", "faithfulness_score"],
    ["retrieval", "faithfulness_score"],
    ["confidence"],
    ["confidence_score"]
  ])
  const promptTokensFromGeneration = getNumericField(generationInfo, [
    ["usage", "prompt_tokens"],
    ["usage", "input_tokens"],
    ["prompt_tokens"],
    ["input_tokens"],
    ["prompt_eval_count"]
  ])
  const completionTokensFromGeneration = getNumericField(generationInfo, [
    ["usage", "completion_tokens"],
    ["usage", "output_tokens"],
    ["completion_tokens"],
    ["output_tokens"],
    ["eval_count"]
  ])
  const totalTokensFromGeneration = getNumericField(generationInfo, [
    ["usage", "total_tokens"],
    ["total_tokens"],
    ["total_token_count"]
  ])
  const costUsdFromGeneration = getNumericField(generationInfo, [
    ["usage", "total_cost_usd"],
    ["usage", "estimated_cost_usd"],
    ["usage", "cost_usd"],
    ["pricing", "total_cost_usd"],
    ["pricing", "estimated_cost_usd"],
    ["pricing", "cost_usd"],
    ["total_cost_usd"],
    ["estimated_cost_usd"],
    ["cost_usd"],
    ["total_cost"],
    ["estimated_cost"]
  ])

  const uniqueSourceKeys = new Set<string>()
  const uniqueSourceLabels = new Set<string>()
  for (const source of sources) {
    if (!isRecord(source)) continue
    const sourceLabel = extractSourceLabel(source)
    if (sourceLabel) {
      uniqueSourceLabels.add(sourceLabel)
    }

    const mediaId = extractCitationMediaId(source)
    if (mediaId !== null) {
      uniqueSourceKeys.add(`media:${mediaId}`)
    } else {
      const metadata = isRecord(source.metadata) ? source.metadata : null
      const label = normalizeText(
        source.name || source.url || metadata?.source || metadata?.title
      )
      if (label) {
        uniqueSourceKeys.add(`label:${label}`)
      }
    }
  }

  const sourceScores = sources
    .map((source) => extractSourceScore(source))
    .filter((score): score is number => score !== null)

  const averageScoreFromSources =
    sourceScores.length > 0
      ? sourceScores.reduce((sum, score) => sum + score, 0) /
        sourceScores.length
      : null

  const diagnostics: RetrievalDiagnostics = {
    chunkCount:
      chunkCountFromGeneration !== null
        ? Math.max(0, Math.round(chunkCountFromGeneration))
        : sources.length > 0
          ? sources.length
          : null,
    sourceCount:
      sourceCountFromGeneration !== null
        ? Math.max(0, Math.round(sourceCountFromGeneration))
        : uniqueSourceKeys.size > 0
          ? uniqueSourceKeys.size
          : null,
    sourceLabels: Array.from(uniqueSourceLabels),
    averageRelevanceScore:
      avgScoreFromGeneration !== null
        ? avgScoreFromGeneration
        : averageScoreFromSources,
    faithfulnessScore,
    promptTokens:
      promptTokensFromGeneration !== null
        ? Math.max(0, Math.round(promptTokensFromGeneration))
        : null,
    completionTokens:
      completionTokensFromGeneration !== null
        ? Math.max(0, Math.round(completionTokensFromGeneration))
        : null,
    totalTokens:
      totalTokensFromGeneration !== null
        ? Math.max(0, Math.round(totalTokensFromGeneration))
        : promptTokensFromGeneration !== null ||
            completionTokensFromGeneration !== null
          ? Math.max(
              0,
              Math.round(
                (promptTokensFromGeneration || 0) +
                  (completionTokensFromGeneration || 0)
              )
            )
          : null,
    costUsd:
      costUsdFromGeneration !== null && costUsdFromGeneration >= 0
        ? costUsdFromGeneration
        : null,
    confidenceLevel: getConfidenceLevel(
      faithfulnessScore !== null
        ? faithfulnessScore
        : avgScoreFromGeneration !== null
          ? avgScoreFromGeneration
          : averageScoreFromSources
    )
  }

  if (
    diagnostics.chunkCount === null &&
    diagnostics.sourceCount === null &&
    diagnostics.averageRelevanceScore === null &&
    diagnostics.faithfulnessScore === null &&
    diagnostics.sourceLabels.length === 0 &&
    diagnostics.totalTokens === null &&
    diagnostics.costUsd === null
  ) {
    return null
  }

  return diagnostics
}
