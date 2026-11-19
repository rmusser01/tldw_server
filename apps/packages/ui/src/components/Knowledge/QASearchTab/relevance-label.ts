/**
 * Plain-language relevance for retrieved chunks (CC-10, #3101).
 *
 * The RAG pipeline returns scores on different scales (rerank probabilities,
 * fused ranks, BM25), so a raw number like "0.873" means little. Each chunk is
 * labelled against the strongest result in the same list instead; the raw
 * score stays in the tooltip.
 */
export type RelevanceLevel = "best" | "good" | "weaker"

export const chunkScore = (doc: {
  score?: number
  relevance?: number
}): number | null => {
  const value = doc.score ?? doc.relevance
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

/** The highest positive score in a list, or null when none is usable. */
export const topChunkScore = (
  docs: ReadonlyArray<{ score?: number; relevance?: number }>
): number | null => {
  let top: number | null = null
  for (const doc of docs) {
    const score = chunkScore(doc)
    if (score !== null && score > 0 && (top === null || score > top)) {
      top = score
    }
  }
  return top
}

export const relevanceLevel = (
  score: number | null,
  topScore: number | null
): RelevanceLevel | null => {
  if (score === null || topScore === null || topScore <= 0) return null
  const ratio = score / topScore
  if (ratio >= 0.9) return "best"
  if (ratio >= 0.6) return "good"
  return "weaker"
}
