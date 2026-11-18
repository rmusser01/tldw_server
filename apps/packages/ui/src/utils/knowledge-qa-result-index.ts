export function getOriginalResultIndex(
  result: { metadata?: Record<string, unknown> },
  fallbackIndex: number
): number {
  const rawIndex = result.metadata?.original_result_index
  if (typeof rawIndex !== "number" || !Number.isFinite(rawIndex)) {
    return fallbackIndex
  }
  return Math.max(0, Math.round(rawIndex))
}
