export const normalizeFlashcardTags = (
  tags?: readonly string[] | null
): string[] => {
  if (!Array.isArray(tags) || tags.length === 0) {
    return []
  }

  const seen = new Set<string>()
  const normalized: string[] = []

  for (const rawTag of tags) {
    const tag = rawTag.trim()
    if (!tag) continue

    const dedupeKey = tag.toLowerCase()
    if (seen.has(dedupeKey)) continue

    seen.add(dedupeKey)
    normalized.push(tag)
  }

  return normalized
}

export const normalizeOptionalFlashcardTags = (
  tags?: readonly string[] | null
): string[] | undefined => {
  const normalized = normalizeFlashcardTags(tags)
  return normalized.length > 0 ? normalized : undefined
}
