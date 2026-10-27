/**
 * Client-side tag identity for Collections tags, which the server keeps
 * case-insensitive with first-seen display casing (TASK-13950.5).
 *
 * NFC plus lowercase is close to the server's NFC plus casefold; the server
 * also folds ß/SS and dedupes again, so this only keeps the UI from showing a
 * case variant as a second chip.
 */
export const tagIdentity = (tag: string): string => tag.trim().normalize("NFC").toLowerCase()

/** Trim tags and drop empties and case variants, keeping the first-seen casing. */
export const dedupeTags = (tags: readonly string[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of tags) {
    const tag = String(raw ?? "").trim()
    const key = tagIdentity(tag)
    if (!tag || seen.has(key)) continue
    seen.add(key)
    out.push(tag)
  }
  return out
}

/** Whether ``tags`` already holds ``tag`` in any case. */
export const hasTag = (tags: readonly string[], tag: string): boolean =>
  tags.some((existing) => tagIdentity(existing) === tagIdentity(tag))
