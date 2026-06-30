// Reserved durable tag used by extension capture flows to back the /notes Inbox view.
// Capture creates add this marker; existing notes are not rewritten or migrated.
export const CAPTURED_NOTE_KEYWORD = "captured"

export const ensureCapturedNoteKeyword = (keywords: string[]): string[] => {
  const next: string[] = []
  const seen = new Set<string>()

  for (const keyword of keywords) {
    const trimmed = String(keyword || "").trim()
    if (!trimmed) continue

    const key = trimmed.toLowerCase()
    if (key === CAPTURED_NOTE_KEYWORD) {
      if (!seen.has(CAPTURED_NOTE_KEYWORD)) {
        next.push(CAPTURED_NOTE_KEYWORD)
        seen.add(CAPTURED_NOTE_KEYWORD)
      }
      continue
    }

    if (seen.has(key)) continue
    next.push(trimmed)
    seen.add(key)
  }

  if (!seen.has(CAPTURED_NOTE_KEYWORD)) {
    next.push(CAPTURED_NOTE_KEYWORD)
  }

  return next
}
