export const CAPTURED_NOTE_KEYWORD = "captured"

const SOURCE_LINE_PREFIX = "Source: "

const normalizeKeyword = (keyword: string) => keyword.trim()

// Reserved marker backing the /notes Captured view. It is a normal tag added
// only to new extension capture saves; existing notes are never rewritten.
export const withCapturedNoteKeyword = (keywords: string[] = []): string[] => {
  const seen = new Set<string>()
  const normalized = keywords.reduce<string[]>((acc, keyword) => {
    const trimmed = normalizeKeyword(keyword)
    if (!trimmed) return acc
    const key = trimmed.toLocaleLowerCase()
    if (key === CAPTURED_NOTE_KEYWORD || seen.has(key)) return acc
    seen.add(key)
    acc.push(trimmed)
    return acc
  }, [])

  return [...normalized, CAPTURED_NOTE_KEYWORD]
}

export const buildCapturedNoteContent = (
  content: string,
  sourceUrl?: string | null
): string => {
  const body = content.trim()
  const source = sourceUrl?.trim()
  if (!source) return body

  const sourceLine = `${SOURCE_LINE_PREFIX}${source}`
  const existingLines = body.split(/\r?\n/)
  if (existingLines.some((line) => line.trim() === sourceLine)) {
    return body
  }

  return `${body}\n\n${sourceLine}`
}
