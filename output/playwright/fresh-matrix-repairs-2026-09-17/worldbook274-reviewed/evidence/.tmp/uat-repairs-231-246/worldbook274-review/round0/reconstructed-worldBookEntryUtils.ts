export type PriorityBand = "low" | "medium" | "high"

export const normalizeKeywordList = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean)
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
  }
  return []
}

export const normalizeWorldBookEntryIdentifier = <T>(entry: T): T => {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return entry
  const record = entry as T & Record<string, unknown>
  if (record.entry_id != null) return entry

  const canonicalId = Number(record.id)
  if (!Number.isSafeInteger(canonicalId) || canonicalId <= 0) return entry

  return { ...record, entry_id: canonicalId }
}

export const estimateEntryTokens = (content: unknown): number => {
  const chars = String(content || "").length
  if (chars === 0) return 0
  return Math.ceil(chars / 4)
}

export const formatEntryContentStats = (content: unknown): string => {
  const chars = String(content || "").length
  const tokens = estimateEntryTokens(content)
  return `${chars} chars / ~${tokens} tokens`
}

export const getPriorityBand = (value: unknown): PriorityBand => {
  const priority = Number(value)
  if (!Number.isFinite(priority)) return "low"
  if (priority >= 67) return "high"
  if (priority >= 34) return "medium"
  return "low"
}

export const getPriorityTagColor = (band: PriorityBand): string => {
  if (band === "high") return "green"
  if (band === "medium") return "blue"
  return "default"
}

export const validateRegexKeywords = (value: unknown): string | null => {
  const keywords = normalizeKeywordList(value)
  for (const keyword of keywords) {
    try {
      // Validate syntax only; safety checks remain backend-authoritative.
      // eslint-disable-next-line no-new
      new RegExp(keyword)
    } catch (error: any) {
      return `Invalid regex pattern: ${error?.message || keyword}`
    }
  }
  return null
}
