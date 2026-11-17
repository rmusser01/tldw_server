export type DictionaryListItem = {
  id?: number
  name?: string | null
  description?: string | null
  category?: string | null
  tags?: string[] | string | null
  is_active?: boolean | null
  entry_count?: number | null
  regex_entries?: number | null
  regex_entry_count?: number | null
  used_by_chat_count?: number | null
  used_by_active_chat_count?: number | null
  updated_at?: string | Date | null
}

export type DictionaryConfirmationConfig = {
  title: string
  content: string
  okText: string
  cancelText: string
}

function safeText(value: unknown): string {
  if (typeof value !== "string") return ""
  return value.trim().toLowerCase()
}

export function normalizeDictionaryTags(value: unknown): string[] {
  if (!value) return []
  if (Array.isArray(value)) {
    const seen = new Set<string>()
    const tags: string[] = []
    for (const item of value) {
      const tag =
        typeof item === "string" ? item.trim() : String(item || "").trim()
      if (!tag) continue
      const key = tag.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)
      tags.push(tag)
    }
    return tags
  }
  if (typeof value === "string") {
    const normalized = value.trim()
    if (!normalized) return []
    try {
      const parsed = JSON.parse(normalized)
      if (Array.isArray(parsed)) return normalizeDictionaryTags(parsed)
    } catch {
      // noop, fall through to comma-split parsing
    }
    return normalizeDictionaryTags(normalized.split(","))
  }
  return []
}

export function compareDictionaryName(
  a: DictionaryListItem,
  b: DictionaryListItem
): number {
  const nameA = String(a.name || "").toLocaleLowerCase()
  const nameB = String(b.name || "").toLocaleLowerCase()
  return nameA.localeCompare(nameB)
}

export function compareDictionaryEntryCount(
  a: DictionaryListItem,
  b: DictionaryListItem
): number {
  const countA = Number(a.entry_count || 0)
  const countB = Number(b.entry_count || 0)
  return countA - countB
}

export function compareDictionaryActive(
  a: DictionaryListItem,
  b: DictionaryListItem
): number {
  const activeA = Boolean(a.is_active)
  const activeB = Boolean(b.is_active)
  if (activeA === activeB) return 0
  return activeA ? 1 : -1
}

export function filterDictionariesBySearch(
  dictionaries: DictionaryListItem[],
  query: string
): DictionaryListItem[] {
  const normalized = safeText(query)
  if (!normalized) return dictionaries
  return dictionaries.filter((item) => {
    const name = safeText(item.name)
    const description = safeText(item.description)
    const category = safeText(item.category)
    const tags = normalizeDictionaryTags(item.tags).map(safeText).join(" ")
    return (
      name.includes(normalized) ||
      description.includes(normalized) ||
      category.includes(normalized) ||
      tags.includes(normalized)
    )
  })
}

function formatQuantity(value: number, unit: string, future: boolean): string {
  const label = `${value} ${unit}${value === 1 ? "" : "s"}`
  return future ? `in ${label}` : `${label} ago`
}

export function parseDictionaryTimestamp(
  input: string | Date | null | undefined
): Date | null {
  if (!input) return null
  if (input instanceof Date) return Number.isNaN(input.getTime()) ? null : input
  if (typeof input !== "string") return null
  const text = input.trim()
  // The deployed dictionary API emits UTC without a zone.
  const value = new Date(
    /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?$/.test(text)
      ? `${text.replace(" ", "T")}Z`
      : text
  )
  return Number.isNaN(value.getTime()) ? null : value
}

export function formatRelativeTimestamp(
  input: string | Date | null | undefined,
  now: Date = new Date()
): string {
  const value = parseDictionaryTimestamp(input)
  if (!value) return "—"

  const diffMs = now.getTime() - value.getTime()
  const absMs = Math.abs(diffMs)
  const seconds = Math.floor(absMs / 1000)

  if (seconds < 30) return "just now"
  if (seconds < 60) return formatQuantity(seconds, "second", diffMs < 0)

  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return formatQuantity(minutes, "minute", diffMs < 0)

  const hours = Math.floor(minutes / 60)
  if (hours < 24) return formatQuantity(hours, "hour", diffMs < 0)

  const days = Math.floor(hours / 24)
  if (days < 30) return formatQuantity(days, "day", diffMs < 0)

  return value.toLocaleDateString()
}

function normalizeExistingNames(
  existingNames: Array<string | null | undefined>
): Set<string> {
  const normalized = new Set<string>()
  for (const name of existingNames) {
    const value = safeText(name)
    if (value) normalized.add(value)
  }
  return normalized
}

export function buildDuplicateDictionaryName(
  baseName: string,
  existingNames: Array<string | null | undefined>
): string {
  const cleanBase = baseName.trim() || "Dictionary"
  const normalizedExisting = normalizeExistingNames(existingNames)
  const copyBase = `${cleanBase} (copy)`
  if (!normalizedExisting.has(copyBase.toLowerCase())) {
    return copyBase
  }

  let suffix = 2
  while (suffix < 10_000) {
    const candidate = `${cleanBase} (copy ${suffix})`
    if (!normalizedExisting.has(candidate.toLowerCase())) {
      return candidate
    }
    suffix += 1
  }
  return `${cleanBase} (copy ${Date.now()})`
}

export function formatDictionaryUsageLabel(
  dictionary: DictionaryListItem
): string {
  const totalChats = Number(dictionary?.used_by_chat_count || 0)
  if (totalChats <= 0) return "—"
  const activeChats = Number(dictionary?.used_by_active_chat_count || 0)
  if (activeChats > 0) return `${totalChats} chats (${activeChats} active)`
  return `${totalChats} chats`
}

export function resolveDictionaryChatReferenceId(chatRef: unknown): string {
  if (!chatRef || typeof chatRef !== "object") return ""
  const asRecord = chatRef as Record<string, unknown>
  const raw = asRecord.chat_id ?? asRecord.id
  if (
    typeof raw !== "string" &&
    !(typeof raw === "number" && Number.isSafeInteger(raw) && raw > 0)
  )
    return ""
  const id = String(raw).trim()
  return /^[A-Za-z0-9_-]+$/.test(id) ? id : ""
}

export function formatDictionaryChatReferenceTitle(chatRef: unknown): string {
  const chatId = resolveDictionaryChatReferenceId(chatRef)
  const shortId = chatId.length > 8 ? chatId.slice(0, 8) : chatId
  if (chatRef && typeof chatRef === "object") {
    const title = String(
      (chatRef as Record<string, unknown>).title || ""
    ).trim()
    if (title) return title
  }
  return shortId ? `Chat ${shortId}` : "Chat"
}

export type DictionaryChatState =
  | "in-progress"
  | "resolved"
  | "backlog"
  | "non-viable"

export function normalizeDictionaryChatState(
  value: unknown
): DictionaryChatState {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
  if (normalized === "resolved") return "resolved"
  if (normalized === "backlog") return "backlog"
  if (normalized === "non-viable") return "non-viable"
  return "in-progress"
}

export function buildDictionaryDeactivationWarning(
  dictionary: DictionaryListItem,
  cancelText: string
): DictionaryConfirmationConfig | null {
  const activeChats = Number(dictionary?.used_by_active_chat_count || 0)
  if (activeChats <= 0) return null

  const totalChats = Number(dictionary?.used_by_chat_count || 0)
  const activeLabel =
    activeChats === 1
      ? "1 active chat session"
      : `${activeChats} active chat sessions`
  const totalLabel =
    totalChats === 1
      ? "1 linked chat session"
      : `${totalChats} linked chat sessions`

  return {
    title: "Deactivate dictionary?",
    content: `This dictionary is currently used by ${activeLabel} (${totalLabel} total). Deactivating may change active conversations.`,
    okText: "Deactivate",
    cancelText
  }
}

export function buildDictionaryDeletionConfirmationCopy(
  dictionary: DictionaryListItem
): string {
  const linkedChats = Number(dictionary?.used_by_chat_count || 0)
  const activeLinkedChats = Number(dictionary?.used_by_active_chat_count || 0)
  if (activeLinkedChats > 0) {
    return `Delete dictionary? This dictionary is linked to ${linkedChats} chat session(s), including ${activeLinkedChats} active session(s).`
  }
  if (linkedChats > 0) {
    return `Delete dictionary? This dictionary is linked to ${linkedChats} chat session(s).`
  }
  return "Delete dictionary?"
}

export function isDictionaryVersionConflictError(error: unknown): boolean {
  const message =
    error instanceof Error && error.message
      ? error.message.toLowerCase()
      : String(error || "").toLowerCase()

  if (!message) return false
  if (message.includes("already exists")) return false

  return (
    message.includes("modified by another session") ||
    message.includes("expected version") ||
    (message.includes("conflict") && message.includes("version"))
  )
}
