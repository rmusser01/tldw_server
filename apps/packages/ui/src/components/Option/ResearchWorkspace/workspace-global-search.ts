import type { Message } from "@/store/option"
import type { WorkspaceNote, WorkspaceSource } from "@/types/workspace"

export type WorkspaceGlobalSearchDomain = "source" | "chat" | "note"

export type WorkspaceGlobalSearchFilter = WorkspaceGlobalSearchDomain | null

export interface WorkspaceGlobalSearchQuery {
  raw: string
  normalized: string
  terms: string[]
  filter: WorkspaceGlobalSearchFilter
}

export interface WorkspaceGlobalSearchResult {
  id: string
  domain: WorkspaceGlobalSearchDomain
  title: string
  subtitle: string
  snippet: string
  score: number
  sourceId?: string
  chatMessageId?: string
  noteId?: number
  noteField?: "title" | "content"
}

export interface WorkspaceGlobalSearchNoteDocument {
  id?: number
  title: string
  content: string
  keywords?: string[]
  isDraft?: boolean
}

const DOMAIN_PREFIX_TO_FILTER: Record<string, WorkspaceGlobalSearchDomain> = {
  s: "source",
  source: "source",
  sources: "source",
  c: "chat",
  chat: "chat",
  chats: "chat",
  m: "chat",
  message: "chat",
  messages: "chat",
  n: "note",
  note: "note",
  notes: "note"
}

const SCORE_DOMAIN_BOOST: Record<WorkspaceGlobalSearchDomain, number> = {
  source: 240,
  note: 210,
  chat: 170
}

const tokenize = (value: string): string[] =>
  value
    .toLowerCase()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean)

export const parseWorkspaceGlobalSearchQuery = (
  rawQuery: string
): WorkspaceGlobalSearchQuery => {
  const trimmed = rawQuery.trim()
  const prefixMatch = /^(\w+):\s*(.*)$/.exec(trimmed)
  if (!prefixMatch) {
    const normalized = trimmed.toLowerCase()
    return {
      raw: rawQuery,
      normalized,
      terms: tokenize(normalized),
      filter: null
    }
  }

  const [, rawPrefix, remainder] = prefixMatch
  const prefix = rawPrefix.toLowerCase()
  const filter = DOMAIN_PREFIX_TO_FILTER[prefix]

  if (!filter) {
    const normalized = trimmed.toLowerCase()
    return {
      raw: rawQuery,
      normalized,
      terms: tokenize(normalized),
      filter: null
    }
  }

  const normalized = remainder.trim().toLowerCase()
  return {
    raw: rawQuery,
    normalized,
    terms: tokenize(normalized),
    filter
  }
}

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

const buildSnippet = (text: string, terms: string[]): string => {
  const normalizedText = text.replace(/\s+/g, " ").trim()
  if (!normalizedText) return ""
  if (terms.length === 0) return normalizedText.slice(0, 140)

  let firstMatchIndex = -1
  for (const term of terms) {
    const index = normalizedText.toLowerCase().indexOf(term)
    if (index >= 0 && (firstMatchIndex < 0 || index < firstMatchIndex)) {
      firstMatchIndex = index
    }
  }

  if (firstMatchIndex < 0) {
    return normalizedText.slice(0, 140)
  }

  const start = Math.max(0, firstMatchIndex - 48)
  const end = Math.min(normalizedText.length, firstMatchIndex + 92)
  const slice = normalizedText.slice(start, end)
  return `${start > 0 ? "..." : ""}${slice}${end < normalizedText.length ? "..." : ""}`
}

const scoreText = (text: string, terms: string[]): number => {
  if (!text.trim() || terms.length === 0) return 0
  const normalizedText = text.toLowerCase()

  let score = 0
  const phrase = terms.join(" ").trim()
  if (phrase && normalizedText.includes(phrase)) {
    score += 120
  }

  for (const term of terms) {
    const escaped = escapeRegExp(term)
    const wordBoundary = new RegExp(`\\b${escaped}`)
    if (wordBoundary.test(normalizedText)) {
      score += 45
      continue
    }
    if (normalizedText.includes(term)) {
      score += 20
    }
  }

  return score
}

export const getWorkspaceChatSearchMessageId = (
  message: Pick<Message, "id" | "serverMessageId" | "createdAt">,
  index: number
): string => {
  const directId = message.id?.trim()
  if (directId) return `msg:${directId}`

  const serverId = message.serverMessageId?.trim()
  if (serverId) return `server:${serverId}`

  const createdAt =
    typeof message.createdAt === "number" && Number.isFinite(message.createdAt)
      ? String(Math.round(message.createdAt))
      : "na"
  return `idx:${index}:${createdAt}`
}

interface BuildWorkspaceGlobalSearchResultsInput {
  query: string
  sources: WorkspaceSource[]
  chatMessages: Message[]
  currentNote: WorkspaceNote | null | undefined
  workspaceNotes?: WorkspaceGlobalSearchNoteDocument[]
  limit?: number
}

const allowDomain = (
  filter: WorkspaceGlobalSearchFilter,
  domain: WorkspaceGlobalSearchDomain
): boolean => !filter || filter === domain

export const buildWorkspaceGlobalSearchResults = ({
  query,
  sources,
  chatMessages,
  currentNote,
  workspaceNotes = [],
  limit = 30
}: BuildWorkspaceGlobalSearchResultsInput): WorkspaceGlobalSearchResult[] => {
  const parsedQuery = parseWorkspaceGlobalSearchQuery(query)
  if (parsedQuery.terms.length === 0) return []

  const results: WorkspaceGlobalSearchResult[] = []

  if (allowDomain(parsedQuery.filter, "source")) {
    for (const source of sources) {
      const titleScore = scoreText(source.title, parsedQuery.terms)
      const typeScore = scoreText(source.type, parsedQuery.terms)
      if (titleScore === 0 && typeScore === 0) continue

      results.push({
        id: `source:${source.id}`,
        domain: "source",
        title: source.title,
        subtitle: `Source · ${source.type}`,
        snippet: source.url || "",
        score: SCORE_DOMAIN_BOOST.source + titleScore * 2 + typeScore,
        sourceId: source.id
      })
    }
  }

  if (allowDomain(parsedQuery.filter, "chat")) {
    chatMessages.forEach((message, index) => {
      const text = String(message.message || "").trim()
      if (!text) return
      const textScore = scoreText(text, parsedQuery.terms)
      if (textScore === 0) return

      const chatMessageId = getWorkspaceChatSearchMessageId(message, index)
      results.push({
        id: `chat:${chatMessageId}`,
        domain: "chat",
        title: message.isBot ? "Assistant message" : "Your message",
        subtitle: "Chat",
        snippet: buildSnippet(text, parsedQuery.terms),
        score: SCORE_DOMAIN_BOOST.chat + textScore,
        chatMessageId
      })
    })
  }

  if (allowDomain(parsedQuery.filter, "note")) {
    const noteDocuments: WorkspaceGlobalSearchNoteDocument[] = []
    const seenNoteIds = new Set<number>()

    if (currentNote) {
      noteDocuments.push({
        id: currentNote.id,
        title: currentNote.title || "",
        content: currentNote.content || "",
        keywords: currentNote.keywords || [],
        isDraft: currentNote.id == null
      })
      if (typeof currentNote.id === "number" && Number.isFinite(currentNote.id)) {
        seenNoteIds.add(currentNote.id)
      }
    }

    for (const workspaceNote of workspaceNotes) {
      const workspaceNoteId =
        typeof workspaceNote.id === "number" && Number.isFinite(workspaceNote.id)
          ? workspaceNote.id
          : undefined
      if (workspaceNoteId != null && seenNoteIds.has(workspaceNoteId)) {
        continue
      }
      noteDocuments.push({
        ...workspaceNote,
        id: workspaceNoteId
      })
      if (workspaceNoteId != null) {
        seenNoteIds.add(workspaceNoteId)
      }
    }

    for (const noteDocument of noteDocuments) {
      const noteTitle = noteDocument.title || ""
      const noteContent = noteDocument.content || ""
      const noteKeywords = (noteDocument.keywords || []).join(" ")

      const titleScore = scoreText(noteTitle, parsedQuery.terms)
      const contentScore = scoreText(noteContent, parsedQuery.terms)
      const keywordScore = scoreText(noteKeywords, parsedQuery.terms)

      if (titleScore === 0 && contentScore === 0 && keywordScore === 0) {
        continue
      }

      const field: "title" | "content" = titleScore >= contentScore ? "title" : "content"
      const snippetSource = field === "title" ? noteTitle : noteContent
      const noteLabel =
        noteTitle.trim() ||
        (noteDocument.id != null ? `Note #${noteDocument.id}` : "Quick note")
      const noteIdKey = noteDocument.id ?? "draft"
      const subtitle = noteDocument.isDraft
        ? field === "title"
          ? "Draft note title"
          : "Draft note content"
        : field === "title"
          ? "Note title"
          : "Note content"

      results.push({
        id: `note:${noteIdKey}:${field}`,
        domain: "note",
        title: noteLabel,
        subtitle,
        snippet: buildSnippet(snippetSource, parsedQuery.terms),
        score:
          SCORE_DOMAIN_BOOST.note +
          Math.max(titleScore * 2, contentScore) +
          Math.round(keywordScore / 2),
        noteId: noteDocument.id,
        noteField: field
      })
    }
  }

  return results
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score
      if (a.domain !== b.domain) return a.domain.localeCompare(b.domain)
      return a.title.localeCompare(b.title)
    })
    .slice(0, Math.max(1, limit))
}
