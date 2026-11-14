/**
 * Note wikilinks (UX review NE-02, #3110, decision D2).
 *
 * Two forms link notes: `[[Title]]` and `[[id:UUID]]`. The server resolves
 * titles when it builds backlinks and graph edges
 * (tldw_Server_API/app/core/Notes/wikilinks.py); this tokenizer must stay in
 * step with that parser. The preview asks the server how each link resolves
 * (POST /api/v1/notes/wikilinks/resolve), so a link opens the same note its
 * backlink points to. The loaded list page is only a fallback until the
 * server answers.
 */

export type WikilinkCandidate = {
  id: string
  title: string
}

export type WikilinkToken = {
  raw: string
  /** Trimmed link text. For an id link this is `id:<uuid>`. */
  title: string
  start: number
  end: number
  /** Set for `[[id:UUID]]` links: the lower-case note id. */
  noteId?: string
}

export type ActiveWikilinkQuery = {
  start: number
  end: number
  query: string
}

export type WikilinkTitleResolution = {
  noteId: string | null
  noteTitle: string | null
  candidateCount: number
}

export type WikilinkIdResolution = {
  noteId: string | null
  noteTitle: string | null
}

/** Server answers, keyed by collapsed title text and by lower-case note id. */
export type WikilinkResolutions = {
  titles: Map<string, WikilinkTitleResolution>
  ids: Map<string, WikilinkIdResolution>
}

export type WikilinkHrefTarget =
  | { kind: "note"; noteId: string }
  | { kind: "create"; title: string }

export type WikilinkRenderOptions = {
  resolutions?: WikilinkResolutions | null
  labels?: {
    /** Tooltip for an unresolved link that offers to create the note. */
    createNote?: (title: string) => string
  }
}

/** The resolve endpoint accepts at most this many titles and ids per request. */
export const MAX_WIKILINK_RESOLVE_ITEMS = 200

// `[[` + text without a newline or a nested `[[` + the first `]]` that is not
// followed by another `]`, so `[[[Draft] Proposal]]` links "[Draft] Proposal".
const WIKILINK_PATTERN = /\[\[((?:(?!\[\[)[^\n])+?)\]\](?!\])/g
const ID_PREFIX = "id:"
const CANONICAL_UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const ASCII_PUNCTUATION = /[!-/:-@[-`{-~]/g

// In-page hrefs pass the Markdown URL sanitizer, which blanks unknown schemes
// such as `note://`.
const NOTE_HREF_PREFIX = "#note:"
const CREATE_HREF_PREFIX = "#note-new:"

/**
 * Classes for the note preview container that make an unresolved link read
 * as missing: muted, with a dashed underline. Tailwind needs the literal
 * href prefix here, so it must match CREATE_HREF_PREFIX.
 */
export const MISSING_WIKILINK_PREVIEW_CLASSES =
  "[&_a[href^='#note-new:']]:text-text-muted [&_a[href^='#note-new:']]:underline [&_a[href^='#note-new:']]:decoration-dashed [&_a[href^='#note-new:']]:underline-offset-2"

export const collapseWikilinkTitle = (title: string): string =>
  String(title || "").trim().replace(/\s+/g, " ")

export const normalizeWikilinkTitle = (title: string): string =>
  collapseWikilinkTitle(title).toLowerCase()

/** Return the lower-case UUID an `[[id:...]]` link names, or null if malformed. */
export const canonicalWikilinkNoteId = (value: string | null | undefined): string | null => {
  const text = String(value || "").trim()
  return CANONICAL_UUID_PATTERN.test(text) ? text.toLowerCase() : null
}

const compareIds = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0)

export const buildWikilinkIndex = (
  candidates: WikilinkCandidate[]
): Map<string, WikilinkCandidate[]> => {
  const index = new Map<string, WikilinkCandidate[]>()
  const seen = new Set<string>()

  for (const candidate of candidates) {
    const id = String(candidate.id || "").trim()
    const title = String(candidate.title || "").trim()
    if (!id || !title) continue
    const key = normalizeWikilinkTitle(title)
    if (!key) continue
    const dedupeKey = `${key}::${id}`
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)
    const bucket = index.get(key)
    if (bucket) {
      bucket.push({ id, title })
    } else {
      index.set(key, [{ id, title }])
    }
  }

  for (const [key, bucket] of index.entries()) {
    index.set(
      key,
      bucket.sort((a, b) => a.title.localeCompare(b.title) || a.id.localeCompare(b.id))
    )
  }

  return index
}

export const tokenizeWikilinks = (content: string): WikilinkToken[] => {
  const tokens: WikilinkToken[] = []
  const input = String(content || "")
  for (const match of input.matchAll(WIKILINK_PATTERN)) {
    const raw = String(match[0] || "")
    const title = String(match[1] || "").trim()
    const index = match.index ?? -1
    if (!raw || !title || index < 0) continue
    const token: WikilinkToken = {
      raw,
      title,
      start: index,
      end: index + raw.length
    }
    if (title.startsWith(ID_PREFIX)) {
      // The `id:` prefix is reserved: a malformed id is not a link at all.
      const noteId = canonicalWikilinkNoteId(title.slice(ID_PREFIX.length))
      if (!noteId) continue
      token.noteId = noteId
    }
    tokens.push(token)
  }
  return tokens
}

/**
 * Resolve a title against locally known notes. This is only a fallback for
 * the server's answer: an exact title match wins, then the lowest id.
 */
export const resolveWikilinkTitle = (
  title: string,
  index: Map<string, WikilinkCandidate[]>
): string | null => {
  const query = String(title || "").trim()
  if (!query) return null
  const normalized = normalizeWikilinkTitle(query)
  const candidates = index.get(normalized) || []
  if (candidates.length === 0) return null
  if (candidates.length === 1) return candidates[0].id

  const exactTitleMatches = candidates.filter((candidate) => candidate.title.trim() === query)
  if (exactTitleMatches.length === 1) return exactTitleMatches[0].id

  const deterministicPool = exactTitleMatches.length > 0 ? exactTitleMatches : candidates
  return deterministicPool.slice().sort((a, b) => compareIds(a.id, b.id))[0].id
}

/** Collect the unique link texts and note ids a note's content links to. */
export const collectWikilinkTargets = (
  content: string
): { titles: string[]; ids: string[] } => {
  const titles = new Map<string, string>()
  const ids = new Set<string>()
  for (const token of tokenizeWikilinks(content)) {
    if (token.noteId) {
      if (ids.size < MAX_WIKILINK_RESOLVE_ITEMS) ids.add(token.noteId)
      continue
    }
    const key = normalizeWikilinkTitle(token.title)
    if (!key || titles.has(key) || titles.size >= MAX_WIKILINK_RESOLVE_ITEMS) continue
    titles.set(key, collapseWikilinkTitle(token.title))
  }
  return { titles: Array.from(titles.values()), ids: Array.from(ids) }
}

/** Parse the resolve endpoint's response. Returns null when there is no answer. */
export const parseWikilinkResolutions = (payload: unknown): WikilinkResolutions | null => {
  if (!payload || typeof payload !== "object") return null
  const raw = payload as { titles?: unknown; ids?: unknown }
  const resolutions: WikilinkResolutions = { titles: new Map(), ids: new Map() }

  for (const item of Array.isArray(raw.titles) ? raw.titles : []) {
    if (!item || typeof item !== "object") continue
    const entry = item as Record<string, unknown>
    if (typeof entry.title !== "string") continue
    const key = collapseWikilinkTitle(entry.title)
    if (!key) continue
    resolutions.titles.set(key, {
      noteId: typeof entry.note_id === "string" && entry.note_id ? entry.note_id : null,
      noteTitle: typeof entry.note_title === "string" ? entry.note_title : null,
      candidateCount: Number(entry.candidate_count) || 0
    })
  }

  for (const item of Array.isArray(raw.ids) ? raw.ids : []) {
    if (!item || typeof item !== "object") continue
    const entry = item as Record<string, unknown>
    const key = typeof entry.id === "string" ? canonicalWikilinkNoteId(entry.id) : null
    if (!key) continue
    resolutions.ids.set(key, {
      noteId: typeof entry.note_id === "string" && entry.note_id ? entry.note_id : null,
      noteTitle: typeof entry.note_title === "string" ? entry.note_title : null
    })
  }

  return resolutions
}

// encodeURIComponent leaves !'()* alone; parentheses would end a Markdown link.
const encodeHrefValue = (value: string): string =>
  encodeURIComponent(value).replace(
    /[!'()*]/g,
    (char) => `%${char.charCodeAt(0).toString(16).toUpperCase()}`
  )

const decodeHrefValue = (value: string): string | null => {
  try {
    return decodeURIComponent(value)
  } catch {
    return null
  }
}

export const buildWikilinkNoteHref = (noteId: string): string =>
  `${NOTE_HREF_PREFIX}${encodeHrefValue(noteId)}`

export const buildWikilinkCreateHref = (title: string): string =>
  `${CREATE_HREF_PREFIX}${encodeHrefValue(title)}`

/** Read back a wikilink href from the note preview. Other hrefs return null. */
export const parseWikilinkHref = (href: string): WikilinkHrefTarget | null => {
  const value = String(href || "")
  if (value.startsWith(CREATE_HREF_PREFIX)) {
    const title = decodeHrefValue(value.slice(CREATE_HREF_PREFIX.length))
    return title ? { kind: "create", title } : null
  }
  if (value.startsWith(NOTE_HREF_PREFIX)) {
    const noteId = decodeHrefValue(value.slice(NOTE_HREF_PREFIX.length))
    return noteId ? { kind: "note", noteId } : null
  }
  return null
}

// Numeric character references keep link text literal: backslash escapes would
// be read as LaTeX delimiters by the Markdown component's preprocessor.
const escapeMarkdownText = (text: string): string =>
  text.replace(ASCII_PUNCTUATION, (char) => `&#${char.charCodeAt(0)};`)

const defaultCreateNoteLabel = (title: string): string => `Create note: ${title}`

type WikilinkRenderTarget =
  | { kind: "note"; noteId: string; label: string }
  | { kind: "create"; title: string }
  | { kind: "plain" }

const resolveRenderTarget = (
  token: WikilinkToken,
  index: Map<string, WikilinkCandidate[]>,
  resolutions: WikilinkResolutions | null | undefined
): WikilinkRenderTarget => {
  if (token.noteId) {
    const answer = resolutions?.ids.get(token.noteId)
    if (answer) {
      // The server confirmed there is no live note with this id.
      if (!answer.noteId) return { kind: "plain" }
      return {
        kind: "note",
        noteId: answer.noteId,
        label: answer.noteTitle ? `[[${answer.noteTitle}]]` : token.raw
      }
    }
    return { kind: "note", noteId: token.noteId, label: token.raw }
  }

  const answer = resolutions?.titles.get(collapseWikilinkTitle(token.title))
  if (answer) {
    return answer.noteId
      ? { kind: "note", noteId: answer.noteId, label: token.raw }
      : { kind: "create", title: collapseWikilinkTitle(token.title) }
  }
  const localNoteId = resolveWikilinkTitle(token.title, index)
  // Without a server answer an unknown title may still exist beyond the
  // loaded page, so it stays plain text instead of offering to create it.
  return localNoteId ? { kind: "note", noteId: localNoteId, label: token.raw } : { kind: "plain" }
}

/**
 * Rewrite wikilinks in note content as Markdown links for the preview.
 * A resolved link points at `#note:<id>`, and a title the server confirmed
 * missing points at `#note-new:<title>` (see `parseWikilinkHref`).
 */
export const renderContentWithResolvedWikilinks = (
  content: string,
  index: Map<string, WikilinkCandidate[]>,
  options: WikilinkRenderOptions = {}
): string => {
  const input = String(content || "")
  const tokens = tokenizeWikilinks(input)
  if (tokens.length === 0) return input

  const createNoteLabel = options.labels?.createNote ?? defaultCreateNoteLabel
  let cursor = 0
  let output = ""
  for (const token of tokens) {
    output += input.slice(cursor, token.start)
    const target = resolveRenderTarget(token, index, options.resolutions)
    if (target.kind === "note") {
      output += `[${escapeMarkdownText(target.label)}](${buildWikilinkNoteHref(target.noteId)})`
    } else if (target.kind === "create") {
      const tooltip = escapeMarkdownText(createNoteLabel(target.title))
      output += `[${escapeMarkdownText(token.raw)}](${buildWikilinkCreateHref(target.title)} "${tooltip}")`
    } else {
      output += token.raw
    }
    cursor = token.end
  }
  output += input.slice(cursor)
  return output
}

export const getActiveWikilinkQuery = (
  content: string,
  cursorIndex: number
): ActiveWikilinkQuery | null => {
  const input = String(content || "")
  const safeCursor = Math.max(0, Math.min(cursorIndex, input.length))
  const start = input.lastIndexOf("[[", safeCursor)
  if (start < 0) return null

  const closeBeforeCursor = input.indexOf("]]", start + 2)
  if (closeBeforeCursor >= 0 && closeBeforeCursor < safeCursor) return null

  const rawQuery = input.slice(start + 2, safeCursor)
  if (!rawQuery) {
    return { start, end: safeCursor, query: "" }
  }
  if (rawQuery.includes("\n")) return null
  if (rawQuery.includes("[") || rawQuery.includes("]")) return null

  return {
    start,
    end: safeCursor,
    query: rawQuery
  }
}

/**
 * Replace the active `[[query` with a link. Pass `noteId` when several notes
 * share the title, so the link names one of them: `[[id:<uuid>]]`.
 */
export const insertWikilinkAtCursor = (
  content: string,
  activeQuery: ActiveWikilinkQuery,
  title: string,
  noteId?: string | null
): { content: string; cursor: number } => {
  const nextTitle = String(title || "").trim()
  const canonicalId = canonicalWikilinkNoteId(noteId)
  const replacement = canonicalId ? `[[${ID_PREFIX}${canonicalId}]]` : `[[${nextTitle}]]`
  const nextContent = `${content.slice(0, activeQuery.start)}${replacement}${content.slice(activeQuery.end)}`
  const nextCursor = activeQuery.start + replacement.length
  return { content: nextContent, cursor: nextCursor }
}
