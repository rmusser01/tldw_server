import { createSafeStorage } from "@/utils/safe-storage"

export const SIDEPANEL_CHAT_HANDOFF_TTL_MS = 10 * 60 * 1000
export const SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX =
  "tldw:sidepanel-chat-handoff:"
export const SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPETS = 4
export const SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPET_CHARS = 4_000
export const SIDEPANEL_CHAT_HANDOFF_MAX_TOTAL_SNIPPET_CHARS = 16_000
export const SIDEPANEL_CHAT_HANDOFF_MAX_DRAFT_CHARS = 32_000
export const SIDEPANEL_CHAT_HANDOFF_MAX_TITLE_CHARS = 512
export const SIDEPANEL_CHAT_HANDOFF_MAX_URL_CHARS = 2_048
export const SIDEPANEL_CHAT_HANDOFF_MAX_ROUTE_PATH_CHARS = 2_048
export const SIDEPANEL_CHAT_HANDOFF_MAX_ROUTE_CHARACTER_ID_CHARS = 512

export type SidepanelChatHandoffSnippet = {
  kind: "selection" | "visible-context" | "captured-snippet"
  text: string
  label?: string
  truncated?: boolean
}

export type SidepanelChatHandoffPageContext = {
  title?: string
  url?: string
  snippets: SidepanelChatHandoffSnippet[]
  truncated?: boolean
}

export type SidepanelChatHandoffPackage = {
  id: string
  source: "sidepanel-chat"
  createdAt: string
  expiresAt: string
  consumedAt?: string
  draft: { text: string; truncated?: boolean }
  pageContext?: SidepanelChatHandoffPageContext
  routeIntent?: {
    path: string
    mode?: "character"
    characterId?: string
  }
}

export type CreateSidepanelChatHandoffInput = {
  draftText: string
  pageContext?: {
    title?: string
    url?: string
    snippets?: SidepanelChatHandoffSnippet[]
  }
  routeIntent?: SidepanelChatHandoffPackage["routeIntent"]
}

const storage = createSafeStorage({ area: "local" })

const storageKey = (id: string) => `${SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX}${id}`

const idFromStorageKey = (key: string): string | null => {
  if (!key.startsWith(SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX)) return null
  const id = key.slice(SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX.length)
  return id.length > 0 ? id : null
}

const snippetKinds = new Set<SidepanelChatHandoffSnippet["kind"]>([
  "selection",
  "visible-context",
  "captured-snippet"
])

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isValidDateString = (value: unknown): value is string =>
  typeof value === "string" && Number.isFinite(Date.parse(value))

const isTestEnvironment = () => {
  const maybeProcess = (
    globalThis as {
      process?: { env?: Record<string, string | undefined> }
    }
  ).process

  return (
    maybeProcess?.env?.VITEST === "true" ||
    maybeProcess?.env?.NODE_ENV === "test"
  )
}

const createHandoffId = (): string => {
  const randomUUID = globalThis.crypto?.randomUUID
  if (typeof randomUUID === "function") {
    return randomUUID.call(globalThis.crypto)
  }

  if (isTestEnvironment()) {
    return `test-${Date.now().toString(36)}-${Math.random()
      .toString(36)
      .slice(2, 12)}`
  }

  throw new Error("crypto.randomUUID is required to create sidepanel chat handoffs.")
}

const truncateText = (
  text: string,
  maxChars: number
): { text: string; truncated?: boolean } => {
  if (text.length <= maxChars) return { text }
  return { text: text.slice(0, maxChars), truncated: true }
}

const isValidSnippetKind = (
  kind: unknown
): kind is SidepanelChatHandoffSnippet["kind"] =>
  typeof kind === "string" && snippetKinds.has(kind as SidepanelChatHandoffSnippet["kind"])

const isValidSnippet = (
  snippet: unknown
): snippet is SidepanelChatHandoffSnippet => {
  if (!isRecord(snippet)) return false
  if (!isValidSnippetKind(snippet.kind)) return false
  if (typeof snippet.text !== "string") return false
  if (snippet.label != null && typeof snippet.label !== "string") return false
  if (snippet.truncated != null && typeof snippet.truncated !== "boolean") {
    return false
  }
  return true
}

const buildBoundedSnippets = (
  snippets: SidepanelChatHandoffSnippet[] = []
): { snippets: SidepanelChatHandoffSnippet[]; truncated: boolean } => {
  let remainingChars = SIDEPANEL_CHAT_HANDOFF_MAX_TOTAL_SNIPPET_CHARS
  let truncated = snippets.length > SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPETS
  const bounded: SidepanelChatHandoffSnippet[] = []

  for (const snippet of snippets.slice(0, SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPETS)) {
    if (!isValidSnippet(snippet) || remainingChars <= 0) {
      truncated = true
      continue
    }

    const maxChars = Math.min(
      SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPET_CHARS,
      remainingChars
    )
    const boundedText = snippet.text.slice(0, maxChars)
    const wasTruncated = snippet.truncated || boundedText.length < snippet.text.length
    remainingChars -= boundedText.length

    bounded.push({
      kind: snippet.kind,
      text: boundedText,
      ...(snippet.label != null ? { label: snippet.label } : {}),
      ...(wasTruncated ? { truncated: true } : {})
    })

    if (wasTruncated) truncated = true
  }

  if (bounded.length < snippets.length) truncated = true

  return { snippets: bounded, truncated }
}

const buildPageContext = (
  pageContext: CreateSidepanelChatHandoffInput["pageContext"]
): SidepanelChatHandoffPageContext | undefined => {
  if (!pageContext) return undefined

  const { snippets, truncated: snippetsTruncated } = buildBoundedSnippets(
    pageContext.snippets
  )
  const title =
    pageContext.title == null
      ? undefined
      : truncateText(pageContext.title, SIDEPANEL_CHAT_HANDOFF_MAX_TITLE_CHARS)
  const url =
    pageContext.url == null
      ? undefined
      : truncateText(pageContext.url, SIDEPANEL_CHAT_HANDOFF_MAX_URL_CHARS)
  const truncated =
    snippetsTruncated || Boolean(title?.truncated) || Boolean(url?.truncated)

  return {
    ...(title ? { title: title.text } : {}),
    ...(url ? { url: url.text } : {}),
    snippets,
    ...(truncated ? { truncated: true } : {})
  }
}

const buildRouteIntent = (
  routeIntent: CreateSidepanelChatHandoffInput["routeIntent"]
): SidepanelChatHandoffPackage["routeIntent"] => {
  if (!routeIntent) return undefined
  const path = truncateText(
    routeIntent.path,
    SIDEPANEL_CHAT_HANDOFF_MAX_ROUTE_PATH_CHARS
  )
  const characterId =
    routeIntent.characterId == null
      ? undefined
      : truncateText(
          routeIntent.characterId,
          SIDEPANEL_CHAT_HANDOFF_MAX_ROUTE_CHARACTER_ID_CHARS
        )

  return {
    path: path.text,
    ...(routeIntent.mode === "character" ? { mode: "character" as const } : {}),
    ...(characterId ? { characterId: characterId.text } : {})
  }
}

const buildPackage = (
  input: CreateSidepanelChatHandoffInput,
  now: number
): SidepanelChatHandoffPackage => {
  const draft = truncateText(
    String(input.draftText ?? ""),
    SIDEPANEL_CHAT_HANDOFF_MAX_DRAFT_CHARS
  )
  const pageContext = buildPageContext(input.pageContext)
  const routeIntent = buildRouteIntent(input.routeIntent)

  return {
    id: createHandoffId(),
    source: "sidepanel-chat",
    createdAt: new Date(now).toISOString(),
    expiresAt: new Date(now + SIDEPANEL_CHAT_HANDOFF_TTL_MS).toISOString(),
    draft,
    ...(pageContext ? { pageContext } : {}),
    ...(routeIntent ? { routeIntent } : {})
  }
}

const parsePageContext = (
  value: unknown
): SidepanelChatHandoffPageContext | null => {
  if (!isRecord(value)) return null
  let title: string | undefined
  let url: string | undefined
  let truncated: boolean | undefined

  if (value.title != null) {
    if (typeof value.title !== "string") return null
    title = value.title
  }
  if (value.url != null) {
    if (typeof value.url !== "string") return null
    url = value.url
  }
  if (!Array.isArray(value.snippets)) return null
  if (!value.snippets.every(isValidSnippet)) return null
  const snippets: SidepanelChatHandoffSnippet[] = value.snippets
  if (value.truncated != null) {
    if (typeof value.truncated !== "boolean") return null
    truncated = value.truncated
  }

  return {
    ...(title ? { title } : {}),
    ...(url ? { url } : {}),
    snippets,
    ...(truncated != null ? { truncated } : {})
  }
}

const parseRouteIntent = (
  value: unknown
): SidepanelChatHandoffPackage["routeIntent"] | null => {
  if (!isRecord(value)) return null
  let characterId: string | undefined

  if (typeof value.path !== "string") return null
  const path: string = value.path
  if (value.mode != null && value.mode !== "character") return null
  const mode = value.mode === "character" ? value.mode : undefined
  if (value.characterId != null) {
    if (typeof value.characterId !== "string") return null
    characterId = value.characterId
  }

  return {
    path,
    ...(mode === "character" ? { mode: "character" } : {}),
    ...(characterId != null ? { characterId } : {})
  }
}

const normalizeStoredPackageValue = (value: unknown): unknown => {
  if (typeof value !== "string") return value

  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

const parsePackage = (
  storedValue: unknown,
  expectedId?: string
): SidepanelChatHandoffPackage | null => {
  const value = normalizeStoredPackageValue(storedValue)

  if (!isRecord(value)) return null
  if (typeof value.id !== "string" || value.id.length === 0) return null
  const id = value.id
  if (expectedId != null && id !== expectedId) return null
  if (value.source !== "sidepanel-chat") return null
  if (!isValidDateString(value.createdAt)) return null
  const createdAt = value.createdAt
  if (!isValidDateString(value.expiresAt)) return null
  const expiresAt = value.expiresAt
  let consumedAt: string | undefined
  if (value.consumedAt != null) {
    if (!isValidDateString(value.consumedAt)) return null
    consumedAt = value.consumedAt
  }
  if (!isRecord(value.draft)) return null
  if (typeof value.draft.text !== "string") return null
  const draftText: string = value.draft.text
  let draftTruncated: boolean | undefined
  if (value.draft.truncated != null) {
    if (typeof value.draft.truncated !== "boolean") {
      return null
    }
    draftTruncated = value.draft.truncated
  }

  const pageContext =
    value.pageContext == null ? undefined : parsePageContext(value.pageContext)
  if (value.pageContext != null && !pageContext) return null

  const routeIntent =
    value.routeIntent == null ? undefined : parseRouteIntent(value.routeIntent)
  if (value.routeIntent != null && !routeIntent) return null

  return {
    id,
    source: "sidepanel-chat",
    createdAt,
    expiresAt,
    ...(consumedAt ? { consumedAt } : {}),
    draft: {
      text: draftText,
      ...(draftTruncated != null ? { truncated: draftTruncated } : {})
    },
    ...(pageContext ? { pageContext } : {}),
    ...(routeIntent ? { routeIntent } : {})
  }
}

const readRawPackage = async (
  id: string
): Promise<SidepanelChatHandoffPackage | null> => {
  const raw = await storage.get(storageKey(id))
  return parsePackage(raw, id)
}

const isExpired = (pkg: SidepanelChatHandoffPackage, now = Date.now()) =>
  Date.parse(pkg.expiresAt) <= now

const removeKeyQuietly = async (key: string) => {
  try {
    await storage.remove(key)
  } catch {
    // Best-effort cleanup after failed writes or malformed records.
  }
}

export const cleanupExpiredSidepanelChatHandoffs = async (): Promise<number> => {
  try {
    const entries = await storage.getAll()
    const now = Date.now()
    const keysToRemove = Object.entries(entries)
      .filter(([key]) => key.startsWith(SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX))
      .filter(([key, value]) => {
        const expectedId = idFromStorageKey(key)
        const pkg = expectedId ? parsePackage(value, expectedId) : null
        return !pkg || pkg.consumedAt != null || isExpired(pkg, now)
      })
      .map(([key]) => key)

    if (keysToRemove.length > 0) {
      await storage.removeMany(keysToRemove)
    }

    return keysToRemove.length
  } catch {
    return 0
  }
}

export const createSidepanelChatHandoff = async (
  input: CreateSidepanelChatHandoffInput
): Promise<SidepanelChatHandoffPackage> => {
  await cleanupExpiredSidepanelChatHandoffs()
  const now = Date.now()
  const pkg = buildPackage(input, now)
  const key = storageKey(pkg.id)

  try {
    await storage.set(key, pkg)
  } catch (error) {
    await removeKeyQuietly(key)
    throw error
  }

  let saved: SidepanelChatHandoffPackage | null
  try {
    saved = await readRawPackage(pkg.id)
  } catch {
    await removeKeyQuietly(key)
    throw new Error("Sidepanel chat handoff could not be saved.")
  }

  if (!saved || saved.id !== pkg.id) {
    await removeKeyQuietly(key)
    throw new Error("Sidepanel chat handoff could not be saved.")
  }

  return saved
}

export const readSidepanelChatHandoff = async (
  id: string
): Promise<SidepanelChatHandoffPackage | null> => {
  const key = storageKey(id)

  try {
    const raw = await storage.get(key)
    if (raw == null) return null

    const pkg = parsePackage(raw, id)
    if (!pkg || pkg.consumedAt != null || isExpired(pkg)) {
      await removeKeyQuietly(key)
      return null
    }

    return pkg
  } catch {
    return null
  }
}

export const consumeSidepanelChatHandoff = async (
  id: string
): Promise<SidepanelChatHandoffPackage | null> => {
  const pkg = await readSidepanelChatHandoff(id)
  if (!pkg) return null

  await storage.remove(storageKey(id))
  return pkg
}

export const buildSidepanelChatHandoffRoute = (
  baseChatPath: string,
  handoffId: string
): string => {
  const [path, rawQuery = ""] = baseChatPath.split("?")
  const params = new URLSearchParams(rawQuery)
  params.set("handoff", handoffId)
  const query = params.toString()
  return query ? `${path}?${query}` : path
}

export const buildSidepanelHandoffMessageForModel = (
  visibleDraft: string,
  pageContext?: SidepanelChatHandoffPageContext
): string => {
  if (!pageContext) return visibleDraft
  const lines = [
    "Sidepanel page context:",
    pageContext.title ? `Title: ${pageContext.title}` : null,
    pageContext.url ? `URL: ${pageContext.url}` : null,
    ...pageContext.snippets.map((snippet, index) =>
      `Snippet ${index + 1}${snippet.label ? ` (${snippet.label})` : ""}: ${snippet.text}`
    ),
    "",
    "User draft:",
    visibleDraft
  ].filter(Boolean)
  return lines.join("\n")
}
