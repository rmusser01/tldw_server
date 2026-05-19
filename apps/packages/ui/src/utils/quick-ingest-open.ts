export type QuickIngestPendingOpenMode = "normal" | "intro"

export type QuickIngestPlaylistSourceKind =
  | "youtube_playlist"
  | "youtube_watch_playlist"
  | "unknown"

export type QuickIngestOpenDetail =
  | {
      source: "manual"
      action?: "normal"
    }
  | {
      source: "extension_active_tab"
      url: string
      sourceKind?: QuickIngestPlaylistSourceKind
      action: "playlist_preflight"
    }
  | {
      source?: string
      action?: string
      url?: string
      [key: string]: unknown
    }

export type QuickIngestPendingOpenOptions = {
  autoProcessQueued?: boolean
  focusTrigger?: boolean
}

export type QuickIngestPendingOpenRequest = {
  mode: QuickIngestPendingOpenMode
  at: number
  detail?: QuickIngestOpenDetail
  options?: QuickIngestPendingOpenOptions
}

type QuickIngestWindow = Window & {
  __tldwPendingQuickIngestOpen?: QuickIngestPendingOpenRequest
}

const getQuickIngestWindow = (): QuickIngestWindow | null => {
  if (typeof window === "undefined") {
    return null
  }
  return window as QuickIngestWindow
}

const buildPendingOpenRequest = (
  mode: QuickIngestPendingOpenMode,
  detail?: QuickIngestOpenDetail,
  options?: QuickIngestPendingOpenOptions
): QuickIngestPendingOpenRequest => ({
  mode,
  at: Date.now(),
  detail,
  options,
})

const normalizeQuickIngestOpenDetail = (
  detail: unknown
): QuickIngestOpenDetail | undefined =>
  detail && typeof detail === "object"
    ? (detail as QuickIngestOpenDetail)
    : undefined

const dispatchQuickIngestOpenEvent = (
  mode: QuickIngestPendingOpenMode,
  detail?: QuickIngestOpenDetail
): void => {
  const scope = getQuickIngestWindow()
  if (!scope) return
  const eventName =
    mode === "intro"
      ? "tldw:open-quick-ingest-intro"
      : "tldw:open-quick-ingest"
  scope.dispatchEvent(new CustomEvent(eventName, { detail }))
}

export const rememberQuickIngestOpenRequest = (
  mode: QuickIngestPendingOpenMode,
  detail?: unknown,
  options?: QuickIngestPendingOpenOptions
): QuickIngestPendingOpenRequest | null => {
  const scope = getQuickIngestWindow()
  if (!scope) return null
  const request = buildPendingOpenRequest(
    mode,
    normalizeQuickIngestOpenDetail(detail),
    options
  )
  scope.__tldwPendingQuickIngestOpen = request
  return request
}

export const requestQuickIngestOpen = (
  detail?: unknown,
  options?: QuickIngestPendingOpenOptions
): QuickIngestPendingOpenRequest | null => {
  const normalizedDetail = normalizeQuickIngestOpenDetail(detail)
  const request = rememberQuickIngestOpenRequest("normal", normalizedDetail, options)
  dispatchQuickIngestOpenEvent("normal", normalizedDetail)
  return request
}

export const requestQuickIngestIntro = (
  detail?: unknown,
  options?: QuickIngestPendingOpenOptions
): QuickIngestPendingOpenRequest | null => {
  const normalizedDetail = normalizeQuickIngestOpenDetail(detail)
  const request = rememberQuickIngestOpenRequest("intro", normalizedDetail, options)
  dispatchQuickIngestOpenEvent("intro", normalizedDetail)
  return request
}

export const consumePendingQuickIngestOpen = (): QuickIngestPendingOpenRequest | null => {
  const scope = getQuickIngestWindow()
  const request = scope?.__tldwPendingQuickIngestOpen || null
  if (scope) {
    delete scope.__tldwPendingQuickIngestOpen
  }
  return request
}

const hostnameMatches = (hostname: string, allowedHost: string): boolean =>
  hostname === allowedHost || hostname.endsWith(`.${allowedHost}`)

export const getQuickIngestPlaylistSourceKind = (
  rawUrl: string
): QuickIngestPlaylistSourceKind | null => {
  try {
    const parsed = new URL(rawUrl.trim())
    const hostname = parsed.hostname.toLowerCase()
    if (!hostnameMatches(hostname, "youtube.com") && !hostnameMatches(hostname, "youtu.be")) {
      return null
    }
    const playlistId = parsed.searchParams.get("list")?.trim()
    if (!playlistId) return null
    const pathname = parsed.pathname.toLowerCase()
    if (pathname === "/playlist") return "youtube_playlist"
    if (pathname === "/watch" || hostnameMatches(hostname, "youtu.be")) {
      return "youtube_watch_playlist"
    }
    return "unknown"
  } catch {
    return null
  }
}

export const buildQuickIngestOpenDetailFromUrl = (
  rawUrl: string
): QuickIngestOpenDetail | null => {
  const url = rawUrl.trim()
  const sourceKind = getQuickIngestPlaylistSourceKind(url)
  if (!sourceKind) return null
  return {
    source: "extension_active_tab",
    url,
    sourceKind,
    action: "playlist_preflight",
  }
}

export const isQuickIngestPlaylistPreflightDetail = (
  detail: QuickIngestOpenDetail | null | undefined
): detail is Extract<
  QuickIngestOpenDetail,
  { source: "extension_active_tab"; action: "playlist_preflight" }
> =>
  Boolean(
    detail &&
      detail.source === "extension_active_tab" &&
      detail.action === "playlist_preflight" &&
      typeof detail.url === "string" &&
      detail.url.trim().length > 0
  )

export const createQuickIngestSessionSeedFromOpenDetail = (
  detail: QuickIngestOpenDetail | null | undefined
): { openDetail: QuickIngestOpenDetail } | null =>
  isQuickIngestPlaylistPreflightDetail(detail)
    ? { openDetail: detail }
    : null
