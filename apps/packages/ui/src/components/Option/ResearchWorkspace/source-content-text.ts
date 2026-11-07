import { tldwClient, type ScopedRequestOptions } from "@/services/tldw/TldwApiClient"
import { useWorkspaceStore } from "@/store/workspace"

/**
 * WP-45 — source-text reuse across ResearchWorkspace consumers.
 *
 * ChatPane ("ask with sources") and StudioPane artifact generation both need
 * the full text of already-ingested workspace sources. Both now go through
 * this resolver, which reuses the in-memory workspace-store cache keyed by
 * (mediaId, version) and only fetches misses. The cache is intentionally NOT
 * part of the persisted store shape (partialize drops it): reuse is
 * in-memory only and never adds persistence coupling.
 *
 * Eventual server-side replacement: batch `POST /media/details/batch {ids}`
 * (tracked as a note on TASK-15186).
 */

/**
 * Cache cap used by the eviction logic in the workspace store; re-exported
 * here so resolver-side consumers share the single definition (no mirrored
 * value that can drift).
 */
export { SOURCE_CONTENT_CACHE_MAX_ENTRIES } from "@/store/workspace-slices/sources-slice"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object"

const firstNonEmpty = (...candidates: unknown[]): string => {
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim()
    }
  }
  return ""
}

/**
 * Extracts the primary text payload from a media-details response. Union of
 * the shapes ChatPane and StudioPane previously parsed independently.
 */
export const extractSourceContentText = (detail: unknown): string => {
  if (!isRecord(detail)) return ""
  if (typeof detail.content === "string") return detail.content.trim()
  const content = isRecord(detail.content) ? detail.content : null
  const processing = isRecord(detail.processing) ? detail.processing : null

  return firstNonEmpty(
    content?.text,
    content?.content,
    content?.full_text,
    content?.transcript,
    content?.raw_text,
    processing?.analysis,
    processing?.transcript,
    processing?.text
  )
}

const extractSourceTitle = (detail: unknown): string | null => {
  if (!isRecord(detail)) return null
  const source = isRecord(detail.source) ? detail.source : null
  return firstNonEmpty(source?.title, detail.title) || null
}

export type SourceContentResolution = {
  text: string
  title: string | null
  fromCache: boolean
}

/**
 * The cache is a strictly optional capability of the workspace store: the
 * production store always exposes it (typed on SourcesActions), but partial
 * store mocks in component harnesses may not. A store without the cache (or
 * without `getState`) degrades to a plain network fetch — the reuse path must
 * never become a new failure mode for content resolution.
 */
type SourceContentCacheStore = {
  getState?: () => {
    getSourceContentText?: (
      mediaId: number,
      version?: number | null
    ) => string | null
    setSourceContentText?: (
      mediaId: number,
      version: number | null,
      text: string,
      title?: string | null
    ) => void
    sourceContentCache?: Map<number, { title?: string | null }>
  }
}

const storeWithCache = () => useWorkspaceStore as unknown as SourceContentCacheStore

const readCachedSourceContent = (
  mediaId: number
): SourceContentResolution | null => {
  const state = storeWithCache().getState?.()
  const cached = state?.getSourceContentText?.(mediaId, null)
  if (typeof cached !== "string" || cached.length === 0) return null
  return {
    text: cached,
    title: state?.sourceContentCache?.get(mediaId)?.title ?? null,
    fromCache: true
  }
}

/**
 * Resolves the full source text for a media item, reusing the workspace
 * store's in-memory cache for repeated asks of the same (mediaId, version).
 * `version` is the content identity the caller holds (null when unpinned);
 * web-capture sources must NOT use this path — their freshness protocol
 * (version uuid + content digest) is stricter and stays on its own route.
 */
export const resolveSourceContentText = async (
  mediaId: number,
  options?: ScopedRequestOptions
): Promise<SourceContentResolution> => {
  const cached = readCachedSourceContent(mediaId)
  if (cached !== null) {
    return cached
  }

  const detail = await tldwClient.getMediaDetails(mediaId, {
    include_content: true,
    include_versions: false,
    include_version_content: false,
    signal: options?.signal,
    requestScope: options?.requestScope
  })

  const text = extractSourceContentText(detail)
  if (text) {
    storeWithCache()
      .getState?.()
      ?.setSourceContentText?.(mediaId, null, text, extractSourceTitle(detail))
  }

  return { text, title: extractSourceTitle(detail), fromCache: false }
}
