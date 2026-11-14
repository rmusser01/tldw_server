import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "@/services/resource-client"

const KEYWORDS_CACHE_TTL_MS = 5 * 60 * 1000

type KeywordCacheEntry = {
  data: string[]
  expiresAt: number
}

export type NoteKeywordStat = {
  keyword: string
  noteCount: number
}

type KeywordStatsCacheEntry = {
  data: NoteKeywordStat[]
  expiresAt: number
}

const listCache = new Map<number, KeywordCacheEntry>()
const listInFlight = new Map<number, Promise<string[]>>()
const allCache = new Map<number, KeywordCacheEntry>()
const allInFlight = new Map<number, Promise<string[]>>()
const allStatsCache = new Map<number, KeywordStatsCacheEntry>()
const allStatsInFlight = new Map<number, Promise<NoteKeywordStat[]>>()

export const normalizeNoteKeyword = (value: any): string | null => {
  const raw =
    value?.keyword ??
    value?.keyword_text ??
    value?.text ??
    value
  if (raw == null) return null
  const text = String(raw).trim()
  return text.length ? text : null
}

const dedupeKeywords = (items: string[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of items) {
    if (seen.has(item)) continue
    seen.add(item)
    out.push(item)
  }
  return out
}

const normalizeNoteCount = (value: any): number => {
  const parsed = Number(value?.note_count ?? value?.count ?? 0)
  if (!Number.isFinite(parsed) || parsed < 0) return 0
  return Math.floor(parsed)
}

const dedupeKeywordStats = (items: NoteKeywordStat[]): NoteKeywordStat[] => {
  const seen = new Map<string, NoteKeywordStat>()
  for (const item of items) {
    const keyword = String(item.keyword || "").trim()
    if (!keyword) continue
    const key = keyword.toLowerCase()
    const existing = seen.get(key)
    if (!existing) {
      seen.set(key, {
        keyword,
        noteCount: Math.max(0, item.noteCount)
      })
      continue
    }
    if (item.noteCount > existing.noteCount) {
      existing.noteCount = item.noteCount
    }
  }
  return Array.from(seen.values())
}

export const getNoteKeywords = async (limit = 200): Promise<string[]> => {
  const now = Date.now()
  const cached = listCache.get(limit)
  if (cached && cached.expiresAt > now) return cached.data

  const inFlight = listInFlight.get(limit)
  if (inFlight) return inFlight

  const request = (async () => {
    const abs = await bgRequest<any>({
      path: `/api/v1/notes/keywords/${buildQuery({ limit })}` as any,
      method: "GET" as any
    })
    const arr = Array.isArray(abs)
      ? abs
          .map((item: any) => normalizeNoteKeyword(item))
          .filter(Boolean) as string[]
      : []
    const deduped = dedupeKeywords(arr)
    listCache.set(limit, {
      data: deduped,
      expiresAt: Date.now() + KEYWORDS_CACHE_TTL_MS
    })
    return deduped
  })()

  listInFlight.set(limit, request)
  try {
    return await request
  } finally {
    listInFlight.delete(limit)
  }
}

export const getAllNoteKeywords = async (pageSize = 1000): Promise<string[]> => {
  const now = Date.now()
  const cached = allCache.get(pageSize)
  if (cached && cached.expiresAt > now) return cached.data

  const inFlight = allInFlight.get(pageSize)
  if (inFlight) return inFlight

  const request = (async () => {
    const stats = await getAllNoteKeywordStats(pageSize)
    const deduped = dedupeKeywords(stats.map((entry) => entry.keyword))
    allCache.set(pageSize, {
      data: deduped,
      expiresAt: Date.now() + KEYWORDS_CACHE_TTL_MS
    })
    return deduped
  })()

  allInFlight.set(pageSize, request)
  try {
    return await request
  } finally {
    allInFlight.delete(pageSize)
  }
}

export const getAllNoteKeywordStats = async (pageSize = 1000): Promise<NoteKeywordStat[]> => {
  const now = Date.now()
  const cached = allStatsCache.get(pageSize)
  if (cached && cached.expiresAt > now) return cached.data

  const inFlight = allStatsInFlight.get(pageSize)
  if (inFlight) return inFlight

  const request = (async () => {
    const out: NoteKeywordStat[] = []
    let offset = 0
    const maxPages = 100

    for (let page = 0; page < maxPages; page += 1) {
      const abs = await bgRequest<any>({
        path: `/api/v1/notes/keywords/${buildQuery({ limit: pageSize, offset, include_note_counts: true })}` as any,
        method: "GET" as any
      })
      const arr = Array.isArray(abs)
        ? abs
            .map((item: any) => {
              const keyword = normalizeNoteKeyword(item)
              if (!keyword) return null
              return {
                keyword,
                noteCount: normalizeNoteCount(item)
              } as NoteKeywordStat
            })
            .filter(Boolean) as NoteKeywordStat[]
        : []
      if (!arr.length) break
      out.push(...arr)
      if (arr.length < pageSize) break
      offset += pageSize
    }

    const deduped = dedupeKeywordStats(out)
    allStatsCache.set(pageSize, {
      data: deduped,
      expiresAt: Date.now() + KEYWORDS_CACHE_TTL_MS
    })
    return deduped
  })()

  allStatsInFlight.set(pageSize, request)
  try {
    return await request
  } finally {
    allStatsInFlight.delete(pageSize)
  }
}

export const searchNoteKeywords = async (
  query: string,
  limit = 10
): Promise<string[]> => {
  const q = String(query || "").trim()
  if (!q) return []
  const abs = await bgRequest<any>({
    path: `/api/v1/notes/keywords/search/${buildQuery({ query: q, limit })}` as any,
    method: "GET" as any
  })
  const arr = Array.isArray(abs)
    ? abs
        .map((item: any) => normalizeNoteKeyword(item))
        .filter(Boolean) as string[]
    : []
  return dedupeKeywords(arr)
}
