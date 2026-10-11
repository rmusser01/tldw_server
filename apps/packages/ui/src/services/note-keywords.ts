import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "@/services/resource-client"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { resolveServicePromptScope } from "@/services/service-prompts"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import {
  createServicePromptScopeChangedError,
  servicePromptPrincipalMatches,
  servicePromptSingleUserApiKeyScopeMatches,
  servicePromptTargetsMatch
} from "@/services/tldw/service-prompt-scope-error"

export type NoteKeywordStat = {
  keyword: string
  noteCount: number
}

type KeywordRequest = (path: string) => Promise<unknown>
const keywordCredentialFields = ["accessToken", "refreshToken", "apiKey", "apiBearer"] as const

const withKeywordRead = async <T>(read: (request: KeywordRequest) => Promise<T>): Promise<T> => {
  const controller = new AbortController()
  let capturedConfig: Readonly<TldwConfig> | null = null
  const stopWatching = watchChatAccountChanges((invalidated, currentConfig) => {
    if (invalidated || (currentConfig !== undefined && (!currentConfig || !capturedConfig ||
      keywordCredentialFields.some(key => currentConfig[key] !== capturedConfig?.[key])))) {
      controller.abort()
    }
  })
  // Config events omit credentials; a later reread cannot detect same-turn ABA.
  const configUpdated = () => controller.abort()
  if (typeof window !== "undefined") window.addEventListener("tldw:config-updated", configUpdated)
  const assertActive = () => {
    if (controller.signal.aborted) throw createServicePromptScopeChangedError()
  }
  try {
    const scope = await resolveServicePromptScope({ signal: controller.signal })
    assertActive()
    const config = await tldwClient.ensureConfigForRequest(true)
    assertActive()
    if (!config || !servicePromptTargetsMatch(config, scope.config)) {
      throw createServicePromptScopeChangedError()
    }
    const configSnapshot = Object.freeze({ ...config })
    capturedConfig = configSnapshot
    const assertCurrent = async () => {
      assertActive()
      const current = await tldwClient.ensureConfigForRequest(true)
      assertActive()
      if (!current || !servicePromptTargetsMatch(current, scope.config) ||
        !servicePromptSingleUserApiKeyScopeMatches(current, scope.config.expectedSingleUserApiKeyScope) ||
        (current.authSource !== "cookie-session" && !servicePromptPrincipalMatches(current, scope.userId)) ||
        keywordCredentialFields.some(key =>
          current[key] !== configSnapshot[key]
        )) throw createServicePromptScopeChangedError()
    }
    const { headers } = requestScopeFields({ config: scope.config, userId: scope.userId })
    const result = await read(async path => {
      await assertCurrent()
      assertActive()
      // These paths use the captured-config transport, not the Service Prompt allowlist.
      const response = await bgRequest({
        path: path as any, method: "GET", configSnapshot, headers, abortSignal: controller.signal
      })
      await assertCurrent()
      assertActive()
      return response
    })
    await assertCurrent()
    assertActive()
    return result
  } finally {
    stopWatching()
    if (typeof window !== "undefined") window.removeEventListener("tldw:config-updated", configUpdated)
  }
}

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

export const getNoteKeywords = async (limit = 200): Promise<string[]> =>
  withKeywordRead(async request => {
    const abs = await request(`/api/v1/notes/keywords/${buildQuery({ limit })}`)
    const arr = Array.isArray(abs)
      ? abs
          .map((item: any) => normalizeNoteKeyword(item))
          .filter(Boolean) as string[]
      : []
    return dedupeKeywords(arr)
  })

export const getAllNoteKeywords = async (pageSize = 1000): Promise<string[]> =>
  withKeywordRead(async request => {
    const stats = await readAllNoteKeywordStats(request, pageSize)
    return dedupeKeywords(stats.map((entry) => entry.keyword))
  })

const readAllNoteKeywordStats = async (request: KeywordRequest, pageSize: number): Promise<NoteKeywordStat[]> => {
  const out: NoteKeywordStat[] = []
  let offset = 0
  const maxPages = 100

  for (let page = 0; page < maxPages; page += 1) {
    const abs = await request(`/api/v1/notes/keywords/${buildQuery({ limit: pageSize, offset, include_note_counts: true })}`)
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

  return dedupeKeywordStats(out)
}

export const getAllNoteKeywordStats = async (pageSize = 1000): Promise<NoteKeywordStat[]> =>
  withKeywordRead(request => readAllNoteKeywordStats(request, pageSize))

export const searchNoteKeywords = async (
  query: string,
  limit = 10
): Promise<string[]> => {
  const q = String(query || "").trim()
  if (!q) return []
  return withKeywordRead(async request => {
    const abs = await request(`/api/v1/notes/keywords/search/${buildQuery({ query: q, limit })}`)
    const arr = Array.isArray(abs)
      ? abs
          .map((item: any) => normalizeNoteKeyword(item))
          .filter(Boolean) as string[]
      : []
    return dedupeKeywords(arr)
  })
}
