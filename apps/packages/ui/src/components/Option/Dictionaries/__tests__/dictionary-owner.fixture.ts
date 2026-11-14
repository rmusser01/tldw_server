import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { vi } from "vitest"

const requestScope: ServicePromptRequestScope = {
  config: {
    serverUrl: "https://dictionary-ui-fixture.test",
    authMode: "multi-user"
  },
  userId: "fixture"
}
export const dictionaryOwnerFixture = {
  ownerKey: "dictionary-ui-fixture",
  serverScopeKey: "dictionary-ui-fixture",
  requestScope,
  isCurrent: () => true,
  verificationState: "ready" as const,
  hasEverOwned: () => true,
  retryVerification: () => undefined
}
export const selectDictionaryChatFixture = vi.fn(() => true)
export const dictionarySnapshotFixture = async (
  _ids: unknown,
  options?: { signal?: AbortSignal }
) => ({
  scopeSignal: options?.signal ?? new AbortController().signal,
  release: () => undefined
})
const queryData = new Map<string, unknown>()

// Match React Query structural sharing in presentation-only catalogue doubles.
export const scopedDictionaryQueryFixture = (
  options: { queryKey?: readonly unknown[] },
  result: Record<string, unknown>
) => {
  if (!Array.isArray(result.data)) return result
  const key = options.queryKey
  if (
    !key ||
    !["tldw:listDictionaries", "tldw:listChatsForDictionaryAssign"].includes(
      String(key[0])
    )
  )
    return result
  const cacheKey = JSON.stringify(key)
  const previous = queryData.get(cacheKey) as
    | { dictionaries?: unknown; chats?: unknown }
    | undefined
  const rows = previous?.dictionaries ?? previous?.chats
  const data =
    JSON.stringify(rows) === JSON.stringify(result.data)
      ? previous
      : key[0] === "tldw:listDictionaries"
        ? {
            ownerIsCurrent: dictionaryOwnerFixture.isCurrent,
            dictionaries: result.data
          }
        : { modalKey: key[3], chats: result.data }
  queryData.set(cacheKey, data)
  return { ...result, fetchStatus: "idle", data }
}

export const scopedDictionaryQueryClientFixture = (
  client: Record<string, unknown>
) => ({
  ...client,
  cancelQueries: async () => undefined,
  getQueryState: (key: readonly unknown[]) => ({
    status: "success",
    fetchStatus: "idle",
    data: queryData.get(JSON.stringify(key))
  })
})
