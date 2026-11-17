import { useEffect, useState, useSyncExternalStore } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { fetchChatModels } from "@/services/tldw-server"
import {
  getChatModelsCacheGeneration,
  subscribeChatModelsInvalidation
} from "@/services/chat-models-invalidation"

export const EMPTY_CHAT_MODELS: Awaited<ReturnType<typeof fetchChatModels>> = []

/**
 * WP-49: revisits inside this window reuse the cached catalog instead of
 * re-fetching the full model catalog per mount. Explicit refresh affordances
 * (retry actions, "refresh models") and cache-generation invalidations still
 * force fresh reads regardless of this window.
 */
export const CHAT_MODEL_CATALOG_STALE_MS = 15 * 60_000

/** The query key of the chat model catalog for one cache generation. */
export const chatModelCatalogQueryKey = (generation: number) =>
  ["playground:chatModels", generation] as const

/** Share authoritative readiness and selection, retiring results on setup/account changes. */
export const useChatModelCatalog = (enabled = true) => {
  const generation = useSyncExternalStore(
    subscribeChatModelsInvalidation,
    getChatModelsCacheGeneration,
    getChatModelsCacheGeneration
  )
  const client = useQueryClient()
  const queryKey = chatModelCatalogQueryKey(generation)
  const previous = client.getQueryState(queryKey)
  const previousPending = previous?.fetchStatus === "fetching"
  const previousEmpty = previous?.status === "success" && Array.isArray(previous.data) && previous.data.length === 0
  const query = useQuery({
    queryKey,
    queryFn: () => fetchChatModels({ requireFresh: true }),
    enabled,
    retry: false,
    staleTime: CHAT_MODEL_CATALOG_STALE_MS,
    gcTime: CHAT_MODEL_CATALOG_STALE_MS
  })
  const [activation, setActivation] = useState({ enabled, generation,
    needsVerification: enabled && (previousPending || previousEmpty), waitForPrevious: previousPending })
  if (activation.generation !== generation) {
    setActivation({ enabled, generation, needsVerification: false, waitForPrevious: false })
  } else if (activation.enabled !== enabled) {
    setActivation({ enabled, generation, needsVerification: enabled, waitForPrevious: previousPending })
  }
  const { refetch } = query
  useEffect(() => {
    if (!activation.needsVerification) return
    let current = true
    void (async () => {
      if (activation.waitForPrevious) {
        await refetch({ cancelRefetch: false, throwOnError: false })
        if (!current) return
      }
      await refetch({ cancelRefetch: false, throwOnError: false })
      if (current) setActivation(value => value === activation ? { ...value, needsVerification: false } : value)
    })()
    return () => { current = false }
  }, [activation, generation, refetch])
  const verified = query.isSuccess && !query.isFetching && !activation.needsVerification
  return {
    ...query,
    isFetching: query.isFetching || activation.needsVerification,
    data: verified ? query.data : undefined,
    isSuccess: verified,
    catalogGeneration: generation
  }
}
