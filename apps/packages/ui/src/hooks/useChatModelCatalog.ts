import { useSyncExternalStore } from "react"
import { useQuery } from "@tanstack/react-query"
import { fetchChatModels } from "@/services/tldw-server"
import {
  getChatModelsCacheGeneration,
  subscribeChatModelsInvalidation
} from "@/services/chat-models-invalidation"

export const EMPTY_CHAT_MODELS: Awaited<ReturnType<typeof fetchChatModels>> = []

/** Share authoritative readiness and selection, retiring results on setup/account changes. */
export const useChatModelCatalog = (enabled = true) => {
  const generation = useSyncExternalStore(
    subscribeChatModelsInvalidation,
    getChatModelsCacheGeneration,
    getChatModelsCacheGeneration
  )
  const query = useQuery({
    queryKey: ["playground:chatModels", generation],
    queryFn: () => fetchChatModels({ requireFresh: true }),
    enabled,
    retry: false,
    staleTime: 30_000
  })
  const verified = query.isSuccess && !query.isFetching
  return {
    ...query,
    data: verified ? query.data : undefined,
    isSuccess: verified,
    catalogGeneration: generation
  }
}
