import { notifyManager, QueryClient } from "@tanstack/react-query"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

const RATE_LIMIT_PATTERN = /(429|rate limit|too many requests)/i
const MAX_QUERY_RETRIES = 1

const isRateLimitError = (error: unknown): boolean => {
  if (!error) return false
  const msg =
    error instanceof Error ? error.message : String(error || "")
  return RATE_LIMIT_PATTERN.test(msg)
}

const buildQueryClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
        retry: (failureCount, error) => {
          if (isRateLimitError(error)) return false
          return failureCount < MAX_QUERY_RETRIES
        }
      }
    }
  })

let singleton: QueryClient | null = null

const stopWatchingReadinessAccount = watchChatAccountChanges(invalidated => {
  if (!invalidated || !singleton) return
  const client = singleton
  const queries = client.getQueryCache().findAll({
    predicate: ({ queryKey }) => queryKey.length === 1 && (
      queryKey[0] === "playground:providersStatus" ||
      queryKey[0] === "playground:chatModels"
    )
  })
  // Reset now; start model reads after all account listeners finish.
  notifyManager.batch(() => { for (const query of queries) query.reset() })
  queueMicrotask(() => {
    void client.refetchQueries({ type: "active", predicate: query => queries.includes(query) })
  })
})
const queryClientHot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
queryClientHot?.dispose(stopWatchingReadinessAccount)

export const getQueryClient = (): QueryClient => {
  if (!singleton) {
    singleton = buildQueryClient()
  }
  return singleton
}

export const createQueryClient = (): QueryClient => buildQueryClient()

/** A held first page must retire before a write's finder refresh starts. */
export const invalidateServerChatHistory = async (
  queryClient: QueryClient,
  isCurrent: () => boolean = () => true
): Promise<void> => {
  if (!isCurrent()) return;
  const filters = { queryKey: ['serverChatHistory'] };
  await queryClient.cancelQueries(filters);
  if (isCurrent()) await queryClient.invalidateQueries(filters);
};
