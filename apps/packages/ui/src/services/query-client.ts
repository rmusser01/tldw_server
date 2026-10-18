import { QueryClient } from "@tanstack/react-query"

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
