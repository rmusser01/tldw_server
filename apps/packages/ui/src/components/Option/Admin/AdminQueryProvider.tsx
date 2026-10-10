import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

/**
 * One QueryClient for the whole admin area (admin perf B-S4 / F11), mounted
 * by AdminRouteShell around every admin page. Reference data therefore
 * dedupes across surfaces: the overview's signal probes, the monitoring
 * dashboard and Server Admin share one fetch per stale window.
 *
 * - `staleTime: 30_000` — repeated mounts within the window are served from
 *   cache instead of re-probing every admin endpoint per visit.
 * - `refetchOnWindowFocus: false` — admin surfaces refresh on their own
 *   controls/pollers; focusing the window must not fan out refetches.
 * - `retry: false` — the swapped loaders were single-attempt with explicit
 *   guard/timeout error handling; retries would only delay the operator
 *   feedback those paths already render.
 */
const ADMIN_QUERY_STALE_TIME_MS = 30_000

export const createAdminQueryClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: ADMIN_QUERY_STALE_TIME_MS,
        refetchOnWindowFocus: false,
        retry: false
      }
    }
  })

let sharedAdminQueryClient: QueryClient | null = null

/**
 * The shared admin client. Exported for tests (and for the module-signal
 * probes, which run outside React but must read/write the same cache).
 */
export const getAdminQueryClient = (): QueryClient => {
  if (!sharedAdminQueryClient) {
    sharedAdminQueryClient = createAdminQueryClient()
  }
  return sharedAdminQueryClient
}

export const AdminQueryProvider: React.FC<{ children: React.ReactNode }> = ({
  children
}) => {
  const [client] = React.useState(getAdminQueryClient)
  return (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

export default AdminQueryProvider
