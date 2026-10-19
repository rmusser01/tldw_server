import React from "react"
import { usePrivateWorkspaceScope, type PrivateWorkspaceScope } from "@/hooks/usePrivateWorkspaceScope"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import type { AdminRequestOptions } from "@/services/tldw/domains/admin"
import { Button } from "@/components/Common/Button"
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

export type AdminQuerySession = {
  client: QueryClient
  scope: PrivateWorkspaceScope
  requestOptions: () => AdminRequestOptions
}

export const AdminQueryContext = React.createContext<AdminQuerySession | null>(null)

export const useAdminQuerySession = (): AdminQuerySession => {
  const session = React.useContext(AdminQueryContext)
  if (!session) throw new Error("Verified Admin workspace required")
  return session
}

const OwnedAdminQueries: React.FC<{ scope: PrivateWorkspaceScope; children: React.ReactNode }> = ({ scope, children }) => {
  const [client] = React.useState(createAdminQueryClient)
  const session = React.useMemo<AdminQuerySession>(() => ({
    client,
    scope,
    requestOptions: () => {
      if (!scope.isCurrent()) throw new Error("Admin workspace changed")
      return {
        configSnapshot: scope.config,
        abortSignal: scope.signal,
        headers: requestScopeFields(scope.requestScope).headers
      }
    }
  }), [client, scope])
  const [retired, setRetired] = React.useState(false)
  React.useLayoutEffect(() => {
    const retire = () => {
      void client.cancelQueries()
      client.clear()
      setRetired(true)
    }
    scope.signal.addEventListener("abort", retire, { once: true })
    if (scope.signal.aborted) retire()
    return () => {
      scope.signal.removeEventListener("abort", retire)
      void client.cancelQueries()
      client.clear()
    }
  }, [client, scope])
  if (retired || !scope.isCurrent()) return null
  return <AdminQueryContext.Provider value={session}>
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  </AdminQueryContext.Provider>
}

const VerifiedAdminQueryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { scope, retry } = usePrivateWorkspaceScope()
  if (!scope) return <div role="status">Resolving Admin workspace… <Button onClick={retry}>Retry connection</Button></div>
  return <OwnedAdminQueries key={scope.key} scope={scope}>{children}</OwnedAdminQueries>
}

export const AdminQueryProvider: React.FC<{ children: React.ReactNode; active?: boolean }> = ({ children, active = true }) =>
  active ? <VerifiedAdminQueryProvider>{children}</VerifiedAdminQueryProvider> : <>{children}</>

export default AdminQueryProvider
