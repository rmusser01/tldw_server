import React from "react"
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query"
import { AdminQueryContext, createAdminQueryClient, type AdminQuerySession } from "../AdminQueryProvider"
import type { PrivateWorkspaceScope } from "@/hooks/usePrivateWorkspaceScope"

export const createAdminTestSession = (client = createAdminQueryClient(), authMode: "single-user" | "multi-user" = "multi-user"): AdminQuerySession => {
  const controller = new AbortController()
  const config = { serverUrl: "http://127.0.0.1:8000", authMode, authSource: "cookie-session" } as const
  const scope = {
    key: "verified-admin-fixture",
    config,
    requestScope: { config: { serverUrl: config.serverUrl }, userId: 1 },
    signal: controller.signal,
    isCurrent: () => !controller.signal.aborted
  } as PrivateWorkspaceScope
  return { client, scope, requestOptions: () => ({ configSnapshot: scope.config, abortSignal: scope.signal, headers: { "X-TLDW-Expected-User-ID": "1" } }) }
}

export const AdminTestQueryProvider: React.FC<{ client: QueryClient; children: React.ReactNode }> = ({ client, children }) => {
  const session = React.useMemo(() => createAdminTestSession(client), [client])
  return <AdminQueryContext.Provider value={session}><QueryClientProvider client={client}>{children}</QueryClientProvider></AdminQueryContext.Provider>
}
