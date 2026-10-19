import React from "react"
import { useQueryClient } from "@tanstack/react-query"
import { createKanbanClient } from "@/services/kanban"
import type { PrivateWorkspaceScope } from "@/hooks/usePrivateWorkspaceScope"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"

type KanbanScope = {
  api: ReturnType<typeof createKanbanClient>
  key: (...parts: unknown[]) => readonly unknown[]
  isActive: () => boolean
}
const ScopeContext = React.createContext<KanbanScope | null>(null)

export const KanbanScopeProvider = ({
  scope, children
}: { scope: PrivateWorkspaceScope; children: React.ReactNode }) => {
  const queryClient = useQueryClient()
  const owner = scope.key
  const currentConfig = React.useRef(scope.config)
  currentConfig.current = scope.config
  const active = React.useRef(true)
  const controller = React.useRef(new AbortController())
  const isActive = React.useCallback(() => active.current && scope.isCurrent(), [scope])
  const [api] = React.useState(() => createKanbanClient(() => {
    if (!active.current || !scope.isCurrent()) throw new DOMException("Kanban workspace retired", "AbortError")
    return { configSnapshot: { ...currentConfig.current }, abortSignal: controller.current.signal,
      headers: requestScopeFields(scope.requestScope).headers }
  }))
  const key = React.useCallback((...parts: unknown[]) => ["kanban", owner, ...parts], [owner])

  React.useLayoutEffect(() => {
    active.current = true
    if (controller.current.signal.aborted) controller.current = new AbortController()
    const retire = () => {
      active.current = false
      controller.current.abort()
      queryClient.removeQueries({ queryKey: ["kanban", owner] })
    }
    scope.signal.addEventListener("abort", retire, { once: true })
    if (scope.signal.aborted) retire()
    return () => { scope.signal.removeEventListener("abort", retire); retire() }
  }, [owner, queryClient, scope])

  const value = React.useMemo(() => ({ api, key, isActive }), [api, key, isActive])
  return <ScopeContext.Provider value={value}>{children}</ScopeContext.Provider>
}

export const useKanbanScope = (): KanbanScope => {
  const scope = React.useContext(ScopeContext)
  if (!scope) throw new Error("Kanban components require an owned workspace")
  return scope
}
