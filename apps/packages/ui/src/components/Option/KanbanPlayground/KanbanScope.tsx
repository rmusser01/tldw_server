import React from "react"
import { useQueryClient } from "@tanstack/react-query"
import { createKanbanClient } from "@/services/kanban"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

type KanbanScope = {
  api: ReturnType<typeof createKanbanClient>
  key: (...parts: unknown[]) => readonly unknown[]
  isActive: () => boolean
}
const ScopeContext = React.createContext<KanbanScope | null>(null)

export const KanbanScopeProvider = ({
  config, children
}: { config: TldwConfig; children: React.ReactNode }) => {
  const queryClient = useQueryClient()
  const owner = deriveConnectionAuthorityId(config)
  const currentConfig = React.useRef(config)
  currentConfig.current = config
  const active = React.useRef(true)
  const controller = React.useRef(new AbortController())
  const isActive = React.useCallback(() => active.current, [])
  const [api] = React.useState(() => createKanbanClient(() => {
    if (!active.current) throw new DOMException("Kanban workspace retired", "AbortError")
    return { configSnapshot: { ...currentConfig.current }, abortSignal: controller.current.signal }
  }))
  const key = React.useCallback((...parts: unknown[]) => ["kanban", owner, ...parts], [owner])

  React.useEffect(() => {
    active.current = true
    if (controller.current.signal.aborted) controller.current = new AbortController()
    return () => {
      active.current = false
      controller.current.abort()
      queryClient.removeQueries({ queryKey: ["kanban", owner] })
    }
  }, [owner, queryClient])

  const value = React.useMemo(() => ({ api, key, isActive }), [api, key, isActive])
  return <ScopeContext.Provider value={value}>{children}</ScopeContext.Provider>
}

export const useKanbanScope = (): KanbanScope => {
  const scope = React.useContext(ScopeContext)
  if (!scope) throw new Error("Kanban components require an owned workspace")
  return scope
}
