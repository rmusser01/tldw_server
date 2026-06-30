import React from "react"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import type {
  WorkspaceApiResponse,
  WorkspaceContextResponse,
  WorkspaceListApiResponse
} from "@/services/tldw/domains/workspace-api"
import type {
  ActiveWorkspaceContextContract,
  WorkspaceMembershipLabel,
  WorkspaceSummaryContract
} from "./contracts"
import {
  createWorkspaceMembershipLookup,
  normalizeActiveWorkspaceContext,
  normalizeWorkspaceSummary
} from "./normalizers"

export interface WorkspaceContextClient {
  getWorkspaceContext(workspaceId: string): Promise<WorkspaceContextResponse>
}

export interface WorkspaceMembershipClient {
  listWorkspaces(): Promise<WorkspaceListApiResponse>
}

export interface UseActiveWorkspaceContextOptions {
  workspaceId?: string | null
  client?: WorkspaceContextClient
  refreshKey?: unknown
}

export interface UseActiveWorkspaceContextResult {
  context: ActiveWorkspaceContextContract
  loading: boolean
  error: Error | null
  refresh: () => Promise<void>
}

export interface UseWorkspaceMembershipLookupOptions {
  client?: WorkspaceMembershipClient
}

export interface UseWorkspaceMembershipLookupResult {
  workspaces: WorkspaceSummaryContract[]
  loading: boolean
  error: Error | null
  refresh: () => Promise<void>
  resolveMembership: (workspaceId?: string | null) => WorkspaceMembershipLabel
}

const normalizeWorkspaceId = (workspaceId?: string | null): string | null => {
  if (typeof workspaceId !== "string") return null
  const trimmed = workspaceId.trim()
  return trimmed || null
}

const toError = (error: unknown): Error =>
  error instanceof Error ? error : new Error(String(error))

export const useActiveWorkspaceContext = ({
  workspaceId,
  client,
  refreshKey
}: UseActiveWorkspaceContextOptions): UseActiveWorkspaceContextResult => {
  const defaultClient = useTldwApiClient() as WorkspaceContextClient
  const resolvedClient = client ?? defaultClient
  const clientRef = React.useRef(resolvedClient)
  const mountedRef = React.useRef(false)
  const requestIdRef = React.useRef(0)
  const normalizedWorkspaceId = normalizeWorkspaceId(workspaceId)

  clientRef.current = resolvedClient

  const [context, setContext] = React.useState<ActiveWorkspaceContextContract>(
    () => normalizeActiveWorkspaceContext(null)
  )
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<Error | null>(null)

  const loadContext = React.useCallback(async () => {
    const currentWorkspaceId = normalizedWorkspaceId
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId
    const isCurrentRequest = () =>
      mountedRef.current && requestIdRef.current === requestId

    if (!currentWorkspaceId) {
      if (isCurrentRequest()) {
        setLoading(false)
        setError(null)
        setContext(normalizeActiveWorkspaceContext(null))
      }
      return
    }

    if (isCurrentRequest()) {
      setLoading(true)
      setError(null)
    }
    try {
      const response = await clientRef.current.getWorkspaceContext(currentWorkspaceId)
      if (isCurrentRequest()) {
        setContext(normalizeActiveWorkspaceContext(response))
      }
    } catch (err) {
      if (isCurrentRequest()) {
        const normalizedError = toError(err)
        setError(normalizedError)
        setContext(
          normalizeActiveWorkspaceContext(null, {
            state: "error",
            reasonCode: "workspace_context_error"
          })
        )
      }
    } finally {
      if (isCurrentRequest()) {
        setLoading(false)
      }
    }
  }, [normalizedWorkspaceId])

  React.useEffect(() => {
    mountedRef.current = true
    void loadContext()

    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
    }
  }, [loadContext, refreshKey])

  return {
    context,
    loading,
    error,
    refresh: loadContext
  }
}

export const useWorkspaceMembershipLookup = ({
  client
}: UseWorkspaceMembershipLookupOptions = {}): UseWorkspaceMembershipLookupResult => {
  const defaultClient = useTldwApiClient() as WorkspaceMembershipClient
  const resolvedClient = client ?? defaultClient
  const clientRef = React.useRef(resolvedClient)
  const mountedRef = React.useRef(false)
  const requestIdRef = React.useRef(0)

  clientRef.current = resolvedClient

  const [workspaces, setWorkspaces] = React.useState<WorkspaceApiResponse[]>([])
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<Error | null>(null)

  const loadWorkspaces = React.useCallback(async () => {
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId
    const isCurrentRequest = () =>
      mountedRef.current && requestIdRef.current === requestId

    if (isCurrentRequest()) {
      setLoading(true)
      setError(null)
    }

    try {
      const response = await clientRef.current.listWorkspaces()
      if (isCurrentRequest()) {
        setWorkspaces(response.items)
      }
    } catch (err) {
      if (isCurrentRequest()) {
        setError(toError(err))
      }
    } finally {
      if (isCurrentRequest()) {
        setLoading(false)
      }
    }
  }, [])

  React.useEffect(() => {
    mountedRef.current = true
    void loadWorkspaces()

    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
    }
  }, [loadWorkspaces])

  const resolveMembership = React.useMemo(
    () => createWorkspaceMembershipLookup(workspaces),
    [workspaces]
  )
  const workspaceSummaries = React.useMemo(
    () => workspaces.map(normalizeWorkspaceSummary),
    [workspaces]
  )

  return {
    workspaces: workspaceSummaries,
    loading,
    error,
    refresh: loadWorkspaces,
    resolveMembership
  }
}
