import {
  type NotesSemanticCapabilities,
  NotesSemanticClientError,
  type NotesSemanticIndexStatus,
  type NotesSemanticMutation,
  type NotesSemanticRun,
  cancelNotesSemanticRun,
  createNotesSemanticCommand,
  createNotesSemanticOfflineError,
  createNotesSemanticRun,
  deleteNotesSemanticIndex,
  enableNotesSemanticIndex,
  getNotesSemanticCapabilities,
  getNotesSemanticRun,
  getNotesSemanticStatus
} from "@/services/note-semantic-index"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import * as React from "react"

const ACTIVE_RUN_STATUSES = new Set<NotesSemanticRun["status"]>([
  "queued",
  "processing"
])
const DEFAULT_POLL_INTERVAL_MS = 1500

export const notesSemanticIndexQueryKey = ["notes-semantic-index"] as const
const notesGraphWorkspaceQueryKey = ["notes-graph-workspace"] as const

export type UseNotesSemanticIndexOptions = {
  authorityScope: string | null
  enabled: boolean
  isOnline: boolean
  datasetId?: string
  pollIntervalMs?: number
}

const authority = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.length > 0 ? value : null

const retryNetworkOnce = (failureCount: number, error: unknown): boolean =>
  failureCount < 1 && (error as { status?: unknown } | null)?.status === 0

export function useNotesSemanticIndex(options: UseNotesSemanticIndexOptions) {
  const queryClient = useQueryClient()
  const authorityScope = authority(options.authorityScope)
  const scopeKey = React.useMemo(
    () =>
      [
        ...notesSemanticIndexQueryKey,
        authorityScope,
        options.datasetId ?? null
      ] as const,
    [authorityScope, options.datasetId]
  )
  const capabilitiesKey = React.useMemo(
    () => [...scopeKey, "capabilities"] as const,
    [scopeKey]
  )
  const statusKey = React.useMemo(
    () => [...scopeKey, "status"] as const,
    [scopeKey]
  )
  const canRead = Boolean(authorityScope && options.enabled && options.isOnline)
  const scopeIdentity = JSON.stringify(scopeKey)
  const currentScope = React.useRef(scopeIdentity)
  currentScope.current = scopeIdentity

  const capabilitiesQuery = useQuery({
    queryKey: capabilitiesKey,
    enabled: canRead,
    retry: retryNetworkOnce,
    refetchOnWindowFocus: false,
    queryFn: () =>
      getNotesSemanticCapabilities({ datasetId: options.datasetId })
  })
  const statusQuery = useQuery({
    queryKey: statusKey,
    enabled: canRead,
    retry: retryNetworkOnce,
    refetchOnWindowFocus: false,
    queryFn: () => getNotesSemanticStatus({ datasetId: options.datasetId })
  })

  const [runState, setRunState] = React.useState<{
    scope: string
    trackedId: string | null
    resolvedId: string | null
    lastTerminal: NotesSemanticRun | null
  }>({
    scope: scopeIdentity,
    trackedId: null,
    resolvedId: null,
    lastTerminal: null
  })
  const effectiveRunState =
    runState.scope === scopeIdentity
      ? runState
      : {
          scope: scopeIdentity,
          trackedId: null,
          resolvedId: null,
          lastTerminal: null
        }
  React.useEffect(() => {
    setRunState((current) =>
      current.scope === scopeIdentity
        ? current
        : {
            scope: scopeIdentity,
            trackedId: null,
            resolvedId: null,
            lastTerminal: null
          }
    )
  }, [scopeIdentity])

  const statusRun = statusQuery.data?.active_run ?? null
  const runId =
    effectiveRunState.trackedId ??
    (statusRun &&
    ACTIVE_RUN_STATUSES.has(statusRun.status) &&
    statusRun.run_id !== effectiveRunState.resolvedId
      ? statusRun.run_id
      : null)
  const runKey = React.useMemo(
    () => [...scopeKey, "run", runId] as const,
    [runId, scopeKey]
  )
  const runQuery = useQuery({
    queryKey: runKey,
    enabled: Boolean(canRead && runId),
    retry: retryNetworkOnce,
    refetchOnWindowFocus: false,
    refetchInterval: (query) =>
      ACTIVE_RUN_STATUSES.has(
        (query.state.data as NotesSemanticRun | undefined)?.status ??
          "completed"
      )
        ? options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS
        : false,
    queryFn: () =>
      getNotesSemanticRun({
        datasetId: options.datasetId,
        runId: runId ?? ""
      })
  })

  const runDetail = runQuery.data ?? null
  const activeRun = runDetail
    ? ACTIVE_RUN_STATUSES.has(runDetail.status)
      ? runDetail
      : null
    : statusRun &&
        statusRun.run_id !== effectiveRunState.resolvedId &&
        ACTIVE_RUN_STATUSES.has(statusRun.status)
      ? statusRun
      : null
  const reconciledTerminal = React.useRef<string | null>(null)

  React.useEffect(() => {
    if (!runDetail || ACTIVE_RUN_STATUSES.has(runDetail.status)) return
    const identity = JSON.stringify([
      scopeIdentity,
      runDetail.run_id,
      runDetail.revision,
      runDetail.status
    ])
    setRunState((current) => ({
      scope: scopeIdentity,
      trackedId: null,
      resolvedId: runDetail.run_id,
      lastTerminal: runDetail
    }))
    if (reconciledTerminal.current === identity) return
    reconciledTerminal.current = identity
    void Promise.all([
      queryClient.invalidateQueries({ queryKey: statusKey, exact: true }),
      authorityScope
        ? queryClient.invalidateQueries({
            queryKey: [...notesGraphWorkspaceQueryKey, authorityScope]
          })
        : Promise.resolve()
    ])
  }, [authorityScope, queryClient, runDetail, scopeIdentity, statusKey])

  const requireCommand = () => {
    if (!authorityScope || !options.enabled) {
      throw new NotesSemanticClientError(422, "notes_semantic_invalid_request")
    }
    if (!options.isOnline) throw createNotesSemanticOfflineError()
    if (!capabilitiesQuery.data?.manage_authorized) {
      throw new NotesSemanticClientError(
        403,
        "notes_semantic_permission_denied"
      )
    }
    if (!statusQuery.data) {
      throw new NotesSemanticClientError(422, "notes_semantic_invalid_request")
    }
  }

  const revokeManageOnDenied = (error: unknown, commandScope: string) => {
    if (
      currentScope.current === commandScope &&
      error instanceof NotesSemanticClientError &&
      error.status === 403
    ) {
      queryClient.setQueryData<NotesSemanticCapabilities>(
        capabilitiesKey,
        (current) =>
          current ? { ...current, manage_authorized: false } : current
      )
    }
  }

  const trackRun = (run: NotesSemanticRun, commandScope: string) => {
    if (currentScope.current !== commandScope) return
    queryClient.setQueryData([...scopeKey, "run", run.run_id], run)
    setRunState({
      scope: scopeIdentity,
      trackedId: run.run_id,
      resolvedId: null,
      lastTerminal: null
    })
  }

  const applyMutation = (
    mutation: NotesSemanticMutation,
    commandScope: string
  ) => {
    if (currentScope.current !== commandScope) return
    queryClient.setQueryData<NotesSemanticIndexStatus>(
      statusKey,
      mutation.resource
    )
    trackRun(mutation.run, commandScope)
  }

  const enableMutation = useMutation({
    mutationKey: [...scopeKey, "enable"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      status: NotesSemanticIndexStatus
      capability: NotesSemanticCapabilities
      idempotencyKey: string
      scope: string
    }) =>
      enableNotesSemanticIndex({
        datasetId: options.datasetId,
        expectedRevision: variables.status.configuration_revision,
        capabilityRevision: variables.capability.capability_revision,
        idempotencyKey: variables.idempotencyKey
      }),
    onSuccess: (mutation, variables) =>
      applyMutation(mutation, variables.scope),
    onError: (error, variables) => revokeManageOnDenied(error, variables.scope)
  })

  const createRunMutation = useMutation({
    mutationKey: [...scopeKey, "create-run"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      mode: "rebuild" | "retry_failed"
      status: NotesSemanticIndexStatus
      idempotencyKey: string
      scope: string
    }) =>
      createNotesSemanticRun({
        datasetId: options.datasetId,
        mode: variables.mode,
        expectedRevision: variables.status.configuration_revision,
        idempotencyKey: variables.idempotencyKey
      }),
    onSuccess: (run, variables) => trackRun(run, variables.scope),
    onError: (error, variables) => revokeManageOnDenied(error, variables.scope)
  })

  const cancelMutation = useMutation({
    mutationKey: [...scopeKey, "cancel"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      run: NotesSemanticRun
      idempotencyKey: string
      scope: string
    }) =>
      cancelNotesSemanticRun({
        datasetId: options.datasetId,
        runId: variables.run.run_id,
        expectedRevision: variables.run.revision,
        idempotencyKey: variables.idempotencyKey
      }),
    onSuccess: (mutation, variables) =>
      applyMutation(mutation, variables.scope),
    onError: (error, variables) => revokeManageOnDenied(error, variables.scope)
  })

  const deleteMutation = useMutation({
    mutationKey: [...scopeKey, "delete"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      status: NotesSemanticIndexStatus
      idempotencyKey: string
      scope: string
    }) =>
      deleteNotesSemanticIndex({
        datasetId: options.datasetId,
        expectedRevision: variables.status.configuration_revision,
        idempotencyKey: variables.idempotencyKey
      }),
    onSuccess: async (mutation, variables) => {
      applyMutation(mutation, variables.scope)
      if (currentScope.current !== variables.scope || !authorityScope) return
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: statusKey, exact: true }),
        queryClient.invalidateQueries({
          queryKey: [...notesGraphWorkspaceQueryKey, authorityScope]
        })
      ])
    },
    onError: (error, variables) => revokeManageOnDenied(error, variables.scope)
  })

  const commandKey = () => createNotesSemanticCommand().idempotencyKey
  const currentCommandData = () => {
    requireCommand()
    return {
      status: statusQuery.data as NotesSemanticIndexStatus,
      capability: capabilitiesQuery.data as NotesSemanticCapabilities,
      scope: scopeIdentity
    }
  }
  const createRun = async (mode: "rebuild" | "retry_failed") => {
    const current = currentCommandData()
    return await createRunMutation.mutateAsync({
      mode,
      status: current.status,
      idempotencyKey: commandKey(),
      scope: current.scope
    })
  }

  return {
    capabilities: capabilitiesQuery.data ?? null,
    status: statusQuery.data ?? null,
    activeRun,
    lastTerminalRun: effectiveRunState.lastTerminal,
    isOffline: !options.isOnline,
    capabilitiesQuery,
    statusQuery,
    runQuery,
    mutations: {
      enable: enableMutation,
      rebuild: createRunMutation,
      retry: createRunMutation,
      cancel: cancelMutation,
      deleteIndex: deleteMutation
    },
    enable: async () => {
      const current = currentCommandData()
      return await enableMutation.mutateAsync({
        status: current.status,
        capability: current.capability,
        idempotencyKey: commandKey(),
        scope: current.scope
      })
    },
    rebuild: () => createRun("rebuild"),
    retryFailed: () => createRun("retry_failed"),
    cancel: async () => {
      requireCommand()
      if (!activeRun) {
        throw new NotesSemanticClientError(
          422,
          "notes_semantic_invalid_request"
        )
      }
      return await cancelMutation.mutateAsync({
        run: activeRun,
        idempotencyKey: commandKey(),
        scope: scopeIdentity
      })
    },
    deleteIndex: async () => {
      const current = currentCommandData()
      return await deleteMutation.mutateAsync({
        status: current.status,
        idempotencyKey: commandKey(),
        scope: current.scope
      })
    }
  }
}

export type NotesSemanticIndexController = ReturnType<
  typeof useNotesSemanticIndex
>
