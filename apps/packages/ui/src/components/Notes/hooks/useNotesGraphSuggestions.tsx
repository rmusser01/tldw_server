import {
  type NotesGraphSuggestion,
  type NotesGraphSuggestionCapabilities,
  NotesGraphSuggestionClientError,
  type NotesGraphSuggestionRun,
  acceptNotesGraphSuggestion,
  cancelNotesGraphSuggestionRun,
  createNotesGraphOfflineError,
  createNotesGraphSuggestionCommand,
  createNotesGraphSuggestionRun,
  getNotesGraphSuggestionCapabilities,
  getNotesGraphSuggestionRun,
  isNotesGraphCapabilitiesChangedError,
  listNotesGraphSuggestionRuns,
  listNotesGraphSuggestions,
  rejectNotesGraphSuggestion,
  resetNotesGraphSuggestionRejections
} from "@/services/note-graph-suggestions"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import * as React from "react"

import { notesGraphWorkspaceQueryKey } from "./useNotesGraphWorkspace"

const ACTIVE_RUN_STATES = new Set<NotesGraphSuggestionRun["state"]>([
  "admitting",
  "queued",
  "running",
  "cancelling",
  "publishing"
])
const DEFAULT_POLL_INTERVAL_MS = 1500

export type ProvisionalNotesGraphOverlay = {
  edge: {
    id: string
    suggestionId: string
    source: string
    target: string
    type: "provisional_suggestion"
    directed: false
  }
  node: {
    id: string
    suggestionId: string
    type: "provisional_note"
    label: "Suggested note"
  } | null
}

export type UseNotesGraphSuggestionsOptions = {
  authorityScope: string | null
  enabled: boolean
  isOnline: boolean
  noteId: string | null
  datasetId?: string
  provider?: string
  model?: string
  loadedNodeIds: ReadonlySet<string>
  pollIntervalMs?: number
}

type Scoped<T> = {
  scopeIdentity: string
  data: T
}

const authority = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.length > 0 ? value : null

const scopedData = <T,>(
  value: Scoped<T> | undefined,
  scopeIdentity: string
): T | undefined =>
  value?.scopeIdentity === scopeIdentity ? value.data : undefined

const authoritativeNodeId = (
  noteId: string,
  loadedNodeIds: ReadonlySet<string>
): string | null => {
  if (loadedNodeIds.has(noteId)) return noteId
  const typedId = noteId.startsWith("note:") ? noteId : `note:${noteId}`
  return loadedNodeIds.has(typedId) ? typedId : null
}

const newestMatchingRun = (
  runs: NotesGraphSuggestionRun[],
  capability: NotesGraphSuggestionCapabilities | undefined
): NotesGraphSuggestionRun | null => {
  if (!capability) return null
  return (
    [...runs]
      .filter(
        (run) =>
          ACTIVE_RUN_STATES.has(run.state) &&
          run.provider === capability.provider &&
          run.model === capability.model
      )
      .sort((left, right) => {
        const time = Date.parse(right.created_at) - Date.parse(left.created_at)
        return time || right.id.localeCompare(left.id)
      })[0] ?? null
  )
}

const retryNetworkOnce = (failureCount: number, error: unknown): boolean => {
  const status = (error as { status?: unknown } | null)?.status
  return (
    failureCount < 1 &&
    !isNotesGraphCapabilitiesChangedError(error) &&
    typeof status === "number" &&
    status === 0
  )
}

export function useNotesGraphSuggestions(
  options: UseNotesGraphSuggestionsOptions
) {
  const queryClient = useQueryClient()
  const authorityScope = authority(options.authorityScope)
  const noteId = options.noteId ?? ""
  const scopeIdentity = JSON.stringify([
    authorityScope,
    noteId,
    options.datasetId ?? null,
    options.provider ?? null,
    options.model ?? null
  ])
  const baseKey = React.useMemo(
    () =>
      [
        "notes-graph-suggestions",
        authorityScope,
        noteId,
        options.datasetId ?? null,
        options.provider ?? null,
        options.model ?? null
      ] as const,
    [authorityScope, noteId, options.datasetId, options.model, options.provider]
  )
  const enabled = Boolean(
    authorityScope && options.enabled && options.isOnline && noteId
  )

  const capabilitiesKey = React.useMemo(
    () => [...baseKey, "capabilities"] as const,
    [baseKey]
  )
  const capabilitiesQueryRaw = useQuery({
    queryKey: capabilitiesKey,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: async () => ({
      scopeIdentity,
      data: await getNotesGraphSuggestionCapabilities({
        noteId,
        datasetId: options.datasetId,
        provider: options.provider,
        model: options.model
      })
    })
  })
  const capabilities = scopedData(capabilitiesQueryRaw.data, scopeIdentity)

  const runsKey = React.useMemo(
    () => [...baseKey, "runs", "active"] as const,
    [baseKey]
  )
  const runsQueryRaw = useQuery({
    queryKey: runsKey,
    enabled: enabled && Boolean(capabilities),
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: async () => ({
      scopeIdentity,
      data: await listNotesGraphSuggestionRuns({
        noteId,
        datasetId: options.datasetId,
        states: Array.from(ACTIVE_RUN_STATES),
        limit: 100
      })
    })
  })
  const runsPage = scopedData(runsQueryRaw.data, scopeIdentity)

  const adoptedRun = React.useMemo(
    () => newestMatchingRun(runsPage?.items ?? [], capabilities),
    [capabilities, runsPage?.items]
  )
  const seedRun = adoptedRun
  const runKey = React.useMemo(
    () => [...baseKey, "run", seedRun?.id ?? null] as const,
    [baseKey, seedRun?.id]
  )
  const runQueryRaw = useQuery({
    queryKey: runKey,
    enabled:
      enabled &&
      Boolean(seedRun?.id) &&
      Boolean(seedRun?.state && ACTIVE_RUN_STATES.has(seedRun.state)),
    retry: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    queryFn: async () => ({
      scopeIdentity,
      data: await getNotesGraphSuggestionRun({
        noteId,
        runId: seedRun?.id ?? "",
        datasetId: options.datasetId
      })
    }),
    refetchInterval: (query) => {
      if (query.state.error) return false
      const current = scopedData(
        query.state.data as Scoped<NotesGraphSuggestionRun> | undefined,
        scopeIdentity
      )
      return ACTIVE_RUN_STATES.has(current?.state ?? seedRun?.state ?? "stale")
        ? Math.max(250, options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS)
        : false
    },
    refetchIntervalInBackground: false
  })
  const runDetail = scopedData(runQueryRaw.data, scopeIdentity)
  const activeRun = runDetail ?? seedRun

  const suggestionsKey = React.useMemo(
    () => [...baseKey, "items"] as const,
    [baseKey]
  )
  const suggestionsQueryRaw = useQuery({
    queryKey: suggestionsKey,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: async () => ({
      scopeIdentity,
      data: await listNotesGraphSuggestions({
        noteId,
        datasetId: options.datasetId,
        states: ["pending", "accepting"],
        limit: 100
      })
    })
  })
  const suggestionPage = scopedData(suggestionsQueryRaw.data, scopeIdentity)
  const suggestions = React.useMemo(
    () => suggestionPage?.items ?? [],
    [suggestionPage?.items]
  )
  const currentCommandAuthority = React.useRef({
    scopeIdentity,
    allowed: enabled,
    epoch: 0
  })
  if (
    currentCommandAuthority.current.scopeIdentity !== scopeIdentity ||
    currentCommandAuthority.current.allowed !== enabled
  ) {
    currentCommandAuthority.current = {
      scopeIdentity,
      allowed: enabled,
      epoch: currentCommandAuthority.current.epoch + 1
    }
  }
  const authorityEpoch = currentCommandAuthority.current.epoch
  const hasCommandAuthority = (
    commandScopeIdentity: string,
    commandAuthorityEpoch: number
  ): boolean =>
    currentCommandAuthority.current.allowed &&
    currentCommandAuthority.current.scopeIdentity === commandScopeIdentity &&
    currentCommandAuthority.current.epoch === commandAuthorityEpoch
  const reconciledTerminalRuns = React.useRef(new Set<string>())

  React.useEffect(() => {
    if (runDetail?.state !== "succeeded") return
    const reconciliationKey = `${scopeIdentity}:${runDetail.id}:${runDetail.revision}`
    if (reconciledTerminalRuns.current.has(reconciliationKey)) return
    reconciledTerminalRuns.current.add(reconciliationKey)
    void queryClient.invalidateQueries({
      queryKey: suggestionsKey,
      exact: true
    })
  }, [queryClient, runDetail, scopeIdentity, suggestionsKey])

  const generationMutation = useMutation({
    mutationKey: [...baseKey, "generate"],
    retry: retryNetworkOnce,
    mutationFn: async (variables: {
      command: ReturnType<typeof createNotesGraphSuggestionCommand>
      capability: NotesGraphSuggestionCapabilities
      scopeIdentity: string
      authorityEpoch: number
      capabilitiesKey: readonly unknown[]
      runsKey: readonly unknown[]
      baseKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      ) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      const capability = variables.capability
      if (!capability)
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      return await createNotesGraphSuggestionRun(
        variables.command,
        capability,
        {
          canRetry: () =>
            hasCommandAuthority(
              variables.scopeIdentity,
              variables.authorityEpoch
            ),
          onCapabilitiesChanged: (nextCapability) => {
            if (
              !hasCommandAuthority(
                variables.scopeIdentity,
                variables.authorityEpoch
              )
            )
              return
            queryClient.setQueryData(variables.capabilitiesKey, {
              scopeIdentity: variables.scopeIdentity,
              data: nextCapability
            } satisfies Scoped<NotesGraphSuggestionCapabilities>)
          }
        }
      )
    },
    onSuccess: (run, variables) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      )
        return
      queryClient.setQueryData<
        Scoped<Awaited<ReturnType<typeof listNotesGraphSuggestionRuns>>>
      >(variables.runsKey, (current) => {
        const page = scopedData(current, variables.scopeIdentity)
        return {
          scopeIdentity: variables.scopeIdentity,
          data: {
            items: [
              run,
              ...(page?.items ?? []).filter((item) => item.id !== run.id)
            ],
            next_cursor: page?.next_cursor ?? null
          }
        }
      })
      queryClient.setQueryData([...variables.baseKey, "run", run.id], {
        scopeIdentity: variables.scopeIdentity,
        data: run
      } satisfies Scoped<NotesGraphSuggestionRun>)
    }
  })

  const cancelMutation = useMutation({
    mutationKey: [...baseKey, "cancel"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      run: NotesGraphSuggestionRun
      idempotencyKey: string
      scopeIdentity: string
      authorityEpoch: number
      baseKey: readonly unknown[]
      runsKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      ) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return cancelNotesGraphSuggestionRun({
        noteId,
        datasetId: options.datasetId,
        runId: variables.run.id,
        expectedRevision: variables.run.revision,
        idempotencyKey: variables.idempotencyKey
      })
    },
    onSuccess: async (_result, variables) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      )
        return
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: [...variables.baseKey, "run"]
        }),
        queryClient.invalidateQueries({
          queryKey: variables.runsKey,
          exact: true
        })
      ])
    }
  })

  const removeSuggestion = React.useCallback(
    (
      suggestionId: string,
      commandScopeIdentity: string,
      commandSuggestionsKey: readonly unknown[]
    ) => {
      queryClient.setQueryData<
        Scoped<Awaited<ReturnType<typeof listNotesGraphSuggestions>>>
      >(commandSuggestionsKey, (current) => {
        const page = scopedData(current, commandScopeIdentity)
        if (!page) return current
        return {
          scopeIdentity: commandScopeIdentity,
          data: {
            ...page,
            items: page.items.filter((item) => item.id !== suggestionId)
          }
        }
      })
    },
    [queryClient]
  )

  const acceptMutation = useMutation({
    mutationKey: [...baseKey, "accept"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      item: NotesGraphSuggestion
      idempotencyKey: string
      scopeIdentity: string
      authorityEpoch: number
      suggestionsKey: readonly unknown[]
      authorityScope: string
    }) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      ) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return acceptNotesGraphSuggestion({
        noteId,
        datasetId: options.datasetId,
        suggestionId: variables.item.id,
        expectedRevision: variables.item.revision,
        expectedSourceFingerprint: variables.item.source_fingerprint,
        expectedTargetFingerprint: variables.item.target_fingerprint,
        idempotencyKey: variables.idempotencyKey
      })
    },
    onSuccess: async (_result, variables) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      )
        return
      removeSuggestion(
        variables.item.id,
        variables.scopeIdentity,
        variables.suggestionsKey
      )
      await queryClient.invalidateQueries({
        queryKey: [...notesGraphWorkspaceQueryKey, variables.authorityScope]
      })
    }
  })

  const rejectMutation = useMutation({
    mutationKey: [...baseKey, "reject"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      item: NotesGraphSuggestion
      idempotencyKey: string
      scopeIdentity: string
      authorityEpoch: number
      suggestionsKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      ) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return rejectNotesGraphSuggestion({
        noteId,
        datasetId: options.datasetId,
        suggestionId: variables.item.id,
        expectedRevision: variables.item.revision,
        expectedSourceFingerprint: variables.item.source_fingerprint,
        expectedTargetFingerprint: variables.item.target_fingerprint,
        idempotencyKey: variables.idempotencyKey
      })
    },
    onSuccess: (_result, variables) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      )
        return
      removeSuggestion(
        variables.item.id,
        variables.scopeIdentity,
        variables.suggestionsKey
      )
    }
  })

  const resetMutation = useMutation({
    mutationKey: [...baseKey, "reset-rejections"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      idempotencyKey: string
      page: NonNullable<typeof suggestionPage>
      scopeIdentity: string
      authorityEpoch: number
      suggestionsKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      ) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      const page = variables.page
      if (!page) throw new Error("Suggestion rejection state is unavailable")
      return resetNotesGraphSuggestionRejections({
        noteId,
        datasetId: options.datasetId,
        expectedRejectionRevision: page.rejection_set_revision,
        sourceFingerprint: page.current_source_fingerprint,
        idempotencyKey: variables.idempotencyKey
      })
    },
    onSuccess: async (_result, variables) => {
      if (
        !hasCommandAuthority(variables.scopeIdentity, variables.authorityEpoch)
      )
        return
      await queryClient.invalidateQueries({
        queryKey: variables.suggestionsKey,
        exact: true
      })
    }
  })

  const idempotencyKey = React.useCallback(
    () =>
      createNotesGraphSuggestionCommand({
        noteId,
        datasetId: options.datasetId
      }).idempotencyKey,
    [noteId, options.datasetId]
  )

  const requireOnline = React.useCallback(() => {
    if (!options.isOnline) throw createNotesGraphOfflineError()
  }, [options.isOnline])

  const requireAuthority = React.useCallback(() => {
    if (!authorityScope || !options.enabled || !noteId) {
      throw new NotesGraphSuggestionClientError(
        422,
        "notes_graph_invalid_request"
      )
    }
  }, [authorityScope, noteId, options.enabled])

  const provisionalBySuggestionId = React.useMemo(() => {
    const overlays: Record<string, ProvisionalNotesGraphOverlay> = {}
    suggestions.forEach((item) => {
      if (item.kind !== "related_note" || !item.target_note_id) return
      const source =
        authoritativeNodeId(item.source_note_id, options.loadedNodeIds) ??
        item.source_note_id
      const target = authoritativeNodeId(
        item.target_note_id,
        options.loadedNodeIds
      )
      const ephemeralTarget = `suggestion-node:${item.id}`
      overlays[item.id] = {
        edge: {
          id: `suggestion-edge:${item.id}`,
          suggestionId: item.id,
          source,
          target: target ?? ephemeralTarget,
          type: "provisional_suggestion",
          directed: false
        },
        node: target
          ? null
          : {
              id: ephemeralTarget,
              suggestionId: item.id,
              type: "provisional_note",
              label: "Suggested note"
            }
      }
    })
    return overlays
  }, [options.loadedNodeIds, suggestions])

  const capabilitiesQuery = {
    ...capabilitiesQueryRaw,
    data: capabilities
  }
  const runsQuery = { ...runsQueryRaw, data: runsPage }
  const runQuery = { ...runQueryRaw, data: runDetail }
  const suggestionsQuery = { ...suggestionsQueryRaw, data: suggestionPage }

  return {
    capabilities: capabilities ?? null,
    capabilitiesQuery,
    runsQuery,
    runQuery,
    suggestionsQuery,
    activeRun: activeRun ?? null,
    suggestions,
    provisionalBySuggestionId,
    isOffline: !options.isOnline,
    generate: async () => {
      requireAuthority()
      requireOnline()
      const command = createNotesGraphSuggestionCommand({
        noteId,
        datasetId: options.datasetId,
        provider: options.provider,
        model: options.model
      })
      if (!capabilities) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return await generationMutation.mutateAsync({
        command,
        capability: capabilities,
        scopeIdentity,
        authorityEpoch,
        capabilitiesKey,
        runsKey,
        baseKey
      })
    },
    cancel: async () => {
      requireAuthority()
      requireOnline()
      if (!activeRun) throw new Error("No active suggestion run")
      return await cancelMutation.mutateAsync({
        run: activeRun,
        idempotencyKey: idempotencyKey(),
        scopeIdentity,
        authorityEpoch,
        baseKey,
        runsKey
      })
    },
    accept: async (item: NotesGraphSuggestion) => {
      requireAuthority()
      requireOnline()
      if (!suggestions.some((current) => current.id === item.id)) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return await acceptMutation.mutateAsync({
        item,
        idempotencyKey: idempotencyKey(),
        scopeIdentity,
        authorityEpoch,
        suggestionsKey,
        authorityScope
      })
    },
    reject: async (item: NotesGraphSuggestion) => {
      requireAuthority()
      requireOnline()
      if (!suggestions.some((current) => current.id === item.id)) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return await rejectMutation.mutateAsync({
        item,
        idempotencyKey: idempotencyKey(),
        scopeIdentity,
        authorityEpoch,
        suggestionsKey
      })
    },
    resetRejections: async () => {
      requireAuthority()
      requireOnline()
      if (!suggestionPage) {
        throw new NotesGraphSuggestionClientError(
          422,
          "notes_graph_invalid_request"
        )
      }
      return await resetMutation.mutateAsync({
        idempotencyKey: idempotencyKey(),
        page: suggestionPage,
        scopeIdentity,
        authorityEpoch,
        suggestionsKey
      })
    },
    mutations: {
      generation: generationMutation,
      cancellation: cancelMutation,
      acceptance: acceptMutation,
      rejection: rejectMutation,
      reset: resetMutation
    }
  }
}
