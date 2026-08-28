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
const SUGGESTION_LIST_STATES = ["pending", "accepting"] as const
const SUGGESTION_LIST_LIMIT = 100

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

const authority = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.length > 0 ? value : null

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
  const commandAuthority = JSON.stringify([
    authorityScope,
    options.datasetId ?? null,
    noteId
  ])
  const suggestionScopeKey = React.useMemo(
    () =>
      [
        "notes-graph-suggestions",
        authorityScope,
        options.datasetId ?? null,
        noteId
      ] as const,
    [authorityScope, noteId, options.datasetId]
  )
  const providerKey = React.useMemo(
    () =>
      [
        ...suggestionScopeKey,
        "provider",
        options.provider ?? null,
        options.model ?? null
      ] as const,
    [options.model, options.provider, suggestionScopeKey]
  )
  const enabled = Boolean(
    authorityScope && options.enabled && options.isOnline && noteId
  )

  const capabilitiesKey = React.useMemo(
    () => [...providerKey, "capabilities"] as const,
    [providerKey]
  )
  const capabilitiesQueryRaw = useQuery({
    queryKey: capabilitiesKey,
    enabled,
    notifyOnChangeProps: "all",
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: () =>
      getNotesGraphSuggestionCapabilities({
        noteId,
        datasetId: options.datasetId,
        provider: options.provider,
        model: options.model
      })
  })
  const capabilities = capabilitiesQueryRaw.data

  const runsKey = React.useMemo(
    () => [...providerKey, "runs", "active"] as const,
    [providerKey]
  )
  const runsQueryRaw = useQuery({
    queryKey: runsKey,
    enabled: enabled && Boolean(capabilities),
    notifyOnChangeProps: "all",
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: () =>
      listNotesGraphSuggestionRuns({
        noteId,
        datasetId: options.datasetId,
        states: Array.from(ACTIVE_RUN_STATES),
        limit: 100
      })
  })
  const runsPage = runsQueryRaw.data

  const adoptedRun = React.useMemo(
    () => newestMatchingRun(runsPage?.items ?? [], capabilities),
    [capabilities, runsPage?.items]
  )
  const seedRun = adoptedRun
  const runKey = React.useMemo(
    () => [...providerKey, "run", seedRun?.id ?? null] as const,
    [providerKey, seedRun?.id]
  )
  const runQueryRaw = useQuery({
    queryKey: runKey,
    enabled:
      enabled &&
      Boolean(seedRun?.id) &&
      Boolean(seedRun?.state && ACTIVE_RUN_STATES.has(seedRun.state)),
    notifyOnChangeProps: "all",
    retry: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    queryFn: () =>
      getNotesGraphSuggestionRun({
        noteId,
        runId: seedRun?.id ?? "",
        datasetId: options.datasetId
      }),
    refetchInterval: (query) => {
      if (query.state.error) return false
      const current = query.state.data as NotesGraphSuggestionRun | undefined
      return ACTIVE_RUN_STATES.has(current?.state ?? seedRun?.state ?? "stale")
        ? Math.max(250, options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS)
        : false
    },
    refetchIntervalInBackground: false
  })
  const runDetail = runQueryRaw.data
  const activeRun = runDetail ?? seedRun

  const suggestionsKey = React.useMemo(
    () =>
      [
        ...suggestionScopeKey,
        "items",
        SUGGESTION_LIST_STATES.join(","),
        SUGGESTION_LIST_LIMIT
      ] as const,
    [suggestionScopeKey]
  )
  const suggestionsQueryRaw = useQuery({
    queryKey: suggestionsKey,
    enabled,
    notifyOnChangeProps: "all",
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: () =>
      listNotesGraphSuggestions({
        noteId,
        datasetId: options.datasetId,
        states: [...SUGGESTION_LIST_STATES],
        limit: SUGGESTION_LIST_LIMIT
      })
  })
  const suggestionPage = suggestionsQueryRaw.data
  const suggestions = React.useMemo(
    () => suggestionPage?.items ?? [],
    [suggestionPage?.items]
  )
  const currentCommandAuthority = React.useRef({
    commandAuthority,
    allowed: enabled,
    epoch: 0
  })
  if (
    currentCommandAuthority.current.commandAuthority !== commandAuthority ||
    currentCommandAuthority.current.allowed !== enabled
  ) {
    currentCommandAuthority.current = {
      commandAuthority,
      allowed: enabled,
      epoch: currentCommandAuthority.current.epoch + 1
    }
  }
  const authorityEpoch = currentCommandAuthority.current.epoch
  const hasCommandAuthority = (
    expectedCommandAuthority: string,
    commandAuthorityEpoch: number
  ): boolean =>
    currentCommandAuthority.current.allowed &&
    currentCommandAuthority.current.commandAuthority ===
      expectedCommandAuthority &&
    currentCommandAuthority.current.epoch === commandAuthorityEpoch
  const reconciliationScope = JSON.stringify(providerKey)
  const terminalReconciliation = React.useRef({
    scope: reconciliationScope,
    identity: null as string | null
  })
  if (terminalReconciliation.current.scope !== reconciliationScope) {
    terminalReconciliation.current = {
      scope: reconciliationScope,
      identity: null
    }
  }

  React.useEffect(() => {
    if (runDetail?.state !== "succeeded") return
    const reconciliationKey = `${runDetail.id}:${runDetail.revision}`
    if (terminalReconciliation.current.identity === reconciliationKey) return
    terminalReconciliation.current.identity = reconciliationKey
    void queryClient.invalidateQueries({
      queryKey: suggestionsKey,
      exact: true
    })
  }, [queryClient, runDetail, suggestionsKey])

  const generationMutation = useMutation({
    mutationKey: [...providerKey, "generate"],
    retry: retryNetworkOnce,
    mutationFn: async (variables: {
      command: ReturnType<typeof createNotesGraphSuggestionCommand>
      capability: NotesGraphSuggestionCapabilities
      commandAuthority: string
      authorityEpoch: number
      capabilitiesKey: readonly unknown[]
      runsKey: readonly unknown[]
      providerKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
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
              variables.commandAuthority,
              variables.authorityEpoch
            ),
          onCapabilitiesChanged: (nextCapability) => {
            if (
              !hasCommandAuthority(
                variables.commandAuthority,
                variables.authorityEpoch
              )
            )
              return
            queryClient.setQueryData(variables.capabilitiesKey, nextCapability)
          }
        }
      )
    },
    onSuccess: (run, variables) => {
      if (
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
      )
        return
      queryClient.setQueryData<
        Awaited<ReturnType<typeof listNotesGraphSuggestionRuns>>
      >(variables.runsKey, (page) => ({
        items: [
          run,
          ...(page?.items ?? []).filter((item) => item.id !== run.id)
        ],
        next_cursor: page?.next_cursor ?? null
      }))
      queryClient.setQueryData([...variables.providerKey, "run", run.id], run)
    }
  })

  const cancelMutation = useMutation({
    mutationKey: [...providerKey, "cancel"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      run: NotesGraphSuggestionRun
      idempotencyKey: string
      commandAuthority: string
      authorityEpoch: number
      providerKey: readonly unknown[]
      runsKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
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
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
      )
        return
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: [...variables.providerKey, "run"]
        }),
        queryClient.invalidateQueries({
          queryKey: variables.runsKey,
          exact: true
        })
      ])
    }
  })

  const removeSuggestion = React.useCallback(
    (suggestionId: string, commandSuggestionsKey: readonly unknown[]) => {
      queryClient.setQueryData<
        Awaited<ReturnType<typeof listNotesGraphSuggestions>>
      >(commandSuggestionsKey, (page) =>
        page
          ? {
              ...page,
              items: page.items.filter((item) => item.id !== suggestionId)
            }
          : page
      )
    },
    [queryClient]
  )

  const acceptMutation = useMutation({
    mutationKey: [...suggestionScopeKey, "accept"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      item: NotesGraphSuggestion
      idempotencyKey: string
      commandAuthority: string
      authorityEpoch: number
      suggestionsKey: readonly unknown[]
      authorityScope: string
    }) => {
      if (
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
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
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
      )
        return
      removeSuggestion(variables.item.id, variables.suggestionsKey)
      await queryClient.invalidateQueries({
        queryKey: [...notesGraphWorkspaceQueryKey, variables.authorityScope]
      })
    }
  })

  const rejectMutation = useMutation({
    mutationKey: [...suggestionScopeKey, "reject"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      item: NotesGraphSuggestion
      idempotencyKey: string
      commandAuthority: string
      authorityEpoch: number
      suggestionsKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
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
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
      )
        return
      removeSuggestion(variables.item.id, variables.suggestionsKey)
    }
  })

  const resetMutation = useMutation({
    mutationKey: [...suggestionScopeKey, "reset-rejections"],
    retry: retryNetworkOnce,
    mutationFn: (variables: {
      idempotencyKey: string
      page: NonNullable<typeof suggestionPage>
      commandAuthority: string
      authorityEpoch: number
      suggestionsKey: readonly unknown[]
    }) => {
      if (
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
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
        !hasCommandAuthority(
          variables.commandAuthority,
          variables.authorityEpoch
        )
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
      const source = authoritativeNodeId(
        item.source_note_id,
        options.loadedNodeIds
      )
      if (!source) return
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

  return {
    capabilities: capabilities ?? null,
    capabilitiesQuery: capabilitiesQueryRaw,
    runsQuery: runsQueryRaw,
    runQuery: runQueryRaw,
    suggestionsQuery: suggestionsQueryRaw,
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
        commandAuthority,
        authorityEpoch,
        capabilitiesKey,
        runsKey,
        providerKey
      })
    },
    cancel: async () => {
      requireAuthority()
      requireOnline()
      if (!activeRun) throw new Error("No active suggestion run")
      return await cancelMutation.mutateAsync({
        run: activeRun,
        idempotencyKey: idempotencyKey(),
        commandAuthority,
        authorityEpoch,
        providerKey,
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
        commandAuthority,
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
        commandAuthority,
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
        commandAuthority,
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
