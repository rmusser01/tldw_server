import {
  type NotesGraphSuggestion,
  type NotesGraphSuggestionCapabilities,
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

const ACTIVE_RUN_STATES = new Set([
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
  enabled: boolean
  isOnline: boolean
  noteId: string | null
  datasetId?: string
  provider?: string
  model?: string
  loadedNodeIds: ReadonlySet<string>
  pollIntervalMs?: number
}

const prefixedNoteId = (noteId: string): string =>
  noteId.startsWith("note:") ? noteId : `note:${noteId}`

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

const retryNetworkOnce = (failureCount: number, error: unknown): boolean =>
  failureCount < 1 &&
  !isNotesGraphCapabilitiesChangedError(error) &&
  Number((error as { status?: number } | null)?.status ?? 0) === 0

export function useNotesGraphSuggestions(
  options: UseNotesGraphSuggestionsOptions
) {
  const queryClient = useQueryClient()
  const noteId = options.noteId ?? ""
  const baseKey = React.useMemo(
    () =>
      ["notes-graph-suggestions", noteId, options.datasetId ?? null] as const,
    [noteId, options.datasetId]
  )
  const enabled = options.enabled && options.isOnline && Boolean(noteId)
  const [createdRun, setCreatedRun] =
    React.useState<NotesGraphSuggestionRun | null>(null)
  const [lastSuggestions, setLastSuggestions] = React.useState<
    NotesGraphSuggestion[]
  >([])
  const lastPageRef = React.useRef<Awaited<
    ReturnType<typeof listNotesGraphSuggestions>
  > | null>(null)

  React.useEffect(() => {
    setCreatedRun(null)
  }, [noteId, options.datasetId, options.provider, options.model])

  React.useEffect(() => {
    setLastSuggestions([])
    lastPageRef.current = null
  }, [noteId, options.datasetId])

  const capabilitiesQuery = useQuery({
    queryKey: [
      ...baseKey,
      "capabilities",
      options.provider ?? null,
      options.model ?? null
    ],
    enabled,
    retry: false,
    queryFn: () =>
      getNotesGraphSuggestionCapabilities({
        noteId,
        datasetId: options.datasetId,
        provider: options.provider,
        model: options.model
      })
  })

  const runsQuery = useQuery({
    queryKey: [...baseKey, "runs", "active"],
    enabled: enabled && Boolean(capabilitiesQuery.data),
    retry: false,
    queryFn: () =>
      listNotesGraphSuggestionRuns({
        noteId,
        datasetId: options.datasetId,
        states: Array.from(ACTIVE_RUN_STATES),
        limit: 100
      })
  })

  const adoptedRun = React.useMemo(
    () =>
      newestMatchingRun(runsQuery.data?.items ?? [], capabilitiesQuery.data),
    [capabilitiesQuery.data, runsQuery.data?.items]
  )
  const seedRun = createdRun ?? adoptedRun
  const runQuery = useQuery({
    queryKey: [...baseKey, "run", seedRun?.id ?? null],
    enabled:
      enabled &&
      Boolean(seedRun?.id) &&
      ACTIVE_RUN_STATES.has(seedRun?.state ?? ""),
    retry: false,
    queryFn: () =>
      getNotesGraphSuggestionRun({
        noteId,
        runId: seedRun?.id ?? "",
        datasetId: options.datasetId
      }),
    refetchInterval: (query) =>
      ACTIVE_RUN_STATES.has(
        (query.state.data as NotesGraphSuggestionRun | undefined)?.state ??
          seedRun?.state ??
          ""
      )
        ? Math.max(250, options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS)
        : false,
    refetchIntervalInBackground: false
  })
  const activeRun = runQuery.data ?? seedRun

  const suggestionsQuery = useQuery({
    queryKey: [...baseKey, "items"],
    enabled,
    retry: false,
    queryFn: () =>
      listNotesGraphSuggestions({
        noteId,
        datasetId: options.datasetId,
        states: ["pending", "accepting"],
        limit: 100
      })
  })

  React.useEffect(() => {
    if (!suggestionsQuery.data) return
    lastPageRef.current = suggestionsQuery.data
    setLastSuggestions(suggestionsQuery.data.items)
  }, [suggestionsQuery.data])

  const updateCapabilities = React.useCallback(
    (capability: NotesGraphSuggestionCapabilities) => {
      queryClient.setQueryData(
        [
          ...baseKey,
          "capabilities",
          options.provider ?? null,
          options.model ?? null
        ],
        capability
      )
    },
    [baseKey, options.model, options.provider, queryClient]
  )

  const generationMutation = useMutation({
    mutationKey: [...baseKey, "generate"],
    retry: retryNetworkOnce,
    mutationFn: async (
      command: ReturnType<typeof createNotesGraphSuggestionCommand>
    ) => {
      const capability = capabilitiesQuery.data
      if (!capability)
        throw new Error("Notes graph suggestion capabilities are unavailable")
      try {
        return await createNotesGraphSuggestionRun(command, capability, {
          onCapabilitiesChanged: updateCapabilities
        })
      } catch (error) {
        if (!isNotesGraphCapabilitiesChangedError(error)) throw error
        const refreshed = await capabilitiesQuery.refetch()
        if (!refreshed.data) throw error
        updateCapabilities(refreshed.data)
        return await createNotesGraphSuggestionRun(command, refreshed.data, {
          onCapabilitiesChanged: updateCapabilities
        })
      }
    },
    onSuccess: async (run) => {
      setCreatedRun(run)
      await queryClient.invalidateQueries({ queryKey: [...baseKey, "runs"] })
    }
  })

  const cancelMutation = useMutation({
    mutationKey: [...baseKey, "cancel"],
    retry: retryNetworkOnce,
    mutationFn: ({
      run,
      idempotencyKey
    }: {
      run: NotesGraphSuggestionRun
      idempotencyKey: string
    }) =>
      cancelNotesGraphSuggestionRun({
        noteId,
        datasetId: options.datasetId,
        runId: run.id,
        expectedRevision: run.revision,
        idempotencyKey
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: [...baseKey, "run"] })
      await queryClient.invalidateQueries({ queryKey: [...baseKey, "runs"] })
    }
  })

  const invalidateDecisions = React.useCallback(
    async (refreshGraph: boolean) => {
      await queryClient.invalidateQueries({
        queryKey: ["notes-graph-suggestions", noteId]
      })
      if (refreshGraph) {
        await queryClient.invalidateQueries({
          queryKey: [...notesGraphWorkspaceQueryKey]
        })
      }
    },
    [noteId, queryClient]
  )

  const acceptMutation = useMutation({
    mutationKey: [...baseKey, "accept"],
    retry: retryNetworkOnce,
    mutationFn: ({
      item,
      idempotencyKey
    }: {
      item: NotesGraphSuggestion
      idempotencyKey: string
    }) =>
      acceptNotesGraphSuggestion({
        noteId,
        datasetId: options.datasetId,
        suggestionId: item.id,
        expectedRevision: item.revision,
        expectedSourceFingerprint: item.source_fingerprint,
        expectedTargetFingerprint: item.target_fingerprint,
        idempotencyKey
      }),
    onSuccess: () => invalidateDecisions(true)
  })

  const rejectMutation = useMutation({
    mutationKey: [...baseKey, "reject"],
    retry: retryNetworkOnce,
    mutationFn: ({
      item,
      idempotencyKey
    }: {
      item: NotesGraphSuggestion
      idempotencyKey: string
    }) =>
      rejectNotesGraphSuggestion({
        noteId,
        datasetId: options.datasetId,
        suggestionId: item.id,
        expectedRevision: item.revision,
        expectedSourceFingerprint: item.source_fingerprint,
        expectedTargetFingerprint: item.target_fingerprint,
        idempotencyKey
      }),
    onSuccess: () => invalidateDecisions(false)
  })

  const resetMutation = useMutation({
    mutationKey: [...baseKey, "reset-rejections"],
    retry: retryNetworkOnce,
    mutationFn: (idempotencyKey: string) => {
      const page = lastPageRef.current
      if (!page) throw new Error("Suggestion rejection state is unavailable")
      return resetNotesGraphSuggestionRejections({
        noteId,
        datasetId: options.datasetId,
        expectedRejectionRevision: page.rejection_set_revision,
        sourceFingerprint: page.current_source_fingerprint,
        idempotencyKey
      })
    },
    onSuccess: () => invalidateDecisions(false)
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

  const provisionalBySuggestionId = React.useMemo(() => {
    const overlays: Record<string, ProvisionalNotesGraphOverlay> = {}
    lastSuggestions.forEach((item) => {
      if (item.kind !== "related_note" || !item.target_note_id) return
      const source = prefixedNoteId(item.source_note_id)
      const target = prefixedNoteId(item.target_note_id)
      const targetLoaded =
        options.loadedNodeIds.has(target) ||
        options.loadedNodeIds.has(item.target_note_id)
      overlays[item.id] = {
        edge: {
          id: `suggestion-edge:${item.id}`,
          suggestionId: item.id,
          source,
          target: targetLoaded ? target : `suggestion-node:${item.id}`,
          type: "provisional_suggestion",
          directed: false
        },
        node: targetLoaded
          ? null
          : {
              id: `suggestion-node:${item.id}`,
              suggestionId: item.id,
              type: "provisional_note",
              label: "Suggested note"
            }
      }
    })
    return overlays
  }, [lastSuggestions, options.loadedNodeIds])

  return {
    capabilities: capabilitiesQuery.data ?? null,
    capabilitiesQuery,
    runsQuery,
    runQuery,
    suggestionsQuery,
    activeRun,
    suggestions: lastSuggestions,
    provisionalBySuggestionId,
    isOffline: !options.isOnline,
    generate: async () => {
      requireOnline()
      const command = createNotesGraphSuggestionCommand({
        noteId,
        datasetId: options.datasetId,
        provider: options.provider,
        model: options.model
      })
      return await generationMutation.mutateAsync(command)
    },
    cancel: async () => {
      requireOnline()
      if (!activeRun) throw new Error("No active suggestion run")
      return await cancelMutation.mutateAsync({
        run: activeRun,
        idempotencyKey: idempotencyKey()
      })
    },
    accept: async (item: NotesGraphSuggestion) => {
      requireOnline()
      return await acceptMutation.mutateAsync({
        item,
        idempotencyKey: idempotencyKey()
      })
    },
    reject: async (item: NotesGraphSuggestion) => {
      requireOnline()
      return await rejectMutation.mutateAsync({
        item,
        idempotencyKey: idempotencyKey()
      })
    },
    resetRejections: async () => {
      requireOnline()
      return await resetMutation.mutateAsync(idempotencyKey())
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
