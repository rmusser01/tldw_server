import React from "react"
import {
  getStructuredApiErrorDetail,
  type StructuredApiErrorDetail
} from "@/services/tldw/api-error"
import {
  isSharedWorkspacePostCommitResponseError,
  SharedWorkspacePostCommitResponseError,
  sharedWorkspacesApi
} from "@/services/tldw/domains/shared-workspaces"
import type {
  SharedChatRequest,
  SharedSource,
  SharedSourcePage,
  SharedSourceQuery
} from "@/types/shared-workspace"
import {
  createInitialSharedResearchWorkspaceState,
  sharedResearchWorkspaceReducer,
  type SharedWorkspaceError
} from "./shared-research-workspace-reducer"

interface SharedResearchWorkspaceOptions {
  createRequestId?: () => string
}

type Operation =
  | "bootstrap"
  | "sources"
  | "selection"
  | "history"
  | "preview"
  | "submission"

const SOURCE_MATERIALIZATION_LIMIT = 200
const MAX_SOURCE_MATERIALIZATION_PAGES = 100
const SOURCE_SELECTION_ERROR: SharedWorkspaceError = {
  code: "shared_source_selection_unavailable",
  message:
    "Couldn't load every queryable source. All queryable sources remain selected.",
  retryable: true,
  recovery_action: "retry"
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException
    ? error.name === "AbortError"
    : isRecord(error) && error.name === "AbortError"

const isAmbiguousPostCommitFailure = (error: unknown): boolean =>
  isSharedWorkspacePostCommitResponseError(error) ||
  (error instanceof TypeError &&
    !(isRecord(error) && typeof error.status === "number"))

const sourceSummariesMatch = (
  left: SharedSourcePage["summary"],
  right: SharedSourcePage["summary"]
): boolean =>
  left.total === right.total &&
  left.queryable === right.queryable &&
  left.processing === right.processing &&
  left.failed === right.failed

const normalizeError = (error: unknown): SharedWorkspaceError => {
  const structured: StructuredApiErrorDetail =
    getStructuredApiErrorDetail(error) ?? {}
  return {
    ...structured,
    status:
      isRecord(error) && typeof error.status === "number"
        ? error.status
        : undefined,
    code: structured.code ?? "shared_workspace_network_error",
    message:
      structured.message ??
      (error instanceof Error ? error.message : "Shared workspace request failed."),
    retryable: structured.retryable ?? true
  }
}

const defaultRequestId = (): string => {
  if (typeof globalThis.crypto?.randomUUID !== "function") {
    throw new Error("Secure request UUID generation is unavailable")
  }
  return globalThis.crypto.randomUUID()
}

const immutableRequest = (
  requestId: string,
  query: string,
  sourceScopeMode: "all" | "include",
  sourceIds: string[],
  provider: string | null,
  model: string | null
): Readonly<SharedChatRequest> => {
  const ids = Object.freeze(
    sourceScopeMode === "all"
      ? []
      : Array.from(new Set(sourceIds)).sort()
  )
  const sourceScope = Object.freeze({
    mode: sourceScopeMode,
    source_ids: ids
  })
  return Object.freeze({
    request_id: requestId,
    query,
    source_scope: sourceScope,
    provider,
    model
  }) as Readonly<SharedChatRequest>
}

export const useSharedResearchWorkspace = (
  shareId: number,
  options: SharedResearchWorkspaceOptions = {}
) => {
  const [state, dispatch] = React.useReducer(
    sharedResearchWorkspaceReducer,
    createInitialSharedResearchWorkspaceState(shareId, 0)
  )
  const generationRef = React.useRef(0)
  const controllersRef = React.useRef<
    Partial<Record<Operation, AbortController>>
  >({})

  const abortAll = React.useCallback(() => {
    Object.values(controllersRef.current).forEach((controller) =>
      controller?.abort()
    )
    controllersRef.current = {}
  }, [])

  const startOperation = React.useCallback((operation: Operation) => {
    controllersRef.current[operation]?.abort()
    const controller = new AbortController()
    controllersRef.current[operation] = controller
    return controller
  }, [])

  const cancelOperation = React.useCallback((operation: Operation) => {
    controllersRef.current[operation]?.abort()
    delete controllersRef.current[operation]
  }, [])

  const releaseOperation = React.useCallback(
    (operation: Operation, controller: AbortController) => {
      if (controllersRef.current[operation] === controller) {
        delete controllersRef.current[operation]
      }
    },
    []
  )

  const isCurrentOperation = React.useCallback(
    (operation: Operation, controller: AbortController): boolean =>
      controllersRef.current[operation] === controller &&
      !controller.signal.aborted,
    []
  )

  React.useLayoutEffect(() => {
    abortAll()
    const generation = generationRef.current + 1
    generationRef.current = generation
    dispatch({ type: "resetForShare", shareId, generation })

    const controller = startOperation("bootstrap")
    void sharedWorkspacesApi
      .bootstrap(shareId, controller.signal)
      .then((bootstrap) => {
        if (!isCurrentOperation("bootstrap", controller)) return
        dispatch({ type: "bootstrapSucceeded", generation, bootstrap })
      })
      .catch((error: unknown) => {
        if (
          isAbortError(error) ||
          !isCurrentOperation("bootstrap", controller)
        ) {
          return
        }
        const normalized = normalizeError(error)
        dispatch({
          type: "bootstrapFailed",
          generation,
          error: normalized,
          notFound:
            normalized.status === 404 ||
            normalized.code === "shared_workspace_not_found"
        })
      })
      .finally(() => releaseOperation("bootstrap", controller))

    return abortAll
  }, [
    abortAll,
    isCurrentOperation,
    releaseOperation,
    shareId,
    startOperation
  ])

  React.useEffect(() => {
    if (state.rateLimitUntil === null) return
    const tick = () => dispatch({ type: "rateLimitTick", now: Date.now() })
    const timer = globalThis.setInterval(tick, 250)
    return () => globalThis.clearInterval(timer)
  }, [state.rateLimitUntil])

  const refreshSources = React.useCallback(
    async (query: SharedSourceQuery = state.sourceQuery): Promise<void> => {
      if (!state.allowedActions.inspect_sources.allowed) return
      const generation = generationRef.current
      const controller = startOperation("sources")
      dispatch({ type: "sourcesStarted", generation })
      try {
        const page = await sharedWorkspacesApi.listSources(
          shareId,
          query,
          controller.signal
        )
        if (!isCurrentOperation("sources", controller)) return
        dispatch({ type: "sourcesSucceeded", generation, query, page })
      } catch (error) {
        if (
          !isAbortError(error) &&
          isCurrentOperation("sources", controller)
        ) {
          dispatch({
            type: "sourcesFailed",
            generation,
            error: normalizeError(error)
          })
        }
      } finally {
        releaseOperation("sources", controller)
      }
    },
    [
      isCurrentOperation,
      releaseOperation,
      shareId,
      startOperation,
      state.allowedActions.inspect_sources.allowed,
      state.sourceQuery
    ]
  )

  const loadOlderHistory = React.useCallback(async (): Promise<void> => {
    if (!state.nextBefore) return
    const generation = generationRef.current
    const controller = startOperation("history")
    try {
      const page = await sharedWorkspacesApi.listMessages(
        shareId,
        state.nextBefore,
        controller.signal
      )
      if (!isCurrentOperation("history", controller)) return
      dispatch({ type: "historySucceeded", generation, page })
    } catch (error) {
      if (
        !isAbortError(error) &&
        isCurrentOperation("history", controller)
      ) {
        dispatch({
          type: "historyFailed",
          generation,
          error: normalizeError(error)
        })
      }
    } finally {
      releaseOperation("history", controller)
    }
  }, [
    isCurrentOperation,
    releaseOperation,
    shareId,
    startOperation,
    state.nextBefore
  ])

  const previewSource = React.useCallback(
    async (sourceId: string, chunkIndex?: number): Promise<void> => {
      if (!state.allowedActions.inspect_sources.allowed) return
      const generation = generationRef.current
      const controller = startOperation("preview")
      dispatch({
        type: "previewStarted",
        generation,
        sourceId,
        chunkIndex: chunkIndex ?? null
      })
      try {
        const preview = await sharedWorkspacesApi.previewSource(
          shareId,
          sourceId,
          chunkIndex,
          controller.signal
        )
        if (!isCurrentOperation("preview", controller)) return
        dispatch({ type: "previewSucceeded", generation, preview })
      } catch (error) {
        if (
          !isAbortError(error) &&
          isCurrentOperation("preview", controller)
        ) {
          dispatch({
            type: "previewFailed",
            generation,
            error: normalizeError(error)
          })
        }
      } finally {
        releaseOperation("preview", controller)
      }
    },
    [
      isCurrentOperation,
      releaseOperation,
      shareId,
      startOperation,
      state.allowedActions.inspect_sources.allowed
    ]
  )

  const toggleSource = React.useCallback(
    async (source: SharedSource, checked: boolean): Promise<void> => {
      if (
        !state.allowedActions.inspect_sources.allowed ||
        !source.retrieval_ready ||
        state.selectionMaterializing
      ) {
        return
      }

      if (state.sourceScopeMode === "include") {
        const selected = new Set(state.selectedSourceIds)
        if (checked) selected.add(source.source_id)
        else selected.delete(source.source_id)
        dispatch({
          type: "selectedSourcesChanged",
          sourceIds: Array.from(selected)
        })
        return
      }

      if (checked) return
      const expectedSummary = state.sourceSummary
      if (
        !expectedSummary ||
        expectedSummary.queryable === 0 ||
        expectedSummary.queryable > 500
      ) {
        return
      }

      const generation = generationRef.current
      const controller = startOperation("selection")
      dispatch({ type: "selectionStarted", generation })
      try {
        const seenSourceIds = new Set<string>()
        const queryableSourceIds: string[] = []
        let offset = 0
        let pageCount = 0
        let complete = false

        while (!complete) {
          pageCount += 1
          if (pageCount > MAX_SOURCE_MATERIALIZATION_PAGES) {
            throw new Error("Shared source selection exceeded the page limit")
          }

          const page = await sharedWorkspacesApi.listSources(
            shareId,
            { offset, limit: SOURCE_MATERIALIZATION_LIMIT },
            controller.signal
          )
          if (!isCurrentOperation("selection", controller)) return

          if (
            page.partial_errors.length > 0 ||
            page.pagination.offset !== offset ||
            page.pagination.total !== expectedSummary.total ||
            !sourceSummariesMatch(page.summary, expectedSummary) ||
            page.items.length > page.pagination.limit
          ) {
            throw new Error("Shared source pagination changed")
          }

          for (const item of page.items) {
            if (seenSourceIds.has(item.source_id)) {
              throw new Error("Shared source pagination contained duplicates")
            }
            seenSourceIds.add(item.source_id)
            if (item.retrieval_ready) queryableSourceIds.push(item.source_id)
          }

          const nextOffset = offset + page.items.length
          if (
            nextOffset > expectedSummary.total ||
            (page.pagination.has_more &&
              (page.items.length === 0 || nextOffset >= expectedSummary.total)) ||
            (!page.pagination.has_more && nextOffset !== expectedSummary.total)
          ) {
            throw new Error("Shared source pagination was incomplete")
          }
          complete = !page.pagination.has_more
          offset = nextOffset
        }

        if (
          seenSourceIds.size !== expectedSummary.total ||
          queryableSourceIds.length !== expectedSummary.queryable ||
          !queryableSourceIds.includes(source.source_id)
        ) {
          throw new Error("Shared source selection was incomplete")
        }

        if (!isCurrentOperation("selection", controller)) return
        dispatch({
          type: "selectionSucceeded",
          generation,
          sourceIds: queryableSourceIds.filter(
            (sourceId) => sourceId !== source.source_id
          )
        })
      } catch (error) {
        if (
          !isAbortError(error) &&
          isCurrentOperation("selection", controller) &&
          generation === generationRef.current
        ) {
          dispatch({
            type: "selectionFailed",
            generation,
            error: SOURCE_SELECTION_ERROR
          })
        }
      } finally {
        releaseOperation("selection", controller)
      }
    },
    [
      isCurrentOperation,
      releaseOperation,
      shareId,
      startOperation,
      state.allowedActions.inspect_sources.allowed,
      state.selectedSourceIds,
      state.selectionMaterializing,
      state.sourceScopeMode,
      state.sourceSummary
    ]
  )

  const sendRequest = React.useCallback(
    async (
      request: Readonly<SharedChatRequest>,
      submittedDraft: string,
      draftRevision: number
    ): Promise<void> => {
      const generation = generationRef.current
      const controller = startOperation("submission")
      dispatch({
        type: "submissionStarted",
        generation,
        request,
        submittedDraft,
        draftRevision
      })
      try {
        const response = await sharedWorkspacesApi.ask(
          shareId,
          request as SharedChatRequest,
          controller.signal
        )
        if (!isCurrentOperation("submission", controller)) return
        if (response.request_id !== request.request_id) {
          throw new SharedWorkspacePostCommitResponseError({
            code: "shared_chat_response_mismatch",
            message:
              "The answer status is uncertain because the response did not match the request. Retry to reconcile this question.",
            retryable: true,
            recovery_action: "retry"
          })
        }
        dispatch({ type: "submissionSucceeded", generation, response })
      } catch (error) {
        if (
          isAbortError(error) ||
          !isCurrentOperation("submission", controller) ||
          generation !== generationRef.current
        ) {
          return
        }
        const normalized = normalizeError(error)
        const sourceChanged = normalized.code === "shared_source_changed"
        const retryAfter = normalized.retry_after_ms
        const rateLimitRemainingMs =
          retryAfter === undefined ||
          (normalized.code !== "shared_chat_rate_limited" &&
            normalized.code !== "request_in_progress")
            ? 0
            : Math.min(retryAfter, 1_800_000)
        dispatch({
          type: "submissionFailed",
          generation,
          error: normalized,
          retryableReceipt: isAmbiguousPostCommitFailure(error),
          rateLimitUntil:
            rateLimitRemainingMs === 0
              ? null
              : Date.now() + rateLimitRemainingMs,
          rateLimitRemainingMs
        })
        if (sourceChanged) await refreshSources()
      } finally {
        releaseOperation("submission", controller)
      }
    },
    [
      isCurrentOperation,
      refreshSources,
      releaseOperation,
      shareId,
      startOperation
    ]
  )

  const submitDraft = React.useCallback(async (): Promise<void> => {
    const query = state.draft.trim()
    if (
      !query ||
      !state.allowedActions.ask_grounded_questions.allowed ||
      !state.provider ||
      !state.model ||
      state.selectionMaterializing ||
      (state.rateLimitUntil !== null && Date.now() < state.rateLimitUntil) ||
      (state.sourceScopeMode === "all"
        ? !state.sourceSummary ||
          state.sourceSummary.queryable === 0 ||
          state.sourceSummary.queryable > 500
        : state.selectedSourceIds.length === 0 ||
          state.selectedSourceIds.length > 500)
    ) {
      return
    }
    const request = immutableRequest(
      (options.createRequestId ?? defaultRequestId)(),
      query,
      state.sourceScopeMode,
      state.selectedSourceIds,
      state.provider,
      state.model
    )
    await sendRequest(request, state.draft, state.draftRevision)
  }, [
    options.createRequestId,
    sendRequest,
    state.allowedActions.ask_grounded_questions.allowed,
    state.draft,
    state.draftRevision,
    state.model,
    state.provider,
    state.rateLimitUntil,
    state.selectionMaterializing,
    state.sourceScopeMode,
    state.sourceSummary,
    state.selectedSourceIds
  ])

  const retryPending = React.useCallback(async (): Promise<void> => {
    const pending = state.pendingSubmission
    if (pending?.status !== "retryable") return
    if (state.rateLimitUntil !== null && Date.now() < state.rateLimitUntil) {
      return
    }
    const currentSourceIds = Array.from(new Set(state.selectedSourceIds)).sort()
    if (
      pending.request.query !== state.draft.trim() ||
      pending.request.provider !== state.provider ||
      pending.request.model !== state.model ||
      pending.request.source_scope.mode !== state.sourceScopeMode ||
      JSON.stringify(pending.request.source_scope.source_ids) !==
        JSON.stringify(
          state.sourceScopeMode === "all" ? [] : currentSourceIds
        )
    ) {
      return
    }
    await sendRequest(
      pending.request,
      pending.submittedDraft,
      pending.draftRevision
    )
  }, [
    sendRequest,
    state.draft,
    state.model,
    state.pendingSubmission,
    state.provider,
    state.rateLimitUntil,
    state.sourceScopeMode,
    state.selectedSourceIds
  ])

  return {
    state,
    refreshSources,
    loadOlderHistory,
    previewSource,
    toggleSource,
    submitDraft,
    retryPending,
    setDraft: (draft: string) => dispatch({ type: "draftChanged", draft }),
    setSourceQuery: (query: SharedSourceQuery) => {
      if (state.allowedActions.inspect_sources.allowed) {
        dispatch({ type: "sourceQueryChanged", query })
      }
    },
    setSelectedSourceIds: (sourceIds: string[]) => {
      if (state.allowedActions.inspect_sources.allowed) {
        cancelOperation("selection")
        dispatch({ type: "selectedSourcesChanged", sourceIds })
      }
    },
    selectAllSources: () => {
      if (state.allowedActions.inspect_sources.allowed) {
        cancelOperation("selection")
        dispatch({ type: "allSourcesSelected" })
      }
    },
    clearSelectedSources: () => {
      if (state.allowedActions.inspect_sources.allowed) {
        cancelOperation("selection")
        dispatch({ type: "selectedSourcesChanged", sourceIds: [] })
      }
    },
    setProvider: (provider: string | null) =>
      dispatch({ type: "providerChanged", provider }),
    setModel: (model: string | null) =>
      dispatch({ type: "modelChanged", model })
  }
}

export type SharedResearchWorkspaceController = ReturnType<
  typeof useSharedResearchWorkspace
>
