import React from "react"
import {
  getStructuredApiErrorDetail,
  type StructuredApiErrorDetail
} from "@/services/tldw/api-error"
import { sharedWorkspacesApi } from "@/services/tldw/domains/shared-workspaces"
import type {
  SharedChatRequest,
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

type Operation = "bootstrap" | "sources" | "history" | "preview" | "submission"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException
    ? error.name === "AbortError"
    : isRecord(error) && error.name === "AbortError"

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
  sourceIds: string[],
  provider: string | null,
  model: string | null
): Readonly<SharedChatRequest> => {
  const ids = Object.freeze(Array.from(new Set(sourceIds)).sort())
  const sourceScope = Object.freeze({
    mode: "include" as const,
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

  const releaseOperation = React.useCallback(
    (operation: Operation, controller: AbortController) => {
      if (controllersRef.current[operation] === controller) {
        delete controllersRef.current[operation]
      }
    },
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
        dispatch({ type: "bootstrapSucceeded", generation, bootstrap })
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return
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
  }, [abortAll, releaseOperation, shareId, startOperation])

  const refreshSources = React.useCallback(
    async (query: SharedSourceQuery = state.sourceQuery): Promise<void> => {
      const generation = generationRef.current
      const controller = startOperation("sources")
      dispatch({ type: "sourcesStarted", generation })
      try {
        const page = await sharedWorkspacesApi.listSources(
          shareId,
          query,
          controller.signal
        )
        dispatch({ type: "sourcesSucceeded", generation, page })
      } catch (error) {
        if (!isAbortError(error)) {
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
    [releaseOperation, shareId, startOperation, state.sourceQuery]
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
      dispatch({ type: "historySucceeded", generation, page })
    } catch (error) {
      if (!isAbortError(error)) {
        dispatch({
          type: "historyFailed",
          generation,
          error: normalizeError(error)
        })
      }
    } finally {
      releaseOperation("history", controller)
    }
  }, [releaseOperation, shareId, startOperation, state.nextBefore])

  const previewSource = React.useCallback(
    async (sourceId: string, chunkIndex?: number): Promise<void> => {
      const generation = generationRef.current
      const controller = startOperation("preview")
      try {
        const preview = await sharedWorkspacesApi.previewSource(
          shareId,
          sourceId,
          chunkIndex,
          controller.signal
        )
        dispatch({ type: "previewSucceeded", generation, preview })
      } catch (error) {
        if (!isAbortError(error)) {
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
    [releaseOperation, shareId, startOperation]
  )

  const sendRequest = React.useCallback(
    async (request: Readonly<SharedChatRequest>): Promise<void> => {
      const generation = generationRef.current
      const controller = startOperation("submission")
      dispatch({ type: "submissionStarted", generation, request })
      try {
        const response = await sharedWorkspacesApi.ask(
          shareId,
          request as SharedChatRequest,
          controller.signal
        )
        dispatch({ type: "submissionSucceeded", generation, response })
      } catch (error) {
        if (
          isAbortError(error) ||
          controller.signal.aborted ||
          generation !== generationRef.current
        ) {
          return
        }
        const structured = getStructuredApiErrorDetail(error)
        const normalized = normalizeError(error)
        const sourceChanged = normalized.code === "shared_source_changed"
        const retryAfter = normalized.retry_after_ms
        dispatch({
          type: "submissionFailed",
          generation,
          error: normalized,
          retryableReceipt: !structured?.code,
          rateLimitUntil:
            retryAfter === undefined ||
            (normalized.code !== "shared_chat_rate_limited" &&
              normalized.code !== "request_in_progress")
              ? null
              : Date.now() + Math.min(retryAfter, 1_800_000)
        })
        if (sourceChanged) await refreshSources()
      } finally {
        releaseOperation("submission", controller)
      }
    },
    [refreshSources, releaseOperation, shareId, startOperation]
  )

  const submitDraft = React.useCallback(async (): Promise<void> => {
    const query = state.draft.trim()
    if (
      !query ||
      !state.allowedActions.ask_grounded_questions.allowed ||
      !state.provider ||
      !state.model ||
      state.selectedSourceIds.length === 0 ||
      state.selectedSourceIds.length > 500
    ) {
      return
    }
    const request = immutableRequest(
      (options.createRequestId ?? defaultRequestId)(),
      query,
      state.selectedSourceIds,
      state.provider,
      state.model
    )
    await sendRequest(request)
  }, [
    options.createRequestId,
    sendRequest,
    state.allowedActions.ask_grounded_questions.allowed,
    state.draft,
    state.model,
    state.provider,
    state.selectedSourceIds
  ])

  const retryPending = React.useCallback(async (): Promise<void> => {
    const pending = state.pendingSubmission
    if (pending?.status !== "retryable") return
    const currentSourceIds = Array.from(new Set(state.selectedSourceIds)).sort()
    if (
      pending.request.query !== state.draft.trim() ||
      pending.request.provider !== state.provider ||
      pending.request.model !== state.model ||
      JSON.stringify(pending.request.source_scope.source_ids) !==
        JSON.stringify(currentSourceIds)
    ) {
      return
    }
    await sendRequest(pending.request)
  }, [
    sendRequest,
    state.draft,
    state.model,
    state.pendingSubmission,
    state.provider,
    state.selectedSourceIds
  ])

  return {
    state,
    refreshSources,
    loadOlderHistory,
    previewSource,
    submitDraft,
    retryPending,
    setDraft: (draft: string) => dispatch({ type: "draftChanged", draft }),
    setSourceQuery: (query: SharedSourceQuery) =>
      dispatch({ type: "sourceQueryChanged", query }),
    setSelectedSourceIds: (sourceIds: string[]) =>
      dispatch({ type: "selectedSourcesChanged", sourceIds }),
    setProvider: (provider: string | null) =>
      dispatch({ type: "providerChanged", provider }),
    setModel: (model: string | null) =>
      dispatch({ type: "modelChanged", model })
  }
}

export type SharedResearchWorkspaceController = ReturnType<
  typeof useSharedResearchWorkspace
>
