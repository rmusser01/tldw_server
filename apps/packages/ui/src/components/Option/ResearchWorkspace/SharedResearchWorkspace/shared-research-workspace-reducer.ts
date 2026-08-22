import type { StructuredApiErrorDetail } from "@/services/tldw/api-error"
import type {
  SharedAllowedActions,
  SharedChatRequest,
  SharedChatResponse,
  SharedMessage,
  SharedMessagePage,
  SharedSourcePage,
  SharedSourcePreview,
  SharedSourceQuery,
  SharedSourceSummary,
  SharedWorkspaceBootstrap
} from "@/types/shared-workspace"

export type SharedWorkspaceStatus =
  | "loading"
  | "loaded"
  | "not-found"
  | "unavailable"

export interface SharedWorkspaceError extends StructuredApiErrorDetail {
  status?: number
}

export interface PendingSharedSubmission {
  request: Readonly<SharedChatRequest>
  status: "submitting" | "retryable"
  submittedDraft: string
  draftRevision: number
}

export interface SharedResearchWorkspaceState {
  shareId: number
  generation: number
  status: SharedWorkspaceStatus
  bootstrap: SharedWorkspaceBootstrap | null
  allowedActions: SharedAllowedActions
  sourceQuery: SharedSourceQuery
  sources: SharedSourcePage | null
  sourceSummary: SharedSourceSummary | null
  sourceScopeMode: "all" | "include"
  selectedSourceIds: string[]
  messages: SharedMessage[]
  nextBefore: string | null
  draft: string
  draftRevision: number
  provider: string | null
  model: string | null
  pendingSubmission: PendingSharedSubmission | null
  preview: SharedSourcePreview | null
  previewLoading: boolean
  previewTarget: { sourceId: string; chunkIndex: number | null } | null
  rateLimitUntil: number | null
  rateLimitRemainingMs: number
  errors: {
    bootstrap: SharedWorkspaceError | null
    sources: SharedWorkspaceError | null
    history: SharedWorkspaceError | null
    preview: SharedWorkspaceError | null
    submission: SharedWorkspaceError | null
  }
}

const deniedAction = { allowed: false, reason_code: "not_loaded" } as const

export const deniedSharedWorkspaceActions = (): SharedAllowedActions => ({
  inspect_sources: { ...deniedAction },
  ask_grounded_questions: { ...deniedAction },
  add_sources: { ...deniedAction },
  edit_workspace: { ...deniedAction },
  clone_workspace: { ...deniedAction }
})

export const createInitialSharedResearchWorkspaceState = (
  shareId: number,
  generation: number
): SharedResearchWorkspaceState => ({
  shareId,
  generation,
  status: "loading",
  bootstrap: null,
  allowedActions: deniedSharedWorkspaceActions(),
  sourceQuery: { offset: 0, limit: 50 },
  sources: null,
  sourceSummary: null,
  sourceScopeMode: "all",
  selectedSourceIds: [],
  messages: [],
  nextBefore: null,
  draft: "",
  draftRevision: 0,
  provider: null,
  model: null,
  pendingSubmission: null,
  preview: null,
  previewLoading: false,
  previewTarget: null,
  rateLimitUntil: null,
  rateLimitRemainingMs: 0,
  errors: {
    bootstrap: null,
    sources: null,
    history: null,
    preview: null,
    submission: null
  }
})

export type SharedResearchWorkspaceAction =
  | { type: "resetForShare"; shareId: number; generation: number }
  | {
      type: "bootstrapSucceeded"
      generation: number
      bootstrap: SharedWorkspaceBootstrap
    }
  | {
      type: "bootstrapFailed"
      generation: number
      error: SharedWorkspaceError
      notFound: boolean
    }
  | { type: "sourcesStarted"; generation: number }
  | {
      type: "sourcesSucceeded"
      generation: number
      query: SharedSourceQuery
      page: SharedSourcePage
    }
  | {
      type: "sourcesFailed"
      generation: number
      error: SharedWorkspaceError
    }
  | {
      type: "historySucceeded"
      generation: number
      page: SharedMessagePage
    }
  | {
      type: "historyFailed"
      generation: number
      error: SharedWorkspaceError
    }
  | {
      type: "previewStarted"
      generation: number
      sourceId: string
      chunkIndex: number | null
    }
  | {
      type: "previewSucceeded"
      generation: number
      preview: SharedSourcePreview
    }
  | {
      type: "previewFailed"
      generation: number
      error: SharedWorkspaceError
    }
  | { type: "draftChanged"; draft: string }
  | { type: "sourceQueryChanged"; query: SharedSourceQuery }
  | { type: "selectedSourcesChanged"; sourceIds: string[] }
  | { type: "allSourcesSelected" }
  | { type: "providerChanged"; provider: string | null }
  | { type: "modelChanged"; model: string | null }
  | {
      type: "submissionStarted"
      generation: number
      request: Readonly<SharedChatRequest>
      submittedDraft: string
      draftRevision: number
    }
  | {
      type: "submissionFailed"
      generation: number
      error: SharedWorkspaceError
      retryableReceipt: boolean
      rateLimitUntil: number | null
      rateLimitRemainingMs: number
    }
  | { type: "rateLimitTick"; now: number }
  | {
      type: "submissionSucceeded"
      generation: number
      response: SharedChatResponse
    }

const keepsGeneration = (
  state: SharedResearchWorkspaceState,
  action: SharedResearchWorkspaceAction
): boolean => !("generation" in action) || action.generation === state.generation

const clearFailedReceipt = (
  state: SharedResearchWorkspaceState
): PendingSharedSubmission | null =>
  state.pendingSubmission?.status === "retryable"
    ? null
    : state.pendingSubmission

const validGenerationDefault = (
  bootstrap: SharedWorkspaceBootstrap
): boolean => {
  const generation = bootstrap.generation_default
  return (
    generation?.ready === true &&
    typeof generation.provider === "string" &&
    Boolean(generation.provider.trim()) &&
    generation.provider.length <= 128 &&
    typeof generation.model === "string" &&
    Boolean(generation.model.trim()) &&
    generation.model.length <= 512
  )
}

const deduplicateHistory = (
  older: SharedMessage[],
  current: SharedMessage[]
): SharedMessage[] => {
  const seen = new Set<string>()
  return [...older, ...current].filter((message) => {
    if (seen.has(message.message_id)) return false
    seen.add(message.message_id)
    return true
  })
}

const uniqueSourceIds = (sourceIds: string[]): string[] =>
  Array.from(new Set(sourceIds))

const isCompleteUnfilteredPage = (
  query: SharedSourceQuery,
  page: SharedSourcePage
): boolean =>
  query.offset === 0 &&
  !query.q?.trim() &&
  !query.state &&
  !page.pagination.has_more

const reconcileSelectedSources = (
  state: SharedResearchWorkspaceState,
  query: SharedSourceQuery,
  page: SharedSourcePage
): string[] => {
  const queryableIds = page.items
    .filter((source) => source.retrieval_ready)
    .map((source) => source.source_id)
  if (isCompleteUnfilteredPage(query, page)) {
    return uniqueSourceIds(
      state.sourceScopeMode === "all"
        ? queryableIds
        : state.selectedSourceIds.filter((id) => queryableIds.includes(id))
    )
  }

  const returnedNonqueryableIds = new Set(
    page.items
      .filter((source) => !source.retrieval_ready)
      .map((source) => source.source_id)
  )
  const selected =
    state.sourceScopeMode === "all"
      ? [...state.selectedSourceIds, ...queryableIds]
      : state.selectedSourceIds
  return uniqueSourceIds(selected).filter(
    (id) => !returnedNonqueryableIds.has(id)
  )
}

export const sharedResearchWorkspaceReducer = (
  state: SharedResearchWorkspaceState,
  action: SharedResearchWorkspaceAction
): SharedResearchWorkspaceState => {
  if (action.type === "resetForShare") {
    return createInitialSharedResearchWorkspaceState(
      action.shareId,
      action.generation
    )
  }
  if (!keepsGeneration(state, action)) return state

  switch (action.type) {
    case "bootstrapSucceeded": {
      const generationReady = validGenerationDefault(action.bootstrap)
      return {
        ...state,
        status: "loaded",
        bootstrap: action.bootstrap,
        allowedActions: action.bootstrap.allowed_actions,
        sources: {
          items: action.bootstrap.sources.items,
          pagination: action.bootstrap.sources.pagination,
          summary: action.bootstrap.source_summary,
          partial_errors: action.bootstrap.partial_errors.filter(
            (error) => error.area === "sources"
          )
        },
        sourceSummary: action.bootstrap.source_summary,
        sourceScopeMode: "all",
        selectedSourceIds: action.bootstrap.sources.items
          .filter((source) => source.retrieval_ready)
          .map((source) => source.source_id),
        messages: action.bootstrap.conversation.messages,
        nextBefore: action.bootstrap.conversation.next_before,
        provider: generationReady
          ? action.bootstrap.generation_default.provider
          : null,
        model: generationReady
          ? action.bootstrap.generation_default.model
          : null,
        errors: { ...state.errors, bootstrap: null }
      }
    }
    case "bootstrapFailed":
      return {
        ...state,
        status: action.notFound ? "not-found" : "unavailable",
        errors: { ...state.errors, bootstrap: action.error }
      }
    case "sourcesStarted":
      return { ...state, errors: { ...state.errors, sources: null } }
    case "sourcesSucceeded": {
      const unfiltered = !action.query.q?.trim() && !action.query.state
      return {
        ...state,
        sources: action.page,
        sourceSummary: unfiltered ? action.page.summary : state.sourceSummary,
        selectedSourceIds: reconcileSelectedSources(
          state,
          action.query,
          action.page
        ),
        errors: { ...state.errors, sources: null }
      }
    }
    case "sourcesFailed":
      return { ...state, errors: { ...state.errors, sources: action.error } }
    case "historySucceeded":
      return {
        ...state,
        messages: deduplicateHistory(action.page.messages, state.messages),
        nextBefore: action.page.next_before,
        errors: { ...state.errors, history: null }
      }
    case "historyFailed":
      return { ...state, errors: { ...state.errors, history: action.error } }
    case "previewStarted":
      return {
        ...state,
        preview: null,
        previewLoading: true,
        previewTarget: {
          sourceId: action.sourceId,
          chunkIndex: action.chunkIndex
        },
        errors: { ...state.errors, preview: null }
      }
    case "previewSucceeded":
      return {
        ...state,
        preview: action.preview,
        previewLoading: false,
        errors: { ...state.errors, preview: null }
      }
    case "previewFailed":
      return {
        ...state,
        preview: null,
        previewLoading: false,
        errors: { ...state.errors, preview: action.error }
      }
    case "draftChanged":
      return {
        ...state,
        draft: action.draft,
        draftRevision: state.draftRevision + 1,
        pendingSubmission: clearFailedReceipt(state),
        errors: { ...state.errors, submission: null }
      }
    case "sourceQueryChanged":
      return {
        ...state,
        sourceQuery: action.query,
        pendingSubmission: clearFailedReceipt(state)
      }
    case "selectedSourcesChanged":
      return {
        ...state,
        sourceScopeMode: "include",
        selectedSourceIds: uniqueSourceIds(action.sourceIds),
        pendingSubmission: clearFailedReceipt(state)
      }
    case "allSourcesSelected":
      return {
        ...state,
        sourceScopeMode: "all",
        pendingSubmission: clearFailedReceipt(state)
      }
    case "providerChanged":
      return {
        ...state,
        provider: action.provider,
        pendingSubmission: clearFailedReceipt(state)
      }
    case "modelChanged":
      return {
        ...state,
        model: action.model,
        pendingSubmission: clearFailedReceipt(state)
      }
    case "submissionStarted":
      return {
        ...state,
        pendingSubmission: {
          request: action.request,
          status: "submitting",
          submittedDraft: action.submittedDraft,
          draftRevision: action.draftRevision
        },
        rateLimitUntil: null,
        rateLimitRemainingMs: 0,
        errors: { ...state.errors, submission: null }
      }
    case "submissionFailed":
      return {
        ...state,
        pendingSubmission:
          action.retryableReceipt && state.pendingSubmission
            ? { ...state.pendingSubmission, status: "retryable" }
            : null,
        rateLimitUntil: action.rateLimitUntil,
        rateLimitRemainingMs: action.rateLimitRemainingMs,
        errors: { ...state.errors, submission: action.error }
      }
    case "rateLimitTick": {
      if (state.rateLimitUntil === null) return state
      const remaining = Math.max(
        0,
        Math.min(state.rateLimitUntil - action.now, 1_800_000)
      )
      return {
        ...state,
        rateLimitUntil: remaining === 0 ? null : state.rateLimitUntil,
        rateLimitRemainingMs: remaining
      }
    }
    case "submissionSucceeded": {
      const pending = state.pendingSubmission
      const userMessage: SharedMessage = {
        ...action.response.turn.user_message,
        citations: []
      }
      const assistantMessage: SharedMessage = {
        ...action.response.turn.assistant_message,
        citations: action.response.citations
      }
      return {
        ...state,
        messages: deduplicateHistory(state.messages, [
          userMessage,
          assistantMessage
        ]),
        draft:
          pending &&
          state.draftRevision === pending.draftRevision &&
          state.draft === pending.submittedDraft
            ? ""
            : state.draft,
        pendingSubmission: null,
        rateLimitUntil: null,
        rateLimitRemainingMs: 0,
        errors: { ...state.errors, submission: null }
      }
    }
  }
}
