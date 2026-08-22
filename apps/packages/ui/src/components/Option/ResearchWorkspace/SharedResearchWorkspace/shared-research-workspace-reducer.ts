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
}

export interface SharedResearchWorkspaceState {
  shareId: number
  generation: number
  status: SharedWorkspaceStatus
  bootstrap: SharedWorkspaceBootstrap | null
  allowedActions: SharedAllowedActions
  sourceQuery: SharedSourceQuery
  sources: SharedSourcePage | null
  selectedSourceIds: string[]
  messages: SharedMessage[]
  nextBefore: string | null
  draft: string
  provider: string | null
  model: string | null
  pendingSubmission: PendingSharedSubmission | null
  preview: SharedSourcePreview | null
  rateLimitUntil: number | null
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
  selectedSourceIds: [],
  messages: [],
  nextBefore: null,
  draft: "",
  provider: null,
  model: null,
  pendingSubmission: null,
  preview: null,
  rateLimitUntil: null,
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
  | { type: "sourcesSucceeded"; generation: number; page: SharedSourcePage }
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
  | { type: "providerChanged"; provider: string | null }
  | { type: "modelChanged"; model: string | null }
  | {
      type: "submissionStarted"
      generation: number
      request: Readonly<SharedChatRequest>
    }
  | {
      type: "submissionFailed"
      generation: number
      error: SharedWorkspaceError
      retryableReceipt: boolean
      rateLimitUntil: number | null
    }
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
      const allowedActions = generationReady
        ? action.bootstrap.allowed_actions
        : {
            ...action.bootstrap.allowed_actions,
            ask_grounded_questions: {
              allowed: false,
              reason_code: "generation_default_unavailable"
            }
          }
      return {
        ...state,
        status: "loaded",
        bootstrap: action.bootstrap,
        allowedActions,
        sources: {
          items: action.bootstrap.sources.items,
          pagination: action.bootstrap.sources.pagination,
          summary: action.bootstrap.source_summary,
          partial_errors: action.bootstrap.partial_errors.filter(
            (error) => error.area === "sources"
          )
        },
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
      const queryable = new Set(
        action.page.items
          .filter((source) => source.retrieval_ready)
          .map((source) => source.source_id)
      )
      return {
        ...state,
        sources: action.page,
        selectedSourceIds: state.selectedSourceIds.filter((id) =>
          queryable.has(id)
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
    case "previewSucceeded":
      return {
        ...state,
        preview: action.preview,
        errors: { ...state.errors, preview: null }
      }
    case "previewFailed":
      return { ...state, errors: { ...state.errors, preview: action.error } }
    case "draftChanged":
      return {
        ...state,
        draft: action.draft,
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
        selectedSourceIds: [...action.sourceIds],
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
        pendingSubmission: { request: action.request, status: "submitting" },
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
        errors: { ...state.errors, submission: action.error }
      }
    case "submissionSucceeded": {
      const pendingQuery = state.pendingSubmission?.request.query
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
        draft: state.draft.trim() === pendingQuery ? "" : state.draft,
        pendingSubmission: null,
        rateLimitUntil: null,
        errors: { ...state.errors, submission: null }
      }
    }
  }
}
