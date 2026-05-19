import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ReviewTab } from "../ReviewTab"
import {
  useCramQueueQuery,
  useDeckDueCountsQuery,
  useDecksQuery,
  useDeleteFlashcardMutation,
  useDueCountsQuery,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useFlashcardShortcuts,
  useHasCardsQuery,
  useNextDueQuery,
  useResetFlashcardSchedulingMutation,
  useReviewAnalyticsSummaryQuery,
  useReviewFlashcardMutation,
  useReviewQuery,
  useUpdateFlashcardMutation
} from "../../hooks"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

const assistantMutateAsync = vi.fn()
const speakMock = vi.fn()
const assistantRefetchMock = vi.fn()
const speechRecognitionState = {
  supported: true,
  isListening: false,
  transcript: "",
  start: vi.fn(),
  stop: vi.fn(),
  resetTranscript: vi.fn()
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => vi.fn()
  }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    speak: speakMock,
    cancel: vi.fn(),
    isSpeaking: false
  })
}))

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => speechRecognitionState
}))

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCramQueueQuery: vi.fn(),
  useReviewQuery: vi.fn(),
  useReviewFlashcardMutation: vi.fn(),
  useEndFlashcardReviewSessionMutation: vi.fn(),
  useRecentFlashcardReviewSessionsQuery: vi.fn(),
  useGlobalFlashcardTagSuggestionsQuery: vi.fn(),
  useUpdateFlashcardMutation: vi.fn(),
  useResetFlashcardSchedulingMutation: vi.fn(),
  useDeleteFlashcardMutation: vi.fn(),
  useFlashcardShortcuts: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  useDueCountsQuery: vi.fn(),
  useDeckDueCountsQuery: vi.fn(),
  useReviewAnalyticsSummaryQuery: vi.fn(),
  useHasCardsQuery: vi.fn(),
  useNextDueQuery: vi.fn(),
  useFlashcardAssistantQuery: vi.fn(),
  useFlashcardAssistantRespondMutation: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("ReviewTab study assistant panel", () => {
  let assistantQueryState: any

  beforeEach(() => {
    vi.clearAllMocks()
    speechRecognitionState.supported = true
    speechRecognitionState.isListening = false
    speechRecognitionState.transcript = ""
    assistantRefetchMock.mockReset()

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "card-1",
        deck_id: 1,
        front: "What filters blood?",
        back: "The glomerulus.",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: ["renal"],
        ef: 2.5,
        interval_days: 2,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 2,
        model_type: "basic",
        reverse: false
      }
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useResetFlashcardSchedulingMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useDeleteFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useGlobalFlashcardTagSuggestionsQuery).mockReturnValue({
      data: { items: [] },
      isLoading: false,
      isFetching: false,
      isError: false
    } as any)
    vi.mocked(useFlashcardShortcuts).mockImplementation(() => undefined)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 }
    } as any)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({ data: {} } as any)
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({
      data: null,
      isLoading: false
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: null } as any)
    assistantQueryState = {
      data: {
        thread: {
          id: 9,
          context_type: "flashcard",
          flashcard_uuid: "card-1",
          quiz_attempt_id: null,
          question_id: null,
          last_message_at: "2026-03-13T08:00:00Z",
          message_count: 1,
          deleted: false,
          client_id: "test",
          version: 1,
          created_at: "2026-03-13T08:00:00Z",
          last_modified: "2026-03-13T08:00:00Z"
        },
        messages: [
          {
            id: 12,
            thread_id: 9,
            role: "assistant",
            action_type: "explain",
            input_modality: "text",
            content: "Earlier explanation",
            structured_payload: {},
            context_snapshot: {},
            provider: "openai",
            model: "gpt-5",
            created_at: "2026-03-13T08:00:01Z",
            client_id: "test"
          }
        ],
        context_snapshot: {},
        available_actions: ["explain", "mnemonic", "follow_up", "fact_check", "freeform"]
      },
      isLoading: false,
      isError: false
    }
    assistantRefetchMock.mockImplementation(async () => {
      const nextMessageId = 12 + (assistantQueryState.data?.messages?.length ?? 1)
      assistantQueryState = {
        ...assistantQueryState,
        data: {
          ...assistantQueryState.data,
          thread: {
            ...assistantQueryState.data.thread,
            version: 2,
            message_count: 2
          },
          messages: [
            ...(assistantQueryState.data?.messages ?? []),
            {
              id: nextMessageId,
              thread_id: 9,
              role: "assistant",
              action_type: "follow_up",
              input_modality: "text",
              content: "Fresh thread update",
              structured_payload: {},
              context_snapshot: {},
              provider: "openai",
              model: "gpt-5",
              created_at: "2026-03-13T08:05:00Z",
              client_id: "test"
            }
          ]
        }
      }
      return assistantQueryState
    })
    vi.mocked(useFlashcardAssistantQuery).mockImplementation(
      () =>
        ({
          ...assistantQueryState,
          refetch: assistantRefetchMock
        }) as any
    )
    vi.mocked(useFlashcardAssistantRespondMutation).mockReturnValue({
      mutateAsync: assistantMutateAsync,
      isPending: false
    } as any)
  })

  const renderReviewTab = () =>
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

  it("renders assistant quick actions and existing history on the active card", () => {
    renderReviewTab()

    expect(screen.getByTestId("flashcards-review-study-assistant")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Explain" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Fact-check me" })).toBeInTheDocument()
    expect(screen.getByText("Earlier explanation")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Play reply" })).toBeInTheDocument()
  })

  it("submits explain actions immediately", async () => {
    assistantMutateAsync.mockResolvedValue({
      assistant_message: { content: "Fresh explanation" }
    })
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Explain" }))

    await waitFor(() => {
      expect(assistantMutateAsync).toHaveBeenCalledWith({
        cardUuid: "card-1",
        request: { action: "explain" }
      })
    })
  })

  it("requires transcript confirmation before submitting fact-check requests", async () => {
    assistantMutateAsync.mockResolvedValue({
      assistant_message: { content: "Fact check reply" }
    })
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Fact-check me" }))
    expect(assistantMutateAsync).not.toHaveBeenCalled()
    expect(screen.getByText("Confirm transcript")).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Transcript"), {
      target: { value: "I think the glomerulus filters blood." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send fact-check" }))

    await waitFor(() => {
      expect(assistantMutateAsync).toHaveBeenCalledWith({
        cardUuid: "card-1",
        request: {
          action: "fact_check",
          message: "I think the glomerulus filters blood.",
          input_modality: "voice_transcript"
        }
      })
    })
  })

  it("plays back assistant replies on demand", async () => {
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Play reply" }))

    await waitFor(() => {
      expect(speakMock).toHaveBeenCalledWith({
        utterance: "Earlier explanation"
      })
    })
  })

  it("shows assistant errors without blocking review controls", async () => {
    assistantMutateAsync.mockRejectedValue(new Error("assistant failed"))
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Explain" }))

    await waitFor(() => {
      expect(
        screen.getByText(
          "Study assistant requires an LLM provider. Configure one in Settings → LLM Providers."
        )
      ).toBeInTheDocument()
    })
    expect(screen.getByTestId("flashcards-review-show-answer")).toBeInTheDocument()
  })

  it("reloads the latest thread and offers retry actions after a conflict", async () => {
    assistantMutateAsync
      .mockRejectedValueOnce(Object.assign(new Error("Version mismatch"), { response: { status: 409 } }))
      .mockResolvedValueOnce({
        assistant_message: { content: "Retried explanation" }
      })
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Explain" }))

    await waitFor(() => {
      expect(assistantRefetchMock).toHaveBeenCalled()
      expect(screen.getByText("Fresh thread update")).toBeInTheDocument()
      expect(screen.getByText("Conversation changed elsewhere.")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry my message" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry my message" }))

    await waitFor(() => {
      expect(assistantMutateAsync).toHaveBeenNthCalledWith(2, {
        cardUuid: "card-1",
        request: { action: "explain" }
      })
    })
  })

  it("clears the pending request when reloading the latest thread", async () => {
    assistantMutateAsync.mockRejectedValueOnce(
      Object.assign(new Error("Version mismatch"), { response: { status: 409 } })
    )
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Explain" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Reload latest/i })).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry my message" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Reload latest/i }))

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Retry my message" })).not.toBeInTheDocument()
    })
    expect(assistantMutateAsync).toHaveBeenCalledTimes(1)
  })

  it("preserves transcript fact-check requests across conflict recovery", async () => {
    assistantMutateAsync
      .mockRejectedValueOnce(Object.assign(new Error("Version mismatch"), { response: { status: 409 } }))
      .mockResolvedValueOnce({
        assistant_message: { content: "Retried fact-check" }
      })
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Fact-check me" }))
    fireEvent.change(screen.getByLabelText("Transcript"), {
      target: { value: "I think the glomerulus filters blood." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send fact-check" }))

    await waitFor(() => {
      expect(screen.getByText("Conversation changed elsewhere.")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry transcript review" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry transcript review" }))

    await waitFor(() => {
      expect(assistantMutateAsync).toHaveBeenNthCalledWith(2, {
        cardUuid: "card-1",
        request: {
          action: "fact_check",
          message: "I think the glomerulus filters blood.",
          input_modality: "voice_transcript"
        }
      })
    })
  })

  it("uses refreshed top-level remediation context after conflict recovery", async () => {
    assistantMutateAsync.mockRejectedValueOnce(
      Object.assign(new Error("Version mismatch"), { response: { status: 409 } })
    )
    assistantRefetchMock.mockImplementationOnce(async () => {
      assistantQueryState = {
        ...assistantQueryState,
        data: {
          ...assistantQueryState.data,
          thread: {
            ...assistantQueryState.data.thread,
            version: 2,
            message_count: 2
          },
          messages: [
            ...(assistantQueryState.data?.messages ?? []),
            {
              id: 13,
              thread_id: 9,
              role: "assistant",
              action_type: "follow_up",
              input_modality: "text",
              content: "Fresh thread update",
              structured_payload: {},
              context_snapshot: {},
              provider: "openai",
              model: "gpt-5",
              created_at: "2026-03-13T08:05:00Z",
              client_id: "test"
            }
          ],
          citations: [
            {
              id: 1,
              flashcard_uuid: "card-1",
              source_type: "note",
              source_id: "88",
              citation_text: "Retry context adds the missing remediation quote.",
              locator: "{\"section\":\"retry-context\"}",
              ordinal: 0,
              deleted: false,
              client_id: "test",
              version: 1
            }
          ],
          primary_citation: {
            id: 1,
            flashcard_uuid: "card-1",
            source_type: "note",
            source_id: "88",
            citation_text: "Retry context adds the missing remediation quote.",
            locator: "{\"section\":\"retry-context\"}",
            ordinal: 0,
            deleted: false,
            client_id: "test",
            version: 1
          },
          deep_dive_target: {
            source_type: "note",
            source_id: "88",
            citation_ordinal: 0,
            route_kind: "exact_locator",
            route: "/notes/88?section=retry-context",
            available: true,
            fallback_reason: null
          }
        }
      }
      return assistantQueryState
    })
    renderReviewTab()

    fireEvent.click(screen.getByRole("button", { name: "Explain" }))

    await waitFor(() => {
      expect(screen.getByText(/Retry context adds the missing remediation quote\./)).toBeInTheDocument()
      expect(screen.getByRole("link", { name: "Deep dive to source" })).toHaveAttribute(
        "href",
        "/notes/88?section=retry-context"
      )
    })
  })
})
