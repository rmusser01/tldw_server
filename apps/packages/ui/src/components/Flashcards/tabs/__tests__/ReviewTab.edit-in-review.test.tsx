import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ReviewTab } from "../ReviewTab"
import { clearSetting } from "@/services/settings/registry"
import {
  FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING,
  FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING
} from "@/services/settings/ui-settings"
import {
  useDecksQuery,
  useCramQueueQuery,
  useReviewQuery,
  useReviewFlashcardMutation,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardShortcuts,
  useDebouncedFormField,
  useDueCountsQuery,
  useDeckDueCountsQuery,
  useReviewAnalyticsSummaryQuery,
  useHasCardsQuery,
  useNextDueQuery
} from "../../hooks"

const { trackShortcutHintTelemetryMock } = vi.hoisted(() => ({
  trackShortcutHintTelemetryMock: vi.fn().mockResolvedValue(undefined)
}))

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
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
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
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    loading: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  })
}))

vi.mock("@/utils/flashcards-shortcut-hint-telemetry", () => ({
  trackFlashcardsShortcutHintTelemetry: trackShortcutHintTelemetryMock
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    speak: vi.fn(),
    cancel: vi.fn(),
    isSpeaking: false
  })
}))

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => ({
    supported: false,
    isListening: false,
    transcript: "",
    start: vi.fn(),
    stop: vi.fn(),
    resetTranscript: vi.fn()
  })
}))

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCramQueueQuery: vi.fn(),
  useReviewQuery: vi.fn(),
  useReviewFlashcardMutation: vi.fn(),
  useEndFlashcardReviewSessionMutation: vi.fn(),
  useRecentFlashcardReviewSessionsQuery: vi.fn(),
  useGlobalFlashcardTagSuggestionsQuery: vi.fn(),
  useFlashcardAssistantQuery: vi.fn(),
  useFlashcardAssistantRespondMutation: vi.fn(),
  useUpdateFlashcardMutation: vi.fn(),
  useResetFlashcardSchedulingMutation: vi.fn(),
  useDeleteFlashcardMutation: vi.fn(),
  useFlashcardShortcuts: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  useDueCountsQuery: vi.fn(),
  useDeckDueCountsQuery: vi.fn(),
  useReviewAnalyticsSummaryQuery: vi.fn(),
  useHasCardsQuery: vi.fn(),
  useNextDueQuery: vi.fn()
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

describe("ReviewTab edit-in-review workflow", () => {
  beforeEach(async () => {
    vi.clearAllMocks()
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)
    await clearSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "review-card-1",
        deck_id: 1,
        front: "Original question",
        back: "Original answer",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 2,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 3,
        model_type: "basic",
        reverse: false,
        source_ref_type: "note",
        source_ref_id: "88"
      }
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn()
    } as any)
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({
      data: null,
      isLoading: false,
      isError: false
    } as any)
    vi.mocked(useFlashcardAssistantRespondMutation).mockReturnValue({
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
    vi.mocked(useDebouncedFormField).mockReturnValue(undefined as any)
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
  })

  it("opens edit drawer from the review card and returns to review on cancel", async () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.getByText("Note #88")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Edit card (E)" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-shortcut-chips-question")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Show answer (Space)" })
    ).toBeInTheDocument()
    const hintToggle = screen.getByTestId("flashcards-review-shortcut-hints-toggle")
    expect(hintToggle).toHaveTextContent("Compact hints")
    await waitFor(() => {
      expect(trackShortcutHintTelemetryMock).toHaveBeenCalledWith({
        type: "flashcards_shortcut_hints_exposed",
        surface: "review",
        density: "expanded"
      })
    })
    fireEvent.click(hintToggle)
    await waitFor(() => {
      expect(screen.getByText("Space / E")).toBeInTheDocument()
    })
    expect(screen.getByTestId("flashcards-review-shortcut-hints-toggle")).toHaveTextContent(
      "Hide hints"
    )
    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    expect(screen.getByTestId("flashcards-review-shortcut-chips-answer")).toBeInTheDocument()
    expect(screen.getByText("1-4 / E / Ctrl+Z")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("flashcards-review-edit-card"))
    expect(screen.getByText("Edit Flashcard")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    const discardButton = screen.queryByRole("button", { name: "Discard" })
    if (discardButton) {
      fireEvent.click(discardButton)
    }
    await waitFor(() => {
      expect(screen.getByText("Edit Flashcard")).not.toBeVisible()
    })
    expect(screen.getByTestId("flashcards-review-edit-card")).toBeInTheDocument()
  }, 15000)

  it("can hide shortcut hints and records dismissal telemetry", async () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    const toggle = screen.getByTestId("flashcards-review-shortcut-hints-toggle")
    fireEvent.click(toggle)
    await waitFor(() => {
      expect(screen.getByText("Space / E")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("flashcards-review-shortcut-hints-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("flashcards-review-shortcut-hints-toggle")).toHaveTextContent(
        "Show hints"
      )
    })

    expect(screen.queryByText("Space / E")).not.toBeInTheDocument()
    expect(screen.queryByText("Space Flip")).not.toBeInTheDocument()
    expect(trackShortcutHintTelemetryMock).toHaveBeenCalledWith({
      type: "flashcards_shortcut_hints_dismissed",
      surface: "review",
      from_density: "compact"
    })
  }, 15000)

  it("passes an edit shortcut callback when a review card is active", () => {
    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={1}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(useFlashcardShortcuts).toHaveBeenCalled()
    const lastCall = vi.mocked(useFlashcardShortcuts).mock.calls.at(-1)
    expect(lastCall?.[0]?.onEdit).toEqual(expect.any(Function))
  }, 15000)
})
