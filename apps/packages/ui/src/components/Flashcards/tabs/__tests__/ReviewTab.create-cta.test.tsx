import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { ReviewTab } from "../ReviewTab"
import { clearSetting } from "@/services/settings/registry"
import {
  FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING,
  FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING
} from "@/services/settings/ui-settings"
import { FLASHCARDS_HELP_LINKS, FLASHCARDS_LAYOUT_GUARDRAILS } from "../../constants"
import {
  useDecksQuery,
  useCramQueueQuery,
  useReviewQuery,
  useReviewFlashcardMutation,
  useEndFlashcardReviewSessionMutation,
  useRecentFlashcardReviewSessionsQuery,
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

const { trackErrorRecoveryTelemetryMock } = vi.hoisted(() => ({
  trackErrorRecoveryTelemetryMock: vi.fn().mockResolvedValue(undefined)
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

vi.mock("@/utils/flashcards-error-recovery-telemetry", () => ({
  trackFlashcardsErrorRecoveryTelemetry: trackErrorRecoveryTelemetryMock
}))

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
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

describe("ReviewTab create CTA visibility", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
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
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(async () => {
    vi.clearAllMocks()
    trackErrorRecoveryTelemetryMock.mockClear()
    Object.values(messageSpies).forEach((spy) => spy.mockReset())
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)
    await clearSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
    vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: null,
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
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
      data: { due: 0, new: 0, learning: 0, total: 0 },
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({
      data: {}
    } as any)
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({
      data: null,
      isLoading: false
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({
      data: false
    } as any)
    vi.mocked(useNextDueQuery).mockReturnValue({
      data: null
    } as any)
    vi.mocked(useEndFlashcardReviewSessionMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
  })

  it("shows the top-bar create action when there is no active review card", () => {
    const onNavigateToCreate = vi.fn()

    render(
      <ReviewTab
        onNavigateToCreate={onNavigateToCreate}
        onNavigateToImport={() => {}}
        reviewDeckId={undefined}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    const createButton = screen.getByTestId("flashcards-review-create-cta")
    expect(createButton).toBeInTheDocument()
    const topbarPrimaryButtons = screen
      .getByTestId("flashcards-review-topbar")
      .querySelectorAll(".ant-btn-primary")
    expect(topbarPrimaryButtons.length).toBeLessThanOrEqual(
      FLASHCARDS_LAYOUT_GUARDRAILS.review.maxTopbarPrimaryCtas.empty
    )

    fireEvent.click(createButton)
    expect(onNavigateToCreate).toHaveBeenCalledTimes(1)
  })

  it("shows first-run onboarding guidance with create/import/generate actions", () => {
    const onNavigateToCreate = vi.fn()
    const onNavigateToImport = vi.fn()

    render(
      <ReviewTab
        onNavigateToCreate={onNavigateToCreate}
        onNavigateToImport={onNavigateToImport}
        reviewDeckId={undefined}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.getByTestId("flashcards-review-onboarding-guide")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-onboarding-doc-link")).toHaveAttribute(
      "href",
      FLASHCARDS_HELP_LINKS.ratings
    )

    fireEvent.click(screen.getByTestId("flashcards-review-empty-create-cta"))
    fireEvent.click(screen.getByTestId("flashcards-review-empty-import-cta"))
    fireEvent.click(screen.getByTestId("flashcards-review-empty-generate-cta"))

    expect(onNavigateToCreate).toHaveBeenCalledTimes(1)
    expect(onNavigateToImport).toHaveBeenCalledTimes(2)
  })

  it("persists onboarding dismissal and supports reopening from help entry point", async () => {
    const { unmount } = render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={undefined}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-onboarding-dismiss"))
    await waitFor(() => {
      expect(screen.queryByTestId("flashcards-review-onboarding-guide")).not.toBeInTheDocument()
    })
    expect(screen.getByTestId("flashcards-review-onboarding-reopen")).toBeInTheDocument()

    unmount()

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={undefined}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-review-onboarding-reopen")).toBeInTheDocument()
    })
    expect(screen.queryByTestId("flashcards-review-onboarding-guide")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("flashcards-review-onboarding-reopen"))
    await waitFor(() => {
      expect(screen.getByTestId("flashcards-review-onboarding-guide")).toBeInTheDocument()
    })
  })

  it("hides the top-bar create action during active review while keeping controls accessible", () => {
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "active-card-1",
        deck_id: 11,
        front: "Question",
        back: "Answer",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 1,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 1,
        model_type: "basic",
        reverse: false
      }
    } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 }
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({
      data: true
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={11}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.queryByTestId("flashcards-review-create-cta")).not.toBeInTheDocument()
    expect(screen.getByRole("combobox")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show answer (Space)" })).toBeInTheDocument()
    const topbarPrimaryButtons = screen
      .getByTestId("flashcards-review-topbar")
      .querySelectorAll(".ant-btn-primary")
    expect(topbarPrimaryButtons).toHaveLength(
      FLASHCARDS_LAYOUT_GUARDRAILS.review.maxTopbarPrimaryCtas.active
    )
  })

  it("shows due counts in deck selector labels when available", () => {
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 11, name: "Biology" }],
      isLoading: false
    } as any)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({
      data: {
        11: { due: 7, new: 3, learning: 2, total: 12 }
      }
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={undefined}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.mouseDown(screen.getByRole("combobox"))
    expect(screen.getByText("Biology (7 due)")).toBeInTheDocument()
  })

  it("force-shows the addressed workspace deck without widening the visible deck list", () => {
    vi.mocked(useDecksQuery).mockImplementation((params: any) => ({
      data: params?.includeWorkspaceItems
        ? [
            {
              id: 9,
              name: "Workspace Biology",
              workspace_id: "workspace-77"
            }
          ]
        : [],
      isLoading: false
    } as any))

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={9}
        onReviewDeckChange={() => {}}
        isActive
        forceShowWorkspaceItems
      />
    )

    expect(screen.getByText("Workspace Biology")).toBeInTheDocument()
    expect(vi.mocked(useHasCardsQuery)).toHaveBeenLastCalledWith(
      expect.objectContaining({
        includeWorkspaceItems: true
      })
    )
  })

  it("matches baseline snapshot for active review state", () => {
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "active-card-snapshot",
        deck_id: 11,
        front: "Question snapshot",
        back: "Answer snapshot",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 1,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 1,
        model_type: "basic",
        reverse: false
      }
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 }
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={11}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.getByTestId("flashcards-review-topbar")).toMatchSnapshot()
    expect(screen.getByTestId("flashcards-review-active-card")).toMatchSnapshot()
  })

  it("matches baseline snapshot for caught-up completion state", () => {
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 0, new: 0, learning: 0, total: 0 }
    } as any)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: null } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={11}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    expect(screen.getByTestId("flashcards-review-topbar")).toMatchSnapshot()
    expect(screen.getByTestId("flashcards-review-empty-card")).toMatchSnapshot()
  })

  it("shows explicit version-conflict guidance when review submission fails with conflict", async () => {
    const refetchReviewMock = vi.fn().mockResolvedValue(undefined)
    const refetchCountsMock = vi.fn().mockResolvedValue(undefined)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "active-card-error",
        deck_id: 11,
        front: "Question error",
        back: "Answer error",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 1,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 1,
        model_type: "basic",
        reverse: false
      },
      refetch: refetchReviewMock
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 },
      refetch: refetchCountsMock
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn().mockRejectedValue({
        status: 409,
        message: "Version mismatch: expected 4, got 5"
      }),
      isPending: false
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={11}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

    await waitFor(() => {
      expect(messageSpies.error).toHaveBeenCalled()
    })
    expect(screen.getByTestId("flashcards-review-retry-alert")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("flashcards-review-reload-button"))
    await waitFor(() => {
      expect(refetchReviewMock).toHaveBeenCalledTimes(1)
      expect(refetchCountsMock).toHaveBeenCalledTimes(1)
      expect(messageSpies.info).toHaveBeenCalled()
    })
    const latestError = String(messageSpies.error.mock.calls.at(-1)?.[0] || "")
    expect(latestError).toContain("FLASHCARDS_VERSION_CONFLICT")
    expect(latestError).toContain("Refresh")
    expect(trackErrorRecoveryTelemetryMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "flashcards_mutation_failed",
        surface: "review",
        error_code: "FLASHCARDS_VERSION_CONFLICT"
      })
    )
    expect(trackErrorRecoveryTelemetryMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "flashcards_recovered_by_reload",
        surface: "review",
        error_code: "FLASHCARDS_VERSION_CONFLICT"
      })
    )
  })

  it("retries failed review submissions with preserved answer timing", async () => {
    const mutateAsync = vi
      .fn()
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce({
        uuid: "active-card-retry",
        ef: 2.6,
        interval_days: 2,
        repetitions: 2,
        lapses: 0,
        due_at: "2026-02-20T09:30:00.000Z",
        version: 2
      })
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "active-card-retry",
        deck_id: 11,
        front: "Retry question",
        back: "Retry answer",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 1,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 1,
        model_type: "basic",
        reverse: false
      },
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 },
      refetch: vi.fn().mockResolvedValue(undefined)
    } as any)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    render(
      <ReviewTab
        onNavigateToCreate={() => {}}
        onNavigateToImport={() => {}}
        reviewDeckId={11}
        onReviewDeckChange={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))
    await waitFor(() => {
      expect(screen.getByTestId("flashcards-review-retry-alert")).toBeInTheDocument()
    })

    const firstAttempt = mutateAsync.mock.calls[0][0]
    expect(typeof firstAttempt.answerTimeMs).toBe("number")

    fireEvent.click(screen.getByTestId("flashcards-review-retry-button"))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(2)
    })
    const secondAttempt = mutateAsync.mock.calls[1][0]
    expect(secondAttempt.answerTimeMs).toBe(firstAttempt.answerTimeMs)
    await waitFor(() => {
      expect(screen.queryByTestId("flashcards-review-retry-alert")).not.toBeInTheDocument()
    })
    expect(trackErrorRecoveryTelemetryMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "flashcards_retry_requested",
        surface: "review",
        error_code: "FLASHCARDS_NETWORK"
      })
    )
    expect(trackErrorRecoveryTelemetryMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "flashcards_retry_succeeded",
        surface: "review",
        error_code: "FLASHCARDS_NETWORK"
      })
    )
  }, 15000)
})
