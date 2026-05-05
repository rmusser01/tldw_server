import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab } from "../ManageTab"
import type { Flashcard } from "@/services/flashcards"
import {
  useCardsKeyboardNav,
  useDecksQuery,
  useDeleteFlashcardMutation,
  useFlashcardDocumentQuery,
  useManageQuery,
  useResetFlashcardSchedulingMutation,
  useTagSuggestionsQuery,
  useUpdateDeckMutation,
  useUpdateFlashcardsBulkMutation,
  useUpdateFlashcardMutation
} from "../../hooks"

const { trackShortcutHintTelemetryMock } = vi.hoisted(() => ({
  trackShortcutHintTelemetryMock: vi.fn().mockResolvedValue(undefined)
}))
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
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/utils/flashcards-shortcut-hint-telemetry", () => ({
  trackFlashcardsShortcutHintTelemetry: trackShortcutHintTelemetryMock
}))

vi.mock("@/utils/chunk-processing", () => ({
  processInChunks: vi.fn(async <T,>(items: T[], worker: (chunk: T[]) => Promise<void>) => {
    await worker(items)
  })
}))

vi.mock("@/utils/flashcards-error-recovery-telemetry", () => ({
  trackFlashcardsErrorRecoveryTelemetry: trackErrorRecoveryTelemetryMock
}))

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>("@tanstack/react-query")
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: vi.fn()
    })
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

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn().mockResolvedValue(true)
}))

vi.mock("../../hooks", () => ({
  DOCUMENT_VIEW_SUPPORTED_SORTS: ["due", "created"],
  getFlashcardDocumentQueryKey: vi.fn(() => ["flashcards:document", 1]),
  useDecksQuery: vi.fn(),
  useManageQuery: vi.fn(),
  useFlashcardDocumentQuery: vi.fn(),
  useTagSuggestionsQuery: vi.fn(),
  useUpdateDeckMutation: vi.fn(),
  useUpdateFlashcardMutation: vi.fn(),
  useUpdateFlashcardsBulkMutation: vi.fn(),
  useResetFlashcardSchedulingMutation: vi.fn(),
  useDeleteFlashcardMutation: vi.fn(),
  useCardsKeyboardNav: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  getManageServerOrderBy: vi.fn(() => "due_at")
}))

vi.mock("../../components", () => ({
  FlashcardMarkdownSnippet: ({ content }: { content: string }) => <div>{content}</div>,
  MarkdownWithBoundary: ({ content }: { content: string }) => <div>{content}</div>,
  FlashcardActionsMenu: ({ onEdit }: { onEdit: () => void }) => (
    <button onClick={onEdit}>Action Edit</button>
  ),
  FlashcardEditDrawer: () => null,
  FlashcardCreateDrawer: () => null
}))

vi.mock("@/services/flashcards", () => ({
  getFlashcard: vi.fn(),
  updateFlashcard: vi.fn(),
  createFlashcard: vi.fn(),
  deleteFlashcard: vi.fn(),
  listFlashcards: vi.fn()
}))

vi.mock("../../utils/error-taxonomy", () => ({
  formatFlashcardsUiErrorMessage: vi.fn(() => "Action failed"),
  mapFlashcardsUiError: vi.fn(() => ({
    code: "FLASHCARDS_UNKNOWN",
    message: "Action failed",
    actionLabel: "Retry",
    rawMessage: "Action failed"
  }))
}))

vi.mock("../../hooks/useFlashcardsShortcutHintDensity", () => ({
  useFlashcardsShortcutHintDensity: () => ["expanded", vi.fn()]
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

const sampleCard: Flashcard = {
  uuid: "card-document-1",
  deck_id: 1,
  front: "Front prompt",
  back: "Back answer",
  notes: "Doc note",
  extra: null,
  is_cloze: false,
  tags: ["biology"],
  ef: 2.6,
  interval_days: 5,
  repetitions: 3,
  lapses: 1,
  due_at: null,
  last_reviewed_at: null,
  queue_state: "learning",
  step_index: 1,
  suspended_reason: null,
  last_modified: null,
  deleted: false,
  client_id: "test",
  version: 4,
  model_type: "basic",
  reverse: false
}

describe("ManageTab document mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Deck 1",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      isLoading: false
    } as any)

    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [sampleCard],
        count: 1,
        total: 1
      },
      isFetching: false
    } as any)

    vi.mocked(useFlashcardDocumentQuery).mockReturnValue({
      items: [sampleCard],
      isFetching: false,
      isLoading: false,
      isTruncated: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      supportedSorts: ["due", "created"],
      data: {
        pages: [
          {
            items: [sampleCard],
            isTruncated: false,
            total: 1
          }
        ]
      }
    } as any)

    vi.mocked(useTagSuggestionsQuery).mockReturnValue({
      data: ["biology"],
      isLoading: false
    } as any)

    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useUpdateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useUpdateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({
        results: []
      }),
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
    vi.mocked(useCardsKeyboardNav).mockImplementation(() => undefined)
  })

  it("renders a document presentation mode and hides unsupported sorts", async () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-density-toggle-document"))

    expect(screen.getByTestId("flashcards-document-view")).toBeInTheDocument()

    const sortSelect = screen.getByTestId("flashcards-manage-sort-select")
    const selector = sortSelect.querySelector(".ant-select") || sortSelect.firstElementChild
    expect(selector).not.toBeNull()
    fireEvent.mouseDown(selector as Element)

    await waitFor(() => {
      expect(screen.queryByText("Sort: Ease factor")).not.toBeInTheDocument()
    })
    expect(screen.queryByText("Sort: Last reviewed")).not.toBeInTheDocument()
  })

  it("shows a truncation banner and disables select-all-across when document results are capped", async () => {
    vi.mocked(useFlashcardDocumentQuery).mockReturnValue({
      items: [sampleCard],
      isFetching: false,
      isLoading: false,
      isTruncated: true,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      supportedSorts: ["due", "created"],
      data: {
        pages: [
          {
            items: [sampleCard],
            isTruncated: true,
            total: 10
          }
        ]
      }
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-density-toggle-document"))
    fireEvent.click(screen.getByTestId(`flashcards-document-row-select-${sampleCard.uuid}`))

    expect(screen.getByTestId("flashcards-document-truncation-banner")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-select-all-across")).toBeDisabled()
  })

  it("shows queue state badges in expanded and document presentations only", async () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-density-toggle"))
    expect(screen.getByTestId("flashcards-manage-queue-state-card-document-1")).toHaveTextContent(
      "Learning"
    )

    fireEvent.click(screen.getByTestId("flashcards-density-toggle-document"))
    expect(
      await screen.findByTestId("flashcards-document-row-queue-state-card-document-1")
    ).toHaveTextContent("Learning")
  })
})
