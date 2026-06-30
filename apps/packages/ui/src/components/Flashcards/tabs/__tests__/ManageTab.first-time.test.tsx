import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab } from "../ManageTab"
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
  FlashcardActionsMenu: () => null,
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

describe("ManageTab first-time state", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
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
        items: [],
        count: 0,
        total: 0
      },
      isFetching: false
    } as any)
    vi.mocked(useFlashcardDocumentQuery).mockReturnValue({
      items: [],
      isFetching: false,
      isLoading: false,
      isTruncated: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      supportedSorts: ["due", "created"],
      data: {
        pages: []
      }
    } as any)
    vi.mocked(useTagSuggestionsQuery).mockReturnValue({
      data: [],
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

  it("suppresses expert manage chrome when there are no cards and no active filters", () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    expect(screen.getByText("No flashcards yet")).toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-shortcut-chips")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-search")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-sort-select")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-density-toggle")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-selection-summary")).not.toBeInTheDocument()
  })

  it("keeps manage filters visible when an empty result comes from an active filter", () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
        initialDeckId={1}
      />
    )

    expect(screen.getByText("No cards match your filters")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-search")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-sort-select")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-density-toggle")).toBeInTheDocument()
  })
})
