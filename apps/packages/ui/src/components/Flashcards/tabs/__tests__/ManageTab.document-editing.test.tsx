import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab } from "../ManageTab"
import type { Flashcard, FlashcardBulkUpdateResponse } from "@/services/flashcards"
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
const getFlashcardMock = vi.fn()
const bulkUpdateMock = vi.fn<(...args: any[]) => Promise<FlashcardBulkUpdateResponse>>()

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
      invalidateQueries: vi.fn().mockResolvedValue(undefined),
      setQueryData: vi.fn()
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
  getFlashcard: (...args: any[]) => getFlashcardMock(...args),
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
  uuid: "row-1",
  deck_id: 1,
  front: "Original front",
  back: "Original back",
  notes: "Original note",
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

const renderManageDocument = () => {
  render(
    <ManageTab
      onNavigateToImport={() => {}}
      onReviewCard={() => {}}
      isActive
    />
  )
  fireEvent.click(screen.getByTestId("flashcards-density-toggle-document"))
}

describe("ManageTab document editing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    bulkUpdateMock.mockReset()
    getFlashcardMock.mockReset()

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
      mutateAsync: bulkUpdateMock,
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

  it("offers reload and reapply actions when a row save conflicts", async () => {
    bulkUpdateMock.mockResolvedValueOnce({
      results: [
        {
          uuid: "row-1",
          status: "conflict",
          error: {
            code: "conflict",
            message: "Version changed elsewhere"
          }
        }
      ]
    })

    renderManageDocument()

    fireEvent.click(screen.getByTestId("flashcards-document-row-front-display-row-1"))
    const frontInput = await screen.findByTestId("flashcards-document-row-front-input-row-1")
    fireEvent.change(frontInput, { target: { value: "Edited front" } })
    fireEvent.blur(frontInput)

    expect(
      await screen.findByTestId("flashcards-document-row-conflict-row-1")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /reload row/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /reapply my edit/i })).toBeInTheDocument()
  })

  it("restores a previous row snapshot when undo is triggered after inline save", async () => {
    bulkUpdateMock
      .mockResolvedValueOnce({
        results: [
          {
            uuid: "row-1",
            status: "updated",
            flashcard: {
              ...sampleCard,
              version: 5,
              front: "Updated front"
            }
          }
        ]
      })
      .mockResolvedValueOnce({
        results: [
          {
            uuid: "row-1",
            status: "updated",
            flashcard: {
              ...sampleCard,
              version: 6
            }
          }
        ]
      })

    renderManageDocument()

    fireEvent.click(screen.getByTestId("flashcards-document-row-front-display-row-1"))
    const frontInput = await screen.findByTestId("flashcards-document-row-front-input-row-1")
    fireEvent.change(frontInput, { target: { value: "Updated front" } })
    fireEvent.blur(frontInput)

    const undoButton = await screen.findByRole("button", { name: /undo/i })
    fireEvent.click(undoButton)

    await waitFor(() => {
      expect(bulkUpdateMock).toHaveBeenLastCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            uuid: "row-1",
            front: "Original front"
          })
        ])
      )
    })
  })

  it("supports keyboard save in document mode", async () => {
    bulkUpdateMock.mockResolvedValueOnce({
      results: [
        {
          uuid: "row-1",
          status: "updated",
          flashcard: {
            ...sampleCard,
            version: 5,
            front: "Keyboard saved"
          }
        }
      ]
    })

    renderManageDocument()

    fireEvent.click(screen.getByTestId("flashcards-document-row-front-display-row-1"))
    const frontInput = await screen.findByTestId("flashcards-document-row-front-input-row-1")
    fireEvent.change(frontInput, { target: { value: "Keyboard saved" } })
    fireEvent.keyDown(frontInput, { key: "Enter", metaKey: true })

    await waitFor(() => {
      expect(bulkUpdateMock).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            uuid: "row-1",
            front: "Keyboard saved"
          })
        ])
      )
    })
  })
})
