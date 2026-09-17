import { fireEvent, render, screen, within } from "@testing-library/react"
import type React from "react"
import { createInstance, type i18n } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"
import ICU from "@/i18n/icu-format"
import option from "@/assets/locale/en/option.json"
import type { Flashcard } from "@/services/flashcards"

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

const localization = vi.hoisted(() => ({ real: false }))
vi.mock("react-i18next", async importOriginal => {
  const actual = await importOriginal<typeof import("react-i18next")>()
  return { ...actual,
  // Existing empty-state tests inspect conditional fallback copy. Count tests
  // exercise the actual ICU plugin and production English resources instead.
  useTranslation: (...args: Parameters<typeof actual.useTranslation>) => localization.real
    ? actual.useTranslation(...args)
    : ({
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
  }) }
})

vi.mock("@/utils/flashcards-shortcut-hint-telemetry", () => ({
  trackFlashcardsShortcutHintTelemetry: trackShortcutHintTelemetryMock
}))

vi.mock("@/utils/chunk-processing", () => ({
  processInChunks: vi.fn(
    async <T,>(
      items: T[],
      chunkSizeOrWorker: number | ((chunk: T[]) => Promise<void>),
      maybeWorker?: (chunk: T[]) => Promise<void>
    ) => {
      const worker =
        typeof chunkSizeOrWorker === "function" ? chunkSizeOrWorker : maybeWorker
      if (worker) await worker(items)
    }
  )
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

const manageElement = (props: Partial<React.ComponentProps<typeof ManageTab>> = {}) => (
    <ManageTab
      onNavigateToImport={() => {}}
      onReviewCard={() => {}}
      isActive
      {...props}
    />
  )
const renderManageTab = (props: Partial<React.ComponentProps<typeof ManageTab>> = {}, instance?: i18n) =>
  render(instance ? <I18nextProvider i18n={instance}>{manageElement(props)}</I18nextProvider> : manageElement(props))

const countCard: Flashcard = {
  uuid: "count-card", deck_id: 1, front: "Counted question", back: "Counted answer", notes: null, extra: null,
  is_cloze: false, tags: [], ef: 2.5, interval_days: 0, repetitions: 0, lapses: 0, queue_state: "new",
  due_at: null, last_reviewed_at: null, last_modified: null, deleted: false, client_id: "test", version: 1,
  model_type: "basic", reverse: false
}
const setCount = (count: number) => vi.mocked(useManageQuery).mockReturnValue({
  data: { items: Array.from({ length: count }, (_, index) => ({ ...countCard, uuid: `count-card-${index}` })), count, total: count },
  isFetching: false
} as ReturnType<typeof useManageQuery>)
const realEnglish = async () => {
  localization.real = true
  const instance = createInstance().use(ICU)
  await instance.init({ lng: "en", resources: { en: { option } }, interpolation: { escapeValue: false } })
  return instance
}

describe("ManageTab count and no-card empty state", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localization.real = false
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          workspace_id: "workspace-a",
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

  it("shows create, import, generate, and primary filters before expert chrome for a true first run", () => {
    renderManageTab()

    expect(screen.getByText("No flashcards yet")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-empty-create-cta")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-empty-import-cta")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-empty-generate-cta")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-search")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-deck-select")).toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-density-toggle")).not.toBeInTheDocument()
  })

  it("routes import and generate empty-state actions through distinct callbacks", () => {
    const onNavigateToImport = vi.fn()
    const onNavigateToGenerate = vi.fn()

    renderManageTab({
      onNavigateToImport,
      onNavigateToGenerate
    })

    fireEvent.click(screen.getByTestId("flashcards-manage-empty-import-cta"))
    fireEvent.click(screen.getByTestId("flashcards-manage-empty-generate-cta"))

    expect(onNavigateToImport).toHaveBeenCalledTimes(1)
    expect(onNavigateToGenerate).toHaveBeenCalledTimes(1)
  })

  it("names the floating create action without requiring tooltip focus", () => {
    renderManageTab()
    expect(screen.getByTestId("flashcards-fab-create")).toHaveAccessibleName("Create card")
  })

  it("treats workspace deck visibility as an active empty-result filter", () => {
    renderManageTab({
      initialShowWorkspaceDecks: true
    })

    expect(screen.getByText("No cards match your filters")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-search")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-show-workspace-decks")).toBeChecked()
    expect(screen.getByTestId("flashcards-density-toggle")).toBeInTheDocument()
  })

  it.each([0, 1, 2])("localizes the unselected summary for %i cards through actual ICU", async count => {
    const instance = await realEnglish()
    setCount(count)
    renderManageTab({ initialShowWorkspaceDecks: true }, instance)
    expect(within(screen.getByTestId("flashcards-manage-selection-summary")).getByText(new RegExp(`^${count} ${count === 1 ? "card" : "cards"}$`, "i"))).toBeVisible()
  })

  it("updates the localized count and keeps page selection separate from the total", async () => {
    const instance = await realEnglish()
    setCount(1)
    const view = renderManageTab({}, instance)
    const summary = screen.getByTestId("flashcards-manage-selection-summary")
    expect(within(summary).getByText("1 card", { exact: true })).toBeVisible()
    setCount(2)
    view.rerender(<I18nextProvider i18n={instance}>{manageElement()}</I18nextProvider>)
    expect(within(summary).getByText("2 cards", { exact: true })).toBeVisible()
    const selectAll = within(summary).getByRole("checkbox", { name: "Select all on page" })
    fireEvent.click(selectAll)
    expect(selectAll).toBeChecked()
    expect(within(summary).getByText("selected on this page", { exact: true })).toBeVisible()
    expect(within(summary).queryByText("2 cards", { exact: true })).not.toBeInTheDocument()
    fireEvent.click(selectAll)
    expect(selectAll).not.toBeChecked()
    expect(within(summary).getByText("2 cards", { exact: true })).toBeVisible()
  })

  it("keeps a genuine first-run summary hidden with production localization", async () => {
    const instance = await realEnglish()
    renderManageTab({}, instance)
    expect(screen.queryByTestId("flashcards-manage-selection-summary")).not.toBeInTheDocument()
  })
})
