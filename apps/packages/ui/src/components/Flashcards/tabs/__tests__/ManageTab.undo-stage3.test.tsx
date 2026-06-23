import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ManageTab } from "../ManageTab"
import { clearSetting } from "@/services/settings/registry"
import { FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING } from "@/services/settings/ui-settings"
import type { Flashcard } from "@/services/flashcards"
import {
  useUpdateFlashcardsBulkMutation,
  useDecksQuery,
  useFlashcardDocumentQuery,
  useGlobalFlashcardTagSuggestionsQuery,
  useManageQuery,
  useTagSuggestionsQuery,
  useUpdateDeckMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useCardsKeyboardNav,
  useDebouncedFormField,
  getManageServerOrderBy
} from "../../hooks"
import { FLASHCARDS_DRAWER_WIDTH_PX } from "../../constants"
import { FLASHCARDS_LAYOUT_GUARDRAILS } from "../../constants/layout-guardrails"
import { getFlashcard, listFlashcards, updateFlashcard } from "@/services/flashcards"

const showUndoNotificationMock = vi.fn()
const updateMutationMock = vi.fn()
const { trackErrorRecoveryTelemetryMock } = vi.hoisted(() => ({
  trackErrorRecoveryTelemetryMock: vi.fn().mockResolvedValue(undefined)
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
  useAntdMessage: () => messageSpies
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: showUndoNotificationMock
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn().mockResolvedValue(true)
}))

vi.mock("../../hooks", () => ({
  DOCUMENT_VIEW_SUPPORTED_SORTS: ["due", "created"],
  getFlashcardDocumentQueryKey: vi.fn(() => ["flashcards:document", 1]),
  useDecksQuery: vi.fn(),
  useFlashcardDocumentQuery: vi.fn(),
  useGlobalFlashcardTagSuggestionsQuery: vi.fn(),
  useManageQuery: vi.fn(),
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

vi.mock("../../components", async () => {
  const actual = await vi.importActual<typeof import("../../components")>("../../components")
  return {
    ...actual,
    FlashcardActionsMenu: ({
      onEdit,
      onMove
    }: {
      onEdit: () => void
      onMove: () => void
    }) => (
      <div>
        <button onClick={onEdit}>Action Edit</button>
        <button onClick={onMove}>Action Move</button>
      </div>
    ),
    FlashcardCreateDrawer: () => null
  }
})

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>("@/services/flashcards")
  return {
    ...actual,
    getFlashcard: vi.fn(),
    updateFlashcard: vi.fn(),
    createFlashcard: vi.fn(),
    deleteFlashcard: vi.fn(),
    listFlashcards: vi.fn()
  }
})

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
  uuid: "card-undo-1",
  deck_id: 1,
  front: "Front prompt",
  back: "Back answer",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: ["biology"],
  ef: 2.6,
  interval_days: 5,
  repetitions: 3,
  lapses: 1,
  queue_state: "review",
  due_at: null,
  last_reviewed_at: null,
  last_modified: null,
  deleted: false,
  client_id: "test",
  version: 4,
  model_type: "basic",
  reverse: false
}

const buildSampleCards = (count: number): Flashcard[] =>
  Array.from({ length: count }, (_item, index) => ({
    ...sampleCard,
    uuid: `card-undo-${index + 1}`,
    front: `Front prompt ${index + 1}`,
    back: `Back answer ${index + 1}`
  }))

describe("ManageTab stage3 undo controls", () => {
  beforeEach(async () => {
    vi.clearAllMocks()
    trackErrorRecoveryTelemetryMock.mockClear()
    updateMutationMock.mockReset()
    vi.mocked(getFlashcard).mockReset()
    vi.mocked(listFlashcards).mockReset()
    vi.mocked(updateFlashcard).mockReset()
    Object.values(messageSpies).forEach((spy) => spy.mockReset())
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Deck 1",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        },
        {
          id: 2,
          name: "Deck 2",
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
      data: ["biology", "chemistry"],
      isLoading: false
    } as any)
    vi.mocked(useGlobalFlashcardTagSuggestionsQuery).mockReturnValue({
      data: { items: [] },
      isLoading: false,
      isFetching: false,
      isError: false
    } as any)
    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({
      mutateAsync: updateMutationMock,
      isPending: false
    } as any)
    vi.mocked(useUpdateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useUpdateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({ results: [] }),
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
    vi.mocked(useDebouncedFormField).mockReturnValue(undefined as any)
    vi.mocked(getManageServerOrderBy).mockReturnValue("due_at")
  })

  it("keeps the top bar free of primary CTAs while managing cards", () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    const topbarPrimaryButtons = screen
      .getByTestId("flashcards-manage-topbar")
      .querySelectorAll(".ant-btn-primary")
    expect(topbarPrimaryButtons).toHaveLength(
      FLASHCARDS_LAYOUT_GUARDRAILS.manage.maxTopbarPrimaryCtas.active
    )
  })

  it("matches baseline snapshot for cards view with active selection", () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId(`flashcard-item-${sampleCard.uuid}-select`))
    expect(screen.getByTestId("flashcards-manage-selection-summary")).toMatchSnapshot()
  })

  it("offers undo for single-card edits", async () => {
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({ ...sampleCard })
      .mockResolvedValueOnce({ ...sampleCard, version: 5, front: "Updated front" })
    updateMutationMock.mockResolvedValue(undefined)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByText("Action Edit"))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })

    const undoConfig = showUndoNotificationMock.mock.calls[0][0]
    expect(undoConfig.duration).toBe(30)
    expect(String(undoConfig.description)).toContain("Undo within 30s")

    await undoConfig.onUndo()

    expect(updateMutationMock).toHaveBeenCalledTimes(2)
    const undoCall = updateMutationMock.mock.calls[1][0]
    expect(undoCall.uuid).toBe(sampleCard.uuid)
    expect(undoCall.update.deck_id).toBe(1)
    expect(undoCall.update.front).toBe("Front prompt")
    expect(undoCall.update.expected_version).toBe(5)
  }, 15000)

  it("shows explicit conflict guidance when an edit hits version mismatch", async () => {
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({ ...sampleCard })
      .mockResolvedValueOnce({
        ...sampleCard,
        version: 5,
        front: "Remote updated front"
      })
    updateMutationMock.mockRejectedValueOnce({
      status: 409,
      message: "Version mismatch: expected 4 got 5"
    })

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByText("Action Edit"))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(messageSpies.warning).toHaveBeenCalled()
    })
    expect(getFlashcard).toHaveBeenCalledTimes(2)
    const warningMessage = String(messageSpies.warning.mock.calls.at(-1)?.[0] || "")
    expect(warningMessage).toContain("FLASHCARDS_VERSION_CONFLICT")
    expect(warningMessage).toContain("Reloaded")
    expect(messageSpies.error).not.toHaveBeenCalled()
    expect(trackErrorRecoveryTelemetryMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "flashcards_recovered_by_reload",
        surface: "cards",
        error_code: "FLASHCARDS_VERSION_CONFLICT"
      })
    )
  }, 15000)

  it("offers undo for move operations", async () => {
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({ ...sampleCard, version: 8, deck_id: 1 })
      .mockResolvedValueOnce({ ...sampleCard, version: 9, deck_id: 2 })
    vi.mocked(updateFlashcard).mockResolvedValue(undefined as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByText("Action Move"))
    const moveWrapper = document.querySelector(".ant-drawer-content-wrapper") as HTMLElement | null
    expect(moveWrapper?.style.width).toBe(`${FLASHCARDS_DRAWER_WIDTH_PX}px`)

    const comboboxes = screen.getAllByRole("combobox")
    fireEvent.mouseDown(comboboxes[comboboxes.length - 1])
    fireEvent.click(screen.getByText("Deck 2"))
    fireEvent.click(screen.getByRole("button", { name: "Move" }))

    await waitFor(() => {
      expect(showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })
    expect(updateFlashcard).toHaveBeenCalledTimes(1)
    expect(vi.mocked(updateFlashcard).mock.calls[0][1]).toMatchObject({
      deck_id: 2,
      expected_version: 8
    })

    const undoConfig = showUndoNotificationMock.mock.calls[0][0]
    expect(String(undoConfig.description)).toContain("Undo within 30s")
    await undoConfig.onUndo()

    expect(updateFlashcard).toHaveBeenCalledTimes(2)
    expect(vi.mocked(updateFlashcard).mock.calls[1][1]).toMatchObject({
      deck_id: 1,
      expected_version: 9
    })
  }, 15000)

  it("applies bulk add tag to selected cards", async () => {
    vi.mocked(getFlashcard).mockResolvedValueOnce({
      ...sampleCard,
      version: 11,
      tags: ["biology"]
    })
    vi.mocked(updateFlashcard).mockResolvedValue(undefined as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId(`flashcard-item-${sampleCard.uuid}-select`))
    const bulkActionBar = document.querySelector(".fixed.bottom-4") as HTMLElement
    fireEvent.click(within(bulkActionBar).getByRole("button", { name: "Add tag" }))
    fireEvent.change(screen.getByTestId("flashcards-bulk-tag-input"), {
      target: { value: "chemistry chapter-1" }
    })
    const addTagButtons = screen.getAllByRole("button", { name: "Add tag" })
    fireEvent.click(addTagButtons[addTagButtons.length - 1])

    await waitFor(() => {
      expect(updateFlashcard).toHaveBeenCalledTimes(1)
    })
    expect(vi.mocked(updateFlashcard).mock.calls[0][0]).toBe(sampleCard.uuid)
    expect(vi.mocked(updateFlashcard).mock.calls[0][1]).toMatchObject({
      tags: ["biology", "chemistry", "chapter-1"],
      expected_version: 11
    })
  }, 15000)

  it("keeps failed cards selected and warns when bulk tag updates partially fail", async () => {
    const cards = buildSampleCards(2)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: cards,
        count: cards.length,
        total: cards.length
      },
      isFetching: false
    } as any)
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({
        ...cards[0],
        version: 11,
        tags: ["biology"]
      })
      .mockResolvedValueOnce({
        ...cards[1],
        version: 12,
        tags: ["biology"]
      })
    vi.mocked(updateFlashcard)
      .mockResolvedValueOnce(undefined as any)
      .mockRejectedValueOnce(new Error("save failed"))

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId("flashcard-item-card-undo-1-select"))
    fireEvent.click(screen.getByTestId("flashcard-item-card-undo-2-select"))
    const retryBulkActionBar = document.querySelector(".fixed.bottom-4") as HTMLElement
    fireEvent.click(within(retryBulkActionBar).getByRole("button", { name: "Add tag" }))
    fireEvent.change(screen.getByTestId("flashcards-bulk-tag-input"), {
      target: { value: "chemistry" }
    })
    const addTagButtons = screen.getAllByRole("button", { name: "Add tag" })
    fireEvent.click(addTagButtons[addTagButtons.length - 1])

    await waitFor(() => {
      expect(messageSpies.warning).toHaveBeenCalledWith(
        "Updated 1 card; 1 failed. Failed cards remain selected so you can retry."
      )
    })
    expect(messageSpies.success).not.toHaveBeenCalled()
    expect(screen.getByTestId("flashcard-item-card-undo-1-select")).not.toBeChecked()
    expect(screen.getByTestId("flashcard-item-card-undo-2-select")).toBeChecked()
  }, 15000)

  it("retries failed off-page cards after select-all-across bulk tag partial failure", async () => {
    const cards = buildSampleCards(2)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [cards[0]],
        count: 1,
        total: cards.length
      },
      isFetching: false
    } as any)
    vi.mocked(listFlashcards).mockResolvedValue({
      items: cards,
      total: cards.length,
      page: 1,
      page_size: cards.length
    } as any)
    vi.mocked(getFlashcard).mockImplementation(async (uuid) => {
      const card = cards.find((item) => item.uuid === uuid)
      if (!card) throw new Error("card not found")
      return {
        ...card,
        version: card.uuid === cards[0].uuid ? 11 : 12,
        tags: ["biology"]
      }
    })
    let offPageAttempts = 0
    vi.mocked(updateFlashcard).mockImplementation(async (uuid) => {
      if (uuid === cards[1].uuid) {
        offPageAttempts += 1
        if (offPageAttempts === 1) {
          throw new Error("save failed")
        }
      }
      return undefined as any
    })

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByLabelText("Select all cards on this page"))
    fireEvent.click(screen.getByTestId("flashcards-select-all-across"))
    fireEvent.click(screen.getByRole("button", { name: "Add tag" }))
    let bulkTagInputs = screen.getAllByTestId("flashcards-bulk-tag-input")
    let activeBulkTagInput = bulkTagInputs[bulkTagInputs.length - 1]
    fireEvent.change(activeBulkTagInput, {
      target: { value: "chemistry" }
    })
    const retryAddTagButtons = screen.getAllByRole("button", { name: "Add tag" })
    fireEvent.click(retryAddTagButtons[retryAddTagButtons.length - 1])

    await waitFor(() => {
      expect(messageSpies.warning).toHaveBeenCalledWith(
        "Updated 1 card; 1 failed. Failed cards remain selected so you can retry."
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("flashcard-item-card-undo-1-select")).not.toBeChecked()
    })
    await waitFor(() => {
      expect(screen.getAllByText("failed cards selected for retry").length).toBeGreaterThan(0)
    })

    const retryBulkActionBar = document.querySelector(".fixed.bottom-4") as HTMLElement
    fireEvent.click(within(retryBulkActionBar).getByRole("button", { name: "Add tag" }))
    bulkTagInputs = screen.getAllByTestId("flashcards-bulk-tag-input")
    activeBulkTagInput = bulkTagInputs[bulkTagInputs.length - 1]
    fireEvent.change(activeBulkTagInput, {
      target: { value: "chemistry" }
    })
    fireEvent.keyDown(activeBulkTagInput, { key: "Enter", code: "Enter" })

    await waitFor(() => {
      expect(updateFlashcard).toHaveBeenCalledTimes(3)
    })
    expect(vi.mocked(updateFlashcard).mock.calls[2][0]).toBe(cards[1].uuid)
    expect(listFlashcards).toHaveBeenCalledTimes(1)
    expect(messageSpies.info).not.toHaveBeenCalledWith("No cards selected.")
  }, 15000)

  it("keeps off-page retry selections when a visible card is toggled", async () => {
    const cards = buildSampleCards(2)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [cards[0]],
        count: 1,
        total: cards.length
      },
      isFetching: false
    } as any)
    vi.mocked(listFlashcards).mockResolvedValue({
      items: cards,
      total: cards.length,
      page: 1,
      page_size: cards.length
    } as any)
    vi.mocked(getFlashcard).mockImplementation(async (uuid) => {
      const card = cards.find((item) => item.uuid === uuid)
      if (!card) throw new Error("card not found")
      return {
        ...card,
        version: card.uuid === cards[0].uuid ? 11 : 12,
        tags: ["biology"]
      }
    })
    let offPageAttempts = 0
    vi.mocked(updateFlashcard).mockImplementation(async (uuid) => {
      if (uuid === cards[1].uuid) {
        offPageAttempts += 1
        if (offPageAttempts === 1) {
          throw new Error("save failed")
        }
      }
      return undefined as any
    })

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByLabelText("Select all cards on this page"))
    fireEvent.click(screen.getByTestId("flashcards-select-all-across"))
    fireEvent.click(screen.getByRole("button", { name: "Add tag" }))
    let bulkTagInputs = screen.getAllByTestId("flashcards-bulk-tag-input")
    let activeBulkTagInput = bulkTagInputs[bulkTagInputs.length - 1]
    fireEvent.change(activeBulkTagInput, {
      target: { value: "chemistry" }
    })
    let addTagButtons = screen.getAllByRole("button", { name: "Add tag" })
    fireEvent.click(addTagButtons[addTagButtons.length - 1])

    await waitFor(() => {
      expect(screen.getAllByText("failed cards selected for retry").length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getByTestId("flashcard-item-card-undo-1-select"))

    const retryBulkActionBar = document.querySelector(".fixed.bottom-4") as HTMLElement
    fireEvent.click(within(retryBulkActionBar).getByRole("button", { name: "Add tag" }))
    bulkTagInputs = screen.getAllByTestId("flashcards-bulk-tag-input")
    activeBulkTagInput = bulkTagInputs[bulkTagInputs.length - 1]
    fireEvent.change(activeBulkTagInput, {
      target: { value: "chemistry" }
    })
    addTagButtons = screen.getAllByRole("button", { name: "Add tag" })
    fireEvent.click(addTagButtons[addTagButtons.length - 1])

    await waitFor(() => {
      expect(updateFlashcard).toHaveBeenCalledTimes(4)
    })
    expect(vi.mocked(updateFlashcard).mock.calls.slice(2).map(([uuid]) => uuid)).toEqual([
      cards[0].uuid,
      cards[1].uuid
    ])
    expect(listFlashcards).toHaveBeenCalledTimes(1)
  }, 15000)

  it("clears failed retry selection when workspace scope changes", async () => {
    const cards = buildSampleCards(2)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [cards[0]],
        count: 1,
        total: cards.length
      },
      isFetching: false
    } as any)
    vi.mocked(listFlashcards).mockResolvedValue({
      items: cards,
      total: cards.length,
      page: 1,
      page_size: cards.length
    } as any)
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({
        ...cards[0],
        version: 11,
        tags: ["biology"]
      })
      .mockResolvedValueOnce({
        ...cards[1],
        version: 12,
        tags: ["biology"]
      })
    vi.mocked(updateFlashcard).mockImplementation(async (uuid) => {
      if (uuid === cards[1].uuid) {
        throw new Error("save failed")
      }
      return undefined as any
    })

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByLabelText("Select all cards on this page"))
    fireEvent.click(screen.getByTestId("flashcards-select-all-across"))
    fireEvent.click(screen.getByRole("button", { name: "Add tag" }))
    const bulkTagInput = screen.getByTestId("flashcards-bulk-tag-input")
    fireEvent.change(bulkTagInput, {
      target: { value: "chemistry" }
    })
    const addTagButtons = screen.getAllByRole("button", { name: "Add tag" })
    fireEvent.click(addTagButtons[addTagButtons.length - 1])

    await waitFor(() => {
      expect(screen.getAllByText("failed cards selected for retry").length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getByTestId("flashcards-manage-show-workspace-decks"))

    await waitFor(() => {
      expect(screen.queryAllByText("failed cards selected for retry")).toHaveLength(0)
    })
    expect(document.querySelector(".fixed.bottom-4")).toBeNull()
  }, 15000)

  it("applies bulk remove tag to selected cards case-insensitively", async () => {
    vi.mocked(getFlashcard).mockResolvedValueOnce({
      ...sampleCard,
      version: 14,
      tags: ["biology", "Chemistry", "review"]
    })
    vi.mocked(updateFlashcard).mockResolvedValue(undefined as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId(`flashcard-item-${sampleCard.uuid}-select`))
    fireEvent.click(screen.getByRole("button", { name: "Remove tag" }))
    fireEvent.change(screen.getByTestId("flashcards-bulk-tag-input"), {
      target: { value: "chemistry, REVIEW" }
    })
    const removeTagButtons = screen.getAllByRole("button", { name: "Remove tag" })
    fireEvent.click(removeTagButtons[removeTagButtons.length - 1])

    await waitFor(() => {
      expect(updateFlashcard).toHaveBeenCalledTimes(1)
    })
    expect(vi.mocked(updateFlashcard).mock.calls[0][0]).toBe(sampleCard.uuid)
    expect(vi.mocked(updateFlashcard).mock.calls[0][1]).toMatchObject({
      tags: ["biology"],
      expected_version: 14
    })
  }, 15000)

  it("keeps failed cards selected and warns when bulk move partially fails", async () => {
    const cards = buildSampleCards(2)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: cards,
        count: cards.length,
        total: cards.length
      },
      isFetching: false
    } as any)
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({
        ...cards[0],
        version: 21,
        deck_id: 1
      })
      .mockResolvedValueOnce({
        ...cards[1],
        version: 22,
        deck_id: 1
      })
    vi.mocked(updateFlashcard)
      .mockResolvedValueOnce(undefined as any)
      .mockRejectedValueOnce(new Error("move failed"))

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId("flashcard-item-card-undo-1-select"))
    fireEvent.click(screen.getByTestId("flashcard-item-card-undo-2-select"))
    fireEvent.click(screen.getByRole("button", { name: "Move" }))
    const comboboxes = screen.getAllByRole("combobox")
    fireEvent.mouseDown(comboboxes[comboboxes.length - 1])
    fireEvent.click(screen.getByText("Deck 2"))
    const moveButtons = screen.getAllByRole("button", { name: "Move" })
    fireEvent.click(moveButtons[moveButtons.length - 1])

    await waitFor(() => {
      expect(messageSpies.warning).toHaveBeenCalledWith(
        "Moved 1 card; 1 failed. Failed cards remain selected so you can retry."
      )
    })
    expect(messageSpies.success).not.toHaveBeenCalled()
    expect(showUndoNotificationMock).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId("flashcard-item-card-undo-1-select")).not.toBeChecked()
    expect(screen.getByTestId("flashcard-item-card-undo-2-select")).toBeChecked()
  }, 15000)

  it("renders large bulk delete warning with the design-system Alert", async () => {
    const cards = buildSampleCards(101)
    vi.mocked(listFlashcards).mockResolvedValue({
      items: cards,
      total: cards.length,
      page: 1,
      page_size: cards.length
    } as any)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [sampleCard],
        count: cards.length,
        total: cards.length
      },
      isFetching: false
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(
      screen.getByLabelText("Select all cards on this page")
    )
    fireEvent.click(screen.getByTestId("flashcards-select-all-across"))
    fireEvent.click(screen.getByRole("button", { name: "Delete" }))

    const warning = await screen.findByText(
      "These cards will move to Trash for 30 seconds."
    )
    expect(listFlashcards).toHaveBeenCalled()
    expect(warning.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  }, 15000)
})
