// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Form } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardCreateDrawer } from "../FlashcardCreateDrawer"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardTemplateMutation,
  useDecksQuery,
  useFlashcardDeckRecentCardsQuery,
  useFlashcardDeckSearchQuery
} from "../../hooks"
import type { Deck, DeckSchedulerSettings, DeckSchedulerSettingsEnvelope } from "@/services/flashcards"

const defaultFsrsSettings = {
  target_retention: 0.9,
  maximum_interval_days: 36500,
  enable_fuzz: false
}

const defaultSm2Settings = {
  new_steps_minutes: [1, 10],
  relearn_steps_minutes: [10],
  graduating_interval_days: 1,
  easy_interval_days: 4,
  easy_bonus: 1.3,
  interval_modifier: 1,
  max_interval_days: 36500,
  leech_threshold: 8,
  enable_fuzz: false
}

const defaultSchedulerSettingsEnvelope: DeckSchedulerSettingsEnvelope = {
  sm2_plus: defaultSm2Settings,
  fsrs: defaultFsrsSettings
}

const makeDeck = (
  id: number,
  name: string,
  schedulerSettings?: DeckSchedulerSettingsEnvelope
) : Deck => ({
    id,
    name,
    description: null,
    review_prompt_side: "front",
    deleted: false,
    client_id: "test",
    version: 1,
    scheduler_type: "sm2_plus",
    scheduler_settings_json: schedulerSettings ? JSON.stringify(schedulerSettings) : null,
    scheduler_settings: schedulerSettings ?? defaultSchedulerSettingsEnvelope
  })

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

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCreateFlashcardMutation: vi.fn(),
  useCreateDeckMutation: vi.fn(),
  useCreateFlashcardTemplateMutation: vi.fn(),
  useDebouncedFormField: vi.fn((form, field) => Form.useWatch(field, form)),
  useFlashcardDeckRecentCardsQuery: vi.fn(),
  useFlashcardDeckSearchQuery: vi.fn()
}))

vi.mock("../MarkdownWithBoundary", () => ({
  MarkdownWithBoundary: ({ content }: { content: string }) => <div>{content}</div>
}))

vi.mock("../FlashcardImageInsertButton", () => ({
  FlashcardImageInsertButton: ({ ariaLabel, buttonLabel }: { ariaLabel: string; buttonLabel: string }) => (
    <button type="button" aria-label={ariaLabel}>
      {buttonLabel}
    </button>
  )
}))

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: ({
    value = [],
    onChange,
    dataTestId,
    placeholder
  }: {
    value?: string[]
    onChange?: (value: string[]) => void
    dataTestId?: string
    placeholder?: string
  }) => {
    const testId = dataTestId ?? "flashcard-tag-picker"

    return (
      <div data-testid={testId}>
        {value.map((tag) => (
          <span key={tag}>{tag}</span>
        ))}
        <input
          data-testid={`${testId}-search-input`}
          placeholder={placeholder}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return
            event.preventDefault()
            const nextTag = event.currentTarget.value.trim()
            if (!nextTag) return
            onChange?.([...value, nextTag])
            event.currentTarget.value = ""
          }}
        />
      </div>
    )
  }
}))

vi.mock("../utils/text-selection", () => ({
  getSelectionFromElement: () => ({ start: 0, end: 0 }),
  insertTextAtSelection: (value: string, _selection: { start: number; end: number }, inserted: string) => ({
    nextValue: `${value}${inserted}`,
    cursor: value.length + inserted.length
  }),
  restoreSelection: () => {}
}))

vi.mock("@/components/Option/WritingPlayground/writing-editor-actions-utils", () => ({
  applyTextAtRange: (value: string, start: number, end: number, inserted: string) => ({
    nextValue: `${value.slice(0, start)}${inserted}${value.slice(end)}`,
    cursor: start + inserted.length
  })
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

describe("FlashcardCreateDrawer deck reference section", () => {
  const createFlashcardMutateAsync = vi.fn()
  const createDeckMutateAsync = vi.fn()
  const recentRefetch = vi.fn()
  const searchRefetch = vi.fn()
  type DrawerProps = Parameters<typeof FlashcardCreateDrawer>[0]
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>
  let recentState: Record<string, unknown>
  let searchState: Record<string, unknown>
  let currentDecks: Deck[]

  beforeEach(() => {
    vi.clearAllMocks()
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    currentDecks = [makeDeck(1, "Biology"), makeDeck(2, "Physics")]

    recentState = {
      data: [],
      isLoading: false,
      isError: false,
      error: null,
      refetch: recentRefetch
    }
    searchState = {
      data: [],
      isLoading: false,
      isError: false,
      error: null,
      refetch: searchRefetch
    }

    vi.mocked(useDecksQuery).mockImplementation(() => ({
      data: currentDecks,
      isLoading: false
    } as any))
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createFlashcardMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: createDeckMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useFlashcardDeckRecentCardsQuery).mockImplementation(() => recentState as any)
    vi.mocked(useFlashcardDeckSearchQuery).mockImplementation(() => searchState as any)
  })

  afterEach(() => {
    expect(consoleErrorSpy).not.toHaveBeenCalled()
    consoleErrorSpy.mockRestore()
  })

  const renderDrawer = (props: Partial<DrawerProps> = {}) =>
    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} {...props} />)

  const selectDeck = async (deckName: string) => {
    const deckInputs = screen.getAllByLabelText("Deck")
    fireEvent.mouseDown(deckInputs[deckInputs.length - 1] as HTMLElement)
    fireEvent.click(await screen.findByText(deckName))
  }

  const expandReferenceSection = async () => {
    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: /existing cards in this deck/i }).length).toBeGreaterThan(0)
    })
    const expandButtons = screen.getAllByRole("button", { name: /existing cards in this deck/i })
    fireEvent.click(expandButtons[expandButtons.length - 1] as HTMLElement)
  }

  const openAdvancedOptions = async () => {
    fireEvent.click(
      screen.getByRole("button", { name: /advanced options/i })
    )
  }

  it("keeps the reference section hidden until a deck is selected", () => {
    renderDrawer()

    expect(screen.queryByText("Existing cards in this deck")).not.toBeInTheDocument()
    expect(screen.queryByText("Search this deck")).not.toBeInTheDocument()
  })

  it("shows the reference section after a deck is selected", async () => {
    renderDrawer()

    await selectDeck("Biology")

    await waitFor(() => {
      expect(
        screen.getByText("Existing cards in this deck")
      ).toBeInTheDocument()
    })
    expect(screen.getByText("Existing cards in this deck")).toBeInTheDocument()
    expect(screen.getByText("Biology deck")).toBeInTheDocument()
  })

  it("forwards workspace visibility options to the drawer deck reference hooks", async () => {
    renderDrawer({
      workspaceId: "workspace-123",
      includeWorkspaceItems: true
    })

    await selectDeck("Biology")
    await expandReferenceSection()

    expect(vi.mocked(useFlashcardDeckRecentCardsQuery)).toHaveBeenLastCalledWith(
      1,
      expect.objectContaining({
        enabled: true,
        limit: 6,
        workspaceId: "workspace-123",
        includeWorkspaceItems: true
      })
    )
    expect(vi.mocked(useFlashcardDeckSearchQuery)).toHaveBeenLastCalledWith(
      expect.objectContaining({
        deckId: 1,
        query: ""
      }),
      expect.objectContaining({
        enabled: false,
        workspaceId: "workspace-123",
        includeWorkspaceItems: true
      })
    )
  })

  it("preserves the selected deck, template, section expansion, and search term after Create & Add Another", async () => {
    createFlashcardMutateAsync.mockResolvedValueOnce({ uuid: "card-1" })

    renderDrawer()

    await selectDeck("Biology")

    fireEvent.mouseDown(screen.getByLabelText("Card model"))
    fireEvent.click(
      await screen.findByText("Cloze (Fill in the blank)", {
        selector: ".ant-select-item-option-content"
      })
    )

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "A {{c1::cell}} is the basic unit of life." }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "Back content" }
    })

    await openAdvancedOptions()

    const tagsField = screen.getByTestId("flashcards-create-tag-picker-search-input")
    fireEvent.change(tagsField, {
      target: { value: "science" }
    })
    fireEvent.keyDown(tagsField, {
      key: "Enter",
      code: "Enter",
      charCode: 13
    })
    fireEvent.change(screen.getByPlaceholderText("Optional hints or explanations..."), {
      target: { value: "Extra hint" }
    })
    fireEvent.change(screen.getByPlaceholderText("Internal notes (not shown during review)..."), {
      target: { value: "Internal note" }
    })

    await expandReferenceSection()
    fireEvent.change(screen.getByPlaceholderText("Search this deck"), {
      target: { value: "mitochondria" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Create & Add Another" }))

    await waitFor(() => {
      expect(createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(recentRefetch).not.toHaveBeenCalled()
    expect(searchRefetch).not.toHaveBeenCalled()

    expect((screen.getByPlaceholderText("Question or prompt...") as HTMLInputElement).value).toBe("")
    expect((screen.getByPlaceholderText("Answer...") as HTMLInputElement).value).toBe("")
    expect(
      (screen.getByPlaceholderText("Optional hints or explanations...") as HTMLInputElement).value
    ).toBe("")
    expect(
      (screen.getByPlaceholderText("Internal notes (not shown during review)...") as HTMLInputElement).value
    ).toBe("")
    expect(screen.queryByText("science")).not.toBeInTheDocument()
    expect(screen.getByText("Existing cards in this deck")).toBeInTheDocument()
    expect(screen.getByText("Biology deck")).toBeInTheDocument()
    expect((screen.getByPlaceholderText("Search this deck") as HTMLInputElement).value).toBe(
      "mitochondria"
    )
    expect(
      screen.getByText("Cloze (Fill in the blank)", {
        selector: ".ant-select-content-value"
      })
    ).toBeInTheDocument()
  })

  it("clears the reference search term when the selected deck changes", async () => {
    renderDrawer()

    await selectDeck("Biology")
    await expandReferenceSection()

    fireEvent.change(screen.getByPlaceholderText("Search this deck"), {
      target: { value: "mitochondria" }
    })

    await selectDeck("Physics")
    await expandReferenceSection()

    expect((screen.getByPlaceholderText("Search this deck") as HTMLInputElement).value).toBe("")
  })

  it("clears the reference search term when the drawer closes and reopens", async () => {
    const { rerender } = render(
      <FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />
    )

    await selectDeck("Biology")
    await expandReferenceSection()

    fireEvent.change(screen.getByPlaceholderText("Search this deck"), {
      target: { value: "mitochondria" }
    })

    rerender(<FlashcardCreateDrawer open={false} onClose={vi.fn()} onSuccess={vi.fn()} />)
    rerender(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await selectDeck("Biology")
    await expandReferenceSection()

    expect((screen.getByPlaceholderText("Search this deck") as HTMLInputElement).value).toBe("")
  }, 15000)

  it("selects the inline-created deck and shows the empty reference state", async () => {
    const createdDeckSettings: DeckSchedulerSettings = {
      new_steps_minutes: [1, 5, 15],
      relearn_steps_minutes: [10],
      graduating_interval_days: 1,
      easy_interval_days: 3,
      easy_bonus: 1.15,
      interval_modifier: 0.9,
      max_interval_days: 3650,
      leech_threshold: 10,
      enable_fuzz: false
    }
    const createdDeckEnvelope: DeckSchedulerSettingsEnvelope = {
      sm2_plus: createdDeckSettings,
      fsrs: defaultFsrsSettings
    }

    const createdDeck: Deck = {
      id: 7,
      name: "New deck",
      description: null,
      review_prompt_side: "front",
      deleted: false,
      client_id: "test",
      version: 1,
      scheduler_type: "sm2_plus",
      scheduler_settings_json: JSON.stringify(createdDeckEnvelope),
      scheduler_settings: createdDeckEnvelope
    }
    createDeckMutateAsync.mockImplementationOnce(async () => {
      currentDecks = [...currentDecks, createdDeck]
      return createdDeck
    })

    recentState = {
      ...recentState,
      data: []
    }

    renderDrawer()

    fireEvent.mouseDown(screen.getByLabelText("Deck"))
    fireEvent.click(await screen.findByText("Create new deck"))

    fireEvent.change(screen.getByPlaceholderText("New deck name"), {
      target: { value: "New deck" }
    })
    fireEvent.click(screen.getByTestId("deck-scheduler-editor-preset-fast_acquisition"))
    fireEvent.click(screen.getByTestId("flashcards-inline-create-deck-submit"))

    await waitFor(() => {
      expect(createDeckMutateAsync).toHaveBeenCalledWith({
        name: "New deck",
        scheduler_type: "sm2_plus",
        scheduler_settings: createdDeckEnvelope
      })
    })
    expect(recentRefetch).not.toHaveBeenCalled()
    expect(searchRefetch).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(
        screen.getByText("Existing cards in this deck")
      ).toBeInTheDocument()
    })
    await expandReferenceSection()
    expect(screen.getByText("No recent cards in this deck yet.")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /edit/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^review$/i })).not.toBeInTheDocument()
  })
})
