import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { FlashcardsManager } from "../FlashcardsManager"

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  useDecksQuery: vi.fn(),
  decks: [{ id: 1, name: "Biology" }],
  locationKey: "initial-location",
  translationKeys: [] as string[]
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
      mocks.translationKeys.push(key)
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

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useNavigate: () => mocks.navigate,
    useLocation: () => ({
      pathname: window.location.pathname,
      search: window.location.search,
      hash: window.location.hash,
      key: mocks.locationKey
    })
  }
})

vi.mock("../hooks", () => ({
  useDecksQuery: (...args: unknown[]) => mocks.useDecksQuery(...args)
}))

vi.mock("../tabs", () => ({
  ReviewTab: (props: {
    onNavigateToCreate: () => void
    onNavigateToImport: () => void
    onNavigateToGenerate?: () => void
    reviewDeckId?: number | null
    onReviewDeckChange: (deckId: number | null | undefined) => void
    onNavigateToManageDeck?: (deckId: number) => void
    onNavigateToSchedulerDeck?: (deckId: number) => void
    onNavigateToExportDeck?: (deckId: number) => void
    onSourceReviewGenerate?: (intent: {
      activity_type: "flashcards" | "cloze"
      text: string
      source_items: Array<{ source_type: "note"; source_id: string }>
    }) => void
    onSourceReviewQuiz?: (intent: {
      occurrence_id: number
      plan_id: number
      activity_type: "quiz"
      source_bundle: { items: Array<{ source_type: "note"; source_id: string }> }
    }) => void
  }) => (
    <div data-testid="mock-review-tab">
      <button onClick={props.onNavigateToCreate}>Route Create</button>
      <button onClick={props.onNavigateToImport}>Route Import</button>
      <button onClick={() => props.onNavigateToGenerate?.()}>Route Generate</button>
      <button onClick={() => props.onReviewDeckChange(12)}>Select Deck 12</button>
      <button onClick={() => props.onReviewDeckChange(21)}>Select Deck 21</button>
      <button onClick={() => props.onReviewDeckChange(undefined)}>Clear Deck</button>
      <button onClick={() => props.onNavigateToManageDeck?.(12)}>Route Manage Deck 12</button>
      <button onClick={() => props.onNavigateToManageDeck?.(21)}>Route Manage Deck 21</button>
      <button onClick={() => props.onNavigateToSchedulerDeck?.(12)}>Route Scheduler Deck 12</button>
      <button onClick={() => props.onNavigateToExportDeck?.(12)}>Route Export Deck 12</button>
      <button onClick={() => props.onNavigateToExportDeck?.(21)}>Route Export Deck 21</button>
      <button
        onClick={() => props.onSourceReviewGenerate?.({
          activity_type: "cloze",
          text: "Grounded source excerpt",
          source_items: [{ source_type: "note", source_id: "note-42" }]
        })}
      >
        Route Source Cloze
      </button>
      <button
        onClick={() => props.onSourceReviewQuiz?.({
          occurrence_id: 31,
          plan_id: 7,
          activity_type: "quiz",
          source_bundle: {
            items: [{ source_type: "note", source_id: "note-42" }]
          }
        })}
      >
        Route Source Quiz
      </button>
      <span data-testid="mock-review-deck-id">{String(props.reviewDeckId ?? "")}</span>
    </div>
  ),
  ManageTab: (props: {
    onNavigateToImport: () => void
    onNavigateToGenerate?: () => void
    onReviewCard: (card: { deck_id: number }) => void
    openCreateSignal?: number
    initialDeckId?: number
    initialDeckHandoffKey?: string | null
    initialShowWorkspaceDecks?: boolean
    createInitialDeckId?: number | null
    createInitialShowWorkspaceDecks?: boolean
    onCreateHandoffConsumed?: () => void
  }) => (
    <div data-testid="mock-manage-tab">
      <button onClick={props.onNavigateToImport}>Route Import</button>
      <button onClick={() => props.onNavigateToGenerate?.()}>Route Generate</button>
      <button onClick={() => props.onReviewCard({ deck_id: 21 })}>Review Managed Card 21</button>
      <button onClick={() => props.onCreateHandoffConsumed?.()}>Consume Create Handoff</button>
      <span data-testid="mock-open-create-signal">{String(props.openCreateSignal ?? 0)}</span>
      <span data-testid="mock-manage-initial-deck-id">{String(props.initialDeckId ?? "")}</span>
      <span data-testid="mock-manage-handoff-key">{String(props.initialDeckHandoffKey ?? "")}</span>
      <span data-testid="mock-manage-show-workspace">{String(props.initialShowWorkspaceDecks ?? false)}</span>
      <span data-testid="mock-create-initial-deck-id">{String(props.createInitialDeckId ?? "")}</span>
      <span data-testid="mock-create-show-workspace">
        {String(props.createInitialShowWorkspaceDecks ?? false)}
      </span>
    </div>
  ),
  ImportExportTab: (props: {
    initialTask?: "create" | "import" | "export" | null
    initialTaskHandoffKey?: string | null
    initialExportDeckId?: number | null
    initialExportDeckHandoffKey?: string | null
    sourceReviewIntent?: {
      activity_type: "flashcards" | "cloze"
      text: string
    } | null
  }) => (
    <div data-testid="mock-transfer-tab">
      Import / Export panel
      <span data-testid="mock-transfer-initial-task">{String(props.initialTask ?? "")}</span>
      <span data-testid="mock-transfer-task-handoff-key">{String(props.initialTaskHandoffKey ?? "")}</span>
      <span data-testid="mock-export-initial-deck-id">{String(props.initialExportDeckId ?? "")}</span>
      <span data-testid="mock-export-handoff-key">{String(props.initialExportDeckHandoffKey ?? "")}</span>
      <span data-testid="mock-source-review-activity">
        {String(props.sourceReviewIntent?.activity_type ?? "")}
      </span>
      <span data-testid="mock-source-review-text">
        {String(props.sourceReviewIntent?.text ?? "")}
      </span>
    </div>
  ),
  TemplatesTab: () => <div data-testid="mock-templates-tab">Templates panel</div>,
  SchedulerTab: (props: {
    initialDeckId?: number | null
    initialDeckHandoffKey?: string | null
    deckVisibilityOptions?: {
      includeWorkspaceItems?: boolean
    }
    onDirtyChange?: (dirty: boolean) => void
    discardSignal?: number
  }) => {
    const [draftState, setDraftState] = React.useState("clean")

    React.useEffect(() => {
      setDraftState("clean")
      props.onDirtyChange?.(false)
    }, [props.discardSignal, props.onDirtyChange])

    return (
      <div data-testid="mock-scheduler-tab">
        Scheduler panel
        <span data-testid="mock-scheduler-initial-deck-id">{String(props.initialDeckId ?? "")}</span>
        <span data-testid="mock-scheduler-handoff-key">{String(props.initialDeckHandoffKey ?? "")}</span>
        <span data-testid="mock-scheduler-include-workspace">
          {String(props.deckVisibilityOptions?.includeWorkspaceItems ?? false)}
        </span>
        <span data-testid="mock-scheduler-discard-signal">{String(props.discardSignal ?? 0)}</span>
        <span data-testid="mock-scheduler-draft-state">{draftState}</span>
        <button
          onClick={() => {
            setDraftState("dirty")
            props.onDirtyChange?.(true)
          }}
        >
          Mark Scheduler Dirty
        </button>
        <button
          onClick={() => {
            setDraftState("clean")
            props.onDirtyChange?.(false)
          }}
        >
          Mark Scheduler Clean
        </button>
      </div>
    )
  }
}))

vi.mock("../components", () => ({
  KeyboardShortcutsModal: () => null
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const renderFlashcardsManager = ({
  decks = mocks.decks
}: {
  decks?: Array<{ id: number; name: string }>
} = {}) => {
  mocks.decks = decks
  window.history.replaceState({}, "", "/flashcards")
  return render(<FlashcardsManager />)
}

describe("FlashcardsManager consistency standards", () => {
  mocks.useDecksQuery.mockImplementation(() => ({
    data: mocks.decks,
    isLoading: false
  }))

  beforeEach(() => {
    mocks.navigate.mockReset()
    mocks.decks = [{ id: 1, name: "Biology" }]
    mocks.locationKey = "initial-location"
    mocks.translationKeys = []
    mocks.useDecksQuery.mockClear()
    mocks.useDecksQuery.mockImplementation(() => ({
      data: mocks.decks,
      isLoading: false
    }))
  })

  it("hydrates review deck and quiz context from quiz-study handoff params", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?tab=review&study_source=quiz&quiz_id=21&attempt_id=88&deck_id=4"
    )

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("4")
    fireEvent.click(screen.getByTestId("flashcards-to-quiz-cta"))

    expect(mocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining(
        "/quiz?tab=take&source=flashcards&start_quiz_id=21&highlight_quiz_id=21&deck_id=4&source_attempt_id=88"
      )
    )
  })

  it("opens Import / Export tab first when URL contains generate intent", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?generate=1&generate_text=Study%20notes"
    )

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
  })

  it("opens the Manage tab with a preselected workspace deck from direct-link params", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?tab=manage&deck_id=9&include_workspace_items=1"
    )

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-manage-initial-deck-id")).toHaveTextContent("9")
    expect(screen.getByTestId("mock-manage-show-workspace")).toHaveTextContent("true")
  })

  it("opens Scheduler with the deck id from direct-link params", () => {
    window.history.replaceState({}, "", "/flashcards?tab=scheduler&deck_id=9")

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-scheduler-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("9")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("initial-location")
  })

  it("passes workspace deck visibility to Scheduler for workspace handoffs", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?tab=scheduler&deck_id=9&include_workspace_items=1"
    )

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-scheduler-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-scheduler-include-workspace")).toHaveTextContent("true")
  })

  it("refreshes Scheduler handoff identity when the same deck deep-link is requested again", () => {
    mocks.locationKey = "scheduler-handoff-1"
    window.history.replaceState({}, "", "/flashcards?tab=scheduler&deck_id=9")

    const { rerender } = render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("9")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent(
      "scheduler-handoff-1"
    )

    mocks.locationKey = "scheduler-handoff-2"
    rerender(<FlashcardsManager />)

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("9")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent(
      "scheduler-handoff-2"
    )
  })

  it("hands the selected Study deck to Scheduler when opening the tab manually", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Select Deck 12"))
    fireEvent.click(screen.getByText("Scheduler"))

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("review:12")
  })

  it("routes deck dashboard actions to the target deck workflows", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Manage Deck 12"))
    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-manage-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-manage-handoff-key")).toHaveTextContent("manage:12:")

    fireEvent.click(screen.getByText("Study"))
    fireEvent.click(screen.getByText("Route Scheduler Deck 12"))
    expect(screen.getByTestId("mock-scheduler-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("scheduler:12:")

    fireEvent.click(screen.getByText("Study"))
    fireEvent.click(screen.getByText("Route Export Deck 12"))
    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-export-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-export-handoff-key")).toHaveTextContent("export:12:")
  })

  it("clears one-shot scheduler handoff after Study deck selection changes", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Scheduler Deck 12"))
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("scheduler:12:")

    fireEvent.click(screen.getByText("Study"))
    fireEvent.click(screen.getByText("Select Deck 21"))
    fireEvent.click(screen.getByText("Scheduler"))

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("21")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("review:21")
  })

  it("clears stale scheduler handoff when dashboard Manage changes the deck", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Scheduler Deck 12"))
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("scheduler:12:")

    fireEvent.click(screen.getByText("Study"))
    fireEvent.click(screen.getByText("Route Manage Deck 21"))
    expect(screen.getByTestId("mock-manage-initial-deck-id")).toHaveTextContent("21")

    fireEvent.click(screen.getByText("Scheduler"))

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("21")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("review:21")
  })

  it("clears stale scheduler handoff when Manage starts reviewing another deck", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Scheduler Deck 12"))
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")

    fireEvent.click(screen.getByText("Manage"))
    fireEvent.click(screen.getByText("Review Managed Card 21"))
    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("21")

    fireEvent.click(screen.getByText("Scheduler"))

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("21")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("review:21")
  })

  it("clears stale scheduler handoff when dashboard Export changes the deck", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Scheduler Deck 12"))
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")

    fireEvent.click(screen.getByText("Study"))
    fireEvent.click(screen.getByText("Route Export Deck 21"))
    expect(screen.getByTestId("mock-export-initial-deck-id")).toHaveTextContent("21")

    fireEvent.click(screen.getByText("Scheduler"))

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("21")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("review:21")
  })

  it("clears stale scheduler handoff when route params change the Study deck", () => {
    window.history.replaceState({}, "", "/flashcards")
    const { rerender } = render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Scheduler Deck 12"))
    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("12")

    window.history.replaceState({}, "", "/flashcards?tab=review&deck_id=21")
    mocks.locationKey = "review-deck-21"
    rerender(<FlashcardsManager />)
    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("21")

    fireEvent.click(screen.getByText("Scheduler"))

    expect(screen.getByTestId("mock-scheduler-initial-deck-id")).toHaveTextContent("21")
    expect(screen.getByTestId("mock-scheduler-handoff-key")).toHaveTextContent("review:21")
  })

  it("uses Study/Manage/Import Export/Templates/Scheduler tab labels", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    expect(screen.getByText("Study")).toBeInTheDocument()
    expect(screen.getByText("Manage")).toBeInTheDocument()
    expect(screen.getByText("Import / Export")).toBeInTheDocument()
    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.getByText("Scheduler")).toBeInTheDocument()
    expect(mocks.translationKeys).toContain("option:flashcards.importExport")
    expect(mocks.translationKeys).not.toContain("option:flashcards.tabImportExport")
  })

  it("keeps tab actions in a responsive wrapping container", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    expect(screen.getByTestId("flashcards-tabs")).toHaveClass("flashcards-responsive-tabs")
    expect(screen.getByTestId("flashcards-tab-actions")).toHaveClass(
      "flex",
      "flex-wrap",
      "min-w-0",
      "max-w-full"
    )
  })

  it("defaults zero-deck users to Study and keeps Scheduler discoverable", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards")

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-review-tab")).toBeInTheDocument()
    expect(screen.getByText("Import / Export")).toBeInTheDocument()
    expect(screen.queryByText("LLM")).not.toBeInTheDocument()
    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /scheduler/i })).toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
  })

  it("documents zero-deck behavior after first-time IA remediation", async () => {
    renderFlashcardsManager({ decks: [] })

    expect(await screen.findByText("Import / Export")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /scheduler/i })).toBeInTheDocument()
  })

  it("still opens Import / Export first for explicit generate and study-pack intents", () => {
    mocks.decks = []
    window.history.replaceState(
      {},
      "",
      "/flashcards?generate=1&generate_text=Study%20notes"
    )

    const { unmount } = render(<FlashcardsManager />)
    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()

    unmount()
    const studyPackPayload = encodeURIComponent(
      JSON.stringify({
        title: "Lecture 5",
        sourceItems: [{ sourceType: "media", sourceId: "42" }]
      })
    )
    window.history.replaceState(
      {},
      "",
      `/flashcards?study_pack=1&study_pack_payload=${studyPackPayload}`
    )

    render(<FlashcardsManager />)
    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
  })

  it("routes template deep-links to the Templates tab", () => {
    window.history.replaceState({}, "", "/flashcards?tab=templates")

    render(<FlashcardsManager />)

    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.getByTestId("mock-templates-tab")).toBeInTheDocument()
  })

  it("keeps the Templates tab reachable when no decks exist", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards?tab=templates")

    render(<FlashcardsManager />)

    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.getByTestId("mock-templates-tab")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /scheduler/i })).toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
  })

  it("requests workspace decks when study links include workspace items", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?tab=review&deck_id=9&include_workspace_items=1"
    )

    render(<FlashcardsManager />)

    expect(mocks.useDecksQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        includeWorkspaceItems: true
      })
    )
  })

  it("opens a Scheduler preview for zero-deck Scheduler deep-links", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards?tab=scheduler&deck_id=9")

    render(<FlashcardsManager />)

    expect(screen.getByRole("tab", { name: /scheduler/i })).toHaveAttribute("aria-selected", "true")
    expect(screen.getByTestId("flashcards-scheduler-empty-preview")).toHaveTextContent(
      /create a deck/i
    )
    expect(screen.getByRole("button", { name: /create a deck/i })).toBeInTheDocument()
    expect(screen.queryByTestId("mock-review-tab")).not.toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
  })

  it("routes secondary create CTA to the Manage tab create entry point", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Create"))
    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-open-create-signal")).toHaveTextContent("1")
  })

  it("routes Study import CTA to the Import file task", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(within(screen.getByTestId("mock-review-tab")).getByText("Route Import"))

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-transfer-initial-task")).toHaveTextContent("import")
    expect(screen.getByTestId("mock-transfer-task-handoff-key")).toHaveTextContent("import:")
  })

  it("routes Study generate CTA to the Create and generate task", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(within(screen.getByTestId("mock-review-tab")).getByText("Route Generate"))

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-transfer-initial-task")).toHaveTextContent("create")
    expect(screen.getByTestId("mock-transfer-task-handoff-key")).toHaveTextContent("create:")
  })

  it("routes source-review cloze text into the generation workspace", () => {
    renderFlashcardsManager()

    fireEvent.click(screen.getByText("Route Source Cloze"))

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-transfer-initial-task")).toHaveTextContent("create")
    expect(screen.getByTestId("mock-source-review-activity")).toHaveTextContent("cloze")
    expect(screen.getByTestId("mock-source-review-text")).toHaveTextContent(
      "Grounded source excerpt"
    )
  })

  it("routes source-review quiz snapshots with a short session token", () => {
    renderFlashcardsManager()

    fireEvent.click(screen.getByText("Route Source Quiz"))

    expect(mocks.navigate).toHaveBeenCalledWith(
      expect.stringMatching(
        /^\/quiz\?tab=generate&source_review=1&source_review_token=[^&]+$/
      )
    )
    expect(mocks.navigate.mock.calls[0][0]).not.toContain("note-42")
  })

  it("clears stale source-review context when a newer URL handoff arrives", async () => {
    const rendered = renderFlashcardsManager()
    fireEvent.click(screen.getByText("Route Source Cloze"))
    expect(screen.getByTestId("mock-source-review-activity")).toHaveTextContent("cloze")

    window.history.replaceState(
      {},
      "",
      "/flashcards?generate=1&generate_text=New%20URL%20source"
    )
    mocks.locationKey = "url-generation-handoff"
    rendered.rerender(<FlashcardsManager />)

    await waitFor(() => {
      expect(screen.getByTestId("mock-source-review-activity")).toBeEmptyDOMElement()
    })
  })

  it("routes Manage empty-state transfer CTAs to distinct transfer tasks", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Manage"))
    fireEvent.click(within(screen.getByTestId("mock-manage-tab")).getByText("Route Import"))

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-transfer-initial-task")).toHaveTextContent("import")
    expect(screen.getByTestId("mock-transfer-task-handoff-key")).toHaveTextContent("import:")

    fireEvent.click(screen.getByText("Manage"))
    fireEvent.click(within(screen.getByTestId("mock-manage-tab")).getByText("Route Generate"))

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-transfer-initial-task")).toHaveTextContent("create")
    expect(screen.getByTestId("mock-transfer-task-handoff-key")).toHaveTextContent("create:")
  })

  it("passes the selected Study deck to the Manage create drawer handoff", () => {
    window.history.replaceState({}, "", "/flashcards?tab=review&deck_id=12")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Create"))

    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-open-create-signal")).toHaveTextContent("1")
    expect(screen.getByTestId("mock-create-initial-deck-id")).toHaveTextContent("12")
  })

  it("does not resurrect a URL Study deck after the live Study selection is cleared", () => {
    window.history.replaceState({}, "", "/flashcards?tab=review&deck_id=12")
    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("12")
    fireEvent.click(screen.getByText("Clear Deck"))
    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("")

    fireEvent.click(screen.getByText("Route Create"))

    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-create-initial-deck-id")).toBeEmptyDOMElement()
  })

  it("does not keep URL workspace visibility after the live Study selection is cleared", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?tab=review&deck_id=12&include_workspace_items=1"
    )
    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("12")
    fireEvent.click(screen.getByText("Clear Deck"))
    expect(screen.getByTestId("mock-review-deck-id")).toHaveTextContent("")

    fireEvent.click(screen.getByText("Route Create"))

    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-create-initial-deck-id")).toBeEmptyDOMElement()
    expect(screen.getByTestId("mock-create-show-workspace")).toHaveTextContent("false")
  })

  it("clears the create drawer Study deck handoff after Manage consumes it", () => {
    window.history.replaceState({}, "", "/flashcards?tab=review&deck_id=12")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Create"))
    expect(screen.getByTestId("mock-create-initial-deck-id")).toHaveTextContent("12")

    fireEvent.click(screen.getByText("Consume Create Handoff"))

    expect(screen.getByTestId("mock-create-initial-deck-id")).toHaveTextContent("")
  })

  it("disables quiz CTA when handoff IDs are invalid", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?study_source=quiz&quiz_id=abc&attempt_id=-1&deck_id=0"
    )
    render(<FlashcardsManager />)

    const quizButton = screen.getByTestId("flashcards-to-quiz-cta")
    expect(quizButton).toBeDisabled()
    fireEvent.click(quizButton)

    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it("enables quiz CTA after selecting a review deck", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Select Deck 12"))

    const quizButton = screen.getByTestId("flashcards-to-quiz-cta")
    expect(quizButton).not.toBeDisabled()
    fireEvent.click(quizButton)

    expect(mocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining("/quiz?tab=take&source=flashcards&deck_id=12")
    )
  })

  it("disables quiz CTA without a valid quiz or deck context", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    expect(screen.getByTestId("flashcards-to-quiz-cta")).toBeDisabled()
  })

  it("prompts before leaving the Scheduler tab when its draft is dirty", () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    confirmSpy.mockReturnValue(false)

    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Scheduler"))
    expect(screen.getByTestId("mock-scheduler-tab")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Mark Scheduler Dirty"))
    expect(screen.getByTestId("mock-scheduler-draft-state")).toHaveTextContent("dirty")
    fireEvent.click(screen.getByText("Manage"))

    expect(confirmSpy).toHaveBeenCalled()
    expect(screen.getByTestId("mock-scheduler-tab")).toBeInTheDocument()

    confirmSpy.mockReturnValue(true)
    fireEvent.click(screen.getByText("Manage"))
    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Scheduler"))
    expect(screen.getByTestId("mock-scheduler-draft-state")).toHaveTextContent("clean")

    confirmSpy.mockRestore()
  })

  it("discards scheduler draft state when Scheduler is replaced by the zero-deck preview", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    window.history.replaceState({}, "", "/flashcards")
    const { rerender } = render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Scheduler"))
    fireEvent.click(screen.getByText("Mark Scheduler Dirty"))
    expect(screen.getByTestId("mock-scheduler-draft-state")).toHaveTextContent("dirty")

    mocks.decks = []
    rerender(<FlashcardsManager />)

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-scheduler-empty-preview")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText("Manage"))

    expect(confirmSpy).not.toHaveBeenCalled()
    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()

    confirmSpy.mockRestore()
  })
})
