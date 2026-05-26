import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { FlashcardsManager } from "../FlashcardsManager"

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  useDecksQuery: vi.fn(),
  decks: [{ id: 1, name: "Biology" }],
  locationKey: "initial-location"
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
    reviewDeckId?: number | null
    onReviewDeckChange: (deckId: number | null | undefined) => void
    onNavigateToManageDeck?: (deckId: number) => void
    onNavigateToSchedulerDeck?: (deckId: number) => void
    onNavigateToExportDeck?: (deckId: number) => void
  }) => (
    <div data-testid="mock-review-tab">
      <button onClick={props.onNavigateToCreate}>Route Create</button>
      <button onClick={() => props.onReviewDeckChange(12)}>Select Deck 12</button>
      <button onClick={() => props.onReviewDeckChange(21)}>Select Deck 21</button>
      <button onClick={() => props.onNavigateToManageDeck?.(12)}>Route Manage Deck 12</button>
      <button onClick={() => props.onNavigateToSchedulerDeck?.(12)}>Route Scheduler Deck 12</button>
      <button onClick={() => props.onNavigateToExportDeck?.(12)}>Route Export Deck 12</button>
      <span data-testid="mock-review-deck-id">{String(props.reviewDeckId ?? "")}</span>
    </div>
  ),
  ManageTab: (props: {
    onNavigateToImport: () => void
    openCreateSignal?: number
    initialDeckId?: number
    initialDeckHandoffKey?: string | null
    initialShowWorkspaceDecks?: boolean
  }) => (
    <div data-testid="mock-manage-tab">
      <button onClick={props.onNavigateToImport}>Route Import</button>
      <span data-testid="mock-open-create-signal">{String(props.openCreateSignal ?? 0)}</span>
      <span data-testid="mock-manage-initial-deck-id">{String(props.initialDeckId ?? "")}</span>
      <span data-testid="mock-manage-handoff-key">{String(props.initialDeckHandoffKey ?? "")}</span>
      <span data-testid="mock-manage-show-workspace">{String(props.initialShowWorkspaceDecks ?? false)}</span>
    </div>
  ),
  ImportExportTab: (props: {
    initialExportDeckId?: number | null
    initialExportDeckHandoffKey?: string | null
  }) => (
    <div data-testid="mock-transfer-tab">
      Import / Export panel
      <span data-testid="mock-export-initial-deck-id">{String(props.initialExportDeckId ?? "")}</span>
      <span data-testid="mock-export-handoff-key">{String(props.initialExportDeckHandoffKey ?? "")}</span>
    </div>
  ),
  TemplatesTab: () => <div data-testid="mock-templates-tab">Templates panel</div>,
  SchedulerTab: (props: {
    initialDeckId?: number | null
    initialDeckHandoffKey?: string | null
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

describe("FlashcardsManager consistency standards", () => {
  mocks.useDecksQuery.mockImplementation(() => ({
    data: mocks.decks,
    isLoading: false
  }))

  beforeEach(() => {
    mocks.navigate.mockReset()
    mocks.decks = [{ id: 1, name: "Biology" }]
    mocks.locationKey = "initial-location"
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

  it("uses Study/Manage/Create & Import/Scheduler tab labels", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    expect(screen.getByText("Study")).toBeInTheDocument()
    expect(screen.getByText("Manage")).toBeInTheDocument()
    expect(screen.getByText("Create & Import")).toBeInTheDocument()
    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.getByText("Scheduler")).toBeInTheDocument()
  })

  it("defaults to Study and keeps Scheduler discoverable when no decks are available", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards")

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-review-tab")).toBeInTheDocument()
    expect(screen.queryByTestId("mock-transfer-tab")).not.toBeInTheDocument()
    expect(screen.getByText("Templates")).toBeInTheDocument()
    const schedulerTab = screen.getByRole("tab", { name: /scheduler/i })
    const schedulerLabel = screen.getByText("Scheduler")
    expect(schedulerTab).not.toHaveAttribute("aria-disabled", "true")
    expect(schedulerLabel).toHaveAttribute(
      "aria-disabled",
      "true"
    )
    fireEvent.click(schedulerTab)
    expect(screen.getByTestId("mock-review-tab")).toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
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
    expect(screen.getByRole("tab", { name: /scheduler/i })).not.toHaveAttribute(
      "aria-disabled",
      "true"
    )
    expect(screen.getByText("Scheduler")).toHaveAttribute(
      "aria-disabled",
      "true"
    )
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

  it("clamps scheduler deep-links to Study when Scheduler is disabled", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards?tab=scheduler&deck_id=9")

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-review-tab")).toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
    const schedulerTab = screen.getByRole("tab", { name: /scheduler/i })
    expect(schedulerTab).not.toHaveAttribute("aria-disabled", "true")
    expect(screen.getByText("Scheduler")).toHaveAttribute(
      "aria-disabled",
      "true"
    )
    fireEvent.click(schedulerTab)
    expect(screen.getByTestId("mock-review-tab")).toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
  })

  it("routes secondary create CTA to the Manage tab create entry point", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Create"))
    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-open-create-signal")).toHaveTextContent("1")
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

  it("disables quiz CTA without a valid quiz handoff context", () => {
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
})
