import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { FlashcardsManager } from "../FlashcardsManager"

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  useDecksQuery: vi.fn(),
  decks: [{ id: 1, name: "Biology" }]
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
      hash: window.location.hash
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
  }) => (
    <div data-testid="mock-review-tab">
      <button onClick={props.onNavigateToCreate}>Route Create</button>
      <button onClick={() => props.onReviewDeckChange(12)}>Select Deck 12</button>
      <span data-testid="mock-review-deck-id">{String(props.reviewDeckId ?? "")}</span>
    </div>
  ),
  ManageTab: (props: {
    onNavigateToImport: () => void
    openCreateSignal?: number
    initialDeckId?: number
    initialShowWorkspaceDecks?: boolean
  }) => (
    <div data-testid="mock-manage-tab">
      <button onClick={props.onNavigateToImport}>Route Import</button>
      <span data-testid="mock-open-create-signal">{String(props.openCreateSignal ?? 0)}</span>
      <span data-testid="mock-manage-initial-deck-id">{String(props.initialDeckId ?? "")}</span>
      <span data-testid="mock-manage-show-workspace">{String(props.initialShowWorkspaceDecks ?? false)}</span>
    </div>
  ),
  ImportExportTab: () => <div data-testid="mock-transfer-tab">Import / Export panel</div>,
  TemplatesTab: () => <div data-testid="mock-templates-tab">Templates panel</div>,
  SchedulerTab: (props: {
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

  it("uses Study/Manage/Transfer/Scheduler tab labels", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    expect(screen.getByText("Study")).toBeInTheDocument()
    expect(screen.getByText("Manage")).toBeInTheDocument()
    expect(screen.getByText("Import / Export")).toBeInTheDocument()
    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.getByText("Scheduler")).toBeInTheDocument()
  })

  it("defaults to Import / Export and hides Scheduler when no decks are available", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards")

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.getByText("Templates")).toBeInTheDocument()
    expect(screen.queryByText("Scheduler")).not.toBeInTheDocument()
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
    expect(screen.queryByText("Scheduler")).not.toBeInTheDocument()
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

  it("clamps scheduler deep-links to Import / Export when Scheduler is hidden", () => {
    mocks.decks = []
    window.history.replaceState({}, "", "/flashcards?tab=scheduler&deck_id=9")

    render(<FlashcardsManager />)

    expect(screen.getByTestId("mock-transfer-tab")).toBeInTheDocument()
    expect(screen.queryByTestId("mock-scheduler-tab")).not.toBeInTheDocument()
  })

  it("routes secondary create CTA to the Manage tab create entry point", () => {
    window.history.replaceState({}, "", "/flashcards")
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByText("Route Create"))
    expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
    expect(screen.getByTestId("mock-open-create-signal")).toHaveTextContent("1")
  })

  it("keeps quiz CTA usable when handoff IDs are invalid", () => {
    window.history.replaceState(
      {},
      "",
      "/flashcards?study_source=quiz&quiz_id=abc&attempt_id=-1&deck_id=0"
    )
    render(<FlashcardsManager />)

    fireEvent.click(screen.getByTestId("flashcards-to-quiz-cta"))

    expect(mocks.navigate).toHaveBeenCalledWith("/quiz?tab=take&source=flashcards")
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
