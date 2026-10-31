import { MemoryRouter } from "react-router-dom"
import React from "react"
import type { TutorialRun } from "@/store/tutorials"
import { describe, expect, it, beforeEach, afterEach, vi } from "vitest"
import { fireEvent, render as renderUI, screen, within } from "@testing-library/react"
import { QUICK_CHAT_WORKFLOW_GUIDES } from "../workflow-guides"
import { QuickChatGuidesPanel } from "../QuickChatGuidesPanel"

const render = (ui: React.ReactNode) => renderUI(<MemoryRouter>{ui}</MemoryRouter>)

const useStorageMock = vi.fn()
const startTutorialMock = vi.fn()
const tutorialStoreSelectorMock = vi.fn()
const tutorialStoreState: {
  completedTutorials: string[]
  activeRun: TutorialRun | null
  startTutorial: typeof startTutorialMock
} = {
  completedTutorials: [],
  activeRun: null,
  startTutorial: startTutorialMock
}

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => useStorageMock(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        if ("defaultValue" in fallbackOrOptions && fallbackOrOptions.defaultValue) {
          return fallbackOrOptions.defaultValue
        }
      }
      if (key === "tutorials:steps") {
        return "5 steps"
      }
      if (key.endsWith(".label")) {
        return key
      }
      return key
    }
  })
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialStore: Object.assign(
    (selector: (state: unknown) => unknown) =>
      tutorialStoreSelectorMock(selector),
    { getState: () => tutorialStoreState }
  )
}))

describe("QuickChatGuidesPanel tutorials section", () => {
  afterEach(() => { startTutorialMock.mock.calls.forEach(call => call[1]?.dispose?.()) })
  beforeEach(() => {
    useStorageMock.mockReset()
    tutorialStoreSelectorMock.mockReset()
    startTutorialMock.mockReset()

    useStorageMock.mockReturnValue([QUICK_CHAT_WORKFLOW_GUIDES])
    tutorialStoreState.completedTutorials = []
    tutorialStoreState.activeRun = null
    startTutorialMock.mockImplementation((_id, focusReturn) => { tutorialStoreState.activeRun = { id: 1, focusReturn } })
    tutorialStoreSelectorMock.mockImplementation(
      (selector: (state: unknown) => unknown) => selector(tutorialStoreState)
    )
  })

  it("renders Tutorials section above workflow guide browser", () => {
    render(
      <QuickChatGuidesPanel
        onAskGuide={vi.fn()}
        onOpenRoute={vi.fn()}
        currentRoute="/prompts"
      />
    )

    const tutorialsTitle = screen.getByText("Tutorials for this page")
    const guidesTitle = screen.getByText("Workflow guide browser")
    const position = tutorialsTitle.compareDocumentPosition(guidesTitle)

    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText("Prompts Basics")).toBeInTheDocument()
  })

  it("starts selected tutorial from tutorials section", () => {
    const onStartTutorial = vi.fn()
    render(
      <QuickChatGuidesPanel
        onAskGuide={vi.fn()}
        onOpenRoute={vi.fn()}
        currentRoute="/prompts"
        onStartTutorial={onStartTutorial}
      />
    )

    const tutorialRow = screen
      .getByText("Prompts Basics")
      .closest("div")
    expect(tutorialRow).toBeTruthy()
    if (!tutorialRow) return

    const startButton = within(tutorialRow.parentElement as HTMLElement).getByRole(
      "button",
      { name: "Start" }
    )
    fireEvent.click(startButton)
    expect(onStartTutorial).toHaveBeenCalledWith("prompts-basics")
    expect(startTutorialMock).not.toHaveBeenCalled()
  })

  it("shows tutorials empty state while workflow cards remain visible when route has no tutorials", () => {
    render(
      <QuickChatGuidesPanel
        onAskGuide={vi.fn()}
        onOpenRoute={vi.fn()}
        currentRoute="/settings/health"
      />
    )

    expect(
      screen.getByText("No tutorials are available for this page yet.")
    ).toBeInTheDocument()
    expect(screen.getByText("Workflow guide browser")).toBeInTheDocument()
    expect(screen.getByText("Ingest + summarize a source")).toBeInTheDocument()
  })

  it("keeps workflow card Ask/Open actions wired after tutorials section changes", () => {
    const onAskGuide = vi.fn()
    const onOpenRoute = vi.fn()

    render(
      <QuickChatGuidesPanel
        onAskGuide={onAskGuide}
        onOpenRoute={onOpenRoute}
        currentRoute="/prompts"
      />
    )

    const guideCard = screen.getByText("Ingest + summarize a source").closest("section")
    expect(guideCard).toBeTruthy()
    if (!guideCard) return

    fireEvent.click(within(guideCard).getByRole("button", { name: "Ask docs mode" }))
    expect(onAskGuide).toHaveBeenCalledWith(
      "How do I ingest a URL or file and then summarize it quickly?"
    )

    fireEvent.click(within(guideCard).getByRole("button", { name: /Open/i }))
    expect(onOpenRoute).toHaveBeenCalledWith("/media")
  })

  it("uses store startTutorial when no external start callback is provided", () => {
    render(
      <QuickChatGuidesPanel
        onAskGuide={vi.fn()}
        onOpenRoute={vi.fn()}
        currentRoute="/prompts"
      />
    )

    const start = screen.getByRole("button", { name: "Start" })
    start.focus()
    fireEvent.click(start)
    expect(startTutorialMock).toHaveBeenCalledWith("prompts-basics", expect.objectContaining({ target: expect.any(HTMLElement), isCurrent: expect.any(Function) }))
  })

  it("shows locked advanced tutorials until prerequisites are completed", () => {
    render(
      <QuickChatGuidesPanel
        onAskGuide={vi.fn()}
        onOpenRoute={vi.fn()}
        currentRoute="/chat"
      />
    )

    const toolsLabel = screen.getByText("Tools & Attachments")
    const toolsRow = toolsLabel.closest("div")?.parentElement
    expect(toolsRow).toBeTruthy()
    if (!toolsRow) return

    expect(within(toolsRow).getByRole("button", { name: "Locked" })).toBeDisabled()
  })

  it("shows replay for completed basics and unlocks advanced tutorials", () => {
    tutorialStoreState.completedTutorials = ["playground-basics"]

    render(
      <QuickChatGuidesPanel
        onAskGuide={vi.fn()}
        onOpenRoute={vi.fn()}
        currentRoute="/chat"
      />
    )

    const chatBasicsLabel = screen.getByText("Chat Basics")
    const chatBasicsRow = chatBasicsLabel.closest("div")?.parentElement
    expect(chatBasicsRow).toBeTruthy()
    if (!chatBasicsRow) return
    expect(within(chatBasicsRow).getByRole("button", { name: "Replay" })).toBeInTheDocument()

    const toolsLabel = screen.getByText("Tools & Attachments")
    const toolsRow = toolsLabel.closest("div")?.parentElement
    expect(toolsRow).toBeTruthy()
    if (!toolsRow) return
    expect(within(toolsRow).queryByRole("button", { name: "Locked" })).toBeNull()
    expect(within(toolsRow).getByRole("button", { name: "Start" })).toBeInTheDocument()
  })
})

it("compact Guides retain workflow links but offer no Start without a runner", () => {
  const original = window.location.pathname
  window.history.replaceState(null, "", "/sidepanel.html")
  try {
    render(<QuickChatGuidesPanel currentRoute="/chat" onAskGuide={vi.fn()} onOpenRoute={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument()
    expect(screen.getByText("Workflow guide browser")).toBeVisible()
    expect(screen.getAllByRole("button", { name: /Open/i }).length).toBeGreaterThan(0)
  } finally { window.history.replaceState(null, "", original) }
})
