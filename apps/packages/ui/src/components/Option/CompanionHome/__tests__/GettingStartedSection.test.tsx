// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOpts?: string | Record<string, unknown>, opts?: Record<string, unknown>) => {
      const template = typeof fallbackOrOpts === "string" ? fallbackOrOpts : key
      const vars = opts ?? (typeof fallbackOrOpts === "object" ? fallbackOrOpts : {})
      return template.replace(/\{\{(\w+)\}\}/g, (_, k) => String((vars as Record<string, unknown>)[k] ?? k))
    }
  })
}))

const mockMissionCards = {
  gettingStartedCards: [] as any[],
  whatsNextCard: null as any,
  completedCount: 0,
  totalCount: 0,
  allComplete: false
}

vi.mock("../hooks/useMissionCards", () => ({
  useMissionCards: () => mockMissionCards
}))

import { GettingStartedSection } from "../GettingStartedSection"

describe("GettingStartedSection", () => {
  beforeEach(() => {
    mockMissionCards.gettingStartedCards = []
    mockMissionCards.allComplete = false
    mockMissionCards.completedCount = 0
    mockMissionCards.totalCount = 0
  })

  it("renders nothing when all complete", () => {
    mockMissionCards.gettingStartedCards = [
      { id: "test-done", title: "Completed step", description: "Already done", icon: () => null, href: "/done", isCompleted: true, category: "getting-started", priority: 1, persona: "all", prerequisiteMilestones: ["first_connection"] }
    ]
    mockMissionCards.allComplete = true
    const { container } = render(<MemoryRouter><GettingStartedSection /></MemoryRouter>)
    expect(container).toBeEmptyDOMElement()
  })

  it("renders nothing when no cards", () => {
    const { container } = render(<MemoryRouter><GettingStartedSection /></MemoryRouter>)
    expect(container).toBeEmptyDOMElement()
  })

  it("renders cards when available", () => {
    mockMissionCards.gettingStartedCards = [
      { id: "test-1", title: "First step", description: "Do this", icon: () => null, href: "/test", isCompleted: false, category: "getting-started", priority: 1, persona: "all", prerequisiteMilestones: [] }
    ]
    mockMissionCards.totalCount = 1
    render(<MemoryRouter><GettingStartedSection /></MemoryRouter>)
    expect(screen.getByTestId("getting-started-section")).toBeInTheDocument()
    expect(screen.getByText("First step")).toBeInTheDocument()
    expect(screen.getByText("0 of 1 complete")).toBeInTheDocument()
  })

  it("shows completed count", () => {
    mockMissionCards.gettingStartedCards = [
      { id: "t1", title: "Done", description: "x", icon: () => null, href: "/", isCompleted: true, category: "getting-started", priority: 1, persona: "all", prerequisiteMilestones: [] },
      { id: "t2", title: "Pending", description: "x", icon: () => null, href: "/", isCompleted: false, category: "getting-started", priority: 2, persona: "all", prerequisiteMilestones: [] }
    ]
    mockMissionCards.completedCount = 1
    mockMissionCards.totalCount = 2
    render(<MemoryRouter><GettingStartedSection /></MemoryRouter>)
    expect(screen.getByText("1 of 2 complete")).toBeInTheDocument()
  })
})
