import React, { useState } from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { FollowUpInput } from "../FollowUpInput"

const state = {
  askFollowUp: vi.fn(),
  isSearching: false,
  startNewTopic: vi.fn(),
  createNewThread: vi.fn(),
  results: [{ id: "r1" }] as Array<{ id: string }>,
  answer: null as string | null,
  isMobile: false,
  setEvidenceRailOpen: (_open: boolean) => {},
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    askFollowUp: state.askFollowUp,
    isSearching: state.isSearching,
    startNewTopic: state.startNewTopic,
    createNewThread: state.createNewThread,
    results: state.results,
    answer: state.answer,
    setEvidenceRailOpen: state.setEvidenceRailOpen,
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => state.isMobile,
}))

describe("FollowUpInput accessibility", () => {
  function createDeferred<T>() {
    let resolve!: (value: T) => void
    let reject!: (reason?: unknown) => void
    const promise = new Promise<T>((res, rej) => {
      resolve = res
      reject = rej
    })
    return { promise, resolve, reject }
  }

  beforeEach(() => {
    vi.clearAllMocks()
    state.isSearching = false
    state.startNewTopic = vi.fn()
    state.results = [{ id: "r1" }]
    state.answer = null
    state.isMobile = false
  })

  it("provides an explicit accessible name for the follow-up input", () => {
    render(<FollowUpInput />)

    const input = screen.getByRole("textbox", { name: "Ask a follow-up question" })
    expect(input).toBeInTheDocument()
    expect(input).toHaveAttribute("maxlength", "20000")
    expect(
      screen.getByText(
        'Follow-up questions maintain context. Click "New Topic" to start fresh.'
      )
    ).toBeInTheDocument()
  })

  it("shows explicit New Topic text alongside the icon action", () => {
    render(<FollowUpInput />)
    expect(screen.getByRole("button", { name: "Start new topic" })).toHaveTextContent(
      "New Topic"
    )
  })

  it("switches to a recovery-oriented follow-up prompt when requested", () => {
    render(<FollowUpInput mode="recovery" />)

    expect(screen.getByText("Try a sharper follow-up")).toBeInTheDocument()
    expect(
      screen.getByText(/Ask for the missing detail, timeframe, or source you still need/i)
    ).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Ask a follow-up question" })).toHaveAttribute(
      "placeholder",
      "Ask a more specific follow-up..."
    )
  })

  it("routes the New Topic action through the fresh-topic lifecycle handler", () => {
    render(<FollowUpInput />)

    fireEvent.click(screen.getByRole("button", { name: "Start new topic" }))

    expect(state.startNewTopic).toHaveBeenCalledTimes(1)
    expect(state.createNewThread).not.toHaveBeenCalled()
  })

  it("prevents duplicate new-topic creation while the first request is still pending", () => {
    const pendingStart = createDeferred<void>()
    state.startNewTopic = vi.fn(() => pendingStart.promise)

    render(<FollowUpInput />)

    const newTopicButton = screen.getByRole("button", { name: "Start new topic" })
    fireEvent.click(newTopicButton)
    fireEvent.click(newTopicButton)

    expect(state.startNewTopic).toHaveBeenCalledTimes(1)
    expect(newTopicButton).toBeDisabled()
  })

  it("renders queued follow-up state while an initial search is in progress", () => {
    state.isSearching = true
    state.results = []
    state.answer = null

    render(<FollowUpInput />)

    const input = screen.getByRole("textbox", { name: "Ask a follow-up question" })
    expect(input).toBeDisabled()
    expect(input).toHaveAttribute("placeholder", "Current search in progress...")
    expect(
      screen.getByText(/follow-up input unlocks when the current search completes/i)
    ).toBeInTheDocument()
  })

  it("prevents duplicate follow-up submission while the first request is still pending", () => {
    const pendingFollowUp = createDeferred<void>()
    state.askFollowUp = vi.fn(() => pendingFollowUp.promise)

    render(<FollowUpInput />)

    const input = screen.getByRole("textbox", { name: "Ask a follow-up question" })
    fireEvent.change(input, { target: { value: "Compare the findings" } })

    const submitButton = screen.getByRole("button", {
      name: "Submit follow-up question",
    })
    fireEvent.click(submitButton)
    fireEvent.click(submitButton)

    expect(state.askFollowUp).toHaveBeenCalledTimes(1)
    expect(state.askFollowUp).toHaveBeenCalledWith("Compare the findings")
    expect(submitButton).toBeDisabled()
    expect(input).toBeDisabled()
  })

  it("offers distinct Evidence, Send and New Topic actions in the mobile composer", () => {
    state.isMobile = true
    function Composer() {
      const [open, setOpen] = useState(false)
      state.setEvidenceRailOpen = setOpen
      return <><FollowUpInput /><output aria-label="Evidence state">{open ? "Open" : "Closed"}</output></>
    }
    render(<Composer />)
    fireEvent.change(screen.getByRole("textbox", { name: "Ask a follow-up question" }), { target: { value: "My follow-up" } })
    expect(screen.getByRole("button", { name: "Submit follow-up question" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Start new topic" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Open evidence panel" }))
    expect(screen.getByLabelText("Evidence state")).toHaveTextContent("Open")
    expect(screen.getByRole("textbox", { name: "Ask a follow-up question" })).toHaveValue("My follow-up")
  })

  it("uses a sticky mobile layout with safe-area padding while keeping actions visible", () => {
    state.isMobile = true

    render(<FollowUpInput />)

    const stickyContainer = screen.getByTestId("knowledge-followup-sticky")
    expect(stickyContainer.className).toContain("fixed inset-x-0 bottom-0")
    expect(stickyContainer.className).toContain("env(safe-area-inset-bottom)")
    expect(screen.getByRole("button", { name: "Start new topic" })).toBeInTheDocument()
    expect(
      screen.getByRole("textbox", { name: "Ask a follow-up question" })
    ).toBeInTheDocument()
  })

  it("shows character-limit feedback near the max and warns at the cap", () => {
    render(<FollowUpInput />)

    const input = screen.getByRole("textbox", { name: "Ask a follow-up question" })

    fireEvent.change(input, { target: { value: "x".repeat(17000) } })
    expect(screen.getByText("17000/20000")).toBeInTheDocument()

    fireEvent.change(input, { target: { value: "x".repeat(21000) } })
    expect(screen.getByText(/20000\/20000/i)).toBeInTheDocument()
    expect(
      screen.getByText(/Max length reached\. Extra text will not be included\./i)
    ).toBeInTheDocument()
  })
})
