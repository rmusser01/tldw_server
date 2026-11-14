// @vitest-environment jsdom
import { act, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { LoadingStatus } from "../ActionInfo"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string, options?: Record<string, unknown>) => {
      const template =
        typeof defaultValue === "string" && defaultValue.length > 0
          ? defaultValue
          : key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

describe("LoadingStatus accessibility announcements", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("announces the start and completion of a reply", () => {
    const { rerender } = render(<LoadingStatus isStreaming />)
    expect(screen.getByText("Generating response... started")).toBeInTheDocument()

    rerender(<LoadingStatus isStreaming={false} isProcessing={false} />)
    expect(screen.getByText("Response complete")).toBeInTheDocument()
  })
})

// CM-06: a long wait is announced plainly, twice, and shows how long it has been.
describe("LoadingStatus during a long wait (CM-06)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  const advance = (ms: number) =>
    act(() => {
      vi.advanceTimersByTime(ms)
    })

  it("announces at about 10 and 30 seconds, naming the model, and never counts checkpoints", () => {
    render(<LoadingStatus isStreaming modelLabel="mock-gpt-4o (Custom OpenAI API)" />)

    advance(9_000)
    expect(screen.queryByText(/Still working/)).toBeNull()
    advance(1_000)
    expect(
      screen.getByText("Still working on the reply from mock-gpt-4o (Custom OpenAI API).")
    ).toBeInTheDocument()

    advance(20_000)
    expect(
      screen.getByText(
        "mock-gpt-4o (Custom OpenAI API) is taking a while. You can stop the reply and try again."
      )
    ).toBeInTheDocument()

    advance(60_000)
    expect(screen.queryByText(/checkpoint/i)).toBeNull()
  })

  it("speaks of the reply when the model is unknown", () => {
    render(<LoadingStatus isProcessing />)
    advance(10_000)
    expect(screen.getByText("Still working on the reply.")).toBeInTheDocument()
    advance(20_000)
    expect(
      screen.getByText("This reply is taking a while. You can stop it and try again.")
    ).toBeInTheDocument()
  })

  it("doesn't suggest stopping a reply that is already streaming text", () => {
    render(<LoadingStatus isStreaming replyStarted modelLabel="mock-gpt-4o" />)
    advance(30_000)
    expect(screen.queryByText(/taking a while/)).toBeNull()
  })

  it("still announces a long wait if a timer tick is skipped", () => {
    render(<LoadingStatus isStreaming modelLabel="mock-gpt-4o" />)
    advance(9_000)
    // Jump past the 10 s mark in one tick, as a blocked main thread would.
    vi.setSystemTime(Date.now() + 3_000)
    advance(1_000)
    expect(screen.getByText("Still working on the reply from mock-gpt-4o.")).toBeInTheDocument()
  })

  it("shows the elapsed time on the status pill after five seconds", () => {
    render(<LoadingStatus isStreaming modelLabel="mock-gpt-4o" />)
    const pill = () => screen.getByTestId("chat-loading-status")

    advance(4_000)
    expect(pill()).not.toHaveTextContent(/\d:\d\d/)
    advance(8_000)
    expect(pill()).toHaveTextContent("0:12")
    advance(60_000)
    expect(pill()).toHaveTextContent("1:12")
  })

  it("starts counting again for the next reply", () => {
    const { rerender } = render(<LoadingStatus isStreaming />)
    advance(20_000)
    rerender(<LoadingStatus isStreaming={false} />)
    rerender(<LoadingStatus isStreaming />)
    advance(6_000)
    expect(screen.getByTestId("chat-loading-status")).toHaveTextContent("0:06")
  })
})
