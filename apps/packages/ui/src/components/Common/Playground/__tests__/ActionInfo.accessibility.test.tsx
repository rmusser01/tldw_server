// @vitest-environment jsdom
import { act, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { formatElapsedWait, LoadingStatus } from "../ActionInfo"

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

const liveText = () =>
  [...document.querySelectorAll("[aria-live]")].map((node) => node.textContent)

describe("LoadingStatus accessibility announcements", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("announces the start and completion of a response", () => {
    const { rerender } = render(<LoadingStatus isStreaming />)
    expect(screen.getByText("Generating response... started")).toBeInTheDocument()

    rerender(<LoadingStatus isStreaming={false} isProcessing={false} />)
    expect(screen.getByText("Response complete")).toBeInTheDocument()
  })

  it("CM-06: shows how long the wait has been and names the model it waits for", () => {
    render(<LoadingStatus isStreaming modelName="gemma3:1b" waitingForReply />)
    expect(screen.getByText("Waiting for gemma3:1b...")).toBeInTheDocument()
    expect(screen.queryByTestId("chat-loading-elapsed")).toBeNull()

    act(() => {
      vi.advanceTimersByTime(12_000)
    })
    const elapsed = screen.getByTestId("chat-loading-elapsed")
    expect(elapsed).toHaveTextContent("0:12")
    // The ticking timer is visual only; it never enters a live region.
    expect(elapsed).toHaveAttribute("aria-hidden", "true")
    expect(screen.getByTestId("chat-loading-status")).not.toHaveAttribute("aria-live")
  })

  it("CM-06: announces a long wait in plain language at about 10 s and 30 s, not every 5 s", () => {
    render(<LoadingStatus isStreaming modelName="gemma3:1b" waitingForReply />)
    act(() => {
      vi.advanceTimersByTime(5_000)
    })
    expect(liveText().join(" ")).not.toMatch(/checkpoint|Still/)

    act(() => {
      vi.advanceTimersByTime(5_000)
    })
    expect(liveText()).toContain("Still waiting for gemma3:1b (10 seconds).")

    act(() => {
      vi.advanceTimersByTime(20_000)
    })
    expect(liveText()).toContain("Still waiting for gemma3:1b (30 seconds).")

    act(() => {
      vi.advanceTimersByTime(60_000)
    })
    // No further announcements after 30 s.
    expect(liveText()).toContain("Still waiting for gemma3:1b (30 seconds).")
  })

  it("keeps the generating label once the reply has started", () => {
    render(<LoadingStatus isStreaming modelName="gemma3:1b" waitingForReply={false} />)
    expect(screen.getByText("Generating response...")).toBeInTheDocument()
    act(() => {
      vi.advanceTimersByTime(10_000)
    })
    expect(liveText()).toContain("Still writing the reply (10 seconds).")
  })

  it("formats elapsed waits as minutes and seconds", () => {
    expect(formatElapsedWait(0)).toBe("0:00")
    expect(formatElapsedWait(12_400)).toBe("0:12")
    expect(formatElapsedWait(125_000)).toBe("2:05")
  })
})
