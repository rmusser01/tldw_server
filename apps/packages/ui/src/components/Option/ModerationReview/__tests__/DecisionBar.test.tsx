import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { DecisionBar } from "../DecisionBar"

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("DecisionBar undo expiry", () => {
  it("expires without another render and resets when a new undo deadline arrives", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-09-10T12:00:00Z"))
    const props = { undoToken: "undo-1", onDecision: vi.fn(), onUndo: vi.fn() }
    const { rerender, unmount } = render(
      <DecisionBar {...props} undoExpiresAt="2026-09-10T12:00:01Z" />
    )
    expect(screen.getByRole("button", { name: "Undo decision" })).toBeEnabled()

    act(() => vi.advanceTimersByTime(1001))
    expect(screen.getByRole("button", { name: "Undo expired" })).toBeDisabled()

    rerender(
      <DecisionBar
        {...props}
        undoToken="undo-2"
        undoExpiresAt="2026-09-10T12:00:05Z"
      />
    )
    expect(screen.getByRole("button", { name: "Undo decision" })).toBeEnabled()
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })

  it("treats an already elapsed deadline as expired on first display", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-09-10T12:00:10Z"))
    render(
      <DecisionBar
        undoToken="undo-1"
        undoExpiresAt="2026-09-10T12:00:01Z"
        onDecision={vi.fn()}
        onUndo={vi.fn()}
      />
    )
    expect(screen.getByRole("button", { name: "Undo expired" })).toBeDisabled()
  })
})
