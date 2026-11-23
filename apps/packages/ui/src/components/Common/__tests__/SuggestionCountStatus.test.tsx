// @vitest-environment jsdom
import React from "react"
import { act, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { SUGGESTION_COUNT_DELAY_MS, SuggestionCountStatus } from "../SuggestionCountStatus"

afterEach(() => {
  vi.useRealTimers()
})

// AX-12 review: typing changes the count on every key; only the count the
// user settles on is announced.
describe("SuggestionCountStatus", () => {
  it("announces the count once typing settles, not on every change", () => {
    vi.useFakeTimers()
    const view = render(<SuggestionCountStatus open text="9 commands" />)
    const region = screen.getByTestId("suggestion-count-status")
    expect(region).toHaveAttribute("aria-live", "polite")
    act(() => vi.advanceTimersByTime(SUGGESTION_COUNT_DELAY_MS / 2))
    view.rerender(<SuggestionCountStatus open text="3 commands" />)
    act(() => vi.advanceTimersByTime(SUGGESTION_COUNT_DELAY_MS / 2))
    view.rerender(<SuggestionCountStatus open text="1 command" />)
    expect(region).toBeEmptyDOMElement()
    act(() => vi.advanceTimersByTime(SUGGESTION_COUNT_DELAY_MS))
    expect(region).toHaveTextContent("1 command")
  })

  it("goes quiet as soon as the menu closes", () => {
    vi.useFakeTimers()
    const view = render(<SuggestionCountStatus open text="2 commands" />)
    act(() => vi.advanceTimersByTime(SUGGESTION_COUNT_DELAY_MS))
    view.rerender(<SuggestionCountStatus open={false} text="2 commands" />)
    expect(screen.getByTestId("suggestion-count-status")).toBeEmptyDOMElement()
  })
})
