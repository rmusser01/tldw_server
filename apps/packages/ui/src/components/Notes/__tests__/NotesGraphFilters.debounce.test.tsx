/**
 * NL-11 review: a native select fires `change` on every arrow key, so the
 * graph's tag filter waits for the reader to settle before it refetches.
 */
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphFilters from "../NotesGraphFilters"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? _key
  })
}))

describe("NotesGraphFilters tag debounce", () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it("applies only the tag the reader settles on", () => {
    const onTagChange = vi.fn()
    render(
      <NotesGraphFilters
        tagOptions={["a", "b", "c"]}
        tag={null}
        linkedOnly={false}
        onTagChange={onTagChange}
        onLinkedOnlyChange={vi.fn()}
      />
    )
    const select = screen.getByRole("combobox", { name: "Tag" })

    fireEvent.change(select, { target: { value: "a" } })
    fireEvent.change(select, { target: { value: "b" } })
    fireEvent.change(select, { target: { value: "c" } })
    expect(onTagChange).not.toHaveBeenCalled()
    expect(select).toHaveValue("c")

    act(() => vi.advanceTimersByTime(400))
    expect(onTagChange).toHaveBeenCalledTimes(1)
    expect(onTagChange).toHaveBeenCalledWith("c")
  })
})
