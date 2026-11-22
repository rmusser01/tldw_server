import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesTrashSearch from "../NotesTrashSearch"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? _key
  })
}))

const Harness = () => {
  const [value, setValue] = React.useState("")
  return (
    <>
      <NotesTrashSearch value={value} onSearch={setValue} />
      <span data-testid="applied">{value}</span>
    </>
  )
}

const input = () => screen.getByTestId("notes-trash-search") as HTMLInputElement

describe("NotesTrashSearch (NL-05)", () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it("keeps a trailing space typed before a pause, so the next word joins it", async () => {
    render(<Harness />)
    fireEvent.change(input(), { target: { value: "budget " } })
    await act(async () => {
      vi.advanceTimersByTime(400)
    })
    expect(screen.getByTestId("applied")).toHaveTextContent("budget")
    expect(input().value).toBe("budget ")

    fireEvent.change(input(), { target: { value: "budget report" } })
    await act(async () => {
      vi.advanceTimersByTime(400)
    })
    expect(input().value).toBe("budget report")
    expect(screen.getByTestId("applied")).toHaveTextContent("budget report")
  })

  it("clears the box when the search is cleared from outside", async () => {
    const { rerender } = render(<NotesTrashSearch value="draft" onSearch={() => undefined} />)
    expect(input().value).toBe("draft")
    rerender(<NotesTrashSearch value="" onSearch={() => undefined} />)
    expect(input().value).toBe("")
  })

  it("limits the search to the 200 characters the server accepts", () => {
    render(<Harness />)
    expect(input()).toHaveAttribute("maxLength", "200")
  })
})
