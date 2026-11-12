import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TagSelector } from "../TagSelector"

vi.mock("@/store/collections", () => ({
  useCollectionsStore: (selector: (state: { availableTags: string[] }) => unknown) =>
    selector({ availableTags: [] })
}))

const enterTag = (value: string) => {
  fireEvent.click(screen.getByRole("button", { name: /add tag/i }))
  const input = screen.getByPlaceholderText("Add tag...")
  fireEvent.change(input, { target: { value } })
  fireEvent.keyDown(input, { key: "Enter" })
}

describe("Collections TagSelector", () => {
  it("keeps the typed casing of a new tag", () => {
    const onChange = vi.fn()
    render(<TagSelector tags={[]} onChange={onChange} />)

    enterTag("  Machine Learning ")

    expect(onChange).toHaveBeenCalledWith(["Machine Learning"])
  })

  it("does not add a case variant of a tag it already has", () => {
    const onChange = vi.fn()
    render(<TagSelector tags={["foo"]} onChange={onChange} />)

    enterTag("FOO")

    expect(onChange).not.toHaveBeenCalled()
  })

  it("hides suggestions that are case variants of chosen tags", () => {
    render(<TagSelector tags={["foo"]} onChange={vi.fn()} suggestions={["Foo", "Food"]} />)

    fireEvent.click(screen.getByRole("button", { name: /add tag/i }))
    fireEvent.change(screen.getByPlaceholderText("Add tag..."), { target: { value: "fo" } })

    expect(screen.queryByRole("button", { name: "Foo" })).toBeNull()
    expect(screen.getByRole("button", { name: "Food" })).toBeTruthy()
  })
})
