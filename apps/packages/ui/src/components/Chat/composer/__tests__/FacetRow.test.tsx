import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Facet, FacetRow } from "../shared/FacetRow"

describe("Facet", () => {
  it("renders key + value as a span when no onClick", () => {
    const { container } = render(<Facet fieldKey="src" value="irb-archive" />)
    expect(container.textContent).toContain("src")
    expect(container.textContent).toContain("irb-archive")
    expect(container.querySelector("button")).toBeNull()
  })

  it("renders as a button when onClick is provided", () => {
    const onClick = vi.fn()
    render(<Facet fieldKey="src" value="irb-archive" onClick={onClick} />)
    fireEvent.click(screen.getByRole("button"))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("sets aria-pressed when active", () => {
    render(<Facet value="v" onClick={() => {}} active />)
    expect(screen.getByRole("button").getAttribute("aria-pressed")).toBe("true")
  })

  it("applies primary text color when active", () => {
    const { container } = render(<Facet value="v" active />)
    const el = container.firstElementChild as HTMLElement
    expect(el.className).toContain("text-primary")
  })

  it("forwards aria-label for icon-only facets", () => {
    render(<Facet value="☼" onClick={() => {}} aria-label="Web search" />)
    expect(screen.getByRole("button", { name: "Web search" })).toBeTruthy()
  })
})

describe("FacetRow", () => {
  it("renders facets as a role=group with aria-label", () => {
    render(
      <FacetRow
        facets={[
          { id: "src", fieldKey: "src", value: "irb-archive" },
          { id: "mdl", fieldKey: "mdl", value: "haiku-4-5" },
        ]}
      />
    )
    expect(screen.getByRole("group", { name: /composer facets/i })).toBeTruthy()
  })

  it("renders each facet", () => {
    render(
      <FacetRow
        facets={[
          { id: "src", fieldKey: "src", value: "irb-archive" },
          { id: "mdl", fieldKey: "mdl", value: "haiku-4-5" },
        ]}
      />
    )
    expect(screen.getByText("irb-archive")).toBeTruthy()
    expect(screen.getByText("haiku-4-5")).toBeTruthy()
  })

  it("places the trailing slot at the end", () => {
    render(
      <FacetRow
        facets={[{ id: "src", fieldKey: "src", value: "irb" }]}
        trailing={<span>TRAIL</span>}
      />
    )
    expect(screen.getByText("TRAIL")).toBeTruthy()
  })

  it("honors a custom aria-label", () => {
    render(
      <FacetRow
        facets={[{ id: "src", fieldKey: "src", value: "irb" }]}
        aria-label="Chat context"
      />
    )
    expect(screen.getByRole("group", { name: "Chat context" })).toBeTruthy()
  })
})
