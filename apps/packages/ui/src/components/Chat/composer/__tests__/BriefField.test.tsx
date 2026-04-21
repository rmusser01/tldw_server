import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { BriefField } from "../shared/BriefField"

describe("BriefField", () => {
  it("renders key + value at desktop density", () => {
    const { container } = render(
      <BriefField fieldKey="src" value="irb-archive · 14" />
    )
    expect(container.textContent).toContain("src")
    expect(container.textContent).toContain("irb-archive")
  })

  it("hides the key when hideKey=true", () => {
    const { container } = render(
      <BriefField fieldKey="src" value="irb-archive" hideKey />
    )
    const text = container.textContent ?? ""
    expect(text).toContain("irb-archive")
    expect(text).not.toContain("src")
  })

  it("renders as <span> when no onClick", () => {
    const { container } = render(<BriefField value="v" />)
    expect(container.querySelector("button")).toBeNull()
  })

  it("renders as <button> when onClick is provided", () => {
    const onClick = vi.fn()
    render(<BriefField value="irb-archive" onClick={onClick} />)
    fireEvent.click(screen.getByRole("button"))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("sets aria-pressed on the button when active", () => {
    render(<BriefField value="v" onClick={() => {}} active />)
    const btn = screen.getByRole("button")
    expect(btn.getAttribute("aria-pressed")).toBe("true")
  })

  it("applies primary tint classes when active", () => {
    const { container } = render(<BriefField value="v" active />)
    const el = container.firstElementChild as HTMLElement
    expect(el.className).toContain("border-primary/40")
    expect(el.className).toContain("bg-primary/")
  })

  it("forwards aria-label to the element", () => {
    render(
      <BriefField
        value="☼"
        onClick={() => {}}
        aria-label="Toggle web search"
      />
    )
    expect(screen.getByRole("button", { name: "Toggle web search" })).toBeTruthy()
  })
})
