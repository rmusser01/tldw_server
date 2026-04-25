import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SourceChip } from "../shared/SourceChip"

describe("SourceChip", () => {
  it("renders the count badge and label", () => {
    render(<SourceChip count={14} label="irb-archive" />)
    expect(screen.getByText("14")).toBeTruthy()
    expect(screen.getByText("irb-archive")).toBeTruthy()
  })

  it("renders the default glyph '▣' when no glyph prop is passed", () => {
    const { container } = render(
      <SourceChip count={14} label="irb-archive" />
    )
    expect(container.textContent).toContain("▣")
  })

  it("renders a custom glyph", () => {
    const { container } = render(
      <SourceChip count={3} label="web" glyph="☼" />
    )
    expect(container.textContent).toContain("☼")
  })

  it("marks the glyph decoration aria-hidden", () => {
    const { container } = render(
      <SourceChip count={3} label="web" glyph="☼" />
    )
    const ariaHiddenEls = container.querySelectorAll("[aria-hidden='true']")
    const hasGlyph = Array.from(ariaHiddenEls).some((el) =>
      el.textContent?.includes("☼")
    )
    expect(hasGlyph).toBe(true)
  })

  it("renders as a <span> when no onClick is provided", () => {
    const { container } = render(
      <SourceChip count={14} label="irb-archive" />
    )
    expect(container.querySelector("button")).toBeNull()
  })

  it("renders as a <button> when onClick is provided", () => {
    const onClick = vi.fn()
    render(
      <SourceChip count={14} label="irb-archive" onClick={onClick} />
    )
    const btn = screen.getByRole("button")
    fireEvent.click(btn)
    expect(onClick).toHaveBeenCalledOnce()
  })
})
