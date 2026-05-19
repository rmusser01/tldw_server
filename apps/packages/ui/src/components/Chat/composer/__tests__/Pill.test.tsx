import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Pill } from "../shared/Pill"

describe("Pill", () => {
  it("renders as <span> when no onClick is provided", () => {
    const { container } = render(<Pill>hello</Pill>)
    expect(container.querySelector("span")?.textContent).toBe("hello")
    expect(container.querySelector("button")).toBeNull()
  })

  it("renders as <button> when onClick is provided (default `as`)", () => {
    const onClick = vi.fn()
    render(<Pill onClick={onClick}>click me</Pill>)
    const btn = screen.getByRole("button", { name: "click me" })
    fireEvent.click(btn)
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("applies the `on` variant classes", () => {
    const { container } = render(<Pill variant="on">on</Pill>)
    const el = container.firstElementChild as HTMLElement
    expect(el.className).toContain("text-primary")
    expect(el.className).toContain("bg-primary/10")
  })

  it("applies the `accent` variant classes", () => {
    const { container } = render(<Pill variant="accent">accent</Pill>)
    const el = container.firstElementChild as HTMLElement
    expect(el.className).toContain("text-accent")
  })

  it("applies the `default` variant classes", () => {
    const { container } = render(<Pill variant="default">default</Pill>)
    const el = container.firstElementChild as HTMLElement
    expect(el.className).toContain("text-text-muted")
    expect(el.className).toContain("bg-surface2")
  })

  it("forwards aria-label on the button element", () => {
    render(
      <Pill onClick={() => {}} aria-label="Toggle web search">
        ☼
      </Pill>
    )
    expect(screen.getByRole("button", { name: "Toggle web search" })).toBeTruthy()
  })

  it("disables the button when disabled=true", () => {
    const onClick = vi.fn()
    render(
      <Pill onClick={onClick} disabled>
        nope
      </Pill>
    )
    const btn = screen.getByRole("button", { name: "nope" }) as HTMLButtonElement
    expect(btn.disabled).toBe(true)
    fireEvent.click(btn)
    expect(onClick).not.toHaveBeenCalled()
  })

  it("honors onClick when `as='span'` is explicitly passed (a11y fallback)", () => {
    const onClick = vi.fn()
    render(
      <Pill as="span" onClick={onClick} aria-label="click me">
        x
      </Pill>
    )
    const el = screen.getByRole("button", { name: "click me" })
    fireEvent.click(el)
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("`as='span'` + onClick activates on Enter key", () => {
    const onClick = vi.fn()
    render(
      <Pill as="span" onClick={onClick} aria-label="click me">
        x
      </Pill>
    )
    fireEvent.keyDown(screen.getByRole("button"), { key: "Enter" })
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("`as='span'` + onClick activates on Space key", () => {
    const onClick = vi.fn()
    render(
      <Pill as="span" onClick={onClick} aria-label="click me">
        x
      </Pill>
    )
    fireEvent.keyDown(screen.getByRole("button"), { key: " " })
    expect(onClick).toHaveBeenCalledOnce()
  })
})
