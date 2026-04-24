import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { IconButton } from "../shared/IconButton"

describe("IconButton", () => {
  it("renders with the required aria-label and title", () => {
    render(<IconButton label="Attach file">⎙</IconButton>)
    const btn = screen.getByRole("button", { name: "Attach file" })
    expect(btn.getAttribute("title")).toBe("Attach file")
  })

  it("marks the icon decoration aria-hidden", () => {
    const { container } = render(<IconButton label="Voice">◉</IconButton>)
    const icon = container.querySelector("[aria-hidden='true']")
    expect(icon?.textContent).toBe("◉")
  })

  it("fires onClick on click", () => {
    const onClick = vi.fn()
    render(
      <IconButton label="Voice" onClick={onClick}>
        ◉
      </IconButton>
    )
    fireEvent.click(screen.getByRole("button", { name: "Voice" }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("does not expose toggle semantics unless pressed is provided", () => {
    render(
      <IconButton label="Voice" active>
        ◉
      </IconButton>
    )
    const btn = screen.getByRole("button", { name: "Voice" })
    expect(btn.hasAttribute("aria-pressed")).toBe(false)
  })

  it("sets aria-pressed when pressed=true", () => {
    render(
      <IconButton label="Voice" pressed>
        ◉
      </IconButton>
    )
    const btn = screen.getByRole("button", { name: "Voice" })
    expect(btn.getAttribute("aria-pressed")).toBe("true")
  })

  it("sets aria-pressed to false when pressed=false", () => {
    render(
      <IconButton label="Voice" pressed={false}>
        ◉
      </IconButton>
    )
    const btn = screen.getByRole("button", { name: "Voice" })
    expect(btn.getAttribute("aria-pressed")).toBe("false")
  })

  it("applies the primary tint when active", () => {
    const { container } = render(
      <IconButton label="Voice" active>
        ◉
      </IconButton>
    )
    const btn = container.querySelector("button")
    expect(btn?.className).toContain("text-primary")
    expect(btn?.className).toContain("bg-primary/10")
  })

  it("disables clicks when disabled=true", () => {
    const onClick = vi.fn()
    render(
      <IconButton label="Voice" onClick={onClick} disabled>
        ◉
      </IconButton>
    )
    const btn = screen.getByRole("button", { name: "Voice" }) as HTMLButtonElement
    expect(btn.disabled).toBe(true)
    fireEvent.click(btn)
    expect(onClick).not.toHaveBeenCalled()
  })

  it("uses compact size classes at density='compact'", () => {
    const { container } = render(
      <IconButton label="x" density="compact">
        ◉
      </IconButton>
    )
    const btn = container.querySelector("button")
    expect(btn?.className).toContain("w-6")
    expect(btn?.className).toContain("h-6")
  })

  it("uses larger size classes at density='desktop'", () => {
    const { container } = render(
      <IconButton label="x" density="desktop">
        ◉
      </IconButton>
    )
    const btn = container.querySelector("button")
    expect(btn?.className).toContain("w-7")
    expect(btn?.className).toContain("h-7")
  })
})
