import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SendButton } from "../shared/SendButton"

describe("SendButton", () => {
  it("renders 'Send' label by default", () => {
    render(<SendButton />)
    expect(screen.getByRole("button", { name: "Send" })).toBeTruthy()
  })

  it("renders a custom label", () => {
    render(<SendButton label="Dispatch" />)
    expect(screen.getByRole("button", { name: "Dispatch" })).toBeTruthy()
  })

  it("swaps to 'Stop' label while streaming", () => {
    render(<SendButton stopping />)
    expect(screen.getByRole("button", { name: "Stop" })).toBeTruthy()
  })

  it("renders a custom stopLabel while streaming", () => {
    render(<SendButton stopping stopLabel="Cancel" />)
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy()
  })

  it("fires onClick on click", () => {
    const onClick = vi.fn()
    render(<SendButton onClick={onClick} />)
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("is disabled when disabled=true", () => {
    const onClick = vi.fn()
    render(<SendButton onClick={onClick} disabled />)
    const btn = screen.getByRole("button", { name: "Send" }) as HTMLButtonElement
    expect(btn.disabled).toBe(true)
    fireEvent.click(btn)
    expect(onClick).not.toHaveBeenCalled()
  })

  it("shows only the ↩ glyph at compact density", () => {
    const { container } = render(<SendButton density="compact" />)
    const btn = container.querySelector("button") as HTMLButtonElement
    expect(btn.textContent).toBe("↩")
    // The kbd hint should NOT be present
    expect(btn.textContent).not.toContain("⌘")
  })

  it("keeps the Send label and ⌘↩ kbd hint at desktop density", () => {
    const { container } = render(<SendButton density="desktop" />)
    const btn = container.querySelector("button") as HTMLButtonElement
    expect(btn.textContent).toContain("Send")
    expect(btn.textContent).toContain("⌘↩")
  })

  it("compact button meets WCAG 2.5.5 min tap target (≥24×24)", () => {
    const { container } = render(<SendButton density="compact" />)
    const btn = container.querySelector("button") as HTMLButtonElement
    expect(btn.className).toContain("min-w-[28px]")
    expect(btn.className).toContain("min-h-[28px]")
  })

  it("applies danger color classes while stopping", () => {
    const { container } = render(<SendButton stopping />)
    const btn = container.querySelector("button") as HTMLButtonElement
    expect(btn.className).toContain("bg-danger")
  })

  it("applies primary color classes when idle", () => {
    const { container } = render(<SendButton />)
    const btn = container.querySelector("button") as HTMLButtonElement
    expect(btn.className).toContain("bg-primary")
  })
})
