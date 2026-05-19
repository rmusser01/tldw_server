import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TerminalStackV1 } from "../variants/TerminalStackV1"

const baseProps = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
}

describe("TerminalStackV1 · slot overrides", () => {
  it("topSlot replaces the default sourceChip + topChips rendering", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        sourceChip={{ count: 14, label: "simple-source" }}
        topChips={[{ id: "x", label: "simple-chip" }]}
        topSlot={<div>CUSTOM_TOP</div>}
      />
    )
    expect(screen.getByText("CUSTOM_TOP")).toBeTruthy()
    expect(screen.queryByText("simple-source")).toBeNull()
    expect(screen.queryByText("simple-chip")).toBeNull()
  })

  it("falls back to the simple chips API when topSlot is not provided", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        sourceChip={{ count: 14, label: "simple-source" }}
      />
    )
    expect(screen.getByText("simple-source")).toBeTruthy()
  })

  it("bottomBarSlot replaces the default bar (chips, iconButtons, tokens, send)", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        bottomChips={[{ id: "m", label: "fallback-chip" }]}
        iconButtons={[{ id: "a", label: "Attach", icon: "⎙" }]}
        tokens={{ used: 10, max: 100 }}
        bottomBarSlot={<div>CUSTOM_BAR</div>}
      />
    )
    expect(screen.getByText("CUSTOM_BAR")).toBeTruthy()
    expect(screen.queryByText("fallback-chip")).toBeNull()
    expect(screen.queryByRole("button", { name: "Attach" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("sendSlot replaces only the built-in SendButton", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        bottomChips={[{ id: "m", label: "visible-chip" }]}
        sendSlot={<button>CUSTOM_SEND</button>}
      />
    )
    // Chip + iconButtons still render
    expect(screen.getByText("visible-chip")).toBeTruthy()
    // Custom send replaces the default
    expect(screen.getByRole("button", { name: "CUSTOM_SEND" })).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("sendSlot is ignored when bottomBarSlot is also set", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        bottomBarSlot={<div>WHOLE_BAR</div>}
        sendSlot={<button>SHOULD_NOT_APPEAR</button>}
      />
    )
    expect(screen.getByText("WHOLE_BAR")).toBeTruthy()
    expect(screen.queryByRole("button", { name: "SHOULD_NOT_APPEAR" })).toBeNull()
  })

  it("noticesSlot renders above the composer box", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        noticesSlot={<div role="alert">UPGRADE_NOTICE</div>}
      />
    )
    expect(screen.getByRole("alert")).toHaveTextContent("UPGRADE_NOTICE")
  })

  it("overlaysSlot renders inside the focus box", () => {
    const { container } = render(
      <TerminalStackV1
        {...baseProps}
        overlaysSlot={
          <div data-testid="slash-menu-overlay">slash-menu</div>
        }
      />
    )
    const overlay = container.querySelector(
      '[data-testid="slash-menu-overlay"]'
    )
    expect(overlay).toBeTruthy()
    // Should be inside the composer box (contains the textarea)
    const box = container.querySelector("[data-variant='v1'] > div")
    expect(box?.contains(overlay)).toBe(true)
  })

  it("omitting top-rail content entirely (no slot, no chips) skips the row", () => {
    const { container } = render(<TerminalStackV1 {...baseProps} />)
    expect(
      container.querySelector('[data-testid="v1-top-rail"]')
    ).toBeNull()
  })

  it("topSlot content WITHOUT simple chips still renders the row", () => {
    const { container } = render(
      <TerminalStackV1 {...baseProps} topSlot={<span>anything</span>} />
    )
    expect(
      container.querySelector('[data-testid="v1-top-rail"]')
    ).toBeTruthy()
  })

  it("textareaSlot replaces the built-in `>_` + textarea", () => {
    render(
      <TerminalStackV1
        {...baseProps}
        textareaSlot={<textarea aria-label="Custom Textarea" />}
      />
    )
    expect(
      screen.getByRole("textbox", { name: "Custom Textarea" })
    ).toBeTruthy()
    expect(screen.queryByRole("textbox", { name: "Message" })).toBeNull()
  })

  it("textareaSlot drops the built-in caret column too", () => {
    const { container } = render(
      <TerminalStackV1
        {...baseProps}
        textareaSlot={<div data-testid="custom-input">x</div>}
      />
    )
    expect(container.querySelector("[data-testid='custom-input']")).toBeTruthy()
    // The "&gt;_" caret span is gone
    expect(container.textContent).not.toContain(">_")
  })
})
