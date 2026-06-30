import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SplitBriefV3 } from "../variants/SplitBriefV3"

const base = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
  briefSections: [
    {
      id: "b",
      fields: [{ id: "src", fieldKey: "src", value: "simple-source" }],
    },
  ],
}

describe("SplitBriefV3 · slot overrides", () => {
  it("briefSlot replaces the default brief panel rendering", () => {
    render(
      <SplitBriefV3 {...base} briefSlot={<div>CUSTOM_BRIEF</div>} />
    )
    expect(screen.getByText("CUSTOM_BRIEF")).toBeTruthy()
    expect(screen.queryByText("simple-source")).toBeNull()
  })

  it("falls back to briefSections when briefSlot is not provided", () => {
    render(<SplitBriefV3 {...base} />)
    expect(screen.getByText("simple-source")).toBeTruthy()
  })

  it("bottomBarSlot replaces the entire bottom bar", () => {
    render(
      <SplitBriefV3
        {...base}
        iconButtons={[{ id: "a", label: "Attach", icon: "⎙" }]}
        tokens={{ used: 10, max: 100 }}
        costLabel="$0.001"
        bottomBarSlot={<div>CUSTOM_BAR</div>}
      />
    )
    expect(screen.getByText("CUSTOM_BAR")).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Attach" })).toBeNull()
    expect(screen.queryByText(/0\.001/)).toBeNull()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("sendSlot replaces only the SendButton", () => {
    render(
      <SplitBriefV3
        {...base}
        tokens={{ used: 10, max: 100 }}
        sendSlot={<button>CUSTOM_SEND</button>}
      />
    )
    // Token meter still renders
    expect(screen.getByLabelText(/of 100 tokens used/i)).toBeTruthy()
    expect(screen.getByRole("button", { name: "CUSTOM_SEND" })).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("noticesSlot renders above the composer box", () => {
    render(
      <SplitBriefV3
        {...base}
        noticesSlot={<div role="alert">WARNING</div>}
      />
    )
    expect(screen.getByRole("alert")).toHaveTextContent("WARNING")
  })

  it("overlaysSlot renders inside the focus box", () => {
    const { container } = render(
      <SplitBriefV3
        {...base}
        overlaysSlot={<div data-testid="mentions-menu">mentions</div>}
      />
    )
    expect(container.querySelector('[data-testid="mentions-menu"]')).toBeTruthy()
  })

  it("briefSlot + bottomBarSlot compose together", () => {
    render(
      <SplitBriefV3
        {...base}
        briefSlot={<div>MY_BRIEF</div>}
        bottomBarSlot={<div>MY_BAR</div>}
      />
    )
    expect(screen.getByText("MY_BRIEF")).toBeTruthy()
    expect(screen.getByText("MY_BAR")).toBeTruthy()
  })

  it("textareaSlot replaces the built-in question textarea", () => {
    render(
      <SplitBriefV3
        {...base}
        textareaSlot={<textarea aria-label="Custom Question" />}
      />
    )
    expect(
      screen.getByRole("textbox", { name: "Custom Question" })
    ).toBeTruthy()
    expect(screen.queryByRole("textbox", { name: "Question" })).toBeNull()
  })
})
