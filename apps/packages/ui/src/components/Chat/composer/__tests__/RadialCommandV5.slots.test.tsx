import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { RadialCommandV5 } from "../variants/RadialCommandV5"

const base = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
}

describe("RadialCommandV5 · slot overrides", () => {
  it("facetsSlot replaces the default facet row", () => {
    render(
      <RadialCommandV5
        {...base}
        facets={[{ id: "f", fieldKey: "src", value: "simple-facet" }]}
        tokens={{ used: 10, max: 100 }}
        facetsSlot={<div>CUSTOM_FACETS</div>}
      />
    )
    expect(screen.getByText("CUSTOM_FACETS")).toBeTruthy()
    expect(screen.queryByText("simple-facet")).toBeNull()
    expect(screen.queryByLabelText(/tokens used/i)).toBeNull()
  })

  it("falls back to facets + tokens simple API when no facetsSlot", () => {
    render(
      <RadialCommandV5
        {...base}
        facets={[{ id: "f", fieldKey: "src", value: "simple-facet" }]}
        tokens={{ used: 10, max: 100 }}
      />
    )
    expect(screen.getByText("simple-facet")).toBeTruthy()
    expect(screen.getByLabelText(/of 100 tokens used/i)).toBeTruthy()
  })

  it("omits facet row entirely when no facets, no tokens, no slot", () => {
    render(<RadialCommandV5 {...base} />)
    expect(
      screen.queryByRole("group", { name: /composer facets/i })
    ).toBeNull()
  })

  it("inlineSlot replaces the default ⌘K + iconButtons region", () => {
    render(
      <RadialCommandV5
        {...base}
        iconButtons={[{ id: "a", label: "Attach", icon: "⎙" }]}
        onPaletteTrigger={() => {}}
        inlineSlot={<div>CUSTOM_INLINE</div>}
      />
    )
    expect(screen.getByText("CUSTOM_INLINE")).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Attach" })).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Open command palette" })
    ).toBeNull()
    // Send button is NOT replaced — inlineSlot doesn't cover it
    expect(screen.getByRole("button", { name: "Send" })).toBeTruthy()
  })

  it("sendSlot replaces only the round SendButton", () => {
    render(
      <RadialCommandV5
        {...base}
        iconButtons={[{ id: "a", label: "Attach", icon: "⎙" }]}
        sendSlot={<button>CUSTOM_SEND</button>}
      />
    )
    expect(screen.getByRole("button", { name: "Attach" })).toBeTruthy()
    expect(screen.getByRole("button", { name: "CUSTOM_SEND" })).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull()
  })

  it("noticesSlot renders above the facet row", () => {
    render(
      <RadialCommandV5
        {...base}
        noticesSlot={<div role="alert">NOTICE_V5</div>}
      />
    )
    expect(screen.getByRole("alert")).toHaveTextContent("NOTICE_V5")
  })

  it("all four slots compose together", () => {
    render(
      <RadialCommandV5
        {...base}
        facetsSlot={<div>F</div>}
        textareaSlot={<textarea aria-label="caller textarea" />}
        inlineSlot={<div>I</div>}
        sendSlot={<button>S</button>}
        noticesSlot={<div>N</div>}
      />
    )
    expect(screen.getByText("F")).toBeTruthy()
    expect(
      screen.getByRole("textbox", { name: "caller textarea" })
    ).toBeTruthy()
    expect(screen.getByText("I")).toBeTruthy()
    expect(screen.getByRole("button", { name: "S" })).toBeTruthy()
    expect(screen.getByText("N")).toBeTruthy()
  })

  it("textareaSlot replaces the built-in textarea AND the >_ caret", () => {
    render(
      <RadialCommandV5
        {...base}
        message="built-in-value"
        textareaSlot={<textarea aria-label="custom ta" />}
      />
    )
    // Caller's textarea is rendered
    expect(
      screen.getByRole("textbox", { name: "custom ta" })
    ).toBeTruthy()
    // Built-in textarea is NOT rendered (would have aria-label="Message")
    expect(
      screen.queryByRole("textbox", { name: "Message" })
    ).toBeNull()
  })

  it("textareaSlot falls through to built-in textarea when omitted", () => {
    render(<RadialCommandV5 {...base} />)
    // Built-in textarea IS rendered
    expect(
      screen.getByRole("textbox", { name: "Message" })
    ).toBeTruthy()
  })
})
