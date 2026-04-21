import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { RadialCommandV5 } from "../variants/RadialCommandV5"

const minimalProps = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
}

describe("RadialCommandV5", () => {
  it("renders the textarea with the default placeholder", () => {
    render(<RadialCommandV5 {...minimalProps} />)
    expect(
      screen.getByPlaceholderText(/ask anything · \/ for commands/i)
    ).toBeTruthy()
  })

  it("forwards message changes", () => {
    const onMessageChange = vi.fn()
    render(<RadialCommandV5 {...minimalProps} onMessageChange={onMessageChange} />)
    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), {
      target: { value: "hi" },
    })
    expect(onMessageChange).toHaveBeenCalledWith("hi")
  })

  it("fires onSend on Cmd+Enter", () => {
    const onSend = vi.fn()
    render(<RadialCommandV5 {...minimalProps} onSend={onSend} />)
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Message" }), {
      key: "Enter",
      metaKey: true,
    })
    expect(onSend).toHaveBeenCalledOnce()
  })

  it("swaps to Stop button and routes clicks to stopStreaming", () => {
    const stopStreaming = vi.fn()
    const onSend = vi.fn()
    render(
      <RadialCommandV5
        {...minimalProps}
        onSend={onSend}
        sending
        stopStreaming={stopStreaming}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Stop" }))
    expect(stopStreaming).toHaveBeenCalledOnce()
    expect(onSend).not.toHaveBeenCalled()
  })

  it("renders the ⌘K trigger button when onPaletteTrigger is provided", () => {
    const onPaletteTrigger = vi.fn()
    render(
      <RadialCommandV5
        {...minimalProps}
        onPaletteTrigger={onPaletteTrigger}
      />
    )
    const btn = screen.getByRole("button", { name: "Open command palette" })
    fireEvent.click(btn)
    expect(onPaletteTrigger).toHaveBeenCalledOnce()
  })

  it("does not render ⌘K trigger when onPaletteTrigger is omitted", () => {
    render(<RadialCommandV5 {...minimalProps} />)
    expect(screen.queryByRole("button", { name: "Open command palette" })).toBeNull()
  })

  it("renders the slash palette when paletteOpen=true", () => {
    render(
      <RadialCommandV5
        {...minimalProps}
        paletteOpen
        paletteQuery="model"
        paletteActiveIndex={0}
        onPaletteActiveIndexChange={vi.fn()}
        onPaletteSelect={vi.fn()}
        paletteGroups={[
          {
            id: "m",
            label: "Models",
            rows: [{ id: "haiku", command: "/model haiku-4-5" }],
          },
        ]}
      />
    )
    expect(screen.getByRole("listbox", { name: /composer slash commands/i })).toBeTruthy()
    expect(screen.getByText("Models")).toBeTruthy()
    expect(screen.getByText("/model haiku-4-5")).toBeTruthy()
  })

  it("renders facets in the meta row", () => {
    render(
      <RadialCommandV5
        {...minimalProps}
        facets={[
          { id: "src", fieldKey: "src", value: "irb-archive · 14", active: true },
          { id: "mdl", fieldKey: "mdl", value: "haiku-4-5" },
        ]}
      />
    )
    expect(screen.getByRole("group", { name: /composer facets/i })).toBeTruthy()
    expect(screen.getByText("irb-archive · 14")).toBeTruthy()
  })

  it("renders the token meter in the facet row trailing slot", () => {
    render(
      <RadialCommandV5
        {...minimalProps}
        facets={[{ id: "m", fieldKey: "mdl", value: "haiku-4-5" }]}
        tokens={{ used: 284, max: 8000 }}
      />
    )
    expect(screen.getByLabelText(/of 8K tokens used/i)).toBeTruthy()
  })

  it("renders round send button", () => {
    const { container } = render(<RadialCommandV5 {...minimalProps} />)
    const sendBtn = screen.getByRole("button", { name: "Send" })
    expect(sendBtn.className).toContain("rounded-full")
  })

  it("applies density attribute", () => {
    const { container } = render(
      <RadialCommandV5 {...minimalProps} density="compact" />
    )
    const root = container.querySelector("[data-variant='v5']")
    expect(root?.getAttribute("data-density")).toBe("compact")
  })

  it("does not render facets row when no facets and no tokens", () => {
    render(<RadialCommandV5 {...minimalProps} />)
    expect(
      screen.queryByRole("group", { name: /composer facets/i })
    ).toBeNull()
  })
})
