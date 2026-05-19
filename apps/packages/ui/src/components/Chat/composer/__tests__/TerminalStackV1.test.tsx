import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TerminalStackV1 } from "../variants/TerminalStackV1"

const minimalProps = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
}

describe("TerminalStackV1", () => {
  it("renders the textarea with the provided placeholder", () => {
    render(<TerminalStackV1 {...minimalProps} placeholder="Ask away" />)
    expect(screen.getByPlaceholderText("Ask away")).toBeTruthy()
  })

  it("forwards message changes to onMessageChange", () => {
    const onMessageChange = vi.fn()
    render(<TerminalStackV1 {...minimalProps} onMessageChange={onMessageChange} />)
    const textarea = screen.getByRole("textbox", { name: "Message" })
    fireEvent.change(textarea, { target: { value: "hello" } })
    expect(onMessageChange).toHaveBeenCalledWith("hello")
  })

  it("renders a source chip with count + label when provided", () => {
    render(
      <TerminalStackV1
        {...minimalProps}
        sourceChip={{ count: 14, label: "irb-archive" }}
      />
    )
    expect(screen.getByText("14")).toBeTruthy()
    expect(screen.getByText("irb-archive")).toBeTruthy()
  })

  it("renders top-rail chips and fires onClick", () => {
    const onClick = vi.fn()
    render(
      <TerminalStackV1
        {...minimalProps}
        topChips={[
          { id: "web", label: "Web search", active: true, onClick },
        ]}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Web search" }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("renders bottom chips + icon buttons + token meter", () => {
    render(
      <TerminalStackV1
        {...minimalProps}
        bottomChips={[{ id: "model", label: "haiku-4-5" }]}
        iconButtons={[
          { id: "attach", label: "Attach file", icon: "⎙" },
        ]}
        tokens={{ used: 127, max: 8000 }}
      />
    )
    expect(screen.getByText("haiku-4-5")).toBeTruthy()
    expect(screen.getByRole("button", { name: "Attach file" })).toBeTruthy()
    expect(screen.getByLabelText(/of 8K tokens used/i)).toBeTruthy()
  })

  it("calls onSend when the Send button is clicked", () => {
    const onSend = vi.fn()
    render(<TerminalStackV1 {...minimalProps} onSend={onSend} />)
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    expect(onSend).toHaveBeenCalledOnce()
  })

  it("renders Send button with Send label on desktop", () => {
    render(<TerminalStackV1 {...minimalProps} />)
    expect(screen.getByRole("button", { name: "Send" })).toBeTruthy()
  })

  it("switches to Stop button when streaming, and routes clicks to stopStreaming", () => {
    const stopStreaming = vi.fn()
    const onSend = vi.fn()
    render(
      <TerminalStackV1
        {...minimalProps}
        onSend={onSend}
        sending={true}
        stopStreaming={stopStreaming}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Stop" }))
    expect(stopStreaming).toHaveBeenCalledOnce()
    expect(onSend).not.toHaveBeenCalled()
  })

  it("fires onSend when Cmd+Enter is pressed in the textarea", () => {
    const onSend = vi.fn()
    render(<TerminalStackV1 {...minimalProps} onSend={onSend} />)
    const textarea = screen.getByRole("textbox", { name: "Message" })
    fireEvent.keyDown(textarea, { key: "Enter", metaKey: true })
    expect(onSend).toHaveBeenCalledOnce()
  })

  it("routes Cmd+Enter to stopStreaming while a response is in flight", () => {
    const onSend = vi.fn()
    const stopStreaming = vi.fn()
    render(
      <TerminalStackV1
        {...minimalProps}
        onSend={onSend}
        sending={true}
        stopStreaming={stopStreaming}
      />
    )
    const textarea = screen.getByRole("textbox", { name: "Message" })
    fireEvent.keyDown(textarea, { key: "Enter", metaKey: true })
    expect(stopStreaming).toHaveBeenCalledOnce()
    expect(onSend).not.toHaveBeenCalled()
  })

  it("does not fire onSend on plain Enter (unmodified)", () => {
    const onSend = vi.fn()
    render(<TerminalStackV1 {...minimalProps} onSend={onSend} />)
    const textarea = screen.getByRole("textbox", { name: "Message" })
    fireEvent.keyDown(textarea, { key: "Enter" })
    expect(onSend).not.toHaveBeenCalled()
  })

  it("does not fire onSend when canSend is false", () => {
    const onSend = vi.fn()
    render(
      <TerminalStackV1 {...minimalProps} onSend={onSend} canSend={false} />
    )
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Message" }), {
      key: "Enter",
      metaKey: true,
    })
    expect(onSend).not.toHaveBeenCalled()
  })

  it("applies compact density class when density='compact'", () => {
    const { container } = render(
      <TerminalStackV1 {...minimalProps} density="compact" />
    )
    const root = container.querySelector("[data-variant='v1']")
    expect(root?.getAttribute("data-density")).toBe("compact")
  })

  it("honors the tokens prop in the meter", () => {
    render(
      <TerminalStackV1 {...minimalProps} tokens={{ used: 50, max: 2000 }} />
    )
    const meter = screen.getByLabelText(/of 2K tokens used/i)
    expect(meter.textContent).toContain("50")
  })
})
