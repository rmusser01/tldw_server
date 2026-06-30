import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SplitBriefV3 } from "../variants/SplitBriefV3"

const minimalProps = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
  briefSections: [
    {
      id: "brief",
      label: "Brief",
      fields: [
        { id: "src", fieldKey: "src", value: "irb-archive · 14", active: true },
        { id: "mdl", fieldKey: "mdl", value: "haiku-4-5" },
      ],
    },
  ],
}

describe("SplitBriefV3", () => {
  it("renders the textarea with the provided placeholder", () => {
    render(<SplitBriefV3 {...minimalProps} placeholder="Your question" />)
    expect(screen.getByPlaceholderText("Your question")).toBeTruthy()
  })

  it("forwards message changes to onMessageChange", () => {
    const onMessageChange = vi.fn()
    render(
      <SplitBriefV3 {...minimalProps} onMessageChange={onMessageChange} />
    )
    const textarea = screen.getByRole("textbox", { name: "Question" })
    fireEvent.change(textarea, { target: { value: "hi" } })
    expect(onMessageChange).toHaveBeenCalledWith("hi")
  })

  it("renders the brief section label at desktop density", () => {
    render(<SplitBriefV3 {...minimalProps} density="desktop" />)
    expect(screen.getByText("Brief")).toBeTruthy()
  })

  it("hides the brief section label at compact density", () => {
    render(<SplitBriefV3 {...minimalProps} density="compact" />)
    expect(screen.queryByText("Brief")).toBeNull()
  })

  it("renders field keys at desktop density", () => {
    render(<SplitBriefV3 {...minimalProps} density="desktop" />)
    expect(screen.getByText("src")).toBeTruthy()
    expect(screen.getByText("mdl")).toBeTruthy()
  })

  it("hides field keys at compact density", () => {
    render(<SplitBriefV3 {...minimalProps} density="compact" />)
    expect(screen.queryByText("src")).toBeNull()
    expect(screen.queryByText("mdl")).toBeNull()
  })

  it("fires field onClick", () => {
    const onClick = vi.fn()
    render(
      <SplitBriefV3
        {...minimalProps}
        briefSections={[
          {
            id: "brief",
            fields: [
              {
                id: "src",
                fieldKey: "src",
                value: "irb-archive",
                onClick,
                "aria-label": "Open source picker",
              },
            ],
          },
        ]}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Open source picker" }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it("renders icon buttons + token meter + cost label", () => {
    render(
      <SplitBriefV3
        {...minimalProps}
        iconButtons={[
          { id: "attach", label: "Attach", icon: "⎙" },
          { id: "mention", label: "Mention", icon: "@" },
        ]}
        tokens={{ used: 284, max: 8000 }}
        costLabel="≈ $0.003"
      />
    )
    expect(screen.getByRole("button", { name: "Attach" })).toBeTruthy()
    expect(screen.getByRole("button", { name: "Mention" })).toBeTruthy()
    expect(screen.getByLabelText(/of 8K tokens used/i)).toBeTruthy()
    expect(screen.getByText(/\$0\.003/)).toBeTruthy()
  })

  it("hides the cost label at compact density to save space", () => {
    render(
      <SplitBriefV3
        {...minimalProps}
        tokens={{ used: 84, max: 8000 }}
        costLabel="≈ $0.001"
        density="compact"
      />
    )
    expect(screen.queryByText(/\$0\.001/)).toBeNull()
  })

  it("fires onSend on Cmd+Enter", () => {
    const onSend = vi.fn()
    render(<SplitBriefV3 {...minimalProps} onSend={onSend} />)
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Question" }), {
      key: "Enter",
      metaKey: true,
    })
    expect(onSend).toHaveBeenCalledOnce()
  })

  it("swaps Send to Stop while streaming and routes clicks to stopStreaming", () => {
    const stopStreaming = vi.fn()
    const onSend = vi.fn()
    render(
      <SplitBriefV3
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

  it("applies density attribute for QA selectors", () => {
    const { container } = render(
      <SplitBriefV3 {...minimalProps} density="compact" />
    )
    const root = container.querySelector("[data-variant='v3']")
    expect(root?.getAttribute("data-density")).toBe("compact")
  })

  it("exposes the Brief as a role=group for AT", () => {
    render(<SplitBriefV3 {...minimalProps} density="desktop" />)
    expect(screen.getByRole("group", { name: /brief/i })).toBeTruthy()
  })
})
