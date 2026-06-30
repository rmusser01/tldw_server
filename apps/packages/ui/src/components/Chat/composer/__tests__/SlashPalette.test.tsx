import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SlashPalette } from "../shared/SlashPalette"

const baseProps = {
  open: true,
  query: "model",
  activeIndex: 0,
  onActiveIndexChange: vi.fn(),
  onSelect: vi.fn(),
  groups: [
    {
      id: "models",
      label: "Models · 2 results",
      rows: [
        {
          id: "haiku",
          icon: "☀",
          command: "/model haiku-4-5",
          hint: "current · 200k ctx",
          kbd: "↩",
        },
        {
          id: "opus",
          icon: "♦",
          command: "/model opus-4-1",
          hint: "deep reasoning",
        },
      ],
    },
  ],
}

describe("SlashPalette", () => {
  it("renders nothing when open=false", () => {
    const { container } = render(
      <SlashPalette {...baseProps} open={false} />
    )
    expect(container.firstChild).toBeNull()
  })

  it("renders the query with a leading slash", () => {
    render(<SlashPalette {...baseProps} />)
    const header = screen.getByRole("listbox", {
      name: /composer slash commands/i,
    })
    expect(header.textContent).toContain("model")
    expect(header.textContent).toContain("/")
  })

  it("renders group labels + rows with command + hint + kbd", () => {
    render(<SlashPalette {...baseProps} />)
    expect(screen.getByText("Models · 2 results")).toBeTruthy()
    expect(screen.getByText("/model haiku-4-5")).toBeTruthy()
    expect(screen.getByText("current · 200k ctx")).toBeTruthy()
    expect(screen.getByText("↩")).toBeTruthy()
  })

  it("marks the active row aria-selected=true", () => {
    render(<SlashPalette {...baseProps} activeIndex={1} />)
    const rows = screen.getAllByRole("option")
    expect(rows[0].getAttribute("aria-selected")).toBe("false")
    expect(rows[1].getAttribute("aria-selected")).toBe("true")
  })

  it("fires onSelect when a row is clicked", () => {
    const onSelect = vi.fn()
    render(<SlashPalette {...baseProps} onSelect={onSelect} />)
    fireEvent.click(screen.getByText("/model opus-4-1"))
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ id: "opus" })
    )
  })

  it("fires onActiveIndexChange on mouseenter", () => {
    const onActiveIndexChange = vi.fn()
    render(
      <SlashPalette
        {...baseProps}
        onActiveIndexChange={onActiveIndexChange}
      />
    )
    fireEvent.mouseEnter(screen.getByText("/model opus-4-1"))
    expect(onActiveIndexChange).toHaveBeenCalledWith(1)
  })

  it("shows the empty label when no groups have rows", () => {
    render(
      <SlashPalette
        {...baseProps}
        groups={[]}
        emptyLabel="Nothing matches"
      />
    )
    expect(screen.getByText("Nothing matches")).toBeTruthy()
  })

  it("renders the match-count label in the footer", () => {
    render(
      <SlashPalette {...baseProps} matchCountLabel="14 commands matched" />
    )
    expect(screen.getByText("14 commands matched")).toBeTruthy()
  })

  it("always renders the footer shortcut hints", () => {
    render(<SlashPalette {...baseProps} />)
    expect(screen.getByText(/↑↓ navigate/)).toBeTruthy()
    // Exact match — "↩ run" also appears inside "⌘↩ run + send"
    expect(screen.getByText("↩ run")).toBeTruthy()
    expect(screen.getByText("⌘↩ run + send")).toBeTruthy()
    expect(screen.getByText(/esc close/)).toBeTruthy()
  })

  it("tracks activeIndex across groups (flat index across all rows)", () => {
    render(
      <SlashPalette
        {...baseProps}
        groups={[
          {
            id: "a",
            label: "A",
            rows: [
              { id: "a1", command: "/a1" },
              { id: "a2", command: "/a2" },
            ],
          },
          {
            id: "b",
            label: "B",
            rows: [{ id: "b1", command: "/b1" }],
          },
        ]}
        activeIndex={2}
      />
    )
    const rows = screen.getAllByRole("option")
    expect(rows).toHaveLength(3)
    expect(rows[2].getAttribute("aria-selected")).toBe("true")
  })
})
