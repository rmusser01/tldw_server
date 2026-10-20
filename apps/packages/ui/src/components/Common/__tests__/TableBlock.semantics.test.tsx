import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

// AX-10 (#3123): Markdown tables in chat answers and the notes preview must
// keep real table semantics, so screen readers can announce rows, columns and
// headers instead of orphan cells.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?: string | { defaultValue?: string; [name: string]: unknown }
    ) => {
      if (typeof fallback === "string") return fallback
      const template = fallback?.defaultValue ?? key
      return template.replace(/\{\{(\w+)\}\}/g, (_, name) => String(fallback?.[name] ?? ""))
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

const GFM_TABLE = [
  "| Name | Score |",
  "| --- | --- |",
  "| Ada | 3 |",
  "| Linus | 5 |"
].join("\n")

afterEach(() => {
  vi.restoreAllMocks()
})

describe("TableBlock table semantics", () => {
  it("renders a Markdown table as a real table with column headers", () => {
    render(<Markdown message={GFM_TABLE} />)

    const table = screen.getByRole("table")
    const headers = within(table).getAllByRole("columnheader")
    expect(headers.map((header) => header.textContent)).toEqual([
      "Name",
      "Score"
    ])
    expect(within(table).getAllByRole("row")).toHaveLength(3)
    expect(within(table).getByRole("cell", { name: "Linus" })).toBeInTheDocument()
  })

  it("puts the table in a labelled scroll region keyboard users can reach", () => {
    render(<Markdown message={GFM_TABLE} />)

    const region = screen.getByRole("region", { name: "Table: Name, Score" })
    expect(region).toHaveAttribute("tabindex", "0")
    expect(within(region).getByRole("table")).toBeInTheDocument()
  })

  // Several tables in one answer were all landmarks named "Table", which
  // screen-reader landmark lists cannot tell apart (axe landmark-unique).
  it("names each table region after its columns so several tables stay distinct", () => {
    const second = ["| City | Population |", "| --- | --- |", "| Oslo | 700k |"].join("\n")
    render(<Markdown message={`${GFM_TABLE}\n\nAnd another:\n\n${second}`} />)

    const names = screen.getAllByRole("region").map((region) => region.getAttribute("aria-label"))
    expect(names).toEqual(["Table: Name, Score", "Table: City, Population"])
  })

  it("still copies the table rows as CSV", async () => {
    const writeText = vi.fn(async () => undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })
    render(<Markdown message={GFM_TABLE} />)

    fireEvent.click(screen.getByRole("button", { name: "Copy as CSV" }))

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("Name,Score\nAda,3\nLinus,5")
    })
  })

  it("shows a visible focus ring on the copy and download buttons", () => {
    render(<Markdown message={GFM_TABLE} />)

    for (const name of ["Copy as CSV", "Download CSV"]) {
      const button = screen.getByRole("button", { name })
      expect(button.className).not.toMatch(/(^|\s)focus:outline-none(\s|$)/)
      expect(button).toHaveClass("focus-visible:ring-2", "focus-visible:ring-focus")
    }
  })
})
