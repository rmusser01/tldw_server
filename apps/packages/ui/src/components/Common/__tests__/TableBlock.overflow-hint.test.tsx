import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TableBlock } from "../TableBlock"

// XS-14: in a narrow side panel a wide markdown table clips its last columns.
// When the table is wider than its box, say so and fade the cut-off edge.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback ?? _key).replace(/\{\{(\w+)\}\}/g, (_match, name) =>
        String(options?.[name] ?? "")
      )
  })
}))

const box = { scrollWidth: 0, clientWidth: 0 }

const Table = () => (
  <TableBlock>
    <table>
      <thead>
        <tr>
          <th>Option</th>
          <th>Pros</th>
          <th>Cons</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>A</td>
          <td>Fast</td>
          <td>Less flexible</td>
        </tr>
      </tbody>
    </table>
  </TableBlock>
)

describe("TableBlock horizontal overflow hint (XS-14)", () => {
  beforeEach(() => {
    const isScroller = (element: Element) =>
      element.getAttribute("data-testid") === "table-scroll-region"
    vi.spyOn(Element.prototype, "scrollWidth", "get").mockImplementation(function (this: Element) {
      return isScroller(this) ? box.scrollWidth : 0
    })
    vi.spyOn(Element.prototype, "clientWidth", "get").mockImplementation(function (this: Element) {
      return isScroller(this) ? box.clientWidth : 0
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says there are more columns when the table is wider than its box", () => {
    box.scrollWidth = 600
    box.clientWidth = 320
    render(<Table />)
    expect(screen.getByText("Scroll for more columns →")).toBeInTheDocument()
    expect(screen.getByTestId("table-scroll-fade")).toBeInTheDocument()
    const region = screen.getByRole("region", { name: "Table: Option" })
    expect(region).toHaveAttribute("tabindex", "0")
    expect(region).toHaveClass("focus-visible:ring-2")
  })

  it("names each wide table by its caption when it has one", () => {
    box.scrollWidth = 600
    box.clientWidth = 320
    render(
      <TableBlock>
        <table>
          <caption>Pricing plans</caption>
          <thead>
            <tr>
              <th>Plan</th>
              <th>Price</th>
            </tr>
          </thead>
        </table>
      </TableBlock>
    )
    expect(screen.getByRole("region", { name: "Table: Pricing plans" })).toBeInTheDocument()
  })

  it("keeps one size watcher while the reply re-renders the table", () => {
    let created = 0
    class CountingResizeObserver {
      constructor() {
        created += 1
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", CountingResizeObserver)
    try {
      box.scrollWidth = 600
      box.clientWidth = 320
      const { rerender } = render(<Table />)
      for (let i = 0; i < 4; i += 1) rerender(<Table />)
      expect(created).toBe(1)
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("hides the hint once the reader scrolls to the last column", () => {
    box.scrollWidth = 600
    box.clientWidth = 320
    render(<Table />)
    const region = screen.getByTestId("table-scroll-region")
    act(() => {
      region.scrollLeft = 280
      fireEvent.scroll(region)
    })
    expect(screen.queryByText("Scroll for more columns →")).not.toBeInTheDocument()
    expect(screen.queryByTestId("table-scroll-fade")).not.toBeInTheDocument()
  })

  it("shows no hint when the whole table fits", () => {
    box.scrollWidth = 320
    box.clientWidth = 320
    render(<Table />)
    expect(screen.queryByText("Scroll for more columns →")).not.toBeInTheDocument()
    expect(screen.queryByTestId("table-scroll-fade")).not.toBeInTheDocument()
  })
})
