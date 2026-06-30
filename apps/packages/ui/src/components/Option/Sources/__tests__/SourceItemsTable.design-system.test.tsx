import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { SourceItemsTable } from "../SourceItemsTable"

vi.mock("@/design-system", () => ({
  getDesignSystemState: vi.fn((key: string) => ({
    key,
    label: key === "degraded" ? "Registry Degraded" : key,
    severity: key === "degraded" ? "warning" : "neutral"
  }))
}))

describe("SourceItemsTable design-system state labels", () => {
  it("uses the design-system registry label for the degraded filter", () => {
    const onFilterChange = vi.fn()

    render(
      <SourceItemsTable
        items={[]}
        filter="all"
        onFilterChange={onFilterChange}
        onReattach={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Registry Degraded" }))

    expect(onFilterChange).toHaveBeenCalledWith("degraded")
  })
})
