import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { StatusBadge, type RunStatus } from "../StatusBadge"

describe("Evaluations StatusBadge design-system adapter", () => {
  it.each([
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
  ] satisfies RunStatus[])(
    "renders the %s status through the shared Badge primitive",
    (status) => {
      const { container } = render(<StatusBadge status={status} />)

      expect(screen.getByText(status)).toBeInTheDocument()
      expect(
        container.querySelector('[data-ds-component="Badge"]')
      ).toBeInTheDocument()
    }
  )

  it("preserves the running status spinner affordance", () => {
    const { container } = render(<StatusBadge status="running" />)

    expect(container.querySelector(".animate-spin")).toBeInTheDocument()
  })
})
