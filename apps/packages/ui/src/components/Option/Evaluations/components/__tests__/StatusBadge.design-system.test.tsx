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

  it("renders unknown statuses through the shared Badge fallback", () => {
    const status = "mystery-state" as RunStatus
    const { container } = render(<StatusBadge status={status} />)

    expect(screen.getByText(status)).toBeInTheDocument()
    expect(
      container.querySelector('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
  })

  it("falls back safely for prototype property status keys", () => {
    const status = "constructor" as RunStatus
    const { container } = render(<StatusBadge status={status} />)

    expect(screen.getByText(status)).toBeInTheDocument()
    expect(
      container.querySelector('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
  })

  it("keeps canonical state labels out of hidden badge copy", () => {
    render(<StatusBadge status="running" />)

    expect(screen.getByText("running")).toBeInTheDocument()
    expect(screen.queryByText("Retrying")).not.toBeInTheDocument()
  })

  it("preserves the running status spinner affordance", () => {
    render(<StatusBadge status="running" />)

    expect(
      screen.getByTestId("evaluations-status-running-spinner")
    ).toHaveAttribute("aria-hidden", "true")
  })
})
