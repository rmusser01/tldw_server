import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { WorldBookBudgetBar } from "../WorldBookBudgetBar"

describe("WorldBookBudgetBar", () => {
  it("renders current and max token values", () => {
    render(<WorldBookBudgetBar estimatedTokens={285} tokenBudget={700} />)

    expect(screen.getByText(/285\/700 tokens/)).toBeInTheDocument()
  })

  it("renders the meter role with correct aria attributes", () => {
    render(<WorldBookBudgetBar estimatedTokens={285} tokenBudget={700} />)

    const meter = screen.getByRole("meter")
    expect(meter).toHaveAttribute("aria-valuenow", "285")
    expect(meter).toHaveAttribute("aria-valuemax", "700")
    expect(meter).toHaveAttribute("aria-label", "Token budget usage")
  })

  it("shows warning when usage exceeds budget", () => {
    render(<WorldBookBudgetBar estimatedTokens={800} tokenBudget={700} />)

    expect(
      screen.getByText(/estimated usage exceeds the configured budget/i)
    ).toBeInTheDocument()
  })

  it("renders nothing when tokenBudget is 0", () => {
    const { container } = render(
      <WorldBookBudgetBar estimatedTokens={285} tokenBudget={0} />
    )

    expect(container.innerHTML).toBe("")
  })

  it("shows projected state when projectedTokens is provided", () => {
    render(
      <WorldBookBudgetBar
        estimatedTokens={285}
        tokenBudget={700}
        projectedTokens={340}
      />
    )

    expect(screen.getByText(/after save/i)).toBeInTheDocument()
    expect(screen.getByText(/340/)).toBeInTheDocument()
  })
})
