import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ExpressionSlotGrid } from "../ExpressionSlotGrid"

describe("ExpressionSlotGrid", () => {
  it("shows custom expression slots after canonical slots", () => {
    render(
      <ExpressionSlotGrid
        slots={[
          { key: "neutral", label: "Neutral", canonical: true },
          { key: "custom:bashful", label: "Bashful", canonical: false },
          { key: "happy", label: "Happy", canonical: true }
        ]}
      />
    )

    expect(
      screen.getAllByTestId("expression-slot-label").map((node) => node.textContent)
    ).toEqual(["Neutral", "Happy", "Bashful"])
  })
})
