import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => (typeof fallback === "string" ? fallback : key)
  })
}))

import { ModelParamSliderField } from "../ModelParamField"

const field = (unavailableReason?: string) => (
  <ModelParamSliderField
    label="Min P"
    tooltip="Minimum probability"
    value={0.1}
    min={0}
    max={1}
    step={0.05}
    precision={2}
    restingValue={0}
    onChange={vi.fn()}
    onReset={vi.fn()}
    unavailableReason={unavailableReason}
  />
)

// #3283: a setting the provider rejects says why, and can't be moved.
describe("ModelParamSliderField when the setting can't be used", () => {
  it("shows the reason instead of the value and disables the slider", () => {
    render(field("Not supported by OpenAI"))
    expect(screen.getByText("Not supported by OpenAI")).toBeInTheDocument()
    expect(screen.queryByText("0.1")).toBeNull()
    expect(screen.getByRole("slider")).toHaveAttribute("aria-disabled", "true")
    expect(screen.queryByRole("button", { name: /Reset/ })).toBeNull()
  })

  it("works as before otherwise", () => {
    render(field())
    expect(screen.getByText("0.1")).toBeInTheDocument()
    expect(screen.getByRole("slider")).not.toHaveAttribute("aria-disabled", "true")
  })
})
