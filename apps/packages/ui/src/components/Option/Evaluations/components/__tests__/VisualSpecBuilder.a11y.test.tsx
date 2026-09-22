import React, { useState } from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { VisualSpecBuilder } from "../VisualSpecBuilder"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue || key }),
}))

function Builder({ type = "exact_match", initial = { case_sensitive: false } }: {
  type?: string
  initial?: Record<string, unknown>
}) {
  const [spec, setSpec] = useState(JSON.stringify(initial))
  return <>
    <VisualSpecBuilder evalType={type} specText={spec} onSpecChange={setSpec} />
    <output data-testid="spec-value">{spec}</output>
  </>
}

const savedSpec = () => JSON.parse(screen.getByTestId("spec-value").textContent || "{}")

describe("Evaluation visual spec accessible controls", () => {
  it("names and keyboard-operates the case-sensitive switch", async () => {
    const user = userEvent.setup()
    render(<Builder />)
    const control = screen.getByRole("switch", { name: "Case sensitive" })
    control.focus()
    await user.keyboard(" ")
    expect(control).toBeChecked()
    expect(savedSpec().case_sensitive).toBe(true)
  })

  it("names the advanced JSON switch and opens its editor by keyboard", async () => {
    const user = userEvent.setup()
    render(<Builder />)
    const control = screen.getByRole("switch", { name: "Advanced: Edit JSON" })
    control.focus()
    await user.keyboard(" ")
    expect(control).toBeChecked()
    expect(screen.getByRole("textbox")).toHaveValue(JSON.stringify({ case_sensitive: false }))
  })

  it("names both threshold controls and preserves keyboard edits", () => {
    render(<Builder type="response_quality" initial={{ thresholds: { min_score: 0.5 } }} />)
    const slider = screen.getByRole("slider", { name: "Min score" })
    const number = screen.getByRole("spinbutton", { name: "Min score" })
    slider.focus()
    // rc-slider consumes legacy keyCode, which user-event does not synthesize.
    fireEvent.keyDown(slider, { key: "ArrowRight", keyCode: 39 })
    expect(savedSpec().thresholds.min_score).toBe(0.55)
    expect(number).toHaveValue("0.55")
    number.focus()
    fireEvent.keyDown(number, { key: "ArrowUp", keyCode: 38 })
    expect(savedSpec().thresholds.min_score).toBe(0.6)
  })
})
