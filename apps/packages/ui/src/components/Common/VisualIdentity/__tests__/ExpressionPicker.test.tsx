import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ExpressionPicker } from "../ExpressionPicker"

describe("ExpressionPicker", () => {
  it("selects an available expression", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()

    render(
      <ExpressionPicker
        value="neutral"
        expressions={[
          { key: "neutral", label: "Neutral", hasAsset: true },
          { key: "happy", label: "Happy", hasAsset: true }
        ]}
        onChange={onChange}
      />
    )

    await user.click(screen.getByRole("button", { name: "Happy" }))

    expect(onChange).toHaveBeenCalledWith("happy")
  })

  it("does not select expressions without assets", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()

    render(
      <ExpressionPicker
        value="neutral"
        expressions={[
          { key: "neutral", label: "Neutral", hasAsset: true },
          { key: "sad", label: "Sad", hasAsset: false }
        ]}
        onChange={onChange}
      />
    )

    const sadButton = screen.getByRole("button", { name: "Sad" })
    expect(sadButton).toBeDisabled()
    await user.click(sadButton)

    expect(onChange).not.toHaveBeenCalled()
  })
})
