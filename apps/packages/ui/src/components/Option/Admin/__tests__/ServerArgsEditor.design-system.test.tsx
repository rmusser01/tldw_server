// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ServerArgsEditor } from "../ServerArgsEditor"

describe("ServerArgsEditor design-system states", () => {
  it("renders JSON validation feedback through the design-system Alert primitive", async () => {
    render(<ServerArgsEditor value={{ threads: 4 }} onChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("switch"))
    const editor = screen.getByRole("textbox")

    fireEvent.change(editor, {
      target: { value: "{" }
    })

    await expectDesignSystemAlert("Invalid JSON")

    fireEvent.change(editor, {
      target: { value: "[]" }
    })

    await expectDesignSystemAlert("Must be a JSON object")
  })
})

async function expectDesignSystemAlert(message: string) {
  const title = await screen.findByText(message)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("role", "alert")
  expect(alertEl).toHaveTextContent(message)
}
