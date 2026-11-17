import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SidepanelPersistenceSwitch } from "../SidepanelPersistenceSwitch"

// AX-16 (#3124): the side panel "Save chat to history" switch was the 28x16 px
// small antd switch. It should use the default size, and its visible label
// should toggle it too, so the target is easy to hit.
describe("SidepanelPersistenceSwitch", () => {
  const renderSwitch = (
    props: Partial<React.ComponentProps<typeof SidepanelPersistenceSwitch>> = {}
  ) => {
    const onChange = vi.fn()
    render(
      <SidepanelPersistenceSwitch
        checked
        onChange={onChange}
        ariaLabel="Save chat to history"
        label="Saved on this device"
        {...props}
      />
    )
    return { onChange }
  }

  it("uses the default-size switch, named by its aria-label", () => {
    renderSwitch()

    const control = screen.getByRole("switch", { name: "Save chat to history" })
    expect(control).toBeChecked()
    expect(control).not.toHaveClass("ant-switch-small")
  })

  it("toggles when the visible label is clicked", () => {
    const { onChange } = renderSwitch()

    fireEvent.click(screen.getByText("Saved on this device"))

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange.mock.calls[0][0]).toBe(false)
  })

  it("does not toggle from the label while disabled", () => {
    const { onChange } = renderSwitch({ disabled: true })

    fireEvent.click(screen.getByText("Saved on this device"))

    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByRole("switch", { name: "Save chat to history" })).toBeDisabled()
  })
})
