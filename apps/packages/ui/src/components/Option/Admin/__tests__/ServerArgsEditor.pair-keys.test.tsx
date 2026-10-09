// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ServerArgsEditor } from "../ServerArgsEditor"

function Harness(props: { initial: Record<string, unknown> }) {
  const [args, setArgs] = React.useState(props.initial)
  return <ServerArgsEditor value={args} onChange={setArgs} />
}

const trashButtons = (container: HTMLElement) =>
  Array.from(container.querySelectorAll("button")).filter(
    // The delete buttons are the only text-less, non-switch buttons (the
    // JSON-mode Switch renders as a text-less button too).
    (button) =>
      button.textContent === "" && button.getAttribute("role") !== "switch"
  )

describe("ServerArgsEditor stable pair keys (C-S5)", () => {
  it("assigns pair ids at creation so row keys survive deletion (focus proof)", () => {
    const { container } = render(<Harness initial={{ alpha: "1", beta: "2" }} />)

    // Focus beta's value input, then delete the sibling alpha row. With
    // index keys the focused row's DOM is unmounted; with stable per-pair
    // ids beta keeps its DOM node (and focus) through the deletion.
    const betaValue = screen.getByDisplayValue("2")
    betaValue.focus()
    expect(document.activeElement).toBe(betaValue)

    fireEvent.click(trashButtons(container)[0])

    expect(screen.getByDisplayValue("2")).toBeInTheDocument()
    expect(screen.getByDisplayValue("beta")).toBeInTheDocument()
    expect(screen.queryByDisplayValue("alpha")).not.toBeInTheDocument()
    expect(document.activeElement).toBe(screen.getByDisplayValue("2"))
  })

  it("renaming a key keeps the onChange contract (value parsed, pair moved)", () => {
    const onChange = vi.fn()
    render(<ServerArgsEditor value={{ alpha: "1", beta: "2" }} onChange={onChange} />)

    fireEvent.change(screen.getByDisplayValue("alpha"), {
      target: { value: "gamma" }
    })

    expect(onChange).toHaveBeenCalledWith({ gamma: 1, beta: "2" })
  })

  it("adds a new empty pair and lets the user name it", () => {
    render(<Harness initial={{ alpha: "1" }} />)

    fireEvent.click(screen.getByRole("button", { name: /add argument/i }))

    const keyInputs = screen.getAllByPlaceholderText("key")
    expect(keyInputs).toHaveLength(2)

    fireEvent.change(keyInputs[1], { target: { value: "threads" } })

    expect(screen.getByDisplayValue("threads")).toBeInTheDocument()
    expect(screen.getByDisplayValue("alpha")).toBeInTheDocument()
  })
})
