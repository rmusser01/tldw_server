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

  it("keeps focus while typing an argument key one keystroke at a time (C-S5 fix1)", () => {
    render(<Harness initial={{ alpha: "1", beta: "2" }} />)

    // Select-all + type "gamma" into alpha's key input: one change event per
    // keystroke. A remounted row destroys the focused DOM node, so holding
    // the same node reference proves the row kept its identity per keystroke.
    const keyInput = screen.getByDisplayValue("alpha")
    keyInput.focus()
    expect(document.activeElement).toBe(keyInput)

    for (const typed of ["g", "ga", "gam", "gamm", "gamma"]) {
      fireEvent.change(keyInput, { target: { value: typed } })
      expect(document.activeElement).toBe(keyInput)
      // The staged rename renders immediately (row-local draft state).
      expect(keyInput).toHaveValue(typed)
    }

    // The rename committed through the echo: same payload contract as before.
    expect(screen.getByDisplayValue("gamma")).toBeInTheDocument()
    expect(screen.getByDisplayValue("beta")).toBeInTheDocument()
    expect(screen.queryByDisplayValue("alpha")).not.toBeInTheDocument()
  })

  it("keeps focus while naming a newly added argument per keystroke (C-S5 fix1)", () => {
    render(<Harness initial={{ threads: "4" }} />)

    fireEvent.click(screen.getByRole("button", { name: /add argument/i }))

    // The new row's key input is the (only) empty input with the "key"
    // placeholder — its value input is empty too.
    const newKeyInput = screen
      .getAllByPlaceholderText("key")
      .find((el) => (el as HTMLInputElement).value === "") as HTMLInputElement
    newKeyInput.focus()

    // The empty new pair starts with key "" — the old id-by-arg-key scheme
    // reassigned its id on the very first keystroke.
    for (const typed of ["c", "ct", "ctx"]) {
      fireEvent.change(newKeyInput, { target: { value: typed } })
      expect(document.activeElement).toBe(newKeyInput)
      expect(newKeyInput).toHaveValue(typed)
    }

    expect(screen.getByDisplayValue("ctx")).toBeInTheDocument()
    expect(screen.getByDisplayValue("threads")).toBeInTheDocument()
  })

  it("keeps focus while typing into a value input per keystroke (C-S5 fix1)", () => {
    render(<Harness initial={{ threads: "4" }} />)

    const valueInput = screen.getByDisplayValue("4")
    valueInput.focus()

    for (const typed of ["4", "48", "480"]) {
      fireEvent.change(valueInput, { target: { value: typed } })
      expect(document.activeElement).toBe(valueInput)
      expect(valueInput).toHaveValue(typed)
    }

    expect(screen.getByDisplayValue("480")).toBeInTheDocument()
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
