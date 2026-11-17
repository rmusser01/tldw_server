import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"
import { renderTabWrappedDialog } from "../dialog-tab-wrap"

// AX-09: a dialog at the end of the document let Tab escape to <body>.
const Dialog: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <>
    <button type="button">Page control</button>
    <div role="dialog" aria-modal="true" aria-label="Example" tabIndex={-1}>
      {renderTabWrappedDialog(children)}
    </div>
  </>
)

describe("renderTabWrappedDialog", () => {
  it("wraps Tab from the last control to the first and Shift+Tab back", async () => {
    const user = userEvent.setup()
    render(
      <Dialog>
        <button type="button">First</button>
        <button type="button">Last</button>
      </Dialog>
    )

    screen.getByRole("button", { name: "Last" }).focus()
    await user.tab()
    expect(screen.getByRole("button", { name: "First" })).toHaveFocus()

    await user.tab({ shift: true })
    expect(screen.getByRole("button", { name: "Last" })).toHaveFocus()
  })

  it("skips disabled, hidden and untabbable controls when finding the ends", async () => {
    const user = userEvent.setup()
    render(
      <Dialog>
        <button type="button">First</button>
        <button type="button">Last</button>
        <button type="button" disabled>
          Disabled
        </button>
        <button type="button" tabIndex={-1}>
          Skipped
        </button>
        <div hidden>
          <button type="button">Hidden</button>
        </div>
      </Dialog>
    )

    screen.getByRole("button", { name: "Last" }).focus()
    await user.tab()
    expect(screen.getByRole("button", { name: "First" })).toHaveFocus()
  })

  it("leaves Tab alone when a control inside has already handled it", async () => {
    const user = userEvent.setup()
    render(
      <Dialog>
        <button type="button">First</button>
        <textarea
          aria-label="Editor"
          onKeyDown={(event) => {
            if (event.key === "Tab") event.preventDefault()
          }}
        />
      </Dialog>
    )

    const editor = screen.getByRole("textbox", { name: "Editor" })
    editor.focus()
    await user.tab()
    expect(editor).toHaveFocus()
  })
})
