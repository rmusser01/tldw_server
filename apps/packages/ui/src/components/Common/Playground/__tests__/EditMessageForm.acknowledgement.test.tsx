// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { EditMessageForm } from "../EditMessageForm"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) =>
      (
        ({
          save: "Save",
          saveAndSubmit: "Save & Send",
          cancel: "Cancel",
          saving: "Saving..."
        }) as Record<string, string>
      )[key] ??
      fallback ??
      key
  })
}))

const deferred = () => {
  let resolve!: (value: unknown) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<unknown>((accept, refuse) => {
    resolve = accept
    reject = refuse
  })
  void promise.catch(() => {})
  return { promise, resolve, reject }
}
const mount = (
  submit: (value: string, isSend: boolean) => unknown = vi.fn()
) => {
  const close = vi.fn()
  const props = {
    value: "Original",
    isBot: false,
    onSumbit: submit,
    onClose: close
  }
  const view = render(<EditMessageForm {...props} />)
  const editor = screen.getByRole("textbox")
  fireEvent.change(editor, { target: { value: "Exact authored edit  \n" } })
  return { ...view, close, props, editor }
}

it.each(["Save", "Save & Send"])(
  "%s waits for the current acknowledgement",
  async (action) => {
    const pending = deferred()
    const submit = vi.fn(() => pending.promise)
    const view = mount(submit)
    fireEvent.click(screen.getByRole("button", { name: action, exact: true }))
    try {
      expect(view.close).not.toHaveBeenCalled()
      expect(view.editor).toHaveValue("Exact authored edit  \n")
      expect(submit).toHaveBeenCalledExactlyOnceWith(
        "Exact authored edit  \n",
        action === "Save & Send",
        expect.any(Function)
      )
    } finally {
      await act(async () => {
        pending.resolve({ status: "submitted" })
        await pending.promise
      })
    }
    await waitFor(() => expect(view.close).toHaveBeenCalledOnce())
  }
)

it.each(["Save", "Save & Send"])(
  "%s retains exact text after rejection",
  async (action) => {
    const pending = deferred()
    const view = mount(() => pending.promise)
    fireEvent.click(screen.getByRole("button", { name: action, exact: true }))
    await act(async () => {
      pending.reject(new Error("Request refused"))
      await pending.promise.catch(() => {})
    })
    expect(view.close).not.toHaveBeenCalled()
    expect(view.editor).toHaveValue("Exact authored edit  \n")
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Your edited text is kept"
    )
    expect(
      screen.getByRole("button", { name: action, exact: true })
    ).toBeEnabled()
  }
)

it.each([
  undefined,
  false,
  { status: "failed", errorMessage: "not saved" },
  { status: "skipped", reason: "view retired" },
  { unexpected: "unconfirmed" }
])("does not close for an unconfirmed outcome %j", async (outcome) => {
  const view = mount(() => Promise.resolve(outcome))
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  await screen.findByRole("alert")
  expect(view.close).not.toHaveBeenCalled()
  expect(view.editor).toHaveValue("Exact authored edit  \n")
})

it.each([false, true])(
  "a held acknowledgement preserves newer input including edit/undo=%s",
  async (undo) => {
    const pending = deferred()
    const view = mount(() => pending.promise)
    fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
    fireEvent.change(view.editor, { target: { value: "Newer unsaved text" } })
    if (undo)
      fireEvent.change(view.editor, {
        target: { value: "Exact authored edit  \n" }
      })
    await act(async () => {
      pending.resolve({ status: "submitted" })
      await pending.promise
    })
    expect(view.close).not.toHaveBeenCalled()
    expect(view.editor).toHaveValue(
      undo ? "Exact authored edit  \n" : "Newer unsaved text"
    )
  }
)

it("Cancel retires a held save before a later acknowledgement", async () => {
  const pending = deferred()
  const view = mount(() => pending.promise)
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }))
  await act(async () => {
    pending.resolve({ status: "submitted" })
    await pending.promise
  })
  expect(view.close).toHaveBeenCalledOnce()
})

it("unmount retires the held save", async () => {
  const pending = deferred()
  const view = mount(() => pending.promise)
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  view.unmount()
  await act(async () => {
    pending.resolve({ status: "submitted" })
    await pending.promise
  })
  expect(view.close).not.toHaveBeenCalled()
})

it("a changed canonical message preserves the authored edit and blocks an unrevised save", async () => {
  const submit = vi.fn()
  const view = mount(submit)
  view.rerender(
    <EditMessageForm {...view.props} value="Changed in another view" />
  )
  expect(view.editor).toHaveValue("Exact authored edit  \n")
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  expect(submit).not.toHaveBeenCalled()
  expect(view.close).not.toHaveBeenCalled()
  expect(screen.getByRole("alert")).toHaveTextContent("message changed")
})

it("one in-flight attempt prevents repeated submit before a render", async () => {
  const pending = deferred()
  const submit = vi.fn(() => pending.promise)
  const view = mount(submit)
  const form = view.editor.closest("form")!
  act(() => {
    fireEvent.submit(form)
    fireEvent.submit(form)
  })
  await act(async () => {
    pending.resolve({ status: "submitted" })
    await pending.promise
  })
  expect(submit).toHaveBeenCalledOnce()
})

it("composition does not submit an unfinished edit", () => {
  const submit = vi.fn()
  const view = mount(submit)
  fireEvent.compositionStart(view.editor)
  fireEvent.submit(view.editor.closest("form")!)
  expect(submit).not.toHaveBeenCalled()
})

it("synchronous legacy saves still close after completion", async () => {
  const view = mount()
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  await waitFor(() => expect(view.close).toHaveBeenCalledOnce())
})

it("accepts the current write's matching canonical value without erasing newer input", async () => {
  const held = deferred()
  const view = mount(() => held.promise)
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  fireEvent.change(view.editor, { target: { value: "Newer authored edit" } })
  view.rerender(
    <EditMessageForm {...view.props} value={"Exact authored edit  \n"} />
  )
  await act(async () => {
    held.resolve({ status: "submitted" })
    await held.promise
  })
  expect(view.close).not.toHaveBeenCalled()
  expect(view.editor).toHaveValue("Newer authored edit")
  expect(screen.queryByRole("alert")).toBeNull()
})

it("a matching canonical value arriving after acknowledgement keeps newer input saveable", async () => {
  const held = deferred()
  const submit = vi.fn(() => held.promise)
  const view = mount(submit)
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  fireEvent.change(view.editor, { target: { value: "Newer authored edit" } })
  await act(async () => {
    held.resolve({ status: "submitted" })
    await held.promise
  })
  view.rerender(
    <EditMessageForm {...view.props} value={"Exact authored edit  \n"} />
  )
  expect(view.editor).toHaveValue("Newer authored edit")
  expect(screen.queryByRole("alert")).toBeNull()
  submit.mockResolvedValue({ status: "submitted" })
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  await waitFor(() => expect(view.close).toHaveBeenCalledOnce())
  expect(submit).toHaveBeenLastCalledWith(
    "Newer authored edit",
    false,
    expect.any(Function)
  )
})

it("canonical edit/undo cannot certify an old held acknowledgement", async () => {
  const held = deferred()
  const view = mount(() => held.promise)
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  view.rerender(<EditMessageForm {...view.props} value="Other view update" />)
  view.rerender(<EditMessageForm {...view.props} />)
  await act(async () => {
    held.resolve({ status: "submitted" })
    await held.promise
  })
  expect(view.close).not.toHaveBeenCalled()
  expect(view.editor).toHaveValue("Exact authored edit  \n")
  expect(screen.getByRole("alert")).toHaveTextContent("message changed")
})

it("an external canonical transition retires an expected delayed acknowledgement", async () => {
  const held = deferred()
  const submit = vi.fn(() => held.promise)
  const view = mount(submit)
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  fireEvent.change(view.editor, { target: { value: "Newer authored draft" } })
  await act(async () => {
    held.resolve({ status: "submitted" })
    await held.promise
  })
  view.rerender(
    <EditMessageForm {...view.props} value="External canonical update" />
  )
  expect(screen.getByRole("alert")).toHaveTextContent("message changed")
  view.rerender(
    <EditMessageForm {...view.props} value={"Exact authored edit  \n"} />
  )
  expect(screen.getByRole("alert")).toHaveTextContent("message changed")
  fireEvent.click(screen.getByRole("button", { name: "Save", exact: true }))
  expect(submit).toHaveBeenCalledOnce()
  expect(view.close).not.toHaveBeenCalled()
  expect(view.editor).toHaveValue("Newer authored draft")
})


it.each(["typing", "undo", "cancel"])(
  "current editor follow lease retires after %s",
  async (change) => {
    const held = deferred()
    const submit = vi.fn((..._args: unknown[]) => held.promise)
    const view = mount(submit)
    fireEvent.click(
      screen.getByRole("button", { name: "Save & Send", exact: true })
    )
    try {
      const lease = submit.mock.calls[0]?.[2]
      expect(typeof lease).toBe("function")
      const isCurrent = lease as () => boolean
      expect(isCurrent()).toBe(true)
      if (change === "cancel")
        fireEvent.click(
          screen.getByRole("button", { name: "Cancel", exact: true })
        )
      else {
        fireEvent.change(view.editor, {
          target: { value: "Newer authored edit" }
        })
        if (change === "undo")
          fireEvent.change(view.editor, {
            target: { value: "Exact authored edit  \n" }
          })
      }
      expect(isCurrent()).toBe(false)
    } finally {
      await act(async () => {
        held.resolve({ status: "submitted" })
        await held.promise
      })
    }
  }
)
