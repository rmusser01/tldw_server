import React from "react"
import { render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import userEvent from "@testing-library/user-event"
import { JsonEditor } from "../JsonEditor"

it("renders and updates exact JSON without spreading React keys", () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  try {
    const first = '{\n  "case_sensitive": false\n}'
    const second = '{\n  "case_sensitive": true,\n  "metrics": ["exact_match"]\n}'
    const { container, rerender } = render(<JsonEditor value={first} onChange={() => {}} />)
    expect(Array.from(container.querySelectorAll("pre > div"), line => line.textContent)).toEqual(["{", '  "case_sensitive": false', "}"])
    rerender(<JsonEditor value={second} onChange={() => {}} />)
    expect(screen.getByRole("textbox")).toHaveValue(second)
    expect(Array.from(container.querySelectorAll("pre > div"), line => line.textContent)).toEqual(["{", '  "case_sensitive": true,', '  "metrics": ["exact_match"]', "}"])
    expect(errors.mock.calls.filter(args => String(args[0]).includes('"key" prop'))).toEqual([])
  } finally {
    errors.mockRestore()
  }
})


it("keeps the preview mounted while focus moves to the following action", async () => {
  const user = userEvent.setup()
  const submit = vi.fn()
  const value = '[{"input":{"output":"ORBIT-742"}}]'
  const { container } = render(<>
    <JsonEditor value={value} onChange={() => {}} />
    <button onClick={submit}>Create</button>
  </>)
  const preview = container.querySelector("pre")
  await user.click(screen.getByRole("textbox"))
  expect(preview).toBeVisible()
  await user.click(screen.getByRole("button", { name: "Create" }))
  expect(submit).toHaveBeenCalledTimes(1)
  expect(container.querySelector("pre")).toBe(preview)
})
