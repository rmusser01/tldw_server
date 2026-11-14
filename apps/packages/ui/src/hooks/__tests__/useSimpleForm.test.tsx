import { act, renderHook } from "@testing-library/react"
import { expect, it } from "vitest"
import { useSimpleForm } from "../useSimpleForm"

it("fences batched edit-and-erase and reset without waiting for a React commit", () => {
  const { result } = renderHook(() => useSimpleForm({ initialValues: { message: "", image: "" } }))
  const form = result.current
  const revision = form.getRevision()
  act(() => {
    form.setFieldValue("message", "new input")
    form.setFieldValue("message", "")
    expect(form.getValues()).toEqual({ message: "", image: "" })
    expect(form.getRevision()).toBe(revision + 2)
    form.setValues({ image: "image" })
    form.reset()
    expect(form.getRevision()).toBe(revision + 4)
  })
  expect(result.current.values).toEqual(form.getValues())
})
