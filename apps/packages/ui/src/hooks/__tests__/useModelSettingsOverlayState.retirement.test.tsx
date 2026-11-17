// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { expect, it } from "vitest"
import { useModelSettingsOverlayState } from "../useModelSettingsOverlayState"

it("retires the editor and its outstanding native completion when its layout source leaves", () => {
  const hook = renderHook(() => useModelSettingsOverlayState())
  act(() => { hook.result.current.setOpen(true) })
  act(() => { hook.result.current.setOpen(false) })
  const finish = hook.result.current.afterClose
  const focusIsCurrent = hook.result.current.isCurrent
  act(() => { hook.result.current.retire() })
  expect(hook.result.current.present).toBe(false)
  expect(focusIsCurrent()).toBe(false)
  act(() => { expect(finish()).toBe(false) })
  act(() => { hook.result.current.setOpen(true) })
  expect(hook.result.current.open).toBe(true)
  hook.unmount()
})

it("keeps the same close completion after a repeated close request", () => {
  const hook = renderHook(() => useModelSettingsOverlayState())
  act(() => { hook.result.current.setOpen(true) })
  act(() => { hook.result.current.setOpen(false) })
  const finish = hook.result.current.afterClose
  act(() => { hook.result.current.setOpen(false) })
  act(() => { expect(finish()).toBe(true) })
  expect(hook.result.current.present).toBe(false)
  hook.unmount()
})
