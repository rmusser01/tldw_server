import { usePageAssist } from "@/context"
import { act, renderHook } from "@testing-library/react"
import React from "react"
import { describe, expect, it } from "vitest"

import { PageAssistProvider } from "../PageAssistProvider"

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <PageAssistProvider>{children}</PageAssistProvider>
)

describe("shared response controller ownership", () => {
  it("exposes the current owner synchronously before another hook render", () => {
    const hook = renderHook(() => usePageAssist(), { wrapper })
    const owner = new AbortController()
    act(() => {
      hook.result.current.setController(owner)
      expect(hook.result.current.getController?.()).toBe(owner)
    })
    expect(hook.result.current.controller).toBe(owner)
  })
  it("an older functional release cannot clear a newer controller in the same event", () => {
    const hook = renderHook(() => usePageAssist(), { wrapper }),
      old = new AbortController(),
      current = new AbortController()
    act(() => {
      hook.result.current.setController(old)
      hook.result.current.setController(current)
      hook.result.current.setController((value) =>
        value === old ? null : value
      )
      expect(hook.result.current.getController?.()).toBe(current)
    })
    expect(hook.result.current.controller).toBe(current)
  })
  it("a retained getter sees replacement and explicit release across renders", () => {
    const hook = renderHook(() => usePageAssist(), { wrapper }),
      getter = hook.result.current.getController
    act(() => hook.result.current.setController(new AbortController()))
    expect(getter?.()).toBe(hook.result.current.controller)
    act(() => hook.result.current.setController(null))
    expect(getter?.()).toBeNull()
  })
})
