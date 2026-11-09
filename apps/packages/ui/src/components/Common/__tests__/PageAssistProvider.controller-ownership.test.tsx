import { usePageAssist } from "@/context"
import { act, render, renderHook, waitFor } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import { PageAssistProvider } from "../PageAssistProvider"

const debugStore = vi.hoisted(() => ({ messages: [], setMessages: vi.fn() }))
vi.mock("@/store/option", () => ({
  useStoreMessageOption: { getState: () => debugStore }
}))

type DebugWindow = Window & { __tldw_pageAssist?: unknown }

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
  it("an older provider cannot remove the newer development bridge", async () => {
    const first = render(<PageAssistProvider><span>First</span></PageAssistProvider>)
    await waitFor(() => expect((window as DebugWindow).__tldw_pageAssist).toBeDefined())
    const previous = (window as DebugWindow).__tldw_pageAssist
    const second = render(<PageAssistProvider><span>Second</span></PageAssistProvider>)
    await waitFor(() => expect((window as DebugWindow).__tldw_pageAssist).not.toBe(previous))
    const current = (window as DebugWindow).__tldw_pageAssist
    first.unmount()
    expect((window as DebugWindow).__tldw_pageAssist).toBe(current)
    second.unmount()
    expect((window as DebugWindow).__tldw_pageAssist).toBeUndefined()
  })

})
