import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  notification: undefined as Record<string, unknown> | undefined,
  fallback: { open: vi.fn(), destroy: vi.fn() }
}))
vi.mock("antd", () => ({
  App: { useApp: () => ({ notification: mocks.notification }) },
  notification: mocks.fallback
}))

import { useAntdNotification } from "../useAntdNotification"

describe("useAntdNotification", () => {
  beforeEach(() => {
    mocks.notification = undefined
    vi.clearAllMocks()
  })

  it("adapts frozen APIs without changing other consumers' methods", () => {
    const delivered: unknown[] = []
    const base = Object.freeze({
      open(config: unknown) {
        expect(this).toBe(base)
        delivered.push(config)
      },
      destroy: vi.fn()
    })
    mocks.notification = base
    const { result, rerender } = renderHook(() => useAntdNotification())
    const first = result.current
    result.current.success({ message: "Saved", description: "Ready" })
    result.current.warning({ title: "Current", message: "Legacy" })
    rerender()
    expect(result.current).toBe(first)
    expect(delivered).toEqual([
      { title: "Saved", description: "Ready", type: "success" },
      { title: "Current", type: "warning" }
    ])
    expect(base).not.toHaveProperty("success")
  })

  it("normalizes existing methods while preserving receiver and destroy", () => {
    const delivered: unknown[] = []
    const base = {
      open(config: unknown) { delivered.push(config) },
      success(config: unknown) { expect(this).toBe(base); delivered.push(config) },
      destroy(key?: unknown) { expect(this).toBe(base); delivered.push(key) }
    }
    const originalSuccess = base.success
    mocks.notification = base
    const { result } = renderHook(() => useAntdNotification())
    result.current.success({ message: "Saved" })
    result.current.open({ title: "Direct", message: "Legacy" })
    result.current.destroy("notice")
    expect(delivered).toEqual([{ title: "Saved" }, { title: "Direct" }, "notice"])
    expect(base.success).toBe(originalSuccess)
  })

  it("uses a normalized static fallback when context has no open method", () => {
    mocks.notification = {}
    const { result } = renderHook(() => useAntdNotification())
    result.current.error({ message: "Failed" })
    expect(mocks.fallback.open).toHaveBeenCalledWith({ title: "Failed", type: "error" })
  })

  it("uses the replacement context API after a provider change", () => {
    const firstOpen = vi.fn()
    const secondOpen = vi.fn()
    mocks.notification = { open: firstOpen }
    const { result, rerender } = renderHook(() => useAntdNotification())
    mocks.notification = { open: secondOpen }
    rerender()
    result.current.info({ message: "New provider" })
    expect(firstOpen).not.toHaveBeenCalled()
    expect(secondOpen).toHaveBeenCalledWith({ title: "New provider", type: "info" })
  })
})
