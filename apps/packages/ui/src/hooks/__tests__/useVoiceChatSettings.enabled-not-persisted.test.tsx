import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const seedStale = () => {
  for (const area of [window.localStorage, window.sessionStorage]) {
    for (const key of [
      "voiceChatEnabled",
      "tldw:voiceChatEnabled",
      "local::voiceChatEnabled",
      "session::voiceChatEnabled"
    ]) {
      area.setItem(key, "true")
      area.setItem(key, JSON.stringify(true))
    }
  }
}

const load = async () =>
  (await import("@/hooks/useVoiceChatSettings")).useVoiceChatSettings

describe("voiceChatEnabled is in-memory only", () => {
  beforeEach(() => {
    vi.resetModules()
    window.localStorage.clear()
    window.sessionStorage.clear()
  })

  it("reads false even with stale true in local and session storage", async () => {
    seedStale()
    const useHook = await load()
    const { result } = renderHook(() => useHook())
    await act(async () => {})
    expect(result.current.voiceChatEnabled).toBe(false)
  })

  it("shares a toggle between two consumers in one page", async () => {
    const useHook = await load()
    const a = renderHook(() => useHook())
    const b = renderHook(() => useHook())
    act(() => a.result.current.setVoiceChatEnabled(true))
    expect(b.result.current.voiceChatEnabled).toBe(true)
    act(() => b.result.current.setVoiceChatEnabled(!b.result.current.voiceChatEnabled))
    expect(a.result.current.voiceChatEnabled).toBe(false)
  })

  it("is false again after a simulated reload with storage intact", async () => {
    const first = await load()
    const a = renderHook(() => first())
    act(() => a.result.current.setVoiceChatEnabled(true))
    a.unmount()
    vi.resetModules()
    const second = await load()
    const b = renderHook(() => second())
    await act(async () => {})
    expect(b.result.current.voiceChatEnabled).toBe(false)
  })
})
