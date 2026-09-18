import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useFirstRunCheck } from "../useFirstRunCheck"

const mocks = vi.hoisted(() => ({ profiles: vi.fn(), setup: vi.fn() }))
vi.mock("@/services/api-send", () => ({ apiSend: mocks.profiles }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getFirstRunState: mocks.setup }
}))
const cache = vi.hoisted(() => ({ status: null as string | null }))
vi.mock("@/hooks/useSetupOnboarding", () => ({
  getCachedFirstRunSetupStatus: () => cache.status
}))

describe("useFirstRunCheck", () => {
  beforeEach(() => {
    localStorage.clear()
    mocks.profiles.mockReset().mockResolvedValue({ ok: true, data: [] })
    mocks.setup.mockReset().mockResolvedValue({ status: "completed" })
    cache.status = "completed"
  })

  it("does not treat a failed profile request as an empty profile list", async () => {
    mocks.profiles.mockResolvedValue({ ok: false, status: 403 })
    const { result } = renderHook(() => useFirstRunCheck())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.shouldShowSetup).toBe(false)
  })

  it("checks once per enabled mount and never checks a disabled remount", async () => {
    const first = renderHook(() => useFirstRunCheck())
    await waitFor(() => expect(first.result.current.loading).toBe(false))
    first.rerender()
    first.rerender()
    expect(mocks.profiles).toHaveBeenCalledTimes(1)
    first.unmount()
    const disabled = renderHook(() => useFirstRunCheck({ enabled: false }))
    expect(mocks.profiles).toHaveBeenCalledTimes(1)
    disabled.unmount()
    const second = renderHook(() => useFirstRunCheck())
    await waitFor(() => expect(second.result.current.loading).toBe(false))
    expect(mocks.profiles).toHaveBeenCalledTimes(2)
  })

  it("waits without fetching or showing setup until enabled", async () => {
    const { result, rerender } = renderHook(
      ({ enabled }) => useFirstRunCheck({ enabled }),
      { initialProps: { enabled: false } }
    )
    expect(mocks.profiles).not.toHaveBeenCalled()
    expect(result.current).toEqual({ shouldShowSetup: false, resumeStep: null, loading: false })
    rerender({ enabled: true })
    await waitFor(() => expect(result.current.shouldShowSetup).toBe(true))
    expect(mocks.profiles).toHaveBeenCalledTimes(1)
  })

  it("clears resume state when disabled and ignores an old response after re-enabling", async () => {
    let resolveOld!: (value: unknown) => void
    const oldResponse = new Promise(resolve => { resolveOld = resolve })
    mocks.profiles.mockReturnValueOnce(oldResponse)
    const { result, rerender } = renderHook(
      ({ enabled }) => useFirstRunCheck({ enabled }),
      { initialProps: { enabled: true } }
    )
    rerender({ enabled: false })
    expect(result.current).toEqual({ shouldShowSetup: false, resumeStep: null, loading: false })
    mocks.profiles.mockResolvedValueOnce({ ok: true, data: [{ setup: { status: "in_progress", current_step: "voice" } }] })
    rerender({ enabled: true })
    await waitFor(() => expect(result.current.resumeStep).toBe("voice"))
    await act(async () => { resolveOld({ ok: true, data: [] }); await oldResponse })
    expect(result.current).toEqual({ shouldShowSetup: false, resumeStep: "voice", loading: false })
    rerender({ enabled: false })
    expect(result.current).toEqual({ shouldShowSetup: false, resumeStep: null, loading: false })
  })

  it("allows an opted-in source destination after unified setup completes", async () => {
    const { result } = renderHook(() =>
      useFirstRunCheck({ allowCompletedSetup: true })
    )
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.shouldShowSetup).toBe(false)
  })

  it("keeps assistant setup available on other routes", async () => {
    const { result } = renderHook(() => useFirstRunCheck())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.shouldShowSetup).toBe(true)
    expect(mocks.setup).not.toHaveBeenCalled()
  })

  it("does not let another server's incomplete cache block source destinations", async () => {
    cache.status = "in_progress"
    mocks.setup.mockResolvedValue({ status: "in_progress" })
    const { result } = renderHook(() =>
      useFirstRunCheck({ allowCompletedSetup: true })
    )
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.shouldShowSetup).toBe(false)
  })

  it("keeps remote source navigation nonblocking without privileged setup probes", async () => {
    cache.status = null
    mocks.setup.mockRejectedValue(new Error("HTTP 403: first-run disabled"))
    const { result, rerender } = renderHook(
      ({ allowCompletedSetup }) => useFirstRunCheck({ allowCompletedSetup }),
      { initialProps: { allowCompletedSetup: true } }
    )
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.shouldShowSetup).toBe(false)
    rerender({ allowCompletedSetup: false })
    await waitFor(() => expect(result.current.shouldShowSetup).toBe(true))
    rerender({ allowCompletedSetup: true })
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.shouldShowSetup).toBe(false)
    expect(mocks.setup).not.toHaveBeenCalled()
  })
})
