import { renderHook, waitFor } from "@testing-library/react"
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
