import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useMediaCapabilities } from "../useMediaCapabilities"
import { bgRequest } from "@/services/background-proxy"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))

describe("Media delete affordance permissions", () => {
  beforeEach(() => vi.clearAllMocks())

  it("stays unavailable until the server confirms permission", async () => {
    vi.mocked(bgRequest).mockResolvedValue({ can_delete: false })
    const { result } = renderHook(() => useMediaCapabilities())
    expect(result.current.canDelete).toBe(false)
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.canDelete).toBe(false)
  })

  it("forgets the old permission immediately at an account boundary", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ can_delete: true }).mockImplementationOnce(() => new Promise(() => {}))
    const { result } = renderHook(() => useMediaCapabilities())
    await waitFor(() => expect(result.current.canDelete).toBe(true))
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")))
    expect(result.current.canDelete).toBe(false)
  })
})
