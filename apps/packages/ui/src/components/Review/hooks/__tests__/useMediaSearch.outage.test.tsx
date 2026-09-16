import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import { useMediaSearch } from "../useMediaSearch"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))
vi.mock("@plasmohq/storage", () => ({
  Storage: class {
    get = async () => undefined
    set = async () => undefined
  }
}))

const t = (key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue ?? key)

const mountSearch = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const feedback: string[] = []
  const message = {
    error: (text: string) => feedback.push(text),
    warning: (text: string) => feedback.push(text)
  }
  const hook = renderHook(() => useMediaSearch({ t, message }), {
    wrapper: ({ children }: React.PropsWithChildren) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  })
  return { ...hook, feedback, client }
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(bgRequest).mockResolvedValue({ items: [], pagination: { total_items: 0 } })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("Media search outage handling", () => {
  it("keeps a transport outage recoverable without reporting a runtime exception", async () => {
    const runtimeErrors = vi.spyOn(console, "error").mockImplementation(() => {})
    vi.spyOn(console, "warn").mockImplementation(() => {})
    const outage = Object.assign(new Error("Failed to fetch (GET /api/v1/media)"), { status: 0 })
    vi.mocked(bgRequest).mockRejectedValue(outage)
    const { result, feedback } = mountSearch()

    await waitFor(() => expect(feedback).toContain("Failed to search media"))
    expect(result.current.mediaApiUnavailable).toBe(false)
    expect(runtimeErrors).not.toHaveBeenCalled()

    vi.mocked(bgRequest).mockResolvedValue({
      items: [{ id: 7, title: "Preserved Rowan source" }],
      pagination: { total_items: 1 }
    })
    await act(async () => { await result.current.refetch() })
    expect(result.current.results.map(item => item.title)).toEqual(["Preserved Rowan source"])
  })

  it("keeps missing-endpoint guidance distinct from a transport outage", async () => {
    vi.mocked(bgRequest).mockRejectedValue(Object.assign(new Error("Not found (GET /api/v1/media/)"), { status: 404 }))
    const { result } = mountSearch()
    await waitFor(() => expect(result.current.mediaApiUnavailable).toBe(true))
  })

  it("retains diagnostics for unexpected failures", async () => {
    const runtimeErrors = vi.spyOn(console, "error").mockImplementation(() => {})
    const unexpected = new Error("Unexpected media mapping failure")
    vi.mocked(bgRequest).mockRejectedValue(unexpected)
    const { feedback } = mountSearch()
    await waitFor(() => expect(feedback).toContain("Failed to search media"))
    expect(runtimeErrors).toHaveBeenCalledWith("Media search error:", unexpected)
  })
})
