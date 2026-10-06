import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, expect, it, vi } from "vitest"
import { createQueryClient } from "@/services/query-client"
import { useChatModelCatalog } from "../useChatModelCatalog"

const io = vi.hoisted(() => ({ fetch: vi.fn(), generation: 0, listeners: new Set<() => void>() }))
vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: (...args: unknown[]) => io.fetch(...args)
}))
vi.mock("@/services/chat-models-invalidation", () => ({
  getChatModelsCacheGeneration: () => io.generation,
  subscribeChatModelsInvalidation: (listener: () => void) => { io.listeners.add(listener); return () => io.listeners.delete(listener) }
}))
const invalidate = () => { io.generation++; io.listeners.forEach(listener => listener()) }
const setup = () => {
  const client = createQueryClient()
  const wrapper = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return { ...renderHook(() => useChatModelCatalog(), { wrapper }), client }
}
beforeEach(() => { io.fetch.mockReset(); io.generation = 0; io.listeners.clear() })

it("refreshes a continuously mounted catalog after setup invalidation", async () => {
  io.fetch.mockResolvedValueOnce([]).mockResolvedValueOnce([{ model: "configured" }])
  const { result } = setup()
  await waitFor(() => expect(result.current.isSuccess).toBe(true))
  act(invalidate)
  await waitFor(() => expect(result.current.data).toEqual([{ model: "configured" }]))
})

it("withholds old account data and ignores a held old discovery", async () => {
  let oldResolve!: (value: unknown[]) => void
  let currentResolve!: (value: unknown[]) => void
  io.fetch.mockReturnValueOnce(new Promise(resolve => { oldResolve = resolve }))
    .mockReturnValueOnce(new Promise(resolve => { currentResolve = resolve }))
  const { result } = setup()
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(1))
  act(invalidate)
  expect(result.current.data).toBeUndefined()
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(2))
  await act(async () => oldResolve([{ model: "account-a" }]))
  expect(result.current.data).toBeUndefined()
  await act(async () => currentResolve([{ model: "account-b" }]))
  await waitFor(() => expect(result.current.data).toEqual([{ model: "account-b" }]))
})

it("retains failure state until an explicit Retry obtains an authoritative catalog", async () => {
  io.fetch.mockRejectedValueOnce(new Error("Metadata unavailable"))
    .mockResolvedValueOnce([{ model: "configured" }])
  const { result } = setup()
  await waitFor(() => expect(result.current.isError).toBe(true))
  expect(result.current.data).toBeUndefined()
  await act(async () => { await result.current.refetch() })
  await waitFor(() => expect(result.current.data).toEqual([{ model: "configured" }]))
})

it("withholds retained successful data when rediscovery fails", async () => {
  io.fetch.mockResolvedValueOnce([{ model: "previously-verified" }])
    .mockRejectedValueOnce(new Error("Metadata unavailable"))
  const { result } = setup()
  await waitFor(() => expect(result.current.data).toEqual([{ model: "previously-verified" }]))
  await act(async () => { await result.current.refetch() })
  await waitFor(() => expect(result.current.isError).toBe(true))
  expect(result.current.data).toBeUndefined()
})
