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

it("a connected admission observer retries an earlier shared discovery failure without picker engagement", async () => {
  let complete!: (models: unknown[]) => void
  io.fetch.mockRejectedValueOnce(new Error("Connection not ready"))
    .mockReturnValueOnce(new Promise(resolve => { complete = resolve }))
  const client = createQueryClient()
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(({ connected }) => ({
    page: useChatModelCatalog(), admission: useChatModelCatalog(connected)
  }), { initialProps: { connected: false }, wrapper })
  await waitFor(() => expect(hook.result.current.page.isError).toBe(true))
  expect(io.fetch).toHaveBeenCalledTimes(1)
  expect(hook.result.current.admission.data).toBeUndefined()
  hook.rerender({ connected: true })
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(2))
  expect(hook.result.current.admission.isFetching).toBe(true)
  expect(hook.result.current.admission.data).toBeUndefined()
  await act(async () => complete([{ model: "gpt-4.1-mini", provider: "openai" }]))
  await waitFor(() => expect(hook.result.current.admission.data).toEqual([{ model: "gpt-4.1-mini", provider: "openai" }]))
  expect(hook.result.current.admission.isSuccess).toBe(true)
  expect(io.fetch).toHaveBeenLastCalledWith({ requireFresh: true })
  hook.unmount()
  client.clear()
})

it("activation verifies an early empty shared catalog instead of treating its fresh cache as ready", async () => {
  let complete!: (models: unknown[]) => void
  io.fetch.mockResolvedValueOnce([])
    .mockReturnValueOnce(new Promise(resolve => { complete = resolve }))
  const client = createQueryClient()
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(({ connected }) => ({
    page: useChatModelCatalog(), admission: useChatModelCatalog(connected)
  }), { initialProps: { connected: false }, wrapper })
  await waitFor(() => expect(hook.result.current.page.isSuccess).toBe(true))
  expect(hook.result.current.page.data).toEqual([])
  expect(io.fetch).toHaveBeenCalledTimes(1)
  hook.rerender({ connected: true })
  expect(hook.result.current.admission.data).toBeUndefined()
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(2))
  expect(hook.result.current.admission.isFetching).toBe(true)
  await act(async () => complete([{ model: "gpt-4.1-mini", provider: "openai" }]))
  await waitFor(() => expect(hook.result.current.admission.data).toEqual([{ model: "gpt-4.1-mini", provider: "openai" }]))
  hook.unmount()
  client.clear()
})

it("an old activation completion cannot certify a newer account's held catalog", async () => {
  let oldComplete!: (models: unknown[]) => void, currentComplete!: (models: unknown[]) => void
  io.fetch.mockResolvedValueOnce([{ model: "previous" }])
    .mockReturnValueOnce(new Promise(resolve => { oldComplete = resolve }))
    .mockReturnValueOnce(new Promise(resolve => { currentComplete = resolve }))
  const client = createQueryClient()
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(({ connected }) => useChatModelCatalog(connected), { initialProps: { connected: false }, wrapper })
  await client.fetchQuery({ queryKey: ["playground:chatModels", 0], queryFn: () => io.fetch({ requireFresh: true }) })
  hook.rerender({ connected: true })
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(2))
  act(invalidate)
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(3))
  await act(async () => oldComplete([{ model: "old-account" }]))
  expect(hook.result.current.data).toBeUndefined()
  expect(hook.result.current.isFetching).toBe(true)
  await act(async () => currentComplete([{ model: "current-account" }]))
  await waitFor(() => expect(hook.result.current.data).toEqual([{ model: "current-account" }]))
  hook.unmount()
  client.clear()
})

it("failed activation withholds a retained previously successful catalog", async () => {
  io.fetch.mockResolvedValueOnce([{ model: "previous" }]).mockRejectedValueOnce(new Error("Discovery failed"))
  const client = createQueryClient()
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(({ connected }) => useChatModelCatalog(connected), { initialProps: { connected: false }, wrapper })
  await client.fetchQuery({ queryKey: ["playground:chatModels", 0], queryFn: () => io.fetch({ requireFresh: true }) })
  hook.rerender({ connected: true })
  await waitFor(() => expect(hook.result.current.isError).toBe(true))
  await waitFor(() => expect(hook.result.current.isFetching).toBe(false))
  expect(hook.result.current.data).toBeUndefined()
  expect(hook.result.current.isSuccess).toBe(false)
  hook.unmount()
  client.clear()
})

it("an initially connected observer verifies a fresh cached empty result", async () => {
  let complete!: (models: unknown[]) => void
  io.fetch.mockReturnValueOnce(new Promise(resolve => { complete = resolve }))
  const client = createQueryClient()
  client.setQueryData(["playground:chatModels", 0], [])
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(() => useChatModelCatalog(true), { wrapper })
  expect(hook.result.current.data).toBeUndefined()
  await waitFor(() => expect(io.fetch).toHaveBeenCalledOnce())
  await act(async () => complete([{ model: "verified-current" }]))
  await waitFor(() => expect(hook.result.current.data).toEqual([{ model: "verified-current" }]))
  hook.unmount()
  client.clear()
})

it("activation cannot certify an earlier same-generation request's empty result", async () => {
  let oldComplete!: (models: unknown[]) => void, currentComplete!: (models: unknown[]) => void
  io.fetch.mockReturnValueOnce(new Promise(resolve => { oldComplete = resolve }))
    .mockReturnValueOnce(new Promise(resolve => { currentComplete = resolve }))
  const client = createQueryClient()
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(({ connected }) => ({ page: useChatModelCatalog(), admission: useChatModelCatalog(connected) }),
    { initialProps: { connected: false }, wrapper })
  await waitFor(() => expect(io.fetch).toHaveBeenCalledOnce())
  hook.rerender({ connected: true })
  expect(hook.result.current.admission.data).toBeUndefined()
  await act(async () => oldComplete([]))
  await waitFor(() => expect(io.fetch).toHaveBeenCalledTimes(2))
  expect(hook.result.current.admission.data).toBeUndefined()
  expect(hook.result.current.admission.isFetching).toBe(true)
  await act(async () => currentComplete([{ model: "post-activation" }]))
  await waitFor(() => expect(hook.result.current.admission.data).toEqual([{ model: "post-activation" }]))
  hook.unmount()
  client.clear()
})

it("an incidental enabled mount reuses a verified nonempty current catalog without another read", () => {
  const client = createQueryClient()
  const models = [{ model: "verified-current", provider: "openai" }]
  client.setQueryData(["playground:chatModels", 0], models)
  const wrapper = ({ children }: React.PropsWithChildren<object>) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const hook = renderHook(() => useChatModelCatalog(true), { wrapper })
  expect(hook.result.current.data).toEqual(models)
  expect(hook.result.current.isSuccess).toBe(true)
  expect(io.fetch).not.toHaveBeenCalled()
  hook.unmount()
  client.clear()
})
