import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, expect, it, vi } from 'vitest'
import { createQueryClient } from '@/services/query-client'
import { useVlmBackends } from '@web/hooks/useVlmBackends'

const io = vi.hoisted(() => ({ get: vi.fn() }))
vi.mock('@web/lib/api', () => ({
  apiClient: { get: (...args: unknown[]) => io.get(...args) },
}))

const CAPABILITIES_URL = '/rag/capabilities'
const DEFAULT_BACKENDS_URL = '/rag/vlm/backends'

const capabilitiesWithEndpoint = (endpoint: string) => ({
  features: { vlm_late_chunking: { backends_endpoint: endpoint } },
})

const setup = () => {
  const client = createQueryClient()
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  return { ...renderHook(() => useVlmBackends(), { wrapper }), client }
}

beforeEach(() => {
  io.get.mockReset()
  io.get.mockImplementation(async (url: string) => {
    if (url === CAPABILITIES_URL) return capabilitiesWithEndpoint('/api/v1/rag/vlm/backends')
    if (url === DEFAULT_BACKENDS_URL) return { backends: { lmserve: { available: true }, native: false } }
    throw new Error(`Unexpected URL: ${url}`)
  })
})

it('discovers the endpoint from capabilities, then reads the backends map once', async () => {
  const { result } = setup()
  await waitFor(() => expect(result.current.loading).toBe(false))
  expect(io.get).toHaveBeenCalledTimes(2)
  expect(io.get).toHaveBeenNthCalledWith(1, CAPABILITIES_URL)
  expect(io.get).toHaveBeenNthCalledWith(2, DEFAULT_BACKENDS_URL)
  expect(result.current.endpoint).toBe('/api/v1/rag/vlm/backends')
  expect(result.current.backends).toEqual({ lmserve: true, native: false })
  expect(result.current.error).toBeNull()
})

it('falls back to the default endpoint when capabilities omit the feature', async () => {
  io.get.mockImplementation(async (url: string) => {
    if (url === CAPABILITIES_URL) return { features: {} }
    if (url === DEFAULT_BACKENDS_URL) return { backends: {} }
    throw new Error(`Unexpected URL: ${url}`)
  })
  const { result } = setup()
  await waitFor(() => expect(result.current.loading).toBe(false))
  expect(result.current.endpoint).toBe('/api/v1/rag/vlm/backends')
  expect(result.current.backends).toEqual({})
  expect(io.get).toHaveBeenNthCalledWith(2, DEFAULT_BACKENDS_URL)
})

it('two mounts within the stale window issue one capabilities read and one backends read', async () => {
  const client = createQueryClient()
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const first = renderHook(() => useVlmBackends(), { wrapper })
  await waitFor(() => expect(first.result.current.loading).toBe(false))
  expect(io.get).toHaveBeenCalledTimes(2)
  first.unmount()

  const second = renderHook(() => useVlmBackends(), { wrapper })
  await waitFor(() => expect(second.result.current.loading).toBe(false))
  expect(second.result.current.backends).toEqual({ lmserve: true, native: false })
  expect(io.get).toHaveBeenCalledTimes(2)
  second.unmount()
  client.clear()
})

it('an explicit reload forces both reads again', async () => {
  const { result } = setup()
  await waitFor(() => expect(result.current.loading).toBe(false))
  expect(io.get).toHaveBeenCalledTimes(2)
  await act(async () => {
    result.current.reload()
  })
  await waitFor(() => expect(io.get).toHaveBeenCalledTimes(4))
  expect(io.get).toHaveBeenNthCalledWith(3, CAPABILITIES_URL)
  expect(io.get).toHaveBeenNthCalledWith(4, DEFAULT_BACKENDS_URL)
})

it('a capabilities failure surfaces an error and skips the backends read', async () => {
  io.get.mockImplementation(async (url: string) => {
    if (url === CAPABILITIES_URL) throw new Error('capabilities unavailable')
    throw new Error(`Unexpected URL: ${url}`)
  })
  const { result } = setup()
  await waitFor(() => expect(result.current.error).toBe('capabilities unavailable'))
  expect(result.current.loading).toBe(false)
  expect(result.current.backends).toBeNull()
  expect(io.get).toHaveBeenCalledTimes(1)
})
