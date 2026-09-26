import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor, act } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const fetchWithTldwAuthMock = vi.hoisted(() => vi.fn())
const getTldwServerURLMock = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/auth-fetch", () => ({
  fetchWithTldwAuth: fetchWithTldwAuthMock
}))

vi.mock("@/services/tldw-server", () => ({
  getTldwServerURL: getTldwServerURLMock
}))

import {
  usePrototypePrivateLinkExchange,
  useSharedWithMe
} from "@/hooks/useSharing"
import { getPrototypeContractState } from "@/test-utils/prototype-contract-fixtures"

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useSharing auth wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getTldwServerURLMock.mockResolvedValue("http://127.0.0.1:8000")
  })

  it("loads shared workspaces through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(JSON.stringify({ items: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    const { result } = renderHook(() => useSharedWithMe("server:one|user:42"), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/sharing/shared-with-me",
      { signal: expect.any(AbortSignal) }
    )
  })

  it("does not fetch recipient data without a verified scope", () => {
    renderHook(() => useSharedWithMe(null), { wrapper: buildWrapper() })
    expect(fetchWithTldwAuthMock).not.toHaveBeenCalled()
  })

  it("does not reuse cached recipient data across scopes", async () => {
    fetchWithTldwAuthMock.mockImplementation(async () => new Response(JSON.stringify({ items: [{ workspace_name: "Private" }], total: 1 })))
    const { result, rerender } = renderHook(({ scope }: { scope: string | null }) => useSharedWithMe(scope), {
      wrapper: buildWrapper(), initialProps: { scope: "server:one|user:42" as string | null }
    })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    rerender({ scope: null })
    expect(result.current.data).toBeUndefined()
    fetchWithTldwAuthMock.mockImplementation(() => new Promise(() => undefined))
    rerender({ scope: "server:one|user:99" })
    expect(result.current.data).toBeUndefined()
  })

  it("exchanges a prototype share token for a collaborator session through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          actor_type: "external_collaborator",
          shared_actor_id: "psa_1",
          session_token: "session-token"
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => usePrototypePrivateLinkExchange(), {
      wrapper: buildWrapper()
    })

    await act(async () => {
      await result.current.mutateAsync({
        token: "prototype-token",
        display_name: "Acme PM",
        password: "demo-pass"
      })
    })

    expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/sharing/public/prototype-token/prototype-session",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json"
        }),
        body: JSON.stringify({
          display_name: "Acme PM",
          password: "demo-pass"
        })
      })
    )
  })

  it("preserves structured prototype link exchange error details for route-state mapping", async () => {
    const invalidLink = getPrototypeContractState("invalid_link")
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify(invalidLink.mockResponse),
        {
          status: invalidLink.httpStatus,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => usePrototypePrivateLinkExchange(), {
      wrapper: buildWrapper()
    })

    await expect(
      result.current.mutateAsync({
        token: "prototype-token",
        display_name: "Acme PM"
      })
    ).rejects.toMatchObject({
      status: invalidLink.httpStatus,
      detail: invalidLink.mockResponse.detail,
      message: invalidLink.mockResponse.detail.message
    })
  })
})
