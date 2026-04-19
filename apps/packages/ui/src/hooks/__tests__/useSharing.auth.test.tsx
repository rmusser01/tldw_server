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
  useCloneWorkspace,
  usePrototypePrivateLinkExchange,
  useSharedWithMe
} from "@/hooks/useSharing"

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

    const { result } = renderHook(() => useSharedWithMe(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/sharing/shared-with-me"
    )
  })

  it("clones shared workspaces through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({ job_id: "job-1", status: "queued", message: "ok" }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => useCloneWorkspace(), {
      wrapper: buildWrapper()
    })

    await act(async () => {
      await result.current.mutateAsync({
        shareId: 9,
        new_name: "My Clone"
      })
    })

    expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/sharing/shared-with-me/9/clone",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json"
        }),
        body: JSON.stringify({ new_name: "My Clone" })
      })
    )
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
})
