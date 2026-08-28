// @vitest-environment jsdom
import { createHash } from "node:crypto"

import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesGraphAuthorityScope } from "../hooks/useNotesGraphAuthorityScope"

const mocks = vi.hoisted(() => ({
  getCurrentUser: vi.fn()
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: mocks.getCurrentUser }
}))

const config = (
  serverUrl: string,
  overrides: Record<string, unknown> = {}
) => ({
  serverUrl,
  authMode: "multi-user" as const,
  ...overrides
})

const activeUser = (id: number) => ({
  id,
  username: `user-${id}`,
  is_active: true
})

const expectedScope = (origin: string, principalId: number | string) => {
  const tuple = JSON.stringify([origin, String(principalId)])
  const digest = createHash("sha256")
    .update(`tldw:notes-graph-authority:v1\0${tuple}`, "utf8")
    .digest("hex")
  return `notes-graph:sha256:${digest}`
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

describe("useNotesGraphAuthorityScope", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("stays guarded without canonical configuration or a verified active principal", async () => {
    mocks.getCurrentUser.mockRejectedValue(new Error("Not authenticated"))
    const { result, rerender } = renderHook(
      ({ canonicalConfig, loading }) =>
        useNotesGraphAuthorityScope({
          config: canonicalConfig,
          loading
        }),
      {
        initialProps: {
          canonicalConfig: null as ReturnType<typeof config> | null,
          loading: true
        }
      }
    )

    expect(result.current).toBeNull()
    expect(mocks.getCurrentUser).not.toHaveBeenCalled()

    rerender({
      canonicalConfig: config("https://notes.example.test"),
      loading: false
    })
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1))
    expect(result.current).toBeNull()

    mocks.getCurrentUser.mockResolvedValue({
      id: 9,
      username: "inactive-user",
      is_active: false
    })
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "switch" }
        })
      )
    )
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2))
    expect(result.current).toBeNull()
  })

  it("becomes ready for a verified cookie session without credential material in the scope", async () => {
    mocks.getCurrentUser.mockResolvedValue(activeUser(42))
    const cookieConfig = config(" HTTPS://Notes.Example.Test:443/api/v1/ ", {
      apiKey: "raw-api-key",
      accessToken: "raw-access-token"
    })
    const { result } = renderHook(() =>
      useNotesGraphAuthorityScope({ config: cookieConfig, loading: false })
    )

    await waitFor(() => {
      expect(result.current).toBe(
        expectedScope("https://notes.example.test", 42)
      )
    })
    expect(result.current).toMatch(/^notes-graph:sha256:[0-9a-f]{64}$/)
    expect(result.current).not.toContain("raw-api-key")
    expect(result.current).not.toContain("raw-access-token")
  })

  it.each([
    ["single-user", { authMode: "single-user", apiKey: "single-user-key" }],
    ["multi-user", { authMode: "multi-user", accessToken: "multi-user-token" }]
  ] as const)(
    "uses the verified principal for %s authentication without exposing credentials",
    async (_mode, authConfig) => {
      mocks.getCurrentUser.mockResolvedValue(activeUser(77))
      const canonicalConfig = config("https://notes.example.test", authConfig)
      const { result } = renderHook(() =>
        useNotesGraphAuthorityScope({
          config: canonicalConfig,
          loading: false
        })
      )

      await waitFor(() => {
        expect(result.current).toBe(
          expectedScope("https://notes.example.test", 77)
        )
      })
      expect(result.current).not.toContain("single-user-key")
      expect(result.current).not.toContain("multi-user-token")
    }
  )

  it("clears across an A-B-A server boundary and ignores every stale authentication completion", async () => {
    const firstA = deferred<ReturnType<typeof activeUser>>()
    const serverB = deferred<ReturnType<typeof activeUser>>()
    const secondA = deferred<ReturnType<typeof activeUser>>()
    mocks.getCurrentUser
      .mockImplementationOnce(() => firstA.promise)
      .mockImplementationOnce(() => serverB.promise)
      .mockImplementationOnce(() => secondA.promise)

    const { result, rerender } = renderHook(
      ({ canonicalConfig }) =>
        useNotesGraphAuthorityScope({
          config: canonicalConfig,
          loading: false
        }),
      {
        initialProps: {
          canonicalConfig: config("https://server-a.example.test")
        }
      }
    )
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1))

    rerender({ canonicalConfig: config("https://server-b.example.test") })
    expect(result.current).toBeNull()
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2))

    rerender({ canonicalConfig: config("https://server-a.example.test") })
    expect(result.current).toBeNull()
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(3))

    await act(async () => {
      serverB.resolve(activeUser(20))
      firstA.resolve(activeUser(10))
      await Promise.resolve()
    })
    expect(result.current).toBeNull()

    await act(async () => {
      secondA.resolve(activeUser(30))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(result.current).toBe(
        expectedScope("https://server-a.example.test", 30)
      )
    })
  })

  it("clears immediately when canonical credentials change on the same server", async () => {
    const nextPrincipal = deferred<ReturnType<typeof activeUser>>()
    mocks.getCurrentUser
      .mockResolvedValueOnce(activeUser(8))
      .mockImplementationOnce(() => nextPrincipal.promise)
    const { result, rerender } = renderHook(
      ({ canonicalConfig }) =>
        useNotesGraphAuthorityScope({
          config: canonicalConfig,
          loading: false
        }),
      {
        initialProps: {
          canonicalConfig: config("https://notes.example.test", {
            accessToken: "first-token"
          })
        }
      }
    )
    await waitFor(() => expect(result.current).not.toBeNull())

    rerender({
      canonicalConfig: config("https://notes.example.test", {
        accessToken: "second-token"
      })
    })
    expect(result.current).toBeNull()
    expect(String(result.current)).not.toContain("second-token")
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2))

    await act(async () => {
      nextPrincipal.resolve(activeUser(8))
      await Promise.resolve()
    })
    await waitFor(() => expect(result.current).not.toBeNull())
  })

  it("clears synchronously on principal events and fences the prior resolution", async () => {
    const stale = deferred<ReturnType<typeof activeUser>>()
    const current = deferred<ReturnType<typeof activeUser>>()
    mocks.getCurrentUser
      .mockImplementationOnce(() => stale.promise)
      .mockImplementationOnce(() => current.promise)
    const canonicalConfig = config("https://notes.example.test")

    const { result } = renderHook(() =>
      useNotesGraphAuthorityScope({
        config: canonicalConfig,
        loading: false
      })
    )
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1))

    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "switch" }
        })
      )
    })
    expect(result.current).toBeNull()
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2))

    await act(async () => {
      stale.resolve(activeUser(11))
      await Promise.resolve()
    })
    expect(result.current).toBeNull()

    await act(async () => {
      current.resolve(activeUser(12))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(result.current).toBe(
        expectedScope("https://notes.example.test", 12)
      )
    })
  })

  it("clears synchronously on canonical config events before re-authentication completes", async () => {
    const nextPrincipal = deferred<ReturnType<typeof activeUser>>()
    mocks.getCurrentUser
      .mockResolvedValueOnce(activeUser(21))
      .mockImplementationOnce(() => nextPrincipal.promise)
    const canonicalConfig = config("https://notes.example.test")
    const { result } = renderHook(() =>
      useNotesGraphAuthorityScope({
        config: canonicalConfig,
        loading: false
      })
    )
    await waitFor(() => {
      expect(result.current).toBe(
        expectedScope("https://notes.example.test", 21)
      )
    })

    act(() => window.dispatchEvent(new Event("tldw:config-updated")))
    expect(result.current).toBeNull()
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2))

    await act(async () => {
      nextPrincipal.resolve(activeUser(22))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(result.current).toBe(
        expectedScope("https://notes.example.test", 22)
      )
    })
  })
})
