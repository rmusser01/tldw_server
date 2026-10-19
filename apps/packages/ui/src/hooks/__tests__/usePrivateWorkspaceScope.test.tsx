import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { usePrivateWorkspaceScope } from "../usePrivateWorkspaceScope"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const state = vi.hoisted(() => ({
  config: { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "cookie-session" } as TldwConfig,
  resolve: vi.fn()
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({ config: state.config, authorityLoading: false })
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: (...args: unknown[]) => state.resolve(...args)
}))

const verified = (userId: number) => ({ config: state.config, userId, scopeKey: `owner-${userId}`, clientPrincipalVerified: true })
beforeEach(() => {
  state.config = { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "cookie-session" }
  state.resolve.mockReset(); state.resolve.mockResolvedValue(verified(1))
})

it("accepts an authenticated cookie principal without a client-side JWT", async () => {
  state.resolve.mockResolvedValueOnce({ ...verified(1), clientPrincipalVerified: false })
  const view = renderHook(() => usePrivateWorkspaceScope())
  await waitFor(() => expect(view.result.current.scope?.requestScope.userId).toBe(1))
})

it("lets the user retry failed verification without reloading or changing configuration", async () => {
  state.resolve.mockRejectedValueOnce(new Error("Server unavailable"))
  const view = renderHook(() => usePrivateWorkspaceScope())
  await act(async () => {})
  expect(view.result.current?.retry).toBeTypeOf("function")
  state.resolve.mockResolvedValueOnce(verified(1))
  act(() => view.result.current!.retry())
  await waitFor(() => expect(view.result.current?.scope?.requestScope.userId).toBe(1))
})

it("refuses verification from a different configured server", async () => {
  state.resolve.mockResolvedValueOnce({ ...verified(1), config: { ...state.config, serverUrl: "https://other.test" } })
  const view = renderHook(() => usePrivateWorkspaceScope())
  await act(async () => {})
  expect(view.result.current.scope).toBeNull()
})

it("refuses a single-user verification bound to another API key", async () => {
  state.config = { serverUrl: "http://localhost:8000", authMode: "single-user", apiKey: "captured-test-key" }
  state.resolve.mockResolvedValueOnce({ ...verified(1), userId: null,
    config: { serverUrl: state.config.serverUrl, authMode: "single-user", expectedSingleUserApiKeyScope: "other-key" } })
  const view = renderHook(() => usePrivateWorkspaceScope())
  await act(async () => {})
  expect(view.result.current.scope).toBeNull()
})

it("aborts the previous owner synchronously and ignores its late verification", async () => {
  const view = renderHook(() => usePrivateWorkspaceScope())
  await waitFor(() => expect(view.result.current.scope).not.toBeNull())
  const alice = view.result.current.scope!
  let stale!: (value: ReturnType<typeof verified>) => void
  state.resolve.mockImplementationOnce(() => new Promise(resolve => { stale = resolve }))
  act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")))
  expect(alice.signal.aborted).toBe(true)
  expect(alice.isCurrent()).toBe(false)
  expect(view.result.current.scope).toBeNull()
  state.resolve.mockResolvedValueOnce(verified(2))
  act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")))
  await waitFor(() => expect(view.result.current.scope?.requestScope.userId).toBe(2))
  await act(async () => { stale(verified(1)) })
  expect(view.result.current.scope?.requestScope.userId).toBe(2)
  view.unmount()
})

it("retires an existing scope when a configuration recheck cannot verify its principal", async () => {
  const view = renderHook(() => usePrivateWorkspaceScope())
  await waitFor(() => expect(view.result.current.scope).not.toBeNull())
  const alice = view.result.current.scope!
  state.resolve.mockRejectedValueOnce(new Error("Unauthorized"))
  act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
  await waitFor(() => expect(view.result.current.scope).toBeNull())
  expect(alice.signal.aborted).toBe(true)
  view.unmount()
})

it("does not grant an unverified principal or grant a scope after unmount", async () => {
  state.config = { ...state.config, authSource: "manual", accessToken: "unverified-test-token" }
  state.resolve.mockResolvedValueOnce({ ...verified(1), clientPrincipalVerified: false })
  const view = renderHook(() => usePrivateWorkspaceScope())
  await act(async () => {})
  expect(view.result.current.scope).toBeNull()
  let finish!: (value: ReturnType<typeof verified>) => void
  state.resolve.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  act(() => window.dispatchEvent(new CustomEvent("tldw:auth-credentials-changed")))
  view.unmount()
  await act(async () => { finish(verified(1)) })
  expect(view.result.current.scope).toBeNull()
})
